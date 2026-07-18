#!/usr/bin/env python3
"""
2_train.py
----------
Parameter-efficient fine-tuning of Qwen2.5-VL-3B-Instruct with Unsloth
FastVisionModel + LoRA (language attention only; ViT fully frozen).

Hardware target: 1x RTX 4090 24GB VRAM, Ubuntu, single process.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Env knobs BEFORE importing torch / unsloth (must be set early)
# ---------------------------------------------------------------------------
# Reduce CUDA memory fragmentation under long multimodal sequences.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Faster HF downloads when hf_transfer is installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Defaults tuned for 24GB VRAM + 1k DocVQA PEFT
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct"  # Unsloth-optimized mirror; falls back to Qwen/
FALLBACK_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATASET_DIR = Path("data/docvqa_1k_qwen_vl")
DEFAULT_OUTPUT_DIR = Path("outputs/qwen25vl_3b_docvqa_lora")

# LoRA rank 16: enough capacity for extractive DocVQA style shift without
# overfitting a 1k set. r>32 on tiny data tends to memorize and forget.
LORA_R = 16
# alpha == r keeps effective scale ~1.0 (deltaW *= alpha/r). Standard Unsloth default.
LORA_ALPHA = 16
# Dropout 0 for PEFT at small N — regularization comes from low rank + weight decay.
LORA_DROPOUT = 0.0

# Sequence length: document images + short Q/A. 2048 fits 4090 with 4-bit + LoRA;
# 4096 often OOMs when images are dense (many vision tokens).
MAX_SEQ_LENGTH = 2048

# Micro-batch 1 is mandatory for VL at this seq length on 24GB.
PER_DEVICE_BATCH_SIZE = 1
# Effective batch = 1 * 8 = 8 → stable grads without large activation memory.
GRAD_ACCUM_STEPS = 8
# 2e-4 is Unsloth's recommended LoRA LR band for 4-bit VL; higher risks collapse.
LEARNING_RATE = 2e-4
# 1 epoch over 1k samples ≈ 1000/8 = 125 optimizer steps — enough for adapter fit,
# low enough to limit catastrophic forgetting of general VLM skills.
NUM_EPOCHS = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
# Cosine decays LR smoothly; better late-stage stability than constant on short runs.
LR_SCHEDULER = "cosine"
SEED = 3407
LOGGING_STEPS = 5
SAVE_STEPS = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth PEFT training for Qwen2.5-VL DocVQA")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--wandb-project", type=str, default="vlm-docvqa-peft")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit; use bf16 full base (needs more VRAM)")
    p.add_argument("--max-steps", type=int, default=-1, help="Override epoch-based schedule if > 0")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument(
        "--finetune-mlp-modules",
        action="store_true",
        help="Also apply LoRA to language MLP (gate/up/down) — more capacity for grounding",
    )
    p.add_argument(
        "--eval-holdout",
        type=float,
        default=0.0,
        help="Fraction of training data held out for per-epoch eval + best-checkpoint selection (0 disables)",
    )
    return p.parse_args()


def assert_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training script requires a GPU "
            "(target: RTX 4090 24GB). Check NVIDIA drivers and CUDA toolkit."
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    logger.info(
        "GPU 0: %s | VRAM: %.1f GB | CC: %d.%d",
        props.name,
        vram_gb,
        props.major,
        props.minor,
    )
    if vram_gb < 20:
        logger.warning(
            "Detected <20GB VRAM. Consider lowering --max-seq-length to 1536 "
            "and keeping batch size 1 / grad accum >= 8."
        )


def setup_wandb(args: argparse.Namespace) -> str:
    """
    Returns report_to value for SFTConfig.
    """
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        return "none"

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed; continuing without logging. pip install wandb")
        return "none"

    # Lazy login: uses WANDB_API_KEY if present; otherwise prompts once in online mode.
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY not set. Run `wandb login` or export WANDB_API_KEY. "
            "Falling back to offline mode."
        )
        os.environ["WANDB_MODE"] = "offline"
    return "wandb"


def load_processed_dataset(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load save_to_disk output from 1_prepare_data.py.
    Return a Python list of {messages: [...]} dicts — required by Unsloth vision
    collator (map()/Arrow can corrupt nested PIL images).
    """
    from datasets import load_from_disk

    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset dir not found: {dataset_dir}. Run 1_prepare_data.py first."
        )

    ds = load_from_disk(str(dataset_dir))
    logger.info("Loaded processed dataset: %d rows from %s", len(ds), dataset_dir)

    # Materialize as list so PIL images stay live Python objects.
    converted: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Row {i}: invalid messages schema")
        converted.append({"messages": messages})

    logger.info("Materialized %d conversational samples for vision collator", len(converted))
    return converted


def load_model_and_tokenizer(args: argparse.Namespace):
    from unsloth import FastVisionModel

    load_in_4bit = args.load_in_4bit and not args.no_4bit
    # 4-bit NF4 quant (bitsandbytes) cuts base weights ~4x so LoRA + vision
    # activations fit in 24GB. FP8 needs Hopper/Ada-specific stacks; 4-bit is
    # the portable production default on 4090.
    model_id = args.model
    last_err: Optional[Exception] = None

    for candidate in (model_id, FALLBACK_MODEL):
        try:
            logger.info(
                "Loading FastVisionModel from %s | 4bit=%s | max_seq=%d",
                candidate,
                load_in_4bit,
                args.max_seq_length,
            )
            model, tokenizer = FastVisionModel.from_pretrained(
                candidate,
                load_in_4bit=load_in_4bit,
                use_gradient_checkpointing="unsloth",  # Unsloth GC: lower VRAM, mild slowdown
                max_seq_length=args.max_seq_length,
            )
            return model, tokenizer, candidate
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Failed loading %s: %s", candidate, exc)

    raise RuntimeError(f"Could not load vision model. Last error: {last_err}")


def apply_lora(model, args: argparse.Namespace):
    """
    Freeze ViT completely; LoRA only language attention projections.
    Why freeze vision:
      - DocVQA adaptation is mostly linguistic grounding of already-good OCR features.
      - ViT activations dominate VRAM; freezing them saves memory and prevents
        destroying general visual features with only 1k samples.
    """
    from unsloth import FastVisionModel

    # Explicit attention targets per user requirement (q/k/v/o).
    # finetune_vision_layers=False is the hard freeze for the ViT tower.
    # finetune_mlp_modules=False keeps gate/up/down frozen → fewer trainable params,
    # less forgetting on tiny data. Attention-only LoRA is enough for VQA phrasing.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # STRICT: freeze entire ViT
        finetune_language_layers=True,
        finetune_attention_modules=True,   # q_proj, k_proj, v_proj, o_proj (+ variants)
        finetune_mlp_modules=args.finetune_mlp_modules,  # optional: MLP LoRA adds capacity
        r=args.lora_r,                     # rank: capacity vs overfit tradeoff
        lora_alpha=args.lora_alpha,        # scaling; alpha==r → scale 1.0
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        # Explicit list pins intent; MLP projections join only when requested.
        target_modules=(
            ["q_proj", "k_proj", "v_proj", "o_proj"]
            + (["gate_proj", "up_proj", "down_proj"] if args.finetune_mlp_modules else [])
        ),
    )

    # Belt-and-suspenders: force requires_grad=False on any vision parameter
    # that might still be open (covers naming differences across Qwen VL versions).
    frozen_n, trainable_n = 0, 0
    for name, param in model.named_parameters():
        lname = name.lower()
        is_vision = any(
            key in lname
            for key in (
                "visual",
                "vision_tower",
                "vision_model",
                "vit",
                "vision_encoder",
                "patch_embed",
            )
        )
        if is_vision:
            param.requires_grad = False
            frozen_n += param.numel()
        elif param.requires_grad:
            trainable_n += param.numel()

    logger.info(
        "Trainable params: %s | Forced-frozen vision tensors: %s elements",
        f"{trainable_n:,}",
        f"{frozen_n:,}",
    )
    return model


def build_trainer(model, tokenizer, train_dataset, eval_dataset, args: argparse.Namespace, report_to: str):
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Qwen2.5 chat markers for response-only loss (ignore user/system tokens).
    # Training on assistant tokens only improves instruction following and
    # prevents the model from learning to rewrite the question.
    instruction_part = "<|im_start|>user\n"
    response_part = "<|im_start|>assistant\n"

    collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        train_on_responses_only=True,
        instruction_part=instruction_part,
        response_part=response_part,
        # completion_only_loss=True ignores padding / vision placeholder noise
        completion_only_loss=True,
        resize="min",  # fit model default image size without upscaling tiny pages
    )

    # bf16 on Ada (4090) is stable and faster than fp16 for mixed precision.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # With a holdout, evaluate + save per epoch and keep the best checkpoint
    # (proper model selection instead of "last epoch wins").
    use_eval = eval_dataset is not None and len(eval_dataset) > 0

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        # Effective batch 8: good SNR for LoRA without large activation memory.
        learning_rate=args.learning_rate,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        eval_strategy="epoch" if use_eval else "no",
        save_strategy="epoch" if use_eval else "steps",
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        per_device_eval_batch_size=1,
        optim="adamw_8bit",  # 8-bit Adam: ~2x optimizer state savings vs fp32 AdamW
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        seed=args.seed,
        report_to=report_to,
        # Required for VL SFT with Unsloth collator — do not use text-only packing.
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=2,  # 2 workers: overlap image decode without RAM thrash
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,  # clip spikes from occasional hard document samples
        logging_first_step=True,
        run_name=args.wandb_run_name or "qwen25vl-3b-docvqa-lora",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        args=sft_args,
    )
    return trainer


def save_lora(model, tokenizer, output_dir: Path) -> Path:
    """
    Unsloth fast save: adapters only (MBs, not full 3B checkpoint).
    """
    from unsloth import FastVisionModel

    adapter_dir = output_dir / "lora_adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained is the Unsloth-recommended path for PEFT adapters.
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Also export a merged 16-bit copy optionally? Skip by default — 24GB + disk.
    # Users can merge later with FastVisionModel for deployment.
    logger.info("Saved LoRA adapters -> %s", adapter_dir)

    # Write a small marker file for the Gradio app.
    meta = adapter_dir / "ADAPTER_READY"
    meta.write_text(f"base_model_hint={DEFAULT_MODEL}\n", encoding="utf-8")
    return adapter_dir


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> int:
    args = parse_args()

    try:
        assert_cuda()
        report_to = setup_wandb(args)

        train_dataset = load_processed_dataset(args.dataset_dir)

        # Holdout split (seeded) for per-epoch eval + best-checkpoint selection.
        eval_dataset = None
        if args.eval_holdout and args.eval_holdout > 0:
            import random

            idx = list(range(len(train_dataset)))
            random.Random(args.seed).shuffle(idx)
            n_eval = max(1, int(len(idx) * args.eval_holdout))
            eval_dataset = [train_dataset[i] for i in sorted(idx[:n_eval])]
            train_dataset = [train_dataset[i] for i in sorted(idx[n_eval:])]
            logger.info(
                "Holdout enabled: train=%d | eval=%d (%.1f%%)",
                len(train_dataset),
                len(eval_dataset),
                100.0 * args.eval_holdout,
            )

        model, tokenizer, loaded_id = load_model_and_tokenizer(args)
        logger.info("Using base model: %s", loaded_id)

        model = apply_lora(model, args)

        # Enable training mode kernels (Unsloth switches fused paths).
        from unsloth import FastVisionModel

        FastVisionModel.for_training(model)

        trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, args, report_to)

        # Show VRAM before train
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            free = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info("Free VRAM before train: %.2f GB", free)

        train_result = trainer.train(resume_from_checkpoint=args.resume_from)
        logger.info("Train metrics: %s", train_result.metrics)

        adapter_dir = save_lora(model, tokenizer, args.output_dir.resolve())

        # Persist trainer state / final checkpoint as well.
        trainer.save_model(str(args.output_dir.resolve() / "final_checkpoint"))
        metrics_path = args.output_dir.resolve() / "train_metrics.json"
        import json

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f, indent=2)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info("Peak VRAM allocated: %.2f GB", peak)

        logger.info("Training complete. Adapters: %s", adapter_dir)
        logger.info("Next: python 3_app.py --adapter-dir %s", adapter_dir)
        free_cuda()
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        free_cuda()
        return 1


if __name__ == "__main__":
    sys.exit(main())
