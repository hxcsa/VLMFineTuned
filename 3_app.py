#!/usr/bin/env python3
"""
3_app.py
--------
Gradio Blocks UI for DocVQA inference with base Qwen2.5-VL + saved LoRA adapters.
Streams tokens token-by-token; aggressively frees CUDA cache after each run.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Generator, List, Optional, Tuple

# Must be set before torch import for allocator behavior on long sessions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

DEFAULT_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER = Path("outputs/qwen25vl_3b_docvqa_lora/lora_adapters")

SYSTEM_PROMPT = (
    "You are a precise document visual question answering assistant. "
    "Read the provided document image carefully and answer the user's question "
    "using only information visible in the image. "
    "Respond with a concise factual answer. Do not invent details."
)

# Serialize GPU access — Gradio can fire concurrent requests.
_INFER_LOCK = threading.Lock()
_MODEL = None
_TOKENIZER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio DocVQA app for fine-tuned Qwen2.5-VL")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.1)
    # Low temp: DocVQA wants deterministic extractive answers, not creative prose.
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def free_cuda() -> None:
    """Release fragmented blocks between requests to avoid long-session OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass


def assert_runtime() -> None:
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA unavailable — inference will run on CPU (very slow for VLMs)."
        )
    else:
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            props.name,
            props.total_memory / (1024**3),
        )


def load_model_with_adapter(
    model_id: str,
    adapter_dir: Path,
    load_in_4bit: bool,
):
    """
    Load base FastVisionModel, then attach PEFT LoRA adapters from disk.
    """
    from unsloth import FastVisionModel
    from peft import PeftModel

    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. "
            "Train first with 2_train.py or pass --adapter-dir."
        )

    last_err: Optional[Exception] = None
    model = tokenizer = None
    loaded_id = model_id

    for candidate in (model_id, FALLBACK_MODEL):
        try:
            logger.info("Loading base model %s (4bit=%s)...", candidate, load_in_4bit)
            model, tokenizer = FastVisionModel.from_pretrained(
                candidate,
                load_in_4bit=load_in_4bit,
                use_gradient_checkpointing=False,  # inference path
            )
            loaded_id = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Base load failed for %s: %s", candidate, exc)

    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load base VLM: {last_err}")

    # Prefer Unsloth / PEFT load of adapter weights.
    try:
        logger.info("Applying LoRA adapters from %s", adapter_dir)
        # If adapter was saved via model.save_pretrained (PEFT layout):
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    except Exception as peft_exc:  # noqa: BLE001
        logger.warning("PeftModel.from_pretrained failed (%s); trying FastVisionModel path", peft_exc)
        try:
            # Some Unsloth versions expose load via get_peft_model + load_adapter
            model.load_adapter(str(adapter_dir))
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load LoRA adapters from {adapter_dir}: {exc2}"
            ) from exc2

    FastVisionModel.for_inference(model)
    model.eval()
    logger.info("Model ready | base=%s | adapter=%s", loaded_id, adapter_dir)
    return model, tokenizer


def build_messages(image: Image.Image, question: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]


def prepare_inputs(tokenizer, image: Image.Image, question: str):
    """
    Qwen2.5-VL processor path via Unsloth tokenizer:
    apply_chat_template → tokenizer(image, text, ...).
    """
    messages = build_messages(image, question)
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        # Qwen3.5-style models: skip the thinking block so answers stay concise.
        # Harmless for templates that don't use it (e.g. Qwen2.5-VL).
        enable_thinking=False,
    )
    # tokenizer here is the multimodal processor wrapper from Unsloth.
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    # Move tensors to device
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            moved[k] = v.to(_DEVICE)
        else:
            moved[k] = v
    return moved


class _Streamer:
    """
    Minimal TextIteratorStreamer-compatible sink used if transformers streamer
    is unavailable; primary path uses TextIteratorStreamer.
    """

    def __init__(self):
        self.text = ""


def stream_answer(
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    """
    Gradio generator: yields partial answer strings token-by-token.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        yield "ERROR: Model not loaded."
        return
    if image is None:
        yield "Please upload a document image."
        return
    if not question or not str(question).strip():
        yield "Please enter a question."
        return

    # Normalize image
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        yield f"ERROR: Could not read image ({exc})"
        return

    # Bound resolution for inference VRAM (same philosophy as data prep).
    w, h = image.size
    max_edge = 1280
    long_edge = max(w, h)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    from transformers import TextIteratorStreamer
    import threading as th

    partial = ""
    try:
        with _INFER_LOCK:
            inputs = prepare_inputs(_TOKENIZER, image, question)
            streamer = TextIteratorStreamer(
                _TOKENIZER,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Greedy-ish decoding for extractive VQA when temperature ~ 0.
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )
            if temperature is not None and float(temperature) > 0:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            else:
                gen_kwargs.update(do_sample=False)

            def _generate():
                try:
                    with torch.inference_mode():
                        _MODEL.generate(**gen_kwargs)
                except Exception:
                    logger.exception("generate() failed")

            worker = th.Thread(target=_generate, daemon=True)
            worker.start()

            for token_text in streamer:
                partial += token_text
                yield partial

            worker.join(timeout=300)

    except torch.cuda.OutOfMemoryError:
        free_cuda()
        yield (
            (partial + "\n\n" if partial else "")
            + "ERROR: CUDA OOM during generation. "
            "Try a smaller image or lower max_new_tokens."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        yield f"ERROR: {exc}\n{traceback.format_exc(limit=2)}"
    finally:
        # Critical for multi-turn Gradio sessions on 24GB cards.
        free_cuda()


def build_ui(args: argparse.Namespace):
    import gradio as gr

    with gr.Blocks(
        title="Qwen2.5-VL DocVQA (LoRA)",
        theme=gr.themes.Soft(),
        css="""
        .output-box textarea {font-family: ui-monospace, monospace;}
        """,
    ) as demo:
        gr.Markdown(
            """
            # Qwen2.5-VL DocVQA — LoRA Inference
            Upload a document page, ask a question, get a streamed extractive answer.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil",
                    label="Document image",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                question_in = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What is the total amount listed?",
                    lines=2,
                )
                with gr.Accordion("Generation settings", open=False):
                    max_new = gr.Slider(
                        16,
                        512,
                        value=args.max_new_tokens,
                        step=8,
                        label="max_new_tokens",
                    )
                    temperature = gr.Slider(
                        0.0,
                        1.5,
                        value=args.temperature,
                        step=0.05,
                        label="temperature (0 = greedy)",
                    )
                    top_p = gr.Slider(
                        0.1,
                        1.0,
                        value=args.top_p,
                        step=0.05,
                        label="top_p",
                    )
                with gr.Row():
                    btn = gr.Button("Answer", variant="primary")
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                answer_out = gr.Textbox(
                    label="Model answer (streamed)",
                    lines=12,
                    elem_classes=["output-box"],
                )
                status = gr.Markdown(
                    f"Device: `{_DEVICE}` | Adapter: `{args.adapter_dir}`"
                )

        btn.click(
            fn=stream_answer,
            inputs=[image_in, question_in, max_new, temperature, top_p],
            outputs=[answer_out],
        )
        clear_btn.click(
            fn=lambda: (None, "", "", f"Device: `{_DEVICE}` | cache cleared"),
            inputs=None,
            outputs=[image_in, question_in, answer_out, status],
        ).then(fn=lambda: free_cuda())

        gr.Examples(
            examples=[],
            inputs=[image_in, question_in],
        )

    return demo


def main() -> int:
    global _MODEL, _TOKENIZER

    args = parse_args()
    assert_runtime()

    load_in_4bit = args.load_in_4bit and not args.no_4bit and torch.cuda.is_available()

    try:
        _MODEL, _TOKENIZER = load_model_with_adapter(
            model_id=args.model,
            adapter_dir=args.adapter_dir,
            load_in_4bit=load_in_4bit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize model: %s", exc)
        return 1

    free_cuda()

    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        logger.error("gradio is required: pip install gradio")
        return 1

    demo = build_ui(args)
    logger.info("Launching Gradio on %s:%d", args.host, args.port)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
