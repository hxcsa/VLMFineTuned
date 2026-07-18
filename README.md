# VLMFineTuned — DocVQA Fine-Tuning with Unsloth LoRA

Parameter-efficient fine-tuning (PEFT) of a vision-language model on **DocVQA**
(document visual question answering) using **Unsloth FastVisionModel + LoRA**.
The model learns to read a document image and return a short, extractive answer.

**Final model:** `Qwen/Qwen3.5-4B` + LoRA adapters (12.6 MB) trained on 1,000
curated DocVQA samples. Adapters live in
`outputs/qwen35_4b_docvqa_lora/lora_adapters/`.

---

## Results at a glance

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` (4-bit NF4 via bitsandbytes) |
| Adapter | LoRA r=16, α=16, dropout 0 — attention only (`q/k/v/o_proj`) |
| Trainable params | 3,145,728 (0.07% of 4.54B) — ViT fully frozen |
| Training data | 1,000 top-quality DocVQA samples (`data/docvqa_1k_qwen_vl/`) |
| Schedule | 1 epoch = 125 steps, effective batch 8 (1 × grad-accum 8) |
| Final avg train loss | **0.355** (1.03 → ~0.2–0.38) |
| Train time | 11m51s on 1× RTX A6000 (peak VRAM 3.75 GB) |

Sanity check (fine-tuned model, greedy decoding, thinking disabled):

| Question | Gold | Prediction |
|---|---|---|
| How was the amount contributed? | CHECK | **CHECK** |
| Which acetabular shell over Porocoat coating? | pinnacle | **54 mm Pinnacle** |
| Name of the company? | itc limited | **ITC Limited** |
| Who wrote the letter? | Kay | **Kay** |

---

## Hardware / environment

Developed on a Vast.ai instance: 1× NVIDIA RTX A6000 (48 GB), Unsloth Studio
image (PyTorch 2.10 + cu128, Unsloth 2026.7.2, TRL, Transformers 5.5).
The scripts were originally parameterized for a 24 GB RTX 4090 and run far
under that budget (peak 3.75 GB), so they work on both.

## Repository layout

```
1_prepare_data.py   # Download DocVQA → filter to top-1,000 → Qwen-VL chat schema
2_train.py          # Unsloth FastVisionModel + LoRA PEFT training
3_app.py            # Gradio demo (base model + LoRA adapters, streaming)
compare.py          # Side-by-side base vs fine-tuned predictions
hf_infer.py         # Plain-Transformers single-sample inference helper
requirements.txt    # Python deps (install torch/CUDA wheels first)
data/               # (gitignored) processed datasets
outputs/            # (gitignored) training runs, adapters, checkpoints
```

## Pipeline

### 1 — Data preprocessing (`1_prepare_data.py`)

- Source: `lmms-lab/DocVQA` **validation** split (5,349 rows; used because it was
  already cached locally — see caveat below).
- Streaming quality filter: requires valid question (≥5 chars), answer (≤256
  chars, shortest of the candidate answers), RGB image, aspect ratio ≤ 4.
  Images are capped at 1024 px on the long edge (LANCZOS) to bound ViT tokens.
- A heuristic `quality_score` (concise answers, well-formed questions,
  readable page sizes, multi-annotator agreement) ranks candidates; the top
  1,000 are kept and shuffled (seed 3407). In practice 3,000 candidates were
  scanned with **0 hard rejects**, so the score did the selection.
- Output schema per row (Unsloth vision-collator compatible):
  `messages = [system, user(image+question), assistant(answer)]` plus a `meta`
  dict for auditing. Saved with `datasets.save_to_disk` → `data/docvqa_1k_qwen_vl/`.

Reproduce:
```bash
python 1_prepare_data.py --dataset lmms-lab/DocVQA --dataset-name DocVQA --split validation
```

### 2 — Fine-tuning (`2_train.py`)

- `FastVisionModel.from_pretrained(Qwen/Qwen3.5-4B, load_in_4bit=True)`,
  gradient checkpointing ("unsloth"), max_seq_length 2048.
- LoRA: r=16, α=16, dropout 0, `bias="none"`, attention projections only
  (`q_proj/k_proj/v_proj/o_proj`). Vision tower frozen twice over:
  `finetune_vision_layers=False` plus an explicit `requires_grad=False` sweep
  over any vision-named parameter.
- Loss on assistant tokens only (`train_on_responses_only` with the
  `<|im_start|>user` / `<|im_start|>assistant` markers).
- Optimizer adamw_8bit, LR 2e-4, cosine schedule, 5% warmup, weight decay 0.01,
  bf16, max_grad_norm 1.0.

Reproduce:
```bash
python 2_train.py \
  --model Qwen/Qwen3.5-4B \
  --dataset-dir data/docvqa_1k_qwen_vl \
  --output-dir outputs/qwen35_4b_docvqa_lora \
  --wandb-mode disabled
```

### 3 — Inference (important gotcha)

`Qwen3.5-4B` is a **thinking-style** model: by default it emits a long
chain-of-thought before the answer, which hides the concise trained behavior
and truncates at small `max_new_tokens`. Disable thinking at inference:

```python
text = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
    enable_thinking=False,   # ← concise, direct answers
)
```

Then greedy decoding with `max_new_tokens=64` returns the short extractive
answers the model was trained for.

Demo app (note it defaults to the 3B model — point it at the right base):
```bash
python 3_app.py --model Qwen/Qwen3.5-4B \
  --adapter-dir outputs/qwen35_4b_docvqa_lora/lora_adapters
```

## What gets published where

Weights and datasets do **not** belong in git — the repo's `.gitignore` already
excludes `data/`, `outputs/`, `*.safetensors`, `*.log` and
`unsloth_compiled_cache/`. GitHub gets code + this README; Hugging Face gets
the artifacts.

### Push to GitHub (code)

```bash
git add -A && git commit -m "Add comparison/inference helpers and project README"
git push origin main        # remote: https://github.com/hxcsa/VLMFineTuned
```

### Push to Hugging Face (artifacts)

1. **LoRA adapters → a model repo** (the main deliverable, ~32 MB incl. tokenizer).
   Only `lora_adapters/` — skip `checkpoint-100/125` (they only add optimizer/
   RNG state for resuming) and `final_checkpoint` (same weights again).
   ```bash
   huggingface-cli login   # token with write access from https://huggingface.co/settings/tokens
   huggingface-cli upload <your-user>/qwen35-4b-docvqa-lora \
     outputs/qwen35_4b_docvqa_lora/lora_adapters .
   ```
2. **Processed dataset → a dataset repo** (optional, ~450 MB with images):
   ```bash
   huggingface-cli upload <your-user>/docvqa-1k-qwen-vl \
     data/docvqa_1k_qwen_vl . --repo-type dataset
   ```
3. Add a model card: the run's `outputs/qwen35_4b_docvqa_lora/README.md`
   (auto-generated) is a starting point — add the results table and the
   `enable_thinking=False` note from this README.

## Known caveats / next steps

- **Train/eval contamination:** the 1k set was drawn from the DocVQA
  *validation* split, so benchmark numbers on that split would be inflated.
  For clean evaluation, retrain from the DocVQA train split or hold out a
  slice of the 1k before training.
- The training runs under `outputs/test_run`, `outputs/optimized_run`,
  `outputs/qwen3.5_test` were 20-sample smoke tests — not real models.
- One epoch was enough for correct extractive answers with thinking disabled;
  if you want the model to be terse even in thinking mode, train 2–3 epochs.
