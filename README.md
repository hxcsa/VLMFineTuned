# VLMFineTuned — DocVQA Fine-Tuning with Unsloth LoRA

Parameter-efficient fine-tuning of a vision-language model for document visual question answering (DocVQA). The model learns to read a document image and return a short, extractive answer.

**Final model:** `Qwen/Qwen3.5-4B` + LoRA (r=32, MLP enabled) — ANLS 0.8833 on clean DocVQA validation (300 samples, seed 1234, 1344px). Adapters: 34 MB.

---

## Project Description

This project implements a complete, reproducible pipeline for fine-tuning vision-language models on the DocVQA dataset using parameter-efficient LoRA adapters. The key contribution is demonstrating that clean train/validation splits, proper LoRA configuration (including MLP modules), and thinking-aware inference can achieve strong DocVQA performance with minimal compute.

### Key Features

- **Clean train/validation methodology** — Trained on DocVQA train split, evaluated on held-out validation split
- **MLP LoRA support** — Extends standard attention-only LoRA with gate/up/down projection adapters
- **Holdout evaluation during training** — Automatic best-checkpoint selection via validation loss
- **Thinking-aware inference** — Proper `enable_thinking=False` handling for Qwen3.5-style models
- **Comprehensive evaluation** — ANLS (official DocVQA metric) + exact match reporting
- **Production-ready artifacts** — Dockerfile, Gradio demo, Hugging Face model/dataset cards
- **Reproducible pipeline** — Single-command data prep, training, and evaluation

### Results Summary

| Configuration | ANLS (300 val) | Exact Match | Notes |
|---------------|----------------|-------------|-------|
| Qwen3.5-4B Zero-Shot | 0.6265 | 0.5933 | Baseline |
| v1 LoRA (1k from val, contaminated) | 0.8626 | 0.8333 | Not clean |
| v2 r16 attn-only (2k train) | 0.8687 | 0.8567 | Clean |
| **v2 r32 + MLP (2k train)** | **0.8833** | **0.8667** | **Winner** |

**Improvement:** +0.26 ANLS over zero-shot, clean train-to-val evaluation.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hxcsa/VLMFineTuned.git
cd VLMFineTuned

# 2. Install dependencies (PyTorch/CUDA first per your platform, then)
pip install -r requirements.txt

# 3. Run the Gradio demo (loads best adapter from HF)
python 3_app.py --model Qwen/Qwen3.5-4B --adapter-dir hxcsa/qwen35-4b-docvqa-lora

# Or use the live demo (if this instance is running):
# https://72.83.150.152:55817/  (external port 10100, token auth)
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Training Framework** | Unsloth (2x faster training, 4-bit quantization) |
| **LoRA / PEFT** | TRL + PEFT (response-only loss) |
| **Base Model** | Qwen3.5-4B (thinking-style) |
| **Hardware** | NVIDIA RTX A6000 (48 GB) |
| **Model Hub** | Hugging Face Hub |
| **Evaluation** | ANLS (ICDAR DocVQA standard) + Exact Match |
| **Serving** | Gradio + Unsloth FastVisionModel |
| **Containerization** | Docker (pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime) |
| **Experiment Tracking** | Optional Weights & Biases |

---

## Pipeline

### 1. Data Preparation (`1_prepare_data.py`)

```bash
# V2 (from train split, 1344px, 2k samples)
python 1_prepare_data.py --dataset HuggingFaceM4/DocumentVQA --split train \
  --target-size 2000 --max-image-edge 1344 --output-dir data/docvqa_2k_train
```

**Process:**
- Loads raw DocVQA (lmms-lab/DocVQA or HuggingFaceM4/DocumentVQA)
- Validates: question ≥5 chars, answer ≤256 chars (shortest candidate), valid RGB image
- Filters aspect ratio ≤4.0, resizes long edge to max (LANCZOS)
- Scores candidates: concise answers, clear questions, readable page sizes, multi-answer agreement
- Takes top-K (3× pool → rank → shuffle with seed 3407)
- Saves in Unsloth-VL conversational schema with PIL images embedded

### 2. Training (`2_train.py`)

```bash
# V2 winner: r=32 + MLP, 2 epochs, 1344px, 2k train samples
python 2_train.py \
  --model Qwen/Qwen3.5-4B \
  --dataset-dir data/docvqa_2k_train \
  --output-dir outputs/v2_r32_mlp \
  --lora-r 32 --lora-alpha 32 --finetune-mlp-modules \
  --num-epochs 2 --eval-holdout 0.05 \
  --max-seq-length 3072 --per-device-batch-size 2 --grad-accum-steps 4 \
  --wandb-mode disabled
```

**Key Settings:**
- Base: `Qwen/Qwen3.5-4B` (4-bit NF4 via bitsandbytes)
- LoRA: r=32, α=32, dropout=0 — **attention** (`q/k/v/o_proj`) + **MLP** (`gate/up/down_proj`)
- ViT frozen: `finetune_vision_layers=False` + explicit `requires_grad=False` sweep
- Loss on assistant tokens only (`train_on_responses_only` with `〈user〉`/`〈assistant〉` markers)
- Optimizer: `adamw_8bit`, LR 2e-4 cosine, 5% warmup, weight decay 0.01
- Holdout eval: 5% per-epoch → best checkpoint auto-restored (`load_best_model_at_end`)
- 2 epochs = 476 steps, ~34 min on A6000, peak VRAM 4.8 GB

### 3. Evaluation (`4_evaluate.py`)

```bash
python 4_evaluate.py \
  --model-id Qwen/Qwen3.5-4B \
  --adapter-dir outputs/v2_r32_mlp/lora_adapters \
  --num-samples 300 --max-image-edge 1344 \
  --output eval_results/v2_r32_mlp_val300.json
```

**ANLS Metric (ICDAR DocVQA Standard):**
- For each prediction, compute best score over all gold answers:
  `score = max(1 - NLD) if NLD < 0.5 else 0` where `NLD = levenshtein / max(len)`
- Also reports case/space-insensitive exact match

---

## Inference Gotcha: `enable_thinking=False`

`Qwen3.5-4B` is a **thinking-style** model — by default it emits long chain-of-thought before the answer, hiding the concise trained behavior and truncating at low `max_new_tokens`. **Always disable thinking:**

```python
text = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
    enable_thinking=False  # critical
)
```

Both `3_app.py` and `4_evaluate.py` do this automatically.

---

## Docker (Inference Only)

Slim runtime image (~4 GB) that fetches adapters from HF at startup:

```bash
# Build
docker build -t vlmfinetuned:latest .

# Run (GPU required)
docker run --gpus all -p 7860:7860 \
  -e MODEL_ID=Qwen/Qwen3.5-4B \
  -e ADAPTER_REPO=hxcsa/qwen35-4b-docvqa-lora \
  vlmfinetuned:latest
```

- Base: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`
- Installs only inference deps (no training stack)
- Entrypoint runs `3_app.py` with env-configurable model/adapter

---

## Reproducing the Full v2 Run

```bash
# 1. Data (2k from train, 1344px)
python 1_prepare_data.py --dataset HuggingFaceM4/DocumentVQA --split train \
  --target-size 2000 --max-image-edge 1344 --output-dir data/docvqa_2k_train

# 2. Parallel training (both fit on 48 GB A6000)
python 2_train.py --model Qwen/Qwen3.5-4B --dataset-dir data/docvqa_2k_train \
  --output-dir outputs/v2_r32_mlp --lora-r 32 --lora-alpha 32 --finetune-mlp-modules \
  --num-epochs 2 --eval-holdout 0.05 --max-seq-length 3072 \
  --per-device-batch-size 2 --grad-accum-steps 4 --wandb-mode disabled &

python 2_train.py --model Qwen/Qwen3.5-4B --dataset-dir data/docvqa_2k_train \
  --output-dir outputs/v2_r16_attn --lora-r 16 --lora-alpha 16 \
  --num-epochs 2 --eval-holdout 0.05 --max-seq-length 3072 \
  --per-device-batch-size 2 --grad-accum-steps 4 --wandb-mode disabled &

# 3. Evaluate both (ANLS on val, seed 1234, 1344px)
python 4_evaluate.py --adapter-dir outputs/v2_r32_mlp/lora_adapters --output eval_v2a.json
python 4_evaluate.py --adapter-dir outputs/v2_r16_attn/lora_adapters --output eval_v2b.json

# 4. Publish winner
hf upload <user>/qwen35-4b-docvqa-lora outputs/v2_r32_mlp/lora_adapters .
hf upload <user>/qwen35-4b-docvqa-lora README.md
```

---

## Artifacts on Hugging Face

| Repository | Contents |
|------------|----------|
| `hxcsa/qwen35-4b-docvqa-lora` | LoRA adapters (34 MB), tokenizer, chat template, **model card with metrics** |
| `hxcsa/docvqa-1k-qwen-vl` | 1k samples from val (v1), 453 MB |
| `hxcsa/docvqa-2k-train` | 2k samples from train (v2), 1.1 GB |

---

## Live Demo

On this Vast.ai instance: external port **10100** (token auth via `OPEN_BUTTON_TOKEN` / `WEB_PASSWORD`), internal Gradio on 7860. Managed by supervisor as `vlm-demo`.

```bash
# Access
curl -H "Authorization: Bearer $TOKEN" http://<PUBLIC_IP>:55817/
```

---

## Hardware / Environment

- GPU: NVIDIA RTX A6000 (48 GB) — plenty of headroom (peak 4.8 GB training, 8 GB eval)
- Image: Vast.ai Unsloth Studio (PyTorch 2.10 + cu128, Unsloth 2026.7)
- CPU: 48 cores, 1.5 TB RAM
- All code runs on the GPU instance — nothing on your local machine.

---

## License

Apache-2.0 (code) / base model license applies to weights.

---

## Next Steps / Ideas

- Scale to full 39k DocVQA train split
- Ablate vision-layer LoRA (currently frozen) on larger VRAM
- Distill thinking traces for multi-step reasoning
- ONNX / TensorRT export for lower-latency serving