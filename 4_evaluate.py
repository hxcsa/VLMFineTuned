#!/usr/bin/env python3
"""
4_evaluate.py
-------------
Quantitative evaluation of a (optionally LoRA-adapted) vision-language model
on DocVQA using the official-style ANLS metric + exact match.

ANLS (Average Normalized Levenshtein Similarity), ICDAR DocVQA standard:
    per-sample score = max over gold answers of t(pred, gold)
    t(a, b) = 1 - NLD(a, b)   if NLD(a, b) <  0.5
            = 0               otherwise
    NLD = levenshtein(a, b) / max(len(a), len(b))

Evaluates directly on a raw DocVQA split (independent of the processed
training dirs), greedy decoding, thinking disabled for Qwen3.5-style models.

Example:
    python 4_evaluate.py --model-id Qwen/Qwen3.5-4B \
        --adapter-dir outputs/qwen35_4b_docvqa_lora/lora_adapters \
        --num-samples 300 --output eval_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

SYSTEM_PROMPT = (
    "You are a precise document visual question answering assistant. "
    "Read the provided document image carefully and answer the user's question "
    "using only information visible in the image. "
    "Respond with a concise factual answer. Do not invent details."
)


# ---------------------------------------------------------------------------
# ANLS
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance. Answers are short (<300 chars), O(n*m) is fine."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def anls_single(prediction: str, golds: List[str]) -> float:
    """Best ANLS over the gold answer list, thresholded at NLD 0.5 (official style)."""
    prediction = prediction.strip()
    best = 0.0
    for gold in golds:
        gold = str(gold).strip()
        if not gold and not prediction:
            return 1.0
        denom = max(len(prediction), len(gold))
        if denom == 0:
            continue
        nld = levenshtein(prediction, gold) / denom
        score = 1.0 - nld if nld < 0.5 else 0.0
        best = max(best, score)
    return best


def exact_match(prediction: str, golds: List[str]) -> float:
    """Case/space-insensitive exact match against any gold."""
    norm = prediction.strip().lower()
    return 1.0 if any(norm == str(g).strip().lower() for g in golds) else 0.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_eval_rows(split: str, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(num_samples, len(ds))]
    rows = []
    for i in idxs:
        r = ds[i]
        img = r.get("image")
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(BytesIO(img["bytes"]))
        if img is None:
            continue
        answers = r.get("answers") or [r.get("answer")]
        answers = [str(a) for a in answers if a is not None and str(a).strip()]
        if not answers:
            continue
        rows.append(
            {
                "idx": int(i),
                "question": str(r["question"]).strip(),
                "answers": answers,
                "image": img.convert("RGB"),
            }
        )
    logger.info("Prepared %d eval rows from split=%s (seed=%d)", len(rows), split, seed)
    return rows


def resize_image(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image
    scale = max_edge / float(long_edge)
    return image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.LANCZOS,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(model_id: str, adapter_dir: Optional[str], load_in_4bit: bool):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
    )
    model.to("cuda")
    FastVisionModel.for_inference(model)
    if adapter_dir:
        from peft import PeftModel

        logger.info("Attaching LoRA adapters from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.to("cuda")
        FastVisionModel.for_inference(model)
    return model, processor


@torch.inference_mode()
def predict(model, processor, image: Image.Image, question: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,  # concise answers; see README gotcha
    )
    inputs = processor(images=image, text=text, add_special_tokens=False, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="ANLS evaluation on DocVQA")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter-dir", type=str, default=None, help="Optional LoRA adapter dir")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-image-edge", type=int, default=1344)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--output", type=Path, default=Path("eval_results.json"))
    args = p.parse_args()

    rows = load_eval_rows(args.split, args.num_samples, args.seed)
    model, processor = load_model(args.model_id, args.adapter_dir, args.load_in_4bit and not args.no_4bit)

    records: List[Dict[str, Any]] = []
    anls_sum, em_sum = 0.0, 0.0
    for n, row in enumerate(rows, 1):
        image = resize_image(row["image"], args.max_image_edge)
        try:
            pred = predict(model, processor, image, row["question"], args.max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed on row %s: %s", row["idx"], exc)
            pred = ""
        s_anls = anls_single(pred, row["answers"])
        s_em = exact_match(pred, row["answers"])
        anls_sum += s_anls
        em_sum += s_em
        records.append(
            {
                "idx": row["idx"],
                "question": row["question"],
                "golds": row["answers"],
                "prediction": pred,
                "anls": round(s_anls, 4),
                "em": s_em,
            }
        )
        if n % 25 == 0 or n == len(rows):
            logger.info(
                "Progress %d/%d | running ANLS=%.4f EM=%.4f",
                n, len(rows), anls_sum / n, em_sum / n,
            )

    n = max(1, len(records))
    summary = {
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "split": args.split,
        "num_samples": len(records),
        "seed": args.seed,
        "max_image_edge": args.max_image_edge,
        "anls": round(anls_sum / n, 4),
        "exact_match": round(em_sum / n, 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("=== RESULT: ANLS=%.4f | EM=%.4f | n=%d ===", summary["anls"], summary["exact_match"], n)
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
