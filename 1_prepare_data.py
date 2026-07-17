#!/usr/bin/env python3
"""
1_prepare_data.py
-----------------
Download DocVQA, aggressively filter to exactly 1,000 high-quality samples,
format into Qwen2.5-VL conversational schema, and save as a local HF Dataset.

Target: single-GPU PEFT run (RTX 4090 24GB) without catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Prefer the requested subset; fall back to public DocVQA mirrors if gated/missing.
DATASET_CANDIDATES: Sequence[Dict[str, Any]] = (
    {"path": "nielsr/docvqa_1200k_sub", "name": None, "split": "train"},
    {"path": "lmms-lab/DocVQA", "name": "DocVQA", "split": "validation"},
    {"path": "HuggingFaceM4/DocumentVQA", "name": None, "split": "train"},
)

TARGET_SIZE = 1_000
# Cap long edge so ViT token count stays bounded on 24GB VRAM.
MAX_IMAGE_EDGE = 1024
# Drop ultra-wide / ultra-tall scans that explode sequence length.
MAX_ASPECT_RATIO = 4.0
# DocVQA answers are short; longer strings are often noisy OCR dumps.
MAX_ANSWER_CHARS = 256
MIN_QUESTION_CHARS = 5
SEED = 3407

SYSTEM_PROMPT = (
    "You are a precise document visual question answering assistant. "
    "Read the provided document image carefully and answer the user's question "
    "using only information visible in the image. "
    "Respond with a concise factual answer. Do not invent details."
)

DEFAULT_OUTPUT_DIR = Path("data/docvqa_1k_qwen_vl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DocVQA subset for Qwen2.5-VL PEFT")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-size", type=int, default=TARGET_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-image-edge", type=int, default=MAX_IMAGE_EDGE)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional HF dataset path override, e.g. lmms-lab/DocVQA",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional HF config/name (subset) for the dataset",
    )
    p.add_argument("--split", type=str, default=None, help="Optional split override")
    return p.parse_args()


def _first_nonempty(*values: Any) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        return v
    return None


def extract_question(sample: Dict[str, Any]) -> Optional[str]:
    q = _first_nonempty(
        sample.get("question"),
        sample.get("query"),
        sample.get("prompt"),
    )
    if q is None:
        return None
    q = str(q).strip()
    return q if len(q) >= MIN_QUESTION_CHARS else None


def extract_answer(sample: Dict[str, Any]) -> Optional[str]:
    """
    DocVQA stores multiple valid answers; pick the shortest non-empty one.
    Short gold answers train extractive VQA better than free-form paragraphs.
    """
    raw = _first_nonempty(
        sample.get("answers"),
        sample.get("answer"),
        sample.get("answers_text"),
        sample.get("ground_truth"),
    )
    if raw is None:
        return None

    candidates: List[str] = []
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                # Some schemas: {"text": "...", "answer_start": ...}
                text = item.get("text") or item.get("answer")
                if text is not None:
                    candidates.append(str(text))
            else:
                candidates.append(str(item))
    else:
        candidates = [str(raw)]

    cleaned = [c.strip() for c in candidates if c and str(c).strip()]
    if not cleaned:
        return None

    # Prefer short, non-empty answers (reduces label noise / over-generation).
    cleaned.sort(key=lambda s: (len(s), s.lower()))
    answer = cleaned[0]
    if len(answer) > MAX_ANSWER_CHARS:
        return None
    return answer


def extract_image(sample: Dict[str, Any]) -> Optional[Image.Image]:
    img = _first_nonempty(
        sample.get("image"),
        sample.get("document"),
        sample.get("img"),
    )
    if img is None:
        # Some datasets nest under "images"
        images = sample.get("images")
        if isinstance(images, (list, tuple)) and images:
            img = images[0]
    if img is None:
        return None

    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, dict) and "bytes" in img:
        from io import BytesIO

        pil = Image.open(BytesIO(img["bytes"]))
    elif isinstance(img, str) and os.path.isfile(img):
        pil = Image.open(img)
    else:
        # datasets Image feature already decoded in most loaders
        try:
            pil = img.convert("RGB") if hasattr(img, "convert") else Image.fromarray(img)
        except Exception:
            return None

    try:
        pil = pil.convert("RGB")
    except Exception:
        return None
    return pil


def resize_image(image: Image.Image, max_edge: int) -> Image.Image:
    """
    Keep aspect ratio; shrink only if longer edge exceeds max_edge.
    Why: Qwen-VL image tokens scale with resolution; 1024 max edge is a
    practical VRAM/quality tradeoff for document pages on 24GB.
    """
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image
    scale = max_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # LANCZOS preserves text edges better than bilinear for OCR/VQA.
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def aspect_ratio_ok(image: Image.Image) -> bool:
    w, h = image.size
    if w == 0 or h == 0:
        return False
    ratio = max(w, h) / min(w, h)
    return ratio <= MAX_ASPECT_RATIO


def quality_score(sample: Dict[str, Any], image: Image.Image, question: str, answer: str) -> float:
    """
    Higher is better. Used to rank candidates before hard cap at 1k.
    Rationale: short extractive answers + readable image sizes transfer best
    for PEFT with tiny data budgets and reduce forgetting.
    """
    score = 0.0
    # Prefer concise answers (DocVQA style).
    score += max(0.0, 40.0 - len(answer) * 0.15)
    # Prefer clear questions (not too short, not essay-length).
    qlen = len(question)
    if 10 <= qlen <= 120:
        score += 15.0
    elif qlen < 10:
        score -= 10.0

    w, h = image.size
    pixels = w * h
    # Sweet spot for document pages after resize.
    if 200_000 <= pixels <= 1_200_000:
        score += 20.0
    elif pixels < 80_000:
        score -= 20.0

    # Prefer samples with explicit answer lists (higher annotation quality).
    answers = sample.get("answers")
    if isinstance(answers, (list, tuple)) and len(answers) >= 2:
        score += 5.0

    # Prefer typed questions if present (table/form/layout etc.).
    qtypes = sample.get("question_types") or sample.get("question_type")
    if qtypes:
        score += 3.0

    return score


def to_qwen_messages(
    image: Image.Image,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Exact conversational schema expected by Unsloth FastVisionModel / Qwen-VL:
      messages = [
        {role: system, content: [{type:text, text:...}]},
        {role: user,   content: [{type:image, image:...}, {type:text, text:...}]},
        {role: assistant, content: [{type:text, text:...}]},
      ]
    Image must be a PIL object (not a path) for UnslothVisionDataCollator.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
    }


def load_docvqa(args: argparse.Namespace) -> Dataset:
    candidates: List[Dict[str, Any]] = []
    if args.dataset:
        candidates.append(
            {
                "path": args.dataset,
                "name": args.dataset_name,
                "split": args.split or "train",
            }
        )
    candidates.extend(list(DATASET_CANDIDATES))

    last_err: Optional[Exception] = None
    for cfg in candidates:
        path = cfg["path"]
        name = cfg.get("name")
        split = cfg.get("split") or "train"
        try:
            logger.info("Loading dataset path=%s name=%s split=%s", path, name, split)
            kwargs: Dict[str, Any] = {"path": path, "split": split}
            if name:
                kwargs["name"] = name
            if args.cache_dir:
                kwargs["cache_dir"] = args.cache_dir
            # trust_remote_code only if required by older dataset scripts
            ds = load_dataset(**kwargs)
            if isinstance(ds, DatasetDict):
                # Prefer train, then validation
                for key in ("train", "validation", "val", "test"):
                    if key in ds:
                        ds = ds[key]
                        break
                else:
                    ds = next(iter(ds.values()))
            logger.info("Loaded %d raw examples from %s", len(ds), path)
            return ds
        except Exception as exc:  # noqa: BLE001 — try next mirror
            last_err = exc
            logger.warning("Failed to load %s: %s", path, exc)

    raise RuntimeError(
        f"Could not load any DocVQA dataset candidate. Last error: {last_err}"
    )


def filter_and_format(
    raw: Dataset,
    target_size: int,
    max_image_edge: int,
    seed: int,
) -> Dataset:
    """
    Stream through raw rows, keep only high-quality (image, q, a) triples,
    rank by quality_score, keep top `target_size`.
    """
    scored: List[tuple[float, Dict[str, Any]]] = []
    skipped = {
        "no_question": 0,
        "no_answer": 0,
        "no_image": 0,
        "bad_aspect": 0,
        "errors": 0,
    }

    # Iterate without map() so PIL images stay as Python objects (Unsloth requirement).
    n = len(raw)
    logger.info("Scanning %d samples for quality filtering...", n)

    for idx in range(n):
        try:
            sample = raw[idx]
            question = extract_question(sample)
            if question is None:
                skipped["no_question"] += 1
                continue
            answer = extract_answer(sample)
            if answer is None:
                skipped["no_answer"] += 1
                continue
            image = extract_image(sample)
            if image is None:
                skipped["no_image"] += 1
                continue
            if not aspect_ratio_ok(image):
                skipped["bad_aspect"] += 1
                continue

            image = resize_image(image, max_image_edge)
            score = quality_score(sample, image, question, answer)
            formatted = to_qwen_messages(image, question, answer)
            # Keep lightweight metadata for debugging / audit.
            formatted["meta"] = {
                "quality_score": score,
                "question": question,
                "answer": answer,
                "image_size": list(image.size),
            }
            scored.append((score, formatted))
        except Exception as exc:  # noqa: BLE001
            skipped["errors"] += 1
            if skipped["errors"] <= 5:
                logger.debug("Row %d skipped due to error: %s", idx, exc)

        if (idx + 1) % 500 == 0:
            logger.info(
                "Progress %d/%d | kept_so_far=%d | skipped=%s",
                idx + 1,
                n,
                len(scored),
                skipped,
            )

        # Early exit if we already have a large pool (3x target) to rank from.
        if len(scored) >= target_size * 3:
            logger.info(
                "Collected %d candidates (>= 3x target); stopping scan early.",
                len(scored),
            )
            break

    if len(scored) < target_size:
        raise RuntimeError(
            f"Only {len(scored)} high-quality samples after filtering; "
            f"need {target_size}. Relax filters or use a larger source split. "
            f"Skip stats: {skipped}"
        )

    # Deterministic ranking: score desc, then question text for stability.
    scored.sort(key=lambda t: (-t[0], t[1]["meta"]["question"]))
    top = [item for _, item in scored[:target_size]]

    # Shuffle final 1k so training order is not quality-sorted (avoids curriculum bias).
    # Use Dataset.shuffle after construction for reproducibility via seed.
    logger.info(
        "Selected top %d / %d candidates. Skip stats: %s",
        len(top),
        len(scored),
        skipped,
    )

    # Build HF Dataset from list of dicts. Images remain PIL in the `messages` field.
    # Note: saving PIL inside nested messages works with datasets >= 2.14 via pickle/arrow.
    ds = Dataset.from_list(top)
    ds = ds.shuffle(seed=seed)
    assert len(ds) == target_size, f"Expected {target_size}, got {len(ds)}"
    return ds


def save_dataset(ds: Dataset, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # save_to_disk persists Arrow table + features for offline train reload.
    ds.save_to_disk(str(output_dir))
    logger.info("Saved %d examples -> %s", len(ds), output_dir)

    # Write a tiny human-readable preview (no images).
    preview_path = output_dir / "preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as f:
        for i in range(min(5, len(ds))):
            row = ds[i]
            meta = row.get("meta", {})
            f.write(
                f'{{"idx": {i}, "question": {meta.get("question")!r}, '
                f'"answer": {meta.get("answer")!r}, '
                f'"score": {meta.get("quality_score")}}}\n'
            )
    logger.info("Wrote preview -> %s", preview_path)


def main() -> int:
    args = parse_args()
    logger.info("CUDA not required for data prep; CPU-only is fine.")

    try:
        raw = load_docvqa(args)
        ds = filter_and_format(
            raw=raw,
            target_size=args.target_size,
            max_image_edge=args.max_image_edge,
            seed=args.seed,
        )
        save_dataset(ds, args.output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data preparation failed: %s", exc)
        return 1

    logger.info("Done. Next: python 2_train.py --dataset-dir %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
