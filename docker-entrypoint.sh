#!/usr/bin/env bash
# Pull LoRA adapters from HF Hub (tiny, ~30MB), then serve the Gradio demo.
set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/adapters}"

if [ -n "${ADAPTER_REPO:-}" ]; then
    echo "[entrypoint] Downloading adapters from ${ADAPTER_REPO} -> ${ADAPTER_DIR}"
    mkdir -p "${ADAPTER_DIR}"
    hf download "${ADAPTER_REPO}" --local-dir "${ADAPTER_DIR}"
fi

echo "[entrypoint] Launching demo: model=${MODEL_ID} port=${PORT}"
exec python /app/3_app.py \
    --model "${MODEL_ID}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}"
