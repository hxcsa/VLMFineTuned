# Slim inference image for the DocVQA LoRA demo (Gradio).
# Training is NOT done in this image — it pulls the fine-tuned adapters
# from the Hugging Face Hub at container start and serves 3_app.py.
#
# Build:  docker build -t ghcr.io/<user>/vlmfinetuned-demo:latest .
# Run:    docker run --gpus all -p 7860:7860 ghcr.io/<user>/vlmfinetuned-demo:latest
#         (private base model/adapter repo? add: -e HF_TOKEN=hf_...)

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/huggingface

# Inference-only deps (no training stack). unsloth pulls the matching
# transformers/accelerate/bitsandbytes itself; keep gradio + peft explicit.
RUN pip install --no-cache-dir \
    unsloth \
    peft \
    gradio>=4.44.0 \
    hf_transfer>=0.1.6 \
    sentencepiece>=0.2.0 \
    protobuf>=4.25.0

WORKDIR /app
COPY 3_app.py /app/3_app.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Runtime-configurable: which base model + which adapter repo to serve.
ENV MODEL_ID="Qwen/Qwen3.5-4B" \
    ADAPTER_REPO="hxcsa/qwen35-4b-docvqa-lora" \
    PORT=7860

EXPOSE 7860

# Volumes for HF cache (mount to avoid re-downloading the base model):
#   docker run -v hf_cache:/root/.cache/huggingface ...
ENTRYPOINT ["/app/docker-entrypoint.sh"]
