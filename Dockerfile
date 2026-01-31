# Qwen-VL RunPod Serverless Container with Network Storage
# Model is loaded from network storage to reduce cold starts
# First run downloads model to network storage; subsequent runs use cached model

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    runpod \
    pillow \
    qwen-vl-utils==0.0.14 \
    huggingface_hub \
    transformers>=4.45.0 \
    accelerate

# Copy handler
COPY handler.py /app/handler.py

# RunPod configuration
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV VLLM_DISABLE_MODEL_SOURCE_CHECK=1
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm_cache

# Extended timeout for first-time model download to network storage
# First cold start may take 5-10 minutes to download 30B+ model
# Subsequent cold starts will be fast (model already on network storage)
ENV RUNPOD_INIT_TIMEOUT=900

# Maximum concurrent jobs per worker (default 5 for 30B model on 48GB GPU)
ENV MAX_CONCURRENCY=3

# Default model - can be overridden in RunPod endpoint settings
# Options:
#   QuantTrio/Qwen3-VL-32B-Instruct-AWQ       (~17GB, 8-bit AWQ, 32B dense)
#   QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ  (~17GB, 8-bit AWQ, MoE 30B/3B active)
#   Qwen/Qwen3-VL-8B-Instruct-FP8            (~8GB, FP8 quantized)
ENV MODEL_NAME="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"

# NOTE: Model is NOT pre-downloaded in this image
# Model will be downloaded to /runpod-volume/models/ on first run
# This keeps the Docker image small and allows easy model switching

# Start handler
CMD ["python", "-u", "/app/handler.py"]
