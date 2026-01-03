# Qwen3-VL-8B RunPod Serverless Container
# Uses vLLM for fast inference

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    runpod \
    pillow \
    qwen-vl-utils==0.0.14

# Pre-download FP8 model during build (must match MODEL_NAME in handler.py)
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct-FP8')"

# Copy handler
COPY handler.py /app/handler.py

# RunPod configuration
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_INIT_TIMEOUT=600

# Start handler
CMD ["python", "-u", "/app/handler.py"]
