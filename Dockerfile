# Qwen3-VL-8B RunPod Serverless Container
# Uses vLLM for fast inference with vision-language model

FROM vllm/vllm-openai:v0.11.0

# Set working directory
WORKDIR /app

# Install additional dependencies for RunPod
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    qwen-vl-utils==0.0.14

# Pre-download the model during build (optional but recommended)
# This reduces cold start time on RunPod
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Download model weights during build
# Uncomment one of the following based on your GPU memory:

# For RTX 4090 / A10 (24GB) - Use FP8 for lower VRAM
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct-FP8')"

# For A100 (40GB+) - Use full precision
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"

# Copy handler
COPY handler.py /app/handler.py

# RunPod configuration
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Increase timeout for model loading
ENV RUNPOD_INIT_TIMEOUT=600

# Start handler
CMD ["python", "-u", "/app/handler.py"]
