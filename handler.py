"""
Qwen3-VL RunPod Serverless Handler with Network Storage

Uses vLLM for fast inference with network storage to reduce cold starts.
Model is downloaded once to network storage and reused across cold starts.

Model: QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ (30B params, 3B active, 8-bit AWQ)
Framework: vLLM
Input: {"image_base64": "..."}
Output: {"markdown": "Bank: ...\n---TRANSACTIONS---\n..."}
"""

import runpod
import base64
import io
import time
import tempfile
import os
from pathlib import Path
from PIL import Image

# ============================================
# NETWORK STORAGE CONFIGURATION
# ============================================
NETWORK_VOLUME = "/runpod-volume"
MODEL_NAME = os.environ.get("MODEL_NAME", "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ")

# Convert model name to safe directory name (replace / with --)
MODEL_DIR_NAME = MODEL_NAME.replace("/", "--")
MODEL_LOCAL_PATH = os.path.join(NETWORK_VOLUME, "models", MODEL_DIR_NAME)

def download_model_to_network_storage():
    """
    Download model to network storage if not already present.
    This runs once on first cold start, then all subsequent starts use cached model.
    """
    model_path = Path(MODEL_LOCAL_PATH)

    # Check if network volume is mounted
    if not os.path.exists(NETWORK_VOLUME):
        print(f"[WARNING] Network volume not mounted at {NETWORK_VOLUME}")
        print(f"[WARNING] Falling back to HuggingFace cache (slower cold starts)")
        return MODEL_NAME  # Return HF model ID for download to default cache

    # Check if model already exists on network storage
    if model_path.exists() and any(model_path.iterdir()):
        print(f"[Qwen-VL] Model found on network storage: {MODEL_LOCAL_PATH}")
        return MODEL_LOCAL_PATH

    # Model not found - download to network storage
    print(f"[Qwen-VL] Model not found on network storage. Downloading {MODEL_NAME}...")
    print(f"[Qwen-VL] This is a one-time download. Future cold starts will use cached model.")

    from huggingface_hub import snapshot_download

    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Download model to network storage
    download_start = time.time()
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_LOCAL_PATH,
        local_dir_use_symlinks=False,  # Copy files, don't symlink
    )
    download_time = time.time() - download_start
    print(f"[Qwen-VL] Model downloaded in {download_time:.1f}s to {MODEL_LOCAL_PATH}")

    return MODEL_LOCAL_PATH

# ============================================
# MODEL LOADING
# ============================================
print(f"[Qwen-VL] Starting model loading process...")
print(f"[Qwen-VL] Configured model: {MODEL_NAME}")
print(f"[Qwen-VL] Network storage path: {MODEL_LOCAL_PATH}")

# Get model path (either network storage or HF cache)
model_path = download_model_to_network_storage()

# vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

print(f"[Qwen-VL] Loading model from: {model_path}")
start_load = time.time()

# Load processor for chat template
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Initialize vLLM with multimodal support
# Adjust settings based on model size
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.95,
    dtype="auto",  # Let vLLM choose based on quantization
    limit_mm_per_prompt={"image": 1},  # One image per prompt
    quantization="awq" if "AWQ" in MODEL_NAME.upper() else None,
)

print(f"[Qwen-VL] Model loaded in {time.time() - start_load:.2f}s")

# ============================================
# PROMPTS AND PARAMETERS
# ============================================

# Optimized prompt for bank statement extraction
# Uses neutral column names to prevent semantic guessing
BANK_STATEMENT_PROMPT = """Extract data from this bank statement image EXACTLY as it appears. You are a scanner, not an interpreter.

STEP 1 - IDENTIFY TABLE COLUMNS:
Read the column headers from left to right. The table has 5 columns:
- Column 1: Date
- Column 2: Description/Details
- Column 3: First amount column (left amount column)
- Column 4: Second amount column (right amount column)
- Column 5: Balance

Report what you see:
COLUMNS: [list the actual column header names from the image, left to right]

STEP 2 - EXTRACT TRANSACTIONS:
Copy each row EXACTLY as it appears. Do not interpret or analyze meanings.

CRITICAL RULES:
- Copy amounts EXACTLY as they appear in their column position
- Amount in Column 3 position → write in Col3 output field
- Amount in Column 4 position → write in Col4 output field
- DO NOT interpret what the amount means - just copy its position
- If a cell is empty, leave it blank between pipes
- You are copying positions, NOT analyzing transaction types

Output format:
Bank: [bank name]
Account: [account number]
Period: [date range]

---TRANSACTIONS---
Date | Description | Col3 | Col4 | Balance
[transactions here, exactly as they appear]
---END---

Additional rules:
- Separator is | (pipe)
- Every row MUST have its date - if same as previous row, still include it
- Description must be single line (as it appears)
- Include ALL visible transactions
- Keep amount format exactly (e.g., 1,580.00)
- Extract as they appear - do not reorder or interpret
"""

# Sampling parameters for deterministic output
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)


def handler(job):
    """
    Process a bank statement image with Qwen-VL.
    """
    start_time = time.time()

    try:
        job_input = job.get("input", {})
        image_base64 = job_input.get("image_base64")
        custom_prompt = job_input.get("prompt")

        if not image_base64:
            return {"error": "Missing image_base64 in input"}

        # Decode image and save to temp file (required for process_vision_info)
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        print(f"[Qwen-VL] Processing image: {image.size[0]}x{image.size[1]}")

        # Save image to temp file for vLLM multimodal processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file, format="PNG")
            temp_image_path = tmp_file.name

        try:
            # Use custom prompt or default
            prompt = custom_prompt if custom_prompt else BANK_STATEMENT_PROMPT

            # Create message with file path for process_vision_info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temp_image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template to get the prompt text
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision info to get image data for vLLM
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True
            )

            # Create vLLM input with multimodal data
            llm_inputs = {
                "prompt": text,
                "multi_modal_data": {
                    "image": image_inputs
                }
            }

            # Generate response using vLLM
            outputs = llm.generate(
                [llm_inputs],
                sampling_params=sampling_params,
            )

            # Extract generated text
            markdown = outputs[0].outputs[0].text

            # Append ---END--- if model stopped early
            if "---END---" not in markdown:
                markdown += "\n---END---"

            processing_time = time.time() - start_time

            print(f"[Qwen-VL] Completed in {processing_time:.2f}s, output: {len(markdown)} chars")

            return {
                "status": "success",
                "markdown": markdown,
                "processing_time": processing_time,
                "pipeline": f"qwen-vl ({MODEL_NAME})",
                "model_location": "network_storage" if model_path == MODEL_LOCAL_PATH else "hf_cache"
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        import traceback
        print(f"[Qwen-VL] Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "processing_time": time.time() - start_time
        }


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
