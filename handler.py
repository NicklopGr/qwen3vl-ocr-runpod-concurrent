"""
Qwen3-VL RunPod Serverless Handler with Network Storage

Uses vLLM for fast inference with network storage to reduce cold starts.
Model is downloaded once to network storage and reused across cold starts.
Supports batch processing of multiple images using 2 GPUs with tensor parallelism.

Model: QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ (30B params, 3B active, 8-bit AWQ)
Framework: vLLM with tensor_parallel_size=2
Input:
  Single: {"image_base64": "..."}
  Batch:  {"images": [{"image_base64": "...", "prompt": "..."}, ...]}
Output: {"markdown": "..."} or {"results": [...]}
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
# CONFIGURATION
# ============================================
NETWORK_VOLUME = "/runpod-volume"
MODEL_NAME = os.environ.get("MODEL_NAME", "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))  # Use 2 GPUs

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
print(f"[Qwen-VL] Tensor parallel size: {TENSOR_PARALLEL_SIZE} GPUs")
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

# Initialize vLLM with multimodal support and tensor parallelism
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.95,
    dtype="auto",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,  # Split model across GPUs
    limit_mm_per_prompt={"image": 1},
    quantization="awq" if "AWQ" in MODEL_NAME.upper() else None,
)

print(f"[Qwen-VL] Model loaded in {time.time() - start_load:.2f}s across {TENSOR_PARALLEL_SIZE} GPUs")

# ============================================
# PROMPTS AND PARAMETERS
# ============================================

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

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)


def process_single_image(image_base64, custom_prompt=None):
    """Process a single image and return the result."""
    # Decode image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file, format="PNG")
        temp_image_path = tmp_file.name

    prompt = custom_prompt if custom_prompt else BANK_STATEMENT_PROMPT

    # Create message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True
    )

    # Create vLLM input
    llm_input = {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs}
    }

    return llm_input, temp_image_path, image.size


def handler(job):
    """
    Process bank statement images with Qwen-VL.

    Supports two input modes:
    1. Single image: {"image_base64": "...", "prompt": "..."}
    2. Batch images: {"images": [{"image_base64": "...", "prompt": "..."}, ...]}

    Batch mode processes all images in parallel using vLLM batching.
    """
    start_time = time.time()
    temp_files = []

    try:
        job_input = job.get("input", {})

        # Check for batch mode
        images_batch = job_input.get("images")

        if images_batch and isinstance(images_batch, list):
            # ============================================
            # BATCH MODE: Process multiple images at once
            # ============================================
            print(f"[Qwen-VL] Batch mode: processing {len(images_batch)} images")

            llm_inputs = []
            image_sizes = []

            for i, img_data in enumerate(images_batch):
                img_base64 = img_data.get("image_base64")
                img_prompt = img_data.get("prompt")

                if not img_base64:
                    print(f"[Qwen-VL] Skipping image {i}: missing image_base64")
                    continue

                llm_input, temp_path, size = process_single_image(img_base64, img_prompt)
                llm_inputs.append(llm_input)
                temp_files.append(temp_path)
                image_sizes.append(size)
                print(f"[Qwen-VL] Prepared image {i+1}: {size[0]}x{size[1]}")

            if not llm_inputs:
                return {"error": "No valid images in batch"}

            # Generate all responses in parallel batch
            print(f"[Qwen-VL] Running batch inference on {len(llm_inputs)} images...")
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

            # Collect results
            results = []
            for i, output in enumerate(outputs):
                markdown = output.outputs[0].text
                if "---END---" not in markdown:
                    markdown += "\n---END---"
                results.append({
                    "index": i,
                    "status": "success",
                    "markdown": markdown,
                    "image_size": f"{image_sizes[i][0]}x{image_sizes[i][1]}"
                })

            processing_time = time.time() - start_time
            print(f"[Qwen-VL] Batch completed in {processing_time:.2f}s ({len(results)} images)")

            return {
                "status": "success",
                "mode": "batch",
                "results": results,
                "total_images": len(results),
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results) if results else 0,
                "pipeline": f"qwen-vl ({MODEL_NAME})",
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "model_location": "network_storage" if model_path == MODEL_LOCAL_PATH else "hf_cache"
            }

        else:
            # ============================================
            # SINGLE MODE: Process one image
            # ============================================
            image_base64 = job_input.get("image_base64")
            custom_prompt = job_input.get("prompt")

            if not image_base64:
                return {"error": "Missing image_base64 in input"}

            llm_input, temp_path, size = process_single_image(image_base64, custom_prompt)
            temp_files.append(temp_path)

            print(f"[Qwen-VL] Processing single image: {size[0]}x{size[1]}")

            # Generate response
            outputs = llm.generate([llm_input], sampling_params=sampling_params)
            markdown = outputs[0].outputs[0].text

            if "---END---" not in markdown:
                markdown += "\n---END---"

            processing_time = time.time() - start_time
            print(f"[Qwen-VL] Completed in {processing_time:.2f}s, output: {len(markdown)} chars")

            return {
                "status": "success",
                "mode": "single",
                "markdown": markdown,
                "processing_time": processing_time,
                "pipeline": f"qwen-vl ({MODEL_NAME})",
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "model_location": "network_storage" if model_path == MODEL_LOCAL_PATH else "hf_cache"
            }

    except Exception as e:
        import traceback
        print(f"[Qwen-VL] Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "processing_time": time.time() - start_time
        }

    finally:
        # Clean up all temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
