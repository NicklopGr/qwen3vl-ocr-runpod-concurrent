"""
Qwen3-VL-8B RunPod Serverless Handler

Uses vLLM for fast inference with Qwen3-VL-8B-Instruct model.
Extracts bank statement data into pipe-separated markdown format.

Model: Qwen/Qwen3-VL-8B-Instruct (or FP8 variant)
Framework: vLLM >= 0.11.0
Input: {"image_base64": "..."}
Output: {"markdown": "Bank: ...\n---TRANSACTIONS---\n..."}
"""

import runpod
import base64
import io
import time
from PIL import Image

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalData

print("[Qwen3-VL-8B] Loading model...")
start_load = time.time()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"  # Or use FP8: "Qwen/Qwen3-VL-8B-Instruct-FP8"

# Initialize vLLM with Qwen3-VL-8B
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=8192,  # Reasonable context for single page
    gpu_memory_utilization=0.9,
    dtype="auto",  # Will use bfloat16 on supported GPUs
)

print(f"[Qwen3-VL-8B] Model loaded in {time.time() - start_load:.2f}s")

# Optimized prompt for bank statement extraction
BANK_STATEMENT_PROMPT = """Extract all data from this bank statement image.

Output format:
Bank: [bank name]
Account: [account number]
Period: [date range]

---TRANSACTIONS---
Date | Description | Debit | Credit | Balance
[one transaction per line]
---END---

Rules:
- Separator is | (pipe), NOT comma
- Empty column = leave blank between pipes
- Every transaction MUST have its date (even if repeated from previous row)
- Description must be single line (no line breaks)
- Include ALL visible transactions
- For amounts, keep the original format (e.g., 1,580.00)
"""

# Sampling parameters for deterministic output
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic
    max_tokens=4096,  # Enough for most pages
    stop=["---END---"],  # Stop after transactions
)


def handler(job):
    """
    Process a bank statement image with Qwen3-VL-8B.

    Args:
        job: RunPod job with input containing:
            - image_base64: Base64 encoded image (PNG/JPG)
            - prompt: Optional custom prompt (defaults to bank statement extraction)

    Returns:
        dict with:
            - markdown: Extracted markdown text
            - processing_time: Time taken in seconds
            - pipeline: "qwen3-vl-8b"
    """
    start_time = time.time()

    try:
        job_input = job.get("input", {})
        image_base64 = job_input.get("image_base64")
        custom_prompt = job_input.get("prompt")

        if not image_base64:
            return {"error": "Missing image_base64 in input"}

        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        print(f"[Qwen3-VL-8B] Processing image: {image.size[0]}x{image.size[1]}")

        # Use custom prompt or default
        prompt = custom_prompt if custom_prompt else BANK_STATEMENT_PROMPT

        # Format prompt for Qwen3-VL (chat format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Generate response
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
        )

        # Extract generated text
        markdown = outputs[0].outputs[0].text

        # Append ---END--- if model stopped early
        if "---END---" not in markdown:
            markdown += "\n---END---"

        processing_time = time.time() - start_time

        print(f"[Qwen3-VL-8B] Completed in {processing_time:.2f}s, output: {len(markdown)} chars")

        return {
            "status": "success",
            "markdown": markdown,
            "processing_time": processing_time,
            "pipeline": "qwen3-vl-8b"
        }

    except Exception as e:
        import traceback
        print(f"[Qwen3-VL-8B] Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "processing_time": time.time() - start_time
        }


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
