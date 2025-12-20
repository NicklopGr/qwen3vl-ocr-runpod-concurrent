"""
Qwen3-VL-8B RunPod Serverless Handler

Uses vLLM for fast inference with Qwen3-VL-8B-Instruct model.
Extracts bank statement data into pipe-separated markdown format.

Model: Qwen/Qwen3-VL-8B-Instruct
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
from PIL import Image

# vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

print("[Qwen3-VL-8B] Loading model and processor...")
start_load = time.time()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Load processor for chat template
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Initialize vLLM with multimodal support
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.95,
    dtype="bfloat16",
    limit_mm_per_prompt={"image": 1},  # One image per prompt
)

print(f"[Qwen3-VL-8B] Model loaded in {time.time() - start_load:.2f}s")

# Optimized prompt for bank statement extraction
# Two-phase approach: identify column headers first, then extract data
BANK_STATEMENT_PROMPT = """Extract data from this bank statement image. Let's do this step by step.

STEP 1 - IDENTIFY TABLE COLUMNS:
First, read the column headers from left to right. Bank statements typically have:
- Column 1: Date
- Column 2: Description/Details
- Column 3: Withdrawals/Debits (money OUT - decreases balance)
- Column 4: Deposits/Credits (money IN - increases balance)
- Column 5: Balance

Report what you see:
COLUMNS: [list the actual column header names from the image, left to right]

STEP 2 - EXTRACT TRANSACTIONS:
For each row, place amounts in the correct column based on their POSITION in the table, NOT based on the description text.

CRITICAL RULES:
- An amount in Column 3 position = DEBIT (withdrawal)
- An amount in Column 4 position = CREDIT (deposit)
- Do NOT guess based on description - use ONLY column position
- If a cell is empty, leave it blank between pipes

Output format:
Bank: [bank name]
Account: [account number]
Period: [date range]

---TRANSACTIONS---
Date | Description | Debit | Credit | Balance
[transactions here]
---END---

Additional rules:
- Separator is | (pipe)
- Every transaction MUST have its date
- Description must be single line
- Include ALL visible transactions
- Keep amount format (e.g., 1,580.00)
"""

# Sampling parameters for deterministic output
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
)


def handler(job):
    """
    Process a bank statement image with Qwen3-VL-8B.
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

        print(f"[Qwen3-VL-8B] Processing image: {image.size[0]}x{image.size[1]}")

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

            print(f"[Qwen3-VL-8B] Completed in {processing_time:.2f}s, output: {len(markdown)} chars")

            return {
                "status": "success",
                "markdown": markdown,
                "processing_time": processing_time,
                "pipeline": "qwen3-vl-8b"
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

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
