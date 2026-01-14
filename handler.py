"""
Qwen3-VL RunPod Serverless Handler with Network Storage

Uses vLLM for fast inference with network storage to reduce cold starts.
Model is downloaded once to network storage and reused across cold starts.
Supports batch processing of multiple images using 1 GPU.

Model: QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ (30B params, 3B active, 8-bit AWQ)
Framework: vLLM with tensor_parallel_size=1, max_model_len=14336
Input:
  Single: {"image_base64": "..."} or {"image_url": "..."}
  Batch:  {"images": [{"image_base64": "...", "prompt": "..."}, ...]}
          {"images": [{"image_url": "...", "prompt": "..."}, ...]}
Output: {"markdown": "..."} or {"results": [...]}
"""

import runpod
import base64
import io
import time
import tempfile
import os
import requests
from pathlib import Path
from PIL import Image

# ============================================
# CONFIGURATION
# ============================================
NETWORK_VOLUME = "/runpod-volume"
MODEL_NAME = os.environ.get("MODEL_NAME", "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))  # Use 1 GPU (avoids shared memory sync issues)

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

# Initialize vLLM with multimodal support
# Using 1 GPU with 14K context to avoid shared memory sync timeouts
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=14336,  # 14K tokens (reduced from 32K to prevent OOM/timeouts)
    gpu_memory_utilization=0.95,
    dtype="auto",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,  # Split model across GPUs
    distributed_executor_backend="mp",  # Required for multi-GPU on RunPod serverless
    limit_mm_per_prompt={"image": 1},
    quantization="awq_marlin" if "AWQ" in MODEL_NAME.upper() else None,  # awq_marlin is faster than awq
)

print(f"[Qwen-VL] Model loaded in {time.time() - start_load:.2f}s across {TENSOR_PARALLEL_SIZE} GPUs")

# ============================================
# PROMPTS AND PARAMETERS
# ============================================

# ============================================
# PROMPT V1 - Legacy (backed up 2025-01-14)
# Uses generic Col3/Col4 output, requires extractor pattern matching
# ============================================
BANK_STATEMENT_PROMPT_V1 = """Extract data from this bank statement image EXACTLY as it appears. You are a scanner, not an interpreter.

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
Account_Owner: [name of account holder if visible, otherwise leave blank]
Period: [date range]
Opening_Balance: [opening/beginning balance if shown, otherwise leave blank]
Closing_Balance: [closing/ending balance if shown, otherwise leave blank]

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
- Only include Opening_Balance and Closing_Balance if explicitly shown on the statement - DO NOT calculate them
"""

# ============================================
# PROMPT V2 - Header-based semantic output (2025-01-14)
# Qwen maps headers to semantic columns, outputs Debit/Credit directly
# ============================================
BANK_STATEMENT_PROMPT = """Extract data from this bank statement image.

STEP 1 - READ THE COLUMN HEADERS:
Look at the table header row. Identify each column by reading its header text exactly.
Bank statements typically have columns for: Date, Description, Debits/Withdrawals, Credits/Deposits, Balance.

Report the headers you see:
HEADERS: [Column1Header, Column2Header, Column3Header, Column4Header, Column5Header]

Map them to semantic columns:
- DATE_COL: [position 1-5] = "[header text]"
- DESC_COL: [position 1-5] = "[header text]"
- DEBIT_COL: [position 1-5] = "[header text]" (look for: withdrawals, debits, cheques, payments, money out)
- CREDIT_COL: [position 1-5] = "[header text]" (look for: deposits, credits, receipts, money in)
- BALANCE_COL: [position 1-5] = "[header text]"

STEP 2 - EXTRACT TRANSACTIONS:
A TRANSACTION is defined as a row that has an AMOUNT (either Debit or Credit).
For each transaction:
- Copy values from their source columns to our structured output
- Value from DEBIT_COL → goes to Debit in output
- Value from CREDIT_COL → goes to Credit in output

IMPORTANT - MULTI-LINE DESCRIPTIONS:
Some transactions have descriptions that span multiple lines on the statement.
You MUST combine them into ONE row in the output:
- If a transaction's description continues on the next line(s), merge all description text into a single Description field
- The output must have ONE ROW PER TRANSACTION (one row per amount)
- Lines without amounts are description continuations - combine them with the transaction they belong to

Example: If the statement shows:
  "15 Apr | PAYROLL DEPOSIT    |        | 5,000.00 |
         | ACME CORP REF#123  |        |          |"
Output as ONE row:
  "15 Apr | PAYROLL DEPOSIT ACME CORP REF#123 | | 5,000.00 |"

Output format:
Bank: [bank name]
Account: [account number]
Account_Owner: [name if visible]
Period: [date range]
Opening_Balance: [if explicitly shown]
Closing_Balance: [if explicitly shown]

---TRANSACTIONS---
Date | Description | Debit | Credit | Balance
[ONE ROW PER TRANSACTION - combine multi-line descriptions]
---END---

CRITICAL RULES:
- ONE ROW PER TRANSACTION: Each output row must have an amount (Debit or Credit). Never output rows without amounts.
- COMBINE DESCRIPTIONS: If description spans multiple lines, merge into single Description field
- Determine Debit vs Credit ONLY from column headers, NOT from transaction descriptions
- If column header says "Withdrawals" or "Debits" or "Cheques" → that's the Debit column
- If column header says "Deposits" or "Credits" → that's the Credit column
- Copy amounts exactly as they appear (keep formatting like 1,580.00)
- If a cell is empty, leave it blank between pipes
- Every row must have its date (if same as previous, still include it)
- Include ALL visible transactions
- Do NOT calculate or infer Opening/Closing Balance - only include if explicitly shown
- If only ONE amount column exists: negative or (parentheses) amounts → Debit, positive → Credit
- SKIP these non-transaction rows: "Balance Forward", "Opening Balance", "Closing Balance", "Monthly Average", summary lines
- BALANCE COLUMN: Only include a balance if that SPECIFIC row has a balance printed next to it.
  - Do NOT repeat the same balance across multiple rows
  - Do NOT copy closing balance to transaction rows that don't show it
  - If unsure, leave Balance BLANK
"""

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)


def download_image_from_url(url: str, timeout: int = 60) -> bytes:
    """Download image from URL (supports SAS URLs from Azure Blob)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def process_single_image(image_base64=None, image_url=None, custom_prompt=None):
    """Process a single image and return the result.

    Supports both base64 and URL input modes.
    """
    # Get image data from either base64 or URL
    if image_url:
        print(f"[Qwen-VL] Downloading image from URL...")
        image_data = download_image_from_url(image_url)
    elif image_base64:
        image_data = base64.b64decode(image_base64)
    else:
        raise ValueError("Either image_base64 or image_url must be provided")

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

    Supports multiple input modes:
    1. Single base64: {"image_base64": "...", "prompt": "..."}
    2. Single URL: {"image_url": "...", "prompt": "..."}
    3. Batch base64: {"images": [{"image_base64": "...", "prompt": "..."}, ...]}
    4. Batch URL: {"images": [{"image_url": "...", "prompt": "..."}, ...]}

    URL mode supports Azure Blob SAS URLs for secure, large-scale processing.
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

            # Detect input mode (URL or base64)
            first_img = images_batch[0] if images_batch else {}
            input_mode = "url" if first_img.get("image_url") else "base64"
            print(f"[Qwen-VL] Input mode: {input_mode}")

            llm_inputs = []
            image_sizes = []

            for i, img_data in enumerate(images_batch):
                img_base64 = img_data.get("image_base64")
                img_url = img_data.get("image_url")
                img_prompt = img_data.get("prompt")

                if not img_base64 and not img_url:
                    print(f"[Qwen-VL] Skipping image {i}: missing image_base64 or image_url")
                    continue

                llm_input, temp_path, size = process_single_image(
                    image_base64=img_base64,
                    image_url=img_url,
                    custom_prompt=img_prompt
                )
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
            image_url = job_input.get("image_url")
            custom_prompt = job_input.get("prompt")

            if not image_base64 and not image_url:
                return {"error": "Missing image_base64 or image_url in input"}

            input_mode = "url" if image_url else "base64"
            print(f"[Qwen-VL] Single mode, input: {input_mode}")

            llm_input, temp_path, size = process_single_image(
                image_base64=image_base64,
                image_url=image_url,
                custom_prompt=custom_prompt
            )
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
