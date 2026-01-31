"""
Qwen3-VL RunPod Serverless Handler (Concurrent Version)

1 image = 1 page. No tiling, no multi-image combining.
Multiple jobs run simultaneously on the same GPU worker via concurrency_modifier.

Key design:
- concurrency_modifier allows multiple RunPod jobs on the same worker
- All inference requests are collected into a single queue
- A background task batches queued requests into ONE llm.generate() call
- vLLM's LLM.generate() is NOT thread-safe; we never call it from multiple threads

Model: Qwen/Qwen3-VL-8B-Instruct-FP8 (8B dense, FP8 quantized, ~8GB VRAM)
Framework: vLLM with tensor_parallel_size=1, max_model_len=16384
"""

import runpod
import asyncio
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
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct-FP8")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "5"))

# How long (seconds) the batch processor waits to collect more requests before firing
BATCH_WAIT_SECONDS = float(os.environ.get("BATCH_WAIT_SECONDS", "0.5"))

MODEL_DIR_NAME = MODEL_NAME.replace("/", "--")
MODEL_LOCAL_PATH = os.path.join(NETWORK_VOLUME, "models", MODEL_DIR_NAME)

# ============================================
# INFERENCE QUEUE (single-writer pattern)
# ============================================
# Each item: {"llm_input": dict, "future": asyncio.Future}
_inference_queue: asyncio.Queue | None = None
_batch_processor_task: asyncio.Task | None = None


def download_model_to_network_storage():
    """Download model to network storage if not already present.
    Cleans up any other models on the network volume to free space."""
    model_path = Path(MODEL_LOCAL_PATH)

    if not os.path.exists(NETWORK_VOLUME):
        print(f"[WARNING] Network volume not mounted at {NETWORK_VOLUME}")
        print(f"[WARNING] Falling back to HuggingFace cache (slower cold starts)")
        return MODEL_NAME

    # Clean up old/different models from network volume
    models_dir = Path(NETWORK_VOLUME) / "models"
    if models_dir.exists():
        for existing in models_dir.iterdir():
            if existing.is_dir() and existing.name != MODEL_DIR_NAME:
                import shutil
                print(f"[Qwen-VL] Removing old model: {existing.name}")
                shutil.rmtree(existing, ignore_errors=True)

    if model_path.exists() and any(model_path.iterdir()):
        print(f"[Qwen-VL] Model found on network storage: {MODEL_LOCAL_PATH}")
        return MODEL_LOCAL_PATH

    print(f"[Qwen-VL] Model not found on network storage. Downloading {MODEL_NAME}...")
    print(f"[Qwen-VL] This is a one-time download. Future cold starts will use cached model.")

    from huggingface_hub import snapshot_download

    model_path.parent.mkdir(parents=True, exist_ok=True)

    download_start = time.time()
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_LOCAL_PATH,
        local_dir_use_symlinks=False,
    )
    download_time = time.time() - download_start
    print(f"[Qwen-VL] Model downloaded in {download_time:.1f}s to {MODEL_LOCAL_PATH}")

    return MODEL_LOCAL_PATH


# ============================================
# MODEL LOADING (runs once at container startup)
# ============================================
print(f"[Qwen-VL] Starting model loading process (concurrent handler, max_concurrency={MAX_CONCURRENCY})...")
print(f"[Qwen-VL] Configured model: {MODEL_NAME}")
print(f"[Qwen-VL] Tensor parallel size: {TENSOR_PARALLEL_SIZE} GPUs")
print(f"[Qwen-VL] Network storage path: {MODEL_LOCAL_PATH}")

model_path = download_model_to_network_storage()

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

print(f"[Qwen-VL] Loading model from: {model_path}")
start_load = time.time()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=16384,
    max_num_seqs=MAX_CONCURRENCY * 2,    # Allow 2x concurrency for sequence scheduling headroom
    gpu_memory_utilization=0.95,
    dtype="auto",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    distributed_executor_backend="mp",
    limit_mm_per_prompt={"image": 1},
    quantization="fp8" if "FP8" in MODEL_NAME.upper() else ("awq_marlin" if "AWQ" in MODEL_NAME.upper() else None),
)

print(f"[Qwen-VL] Model loaded in {time.time() - start_load:.2f}s across {TENSOR_PARALLEL_SIZE} GPUs")

# ============================================
# PROMPTS AND PARAMETERS
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
    """Process a single image from base64 or URL. Returns llm_input, temp_path, size."""
    if image_url:
        print(f"[Qwen-VL] Downloading image from URL...")
        image_data = download_image_from_url(image_url)
    elif image_base64:
        image_data = base64.b64decode(image_base64)
    else:
        raise ValueError("Either image_base64 or image_url must be provided")

    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file, format="PNG")
        temp_image_path = tmp_file.name

    prompt = custom_prompt if custom_prompt else BANK_STATEMENT_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    llm_input = {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs}
    }

    return llm_input, temp_image_path, image.size


# ============================================
# BATCH INFERENCE PROCESSOR
# ============================================
# Single background task that drains the queue and calls llm.generate() once per batch.
# This avoids the thread-safety issue with concurrent llm.generate() calls.

def _run_generate(llm_inputs):
    """Synchronous generate — runs in a single dedicated thread to keep event loop free."""
    return llm.generate(llm_inputs, sampling_params=sampling_params)


async def _batch_inference_loop():
    """Background task: collect queued requests, run ONE llm.generate() per batch.

    generate() is blocking (30-60s for vision batches), so we run it in a
    ThreadPoolExecutor.  Only this loop ever calls generate(), so there is
    never more than one concurrent call — thread-safety is preserved while
    the event loop stays responsive for new RunPod jobs arriving.
    """
    global _inference_queue
    print(f"[Qwen-VL] Batch inference loop started (wait={BATCH_WAIT_SECONDS}s, max_concurrency={MAX_CONCURRENCY})")

    loop = asyncio.get_event_loop()

    while True:
        # Wait for at least one request
        first_item = await _inference_queue.get()
        batch = [first_item]

        # Collect more requests that arrived while we waited
        await asyncio.sleep(BATCH_WAIT_SECONDS)
        while not _inference_queue.empty():
            try:
                item = _inference_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        # Build a single list of llm_inputs for one generate() call
        llm_inputs = [item["llm_input"] for item in batch]
        futures = [item["future"] for item in batch]

        print(f"[Qwen-VL] Batch generate: {len(llm_inputs)} prompts")
        gen_start = time.time()

        try:
            # Run in executor so event loop stays responsive for new jobs
            # Only ONE thread ever calls generate() — thread-safety preserved
            outputs = await loop.run_in_executor(None, _run_generate, llm_inputs)
            gen_time = time.time() - gen_start
            print(f"[Qwen-VL] Batch generate done: {len(outputs)} outputs in {gen_time:.2f}s")

            # Distribute results back to each waiting job
            for future, output in zip(futures, outputs):
                if not future.cancelled():
                    future.set_result(output)
        except Exception as e:
            print(f"[Qwen-VL] Batch generate error: {e}")
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)


def _ensure_batch_processor():
    """Lazily start the batch processor on first request."""
    global _inference_queue, _batch_processor_task
    if _inference_queue is None:
        _inference_queue = asyncio.Queue()
    if _batch_processor_task is None or _batch_processor_task.done():
        _batch_processor_task = asyncio.get_event_loop().create_task(_batch_inference_loop())


async def run_inference_single(llm_input):
    """Queue a single prompt and await its result."""
    _ensure_batch_processor()
    future = asyncio.get_event_loop().create_future()
    await _inference_queue.put({"llm_input": llm_input, "future": future})
    return await future


async def run_inference_batch(llm_inputs):
    """Queue multiple prompts and await all results (preserving order)."""
    _ensure_batch_processor()
    loop = asyncio.get_event_loop()
    futures = []
    for inp in llm_inputs:
        future = loop.create_future()
        await _inference_queue.put({"llm_input": inp, "future": future})
        futures.append(future)
    return await asyncio.gather(*futures)


# ============================================
# RUNPOD HANDLER
# ============================================

async def handler(job):
    """
    Process bank statement images with Qwen-VL (async, concurrent-capable).
    1 image = 1 page. No tiling or multi-image combining.

    Supports:
    - Batch mode: {"images": [{"image_base64": ...}, {"image_url": ...}]}
    - Single mode: {"image_base64": ...} or {"image_url": ...}
    """
    start_time = time.time()
    temp_files = []

    try:
        job_input = job.get("input", {})
        images_batch = job_input.get("images")

        if images_batch and isinstance(images_batch, list):
            # BATCH MODE: multiple images in one job
            print(f"[Qwen-VL] Batch mode: processing {len(images_batch)} images (concurrent handler)")

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

            print(f"[Qwen-VL] Queuing batch of {len(llm_inputs)} images for inference...")
            outputs = await run_inference_batch(llm_inputs)

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
            # SINGLE MODE (base64 or URL)
            image_base64 = job_input.get("image_base64")
            image_url = job_input.get("image_url")
            custom_prompt = job_input.get("prompt")

            if not image_base64 and not image_url:
                return {"error": "Missing image_base64, image_url, or images in input"}

            input_mode = "url" if image_url else "base64"
            print(f"[Qwen-VL] Single mode, input: {input_mode} (concurrent handler)")

            llm_input, temp_path, size = process_single_image(
                image_base64=image_base64,
                image_url=image_url,
                custom_prompt=custom_prompt
            )
            temp_files.append(temp_path)

            print(f"[Qwen-VL] Processing single image: {size[0]}x{size[1]}")

            output = await run_inference_single(llm_input)
            markdown = output.outputs[0].text

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
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def concurrency_modifier(current_concurrency: int) -> int:
    """
    Allow up to MAX_CONCURRENCY concurrent jobs.
    Jobs queue their inference requests; the batch processor runs them
    in a single llm.generate() call, so concurrency is safe.
    """
    return MAX_CONCURRENCY


# Start RunPod serverless handler with concurrency support
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
