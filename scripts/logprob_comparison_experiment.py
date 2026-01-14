#!/usr/bin/env python3
"""
Logprob Comparison Experiment: Megatron vs vLLM

This script verifies that LoRA weights exported from Megatron are correctly
applied in vLLM by comparing logprobs for the same input sequence.

Key metrics:
1. Mean absolute difference in logprobs
2. Max absolute difference
3. Token-by-token comparison

If the differences are significant (>0.1), there's likely a bug in the
LoRA export/import logic.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# Test configuration
MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
RENDERER_NAME = "kimi_k2"
TEST_PROMPT = """<|im_start|>user
I have the numbers [25, 4, 2, 6] and need to reach 100. Can you find an arithmetic expression using +, -, *, / that equals 100?<|im_end|>
<|im_start|>assistant
"""

EXPECTED_COMPLETION_START = "Let me think about this step by step"


async def make_request(client: httpx.AsyncClient, method: str, endpoint: str, json_data: dict | None = None) -> dict:
    """Make HTTP request to the server."""
    headers = {"X-API-Key": API_KEY}
    url = f"{BASE_URL}/api/v1{endpoint}"

    if method == "GET":
        response = await client.get(url, headers=headers, timeout=300.0)
    elif method == "POST":
        response = await client.post(url, headers=headers, json=json_data, timeout=300.0)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if response.status_code >= 400:
        logger.error(f"Request failed: {response.status_code} - {response.text}")
        raise Exception(f"Request failed: {response.status_code}")

    return response.json()


async def wait_for_future(client: httpx.AsyncClient, request_id: str, timeout: float = 600.0) -> dict:
    """Poll for future completion."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = await client.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers={"X-API-Key": API_KEY},
            json={"request_id": request_id},
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 408:
            # Still pending
            await asyncio.sleep(2.0)
        else:
            raise Exception(f"Unexpected status: {response.status_code} - {response.text}")

    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


async def create_training_session(client: httpx.AsyncClient) -> str:
    """Create a training session and return model_id."""
    logger.info("Creating training session...")

    response = await make_request(client, "POST", "/create_session", {
        "base_model": MODEL_NAME,
        "lora_rank": 16,
        "learning_rate": 5e-5,
    })

    request_id = response["request_id"]
    logger.info(f"Waiting for session creation (request_id={request_id})...")

    result = await wait_for_future(client, request_id, timeout=1800.0)
    model_id = result.get("model_id")
    logger.info(f"Training session created: model_id={model_id}")
    return model_id


async def train_one_step(client: httpx.AsyncClient, model_id: str) -> dict:
    """Run one training step with a simple batch."""
    logger.info("Running one training step...")

    # Create a simple training batch
    batch = {
        "prompts": [TEST_PROMPT],
        "completions": [EXPECTED_COMPLETION_START + "\n25 * 4 = 100\nSo the answer is 25 * 4 = 100."],
        "rewards": [1.0],
    }

    response = await make_request(client, "POST", "/train", {
        "model_id": model_id,
        "data": batch,
    })

    request_id = response["request_id"]
    logger.info(f"Waiting for training step (request_id={request_id})...")

    result = await wait_for_future(client, request_id, timeout=600.0)
    logger.info(f"Training step completed: {result}")
    return result


async def export_lora_weights(client: httpx.AsyncClient, model_id: str) -> dict:
    """Export LoRA weights via save_weights endpoint."""
    logger.info("Exporting LoRA weights...")

    response = await make_request(client, "POST", "/save_weights", {
        "model_id": model_id,
        "path": "logprob_test",
    })

    request_id = response["request_id"]
    logger.info(f"Waiting for weight export (request_id={request_id})...")

    result = await wait_for_future(client, request_id, timeout=300.0)
    logger.info(f"Weight export completed. Keys: {len(result.get('state_dict_keys', []))}")

    # Log some exported key names for debugging
    keys = result.get('state_dict_keys', [])
    logger.info(f"Sample exported keys: {keys[:10]}")

    return result


async def get_megatron_logprobs(client: httpx.AsyncClient, model_id: str, prompt: str, completion: str) -> dict:
    """Get logprobs from Megatron training backend."""
    logger.info("Getting logprobs from Megatron...")

    # Use the compute_logprobs endpoint if available, otherwise train with return_logprobs
    # For now, we'll use the training endpoint with a batch of size 1
    batch = {
        "prompts": [prompt],
        "completions": [completion],
        "rewards": [0.0],  # Dummy reward, we just want logprobs
        "return_logprobs": True,
    }

    response = await make_request(client, "POST", "/train", {
        "model_id": model_id,
        "data": batch,
        "no_update": True,  # Just compute logprobs, don't update weights
    })

    request_id = response["request_id"]
    result = await wait_for_future(client, request_id, timeout=300.0)

    logprobs = result.get("logprobs", [])
    logger.info(f"Megatron logprobs shape: {len(logprobs)}")
    return {"logprobs": logprobs, "source": "megatron"}


async def get_vllm_logprobs(client: httpx.AsyncClient, model_id: str, prompt: str, completion: str) -> dict:
    """Get logprobs from vLLM inference backend."""
    logger.info("Getting logprobs from vLLM...")

    # Use asample with logprobs enabled
    response = await make_request(client, "POST", "/asample", {
        "model_id": model_id,
        "prompt": prompt,
        "max_tokens": len(completion.split()),  # Approximate
        "temperature": 0.0,  # Greedy for deterministic comparison
        "logprobs": True,
        "top_logprobs": 1,
    })

    request_id = response["request_id"]
    result = await wait_for_future(client, request_id, timeout=300.0)

    logprobs = result.get("logprobs", [])
    logger.info(f"vLLM logprobs shape: {len(logprobs)}")
    return {"logprobs": logprobs, "source": "vllm"}


async def compare_logprobs(megatron_result: dict, vllm_result: dict) -> dict:
    """Compare logprobs from both backends."""
    logger.info("Comparing logprobs...")

    m_logprobs = megatron_result.get("logprobs", [])
    v_logprobs = vllm_result.get("logprobs", [])

    if not m_logprobs or not v_logprobs:
        logger.warning("One or both backends returned empty logprobs")
        return {"error": "Empty logprobs"}

    # Align lengths
    min_len = min(len(m_logprobs), len(v_logprobs))
    if min_len == 0:
        return {"error": "No comparable logprobs"}

    # Compute differences
    diffs = []
    for i in range(min_len):
        m_lp = m_logprobs[i] if isinstance(m_logprobs[i], (int, float)) else m_logprobs[i].get("logprob", 0)
        v_lp = v_logprobs[i] if isinstance(v_logprobs[i], (int, float)) else v_logprobs[i].get("logprob", 0)
        diffs.append(abs(m_lp - v_lp))

    mean_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)

    result = {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "num_tokens": len(diffs),
        "diffs": diffs[:20],  # First 20 for inspection
    }

    logger.info(f"Comparison results:")
    logger.info(f"  Mean absolute difference: {mean_diff:.6f}")
    logger.info(f"  Max absolute difference: {max_diff:.6f}")
    logger.info(f"  Number of tokens compared: {len(diffs)}")

    if mean_diff > 0.1:
        logger.warning("SIGNIFICANT DIFFERENCE DETECTED - likely export/import bug")
    else:
        logger.info("Logprobs are consistent between Megatron and vLLM")

    return result


async def run_experiment():
    """Run the full logprob comparison experiment."""
    logger.info("=" * 60)
    logger.info("Logprob Comparison Experiment: Megatron vs vLLM")
    logger.info("=" * 60)

    async with httpx.AsyncClient() as client:
        # Check server health
        try:
            health = await make_request(client, "GET", "/healthz")
            logger.info(f"Server health: {health}")
        except Exception as e:
            logger.error(f"Server not available: {e}")
            return

        # Create training session
        model_id = await create_training_session(client)

        # Train one step to initialize LoRA weights
        await train_one_step(client, model_id)

        # Export LoRA weights (this also registers for vLLM sampling)
        export_result = await export_lora_weights(client, model_id)

        # Check if sampling was registered
        if not export_result.get("sampling_registered", False):
            logger.error("LoRA weights were NOT registered for vLLM sampling")
            logger.error("This indicates a problem with the export/registration flow")
            return

        # Define test sequence for comparison
        test_prompt = TEST_PROMPT
        test_completion = EXPECTED_COMPLETION_START[:50]  # Short completion for testing

        # Get logprobs from both backends
        try:
            megatron_result = await get_megatron_logprobs(client, model_id, test_prompt, test_completion)
        except Exception as e:
            logger.error(f"Failed to get Megatron logprobs: {e}")
            megatron_result = {"error": str(e)}

        try:
            vllm_result = await get_vllm_logprobs(client, model_id, test_prompt, test_completion)
        except Exception as e:
            logger.error(f"Failed to get vLLM logprobs: {e}")
            vllm_result = {"error": str(e)}

        # Compare results
        if "error" not in megatron_result and "error" not in vllm_result:
            comparison = await compare_logprobs(megatron_result, vllm_result)
        else:
            comparison = {
                "megatron_error": megatron_result.get("error"),
                "vllm_error": vllm_result.get("error"),
            }

        # Save results
        results = {
            "model_id": model_id,
            "export_result": {
                "keys_count": len(export_result.get("state_dict_keys", [])),
                "sample_keys": export_result.get("state_dict_keys", [])[:20],
                "sampling_registered": export_result.get("sampling_registered", False),
            },
            "megatron_result": megatron_result,
            "vllm_result": vllm_result,
            "comparison": comparison,
        }

        output_path = "/tmp/logprob_comparison_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

        return results


if __name__ == "__main__":
    results = asyncio.run(run_experiment())

    if results:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(json.dumps(results.get("comparison", {}), indent=2, default=str))
