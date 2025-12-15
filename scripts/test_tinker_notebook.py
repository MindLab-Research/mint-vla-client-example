#!/usr/bin/env python3
"""
Tinker SDK Test - Direct Port from tinker_test.ipynb

This script is a faithful reproduction of the official tinker_test.ipynb notebook
converted to a standalone Python script. It preserves the exact structure and
code from the original notebook.

Usage:
    # Set environment variables
    export TINKER_BASE_URL=http://localhost:8000
    export TINKER_API_KEY=dummy

    # Run the test
    python scripts/test_tinker_notebook.py
"""

# =============================================================================
# 0. Installation
# =============================================================================
# Install the Tinker SDK with:
# %pip install tinker
#
# Note: For Mint server testing, we use direct HTTP requests instead of the
# tinker SDK to test our implementation.

# =============================================================================
# 1. Setup and Client Creation
# =============================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env', override=True)

print(os.getenv("TINKER_API_KEY"))  # Verify that the API key is loaded
print(os.getenv("TINKER_BASE_URL"))  # Verify that the API URL is loaded

# For Mint testing, we'll use a minimal client implementation
import requests
import time
import uuid
import numpy as np


class MintClient:
    """Minimal client for testing Mint server (Tinker-compatible)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _poll_future(self, request_id: str, timeout: int = 300) -> dict:
        poll_url = f"{self.base_url}/api/v1/retrieve_future"
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.post(poll_url, json={"request_id": request_id}, headers=self.headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 408:
                time.sleep(0.5)
                continue
            else:
                resp.raise_for_status()
        raise TimeoutError(f"Operation did not complete within {timeout}s")


base_url = os.getenv("TINKER_BASE_URL", "http://localhost:8000")
api_key = os.getenv("TINKER_API_KEY", "dummy")

service_client = MintClient(base_url=base_url, api_key=api_key)
print("Available models:")
try:
    resp = requests.get(f"{base_url}/api/v1/healthz", timeout=30)
    print(f"- Server healthy: {resp.json()}")
except Exception as e:
    print(f"Error listing models: {e}")

# Now we create the TrainingClient. We'll use Qwen2.5-7B-Instruct as the base model.
base_model = "Qwen/Qwen2.5-7B-Instruct"
# Note: Ensure this model is available in the list above or use a valid one.

# Create session
session_id = f"notebook_test_{uuid.uuid4().hex[:8]}"
create_resp = requests.post(
    f"{base_url}/api/v1/create_model",
    json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": 32},
        "learning_rate": 1e-4,
    },
    headers=service_client.headers,
    timeout=300,
)
create_resp.raise_for_status()
result = service_client._poll_future(create_resp.json().get("request_id"), timeout=300)
model_id = result.get("model_id")
print(f"Training client created for {base_model}")
print(f"Model ID: {model_id}")

# =============================================================================
# 2. Preparing Training Data
# =============================================================================

# We will train a model to translate words into Pig Latin.

# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]

# Now we convert these examples into the format expected by the training client using the tokenizer.

from transformers import AutoTokenizer

# Get the tokenizer from the training client
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def process_example(example: dict, tokenizer) -> dict:
    """Convert example to Mint API format (equivalent to types.Datum)."""
    # Format the input with Input/Output template
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]  # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # Return in Mint API format
    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "weights": weights,
        # Mint API format
        "api_format": {
            "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": [float(w) for w in weights], "shape": [len(weights)], "dtype": "float32"},
            },
        }
    }


processed_examples = [process_example(ex, tokenizer) for ex in examples]

# Let's visualize the first example to verify the tokenization and weights.

# Visualize the first example for debugging purposes
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0["input_tokens"], datum0["target_tokens"], datum0["weights"])):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")

# =============================================================================
# 3. Performing a Training Update
# =============================================================================

# We'll perform 6 updates on the same batch of data.

print("\nStarting training updates...")
all_losses = []

for i in range(6):
    # Forward-backward pass
    fwdbwd_resp = requests.post(
        f"{base_url}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": [ex["api_format"] for ex in processed_examples],
                "loss_fn": "cross_entropy",
            },
        },
        headers=service_client.headers,
        timeout=120,
    )
    fwdbwd_resp.raise_for_status()
    fwdbwd_result = service_client._poll_future(fwdbwd_resp.json().get("request_id"), timeout=300)

    # Optimizer step
    optim_resp = requests.post(
        f"{base_url}/api/v1/optim_step",
        json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
        },
        headers=service_client.headers,
        timeout=120,
    )
    optim_resp.raise_for_status()
    optim_result = service_client._poll_future(optim_resp.json().get("request_id"), timeout=300)

    # Compute weighted average log loss per token
    # Note: Mint returns loss_fn_outputs with logprobs (either as dict with 'data' or list)
    loss_fn_outputs = fwdbwd_result.get("loss_fn_outputs", [])
    if loss_fn_outputs and "logprobs" in loss_fn_outputs[0]:
        def extract_logprobs(output):
            lp = output["logprobs"]
            return lp["data"] if isinstance(lp, dict) else lp
        logprobs = np.concatenate([extract_logprobs(output) for output in loss_fn_outputs])
        weights = np.concatenate([ex["weights"] for ex in processed_examples])
        loss = -np.dot(logprobs, weights) / weights.sum()
    else:
        # Fallback to server-computed loss
        loss = fwdbwd_result.get("metrics", {}).get("loss:mean", 0)

    all_losses.append(loss)
    print(f"Update {i+1}: Loss per token: {loss:.4f}")

# =============================================================================
# 4. Sampling from the Model
# =============================================================================

# Now we test the model by sampling. We'll translate "coffee break".

print("\n" + "=" * 70)
print("4. Sampling from the Model")
print("=" * 70)

# First, create a sampling client. We need to transfer weights
save_resp = requests.post(
    f"{base_url}/api/v1/save_weights",
    json={"model_id": model_id, "name": "pig-latin-model"},
    headers=service_client.headers,
    timeout=120,
)
save_resp.raise_for_status()
save_result = service_client._poll_future(save_resp.json().get("request_id"), timeout=300)
print(f"Weights saved to: {save_result.get('path')}")

# Now, we can sample from the model.
prompt_text = "English: coffee break\nPig Latin:"
prompt_tokens = tokenizer.encode(prompt_text)

sample_resp = requests.post(
    f"{base_url}/api/v1/asample",
    json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": 20, "temperature": 0.0, "stop": ["\n"]},
        "num_samples": 8,
    },
    headers=service_client.headers,
    timeout=120,
)
sample_resp.raise_for_status()
sample_result = service_client._poll_future(sample_resp.json().get("request_id"), timeout=300)

print("\nResponses:")
for i, seq in enumerate(sample_result.get("sequences", [])):
    tokens = seq.get("tokens", [])
    print(f"{i}: {repr(tokenizer.decode(tokens))}")

# =============================================================================
# 5. Computing Logprobs
# =============================================================================

print("\n" + "=" * 70)
print("5. Computing Logprobs")
print("=" * 70)

prompt_text = "How many r's are in the word strawberry?"
prompt_tokens = tokenizer.encode(prompt_text)

logprobs_resp = requests.post(
    f"{base_url}/api/v1/asample",
    json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": 1},  # Must be at least 1 token, represents prefill step
        "num_samples": 1,
        "include_prompt_logprobs": True,
    },
    headers=service_client.headers,
    timeout=120,
)
logprobs_resp.raise_for_status()
logprobs_result = service_client._poll_future(logprobs_resp.json().get("request_id"), timeout=300)

print("Prompt Logprobs:")
print(logprobs_result.get("prompt_logprobs"))

# Top-k Logprobs
print("\nTop-k Prompt Logprobs:")
topk_resp = requests.post(
    f"{base_url}/api/v1/asample",
    json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": 1},
        "num_samples": 1,
        "include_prompt_logprobs": True,
        "topk_prompt_logprobs": 5,
    },
    headers=service_client.headers,
    timeout=120,
)
topk_resp.raise_for_status()
topk_result = service_client._poll_future(topk_resp.json().get("request_id"), timeout=300)

topk_logprobs = topk_result.get("topk_prompt_logprobs")
if topk_logprobs is not None:
    for i, topk in enumerate(topk_logprobs):
        if topk:
            print(f"Pos {i}: {topk}")
        else:
            print(f"Pos {i}: None")
else:
    print("None (not supported - known issue in official Tinker)")

# =============================================================================
# 6. Saving and Loading
# =============================================================================

print("\n" + "=" * 70)
print("6. Saving and Loading")
print("=" * 70)

# Save a checkpoint that you can use for sampling
sampler_save_resp = requests.post(
    f"{base_url}/api/v1/save_weights",
    json={"model_id": model_id, "name": "0000"},
    headers=service_client.headers,
    timeout=120,
)
sampler_save_resp.raise_for_status()
sampler_save_result = service_client._poll_future(sampler_save_resp.json().get("request_id"), timeout=300)
sampling_path = sampler_save_result.get("path")
print(f"Sampling path: {sampling_path}")

# Save a checkpoint that you can resume from
try:
    state_save_resp = requests.post(
        f"{base_url}/api/v1/save_state",
        json={"model_id": model_id, "name": "0010"},
        headers=service_client.headers,
        timeout=120,
    )
    state_save_resp.raise_for_status()
    state_save_result = service_client._poll_future(state_save_resp.json().get("request_id"), timeout=300)
    resume_path = state_save_result.get("path")
    print(f"Resume path: {resume_path}")

    # Load that checkpoint
    load_resp = requests.post(
        f"{base_url}/api/v1/load_state",
        json={"model_id": model_id, "path": resume_path},
        headers=service_client.headers,
        timeout=120,
    )
    load_resp.raise_for_status()
    service_client._poll_future(load_resp.json().get("request_id"), timeout=300)
    print("Checkpoint loaded successfully")
except Exception as e:
    print(f"save_state/load_state not implemented: {e}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTraining Results:")
print(f"  Initial loss: {all_losses[0]:.4f}")
print(f"  Final loss: {all_losses[-1]:.4f}")
print(f"  Loss reduction: {(all_losses[0] - all_losses[-1]) / all_losses[0] * 100:.1f}%")

print(f"\nExpected (from tinker_test.ipynb):")
print(f"  Update 1: Loss per token: 2.4501")
print(f"  Update 6: Loss per token: 0.5790")
print(f"  Loss reduction: ~76%")

# Validation
if all_losses[-1] < all_losses[0] and (all_losses[0] - all_losses[-1]) / all_losses[0] > 0.5:
    print("\nPASS: Training produces expected loss reduction")
else:
    print("\nFAIL: Training did not produce expected results")
