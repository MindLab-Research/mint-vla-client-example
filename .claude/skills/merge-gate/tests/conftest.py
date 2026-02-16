"""Shared fixtures and utilities for merge gate tests."""

import os
import time
import uuid


import pytest
import requests

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

DEFAULT_POLL_TIMEOUT_S = int(os.environ.get("MINT_MERGE_GATE_POLL_TIMEOUT_S", "900"))

DENSE_MODEL = "Qwen/Qwen3-0.6B"  # Primary dense test model (Phase 1)
DENSE_SMALL_MODEL = DENSE_MODEL  # Alias used by some tests for clarity
DENSE_MEDIUM_MODEL = "Qwen/Qwen3-4B"  # Medium dense model (Phase 3 stress tests)
DENSE_LARGE_MODEL = "Qwen/Qwen3-8B"  # Large dense model (Phase 3 stress tests)
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MOONLIGHT_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"  # DeepseekV3 MLA architecture


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


# =============================================================================
# API Helpers
# =============================================================================

def poll_future(request_id: str, timeout: int = 300, request_timeout: int = 120) -> dict:
    """Poll for async operation result.

    Args:
        request_id: The async request ID to poll.
        timeout: Total timeout in seconds for the polling loop.
        request_timeout: Per-request HTTP timeout. Set to 120s to accommodate
            vLLM MoE model initialization which takes ~62+ seconds.
    """
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=request_timeout)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation request_id={request_id} did not complete within {timeout}s")


def create_session(base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session. Returns (session_id, model_id)."""
    session_id = f"merge_gate_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=DEFAULT_POLL_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def forward_backward(model_id: str, data: list, loss_fn: str = "cross_entropy",
                     loss_fn_config: dict = None) -> dict:
    """Run forward-backward pass."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
    }
    if loss_fn_config:
        payload["forward_backward_input"]["loss_fn_config"] = loss_fn_config

    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=DEFAULT_POLL_TIMEOUT_S)


def optim_step(model_id: str, lr: float = 1e-4) -> dict:
    """Run optimizer step."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=60)


def train_step(model_id: str, data: list, lr: float = 1e-4, loss_fn: str = "cross_entropy") -> dict:
    """Combined forward_backward + optim_step (for MoE)."""
    url = f"{BASE_URL}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=DEFAULT_POLL_TIMEOUT_S)


def save_weights(model_id: str, name: str = "test") -> dict:
    """Save weights for sampling.

    Note: For MoE models, vLLM engine creation + CUDA graph capture takes ~120s,
    so we use a longer timeout.
    """
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    # MoE models need longer timeout for vLLM engine + CUDA graph capture
    return poll_future(
        resp.json().get("request_id"),
        timeout=DEFAULT_POLL_TIMEOUT_S,
        request_timeout=180,
    )


def sample(model_id: str, prompt_tokens: list, max_tokens: int = 20,
           temperature: float = 0.0, include_prompt_logprobs: bool = False) -> dict:
    """Sample from model.

    Returns dict with 'sequences' key containing list of SampledSequence dicts,
    each with 'tokens', 'logprobs', 'stop_reason'.
    If include_prompt_logprobs=True, 'prompt_logprobs' at root level.
    """
    url = f"{BASE_URL}/api/v1/asample"
    payload = {
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
        "include_prompt_logprobs": include_prompt_logprobs,
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)


def list_actors(actor_type: str | None = None, model_name: str | None = None) -> dict:
    """List ResourcePool actors via server endpoint.

    Used by merge-gate eviction sentries to observe real eviction events.
    """
    url = f"{BASE_URL}/api/v1/actors"
    params: dict[str, str] = {}
    if actor_type is not None:
        params["type"] = actor_type
    if model_name is not None:
        params["model_name"] = model_name
    resp = requests.get(url, params=params, headers=get_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """Get tokenizer for Dense model (Qwen3-0.6B)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def small_tokenizer():
    """Get tokenizer for small Dense model (alias of primary dense model)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DENSE_SMALL_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def moe_tokenizer():
    """Get tokenizer for MoE model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MOE_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def moonlight_tokenizer():
    """Get tokenizer for Moonlight model (DeepseekV3 MLA)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MOONLIGHT_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def medium_tokenizer():
    """Get tokenizer for medium Dense model (Qwen3-4B)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DENSE_MEDIUM_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def large_tokenizer():
    """Get tokenizer for large Dense model (Qwen3-8B)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DENSE_LARGE_MODEL, trust_remote_code=True)


@pytest.fixture
def dense_session():
    """Create a Dense model session."""
    session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
    yield session_id, model_id
    # Cleanup not needed - sessions auto-expire


@pytest.fixture
def moe_session():
    """Create a MoE model session."""
    session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
    yield session_id, model_id


# =============================================================================
# Test Data Builders
# =============================================================================

def make_sft_datum(input_tokens: list, target_tokens: list, loss_mask: list) -> dict:
    """Create a single SFT training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def make_rl_datum(input_tokens: list, target_tokens: list, loss_mask: list,
                  old_logprobs: list, advantages: list) -> dict:
    """Create a single RL training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        },
    }


def make_dpo_datum(input_tokens: list, target_tokens: list, loss_mask: list,
                   ref_logprobs: list, is_chosen: bool) -> dict:
    """Create a single DPO training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            "ref_logprobs": {"data": ref_logprobs, "shape": [len(ref_logprobs)], "dtype": "float32"},
            "is_chosen": is_chosen,
        },
    }


# =============================================================================
# Checkpoint APIs
# =============================================================================

def save_state(model_id: str, name: str = "checkpoint") -> dict:
    """Save full checkpoint (LoRA + optimizer state) for training resume."""
    url = f"{BASE_URL}/api/v1/save_state"
    payload = {"model_id": model_id, "path": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def load_state(model_id: str, path: str, load_optimizer: bool = True) -> dict:
    """Load checkpoint to resume training."""
    url = f"{BASE_URL}/api/v1/load_state"
    payload = {"model_id": model_id, "path": path, "optimizer": load_optimizer}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)
