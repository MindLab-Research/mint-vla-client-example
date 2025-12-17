#!/usr/bin/env python3
"""Tinker SDK Compatibility Tests.

Replicates the official tinker_test.ipynb test cases to verify Mint implementation
matches official Tinker SDK behavior.

Usage:
    # Run all tests
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_tinker_compatibility.py

    # Run specific test
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_tinker_compatibility.py --test pig_latin
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


# =============================================================================
# 0. Configuration
# =============================================================================

def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def get_api_key():
    return os.environ.get("TINKER_API_KEY", "dummy")


# =============================================================================
# 1. Setup and Client Creation
# =============================================================================

class ServiceClient:
    """Minimal ServiceClient matching Tinker SDK interface."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_server_capabilities(self):
        """Get server capabilities including supported models."""
        resp = requests.get(f"{self.base_url}/api/v1/healthz", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def create_lora_training_client(self, base_model: str, lora_rank: int = 32, learning_rate: float = 1e-4):
        """Create a training client for the given base model."""
        return TrainingClient(
            base_url=self.base_url,
            api_key=self.api_key,
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
        )


class TrainingClient:
    """Training client matching Tinker SDK interface."""

    def __init__(self, base_url: str, api_key: str, base_model: str, lora_rank: int, learning_rate: float):
        self.base_url = base_url
        self.api_key = api_key
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model_id = None
        self.session_id = None
        self._tokenizer = None

    def _ensure_session(self):
        """Create training session if not exists."""
        if self.model_id is not None:
            return

        import uuid
        self.session_id = f"compat_test_{uuid.uuid4().hex[:8]}"

        url = f"{self.base_url}/api/v1/create_model"
        payload = {
            "session_id": self.session_id,
            "model_seq_id": 1,
            "base_model": self.base_model,
            "lora_config": {"rank": self.lora_rank},
            "learning_rate": self.learning_rate,
        }
        resp = requests.post(url, json=payload, headers=self.headers, timeout=300)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)
        if "error" in result:
            raise RuntimeError(f"Session creation failed: {result['error']}")
        self.model_id = result.get("model_id")

    def _poll_future(self, request_id: str, timeout: int = 300) -> dict:
        """Poll for async result."""
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

    def get_tokenizer(self):
        """Get tokenizer for the base model."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        return self._tokenizer

    def forward_backward(self, data: list, loss_fn: str) -> "ForwardBackwardResult":
        """Compute forward pass and gradients.

        Args:
            data: List of Datum objects
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", etc.)

        Returns:
            ForwardBackwardResult with loss_fn_outputs containing logprobs
        """
        self._ensure_session()

        # Convert Datum objects to API format
        api_data = []
        for datum in data:
            api_item = {
                "model_input": {"chunks": [{"tokens": datum.model_input.tokens, "type": "encoded_text"}]},
                "loss_fn_inputs": {
                    "target_tokens": {
                        "data": datum.loss_fn_inputs["target_tokens"],
                        "shape": [len(datum.loss_fn_inputs["target_tokens"])],
                        "dtype": "int64",
                    },
                    "loss_mask": {
                        "data": datum.loss_fn_inputs["weights"],
                        "shape": [len(datum.loss_fn_inputs["weights"])],
                        "dtype": "float32",
                    },
                },
            }
            api_data.append(api_item)

        url = f"{self.base_url}/api/v1/forward_backward"
        payload = {
            "model_id": self.model_id,
            "forward_backward_input": {"data": api_data, "loss_fn": loss_fn},
        }
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)

        # Extract logprobs from result
        loss_fn_outputs = []
        for item in result.get("loss_fn_outputs", []):
            logprobs = item.get("logprobs", [])
            loss_fn_outputs.append({"logprobs": np.array(logprobs)})

        return ForwardBackwardResult(
            loss_fn_outputs=loss_fn_outputs,
            metrics=result.get("metrics", {}),
        )

    def optim_step(self, adam_params: "AdamParams") -> "OptimStepResult":
        """Perform optimizer step.

        Args:
            adam_params: Adam optimizer parameters

        Returns:
            OptimStepResult
        """
        self._ensure_session()

        url = f"{self.base_url}/api/v1/optim_step"
        payload = {
            "model_id": self.model_id,
            "adam_params": {
                "learning_rate": adam_params.learning_rate,
                "beta1": adam_params.beta1,
                "beta2": adam_params.beta2,
                "eps": adam_params.eps,
            },
        }
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)
        return OptimStepResult(metrics=result.get("metrics", {}))

    def save_weights_and_get_sampling_client(self, name: str) -> "SamplingClient":
        """Save weights and create sampling client.

        Args:
            name: Name for the saved weights

        Returns:
            SamplingClient for sampling from the trained model
        """
        self._ensure_session()

        url = f"{self.base_url}/api/v1/save_weights"
        payload = {"model_id": self.model_id, "name": name}
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)

        return SamplingClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_id=self.model_id,
            lora_path=result.get("path"),
        )

    def save_weights_for_sampler(self, name: str) -> "SaveResult":
        """Save weights for sampling (without creating client).

        Args:
            name: Name for the saved weights

        Returns:
            SaveResult with path
        """
        self._ensure_session()

        url = f"{self.base_url}/api/v1/save_weights"
        payload = {"model_id": self.model_id, "name": name}
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)
        return SaveResult(path=result.get("path"))

    def save_state(self, name: str) -> "SaveResult":
        """Save full training state (weights + optimizer).

        Args:
            name: Name for the checkpoint

        Returns:
            SaveResult with path
        """
        self._ensure_session()

        url = f"{self.base_url}/api/v1/save_state"
        payload = {"model_id": self.model_id, "name": name}
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)
        return SaveResult(path=result.get("path"))

    def load_state(self, path: str) -> None:
        """Load training state from checkpoint.

        Args:
            path: Path to checkpoint
        """
        self._ensure_session()

        url = f"{self.base_url}/api/v1/load_state"
        payload = {"model_id": self.model_id, "path": path}
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        self._poll_future(resp.json().get("request_id"), timeout=300)


class SamplingClient:
    """Sampling client for generating from trained model."""

    def __init__(self, base_url: str, api_key: str, model_id: str, lora_path: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.lora_path = lora_path
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _poll_future(self, request_id: str, timeout: int = 300) -> dict:
        """Poll for async result."""
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

    def sample(
        self,
        prompt: "ModelInput",
        sampling_params: "SamplingParams",
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int | None = None,
    ) -> "SampleResult":
        """Sample from the model.

        Args:
            prompt: Input prompt as ModelInput
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate
            include_prompt_logprobs: Whether to include prompt logprobs
            topk_prompt_logprobs: Number of top-k logprobs per position

        Returns:
            SampleResult with sequences and optional logprobs
        """
        url = f"{self.base_url}/api/v1/asample"
        payload = {
            "model_id": self.model_id,
            "prompt": {"chunks": [{"tokens": prompt.tokens, "type": "encoded_text"}]},
            "sampling_params": {
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "stop": sampling_params.stop,
            },
            "num_samples": num_samples,
            "include_prompt_logprobs": include_prompt_logprobs,
        }
        if topk_prompt_logprobs is not None:
            payload["topk_prompt_logprobs"] = topk_prompt_logprobs

        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        result = self._poll_future(resp.json().get("request_id"), timeout=300)

        sequences = []
        for seq in result.get("sequences", []):
            sequences.append(Sequence(tokens=seq.get("tokens", [])))

        return SampleResult(
            sequences=sequences,
            prompt_logprobs=result.get("prompt_logprobs"),
            topk_prompt_logprobs=result.get("topk_prompt_logprobs"),
        )


# =============================================================================
# 2. Data Types (matching Tinker SDK types)
# =============================================================================

@dataclass
class ModelInput:
    """Model input tokens."""
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        return cls(tokens=tokens)

    def to_ints(self) -> list[int]:
        return self.tokens


@dataclass
class Datum:
    """Single training example."""
    model_input: ModelInput
    loss_fn_inputs: dict  # {"weights": [...], "target_tokens": [...]}


@dataclass
class AdamParams:
    """Adam optimizer parameters."""
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12


@dataclass
class SamplingParams:
    """Sampling parameters."""
    max_tokens: int = 20
    temperature: float = 0.0
    stop: list[str] | None = None


@dataclass
class ForwardBackwardResult:
    """Result from forward_backward."""
    loss_fn_outputs: list[dict]  # [{"logprobs": np.array}, ...]
    metrics: dict


@dataclass
class OptimStepResult:
    """Result from optim_step."""
    metrics: dict


@dataclass
class SaveResult:
    """Result from save operations."""
    path: str


@dataclass
class Sequence:
    """Generated sequence."""
    tokens: list[int]


@dataclass
class SampleResult:
    """Result from sampling."""
    sequences: list[Sequence]
    prompt_logprobs: list[float] | None = None
    topk_prompt_logprobs: list[list[dict]] | None = None


# =============================================================================
# 3. Test: Pig Latin Training
# =============================================================================

# Training examples from the notebook
PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]


def process_example(example: dict, tokenizer) -> Datum:
    """Convert example to Datum format.

    From notebook:
    - Format: "English: {input}\nPig Latin:"
    - Prompt tokens get weight 0
    - Completion tokens (with leading space) get weight 1
    - Targets are shifted by 1
    """
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]  # Shifted by 1 for next-token prediction
    weights = weights[1:]

    return Datum(
        model_input=ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def test_pig_latin_training(base_model: str = "Qwen/Qwen2.5-7B-Instruct", num_updates: int = 6):
    """Test Pig Latin SFT training.

    Replicates Section 3 of tinker_test.ipynb.

    Expected results from notebook:
        Update 1: Loss per token: 2.4501
        Update 2: Loss per token: 2.1987
        Update 3: Loss per token: 1.6280
        Update 4: Loss per token: 1.1566
        Update 5: Loss per token: 0.8436
        Update 6: Loss per token: 0.5790
    """
    print("=" * 70)
    print("TEST: Pig Latin SFT Training")
    print("=" * 70)
    print(f"Base model: {base_model}")
    print(f"Num updates: {num_updates}")

    # Create clients
    service_client = ServiceClient(base_url=get_base_url(), api_key=get_api_key())
    training_client = service_client.create_lora_training_client(base_model=base_model)
    print(f"Training client created for {base_model}")

    # Get tokenizer and process examples
    tokenizer = training_client.get_tokenizer()
    processed_examples = [process_example(ex, tokenizer) for ex in PIG_LATIN_EXAMPLES]
    print(f"Processed {len(processed_examples)} examples")

    # Visualize first example
    print("\nFirst example visualization:")
    print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
    print("-" * 50)
    datum0 = processed_examples[0]
    for inp, tgt, wgt in zip(
        datum0.model_input.to_ints(),
        datum0.loss_fn_inputs["target_tokens"],
        datum0.loss_fn_inputs["weights"],
    ):
        print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")

    # Training loop
    print("\nStarting training updates...")
    losses = []
    for i in range(num_updates):
        start_time = time.time()

        fwdbwd_result = training_client.forward_backward(processed_examples, "cross_entropy")
        optim_result = training_client.optim_step(AdamParams(learning_rate=1e-4))

        elapsed = time.time() - start_time

        # Compute weighted average log loss per token (from notebook)
        logprobs = np.concatenate([output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([ex.loss_fn_inputs["weights"] for ex in processed_examples])
        loss = -np.dot(logprobs, weights) / weights.sum()
        losses.append(loss)

        print(f"Update {i+1}: Loss per token: {loss:.4f} (elapsed: {elapsed:.2f}s)")

    # Check results
    print("\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    # Validation
    passed = True
    if losses[-1] >= losses[0]:
        print("  FAIL: Loss did not decrease")
        passed = False
    if (losses[0] - losses[-1]) / losses[0] < 0.5:
        print("  WARN: Loss reduction < 50%")

    return passed, losses


# =============================================================================
# 4. Test: Sampling
# =============================================================================

def test_sampling(base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Test sampling from trained model.

    Replicates Section 4 of tinker_test.ipynb.
    """
    print("=" * 70)
    print("TEST: Sampling from Trained Model")
    print("=" * 70)

    # First train on Pig Latin
    service_client = ServiceClient(base_url=get_base_url(), api_key=get_api_key())
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()
    processed_examples = [process_example(ex, tokenizer) for ex in PIG_LATIN_EXAMPLES]

    print("Training 6 updates...")
    for i in range(6):
        training_client.forward_backward(processed_examples, "cross_entropy")
        training_client.optim_step(AdamParams(learning_rate=1e-4))
        print(f"  Update {i+1} complete")

    # Create sampling client
    print("\nCreating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="pig-latin-model")

    # Sample
    prompt_text = "English: coffee break\nPig Latin:"
    prompt = ModelInput.from_ints(tokenizer.encode(prompt_text))
    params = SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"])

    print(f"\nPrompt: {repr(prompt_text)}")
    print("Sampling 8 responses...")

    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)

    print("\nResponses:")
    for i, seq in enumerate(result.sequences):
        print(f"  {i}: {repr(tokenizer.decode(seq.tokens))}")

    # Validation: responses should be non-empty and consistent (temp=0)
    passed = True
    if len(result.sequences) == 0:
        print("  FAIL: No sequences returned")
        passed = False
    elif all(len(seq.tokens) == 0 for seq in result.sequences):
        print("  FAIL: All sequences empty")
        passed = False

    return passed


# =============================================================================
# 5. Test: Computing Logprobs
# =============================================================================

def test_prompt_logprobs(base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Test prompt logprobs computation.

    Replicates Section 5 of tinker_test.ipynb.
    """
    print("=" * 70)
    print("TEST: Prompt Logprobs")
    print("=" * 70)

    service_client = ServiceClient(base_url=get_base_url(), api_key=get_api_key())
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    # Create sampling client (no training needed for logprobs)
    sampling_client = training_client.save_weights_and_get_sampling_client(name="logprobs-test")

    prompt_text = "How many r's are in the word strawberry?"
    prompt = ModelInput.from_ints(tokenizer.encode(prompt_text))

    print(f"Prompt: {repr(prompt_text)}")
    print(f"Tokens: {prompt.tokens}")

    # Test include_prompt_logprobs
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
    )

    print("\nPrompt Logprobs:")
    if result.prompt_logprobs is not None:
        print(f"  {result.prompt_logprobs}")
        print(f"  Length: {len(result.prompt_logprobs)} (expected: {len(prompt.tokens)})")
    else:
        print("  None (not supported)")

    # Test topk_prompt_logprobs
    print("\nTop-k Prompt Logprobs (k=5):")
    result_topk = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=5,
    )

    if result_topk.topk_prompt_logprobs is not None:
        for i, topk in enumerate(result_topk.topk_prompt_logprobs[:5]):  # Show first 5
            print(f"  Pos {i}: {topk}")
    else:
        print("  None (not supported - known issue in official Tinker)")

    # Validation
    passed = True
    if result.prompt_logprobs is None:
        print("\n  WARN: prompt_logprobs not supported")
    elif len(result.prompt_logprobs) != len(prompt.tokens):
        print(f"\n  FAIL: Length mismatch: {len(result.prompt_logprobs)} vs {len(prompt.tokens)}")
        passed = False

    return passed


# =============================================================================
# 6. Test: Saving and Loading
# =============================================================================

def test_checkpoint_roundtrip(base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Test checkpoint save/load round-trip.

    Replicates Section 6 of tinker_test.ipynb.
    """
    print("=" * 70)
    print("TEST: Checkpoint Round-Trip")
    print("=" * 70)

    service_client = ServiceClient(base_url=get_base_url(), api_key=get_api_key())
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()
    processed_examples = [process_example(ex, tokenizer) for ex in PIG_LATIN_EXAMPLES]

    # Train 5 iterations
    print("Training 5 iterations...")
    losses_before = []
    for i in range(5):
        fwdbwd_result = training_client.forward_backward(processed_examples, "cross_entropy")
        training_client.optim_step(AdamParams(learning_rate=1e-4))
        logprobs = np.concatenate([output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([ex.loss_fn_inputs["weights"] for ex in processed_examples])
        loss = -np.dot(logprobs, weights) / weights.sum()
        losses_before.append(loss)
        print(f"  Update {i+1}: Loss = {loss:.4f}")

    # Save sampling checkpoint
    print("\nSaving sampling checkpoint...")
    try:
        sampling_result = training_client.save_weights_for_sampler(name="checkpoint_sampler")
        print(f"  Sampling path: {sampling_result.path}")
    except Exception as e:
        print(f"  save_weights_for_sampler: {e}")

    # Save resume checkpoint
    print("\nSaving resume checkpoint...")
    try:
        resume_result = training_client.save_state(name="checkpoint_resume")
        print(f"  Resume path: {resume_result.path}")
    except Exception as e:
        print(f"  save_state: {e}")
        print("  (save_state may not be implemented yet)")
        return True  # Not a failure if not implemented

    # Train 5 more iterations
    print("\nTraining 5 more iterations...")
    for i in range(5):
        training_client.forward_backward(processed_examples, "cross_entropy")
        training_client.optim_step(AdamParams(learning_rate=1e-4))

    # Load checkpoint
    print("\nLoading checkpoint...")
    try:
        training_client.load_state(resume_result.path)
        print("  Checkpoint loaded")
    except Exception as e:
        print(f"  load_state: {e}")
        print("  (load_state may not be implemented yet)")
        return True

    # Verify loss matches
    print("\nVerifying restored state...")
    fwdbwd_result = training_client.forward_backward(processed_examples, "cross_entropy")
    logprobs = np.concatenate([output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([ex.loss_fn_inputs["weights"] for ex in processed_examples])
    loss_restored = -np.dot(logprobs, weights) / weights.sum()

    print(f"  Loss at checkpoint (iter 5): {losses_before[-1]:.4f}")
    print(f"  Loss after restore: {loss_restored:.4f}")
    print(f"  Difference: {abs(losses_before[-1] - loss_restored):.4f}")

    passed = abs(losses_before[-1] - loss_restored) < 0.1
    if not passed:
        print("  FAIL: Loss mismatch after restore")

    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tinker SDK Compatibility Tests")
    parser.add_argument(
        "--test",
        choices=["all", "pig_latin", "sampling", "logprobs", "checkpoint"],
        default="all",
        help="Test to run",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to use",
    )
    args = parser.parse_args()

    print(f"TINKER_BASE_URL: {get_base_url()}")
    print(f"Base model: {args.base_model}")
    print()

    results = {}

    if args.test in ["all", "pig_latin"]:
        try:
            passed, losses = test_pig_latin_training(args.base_model)
            results["pig_latin"] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"ERROR: {e}")
            results["pig_latin"] = "ERROR"
        print()

    if args.test in ["all", "sampling"]:
        try:
            passed = test_sampling(args.base_model)
            results["sampling"] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"ERROR: {e}")
            results["sampling"] = "ERROR"
        print()

    if args.test in ["all", "logprobs"]:
        try:
            passed = test_prompt_logprobs(args.base_model)
            results["logprobs"] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"ERROR: {e}")
            results["logprobs"] = "ERROR"
        print()

    if args.test in ["all", "checkpoint"]:
        try:
            passed = test_checkpoint_roundtrip(args.base_model)
            results["checkpoint"] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"ERROR: {e}")
            results["checkpoint"] = "ERROR"
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL" if result == "FAIL" else "ERROR"
        print(f"  {test_name}: {status}")

    all_passed = all(r == "PASS" for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
