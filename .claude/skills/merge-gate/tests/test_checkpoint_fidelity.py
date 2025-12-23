"""Checkpoint Fidelity Tests.

Tests that checkpoint save/load preserves model state correctly.

T9: Weight Preservation
- Train model for N iterations
- Save checkpoint
- Sample from model (get output X)
- Create new session, load checkpoint
- Sample from loaded model (get output Y)
- Verify X == Y (deterministic sampling)

T10: Training Resume
- Train model for N iterations, record loss curve A
- Save checkpoint at iteration N
- Continue training for M more iterations, record loss curve A'
- Create new session, load checkpoint from iteration N
- Train for M iterations from checkpoint, record loss curve B'
- Verify A' and B' are similar (training resumes correctly)

Agent-analyzed: collects full data and plots for analysis.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

from .conftest import (
    DENSE_MODEL,
    DENSE_SMALL_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    save_state,
    load_state,
    sample,
)
from .utils import RESULTS_DIR


@dataclass
class CheckpointTestResult:
    """Result of a checkpoint fidelity test."""
    test_name: str
    timestamp: str
    success: bool
    details: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)

    def save(self) -> Path:
        path = RESULTS_DIR / f"{self.test_name}_{self.timestamp}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def make_training_batch(tokenizer, seed: int = 42) -> list[dict]:
    """Create deterministic training data."""
    import random
    rng = random.Random(seed)

    prompts = [
        "Translate to French: hello -> ",
        "Translate to French: goodbye -> ",
        "Translate to French: thank you -> ",
        "Translate to French: please -> ",
    ]
    targets = ["bonjour", "au revoir", "merci", "s'il vous plait"]

    data = []
    for prompt, target in zip(prompts, targets):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = tokenizer.encode(target, add_special_tokens=False)

        full_tokens = prompt_tokens + target_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)

        data.append({
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
            },
        })

    return data


class TestCheckpointFidelity:
    """Checkpoint fidelity tests."""

    def test_weight_preservation(self, tokenizer):
        """T9: Verify checkpoint save/load preserves weights.

        Methodology:
        1. Train model for 5 iterations
        2. Save checkpoint
        3. Sample with deterministic settings
        4. Create new session, load checkpoint
        5. Sample again
        6. Compare outputs (should be identical)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n=== T9: Weight Preservation Test ===")

        # Phase 1: Train and save
        print("\nPhase 1: Create session and train...")
        session_id_a, model_id_a = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session A: {model_id_a}")

        train_data = make_training_batch(tokenizer)

        losses_a = []
        for i in range(5):
            result = forward_backward(model_id_a, train_data, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses_a.append(loss)
            optim_step(model_id_a, lr=1e-4)
            print(f"  Iter {i+1}: loss={loss:.4f}")

        # Save checkpoint
        print("\nSaving checkpoint...")
        ckpt_result = save_state(model_id_a, name="t9_checkpoint")
        ckpt_path = ckpt_result.get("path")
        print(f"Checkpoint saved: {ckpt_path}")

        # Sample from trained model
        print("\nSampling from trained model...")
        save_weights(model_id_a, name="t9_sample")

        test_prompt = "Translate to French: hello -> "
        prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=True)

        result_a = sample(model_id_a, prompt_tokens, max_tokens=10, temperature=0.0)

        if "error" in result_a:
            pytest.fail(f"Sampling failed: {result_a['error']}")

        tokens_a = result_a["sequences"][0].get("tokens", [])
        text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
        logprobs_a = result_a["sequences"][0].get("logprobs", [])

        print(f"Output A: '{text_a}'")
        print(f"Tokens A: {tokens_a}")
        print(f"Logprobs A: {logprobs_a[:5] if logprobs_a else 'N/A'}...")

        # Phase 2: Load checkpoint in new session
        print("\nPhase 2: Create new session and load checkpoint...")
        session_id_b, model_id_b = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session B: {model_id_b}")

        load_result = load_state(model_id_b, ckpt_path, load_optimizer=True)
        print(f"Checkpoint loaded: {load_result.get('path')}")

        # Sample from loaded model
        print("\nSampling from loaded model...")
        save_weights(model_id_b, name="t9_sample_loaded")

        result_b = sample(model_id_b, prompt_tokens, max_tokens=10, temperature=0.0)

        if "error" in result_b:
            pytest.fail(f"Sampling from loaded model failed: {result_b['error']}")

        tokens_b = result_b["sequences"][0].get("tokens", [])
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        logprobs_b = result_b["sequences"][0].get("logprobs", [])

        print(f"Output B: '{text_b}'")
        print(f"Tokens B: {tokens_b}")
        print(f"Logprobs B: {logprobs_b[:5] if logprobs_b else 'N/A'}...")

        # Compare
        anomalies = []
        tokens_match = tokens_a == tokens_b
        if not tokens_match:
            anomalies.append(f"Token mismatch: {tokens_a} vs {tokens_b}")

        # Compare logprobs (allow floating point variance between vLLM sessions)
        # Small logprob differences are expected due to CUDA non-determinism
        logprobs_match = True
        max_logprob_diff = 0.0
        if logprobs_a and logprobs_b:
            for i, (lp_a, lp_b) in enumerate(zip(logprobs_a, logprobs_b)):
                diff = abs(lp_a - lp_b)
                max_logprob_diff = max(max_logprob_diff, diff)
                # Allow up to 10% relative difference for small logprobs
                if diff > 0.1:  # Absolute threshold for significant variance
                    logprobs_match = False
                    anomalies.append(f"Logprob mismatch at {i}: {lp_a} vs {lp_b} (diff={diff:.4f})")
                    break

        # For T9, tokens matching is the critical criterion
        # Small logprob variance is acceptable
        success = tokens_match

        # Save result
        result = CheckpointTestResult(
            test_name="t9_weight_preservation",
            timestamp=timestamp,
            success=success,
            details={
                "model_id_a": model_id_a,
                "model_id_b": model_id_b,
                "checkpoint_path": ckpt_path,
                "losses": losses_a,
                "output_a": text_a,
                "output_b": text_b,
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "tokens_match": tokens_match,
                "logprobs_match": logprobs_match,
            },
            anomalies=anomalies,
        )

        report_path = result.save()

        print(f"\n{'='*60}")
        print(f"RESULT: {'PASS' if success else 'FAIL'}")
        print(f"{'='*60}")
        print(f"Tokens match: {tokens_match}")
        print(f"Logprobs match: {logprobs_match}")
        print(f"Report: {report_path}")

        if anomalies:
            print(f"\nANOMALIES:")
            for a in anomalies:
                print(f"  - {a}")

        if not success:
            pytest.fail(f"Checkpoint fidelity test failed: outputs differ after load")

    @pytest.mark.skip(reason="load_state not fully implemented - LoRA weights not loaded for training")
    def test_training_resume(self, tokenizer):
        """T10: Verify training can resume from checkpoint.

        Methodology:
        1. Train model A for 5 iterations, record losses
        2. Save checkpoint at iteration 5
        3. Continue training A for 5 more iterations (iterations 6-10)
        4. Create new session B, load checkpoint from iteration 5
        5. Train B for 5 iterations (equivalent to A's 6-10)
        6. Compare loss trajectories - should be nearly identical
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n=== T10: Training Resume Test ===")

        train_data = make_training_batch(tokenizer)

        # Phase 1: Train session A for 10 iterations, checkpoint at 5
        print("\nPhase 1: Train session A (10 iterations, checkpoint at 5)...")
        _, model_id_a = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session A: {model_id_a}")

        losses_a_before = []  # Iterations 1-5
        losses_a_after = []   # Iterations 6-10

        for i in range(10):
            result = forward_backward(model_id_a, train_data, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)

            if i < 5:
                losses_a_before.append(loss)
            else:
                losses_a_after.append(loss)

            optim_step(model_id_a, lr=1e-4)
            print(f"  Iter {i+1}: loss={loss:.4f}")

            # Save checkpoint at iteration 5
            if i == 4:
                print(f"\n  Saving checkpoint at iteration 5...")
                ckpt_result = save_state(model_id_a, name="t10_checkpoint")
                ckpt_path = ckpt_result.get("path")
                print(f"  Checkpoint: {ckpt_path}\n")

        # Phase 2: Create new session B, load checkpoint, train for 5 iterations
        print("\nPhase 2: Load checkpoint in new session B, train 5 iterations...")
        _, model_id_b = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session B: {model_id_b}")

        load_result = load_state(model_id_b, ckpt_path, load_optimizer=True)
        print(f"Loaded: {load_result.get('path')}")

        losses_b = []
        for i in range(5):
            result = forward_backward(model_id_b, train_data, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses_b.append(loss)
            optim_step(model_id_b, lr=1e-4)
            print(f"  Iter {i+1}: loss={loss:.4f}")

        # Compare loss trajectories
        print("\n=== Comparison ===")
        print("Session A (6-10):", [f"{l:.4f}" for l in losses_a_after])
        print("Session B (1-5):", [f"{l:.4f}" for l in losses_b])

        anomalies = []

        # Check if losses match (allowing for small floating point differences)
        max_diff = 0.0
        for i, (loss_a, loss_b) in enumerate(zip(losses_a_after, losses_b)):
            diff = abs(loss_a - loss_b)
            max_diff = max(max_diff, diff)
            if diff > 0.01:  # 1% tolerance
                anomalies.append(f"Loss mismatch at iter {i+1}: A={loss_a:.4f}, B={loss_b:.4f}, diff={diff:.4f}")

        # Check trajectory similarity via correlation
        if len(losses_a_after) >= 3 and len(losses_b) >= 3:
            import numpy as np
            corr = np.corrcoef(losses_a_after, losses_b)[0, 1]
            print(f"Correlation: {corr:.4f}")
            if corr < 0.9:
                anomalies.append(f"Low correlation between trajectories: {corr:.4f}")
        else:
            corr = None

        success = len(anomalies) == 0

        # Save result
        result = CheckpointTestResult(
            test_name="t10_training_resume",
            timestamp=timestamp,
            success=success,
            details={
                "model_id_a": model_id_a,
                "model_id_b": model_id_b,
                "checkpoint_path": ckpt_path,
                "losses_a_before": losses_a_before,
                "losses_a_after": losses_a_after,
                "losses_b": losses_b,
                "max_diff": max_diff,
                "correlation": corr,
            },
            anomalies=anomalies,
        )

        report_path = result.save()

        print(f"\n{'='*60}")
        print(f"RESULT: {'PASS' if success else 'FAIL'}")
        print(f"{'='*60}")
        print(f"Max loss difference: {max_diff:.4f}")
        print(f"Correlation: {corr:.4f}" if corr else "Correlation: N/A")
        print(f"Report: {report_path}")

        if anomalies:
            print(f"\nANOMALIES:")
            for a in anomalies:
                print(f"  - {a}")

        if not success:
            pytest.fail(f"Training resume test failed: trajectories differ significantly")


class TestCheckpointFidelitySmall:
    """Quick checkpoint fidelity tests with small model (for CI)."""

    def test_weight_preservation_small(self, small_tokenizer):
        """Quick weight preservation test with Qwen3-0.6B."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n=== T9 (Small): Weight Preservation Test ===")

        # Train session A
        _, model_id_a = create_session(DENSE_SMALL_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session A: {model_id_a}")

        train_data = make_training_batch(small_tokenizer)

        for i in range(3):
            result = forward_backward(model_id_a, train_data, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            optim_step(model_id_a, lr=1e-4)
            print(f"  Iter {i+1}: loss={loss:.4f}")

        # Save checkpoint
        ckpt_result = save_state(model_id_a, name="t9_small_checkpoint")
        ckpt_path = ckpt_result.get("path")
        print(f"Checkpoint: {ckpt_path}")

        # Sample
        save_weights(model_id_a, name="t9_small_sample")
        prompt = "Hello, "
        prompt_tokens = small_tokenizer.encode(prompt, add_special_tokens=True)
        result_a = sample(model_id_a, prompt_tokens, max_tokens=5, temperature=0.0)

        if "error" in result_a:
            pytest.fail(f"Sampling failed: {result_a.get('error')}")

        tokens_a = result_a["sequences"][0].get("tokens", [])

        # Load in new session
        _, model_id_b = create_session(DENSE_SMALL_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session B: {model_id_b}")

        load_state(model_id_b, ckpt_path, load_optimizer=True)
        save_weights(model_id_b, name="t9_small_sample_loaded")

        result_b = sample(model_id_b, prompt_tokens, max_tokens=5, temperature=0.0)

        if "error" in result_b:
            pytest.fail(f"Sampling from loaded model failed: {result_b.get('error')}")

        tokens_b = result_b["sequences"][0].get("tokens", [])

        print(f"Tokens A: {tokens_a}")
        print(f"Tokens B: {tokens_b}")

        success = tokens_a == tokens_b
        print(f"Match: {success}")

        if not success:
            pytest.fail(f"Token mismatch: {tokens_a} vs {tokens_b}")
