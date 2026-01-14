"""MoE API Test: Sampling, Logprobs, Checkpoint.

Tests core API operations on MoE model (Qwen3-30B-A3B):
- Model creation
- Weight saving
- Sampling with various parameters
- Prompt logprob extraction
- Train-then-sample workflow

Pass criteria: All API operations complete without error.

Note: MoE tests require 12 GPUs (8 Megatron + 4 vLLM).
"""

import numpy as np
import pytest

from .conftest import (
    MOE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    sample,
)


class TestMoEAPI:
    """MoE model API tests."""

    def test_session_creation(self, moe_tokenizer):
        """Test MoE session creation."""
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        assert session_id is not None, "Session creation failed"
        assert model_id is not None, "Model ID not returned"
        print(f"MoE session: session={session_id}, model={model_id}")

    def test_weight_save_and_sample(self, moe_tokenizer):
        """Test MoE weight saving and sampling flow."""
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)

        # Save initial weights
        save_result = save_weights(model_id, name="moe_api_test_initial")
        assert "error" not in save_result, f"Weight save failed: {save_result.get('error')}"

        # Prepare prompt
        prompt = "The capital of France is"
        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)

        # Sample with temperature=0 (deterministic)
        result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.0)
        assert "error" not in result, f"Sampling failed: {result.get('error')}"
        assert "sequences" in result, "No sequences returned"
        assert len(result["sequences"]) > 0, "Empty sequences list"

        sample_data = result["sequences"][0]
        generated_tokens = sample_data.get("tokens", [])
        assert len(generated_tokens) > 0, "No tokens generated"

        generated_text = moe_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")

    def test_sampling_with_temperature(self, moe_tokenizer):
        """Test MoE sampling with various temperatures."""
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_temp_test")

        prompt = "Once upon a time"
        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)

        temperatures = [0.0, 0.5, 1.0]
        for temp in temperatures:
            result = sample(model_id, prompt_tokens, max_tokens=20, temperature=temp)
            assert "error" not in result, f"Sampling failed at temp={temp}: {result.get('error')}"

            sample_data = result["sequences"][0]
            generated_text = moe_tokenizer.decode(sample_data.get("tokens", []), skip_special_tokens=True)
            print(f"Temp {temp}: {generated_text}")

    def test_prompt_logprobs(self, moe_tokenizer):
        """Test MoE prompt logprob extraction."""
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_logprob_test")

        prompt = "The quick brown fox jumps"
        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)

        result = sample(
            model_id,
            prompt_tokens,
            max_tokens=5,
            temperature=0.0,
            include_prompt_logprobs=True
        )
        assert "error" not in result, f"Sampling failed: {result.get('error')}"

        # prompt_logprobs is at root level of response
        prompt_logprobs = result.get("prompt_logprobs", [])

        # Should have logprobs for each prompt token (except first)
        expected_len = len(prompt_tokens) - 1
        assert len(prompt_logprobs) >= expected_len - 1, (
            f"Expected ~{expected_len} prompt logprobs, got {len(prompt_logprobs)}"
        )

        # Logprobs should be negative
        for i, lp in enumerate(prompt_logprobs):
            assert lp <= 0, f"Logprob at position {i} is positive: {lp}"

        print(f"Prompt tokens: {len(prompt_tokens)}")
        print(f"Prompt logprobs: {len(prompt_logprobs)}")
        print(f"Sample logprobs: {prompt_logprobs[:5]}...")

    def test_train_then_sample(self, moe_tokenizer):
        """Test MoE train-then-sample workflow."""
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)

        # Prepare simple training data
        prompt = "Q: What is 2+2?\nA: "
        target = "4"

        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = moe_tokenizer.encode(target, add_special_tokens=False)

        full_tokens = prompt_tokens + target_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)

        api_data = [{
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
            },
        }]

        # Train for a few iterations
        losses = []
        for i in range(3):
            result = forward_backward(model_id, api_data, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses.append(loss)
            optim_step(model_id, lr=1e-4)
            print(f"Iteration {i+1}: loss={loss:.4f}")

        # Save and sample
        save_weights(model_id, name="moe_trained")
        result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.0)

        assert "error" not in result, f"Post-training sampling failed: {result.get('error')}"

        sample_data = result["sequences"][0]
        generated_text = moe_tokenizer.decode(sample_data.get("tokens", []), skip_special_tokens=True)
        print(f"After training: {prompt}{generated_text}")

        # Loss should decrease
        if len(losses) >= 2:
            assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
