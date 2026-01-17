"""Long Context Test: Verify models use native context window.

Tests issue #9 fix: vLLM should use model's max_position_embeddings
instead of hardcoded defaults.

Tests:
- Dense model (Qwen3-0.6B): 40K context
- MoE model (Qwen3-30B-A3B): Maximum supported context (both inference AND training)
- Prompt near context limit should succeed
- Prompt exceeding context limit should give clear error

Pass criteria:
- Near-limit prompts generate output without error
- Long context training completes without OOM
- Exceed-limit prompts return clear error message (not cryptic vLLM error)
"""

import pytest

from .conftest import (
    DENSE_MODEL,
    MOE_MODEL,
    create_session,
    save_weights,
    sample,
    train_step,
    make_sft_datum,
)


# Context limits (from model_registry.py max_model_len)
DENSE_CONTEXT = 40960   # Qwen3-0.6B (40K context)
MOE_CONTEXT = 40960     # Qwen3-30B-A3B-Instruct-2507 (40K context)


def generate_tokens(tokenizer, target_length: int) -> list[int]:
    """Generate a list of tokens with approximately target_length."""
    base = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base, add_special_tokens=False)
    reps = (target_length // len(base_tokens)) + 1
    return (base_tokens * reps)[:target_length]


class TestDenseLongContext:
    """Long context tests for Dense model (40K vLLM context)."""

    def test_near_context_limit(self, tokenizer):
        """Test prompt near context limit succeeds."""
        # Use 30K tokens (large prompt, leaving room for response)
        prompt_size = 30000
        prompt_tokens = generate_tokens(tokenizer, prompt_size)

        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="long_context_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        assert "error" not in result, f"Near-limit sampling failed: {result.get('error')}"
        assert "sequences" in result, "No sequences returned"
        assert len(result["sequences"][0].get("tokens", [])) > 0, "No tokens generated"

        print(f"Dense near-limit ({prompt_size:,} tokens): Generated {len(result['sequences'][0]['tokens'])} tokens")

    def test_exceed_context_limit(self, tokenizer):
        """Test prompt exceeding context limit gives clear error."""
        prompt_size = 45000  # Exceeds DENSE_CONTEXT
        prompt_tokens = generate_tokens(tokenizer, prompt_size)

        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="exceed_context_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        # Should get an error
        assert "error" in result or "sequences" not in result or not result.get("sequences"), \
            f"Expected error for {prompt_size:,} token prompt exceeding {DENSE_CONTEXT:,} limit"

        # Error should be clear (mention context/length/tokens)
        error_msg = str(result.get("error", ""))
        assert any(kw in error_msg.lower() for kw in ["context", "length", "token", "exceed", "limit"]), \
            f"Error message not clear: {error_msg[:200]}"

        print(f"Dense exceed-limit ({prompt_size:,} tokens): Got expected clear error")


@pytest.mark.moe
class TestMoELongContext:
    """Long context tests for MoE model (40K vLLM context).

    Marked with @pytest.mark.moe - these tests require 4 GPUs and take longer.
    Skip with: pytest -m "not moe"
    """

    def test_moderate_context(self, moe_tokenizer):
        """Test MoE with moderate prompt (8K tokens)."""
        prompt_size = 8000
        prompt_tokens = generate_tokens(moe_tokenizer, prompt_size)

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_context_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        assert "error" not in result, f"MoE moderate prompt failed: {result.get('error')}"
        assert "sequences" in result, "No sequences returned"
        assert len(result["sequences"][0].get("tokens", [])) > 0, "No tokens generated"

        print(f"MoE moderate ({prompt_size:,} tokens): Generated {len(result['sequences'][0]['tokens'])} tokens")

    @pytest.mark.slow
    def test_large_context(self, moe_tokenizer):
        """Test MoE with large prompt (30K tokens).

        Marked slow - takes significant time for prefill.
        Skip with: pytest -m "not slow"
        """
        prompt_size = 30000
        prompt_tokens = generate_tokens(moe_tokenizer, prompt_size)

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_large_context_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        assert "error" not in result, f"MoE large prompt failed: {result.get('error')}"
        assert "sequences" in result, "No sequences returned"
        assert len(result["sequences"][0].get("tokens", [])) > 0, "No tokens generated"

        print(f"MoE large ({prompt_size:,} tokens): Generated {len(result['sequences'][0]['tokens'])} tokens")

    @pytest.mark.slow
    def test_near_context_limit(self, moe_tokenizer):
        """Test MoE with prompt near 40K limit (38K tokens).

        Marked slow - stress test for memory.
        Skip with: pytest -m "not slow"
        """
        prompt_size = 38000
        prompt_tokens = generate_tokens(moe_tokenizer, prompt_size)

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_near_limit_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        assert "error" not in result, f"MoE near-limit prompt failed: {result.get('error')}"
        assert "sequences" in result, "No sequences returned"
        assert len(result["sequences"][0].get("tokens", [])) > 0, "No tokens generated"

        print(f"MoE near-limit ({prompt_size:,} tokens): Generated {len(result['sequences'][0]['tokens'])} tokens")

    def test_exceed_context_limit(self, moe_tokenizer):
        """Test MoE prompt exceeding 40K limit gives clear error."""
        prompt_size = 45000
        prompt_tokens = generate_tokens(moe_tokenizer, prompt_size)

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        save_weights(model_id, name="moe_exceed_context_test")

        result = sample(model_id, prompt_tokens, max_tokens=50, temperature=0.7)

        # Should get an error
        assert "error" in result or "sequences" not in result or not result.get("sequences"), \
            f"Expected error for {prompt_size:,} token prompt exceeding {MOE_CONTEXT:,} limit"

        # Error should be clear (mention context/length/tokens)
        error_msg = str(result.get("error", ""))
        assert any(kw in error_msg.lower() for kw in ["context", "length", "token", "exceed", "limit"]), \
            f"Error message not clear: {error_msg[:200]}"

        print(f"MoE exceed-limit ({prompt_size:,} tokens): Got expected clear error")

    @pytest.mark.slow
    def test_long_context_training(self, moe_tokenizer):
        """Test MoE training with long context (near 40K limit).

        CRITICAL: 30B MoE must support maximum context for both inference AND training.
        This test verifies training does not OOM on long sequences.

        Marked slow - memory intensive.
        Skip with: pytest -m "not slow"
        """
        # Near-limit training: 38K tokens (near 40K limit)
        context_size = 38000
        tokens = generate_tokens(moe_tokenizer, context_size)

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)

        # Create training datum with long context
        # input_tokens: all tokens except last
        # target_tokens: all tokens (shifted by 1 internally)
        # loss_mask: 1.0 for all positions
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = [1.0] * len(target_tokens)

        datum = make_sft_datum(input_tokens, target_tokens, loss_mask)

        # Run one training step - should not OOM
        result = train_step(model_id, [datum], lr=1e-5, loss_fn="cross_entropy")

        assert "error" not in result, f"Long context training failed: {result.get('error')}"
        loss = result.get("metrics", {}).get("loss:mean")
        assert loss is not None, "No loss returned from training metrics"
        assert loss > 0, f"Invalid loss: {loss}"
        assert loss < 100, f"Loss suspiciously high: {loss}"

        print(f"MoE long context training ({context_size:,} tokens): loss={loss:.4f}")

        # Verify we can still sample after training
        save_weights(model_id, name="moe_long_train_test")
        sample_result = sample(model_id, tokens[:100], max_tokens=10, temperature=0.0)
        assert "error" not in sample_result, f"Sampling after long training failed: {sample_result.get('error')}"

        print(f"MoE long context training PASS: trained and sampled successfully")
