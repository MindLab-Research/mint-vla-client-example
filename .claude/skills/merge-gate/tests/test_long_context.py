"""Long Context Test: Verify models use native context window.

Tests issue #9 fix: vLLM should use model's max_position_embeddings
instead of hardcoded defaults.

Tests:
- Dense model (Qwen2.5-7B): 32K context
- MoE model (Qwen3-30B-A3B): 40K context (vLLM limit, model supports 262K)
- Prompt near context limit should succeed
- Prompt exceeding context limit should give clear error

Pass criteria:
- Near-limit prompts generate output without error
- Exceed-limit prompts return clear error message (not cryptic vLLM error)
"""

import pytest

from .conftest import (
    DENSE_MODEL,
    MOE_MODEL,
    create_session,
    save_weights,
    sample,
)


# Context limits (vLLM operational limits)
DENSE_CONTEXT = 32768   # Qwen2.5-7B-Instruct
MOE_CONTEXT = 40960     # Qwen3-30B-A3B-Instruct-2507 (vLLM limit; model supports 262K)


def generate_tokens(tokenizer, target_length: int) -> list[int]:
    """Generate a list of tokens with approximately target_length."""
    base = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base, add_special_tokens=False)
    reps = (target_length // len(base_tokens)) + 1
    return (base_tokens * reps)[:target_length]


class TestDenseLongContext:
    """Long context tests for Dense model (32K context)."""

    def test_near_context_limit(self, tokenizer):
        """Test prompt near 32K limit succeeds."""
        # Use 30K tokens (near 32K limit, leaving room for response)
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
        """Test prompt exceeding 32K limit gives clear error."""
        # Use 35K tokens (exceeds 32K limit)
        prompt_size = 35000
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
