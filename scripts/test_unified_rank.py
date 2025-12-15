#!/usr/bin/env python3
"""Test Phase 7: Unified rank support via max-rank padding.

Tests the pad/truncate functionality:
1. Unit test pad_lora_state_dict()
2. Unit test truncate_lora_state_dict()
3. Integration test with Megatron actor (if available)

Run from tinker-server root directory.
"""

import argparse
import sys

import torch


def test_pad_truncate_utilities():
    """Test the pad/truncate utility functions."""
    import importlib.util
    from pathlib import Path

    # Direct load to avoid ray dependency in backend/__init__.py
    lora_utils_path = Path(__file__).parent.parent / "tinker_server" / "backend" / "lora_utils.py"
    spec = importlib.util.spec_from_file_location("lora_utils", lora_utils_path)
    lora_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lora_utils)

    pad_lora_state_dict = lora_utils.pad_lora_state_dict
    truncate_lora_state_dict = lora_utils.truncate_lora_state_dict
    get_lora_rank_from_state_dict = lora_utils.get_lora_rank_from_state_dict

    print("\n=== Unit Test: pad_lora_state_dict ===")

    # Create sample state dict with rank=32
    actual_rank = 32
    hidden_dim = 4096
    state_dict = {
        "model.layers.0.self_attn.q_proj.adapter.lora_A.weight": torch.randn(actual_rank, hidden_dim),
        "model.layers.0.self_attn.q_proj.adapter.lora_B.weight": torch.randn(hidden_dim, actual_rank),
        "model.layers.0.self_attn.k_proj.adapter.lora_A.weight": torch.randn(actual_rank, hidden_dim),
        "model.layers.0.self_attn.k_proj.adapter.lora_B.weight": torch.randn(hidden_dim, actual_rank),
    }

    # Infer rank
    inferred_rank = get_lora_rank_from_state_dict(state_dict)
    print(f"  Inferred rank: {inferred_rank}")
    assert inferred_rank == actual_rank, f"Expected {actual_rank}, got {inferred_rank}"

    # Pad to trainer_rank=64
    trainer_rank = 64
    padded = pad_lora_state_dict(state_dict, actual_rank, trainer_rank)

    # Verify shapes
    for name, tensor in padded.items():
        if "lora_A" in name:
            expected_shape = (trainer_rank, hidden_dim)
            assert tensor.shape == expected_shape, f"{name}: expected {expected_shape}, got {tensor.shape}"
            # Verify padding is zeros
            assert torch.allclose(tensor[actual_rank:], torch.zeros(trainer_rank - actual_rank, hidden_dim))
            print(f"  {name}: {state_dict[name].shape} -> {tensor.shape} (padded)")
        elif "lora_B" in name:
            expected_shape = (hidden_dim, trainer_rank)
            assert tensor.shape == expected_shape, f"{name}: expected {expected_shape}, got {tensor.shape}"
            # Verify padding is zeros
            assert torch.allclose(tensor[:, actual_rank:], torch.zeros(hidden_dim, trainer_rank - actual_rank))
            print(f"  {name}: {state_dict[name].shape} -> {tensor.shape} (padded)")

    print("  PASS: All tensors padded correctly")

    print("\n=== Unit Test: truncate_lora_state_dict ===")

    # Truncate back to actual_rank=32
    truncated = truncate_lora_state_dict(padded, trainer_rank, actual_rank)

    # Verify shapes match original
    for name, tensor in truncated.items():
        original_tensor = state_dict[name]
        assert tensor.shape == original_tensor.shape, f"{name}: shape mismatch"
        # Verify values preserved (truncation should recover original)
        assert torch.allclose(tensor, original_tensor), f"{name}: values differ after truncation"
        print(f"  {name}: {padded[name].shape} -> {tensor.shape} (truncated)")

    print("  PASS: All tensors truncated correctly, values preserved")

    print("\n=== Unit Test: No-op cases ===")

    # Test no-op: actual_rank >= trainer_rank
    no_op_result = pad_lora_state_dict(state_dict, actual_rank=64, trainer_rank=64)
    assert no_op_result is state_dict, "Expected same object when no padding needed"
    print("  PASS: pad_lora_state_dict returns same dict when actual >= trainer")

    no_op_result = truncate_lora_state_dict(padded, trainer_rank=32, actual_rank=64)
    assert no_op_result is padded, "Expected same object when no truncation needed"
    print("  PASS: truncate_lora_state_dict returns same dict when actual >= trainer")


def test_integration_with_actor():
    """Test integration with Megatron actor (if available)."""
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    from tinker_server.backend.megatron_distributed import (
        is_megatron_actor_running,
        PERSISTENT_MEGATRON_ACTOR_NAME,
        PERSISTENT_NAMESPACE,
    )

    print("\n=== Integration Test: Megatron Actor ===")

    if not is_megatron_actor_running():
        print("  SKIP: No Megatron actor running")
        print("  To test integration, start MoE training first")
        return

    actor = ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
    print(f"  Connected to Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}")

    # Get session info
    session_info = ray.get(actor.get_session_info.remote())
    print(f"  max_lora_rank: {session_info.get('max_lora_rank')}")
    print(f"  actual_rank: {session_info.get('actual_rank')}")
    print(f"  current_session: {session_info.get('current_session')}")

    print("  PASS: Session info retrieved with Phase 7 fields")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 7 unified rank support")
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration test with Megatron actor",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 7: Unified Rank Support Test")
    print("=" * 60)

    # Unit tests (always run)
    test_pad_truncate_utilities()

    # Integration test (optional)
    if not args.skip_integration:
        test_integration_with_actor()

    print("\n" + "=" * 60)
    print("Phase 7 Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
