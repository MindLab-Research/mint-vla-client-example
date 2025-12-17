#!/usr/bin/env python3
"""Test K2 model configuration without loading the model.

This validates that:
1. K2 is detected as MoE model
2. K2 requires FP8 quantization
3. Training parallelism is correct (TP=8, EP=8)
4. Inference parallelism is correct (TP=8, DP=1)
"""

import sys
sys.path.insert(0, "/home/yiwen/tinker_project/tinker-server")

from tinker_server.backend.model_registry import (
    get_model_config,
    is_moe_model,
    is_supported_model,
    get_training_parallelism,
    get_recommended_parallelism,
    requires_fp8,
)


def test_k2_instruct():
    """Test Kimi-K2-Instruct configuration."""
    model = "moonshotai/Kimi-K2-Instruct"

    print(f"\n=== Testing {model} ===")

    # Check supported
    assert is_supported_model(model), f"{model} should be supported"
    print(f"  is_supported_model: True")

    # Check MoE
    assert is_moe_model(model), f"{model} should be MoE"
    print(f"  is_moe_model: True")

    # Check FP8
    assert requires_fp8(model), f"{model} should require FP8"
    print(f"  requires_fp8: True")

    # Check training parallelism
    train_tp, train_ep = get_training_parallelism(model)
    print(f"  training: TP={train_tp}, EP={train_ep}, world_size={train_tp * train_ep}")
    assert train_tp == 8, f"Expected train_tp=8, got {train_tp}"
    assert train_ep == 8, f"Expected train_ep=8, got {train_ep}"

    # Check inference parallelism
    infer_tp, infer_dp = get_recommended_parallelism(model)
    print(f"  inference: TP={infer_tp}, DP={infer_dp}, total_gpus={infer_tp * infer_dp}")
    assert infer_tp == 8, f"Expected infer_tp=8, got {infer_tp}"
    assert infer_dp == 1, f"Expected infer_dp=1, got {infer_dp}"

    # Check full config
    config = get_model_config(model)
    print(f"  Full config: is_moe={config.is_moe}, use_fp8={config.use_fp8}")
    print(f"    train_gpus={config.train_gpus}, total_gpus={config.total_gpus}")


def test_k2_thinking():
    """Test Kimi-K2-Thinking configuration."""
    model = "moonshotai/Kimi-K2-Thinking"

    print(f"\n=== Testing {model} ===")

    assert is_supported_model(model)
    assert is_moe_model(model)
    assert requires_fp8(model)

    train_tp, train_ep = get_training_parallelism(model)
    print(f"  training: TP={train_tp}, EP={train_ep}")

    config = get_model_config(model)
    print(f"  Full config: is_moe={config.is_moe}, use_fp8={config.use_fp8}")


def test_qwen_30b():
    """Test Qwen3-30B-A3B (existing MoE, no FP8)."""
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    print(f"\n=== Testing {model} ===")

    assert is_supported_model(model)
    assert is_moe_model(model)
    assert not requires_fp8(model), f"{model} should NOT require FP8"

    train_tp, train_ep = get_training_parallelism(model)
    print(f"  training: TP={train_tp}, EP={train_ep}")
    assert train_tp == 4
    assert train_ep == 1

    infer_tp, infer_dp = get_recommended_parallelism(model)
    print(f"  inference: TP={infer_tp}, DP={infer_dp}")


def test_dense_model():
    """Test dense model (no FP8, no MoE)."""
    model = "Qwen/Qwen3-0.6B"

    print(f"\n=== Testing {model} ===")

    assert is_supported_model(model)
    assert not is_moe_model(model)
    assert not requires_fp8(model)

    train_tp, train_ep = get_training_parallelism(model)
    print(f"  training: TP={train_tp}, EP={train_ep}")
    assert train_tp == 1
    assert train_ep == 1


def main():
    print("K2 Configuration Test")
    print("=" * 50)

    try:
        test_k2_instruct()
        test_k2_thinking()
        test_qwen_30b()
        test_dense_model()

        print("\n" + "=" * 50)
        print("All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
