#!/usr/bin/env python3
"""Test if vLLM 0.12.0 supports FusedMoE expert LoRA.

This script checks:
1. vLLM version
2. Whether FusedMoE LoRA classes exist
3. Whether we can load an MoE model with expert LoRA weights
"""

import ray

VLLM_PFS_PATH = "/vePFS-Mindverse/share/code/vllm-0.12.0"
MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/4f41ec5a0313c8f4e2a5efde73fa2e999c596ef3"


@ray.remote
def check_vllm_moe_lora_modules():
    """Check if vLLM has FusedMoE LoRA modules (no GPU needed)."""
    import vllm

    results = {
        "vllm_version": vllm.__version__,
        "vllm_path": vllm.__file__,
    }

    # Check 1: FusedMoE LoRA module exists?
    try:
        from vllm.lora.layers import fused_moe
        results["fused_moe_lora_module"] = True
        results["fused_moe_lora_path"] = fused_moe.__file__

        # Check what classes are available
        results["fused_moe_lora_classes"] = [
            name for name in dir(fused_moe)
            if not name.startswith('_') and 'LoRA' in name
        ]
    except ImportError as e:
        results["fused_moe_lora_module"] = False
        results["fused_moe_lora_error"] = str(e)

    # Check 2: Main fused_moe module classes
    try:
        from vllm.model_executor.layers import fused_moe as main_fused_moe
        results["main_fused_moe_classes"] = [
            name for name in dir(main_fused_moe)
            if not name.startswith('_') and ('LoRA' in name or 'Unfused' in name or 'Expert' in name)
        ]
    except ImportError as e:
        results["main_fused_moe_error"] = str(e)

    # Check 3: Check for the specific PR #29708 class
    try:
        # Try different import paths
        unfused_class = None
        for module_path in [
            "vllm.model_executor.layers.fused_moe.fused_moe",
            "vllm.model_executor.layers.fused_moe",
            "vllm.model_executor.layers.fused_moe.triton_moe"
        ]:
            try:
                mod = __import__(module_path, fromlist=['UnfusedOAITritonExperts'])
                if hasattr(mod, 'UnfusedOAITritonExperts'):
                    unfused_class = f"{module_path}.UnfusedOAITritonExperts"
                    break
            except (ImportError, AttributeError):
                continue

        results["unfused_triton_experts"] = unfused_class is not None
        if unfused_class:
            results["unfused_triton_experts_path"] = unfused_class
    except Exception as e:
        results["unfused_triton_experts"] = False
        results["unfused_triton_experts_error"] = str(e)

    # Check 4: Check lora/layers module contents
    try:
        from vllm.lora import layers as lora_layers
        results["lora_layers_contents"] = [
            name for name in dir(lora_layers)
            if not name.startswith('_')
        ]
    except ImportError as e:
        results["lora_layers_error"] = str(e)

    return results


@ray.remote(num_gpus=4)
def test_moe_model_with_lora():
    """Test loading MoE model with enable_lora (requires GPU)."""
    from vllm import LLM

    results = {}

    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=4,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=1024,
            enforce_eager=True,
        )
        results["moe_lora_init"] = True
        results["model_loaded"] = True

        # Check if there's any warning about FusedMoE LoRA
        del llm

    except Exception as e:
        results["moe_lora_init"] = False
        results["moe_lora_error"] = str(e)
        error_str = str(e).lower()
        if "fusedmoe" in error_str or "expert" in error_str or "lora" in error_str:
            results["error_type"] = "fusedmoe_lora_related"
        else:
            results["error_type"] = "other"

    return results


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{VLLM_PFS_PATH}:$PYTHONPATH",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
        }
    }

    print("=" * 60)
    print("Testing vLLM FusedMoE LoRA support")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print()

    # Step 1: Check modules (no GPU)
    print("Step 1: Checking vLLM modules (no GPU)...")
    try:
        result = ray.get(
            check_vllm_moe_lora_modules.options(runtime_env=runtime_env).remote(),
            timeout=60,
        )

        print(f"  vLLM version: {result.get('vllm_version')}")
        print(f"  FusedMoE LoRA module exists: {result.get('fused_moe_lora_module')}")
        if result.get('fused_moe_lora_path'):
            print(f"    Path: {result.get('fused_moe_lora_path')}")
        if result.get('fused_moe_lora_classes'):
            print(f"    LoRA classes: {result.get('fused_moe_lora_classes')}")
        print(f"  UnfusedOAITritonExperts (PR #29708): {result.get('unfused_triton_experts')}")
        if result.get('unfused_triton_experts_path'):
            print(f"    Path: {result.get('unfused_triton_experts_path')}")
        if result.get('main_fused_moe_classes'):
            print(f"  Main FusedMoE classes: {result.get('main_fused_moe_classes')}")
        print()

        modules_ok = result.get('fused_moe_lora_module', False)

    except Exception as e:
        print(f"  ERROR: {e}")
        modules_ok = False

    # Step 2: Test model loading (requires GPU)
    print("Step 2: Testing MoE model with enable_lora (4 GPUs)...")
    try:
        result2 = ray.get(
            test_moe_model_with_lora.options(runtime_env=runtime_env).remote(),
            timeout=300,
        )

        print(f"  MoE model init with enable_lora: {result2.get('moe_lora_init')}")
        if result2.get('moe_lora_error'):
            print(f"  Error: {result2.get('moe_lora_error')[:500]}")
            print(f"  Error type: {result2.get('error_type')}")
        print()

        model_ok = result2.get('moe_lora_init', False)

    except Exception as e:
        print(f"  ERROR: {e}")
        model_ok = False

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if modules_ok and model_ok:
        print("PASS: vLLM 0.12.0 supports MoE with LoRA")
        print("Next: Remove MLP filter and test with actual expert LoRA weights")
    elif modules_ok and not model_ok:
        print("PARTIAL: Modules exist but model loading failed")
        print("May need specific configuration or adapter format")
    else:
        print("FAIL: vLLM does not have FusedMoE LoRA support")


if __name__ == "__main__":
    main()
