#!/usr/bin/env python3
"""Quick test for MoE LoRA inference with vLLM 0.13.0rc2."""

import ray

# Use model name - HF cache will resolve to local snapshot
MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote
def check_vllm_moe_lora_modules():
    """Check if vLLM has FusedMoE LoRA modules (no GPU needed)."""
    import vllm
    results = {"vllm_version": vllm.__version__}

    # Check FusedMoE LoRA module
    try:
        from vllm.lora.layers import fused_moe
        results["fused_moe_lora_module"] = True
        results["fused_moe_lora_classes"] = [
            name for name in dir(fused_moe)
            if not name.startswith("_") and "LoRA" in name
        ]
    except ImportError as e:
        results["fused_moe_lora_module"] = False
        results["fused_moe_lora_error"] = str(e)

    return results


@ray.remote(num_gpus=4)
def test_moe_model_with_lora():
    """Test loading MoE model with enable_lora (requires GPU)."""
    from vllm import LLM, SamplingParams
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

        # Quick generation test
        output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
        results["generation"] = output[0].outputs[0].text[:50]
        del llm

    except Exception as e:
        results["moe_lora_init"] = False
        results["moe_lora_error"] = str(e)[:500]

    return results


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    runtime_env = {
        "env_vars": {
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
        }
    }

    print("Step 1: Checking vLLM modules...")
    result1 = ray.get(
        check_vllm_moe_lora_modules.options(runtime_env=runtime_env).remote(),
        timeout=60
    )
    version = result1.get("vllm_version")
    has_module = result1.get("fused_moe_lora_module")
    classes = result1.get("fused_moe_lora_classes", [])
    error = result1.get("fused_moe_lora_error")

    print(f"  vLLM version: {version}")
    print(f"  FusedMoE LoRA module: {has_module}")
    if classes:
        print(f"  LoRA classes: {classes}")
    if error:
        print(f"  Error: {error}")

    print()
    print("Step 2: Testing MoE model with enable_lora (4 GPUs)...")
    result2 = ray.get(
        test_moe_model_with_lora.options(runtime_env=runtime_env).remote(),
        timeout=300
    )
    init_ok = result2.get("moe_lora_init")
    gen = result2.get("generation")
    err = result2.get("moe_lora_error")

    print(f"  MoE LoRA init: {init_ok}")
    if gen:
        print(f"  Generation: {gen}")
    if err:
        print(f"  Error: {err}")

    print()
    print("RESULT:", "PASS" if init_ok else "FAIL")


if __name__ == "__main__":
    main()
