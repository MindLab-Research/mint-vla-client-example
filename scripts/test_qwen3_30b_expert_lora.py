#!/usr/bin/env python3
"""Test Qwen3-30B-A3B-Instruct-2507 Expert LoRA loading on Ray cluster.

Tests whether MoE expert layer LoRA works on this model.
Config: 48 layers, 128 experts, moe_intermediate_size=768, hidden_size=2048
"""

import ray
import os
import json
import traceback

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LOCAL_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


@ray.remote(num_gpus=4)
def test_qwen3_expert_lora():
    """Test expert LoRA loading on Qwen3-30B-A3B with TP=4."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file

    results = {
        "test_name": "qwen3_30b_expert_lora",
    }

    # Get vLLM version
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
    except:
        results["vllm_version"] = "unknown"

    # Model dimensions for Qwen3-30B-A3B
    hidden_size = 2048
    num_layers = 48
    num_experts = 128
    moe_intermediate_size = 768
    rank = 8

    results["model_config"] = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "moe_intermediate_size": moe_intermediate_size,
    }
    print(f"Model config: {results['model_config']}", flush=True)

    # Step 1: Initialize vLLM
    print("\n[1] Initializing vLLM with enable_lora=True, TP=4...", flush=True)
    try:
        llm = LLM(
            model=LOCAL_MODEL_PATH,
            tensor_parallel_size=4,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )
        results["init_success"] = True
        print("    Model initialized successfully", flush=True)
    except Exception as e:
        results["init_error"] = str(e)
        results["init_traceback"] = traceback.format_exc()
        return results

    # Step 2: Baseline generation
    print("\n[2] Testing baseline generation...", flush=True)
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        results["baseline_output"] = output[0].outputs[0].text
        print(f"    Baseline: {results['baseline_output']}", flush=True)
    except Exception as e:
        results["baseline_error"] = str(e)
        del llm
        return results

    # Step 3: Create expert LoRA adapter for ALL 128 experts
    print("\n[3] Creating expert LoRA adapter (ALL 128 experts)...", flush=True)
    lora_dir = "/tmp/test_qwen3_30b_expert_lora"
    os.makedirs(lora_dir, exist_ok=True)

    # Build target_modules list for all experts
    target_modules = []
    for e in range(num_experts):
        target_modules.extend([
            f"experts.{e}.gate_proj",
            f"experts.{e}.up_proj",
            f"experts.{e}.down_proj",
        ])

    adapter_config = {
        "base_model_name_or_path": MODEL_PATH,
        "r": rank,
        "lora_alpha": 16,
        "target_modules": target_modules,
        "modules_to_save": None,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "inference_mode": True,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM"
    }
    with open(f"{lora_dir}/adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Create LoRA weights for ALL experts in ALL layers
    print("    Creating tensors (this may take a moment)...", flush=True)
    lora_tensors = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            # gate_proj (w1): [moe_intermediate_size, hidden_size] = [768, 2048]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_A.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_B.weight"] = \
                torch.zeros(moe_intermediate_size, rank, dtype=torch.float16)

            # up_proj (w3): [moe_intermediate_size, hidden_size] = [768, 2048]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_A.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_B.weight"] = \
                torch.zeros(moe_intermediate_size, rank, dtype=torch.float16)

            # down_proj (w2): [hidden_size, moe_intermediate_size] = [2048, 768]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_A.weight"] = \
                torch.randn(rank, moe_intermediate_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_B.weight"] = \
                torch.zeros(hidden_size, rank, dtype=torch.float16)

    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    num_expected = num_layers * num_experts * 3 * 2  # 48 * 128 * 3 * 2 = 36864
    results["lora_tensors_count"] = len(lora_tensors)
    results["expected_tensors"] = num_expected
    print(f"    Created {len(lora_tensors)} tensors (expected: {num_expected})", flush=True)

    # Step 4: Load expert LoRA adapter
    print("\n[4] Loading expert LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="test_qwen3_30b_expert",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["lora_success"] = True
        print(f"    LoRA output: {results['lora_output']}", flush=True)
        print("    EXPERT LORA: SUCCESS!", flush=True)

    except Exception as e:
        results["lora_error"] = str(e)
        results["lora_error_type"] = type(e).__name__
        results["lora_traceback"] = traceback.format_exc()
        results["lora_success"] = False
        print(f"    EXPERT LORA FAILED: {e}", flush=True)

        # Save error for analysis
        with open("/tmp/qwen3_30b_expert_lora_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())

    del llm
    return results


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    runtime_env = {
        "env_vars": {
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
        }
    }

    print("=" * 70)
    print("Qwen3-30B-A3B Expert LoRA Test on Ray Cluster")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: 48 layers, 128 experts, moe_intermediate_size=768")
    print()

    try:
        result = ray.get(
            test_qwen3_expert_lora.options(runtime_env=runtime_env).remote(),
            timeout=1200  # 20 min timeout (large model + many tensors)
        )
    except Exception as e:
        print(f"Ray task failed: {e}")
        print(traceback.format_exc())
        return

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for k, v in result.items():
        if not k.endswith("_traceback"):
            print(f"{k}: {v}")

    for k, v in result.items():
        if k.endswith("_traceback"):
            print(f"\n{k}:")
            print("-" * 40)
            print(v)

    print("=" * 70)

    if result.get("lora_success"):
        print("\nSUMMARY: Expert LoRA SUCCEEDED on Qwen3-30B-A3B")
    else:
        print("\nSUMMARY: Expert LoRA FAILED on Qwen3-30B-A3B")
        if "lora_error" in result:
            print(f"Error: {result['lora_error']}")


if __name__ == "__main__":
    main()
