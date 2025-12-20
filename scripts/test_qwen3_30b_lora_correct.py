#!/usr/bin/env python3
"""Test Qwen3-30B-A3B-Instruct-2507 LoRA loading with CORRECT target modules.

Previous test failed because qkv_proj is NOT supported for Qwen3MoeForCausalLM.
Must use separate q_proj, k_proj, v_proj instead.
"""

import ray
import os
import json
import traceback

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LOCAL_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


@ray.remote(num_gpus=4)
def test_qwen3_lora_correct():
    """Test LoRA loading on Qwen3-30B-A3B with correct target modules."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file
    from transformers import AutoConfig

    results = {
        "test_name": "qwen3_30b_lora_correct_modules",
    }

    # Get vLLM version
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
    except:
        results["vllm_version"] = "unknown"

    # Step 1: Get model config
    print("[1] Getting model config...", flush=True)
    config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    head_dim = hidden_size // num_heads

    q_dim = num_heads * head_dim      # 32 * 64 = 2048
    kv_dim = num_kv_heads * head_dim  # 4 * 64 = 256

    results["model_config"] = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_experts": getattr(config, "num_experts", None),
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "head_dim": head_dim,
    }
    print(f"    Config: {results['model_config']}", flush=True)

    # Step 2: Initialize vLLM
    print("\n[2] Initializing vLLM with enable_lora=True, TP=4...", flush=True)
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

    # Step 3: Baseline generation
    print("\n[3] Testing baseline generation...", flush=True)
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        results["baseline_output"] = output[0].outputs[0].text
        print(f"    Baseline: {results['baseline_output']}", flush=True)
    except Exception as e:
        results["baseline_error"] = str(e)
        del llm
        return results

    # Step 4: Create LoRA adapter with CORRECT target modules
    print("\n[4] Creating LoRA adapter with q_proj, k_proj, v_proj...", flush=True)
    lora_dir = "/tmp/test_qwen3_30b_lora_correct"
    os.makedirs(lora_dir, exist_ok=True)

    # Use separate q/k/v projections (NOT qkv_proj)
    adapter_config = {
        "base_model_name_or_path": MODEL_PATH,
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj"],  # Correct modules
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

    # Create LoRA weights for q/k/v separately
    lora_tensors = {}
    for layer_idx in range(num_layers):
        # q_proj: [q_dim, hidden_size] = [2048, 2048]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"] = \
            torch.zeros(q_dim, 8, dtype=torch.float16)

        # k_proj: [kv_dim, hidden_size] = [256, 2048]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight"] = \
            torch.zeros(kv_dim, 8, dtype=torch.float16)

        # v_proj: [kv_dim, hidden_size] = [256, 2048]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"] = \
            torch.zeros(kv_dim, 8, dtype=torch.float16)

    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    results["lora_tensors_count"] = len(lora_tensors)
    results["lora_target_modules"] = adapter_config["target_modules"]
    print(f"    Created {len(lora_tensors)} tensors for {adapter_config['target_modules']}", flush=True)

    # Step 5: Load LoRA adapter
    print("\n[5] Loading LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="test_qwen3_30b_correct",
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
        print("    ATTENTION LORA: SUCCESS!", flush=True)

    except Exception as e:
        results["lora_error"] = str(e)
        results["lora_error_type"] = type(e).__name__
        results["lora_traceback"] = traceback.format_exc()
        results["lora_success"] = False
        print(f"    ATTENTION LORA FAILED: {e}", flush=True)

        # Save error for analysis
        with open("/tmp/qwen3_30b_attn_lora_error.txt", "w") as f:
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
    print("Qwen3-30B-A3B LoRA Test with Correct Target Modules")
    print("=" * 70)
    print(f"Target modules: q_proj, k_proj, v_proj (NOT qkv_proj)")
    print()

    try:
        result = ray.get(
            test_qwen3_lora_correct.options(runtime_env=runtime_env).remote(),
            timeout=900
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
        print("\nSUMMARY: Attention LoRA SUCCEEDED on Qwen3-30B-A3B")
    else:
        print("\nSUMMARY: Attention LoRA FAILED on Qwen3-30B-A3B")
        if "lora_error" in result:
            print(f"Error: {result['lora_error']}")


if __name__ == "__main__":
    main()
