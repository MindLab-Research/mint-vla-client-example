#!/usr/bin/env python3
"""Test Qwen3-30B-A3B-Instruct-2507 LoRA loading on Ray cluster.

Identifies the exact error when loading LoRA adapters on this MoE model.
"""

import ray
import os
import json
import traceback

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LOCAL_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


@ray.remote(num_gpus=4)
def test_qwen3_lora():
    """Test LoRA loading on Qwen3-30B-A3B with TP=4."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file
    from transformers import AutoConfig

    results = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }

    # Get vLLM version
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
    except:
        results["vllm_version"] = "unknown"

    # Step 1: Get model config
    print("[1] Getting model config...", flush=True)
    try:
        config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        results["model_config"] = {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_experts": getattr(config, "num_experts", None),
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "architectures": config.architectures,
        }
        print(f"    Config: {results['model_config']}", flush=True)
    except Exception as e:
        results["config_error"] = str(e)
        return results

    # Step 2: Initialize vLLM with enable_lora=True
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
        print(f"    INIT FAILED: {e}", flush=True)
        return results

    # Step 3: Test baseline generation
    print("\n[3] Testing baseline generation (no LoRA)...", flush=True)
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        results["baseline_output"] = output[0].outputs[0].text
        print(f"    Baseline: {results['baseline_output']}", flush=True)
    except Exception as e:
        results["baseline_error"] = str(e)
        results["baseline_traceback"] = traceback.format_exc()
        del llm
        return results

    # Step 4: Create LoRA adapter
    print("\n[4] Creating LoRA adapter...", flush=True)
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    head_dim = hidden_size // num_heads

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_output_dim = q_dim + 2 * kv_dim

    lora_dir = "/tmp/test_qwen3_30b_lora"
    os.makedirs(lora_dir, exist_ok=True)

    # Try qkv_proj first (Qwen3 may use fused)
    adapter_config = {
        "base_model_name_or_path": MODEL_PATH,
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["qkv_proj"],
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

    # Create LoRA weights for all layers
    lora_tensors = {}
    for layer_idx in range(num_layers):
        lora_a = torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_b = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)

        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    results["lora_tensors_count"] = len(lora_tensors)
    results["lora_target_modules"] = adapter_config["target_modules"]
    print(f"    Created {len(lora_tensors)} tensors", flush=True)

    # Step 5: Load LoRA adapter
    print("\n[5] Loading LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="test_qwen3_30b",
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
        print("    SUCCESS!", flush=True)

    except Exception as e:
        results["lora_error"] = str(e)
        results["lora_error_type"] = type(e).__name__
        results["lora_traceback"] = traceback.format_exc()
        results["lora_success"] = False
        print(f"    LORA FAILED: {e}", flush=True)
        print(f"    Error type: {type(e).__name__}", flush=True)

        # Save detailed error to file
        with open("/tmp/qwen3_30b_lora_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"vLLM version: {results.get('vllm_version', 'unknown')}\n")
            f.write(f"TP size: 4\n\n")
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
    print("Qwen3-30B-A3B-Instruct-2507 LoRA Test on Ray Cluster")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Local path: {LOCAL_MODEL_PATH}")
    print()

    try:
        result = ray.get(
            test_qwen3_lora.options(runtime_env=runtime_env).remote(),
            timeout=900  # 15 min timeout
        )
    except Exception as e:
        print(f"Ray task failed: {e}")
        print(traceback.format_exc())
        return

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Print non-traceback results
    for k, v in result.items():
        if not k.endswith("_traceback"):
            print(f"{k}: {v}")

    # Print tracebacks separately
    for k, v in result.items():
        if k.endswith("_traceback"):
            print(f"\n{k}:")
            print("-" * 40)
            print(v)

    print("=" * 70)

    # Summary
    if result.get("lora_success"):
        print("\nSUMMARY: LoRA loading SUCCEEDED")
    else:
        print("\nSUMMARY: LoRA loading FAILED")
        if "lora_error" in result:
            print(f"Error: {result['lora_error']}")
            print(f"Type: {result.get('lora_error_type', 'unknown')}")


if __name__ == "__main__":
    main()
