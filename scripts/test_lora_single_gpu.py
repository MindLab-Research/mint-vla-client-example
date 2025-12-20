#!/usr/bin/env python3
"""Single-GPU LoRA test for clearer error messages.

Uses TP=1 to reduce multiprocess complexity and get clearer stack traces.
"""

import ray
import os
import json
import sys

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=1)
def test_lora_single_gpu():
    """Test LoRA with single GPU for clearer errors."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import traceback

    results = {}

    # Initialize model with TP=1
    print("Initializing model with TP=1...", flush=True)
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,  # Single GPU
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.5,  # Lower memory for single GPU
        )
        results["init_success"] = True
    except Exception as e:
        results["init_error"] = str(e)
        results["init_tb"] = traceback.format_exc()
        return results

    # Get model config
    print("Getting model config...", flush=True)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        trust_remote_code=True
    )
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    head_dim = hidden_size // num_heads

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_output_dim = q_dim + 2 * kv_dim

    results["hidden_size"] = hidden_size
    results["num_layers"] = num_layers
    results["qkv_output_dim"] = qkv_output_dim

    # Baseline test
    print("Testing baseline...", flush=True)
    prompt = "2+2="
    try:
        baseline = llm.generate([prompt], SamplingParams(max_tokens=5, temperature=0))
        results["baseline"] = baseline[0].outputs[0].text
        print(f"Baseline: {results['baseline']}", flush=True)
    except Exception as e:
        results["baseline_error"] = str(e)
        results["baseline_tb"] = traceback.format_exc()
        return results

    # Create LoRA adapter
    print("Creating LoRA adapter...", flush=True)
    lora_dir = "/tmp/test_single_gpu_lora"
    os.makedirs(lora_dir, exist_ok=True)

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

    # Create weights for ALL layers
    lora_tensors = {}
    for layer_idx in range(num_layers):
        lora_a = torch.zeros(8, hidden_size, dtype=torch.float16)
        lora_b = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)
        torch.nn.init.normal_(lora_a, std=0.01)

        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

    from safetensors.torch import save_file
    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    results["lora_created"] = True

    # Load LoRA
    print("Loading LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="test_single",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=5, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["success"] = True

    except Exception as e:
        results["lora_error"] = str(e)
        results["lora_tb"] = traceback.format_exc()
        results["success"] = False

        # Write full error to file
        with open("/tmp/lora_single_gpu_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())

        print(f"ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

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

    print("Running single-GPU LoRA test...")
    print("=" * 60)

    result = ray.get(
        test_lora_single_gpu.options(runtime_env=runtime_env).remote(),
        timeout=900  # 15 min timeout
    )

    print("\n=== Results ===")
    for k, v in result.items():
        if k.endswith("_tb"):
            print(f"\n{k}:\n{v}")
        else:
            print(f"{k}: {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
