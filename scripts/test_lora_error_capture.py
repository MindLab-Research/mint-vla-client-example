#!/usr/bin/env python3
"""Capture the full LoRA loading error traceback."""

import ray
import os
import json
import sys

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=4)
def test_lora_error():
    """Test LoRA loading and capture full error."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import traceback

    results = {}

    # Initialize model
    print("Initializing model...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,
    )

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

    results["config"] = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "qkv_output_dim": qkv_output_dim,
    }

    # Create LoRA adapter
    print("Creating LoRA adapter...", flush=True)
    lora_dir = "/tmp/test_lora_error"
    os.makedirs(lora_dir, exist_ok=True)

    # Adapter config
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

    # Create LoRA weights for ALL layers
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
    results["lora_keys_sample"] = list(lora_tensors.keys())[:2]

    # Test baseline
    print("Testing baseline generation...", flush=True)
    prompt = "What is 2+2?"
    baseline = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
    results["baseline"] = baseline[0].outputs[0].text
    print(f"Baseline: {results['baseline']}", flush=True)

    # Load LoRA adapter
    print("Loading LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="test_error_lora",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["success"] = True

    except Exception as e:
        # Capture full traceback
        full_tb = traceback.format_exc()
        results["error"] = str(e)
        results["error_type"] = type(e).__name__
        results["full_traceback"] = full_tb
        results["success"] = False
        print(f"ERROR: {e}", flush=True)
        print(f"Full traceback:\n{full_tb}", flush=True)

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

    print("Running LoRA error capture test...")
    print("=" * 70)

    result = ray.get(
        test_lora_error.options(runtime_env=runtime_env).remote(),
        timeout=600
    )

    print("\n=== Results ===")
    print(f"Config: {result.get('config')}")
    print(f"LoRA created: {result.get('lora_created')}")
    print(f"LoRA keys sample: {result.get('lora_keys_sample')}")
    print(f"Baseline: {result.get('baseline')}")
    print(f"Success: {result.get('success')}")

    if result.get('success'):
        print(f"LoRA output: {result.get('lora_output')}")
    else:
        print(f"\nError type: {result.get('error_type')}")
        print(f"Error: {result.get('error')}")
        print(f"\n{'=' * 70}")
        print("FULL TRACEBACK:")
        print("=" * 70)
        print(result.get('full_traceback'))

    print("=" * 70)


if __name__ == "__main__":
    main()
