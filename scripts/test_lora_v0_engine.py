#!/usr/bin/env python3
"""Test LoRA adapter loading with V0 engine (VLLM_USE_V1=0).

The V1 engine fails at set_active_loras. This test checks if V0 works.
"""

import ray
import os
import json

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=4)
def test_lora_v0():
    """Test LoRA loading with V0 engine."""
    # Force V0 engine
    os.environ["VLLM_USE_V1"] = "0"

    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    results = {}

    # Initialize model with V0 engine
    print("Initializing model with V0 engine (VLLM_USE_V1=0)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,
    )
    results["init_success"] = True

    # Get model config
    print("Getting model config...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        trust_remote_code=True
    )
    results["hidden_size"] = config.hidden_size
    results["num_hidden_layers"] = config.num_hidden_layers
    results["num_attention_heads"] = config.num_attention_heads
    results["num_key_value_heads"] = config.num_key_value_heads

    # Get supported modules
    print("Getting supported LoRA modules...")
    try:
        from vllm.lora.utils import get_supported_lora_modules
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        supported = get_supported_lora_modules(model)
        results["supported_modules"] = sorted(supported)
    except Exception as e:
        import traceback
        results["get_modules_error"] = str(e)
        results["get_modules_tb"] = traceback.format_exc()

    # Baseline generation
    print("Testing baseline generation...")
    prompt = "What is 2+2? Answer briefly:"
    output = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
    results["baseline_output"] = output[0].outputs[0].text

    # Create LoRA adapter
    print("Creating LoRA adapter...")
    lora_dir = "/tmp/test_v0_lora"
    os.makedirs(lora_dir, exist_ok=True)

    try:
        hidden_size = results["hidden_size"]
        num_heads = results["num_attention_heads"]
        num_kv_heads = results["num_key_value_heads"]
        num_layers = results["num_hidden_layers"]
        head_dim = hidden_size // num_heads

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        qkv_output_dim = q_dim + 2 * kv_dim

        results["qkv_output_dim"] = qkv_output_dim

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

        results["lora_adapter_created"] = True
        results["lora_num_layers"] = num_layers

    except Exception as e:
        import traceback
        results["create_adapter_error"] = str(e)
        results["create_adapter_tb"] = traceback.format_exc()

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    try:
        lora_request = LoRARequest(
            lora_name="test_v0_lora",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=20, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["lora_load_success"] = True

    except Exception as e:
        import traceback
        results["lora_load_error"] = str(e)
        results["lora_load_tb"] = traceback.format_exc()

    del llm
    return results


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    runtime_env = {
        "env_vars": {
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
            "VLLM_USE_V1": "0",  # Force V0 engine
        }
    }

    print("Running LoRA test with V0 engine...")
    print("=" * 60)

    result = ray.get(
        test_lora_v0.options(runtime_env=runtime_env).remote(),
        timeout=600
    )

    print("\n=== Model Config ===")
    print(f"Hidden size: {result.get('hidden_size')}")
    print(f"Num layers: {result.get('num_hidden_layers')}")
    print(f"Num attention heads: {result.get('num_attention_heads')}")
    print(f"Num KV heads: {result.get('num_key_value_heads')}")

    print("\n=== Supported LoRA Modules ===")
    modules = result.get("supported_modules", [])
    print(f"Total: {len(modules)}")
    for m in modules:
        print(f"  - {m}")
    if result.get("get_modules_error"):
        print(f"Error: {result.get('get_modules_error')}")

    print("\n=== Baseline Generation ===")
    print(f"Output: {result.get('baseline_output')}")

    print("\n=== LoRA Adapter ===")
    print(f"Created: {result.get('lora_adapter_created')}")
    print(f"QKV output dim: {result.get('qkv_output_dim')}")
    print(f"Num layers: {result.get('lora_num_layers')}")
    if result.get("create_adapter_error"):
        print(f"Create error: {result.get('create_adapter_error')}")

    print("\n=== LoRA Loading Test ===")
    print(f"Load success: {result.get('lora_load_success')}")
    if result.get('lora_load_success'):
        print(f"LoRA output: {result.get('lora_output')}")
    else:
        print(f"Error: {result.get('lora_load_error')}")
        print(f"\nFull traceback:\n{result.get('lora_load_tb')}")

    print("\n" + "=" * 60)
    print("VERDICT:")
    if result.get('lora_load_success'):
        print("  V0 Engine LoRA: WORKS")
    else:
        print("  V0 Engine LoRA: FAILED")


if __name__ == "__main__":
    main()
