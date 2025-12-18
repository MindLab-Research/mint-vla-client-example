#!/usr/bin/env python3
"""Debug vLLM LoRA adapter loading for Qwen3MoE.

This script:
1. Inspects supported_lora_modules and packed_modules_mapping
2. Investigates the exact expected weight format
3. Tests loading with verbose error output
"""

import ray
import os
import json

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=4)
def debug_lora_loading():
    """Debug LoRA loading for Qwen3MoE model."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    results = {}

    # Initialize model
    print("Initializing model...")
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
    results["num_experts"] = config.num_experts
    results["moe_intermediate_size"] = config.moe_intermediate_size

    # Get supported modules
    print("Getting supported LoRA modules...")
    try:
        from vllm.lora.utils import get_supported_lora_modules
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        supported = get_supported_lora_modules(model)
        results["supported_modules"] = sorted(supported)
        results["num_supported_modules"] = len(supported)
    except Exception as e:
        import traceback
        results["get_modules_error"] = str(e)
        results["get_modules_tb"] = traceback.format_exc()

    # Try to get adapter manager info
    print("Getting adapter manager info...")
    try:
        # Access the lora manager
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
        if hasattr(model_runner, 'lora_manager'):
            lora_manager = model_runner.lora_manager
            adapter_manager = lora_manager._adapter_manager

            results["adapter_supported_modules"] = list(adapter_manager.supported_lora_modules)
            results["packed_modules_mapping"] = dict(adapter_manager.packed_modules_mapping)

            # Get embedding modules
            if hasattr(lora_manager, 'embedding_modules'):
                results["embedding_modules"] = lora_manager.embedding_modules
        else:
            results["lora_manager_note"] = "lora_manager not found on model_runner"
    except Exception as e:
        import traceback
        results["adapter_manager_error"] = str(e)
        results["adapter_manager_tb"] = traceback.format_exc()

    # Generate baseline
    print("Testing baseline generation...")
    prompt = "What is 2+2? Answer briefly:"
    output = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
    results["baseline_output"] = output[0].outputs[0].text

    # Create LoRA adapter with proper qkv_proj dimensions
    print("Creating LoRA adapter...")
    lora_dir = "/tmp/test_debug_lora"
    os.makedirs(lora_dir, exist_ok=True)

    try:
        hidden_size = results["hidden_size"]
        num_heads = results["num_attention_heads"]
        num_kv_heads = results["num_key_value_heads"]
        num_layers = results["num_hidden_layers"]
        head_dim = hidden_size // num_heads

        # For Qwen3MoE, qkv_proj output dim should be:
        # Q: num_heads * head_dim = hidden_size
        # K: num_kv_heads * head_dim
        # V: num_kv_heads * head_dim
        q_dim = num_heads * head_dim  # = hidden_size
        kv_dim = num_kv_heads * head_dim
        qkv_output_dim = q_dim + 2 * kv_dim

        results["calculated_dims"] = {
            "head_dim": head_dim,
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_output_dim": qkv_output_dim,
        }

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

        # Create LoRA weights for ALL layers (not just 2)
        lora_tensors = {}
        for layer_idx in range(num_layers):
            # LoRA A: (rank, hidden_size)
            # LoRA B: (qkv_output_dim, rank)
            lora_a = torch.zeros(8, hidden_size, dtype=torch.float16)
            lora_b = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)

            # Small random init for A, zero for B
            torch.nn.init.normal_(lora_a, std=0.01)

            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

        from safetensors.torch import save_file
        save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")

        results["lora_adapter_created"] = True
        results["lora_num_layers"] = num_layers
        results["lora_tensor_keys_sample"] = list(lora_tensors.keys())[:4]
        results["lora_a_shape"] = list(lora_a.shape)
        results["lora_b_shape"] = list(lora_b.shape)

    except Exception as e:
        import traceback
        results["create_adapter_error"] = str(e)
        results["create_adapter_tb"] = traceback.format_exc()

    # Try loading the LoRA adapter with detailed error capture
    print("Loading LoRA adapter...")
    try:
        lora_request = LoRARequest(
            lora_name="test_debug_lora",
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
        }
    }

    print("Running LoRA debug test...")
    print("=" * 60)

    result = ray.get(
        debug_lora_loading.options(runtime_env=runtime_env).remote(),
        timeout=600
    )

    print("\n=== Model Config ===")
    print(f"Hidden size: {result.get('hidden_size')}")
    print(f"Num layers: {result.get('num_hidden_layers')}")
    print(f"Num attention heads: {result.get('num_attention_heads')}")
    print(f"Num KV heads: {result.get('num_key_value_heads')}")
    print(f"Num experts: {result.get('num_experts')}")
    print(f"MoE intermediate size: {result.get('moe_intermediate_size')}")

    print("\n=== Supported LoRA Modules ===")
    modules = result.get("supported_modules", [])
    print(f"Total: {len(modules)}")
    for m in modules:
        print(f"  - {m}")
    if result.get("get_modules_error"):
        print(f"Error: {result.get('get_modules_error')}")

    print("\n=== Adapter Manager Info ===")
    if result.get("adapter_supported_modules"):
        print(f"Adapter supported modules: {result.get('adapter_supported_modules')}")
    if result.get("packed_modules_mapping"):
        print(f"Packed modules mapping: {result.get('packed_modules_mapping')}")
    if result.get("adapter_manager_error"):
        print(f"Error: {result.get('adapter_manager_error')}")
        print(f"TB: {result.get('adapter_manager_tb')}")

    print("\n=== Calculated Dimensions ===")
    dims = result.get("calculated_dims", {})
    for k, v in dims.items():
        print(f"  {k}: {v}")

    print("\n=== Baseline Generation ===")
    print(f"Output: {result.get('baseline_output')}")

    print("\n=== LoRA Adapter Creation ===")
    print(f"Created: {result.get('lora_adapter_created')}")
    print(f"Num layers: {result.get('lora_num_layers')}")
    print(f"LoRA A shape: {result.get('lora_a_shape')}")
    print(f"LoRA B shape: {result.get('lora_b_shape')}")
    print(f"Sample keys: {result.get('lora_tensor_keys_sample')}")
    if result.get("create_adapter_error"):
        print(f"Error: {result.get('create_adapter_error')}")

    print("\n=== LoRA Loading Test ===")
    print(f"Load success: {result.get('lora_load_success')}")
    if result.get('lora_load_success'):
        print(f"LoRA output: {result.get('lora_output')}")
    else:
        print(f"Error: {result.get('lora_load_error')}")
        print(f"\nFull traceback:\n{result.get('lora_load_tb')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
