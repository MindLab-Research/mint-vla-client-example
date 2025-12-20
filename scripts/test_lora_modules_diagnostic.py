#!/usr/bin/env python3
"""Diagnostic test: list supported LoRA modules and test minimal attention LoRA.

This script:
1. Lists all LoRA-supported modules for Qwen3MoE
2. Tests loading a minimal attention-only adapter
"""

import ray
import os
import json

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=4)
def diagnose_lora_support():
    """Diagnose LoRA support for Qwen3MoE model."""
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

    # Get supported modules
    print("Getting supported LoRA modules...")
    try:
        from vllm.lora.utils import get_supported_lora_modules
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        supported = get_supported_lora_modules(model)
        results["supported_modules"] = sorted(supported)
        results["num_supported_modules"] = len(supported)

        # Categorize
        results["has_qkv_proj"] = "qkv_proj" in supported
        results["has_o_proj"] = "o_proj" in supported
        results["has_experts"] = "experts" in supported
        results["has_gate"] = "gate" in supported

    except Exception as e:
        import traceback
        results["get_modules_error"] = str(e)
        results["get_modules_tb"] = traceback.format_exc()[:1000]

    # Get model config
    print("Getting model config...")
    try:
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
    except Exception as e:
        results["config_error"] = str(e)

    # Test baseline generation
    print("Testing baseline generation...")
    prompt = "What is 2+2? Answer briefly:"
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
        results["baseline_output"] = output[0].outputs[0].text
    except Exception as e:
        results["baseline_error"] = str(e)

    # Create and test minimal LoRA adapter (attention only)
    print("Creating minimal attention LoRA adapter...")
    lora_dir = "/tmp/test_attention_lora"
    os.makedirs(lora_dir, exist_ok=True)

    try:
        hidden_size = results.get("hidden_size", 2048)
        num_heads = results.get("num_attention_heads", 32)
        num_kv_heads = results.get("num_key_value_heads", 4)
        num_layers = results.get("num_hidden_layers", 48)
        head_dim = hidden_size // num_heads

        # qkv_proj: input=hidden_size, output=num_heads*head_dim + 2*num_kv_heads*head_dim
        q_dim = num_heads * head_dim
        kv_dim = 2 * num_kv_heads * head_dim
        qkv_output_dim = q_dim + kv_dim

        results["qkv_output_dim"] = qkv_output_dim
        results["head_dim"] = head_dim

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

        # Create minimal LoRA weights - just first 2 layers to keep it small
        lora_tensors = {}
        num_test_layers = 2
        for layer_idx in range(num_test_layers):
            # LoRA A: (rank, hidden_size)
            # LoRA B: (qkv_output_dim, rank)
            lora_a = torch.zeros(8, hidden_size, dtype=torch.float16)
            lora_b = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)

            # Small random init for A, zero for B (standard LoRA)
            torch.nn.init.normal_(lora_a, std=0.01)

            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

        from safetensors.torch import save_file
        save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")

        results["lora_adapter_created"] = True
        results["lora_num_test_layers"] = num_test_layers
        results["lora_tensor_keys"] = list(lora_tensors.keys())

    except Exception as e:
        import traceback
        results["create_adapter_error"] = str(e)
        results["create_adapter_tb"] = traceback.format_exc()[:1000]

    # Try loading the LoRA adapter
    print("Loading LoRA adapter...")
    try:
        lora_request = LoRARequest(
            lora_name="test_attention_lora",
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
        results["lora_load_error"] = str(e)[:1000]
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

    print("Running LoRA diagnostic test...")
    print("=" * 60)

    result = ray.get(
        diagnose_lora_support.options(runtime_env=runtime_env).remote(),
        timeout=600
    )

    print("\n=== Supported LoRA Modules ===")
    modules = result.get("supported_modules", [])
    print(f"Total: {len(modules)}")
    for m in modules:
        print(f"  - {m}")

    print(f"\nAttention (qkv_proj): {result.get('has_qkv_proj')}")
    print(f"Attention (o_proj): {result.get('has_o_proj')}")
    print(f"Expert (experts): {result.get('has_experts')}")
    print(f"Router (gate): {result.get('has_gate')}")

    print("\n=== Model Config ===")
    print(f"Hidden size: {result.get('hidden_size')}")
    print(f"Num layers: {result.get('num_hidden_layers')}")
    print(f"Num attention heads: {result.get('num_attention_heads')}")
    print(f"Num KV heads: {result.get('num_key_value_heads')}")
    print(f"Num experts: {result.get('num_experts')}")
    print(f"MoE intermediate size: {result.get('moe_intermediate_size')}")

    print("\n=== Generation Test ===")
    print(f"Baseline: {result.get('baseline_output')}")
    print(f"Baseline error: {result.get('baseline_error')}")

    print("\n=== LoRA Adapter Test ===")
    print(f"Adapter created: {result.get('lora_adapter_created')}")
    print(f"QKV output dim: {result.get('qkv_output_dim')}")
    if result.get('create_adapter_error'):
        print(f"Create error: {result.get('create_adapter_error')}")

    print(f"LoRA load success: {result.get('lora_load_success')}")
    if result.get('lora_load_success'):
        print(f"LoRA output: {result.get('lora_output')}")
    else:
        print(f"LoRA load error: {result.get('lora_load_error')}")
        print(f"\nTraceback:\n{result.get('lora_load_tb')}")

    print("\n" + "=" * 60)
    print("VERDICT:")
    if result.get('has_experts'):
        print("  Expert LoRA modules: REGISTERED")
    else:
        print("  Expert LoRA modules: NOT REGISTERED")

    if result.get('lora_load_success'):
        print("  Attention LoRA loading: WORKS")
    else:
        print("  Attention LoRA loading: FAILED")


if __name__ == "__main__":
    main()
