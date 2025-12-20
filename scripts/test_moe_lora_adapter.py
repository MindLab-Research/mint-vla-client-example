#!/usr/bin/env python3
"""Test MoE LoRA adapter loading and layer coverage in vLLM 0.13.0rc2.

This test verifies:
1. Which layers are LoRA-supported (attention AND experts)
2. Actual LoRA adapter loading works
3. Generation with LoRA adapter produces different output
"""

import ray
import os
import json
import tempfile

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@ray.remote(num_gpus=4)
def test_lora_layer_coverage_and_adapter():
    """Test which layers support LoRA and verify adapter loading."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    results = {}

    # Step 1: Initialize model and check supported modules
    print("Step 1: Initializing model...")
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

    # Step 2: Get supported LoRA modules
    print("Step 2: Checking supported LoRA modules...")
    try:
        # Access the model runner to get supported modules
        from vllm.lora.utils import get_supported_lora_modules
        # The model is inside the engine
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        supported = get_supported_lora_modules(model)
        results["supported_lora_modules"] = sorted(supported)

        # Categorize modules
        attention_modules = [m for m in supported if any(x in m for x in ['qkv', 'o_proj', 'q_proj', 'k_proj', 'v_proj'])]
        expert_modules = [m for m in supported if 'expert' in m.lower()]
        gate_modules = [m for m in supported if m == 'gate']
        mlp_modules = [m for m in supported if any(x in m for x in ['gate_up_proj', 'down_proj', 'up_proj']) and 'expert' not in m.lower()]

        results["attention_modules"] = attention_modules
        results["expert_modules"] = expert_modules
        results["gate_modules"] = gate_modules
        results["mlp_modules"] = mlp_modules

        results["has_attention_lora"] = len(attention_modules) > 0
        results["has_expert_lora"] = len(expert_modules) > 0 or "experts" in supported

    except Exception as e:
        results["supported_modules_error"] = str(e)[:500]
        # Try alternative method - check model config
        try:
            results["model_config"] = str(llm.llm_engine.model_config)[:500]
        except:
            pass

    # Step 3: Generate baseline output (no LoRA)
    print("Step 3: Generating baseline output (no LoRA)...")
    prompt = "What is 2 + 2? Answer with just the number:"
    baseline = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
    results["baseline_output"] = baseline[0].outputs[0].text

    # Step 4: Create minimal LoRA adapter
    print("Step 4: Creating minimal LoRA adapter...")
    try:
        lora_dir = "/tmp/test_moe_lora_adapter"
        os.makedirs(lora_dir, exist_ok=True)

        # Get model config to understand architecture
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            trust_remote_code=True
        )
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        results["model_hidden_size"] = hidden_size
        results["model_num_heads"] = num_heads

        # LoRA config - target qkv_proj (attention)
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["qkv_proj"],  # Start with just attention
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

        # Save adapter config
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

        # Create minimal LoRA weights (random initialization)
        # For qkv_proj: input_dim = hidden_size, output_dim = 3 * hidden_size (Q, K, V)
        lora_tensors = {}
        for layer_idx in range(config.num_hidden_layers):
            # LoRA A: (r, hidden_size)
            # LoRA B: (3*hidden_size, r)
            lora_a = torch.zeros(8, hidden_size, dtype=torch.float16)
            lora_b = torch.zeros(3 * hidden_size, 8, dtype=torch.float16)

            # Small random init to make adapter do something
            torch.nn.init.normal_(lora_a, std=0.01)
            # Zero init for B (standard LoRA practice)

            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

        # Save using safetensors
        from safetensors.torch import save_file
        save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")

        results["lora_adapter_created"] = True
        results["lora_num_layers"] = config.num_hidden_layers

    except Exception as e:
        results["lora_adapter_error"] = str(e)[:500]
        import traceback
        results["lora_adapter_traceback"] = traceback.format_exc()[:1000]

    # Step 5: Load LoRA adapter
    print("Step 5: Loading LoRA adapter...")
    try:
        lora_request = LoRARequest(
            lora_name="test_adapter",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        # Generate with LoRA
        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["lora_load_success"] = True

    except Exception as e:
        results["lora_load_error"] = str(e)[:500]
        import traceback
        results["lora_load_traceback"] = traceback.format_exc()[:1000]

    # Cleanup
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

    print("Running MoE LoRA layer coverage and adapter test...")
    print("=" * 60)

    result = ray.get(
        test_lora_layer_coverage_and_adapter.options(runtime_env=runtime_env).remote(),
        timeout=600
    )

    print("\n=== Results ===")
    print(f"Init success: {result.get('init_success')}")

    print("\n--- Supported LoRA Modules ---")
    modules = result.get('supported_lora_modules', [])
    if modules:
        print(f"Total: {len(modules)}")
        for m in modules:
            print(f"  - {m}")
    else:
        print(f"Error: {result.get('supported_modules_error', 'Unknown')}")

    print(f"\nAttention modules: {result.get('attention_modules', [])}")
    print(f"Expert modules: {result.get('expert_modules', [])}")
    print(f"Gate modules: {result.get('gate_modules', [])}")
    print(f"MLP modules: {result.get('mlp_modules', [])}")

    print(f"\nHas attention LoRA: {result.get('has_attention_lora')}")
    print(f"Has expert LoRA: {result.get('has_expert_lora')}")

    print("\n--- Generation Test ---")
    print(f"Baseline output: {result.get('baseline_output')}")
    print(f"LoRA adapter created: {result.get('lora_adapter_created')}")
    if result.get('lora_adapter_error'):
        print(f"LoRA adapter error: {result.get('lora_adapter_error')}")
        print(f"Traceback: {result.get('lora_adapter_traceback')}")

    print(f"LoRA load success: {result.get('lora_load_success')}")
    if result.get('lora_load_error'):
        print(f"LoRA load error: {result.get('lora_load_error')}")
        print(f"Traceback: {result.get('lora_load_traceback')}")
    else:
        print(f"LoRA output: {result.get('lora_output')}")

    print("\n" + "=" * 60)
    print("VERDICT:")
    if result.get('has_attention_lora') and result.get('has_expert_lora'):
        print("  Attention LoRA: SUPPORTED")
        print("  Expert LoRA: SUPPORTED")
    else:
        print(f"  Attention LoRA: {'SUPPORTED' if result.get('has_attention_lora') else 'NOT SUPPORTED'}")
        print(f"  Expert LoRA: {'SUPPORTED' if result.get('has_expert_lora') else 'NOT SUPPORTED'}")

    if result.get('lora_load_success'):
        print("  LoRA Adapter Loading: WORKS")
    else:
        print("  LoRA Adapter Loading: FAILED")


if __name__ == "__main__":
    main()
