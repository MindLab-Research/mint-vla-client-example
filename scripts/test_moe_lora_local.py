#!/usr/bin/env python3
"""Minimal MoE LoRA test on local GPUs (no Ray cluster).

Runs directly on volcano's 2x A800 GPUs with TP=2.
"""

import os
import json
import traceback

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"

MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def main():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("=" * 60)
    print("Minimal MoE LoRA Test (Local, TP=2)")
    print("=" * 60)

    # Check GPUs
    print(f"\nAvailable GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

    # Initialize model
    print("\n[1] Initializing vLLM with enable_lora=True...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,  # Use both local GPUs
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )
        print("    Model initialized")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return

    # Test baseline generation
    print("\n[2] Testing baseline generation (no LoRA)...")
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        print(f"    Output: {output[0].outputs[0].text}")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return

    # Get model config
    print("\n[3] Getting model config...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    head_dim = hidden_size // num_heads

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_output_dim = q_dim + 2 * kv_dim

    print(f"    hidden_size: {hidden_size}")
    print(f"    num_layers: {num_layers}")
    print(f"    num_experts: {config.num_experts}")
    print(f"    qkv_output_dim: {qkv_output_dim}")

    # Create LoRA adapter
    print("\n[4] Creating LoRA adapter...")
    lora_dir = "/tmp/test_local_moe_lora"
    os.makedirs(lora_dir, exist_ok=True)

    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
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
        lora_a = torch.zeros(8, hidden_size, dtype=torch.float16)
        lora_b = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)
        torch.nn.init.normal_(lora_a, std=0.01)

        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_A.weight"] = lora_a
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.qkv_proj.lora_B.weight"] = lora_b

    from safetensors.torch import save_file
    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    print(f"    Created adapter with {len(lora_tensors)} tensors")

    # Load LoRA adapter
    print("\n[5] Loading LoRA adapter...")
    try:
        lora_request = LoRARequest(
            lora_name="test_local",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        print(f"    LoRA output: {lora_output[0].outputs[0].text}")
        print("\n" + "=" * 60)
        print("SUCCESS: LoRA loading works!")
        print("=" * 60)

    except Exception as e:
        print(f"    FAILED: {e}")
        print("\n" + "=" * 60)
        print("FAILURE: LoRA loading failed")
        print("=" * 60)
        print("\nFull traceback:")
        traceback.print_exc()

        # Write error to file for analysis
        with open("/tmp/moe_lora_local_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        print("\nError details saved to /tmp/moe_lora_local_error.txt")

    del llm


if __name__ == "__main__":
    main()
