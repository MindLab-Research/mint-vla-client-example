#!/usr/bin/env python3
"""Minimal test to capture complete LoRA error."""
import ray
import os
import json
import sys

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"

@ray.remote(num_gpus=4)
def test_lora_minimal():
    """Minimal LoRA test with full error capture."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import traceback

    # Initialize
    print("Init model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,
    )
    
    # Get config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        trust_remote_code=True
    )
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    qkv_output_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
    
    # Create adapter
    lora_dir = "/tmp/test_minimal_lora"
    os.makedirs(lora_dir, exist_ok=True)
    
    adapter_config = {
        "base_model_name_or_path": MODEL_PATH,
        "r": 8, "lora_alpha": 16,
        "target_modules": ["qkv_proj"],
        "lora_dropout": 0.0, "bias": "none",
        "inference_mode": True, "peft_type": "LORA", "task_type": "CAUSAL_LM"
    }
    with open(f"{lora_dir}/adapter_config.json", "w") as f:
        json.dump(adapter_config, f)

    lora_tensors = {}
    for i in range(num_layers):
        lora_tensors[f"base_model.model.model.layers.{i}.self_attn.qkv_proj.lora_A.weight"] = torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{i}.self_attn.qkv_proj.lora_B.weight"] = torch.zeros(qkv_output_dim, 8, dtype=torch.float16)
    
    from safetensors.torch import save_file
    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    
    # Test baseline
    print("Baseline...")
    prompt = "2+2="
    baseline = llm.generate([prompt], SamplingParams(max_tokens=5, temperature=0))
    print(f"Baseline: {baseline[0].outputs[0].text}")
    
    # Test LoRA - capture full error
    print("Loading LoRA...")
    try:
        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=5, temperature=0),
            lora_request=LoRARequest("test", 1, lora_dir)
        )
        return {"success": True, "output": lora_output[0].outputs[0].text}
    except Exception as e:
        tb = traceback.format_exc()
        # Write to file for complete capture
        with open("/tmp/lora_error_full.txt", "w") as f:
            f.write(f"Error type: {type(e).__name__}\n")
            f.write(f"Error message: {str(e)}\n\n")
            f.write(f"Full traceback:\n{tb}")
        return {"success": False, "error": str(e), "traceback": tb}

def main():
    ray.init(address="auto", ignore_reinit_error=True)
    result = ray.get(test_lora_minimal.options(
        runtime_env={"env_vars": {"HF_HUB_OFFLINE": "1", "HF_HOME": "/vePFS-Mindverse/share/huggingface"}}
    ).remote(), timeout=600)
    print(f"\n=== Result ===\nSuccess: {result.get('success')}")
    if not result.get('success'):
        print(f"Error: {result.get('error')}")
        print(f"\nFull traceback:\n{result.get('traceback')}")

if __name__ == "__main__":
    main()
