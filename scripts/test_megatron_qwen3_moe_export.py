#!/usr/bin/env python3
"""Integration test: Megatron -> vLLM MoE LoRA export for Qwen3-30B-A3B.

Tests the full pipeline:
1. Simulate Megatron-Bridge weight export (using HF naming convention)
2. Convert to PEFT format
3. Save as adapter
4. Load into vLLM on Ray cluster (TP=4)

Qwen3-30B-A3B config:
- 48 layers, 128 experts
- hidden_size=2048, moe_intermediate_size=768
- Separate q_proj, k_proj, v_proj (NOT fused qkv_proj)
"""

import ray
import os
import json
import re
import traceback

MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LOCAL_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def convert_megatron_to_peft(name: str) -> str | None:
    """Convert Megatron/HF LoRA param name to PEFT format.

    Handles both attention (separate q/k/v) and MoE expert modules.

    Megatron-Bridge may output:
        decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
        decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight

    Or HF-style (from verl bridge):
        model.layers.0.self_attn.q_proj.lora_a.weight
        model.layers.0.mlp.experts.0.gate_proj.lora_a.weight

    PEFT expects:
        base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight
    """
    # Extract layer number
    match = re.search(r'layers\.(\d+)\.', name)
    if not match:
        return None
    layer_num = match.group(1)

    # Determine lora_A vs lora_B
    name_lower = name.lower()
    if 'adapter.linear_in' in name or '.lora_a.' in name_lower:
        lora_type = 'lora_A'
    elif 'adapter.linear_out' in name or '.lora_b.' in name_lower:
        lora_type = 'lora_B'
    else:
        return None

    # Check for MoE expert pattern first (more specific)
    # Pattern: experts.{idx}.{proj} or local_experts.{idx}.{proj}
    expert_match = re.search(r'(?:local_experts\.)?(\d+)\.(gate_proj|up_proj|down_proj|linear_fc1|linear_fc2)', name)
    if expert_match:
        expert_idx = expert_match.group(1)
        proj_type = expert_match.group(2)

        # Map Megatron naming to HF naming
        if proj_type == 'linear_fc1':
            proj_type = 'gate_proj'  # Megatron fc1 -> gate_proj
        elif proj_type == 'linear_fc2':
            proj_type = 'down_proj'  # Megatron fc2 -> down_proj

        return f"base_model.model.model.layers.{layer_num}.mlp.experts.{expert_idx}.{proj_type}.{lora_type}.weight"

    # Attention modules (separate q/k/v for Qwen3)
    if 'q_proj' in name or ('linear_qkv' in name and 'q_proj' in name):
        target = 'self_attn.q_proj'
    elif 'k_proj' in name:
        target = 'self_attn.k_proj'
    elif 'v_proj' in name:
        target = 'self_attn.v_proj'
    elif 'o_proj' in name or 'linear_proj' in name:
        target = 'self_attn.o_proj'
    elif 'gate_proj' in name and 'expert' not in name_lower:
        # Shared expert or dense MLP
        target = 'mlp.gate_proj'
    elif 'up_proj' in name and 'expert' not in name_lower:
        target = 'mlp.up_proj'
    elif 'down_proj' in name and 'expert' not in name_lower:
        target = 'mlp.down_proj'
    else:
        return None

    return f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"


@ray.remote(num_gpus=4)
def test_megatron_export_to_vllm():
    """Test full Megatron -> vLLM export pipeline for Qwen3-30B-A3B."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file

    results = {
        "test_name": "megatron_qwen3_30b_moe_export",
    }

    # Get vLLM version
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
    except:
        results["vllm_version"] = "unknown"

    # Qwen3-30B-A3B model dimensions
    hidden_size = 2048
    num_layers = 48
    num_experts = 128
    moe_intermediate_size = 768
    num_heads = 32
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    q_dim = num_heads * head_dim      # 2048
    kv_dim = num_kv_heads * head_dim  # 256
    rank = 8

    results["model_config"] = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "moe_intermediate_size": moe_intermediate_size,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
    }
    print(f"Model config: {results['model_config']}", flush=True)

    # Step 1: Simulate Megatron weight export (using HF naming as verl bridge does)
    print("\n[1] Simulating Megatron weight export...", flush=True)
    megatron_weights = {}

    # Attention LoRA (separate q/k/v for Qwen3)
    for layer_idx in range(num_layers):
        for proj, dim in [("q_proj", q_dim), ("k_proj", kv_dim), ("v_proj", kv_dim)]:
            # Simulate HF-style names from verl bridge
            megatron_weights[f"model.layers.{layer_idx}.self_attn.{proj}.lora_a.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            megatron_weights[f"model.layers.{layer_idx}.self_attn.{proj}.lora_b.weight"] = \
                torch.zeros(dim, rank, dtype=torch.float16)

    # Expert LoRA (all 128 experts)
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            # gate_proj (w1): [moe_intermediate_size, hidden_size]
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_a.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_b.weight"] = \
                torch.zeros(moe_intermediate_size, rank, dtype=torch.float16)

            # up_proj (w3): [moe_intermediate_size, hidden_size]
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_a.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_b.weight"] = \
                torch.zeros(moe_intermediate_size, rank, dtype=torch.float16)

            # down_proj (w2): [hidden_size, moe_intermediate_size]
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_a.weight"] = \
                torch.randn(rank, moe_intermediate_size, dtype=torch.float16) * 0.01
            megatron_weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_b.weight"] = \
                torch.zeros(hidden_size, rank, dtype=torch.float16)

    results["megatron_tensor_count"] = len(megatron_weights)
    print(f"    Created {len(megatron_weights)} Megatron-style tensors", flush=True)

    # Step 2: Convert to PEFT format
    print("\n[2] Converting to PEFT format...", flush=True)
    peft_weights = {}
    conversion_failures = []

    for name, tensor in megatron_weights.items():
        peft_name = convert_megatron_to_peft(name)
        if peft_name:
            peft_weights[peft_name] = tensor
        else:
            conversion_failures.append(name)

    results["peft_tensor_count"] = len(peft_weights)
    results["conversion_failures"] = len(conversion_failures)
    print(f"    Converted to {len(peft_weights)} PEFT tensors", flush=True)
    if conversion_failures:
        print(f"    WARNING: {len(conversion_failures)} conversion failures", flush=True)
        results["failed_conversions"] = conversion_failures[:5]  # Sample

    # Step 3: Save PEFT adapter
    print("\n[3] Saving PEFT adapter...", flush=True)
    lora_dir = "/tmp/test_megatron_qwen3_30b_export"
    os.makedirs(lora_dir, exist_ok=True)

    # Build target_modules list
    target_modules = ["q_proj", "k_proj", "v_proj"]
    for e in range(num_experts):
        target_modules.extend([
            f"experts.{e}.gate_proj",
            f"experts.{e}.up_proj",
            f"experts.{e}.down_proj",
        ])

    adapter_config = {
        "base_model_name_or_path": MODEL_PATH,
        "r": rank,
        "lora_alpha": 16,
        "target_modules": target_modules,
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

    save_file(peft_weights, f"{lora_dir}/adapter_model.safetensors")
    print(f"    Saved adapter to {lora_dir}", flush=True)

    # Step 4: Initialize vLLM
    print("\n[4] Initializing vLLM with enable_lora=True, TP=4...", flush=True)
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

    # Step 5: Baseline generation
    print("\n[5] Testing baseline generation...", flush=True)
    prompt = "The capital of France is"
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
        results["baseline_output"] = output[0].outputs[0].text
        print(f"    Baseline: {results['baseline_output']}", flush=True)
    except Exception as e:
        results["baseline_error"] = str(e)
        del llm
        return results

    # Step 6: Load Megatron-exported LoRA
    print("\n[6] Loading Megatron-exported LoRA adapter...", flush=True)
    try:
        lora_request = LoRARequest(
            lora_name="megatron_qwen3_30b",
            lora_int_id=1,
            lora_path=lora_dir,
        )

        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=20, temperature=0),
            lora_request=lora_request
        )
        results["lora_output"] = lora_output[0].outputs[0].text
        results["lora_success"] = True
        print(f"    LoRA output: {results['lora_output']}", flush=True)
        print("    MEGATRON -> VLLM EXPORT: SUCCESS!", flush=True)

    except Exception as e:
        results["lora_error"] = str(e)
        results["lora_error_type"] = type(e).__name__
        results["lora_traceback"] = traceback.format_exc()
        results["lora_success"] = False
        print(f"    LORA LOAD FAILED: {e}", flush=True)

        with open("/tmp/megatron_qwen3_30b_export_error.txt", "w") as f:
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
    print("Megatron -> vLLM MoE LoRA Export Integration Test")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print("Testing: Attention + Expert LoRA (48 layers, 128 experts)")
    print()

    try:
        result = ray.get(
            test_megatron_export_to_vllm.options(runtime_env=runtime_env).remote(),
            timeout=1200  # 20 min timeout
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

    # Summary
    print("\nSUMMARY:")
    print(f"  Megatron tensors: {result.get('megatron_tensor_count', 'N/A')}")
    print(f"  PEFT tensors: {result.get('peft_tensor_count', 'N/A')}")
    print(f"  Conversion failures: {result.get('conversion_failures', 'N/A')}")

    if result.get("lora_success"):
        print("\n  INTEGRATION TEST: PASSED")
        print("  Megatron -> vLLM MoE LoRA export works for Qwen3-30B-A3B")
    else:
        print("\n  INTEGRATION TEST: FAILED")
        if "lora_error" in result:
            print(f"  Error: {result['lora_error']}")


if __name__ == "__main__":
    main()
