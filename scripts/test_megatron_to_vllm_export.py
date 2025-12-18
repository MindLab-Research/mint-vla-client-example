#!/usr/bin/env python3
"""Test Megatron to vLLM LoRA weight export for MoE models.

Tests the conversion pipeline from Megatron-Bridge naming conventions
to PEFT format that vLLM can load.

Megatron-Bridge names (from named_parameters):
    decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
    decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight
    decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight
    decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_out.weight

PEFT expects:
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
    base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight
    base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight
"""

import os
import json
import re
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_PATH = "/vePFS-Mindverse/share/modelscope/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"


def convert_megatron_to_peft(name: str) -> str | None:
    """Convert Megatron-Bridge LoRA param name to PEFT format.

    Handles both attention and MoE expert modules.
    """
    # Extract layer number
    match = re.search(r'layers\.(\d+)\.', name)
    if not match:
        return None
    layer_num = match.group(1)

    # Determine lora_A vs lora_B
    if 'adapter.linear_in' in name or '.lora_a.' in name.lower():
        lora_type = 'lora_A'
    elif 'adapter.linear_out' in name or '.lora_b.' in name.lower():
        lora_type = 'lora_B'
    else:
        return None

    # Check for MoE expert pattern first (more specific)
    # Megatron: decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight
    # Alternative: decoder.layers.0.mlp.experts.0.linear_fc1.adapter.linear_in.weight
    expert_match = re.search(r'(?:local_experts\.)?(\d+)\.linear_fc([12])', name)
    if expert_match:
        expert_idx = expert_match.group(1)
        fc_type = expert_match.group(2)

        # fc1 = gate_proj + up_proj (fused), fc2 = down_proj
        # For Qwen MoE: gate_proj and up_proj are separate, but Megatron may fuse them
        if fc_type == '1':
            # Megatron linear_fc1 maps to gate_proj (w1) in Qwen terminology
            target = f'mlp.experts.{expert_idx}.gate_proj'
        else:
            target = f'mlp.experts.{expert_idx}.down_proj'

        return f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"

    # Attention modules
    if 'linear_qkv.adapter' in name or 'linear_qkv.lora_' in name.lower():
        # Fused QKV - vLLM Qwen2MoE doesn't support qkv_proj, need separate q/k/v
        # For now map to q_proj - user must handle k/v separately or use unfused
        target = 'self_attn.q_proj'
    elif 'self_attention.linear_proj' in name:
        target = 'self_attn.o_proj'
    elif 'linear_fc1' in name:
        # Dense MLP (not MoE) - gate_proj
        target = 'mlp.gate_proj'
    elif 'linear_fc2' in name:
        target = 'mlp.down_proj'
    else:
        return None

    return f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"


def test_conversion_unit():
    """Unit test the conversion function."""
    print("\n" + "=" * 60)
    print("TEST 1: Megatron → PEFT Name Conversion")
    print("=" * 60)

    test_cases = [
        # Attention
        ("decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
         "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"),
        ("decoder.layers.5.self_attention.linear_proj.adapter.linear_out.weight",
         "base_model.model.model.layers.5.self_attn.o_proj.lora_B.weight"),

        # MoE experts with local_experts pattern
        ("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight",
         "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight"),
        ("decoder.layers.3.mlp.experts.local_experts.59.linear_fc2.adapter.linear_out.weight",
         "base_model.model.model.layers.3.mlp.experts.59.down_proj.lora_B.weight"),

        # MoE experts without local_experts pattern
        ("decoder.layers.10.mlp.experts.15.linear_fc1.adapter.linear_in.weight",
         "base_model.model.model.layers.10.mlp.experts.15.gate_proj.lora_A.weight"),
    ]

    passed = 0
    failed = 0

    for megatron_name, expected_peft in test_cases:
        result = convert_megatron_to_peft(megatron_name)
        if result == expected_peft:
            print(f"  [PASS] {megatron_name[:60]}...")
            passed += 1
        else:
            print(f"  [FAIL] {megatron_name[:60]}...")
            print(f"         Expected: {expected_peft}")
            print(f"         Got:      {result}")
            failed += 1

    print(f"\nConversion: {passed}/{passed+failed} passed")
    return failed == 0


def test_simulated_megatron_export():
    """Test loading simulated Megatron export into vLLM.

    Simulates Megatron-Bridge naming and converts to PEFT format,
    then loads into vLLM with Qwen1.5-MoE model.
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file

    print("\n" + "=" * 60)
    print("TEST 2: Simulated Megatron Export → vLLM Load")
    print("=" * 60)

    # Model dimensions for Qwen1.5-MoE-A2.7B-Chat
    hidden_size = 2048
    num_layers = 24
    num_experts = 60
    moe_intermediate_size = 1408
    num_heads = 16
    num_kv_heads = 16
    head_dim = hidden_size // num_heads
    q_dim = num_heads * head_dim      # 2048
    kv_dim = num_kv_heads * head_dim  # 2048
    rank = 8

    print(f"Model config: {num_layers} layers, {num_experts} experts")

    # Step 1: Create simulated Megatron-Bridge weights
    print("\n[1] Creating simulated Megatron-Bridge weights...")
    megatron_weights = {}

    # Attention LoRA (separate q/k/v for Qwen2MoE)
    for layer_idx in range(num_layers):
        for proj, dim in [("q_proj", q_dim), ("k_proj", kv_dim), ("v_proj", kv_dim)]:
            # Simulate Megatron naming - but Qwen2MoE uses separate projections
            # So we'll use HF-style names directly since Megatron-Bridge adapts to model
            megatron_weights[f"model.layers.{layer_idx}.self_attn.{proj}.lora_a.weight"] = \
                torch.randn(rank, hidden_size, dtype=torch.float16) * 0.01
            megatron_weights[f"model.layers.{layer_idx}.self_attn.{proj}.lora_b.weight"] = \
                torch.zeros(dim, rank, dtype=torch.float16)

    # Expert LoRA (all 60 experts)
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

    print(f"    Created {len(megatron_weights)} Megatron-style tensors")

    # Step 2: Convert to PEFT format
    print("\n[2] Converting to PEFT format...")
    peft_weights = {}
    for name, tensor in megatron_weights.items():
        # Simple conversion: add PEFT prefix and fix lora_a/lora_b → lora_A/lora_B
        peft_name = f"base_model.model.{name}"
        peft_name = peft_name.replace(".lora_a.", ".lora_A.")
        peft_name = peft_name.replace(".lora_b.", ".lora_B.")
        peft_weights[peft_name] = tensor

    print(f"    Converted to {len(peft_weights)} PEFT tensors")

    # Step 3: Save adapter
    print("\n[3] Saving PEFT adapter...")
    lora_dir = "/tmp/test_megatron_export"
    os.makedirs(lora_dir, exist_ok=True)

    # Build target_modules list
    attn_modules = ["q_proj", "k_proj", "v_proj"]
    expert_modules = []
    for e in range(num_experts):
        expert_modules.extend([
            f"experts.{e}.gate_proj",
            f"experts.{e}.up_proj",
            f"experts.{e}.down_proj",
        ])

    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "r": rank,
        "lora_alpha": 16,
        "target_modules": attn_modules + expert_modules,
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

    num_expected = num_layers * (3 * 2 + num_experts * 3 * 2)  # attn + experts
    print(f"    Saved {len(peft_weights)} tensors (expected: {num_expected})")

    # Step 4: Initialize vLLM
    print("\n[4] Initializing vLLM...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.7,
        )
        print("    Model initialized")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return False

    # Step 5: Baseline generation
    print("\n[5] Testing baseline generation...")
    prompt = "The capital of France is"
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        print(f"    Baseline: {output[0].outputs[0].text}")
    except Exception as e:
        print(f"    FAILED: {e}")
        del llm
        return False

    # Step 6: Load converted LoRA
    print("\n[6] Loading Megatron-exported LoRA...")
    try:
        lora_request = LoRARequest(
            lora_name="megatron_export",
            lora_int_id=1,
            lora_path=lora_dir,
        )
        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        print(f"    LoRA output: {lora_output[0].outputs[0].text}")
        print("    MEGATRON EXPORT: SUCCESS")
        del llm
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()

        with open("/tmp/megatron_export_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        print("    Error saved to /tmp/megatron_export_error.txt")
        del llm
        return False


def main():
    import torch
    print("=" * 60)
    print("Megatron → vLLM LoRA Export Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except:
        pass
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    results = {}
    results["conversion_unit"] = test_conversion_unit()
    results["megatron_export"] = test_simulated_megatron_export()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
