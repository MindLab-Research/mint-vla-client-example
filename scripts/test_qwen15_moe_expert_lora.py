#!/usr/bin/env python3
"""Test LoRA on Qwen1.5-MoE-A2.7B-Chat including expert layers.

vLLM 0.13.0rc2 supported modules for Qwen2MoE:
- Attention: q_proj, k_proj, v_proj, o_proj (NOT fused qkv_proj)
- Shared expert: gate_proj, up_proj, down_proj
- MoE experts: experts.{i}.gate_proj, experts.{i}.up_proj, experts.{i}.down_proj
- Router: gate
"""

import os
import json
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_PATH = "/vePFS-Mindverse/share/modelscope/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"


def test_attention_lora():
    """Test attention LoRA with separate q/k/v projections."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file

    print("\n" + "=" * 60)
    print("TEST 1: Attention LoRA (q_proj, k_proj, v_proj)")
    print("=" * 60)

    # Model dimensions
    hidden_size = 2048
    num_heads = 16
    num_kv_heads = 16
    num_layers = 24
    head_dim = hidden_size // num_heads

    q_dim = num_heads * head_dim      # 2048
    kv_dim = num_kv_heads * head_dim  # 2048

    print(f"hidden_size: {hidden_size}")
    print(f"q_dim: {q_dim}, kv_dim: {kv_dim}")

    # Initialize vLLM
    print("\n[1] Initializing vLLM...")
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

    # Baseline generation
    print("\n[2] Testing baseline generation...")
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        print(f"    Output: {output[0].outputs[0].text}")
    except Exception as e:
        print(f"    FAILED: {e}")
        del llm
        return False

    # Create attention LoRA with separate q/k/v
    print("\n[3] Creating attention LoRA adapter...")
    lora_dir = "/tmp/test_qwen15_attn_lora"
    os.makedirs(lora_dir, exist_ok=True)

    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
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

    lora_tensors = {}
    for layer_idx in range(num_layers):
        # q_proj: [q_dim, hidden_size]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"] = \
            torch.zeros(q_dim, 8, dtype=torch.float16)

        # k_proj: [kv_dim, hidden_size]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight"] = \
            torch.zeros(kv_dim, 8, dtype=torch.float16)

        # v_proj: [kv_dim, hidden_size]
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"] = \
            torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
        lora_tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"] = \
            torch.zeros(kv_dim, 8, dtype=torch.float16)

    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    print(f"    Created {len(lora_tensors)} tensors")

    # Load attention LoRA
    print("\n[4] Loading attention LoRA...")
    try:
        lora_request = LoRARequest(
            lora_name="attn_qkv",
            lora_int_id=1,
            lora_path=lora_dir,
        )
        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        print(f"    LoRA output: {lora_output[0].outputs[0].text}")
        print("    ATTENTION LORA: SUCCESS")
        del llm
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        del llm
        return False


def test_expert_lora():
    """Test expert layer LoRA.

    vLLM pack_moe() requires ALL experts to have LoRA weights - partial not supported.
    Each expert needs: gate_proj (w1), down_proj (w2), up_proj (w3)
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from safetensors.torch import save_file

    print("\n" + "=" * 60)
    print("TEST 2: Expert Layer LoRA (ALL 60 experts)")
    print("=" * 60)

    # Model dimensions
    hidden_size = 2048
    num_layers = 24
    num_experts = 60
    moe_intermediate_size = 1408

    print(f"hidden_size: {hidden_size}")
    print(f"num_experts: {num_experts}")
    print(f"moe_intermediate_size: {moe_intermediate_size}")

    # Initialize vLLM
    print("\n[1] Initializing vLLM...")
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

    # Baseline generation
    print("\n[2] Testing baseline generation...")
    prompt = "2+2="
    try:
        output = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
        print(f"    Output: {output[0].outputs[0].text}")
    except Exception as e:
        print(f"    FAILED: {e}")
        del llm
        return False

    # Create expert LoRA for ALL experts
    print("\n[3] Creating expert LoRA adapter (ALL 60 experts)...")
    lora_dir = "/tmp/test_qwen15_expert_lora"
    os.makedirs(lora_dir, exist_ok=True)

    # Target ALL expert modules - vLLM pack_moe() requires complete set
    target_modules = []
    for e in range(num_experts):
        target_modules.extend([
            f"experts.{e}.gate_proj",
            f"experts.{e}.up_proj",
            f"experts.{e}.down_proj",
        ])

    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "r": 8,
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

    # Create LoRA weights for ALL experts in ALL layers
    lora_tensors = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):  # All 60 experts
            # gate_proj (w1): [moe_intermediate_size, hidden_size]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_A.weight"] = \
                torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_B.weight"] = \
                torch.zeros(moe_intermediate_size, 8, dtype=torch.float16)

            # up_proj (w3): [moe_intermediate_size, hidden_size]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_A.weight"] = \
                torch.randn(8, hidden_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_B.weight"] = \
                torch.zeros(moe_intermediate_size, 8, dtype=torch.float16)

            # down_proj (w2): [hidden_size, moe_intermediate_size]
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_A.weight"] = \
                torch.randn(8, moe_intermediate_size, dtype=torch.float16) * 0.01
            lora_tensors[f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_B.weight"] = \
                torch.zeros(hidden_size, 8, dtype=torch.float16)

    save_file(lora_tensors, f"{lora_dir}/adapter_model.safetensors")
    num_expected = num_layers * num_experts * 3 * 2  # 24 * 60 * 3 * 2 = 8640
    print(f"    Created {len(lora_tensors)} tensors (expected: {num_expected})")

    # Load expert LoRA
    print("\n[4] Loading expert LoRA...")
    try:
        lora_request = LoRARequest(
            lora_name="expert_lora",
            lora_int_id=1,
            lora_path=lora_dir,
        )
        lora_output = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0),
            lora_request=lora_request
        )
        print(f"    LoRA output: {lora_output[0].outputs[0].text}")
        print("    EXPERT LORA: SUCCESS")
        del llm
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()

        with open("/tmp/qwen15_expert_lora_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Type: {type(e).__name__}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        print("    Error saved to /tmp/qwen15_expert_lora_error.txt")
        del llm
        return False


def main():
    import torch
    print("=" * 60)
    print("Qwen1.5-MoE-A2.7B-Chat LoRA Test")
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
    results["attention_lora"] = test_attention_lora()
    results["expert_lora"] = test_expert_lora()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
