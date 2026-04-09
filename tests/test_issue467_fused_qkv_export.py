from types import SimpleNamespace

import pytest
import torch

from tinker_server.backend.megatron_distributed import MegatronRankWorker


def test_issue467_standard_attention_qkv_sizes() -> None:
    cfg = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
    )
    assert MegatronRankWorker._get_standard_attention_qkv_sizes(cfg) == (4096, 512, 5120)


def test_issue467_split_fused_qkv_lora_a_duplicates_tensor() -> None:
    a = torch.arange(12, dtype=torch.float32).view(3, 4)
    out = MegatronRankWorker._split_standard_attention_qkv_peft_entries(
        peft_name="base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        tensor=a,
        lora_type="lora_A",
        q_size=4,
        kv_size=2,
    )
    assert sorted(out) == [
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
    ]
    assert torch.equal(out["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"], a)
    assert torch.equal(out["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"], a)
    assert torch.equal(out["base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"], a)
    assert out["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] is not a


def test_issue467_split_fused_qkv_lora_b_slices_q_k_v() -> None:
    b = torch.arange(16, dtype=torch.float32).view(8, 2)
    out = MegatronRankWorker._split_standard_attention_qkv_peft_entries(
        peft_name="base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        tensor=b,
        lora_type="lora_B",
        q_size=4,
        kv_size=2,
    )
    assert torch.equal(
        out["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
        b[:4, :],
    )
    assert torch.equal(
        out["base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"],
        b[4:6, :],
    )
    assert torch.equal(
        out["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"],
        b[6:8, :],
    )


def test_issue467_split_fused_qkv_lora_b_shape_mismatch_raises() -> None:
    b = torch.zeros(7, 2)
    with pytest.raises(ValueError, match="Fused QKV lora_B shape mismatch"):
        MegatronRankWorker._split_standard_attention_qkv_peft_entries(
            peft_name="base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            tensor=b,
            lora_type="lora_B",
            q_size=4,
            kv_size=2,
        )
