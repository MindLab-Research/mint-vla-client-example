import json

import pytest
import torch
from safetensors.torch import save_file

from tinker_server.backend.lora_utils import validate_peft_adapter_checkpoint_shapes


def _write_base_model(tmp_path, *, q_shape=(4, 2), k_shape=(1, 2), v_shape=(1, 2), o_shape=(2, 4)):
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(q_shape),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(k_shape),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros(v_shape),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(o_shape),
        },
        str(base_dir / "model.safetensors"),
    )
    return base_dir


def _write_adapter(tmp_path, tensors, *, rank=3):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": rank * 2,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    return adapter_dir


def test_issue_467_validate_peft_adapter_checkpoint_shapes_accepts_matching_qkv_shapes(tmp_path):
    base_dir = _write_base_model(tmp_path)
    adapter_dir = _write_adapter(
        tmp_path,
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((3, 2)),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((4, 3)),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.zeros((3, 2)),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.zeros((1, 3)),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.zeros((3, 2)),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.zeros((1, 3)),
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.zeros((3, 4)),
            "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.zeros((2, 3)),
        },
    )

    validate_peft_adapter_checkpoint_shapes(str(adapter_dir), str(base_dir))


def test_issue_467_validate_peft_adapter_checkpoint_shapes_rejects_fused_qkv_disguised_as_q_proj(tmp_path):
    base_dir = _write_base_model(tmp_path)
    adapter_dir = _write_adapter(
        tmp_path,
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((3, 2)),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((6, 3)),
        },
    )

    with pytest.raises(ValueError, match="lora_B output dim mismatch"):
        validate_peft_adapter_checkpoint_shapes(str(adapter_dir), str(base_dir))
