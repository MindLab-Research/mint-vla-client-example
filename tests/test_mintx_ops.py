from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tinker_server.backend.mintx_ops import (
    compute_teacher_log_probs_cpu,
    interpolate_checkpoints_to_dir,
    reverse_kl_from_teacher_log_probs,
)


def _write_checkpoint(path: Path, *, weight: float, include_rank_shards: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "a": torch.full((2, 2), weight, dtype=torch.float32),
            "b": torch.full((3,), weight + 1.0, dtype=torch.float16),
        },
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": path.name,
                "owner_id": "user-a",
                "model_id": "model-a",
                "model_name": "Qwen/Qwen3-4B",
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "backend": "megatron" if include_rank_shards else "peft",
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )
    if include_rank_shards:
        torch.save(
            {"adapter_state_dict": {"rank_a": torch.full((2, 1), weight, dtype=torch.float32)}},
            path / "mp_rank_00_adapter.pt",
        )
        torch.save(
            {"adapter_state_dict": {"rank_a": torch.full((2, 1), weight + 2.0, dtype=torch.float32)}},
            path / "mp_rank_01_adapter.pt",
        )


def test_interpolate_checkpoints_to_dir_interpolates_sampler_and_rank_shards(tmp_path: Path) -> None:
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    out = tmp_path / "out"
    _write_checkpoint(src1, weight=1.0, include_rank_shards=True)
    _write_checkpoint(src2, weight=3.0, include_rank_shards=True)

    artifacts = interpolate_checkpoints_to_dir(
        source_paths=[str(src1), str(src2)],
        coefficients=[0.75, 0.25],
        output_dir=str(out),
        checkpoint_name="mix",
        user_id="user-a",
    )

    merged = load_file(str(out / "adapter_model.safetensors"), device="cpu")
    assert torch.allclose(merged["a"], torch.full((2, 2), 1.5))
    assert torch.allclose(merged["b"].float(), torch.full((3,), 2.5))

    rank0 = torch.load(out / "mp_rank_00_adapter.pt", map_location="cpu")
    rank1 = torch.load(out / "mp_rank_01_adapter.pt", map_location="cpu")
    assert torch.allclose(rank0["adapter_state_dict"]["rank_a"], torch.full((2, 1), 1.5))
    assert torch.allclose(rank1["adapter_state_dict"]["rank_a"], torch.full((2, 1), 3.5))
    assert artifacts.has_rank_shards is True
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_type"] == "sampler"
    assert metadata["mintx_operation"] == "checkpoints.interpolate"


def test_reverse_kl_from_teacher_log_probs_matches_naive_formula() -> None:
    student_logits = torch.tensor([[0.2, -0.1, 1.0], [1.2, 0.4, -0.3]], dtype=torch.float32)
    teacher_logits = torch.tensor([[0.5, 0.0, 0.7], [0.8, -0.2, 0.3]], dtype=torch.float32)
    temperature = 1.7

    teacher_log_probs = compute_teacher_log_probs_cpu(
        teacher_logits,
        temperature=temperature,
        block_size=2,
    )
    actual = reverse_kl_from_teacher_log_probs(
        student_logits.clone(),
        teacher_log_probs,
        temperature=temperature,
        block_size=2,
    )

    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs_naive = torch.log_softmax(teacher_logits / temperature, dim=-1)
    expected = (
        torch.exp(student_log_probs)
        * (student_log_probs - teacher_log_probs_naive)
    ).sum(dim=-1) * (temperature * temperature)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
