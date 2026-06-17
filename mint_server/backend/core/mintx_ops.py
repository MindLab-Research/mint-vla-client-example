from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from mint_server.checkpoints.checkpoints import read_checkpoint_metadata, write_checkpoint_metadata
from mint_server.utils.model_input_utils import flatten_encoded_text_chunks


def _isoformat_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ReverseKLTensors:
    prefix_tokens: list[int]
    completion_tokens: list[int]
    weights: list[float]


def build_scoring_sequence(prefix_tokens: list[int], completion_tokens: list[int]) -> tuple[list[int], int]:
    if not prefix_tokens:
        raise ValueError("input context must contain at least one token")
    if not completion_tokens:
        raise ValueError("target_tokens must contain at least one token")
    input_tokens = list(prefix_tokens) + list(completion_tokens[:-1])
    completion_start = len(prefix_tokens) - 1
    return input_tokens, completion_start


@dataclass(frozen=True)
class InterpolationArtifacts:
    output_checkpoint_type: str
    backend: str | None
    model_id: str
    model_name: str | None
    adapter_config: dict[str, Any]
    source_paths: list[str]
    coefficients: list[float]
    has_rank_shards: bool


def parse_reverse_kl_item(item: dict, *, input_key: str) -> ReverseKLTensors:
    model_input = item.get(input_key, {})
    input_tokens = flatten_encoded_text_chunks(model_input)
    if not input_tokens:
        raise ValueError(f"{input_key} has no encoded_text tokens")

    target_raw = item.get("target_tokens", {})
    target_tokens = target_raw.get("data") if isinstance(target_raw, dict) else None
    if not isinstance(target_tokens, list) or not target_tokens:
        raise ValueError("target_tokens.data must be a non-empty list[int]")

    weights_raw = item.get("weights", {})
    weights = weights_raw.get("data") if isinstance(weights_raw, dict) else None
    if not isinstance(weights, list) or not weights:
        raise ValueError("weights.data must be a non-empty list[float]")

    if len(weights) != len(target_tokens):
        raise ValueError(
            f"weights length {len(weights)} != target_tokens length {len(target_tokens)}"
        )

    try:
        target_ints = [int(x) for x in target_tokens]
        weight_floats = [float(x) for x in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("target_tokens and weights must be numeric") from exc

    return ReverseKLTensors(
        prefix_tokens=[int(x) for x in input_tokens],
        completion_tokens=target_ints,
        weights=weight_floats,
    )


def compute_teacher_log_probs_cpu(
    logits: torch.Tensor,
    *,
    temperature: float,
    block_size: int,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape={tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    scaled = logits / float(temperature)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    rows, vocab = scaled.shape
    out = torch.empty((rows, vocab), dtype=torch.float32, device="cpu")
    for start in range(0, vocab, block_size):
        end = min(vocab, start + block_size)
        block = (scaled[:, start:end] - log_z).to(dtype=torch.float32)
        out[:, start:end].copy_(block.detach().cpu(), non_blocking=False)
        del block
    return out


def reverse_kl_from_teacher_log_probs(
    student_logits: torch.Tensor,
    teacher_log_probs_cpu: torch.Tensor,
    *,
    temperature: float,
    block_size: int,
) -> torch.Tensor:
    if student_logits.dim() != 2:
        raise ValueError(f"Expected 2D student logits, got shape={tuple(student_logits.shape)}")
    if teacher_log_probs_cpu.dim() != 2:
        raise ValueError(
            f"Expected 2D teacher log probs, got shape={tuple(teacher_log_probs_cpu.shape)}"
        )
    if student_logits.shape != teacher_log_probs_cpu.shape:
        raise ValueError(
            f"student_logits shape {tuple(student_logits.shape)} != teacher_log_probs shape {tuple(teacher_log_probs_cpu.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")

    scaled = student_logits / float(temperature)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    token_kl = torch.zeros(student_logits.shape[0], dtype=torch.float32, device=student_logits.device)
    vocab = scaled.shape[1]
    for start in range(0, vocab, block_size):
        end = min(vocab, start + block_size)
        student_block_logp = scaled[:, start:end] - log_z
        student_block_prob = student_block_logp.exp()
        teacher_block_logp = teacher_log_probs_cpu[:, start:end].to(
            device=student_logits.device,
            dtype=student_block_logp.dtype,
            non_blocking=False,
        )
        token_kl = token_kl + (
            student_block_prob * (student_block_logp - teacher_block_logp)
        ).sum(dim=-1).to(dtype=torch.float32)
        del student_block_logp, student_block_prob, teacher_block_logp
    token_kl = token_kl * float(temperature) * float(temperature)
    return token_kl


def _validate_source_metadata(source_paths: list[str]) -> tuple[str, str | None, str | None, dict[str, Any]]:
    if not source_paths:
        raise ValueError("source_paths must not be empty")

    first_meta = read_checkpoint_metadata(source_paths[0])
    model_id = first_meta.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"Missing model_id in metadata for {source_paths[0]!r}")
    model_name = first_meta.get("model_name") if isinstance(first_meta.get("model_name"), str) else None
    backend = first_meta.get("backend") if isinstance(first_meta.get("backend"), str) else None
    owner_id = first_meta.get("owner_id") if isinstance(first_meta.get("owner_id"), str) else None

    for source_path in source_paths[1:]:
        meta = read_checkpoint_metadata(source_path)
        if model_name is not None and meta.get("model_name") != model_name:
            raise ValueError("All source checkpoints must have the same model_name")
        if backend is not None and meta.get("backend") != backend:
            raise ValueError("All source checkpoints must have the same backend")
        other_owner_id = meta.get("owner_id") if isinstance(meta.get("owner_id"), str) else None
        if owner_id is not None and other_owner_id is not None and other_owner_id != owner_id:
            raise ValueError("All source checkpoints must have the same owner_id")

    return model_id, model_name, backend, first_meta


def _load_adapter_config(path: str) -> dict[str, Any]:
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"Missing adapter_config.json in checkpoint: {path}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _assert_same_keys_and_shapes(state_dicts: list[dict[str, torch.Tensor]], *, label: str) -> None:
    if not state_dicts:
        raise ValueError(f"No state_dicts for {label}")
    base_keys = set(state_dicts[0].keys())
    for idx, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != base_keys:
            missing = sorted(base_keys - set(state_dict.keys()))
            extra = sorted(set(state_dict.keys()) - base_keys)
            raise ValueError(
                f"{label} checkpoint {idx} keys mismatch: missing={missing[:5]} extra={extra[:5]}"
            )
    for key in sorted(base_keys):
        base_shape = tuple(state_dicts[0][key].shape)
        for idx, state_dict in enumerate(state_dicts[1:], start=1):
            if tuple(state_dict[key].shape) != base_shape:
                raise ValueError(
                    f"{label} tensor shape mismatch for {key!r}: {base_shape} vs {tuple(state_dict[key].shape)}"
                )


def _interpolate_tensor_list(tensors: list[torch.Tensor], coefficients: list[float]) -> torch.Tensor:
    if len(tensors) != len(coefficients):
        raise ValueError("tensor/coefficient length mismatch")
    acc = torch.zeros_like(tensors[0], dtype=torch.float32, device="cpu")
    out_dtype = tensors[0].dtype
    for tensor, coeff in zip(tensors, coefficients, strict=True):
        acc.add_(tensor.detach().to(device="cpu", dtype=torch.float32), alpha=float(coeff))
    return acc.to(dtype=out_dtype)


def _interpolate_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    coefficients: list[float],
) -> dict[str, torch.Tensor]:
    _assert_same_keys_and_shapes(state_dicts, label="adapter")
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(state_dicts[0].keys()):
        merged[key] = _interpolate_tensor_list([sd[key] for sd in state_dicts], coefficients)
    return merged


def _rank_adapter_files(path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(path, "mp_rank_*_adapter.pt")))


def _interpolate_rank_shards(
    source_paths: list[str],
    coefficients: list[float],
    output_dir: str,
) -> bool:
    source_rank_files = [_rank_adapter_files(path) for path in source_paths]
    has_any = any(files for files in source_rank_files)
    if not has_any:
        return False
    if not all(files for files in source_rank_files):
        raise ValueError("Either all or none of the source checkpoints must contain Megatron rank adapter shards")

    base_names = [tuple(os.path.basename(f) for f in files) for files in source_rank_files]
    if any(names != base_names[0] for names in base_names[1:]):
        raise ValueError("Megatron rank shard layouts differ across source checkpoints")

    for shard_idx, shard_name in enumerate(base_names[0]):
        shard_dicts: list[dict[str, torch.Tensor]] = []
        for source_idx, source_path in enumerate(source_paths):
            raw = torch.load(source_rank_files[source_idx][shard_idx], map_location="cpu")
            adapter_state = raw.get("adapter_state_dict")
            if not isinstance(adapter_state, dict):
                raise ValueError(
                    f"Megatron rank shard missing adapter_state_dict: {source_rank_files[source_idx][shard_idx]}"
                )
            shard_dicts.append(adapter_state)
        merged = _interpolate_state_dicts(shard_dicts, coefficients)
        torch.save({"adapter_state_dict": merged}, os.path.join(output_dir, shard_name))
    return True


def interpolate_checkpoints_to_dir(
    *,
    source_paths: list[str],
    coefficients: list[float],
    output_dir: str,
    checkpoint_name: str,
    user_id: str | None,
    output_checkpoint_type: str = "sampler",
) -> InterpolationArtifacts:
    if output_checkpoint_type != "sampler":
        raise ValueError(
            f"output_checkpoint_type={output_checkpoint_type!r} is not supported; only 'sampler' is allowed"
        )
    if len(source_paths) < 2:
        raise ValueError("At least two source checkpoints are required for interpolation")
    if len(source_paths) != len(coefficients):
        raise ValueError("source_paths and coefficients must have the same length")
    if not all(math.isfinite(float(c)) for c in coefficients):
        raise ValueError("coefficients must be finite")

    model_id, model_name, backend, first_meta = _validate_source_metadata(source_paths)
    adapter_configs = [_load_adapter_config(path) for path in source_paths]
    if any(cfg != adapter_configs[0] for cfg in adapter_configs[1:]):
        raise ValueError("All source checkpoints must have identical adapter_config.json")

    if os.path.exists(output_dir):
        raise ValueError(f"Output checkpoint already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=False)

    state_dicts = [load_file(os.path.join(path, "adapter_model.safetensors"), device="cpu") for path in source_paths]
    merged_state = _interpolate_state_dicts(state_dicts, coefficients)
    save_file(merged_state, os.path.join(output_dir, "adapter_model.safetensors"))

    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_configs[0], f, indent=2, sort_keys=True)

    has_rank_shards = _interpolate_rank_shards(source_paths, coefficients, output_dir)

    training_meta = {
        "current_step": None,
        "learning_rate": None,
        "mintx_interpolation": {
            "source_paths": list(source_paths),
            "coefficients": [float(c) for c in coefficients],
        },
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2, sort_keys=True)

    metadata = {
        "checkpoint_id": checkpoint_name,
        "owner_id": user_id,
        "model_id": model_id,
        "model_name": model_name,
        "created_at": _isoformat_utc_now(),
        "step": None,
        "checkpoint_type": "sampler",
        "optimizer_present": False,
        "backend": backend,
        "type": "sampler",
        "storage_tier": "persistent_cache",
        "mintx_operation": "checkpoints.interpolate",
        "mintx_sources": list(source_paths),
        "mintx_coefficients": [float(c) for c in coefficients],
        "mintx_has_rank_shards": bool(has_rank_shards),
        "mintx_parent_metadata": {
            "model_id": model_id,
            "backend": backend,
            "owner_id": first_meta.get("owner_id"),
        },
    }
    write_checkpoint_metadata(output_dir, metadata)

    return InterpolationArtifacts(
        output_checkpoint_type="sampler",
        backend=backend,
        model_id=model_id,
        model_name=model_name,
        adapter_config=adapter_configs[0],
        source_paths=list(source_paths),
        coefficients=[float(c) for c in coefficients],
        has_rank_shards=has_rank_shards,
    )


class _VocabParallelCrossEntropyAgainstLogQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as mpu

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized = vocab_parallel_logits - logits_max
        exp_logits = normalized.exp()
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
        softmax = exp_logits / sum_exp
        local_ce = (softmax * (-teacher_log_probs)).sum(dim=-1, keepdim=True)
        dist.all_reduce(local_ce, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
        ctx.save_for_backward(softmax, teacher_log_probs, local_ce)
        return local_ce.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        softmax, teacher_log_probs, local_ce = ctx.saved_tensors
        ce = local_ce.squeeze(dim=-1).unsqueeze(dim=-1)
        cost = -teacher_log_probs
        grad = softmax * (cost - ce)
        grad = grad * grad_output.unsqueeze(dim=-1)
        return grad, None


def vocab_parallel_cross_entropy_against_log_q(
    vocab_parallel_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return _VocabParallelCrossEntropyAgainstLogQ.apply(vocab_parallel_logits, teacher_log_probs)


class _VocabParallelReverseKLAgainstLogQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as mpu

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized = vocab_parallel_logits - logits_max
        exp_logits = normalized.exp()
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
        softmax = exp_logits / sum_exp
        student_log_probs = normalized - sum_exp.log()

        entropy = -(softmax * student_log_probs).sum(dim=-1)
        dist.all_reduce(entropy, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

        cross_entropy = -(softmax * teacher_log_probs).sum(dim=-1)
        dist.all_reduce(cross_entropy, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

        token_kl = cross_entropy - entropy
        ctx.save_for_backward(softmax, student_log_probs, teacher_log_probs, token_kl)
        return token_kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        softmax, student_log_probs, teacher_log_probs, token_kl = ctx.saved_tensors
        grad = softmax * (student_log_probs - teacher_log_probs - token_kl.unsqueeze(dim=-1))
        grad = grad * grad_output.unsqueeze(dim=-1)
        return grad, None


def vocab_parallel_reverse_kl_against_log_q(
    vocab_parallel_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return _VocabParallelReverseKLAgainstLogQ.apply(vocab_parallel_logits, teacher_log_probs)


def vocab_parallel_log_probs_from_logits_no_grad(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as mpu

    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
    normalized = vocab_parallel_logits - logits_max
    exp_logits = normalized.exp()
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
    return normalized - sum_exp.log()


@contextlib.contextmanager
def temporary_adapter_snapshot_dir(prefix: str) -> Any:
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
