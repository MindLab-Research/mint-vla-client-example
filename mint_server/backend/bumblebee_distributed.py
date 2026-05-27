"""Bumblebee-backed distributed MoE LoRA training actors.

The production path is a resident Ray worker group: MinT sends serialized data
through Ray calls, and each rank owns a Bumblebee runtime handle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray

from mint_server.backend.megatron_distributed import (
    DistributedConfig,
    _bundle_node_ip,
    _get_or_create_megatron_placement_group,
    _node_affinity_resources,
    get_node_ip_and_free_port,
)
from mint_server.backend.model_registry import is_topology_desired_model
from mint_server.backend.node_placement import (
    assert_node_ip_capacity,
    parse_model_gpu_placement,
)
from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE, actor_runtime_env_vars, config as server_config, otel_env_vars
from mint_server.logging_context import get_current_traceparent, get_request_id
from mint_server.ray_utils import init_ray

from . import ray_kill

logger = logging.getLogger(__name__)

PERSISTENT_NAMESPACE = RAY_NAMESPACE
BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT = "mint_bumblebee_adapter_train_state_checkpoint_v1"
BUMBLEBEE_TRAIN_STATE_FILE = "adapter_train_state.pt"
BUMBLEBEE_TRAIN_STATE_META_FILE = "training_meta.json"

_bumblebee_create_locks: dict[str, threading.Lock] = {}
_bumblebee_create_locks_guard = threading.Lock()


def _get_bumblebee_create_lock(actor_name: str) -> threading.Lock:
    with _bumblebee_create_locks_guard:
        lock = _bumblebee_create_locks.get(actor_name)
        if lock is None:
            lock = threading.Lock()
            _bumblebee_create_locks[actor_name] = lock
        return lock


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected int", name, raw)
        return default


def _model_key_from_base_model(base_model: str) -> str:
    import re

    match = re.search(r"models--([^/]+)--([^/]+)/snapshots", str(base_model))
    if match:
        org, model = match.groups()
        return f"{org}/{model}"
    return str(base_model)


def _make_bumblebee_actor_name(base_model: str) -> str:
    import re

    match = re.search(r"models--([^/]+)--([^/]+)/snapshots", str(base_model))
    model_name = match.group(2) if match else str(base_model).split("/")[-1]
    model_name = model_name.lower().replace("-", "_").replace(".", "_")
    return f"mint_bumblebee_{model_name}"


def _make_bumblebee_pg_name(base_model: str) -> str:
    return f"{_make_bumblebee_actor_name(base_model)}_pg"


def _is_qwen3_235b_model(model: str | None) -> bool:
    return "qwen3-235b-a22b" in str(model or "").lower()


def _bumblebee_runtime_etp(base_model: str, config: DistributedConfig) -> int | None:
    etp = config.expert_tensor_parallel_size
    if etp is not None:
        return int(etp)
    if _is_qwen3_235b_model(_model_key_from_base_model(base_model)):
        return 1
    return None


def _bumblebee_repo_path() -> str:
    configured = os.environ.get("MINT_BUMBLEBEE_REPO_PATH")
    if configured:
        return configured
    for candidate in (
        "/vePFS-Mindverse/user/nolanho/code/bumblebee",
        "/root/code/bumblebee",
    ):
        if (Path(candidate) / "bumblebee" / "__init__.py").exists():
            return candidate
    return "/root/code/bumblebee"


def _ensure_bumblebee_repo_importable() -> str:
    repo = str(Path(_bumblebee_repo_path()).expanduser())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if repo not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{repo}:{pythonpath}" if pythonpath else repo
    return repo


def _target_modules(*, train_attn: bool, train_mlp: bool, train_unembed: bool) -> list[str]:
    targets: list[str] = []
    if train_attn:
        targets.extend(["linear_qkv", "linear_proj"])
    if train_mlp:
        targets.extend(["linear_fc1", "linear_fc2"])
    if train_unembed:
        logger.warning("Bumblebee Qwen3-MoE lite does not support train_unembed; ignoring")
    if not targets:
        raise ValueError("Bumblebee LoRA requires train_attn or train_mlp to be enabled")
    return targets


def _coerce_scalar(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        elif hasattr(value, "item"):
            value = value.item()
        return int(value)
    except Exception:
        return int(default)


def _record_benchmark_logprobs_enabled() -> bool:
    return os.environ.get("MINT_BENCH_RECORD_LOGPROBS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _record_benchmark_logits_enabled() -> bool:
    return os.environ.get("MINT_BENCH_RECORD_LOGITS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _record_benchmark_inputs_enabled() -> bool:
    return os.environ.get("MINT_BENCH_RECORD_INPUTS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _record_benchmark_model_state_enabled() -> bool:
    return os.environ.get("MINT_BENCH_RECORD_MODEL_STATE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _json_debug_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_debug_value(v) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_json_debug_value(v) for v in list(value)[:64]]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _tensor_debug_signature(name: str, tensor: Any) -> dict[str, Any]:
    import torch

    detached = tensor.detach()
    stats = detached.float()
    flat = stats.reshape(-1)
    sample_flat = flat[: min(4096, flat.numel())]
    if sample_flat.numel() == 0:
        sample = []
        tensor_sum = tensor_abs_mean = tensor_norm = 0.0
    else:
        sample = sample_flat[: min(8, sample_flat.numel())].cpu().tolist()
        tensor_sum = float(sample_flat.sum().detach().cpu().item())
        tensor_abs_mean = float(sample_flat.abs().mean().detach().cpu().item())
        tensor_norm = float(torch.linalg.vector_norm(sample_flat).detach().cpu().item())
    return {
        "name": str(name),
        "shape": [int(dim) for dim in getattr(detached, "shape", ())],
        "dtype": str(getattr(detached, "dtype", "unknown")).replace("torch.", ""),
        "device": str(getattr(detached, "device", "unknown")),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "numel": int(detached.numel()),
        "signature_numel": int(sample_flat.numel()),
        "sample_sum_fp32": tensor_sum,
        "sample_abs_mean_fp32": tensor_abs_mean,
        "sample_norm_fp32": tensor_norm,
        "sample_fp32": [float(value) for value in sample],
    }


def _select_debug_parameter_signatures(
    named_parameters: Any,
    *,
    max_base: int = 24,
    max_adapter: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_signatures: list[dict[str, Any]] = []
    adapter_signatures: list[dict[str, Any]] = []
    counts = {
        "param_tensors": 0,
        "trainable_tensors": 0,
        "adapter_tensors": 0,
        "base_tensors": 0,
        "trainable_names": [],
    }
    base_patterns = (
        "embedding",
        "embed",
        "qkv",
        "proj",
        "linear_qkv",
        "linear_proj",
        "fc1",
        "fc2",
        "linear_fc1",
        "linear_fc2",
        "gate",
        "router",
        "norm",
        "output",
        "head",
    )
    seen_base_patterns: set[str] = set()
    for name, param in named_parameters:
        lname = str(name).lower()
        counts["param_tensors"] += 1
        if bool(getattr(param, "requires_grad", False)):
            counts["trainable_tensors"] += 1
            if len(counts["trainable_names"]) < 64:
                counts["trainable_names"].append(str(name))
        is_adapter = "lora" in lname or "adapter" in lname
        if is_adapter:
            counts["adapter_tensors"] += 1
            if len(adapter_signatures) < max_adapter:
                adapter_signatures.append(_tensor_debug_signature(str(name), param))
            continue
        counts["base_tensors"] += 1
        if len(base_signatures) >= max_base:
            continue
        matched = next((pattern for pattern in base_patterns if pattern in lname), None)
        if matched is None:
            continue
        # Keep coverage broad before adding repeated examples.
        if matched in seen_base_patterns and len(base_signatures) >= len(base_patterns):
            continue
        seen_base_patterns.add(matched)
        base_signatures.append(_tensor_debug_signature(str(name), param))
    counts["trainable_names"] = list(counts["trainable_names"])
    return base_signatures, adapter_signatures, counts


def _flatten_tensor_values(value: Any) -> list[float]:
    if value is None:
        return []
    if getattr(value, "is_nested", False):
        rows = [row.reshape(-1) for row in value.unbind()]
        if not rows:
            return []
        import torch

        value = torch.cat(rows, dim=0)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return [float(item) for item in value]


def _batch_seq_lens(batch: Any, default_count: int) -> list[int]:
    seq_lens = getattr(batch, "seq_lens", None)
    if seq_lens is None and isinstance(batch, dict):
        seq_lens = batch.get("seq_lens")
    values = _flatten_tensor_values(seq_lens)
    if values:
        return [int(value) for value in values]
    return [default_count]


def _split_flat_debug_values(
    values: Any,
    seq_lens: list[int],
    *,
    width: int = 1,
    cast=float,
) -> list[list[Any]]:
    if values is None:
        return [[] for _ in seq_lens]
    flat = [cast(value) for value in values]
    rows: list[list[Any]] = []
    offset = 0
    width = max(1, int(width))
    for seq_len in seq_lens:
        count = int(seq_len) * width
        rows.append(flat[offset : offset + count])
        offset += count
    return rows


@dataclass
class BumblebeeSessionMeta:
    step_count: int = 0
    learning_rate: float = 0.0
    actual_rank: int | None = None


@ray.remote(num_gpus=1, num_cpus=0)
class BumblebeeRankWorker:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        distributed_config: DistributedConfig,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.master_addr = str(master_addr)
        self.master_port = int(master_port)
        self.base_model = str(base_model)
        self.lora_rank = int(lora_rank)
        self.learning_rate = float(learning_rate)
        self.config = distributed_config
        self.rt = None
        self.handle = None
        self._current_session: str | None = None
        self._session_meta: dict[str, BumblebeeSessionMeta] = {}

    def __ray_ready__(self) -> bool:
        return True

    def initialize(self) -> dict[str, Any]:
        repo = _ensure_bumblebee_repo_importable()
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("HF_HOME", "/vePFS-Mindverse/share/huggingface")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from bumblebee.runtime import RuntimeConfig, create_runtime
        from bumblebee.runtime.backends.bb.config import BBConfig
        from bumblebee.runtime.contracts.config import OptimizerConfig, ParallelConfig

        etp = _bumblebee_runtime_etp(self.base_model, self.config)
        bb_cfg = BBConfig(
            model_name="qwen3_moe",
            impl=os.environ.get("MINT_BUMBLEBEE_IMPL", "lite"),
            hf_path=self.base_model,
            parallel=ParallelConfig(
                tp=int(self.config.tensor_parallel_size),
                pp=int(self.config.pipeline_parallel_size),
                ep=int(self.config.expert_parallel_size),
                cp=int(self.config.context_parallel_size),
                etp=etp,
            ),
            optimizer=OptimizerConfig(lr=float(self.learning_rate)),
            load_hf_weights=not _env_flag("MINT_BUMBLEBEE_SKIP_HF_LOAD"),
            impl_cfg={
                "optimizer": os.environ.get("MINT_BUMBLEBEE_OPTIMIZER", "mc_full"),
                "use_thd": True,
                "lora": {
                    "rank": int(self.lora_rank),
                    "max_rank": int(self.lora_rank),
                    "alpha": int(_env_int("MINT_BUMBLEBEE_LORA_ALPHA", self.lora_rank * 2)),
                    "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
                },
            },
        )
        self.rt = create_runtime(RuntimeConfig(backend="bb", hf_path=self.base_model, backend_cfg=bb_cfg))
        self.handle = self.rt.build_model()
        return {"rank": self.rank, "world_size": self.world_size, "backend": "bumblebee", "bumblebee_repo": repo}

    def heartbeat(self) -> dict[str, Any]:
        return {
            "ok": True,
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "session_id": self._current_session,
        }

    def _require_runtime(self):
        if self.rt is None or self.handle is None:
            raise RuntimeError("Bumblebee runtime is not initialized")
        return self.rt, self.handle

    def _iter_lora_modules(self):
        _, handle = self._require_runtime()
        chunks = handle._extras.get("model_chunks", [handle._model])
        for chunk in chunks:
            for _name, module in chunk.named_modules():
                if (
                    hasattr(module, "lora_a")
                    or hasattr(module, "lora_b")
                    or hasattr(module, "lora_A")
                    or hasattr(module, "lora_B")
                ):
                    yield module

    def _set_logical_rank(self, actual_rank: int | None) -> None:
        rank = int(actual_rank if actual_rank is not None else self.lora_rank)
        if rank <= 0 or rank > self.lora_rank:
            raise ValueError(f"actual_rank must be in [1, {self.lora_rank}], got {rank}")
        for module in self._iter_lora_modules():
            setter = getattr(module, "set_logical_rank", None)
            if callable(setter):
                setter(rank)
            else:
                module.rank = rank
                module.logical_rank = rank

    def _reset_adapter_parameters(self) -> None:
        import torch

        for module in self._iter_lora_modules():
            for name in ("lora_a", "lora_A"):
                param = getattr(module, name, None)
                if param is not None:
                    torch.nn.init.xavier_uniform_(param)
            for name in ("lora_b", "lora_B"):
                param = getattr(module, name, None)
                if param is not None:
                    torch.nn.init.zeros_(param)

    def _reset_optimizer_state(self) -> None:
        _, handle = self._require_runtime()
        self._zero_gradients()
        optimizer = handle._optimizer
        if optimizer is None:
            return
        seen: set[int] = set()

        def clear_state(opt: Any) -> None:
            if opt is None or id(opt) in seen:
                return
            seen.add(id(opt))
            for child in getattr(opt, "chained_optimizers", ()) or ():
                clear_state(child)
            state = getattr(opt, "state", None)
            if isinstance(state, dict):
                state.clear()
            try:
                inner = getattr(opt, "optimizer", None)
            except AssertionError:
                inner = None
            if inner is not None:
                clear_state(inner)

        clear_state(optimizer)

    def _zero_gradients(self) -> None:
        rt, handle = self._require_runtime()
        zero_grad = getattr(rt, "zero_grad", None)
        if callable(zero_grad):
            zero_grad(handle)
            return
        optimizer = getattr(handle, "_optimizer", None)
        if optimizer is not None and hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad()
        extras = getattr(handle, "_extras", {}) or {}
        model = getattr(handle, "_model", None)
        for chunk in extras.get("model_chunks", [model] if model is not None else []):
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()

    def _benchmark_model_state_debug(self) -> dict[str, Any]:
        _, handle = self._require_runtime()
        extras = getattr(handle, "_extras", {}) or {}
        chunks = extras.get("model_chunks")
        if not chunks:
            model = getattr(handle, "_model", None)
            chunks = [model] if model is not None else []
        named_params: list[tuple[str, Any]] = []
        chunk_classes: list[str] = []
        config_payload: dict[str, Any] = {}
        for chunk_idx, chunk in enumerate(chunks):
            if chunk is None:
                continue
            chunk_classes.append(type(chunk).__name__)
            config_obj = getattr(chunk, "config", None)
            if config_obj is not None and not config_payload:
                if hasattr(config_obj, "to_dict"):
                    config_payload = _json_debug_value(config_obj.to_dict())
                else:
                    config_payload = _json_debug_value(vars(config_obj))
            for name, param in chunk.named_parameters():
                named_params.append((f"chunk{chunk_idx}.{name}", param))
        base_signatures, adapter_signatures, counts = _select_debug_parameter_signatures(named_params)
        ps = getattr(handle, "_parallel_state", None)
        parallel = {
            "tp_size": getattr(ps, "tp_size", None),
            "tp_rank": getattr(ps, "tp_rank", None),
            "pp_size": getattr(ps, "pp_size", None),
            "pp_rank": getattr(ps, "pp_rank", None),
            "ep_size": getattr(ps, "ep_size", None),
            "ep_rank": getattr(ps, "ep_rank", None),
            "cp_size": getattr(ps, "cp_size", None),
            "cp_rank": getattr(ps, "cp_rank", None),
            "etp_size": getattr(ps, "etp_size", None),
            "etp_rank": getattr(ps, "etp_rank", None),
        }
        return {
            "debug_model_backend": "bumblebee",
            "debug_model_rank": int(self.rank),
            "debug_model_world_size": int(self.world_size),
            "debug_model_base_model": str(self.base_model),
            "debug_model_impl": os.environ.get("MINT_BUMBLEBEE_IMPL", "lite"),
            "debug_model_optimizer": os.environ.get("MINT_BUMBLEBEE_OPTIMIZER", "mc_full"),
            "debug_model_lora_rank": int(self.lora_rank),
            "debug_model_lora_alpha": int(_env_int("MINT_BUMBLEBEE_LORA_ALPHA", self.lora_rank * 2)),
            "debug_model_active_adapter_id": str(extras.get("active_adapter_id")),
            "debug_model_chunk_classes": chunk_classes,
            "debug_model_config": config_payload,
            "debug_model_parallel": _json_debug_value(parallel),
            "debug_model_param_counts": counts,
            "debug_model_base_param_signatures": base_signatures,
            "debug_model_adapter_param_signatures": adapter_signatures,
        }

    def _mint_batch_to_runtime_dict(self, batch: Any) -> dict[str, Any]:
        _, handle = self._require_runtime()
        from bumblebee.runtime.backends.bridge.thd import preprocess_thd
        from bumblebee.runtime.contracts.rl import RLPackedActorBatch

        ps = handle._parallel_state
        thd = preprocess_thd(
            batch,
            tp_size=int(getattr(ps, "tp_size", 1)),
            cp_size=int(getattr(ps, "cp_size", 1)),
            cp_rank=int(getattr(ps, "cp_rank", 0)),
            cp_group=getattr(ps, "cp_group", None),
        )
        runtime_batch = {
            "input_ids": thd.input_ids,
            "target_tokens": thd.labels,
            "labels": thd.labels,
            "loss_mask": thd.loss_mask,
            "position_ids": thd.position_ids,
            "packed_seq_params": thd.packed_seq_params,
        }
        if isinstance(batch, RLPackedActorBatch):
            if int(getattr(ps, "cp_size", 1)) != 1:
                raise NotImplementedError("Bumblebee MinT RL actor payloads are not wired for CP>1 yet")
            runtime_batch["rollout_logprobs"] = self._pad_flat_actor_tensor_to_thd(
                batch.rollout_logprobs,
                batch=batch,
                thd=thd,
                name="rollout_logprobs",
            )
            runtime_batch["advantages"] = self._pad_flat_actor_tensor_to_thd(
                batch.advantages,
                batch=batch,
                thd=thd,
                name="advantages",
            )
            runtime_batch["return_log_probs"] = True
        return runtime_batch

    def _pad_flat_actor_tensor_to_thd(self, tensor: Any, *, batch: Any, thd: Any, name: str) -> Any:
        if thd.loss_mask is None:
            raise ValueError(f"{name} cannot be aligned without THD loss_mask")
        if tuple(getattr(tensor, "shape", ())) == tuple(thd.loss_mask.shape):
            return tensor
        flat = tensor.reshape(-1)
        seq_lens = [int(value) for value in batch.sizes().tolist()]
        total_tokens = sum(seq_lens)
        if int(flat.numel()) != total_tokens:
            raise ValueError(f"{name} numel {flat.numel()} != batch total_tokens {total_tokens}")
        packed_params = thd.packed_seq_params
        cu_padded = packed_params.cu_seqlens_q_padded.tolist()
        out = flat.new_zeros(thd.loss_mask.shape)
        out_flat = out.reshape(-1)
        src_offset = 0
        for idx, seq_len in enumerate(seq_lens):
            dst_start = int(cu_padded[idx])
            out_flat[dst_start : dst_start + seq_len] = flat[src_offset : src_offset + seq_len]
            src_offset += seq_len
        return out

    def _unpad_thd_actor_tensor_to_flat(
        self,
        tensor: Any,
        *,
        batch: Any,
        thd_loss_mask: Any,
        packed_seq_params: Any,
        name: str,
    ) -> Any:
        if tensor is None:
            return None
        actor_loss_mask = getattr(batch, "actor_loss_mask", None)
        target_numel = int(actor_loss_mask.numel()) if actor_loss_mask is not None else sum(
            int(value) for value in batch.sizes().tolist()
        )
        if int(tensor.numel()) == target_numel:
            if actor_loss_mask is not None:
                return tensor.reshape_as(actor_loss_mask)
            return tensor.reshape(-1)
        if thd_loss_mask is None:
            raise ValueError(f"{name} cannot be unpadded without THD loss_mask")
        if int(tensor.numel()) != int(thd_loss_mask.numel()):
            raise ValueError(
                f"{name} numel {tensor.numel()} is neither flat tokens {target_numel} "
                f"nor THD padded tokens {thd_loss_mask.numel()}"
            )

        source = tensor
        if source.dim() >= 2 and int(source.shape[0]) == 1:
            source = source[0]
        if source.dim() == 0:
            raise ValueError(f"{name} cannot be unpadded from scalar tensor")
        if int(source.shape[0]) != int(thd_loss_mask.numel()):
            source = source.reshape(-1)

        import torch

        seq_lens = [int(value) for value in batch.sizes().tolist()]
        cu_padded = packed_seq_params.cu_seqlens_q_padded.tolist()
        pieces = []
        for idx, seq_len in enumerate(seq_lens):
            src_start = int(cu_padded[idx])
            pieces.append(source[src_start : src_start + seq_len])
        if not pieces:
            out = source.new_empty((0,))
        else:
            out = torch.cat(pieces, dim=0)
        if actor_loss_mask is not None:
            return out.reshape_as(actor_loss_mask)
        return out.reshape(-1)

    def _unpad_thd_forward_result_actor_outputs(self, result: Any, *, batch: Any, runtime_batch: dict[str, Any]) -> None:
        model_output = getattr(result, "model_output", None)
        if model_output is None:
            return
        log_probs = getattr(model_output, "log_probs", None)
        if log_probs is None:
            return
        model_output.log_probs = self._unpad_thd_actor_tensor_to_flat(
            log_probs,
            batch=batch,
            thd_loss_mask=runtime_batch.get("loss_mask"),
            packed_seq_params=runtime_batch["packed_seq_params"],
            name="actor_logprobs",
        )

    def _ensure_session_loaded(self, session_id: str, actual_rank: int | None) -> dict[str, Any]:
        rt, handle = self._require_runtime()
        if self._current_session == session_id:
            return {"session_state": "current", "rank": self.rank}
        if self._current_session is not None:
            rt.save_adapter_train_state(
                handle,
                self._current_session,
                include_optimizer=True,
                metadata={"rank": self.rank},
            )
        store = handle._extras.setdefault("adapter_train_states", {})
        if session_id in store:
            rt.switch_active_adapter(
                handle,
                session_id,
                save_current=False,
                include_optimizer=True,
                clear_grad=True,
            )
            session_state = "restored"
        else:
            self._set_logical_rank(actual_rank)
            self._reset_adapter_parameters()
            self._reset_optimizer_state()
            handle._extras["active_adapter_id"] = session_id
            rt.save_adapter_train_state(
                handle,
                session_id,
                include_optimizer=True,
                metadata={"rank": self.rank, "fresh": True},
            )
            session_state = "fresh"
        self._set_logical_rank(actual_rank)
        self._current_session = session_id
        self._session_meta.setdefault(
            session_id,
            BumblebeeSessionMeta(learning_rate=float(self.learning_rate), actual_rank=actual_rank),
        )
        return {"session_state": session_state, "rank": self.rank}

    def forward_backward(
        self,
        data_items: list[dict[str, Any]],
        loss_fn: str,
        loss_fn_config: dict[str, Any],
        rollout_correction_config: dict[str, Any] | None,
        session_id: str,
        actual_rank: int | None,
        *,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del train_attn, train_mlp, train_unembed
        rt, handle = self._require_runtime()
        switch = self._ensure_session_loaded(session_id, actual_rank)
        loss_cfg = dict(loss_fn_config or {})
        if rollout_correction_config:
            loss_cfg["rollout_correction_config"] = dict(rollout_correction_config)

        from bumblebee.runtime.adapters.mint import (
            actor_update_output_to_mint_forward_backward,
            make_mint_actor_loss_fn,
            make_mint_sft_loss_fn,
            mint_datums_to_packed_batch,
        )
        from mint_server.backend.megatron_training import benchmark_debug_input_entries

        device = "cuda"
        self._zero_gradients()
        if loss_fn in {"ppo", "importance_sampling", "grpo"}:
            batch = mint_datums_to_packed_batch(data_items, loss_fn=loss_fn, device=device)
            runtime_batch = self._mint_batch_to_runtime_dict(batch)
            result = rt.forward_backward(
                handle,
                [runtime_batch],
                make_mint_actor_loss_fn(loss_fn, loss_cfg),
                num_microbatches=1,
            )
            self._unpad_thd_forward_result_actor_outputs(result, batch=batch, runtime_batch=runtime_batch)
            from bumblebee.runtime.adapters.rl import actor_update_output_from_forward_result
            from bumblebee.runtime.contracts.rl import RLActorUpdateRequest
            from bumblebee.runtime.adapters.mint import mint_loss_fn_to_actor_objective

            output = actor_update_output_from_forward_result(
                RLActorUpdateRequest(batch=batch, objective=mint_loss_fn_to_actor_objective(loss_fn, loss_cfg)),
                result,
            )
            payload = actor_update_output_to_mint_forward_backward(
                output,
                loss_fn_output_type=f"{loss_fn}_loss",
            )
        else:
            batch = mint_datums_to_packed_batch(data_items, loss_fn=loss_fn, device=device)
            runtime_batch = self._mint_batch_to_runtime_dict(batch)
            result = rt.forward_backward(
                handle,
                [runtime_batch],
                make_mint_sft_loss_fn(),
                num_microbatches=1,
            )
            metrics = dict(result.metrics)
            loss_value = _coerce_scalar(metrics.get("loss", metrics.get("loss:mean")))
            metrics["loss"] = loss_value
            logprobs_by_sample: list[list[float]] = [[] for _ in data_items]
            seq_lens = _batch_seq_lens(batch, 0)
            if _record_benchmark_logprobs_enabled():
                flat_logprobs = metrics.pop("_mint_sft_logprobs", None)
                if flat_logprobs is None:
                    flat_logprobs = _flatten_tensor_values(result.model_output.log_probs)
                else:
                    flat_logprobs = [float(value) for value in flat_logprobs]
                if not seq_lens or seq_lens == [0]:
                    seq_lens = _batch_seq_lens(batch, len(flat_logprobs))
                logprobs_by_sample = []
                offset = 0
                for seq_len in seq_lens[: len(data_items)]:
                    logprobs_by_sample.append(flat_logprobs[offset : offset + seq_len])
                    offset += seq_len
                while len(logprobs_by_sample) < len(data_items):
                    logprobs_by_sample.append([])
            target_logits_by_sample: list[list[float]] = [[] for _ in data_items]
            logsumexp_by_sample: list[list[float]] = [[] for _ in data_items]
            topk_indices_by_sample: list[list[int]] = [[] for _ in data_items]
            topk_logits_by_sample: list[list[float]] = [[] for _ in data_items]
            topk_k = int(metrics.pop("_mint_sft_topk_k", 0) or 0)
            if _record_benchmark_logits_enabled():
                target_logits_by_sample = _split_flat_debug_values(
                    metrics.pop("_mint_sft_target_logits", None),
                    seq_lens[: len(data_items)],
                    cast=float,
                )
                logsumexp_by_sample = _split_flat_debug_values(
                    metrics.pop("_mint_sft_logsumexp", None),
                    seq_lens[: len(data_items)],
                    cast=float,
                )
                if topk_k > 0:
                    topk_indices_by_sample = _split_flat_debug_values(
                        metrics.pop("_mint_sft_topk_indices", None),
                        seq_lens[: len(data_items)],
                        width=topk_k,
                        cast=int,
                    )
                    topk_logits_by_sample = _split_flat_debug_values(
                        metrics.pop("_mint_sft_topk_logits", None),
                        seq_lens[: len(data_items)],
                        width=topk_k,
                        cast=float,
                    )
                while len(target_logits_by_sample) < len(data_items):
                    target_logits_by_sample.append([])
                    logsumexp_by_sample.append([])
                    topk_indices_by_sample.append([])
                    topk_logits_by_sample.append([])
            loss_fn_outputs = []
            for item_idx, logprobs in enumerate(logprobs_by_sample):
                output_entry: dict[str, Any] = {
                    "loss": {"data": [loss_value], "shape": [1], "dtype": "float32"},
                }
                if _record_benchmark_logprobs_enabled():
                    output_entry["logprobs"] = {
                        "data": logprobs,
                        "shape": [len(logprobs)],
                        "dtype": "float32",
                    }
                if _record_benchmark_logits_enabled():
                    target_logits = target_logits_by_sample[item_idx] if item_idx < len(target_logits_by_sample) else []
                    logsumexp = logsumexp_by_sample[item_idx] if item_idx < len(logsumexp_by_sample) else []
                    output_entry["target_logits"] = {
                        "data": target_logits,
                        "shape": [len(target_logits)],
                        "dtype": "float32",
                    }
                    output_entry["logsumexp"] = {
                        "data": logsumexp,
                        "shape": [len(logsumexp)],
                        "dtype": "float32",
                    }
                    if topk_k > 0:
                        topk_indices = topk_indices_by_sample[item_idx] if item_idx < len(topk_indices_by_sample) else []
                        topk_logits = topk_logits_by_sample[item_idx] if item_idx < len(topk_logits_by_sample) else []
                        output_entry["topk_indices"] = {
                            "data": topk_indices,
                            "shape": [len(topk_indices) // topk_k, topk_k],
                            "dtype": "int64",
                        }
                        output_entry["topk_logits"] = {
                            "data": topk_logits,
                            "shape": [len(topk_logits) // topk_k, topk_k],
                            "dtype": "float32",
                        }
                loss_fn_outputs.append(output_entry)
            payload = {
                "loss_fn_output_type": f"{loss_fn}_loss",
                "loss_fn_outputs": loss_fn_outputs,
                "metrics": metrics,
            }

        if _record_benchmark_inputs_enabled():
            loss_fn_outputs = payload.setdefault("loss_fn_outputs", [])
            for item_idx, debug_entry in enumerate(benchmark_debug_input_entries(data_items)):
                while len(loss_fn_outputs) <= item_idx:
                    loss_fn_outputs.append({})
                loss_fn_outputs[item_idx].update(debug_entry)

        payload.setdefault("metrics", {})
        payload["metrics"].update(
            {
                "backend": "bumblebee",
                "rank": self.rank,
                "session_state": switch["session_state"],
            }
        )
        if _record_benchmark_model_state_enabled():
            payload["metrics"].update(self._benchmark_model_state_debug())
        return payload

    def forward(
        self,
        data_items: list[dict[str, Any]],
        session_id: str,
        actual_rank: int | None,
        *,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del train_attn, train_mlp, train_unembed
        rt, handle = self._require_runtime()
        switch = self._ensure_session_loaded(session_id, actual_rank)

        from bumblebee.runtime.adapters.mint import make_mint_sft_loss_fn, mint_datums_to_packed_batch

        batch = mint_datums_to_packed_batch(data_items, loss_fn="cross_entropy", device="cuda")
        runtime_batch = self._mint_batch_to_runtime_dict(batch)
        result = rt.forward_backward(
            handle,
            [runtime_batch],
            make_mint_sft_loss_fn(),
            num_microbatches=1,
            forward_only=True,
        )
        metrics = dict(result.metrics)
        loss_value = _coerce_scalar(metrics.get("loss", metrics.get("loss:mean")))
        metrics.update(
            {
                "backend": "bumblebee",
                "loss": loss_value,
                "rank": self.rank,
                "session_state": switch["session_state"],
            }
        )
        payload = {
            "loss_fn_output_type": "cross_entropy_loss",
            "loss_fn_outputs": [],
            "metrics": metrics,
        }
        return payload

    def optimizer_step(
        self,
        learning_rate: float | None,
        session_id: str,
        actual_rank: int | None,
    ) -> dict[str, Any]:
        rt, handle = self._require_runtime()
        self._ensure_session_loaded(session_id, actual_rank)
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        update_successful, grad_norm, num_zeros = rt.optimizer_step(handle)
        lr = rt.lr_scheduler_step(handle)
        self._zero_gradients()
        meta = self._session_meta.setdefault(session_id, BumblebeeSessionMeta())
        meta.step_count += 1
        meta.learning_rate = float(self.learning_rate)
        meta.actual_rank = actual_rank
        return {
            "metrics": {
                "backend": "bumblebee",
                "update_successful": bool(update_successful),
                "grad_norm": float(grad_norm),
                "num_zeros_in_grad": _coerce_int(num_zeros),
                "learning_rate": float(lr[0] if isinstance(lr, list) and lr else lr or self.learning_rate),
                "rank": self.rank,
            }
        }

    def save_lora_weights(
        self,
        save_path: str,
        *,
        session_id: str,
        actual_rank: int | None,
    ) -> dict[str, Any]:
        self._ensure_session_loaded(session_id, actual_rank)
        _, handle = self._require_runtime()
        from bumblebee.model.qwen3_moe.lite.lora_adapter import save_lora_adapter

        chunks = handle._extras.get("model_chunks", [handle._model])
        meta = save_lora_adapter(
            chunks,
            handle._extras["model_cfg"],
            handle._parallel_state,
            save_path,
            base_model_name_or_path=self.base_model,
            lora_config={
                "rank": int(actual_rank if actual_rank is not None else self.lora_rank),
                "max_rank": int(self.lora_rank),
                "alpha": int(_env_int("MINT_BUMBLEBEE_LORA_ALPHA", self.lora_rank * 2)),
                "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            },
            metadata={
                "backend": "bumblebee",
                "session_id": session_id,
                "rank": self.rank,
            },
        )
        session_meta = self._session_meta.get(session_id, BumblebeeSessionMeta())
        meta.update(
            {
                "current_step": int(session_meta.step_count),
                "learning_rate": float(session_meta.learning_rate or self.learning_rate),
                "actual_rank": int(actual_rank if actual_rank is not None else self.lora_rank),
                "checkpoint_path": str(Path(save_path).resolve()),
            }
        )
        return meta

    def save_training_state(
        self,
        save_path: str,
        *,
        session_id: str,
        actual_rank: int | None,
        include_optimizer: bool = True,
    ) -> dict[str, Any]:
        import torch

        self._ensure_session_loaded(session_id, actual_rank)
        rt, handle = self._require_runtime()
        session_meta = self._session_meta.get(session_id, BumblebeeSessionMeta())
        logical_rank = int(actual_rank if actual_rank is not None else self.lora_rank)
        state = rt.save_adapter_train_state(
            handle,
            session_id,
            include_optimizer=include_optimizer,
            include_lr_scheduler=include_optimizer,
            include_rng=include_optimizer,
            metadata={
                "backend": "bumblebee",
                "session_id": session_id,
                "rank": self.rank,
                "actual_rank": logical_rank,
                "current_step": int(session_meta.step_count),
                "learning_rate": float(session_meta.learning_rate or self.learning_rate),
            },
            store=True,
        )

        root = Path(save_path).resolve()
        rank_dir = root / f"rank_{self.rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        state_path = rank_dir / BUMBLEBEE_TRAIN_STATE_FILE
        payload = {
            "format": BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT,
            "state_format": state.format,
            "adapter_id": str(state.adapter_id),
            "tensors": state.tensors,
            "module_metadata": state.module_metadata,
            "optimizer_state": state.optimizer_state,
            "lr_scheduler_state": state.lr_scheduler_state,
            "rng_state": state.rng_state,
            "step": int(session_meta.step_count),
            "metadata": dict(state.metadata or {}),
            "revision": int(state.revision),
        }
        torch.save(payload, state_path)

        meta = {
            "backend": "bumblebee",
            "format": BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT,
            "session_id": session_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "current_step": int(session_meta.step_count),
            "learning_rate": float(session_meta.learning_rate or self.learning_rate),
            "actual_rank": logical_rank,
            "checkpoint_path": str(root),
            "rank_state_path": str(state_path),
            "optimizer_restored": bool(include_optimizer and state.optimizer_state is not None),
            "has_optimizer": state.optimizer_state is not None,
            "has_lr_scheduler": state.lr_scheduler_state is not None,
            "has_rng": state.rng_state is not None,
        }
        if self.rank == 0:
            (root / BUMBLEBEE_TRAIN_STATE_META_FILE).write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return meta

    def load_checkpoint(
        self,
        load_path: str,
        load_optimizer: bool = True,
        *,
        session_id: str,
        actual_rank: int | None = None,
    ) -> dict[str, Any]:
        return self.load_training_state(
            load_path,
            load_optimizer=load_optimizer,
            session_id=session_id,
            actual_rank=actual_rank,
        )

    def load_training_state(
        self,
        load_path: str,
        load_optimizer: bool = True,
        *,
        session_id: str,
        actual_rank: int | None = None,
    ) -> dict[str, Any]:
        import torch

        self._ensure_session_loaded(session_id, actual_rank)
        rt, handle = self._require_runtime()
        root = Path(load_path).resolve()
        state_path = root / f"rank_{self.rank:05d}" / BUMBLEBEE_TRAIN_STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(
                f"Bumblebee training checkpoint is missing rank state: {state_path}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if payload.get("format") != BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported Bumblebee checkpoint format {payload.get('format')!r} in {state_path}"
            )
        if load_optimizer and payload.get("optimizer_state") is None:
            raise RuntimeError(
                f"Bumblebee checkpoint {state_path} does not include optimizer state"
            )
        state = SimpleNamespace(
            adapter_id=str(session_id),
            format=str(payload["state_format"]),
            tensors=payload.get("tensors") or {},
            module_metadata=payload.get("module_metadata") or {},
            optimizer_state=payload.get("optimizer_state"),
            lr_scheduler_state=payload.get("lr_scheduler_state"),
            rng_state=payload.get("rng_state"),
            step=payload.get("step"),
            metadata=dict(payload.get("metadata") or {}),
            revision=int(payload.get("revision") or 0),
        )
        rt.load_adapter_train_state(
            handle,
            state,
            restore_optimizer=bool(load_optimizer),
            restore_lr_scheduler=bool(load_optimizer),
            restore_rng=bool(load_optimizer),
            clear_grad=True,
        )
        if not load_optimizer:
            self._reset_optimizer_state()
        session_meta = self._session_meta.setdefault(
            session_id,
            BumblebeeSessionMeta(learning_rate=float(self.learning_rate), actual_rank=actual_rank),
        )
        meta_path = root / BUMBLEBEE_TRAIN_STATE_META_FILE
        loaded_meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to read Bumblebee training_meta.json from %s", meta_path, exc_info=True)
        step = loaded_meta.get("current_step", payload.get("step", 0))
        if step is not None:
            session_meta.step_count = int(step)
        lr = loaded_meta.get("learning_rate", state.metadata.get("learning_rate"))
        if lr is not None:
            session_meta.learning_rate = float(lr)
        session_meta.actual_rank = actual_rank
        self._current_session = session_id
        return {
            "backend": "bumblebee",
            "current_step": int(session_meta.step_count),
            "learning_rate": float(session_meta.learning_rate or self.learning_rate),
            "actual_rank": int(actual_rank if actual_rank is not None else self.lora_rank),
            "checkpoint_path": str(root),
            "rank_state_path": str(state_path),
            "optimizer_restored": bool(load_optimizer),
            "has_optimizer": payload.get("optimizer_state") is not None,
            "has_lr_scheduler": payload.get("lr_scheduler_state") is not None,
            "has_rng": payload.get("rng_state") is not None,
        }

    def mark_session_loaded(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        meta = self._session_meta.setdefault(session_id, BumblebeeSessionMeta())
        if kwargs.get("step_count") is not None:
            meta.step_count = int(kwargs["step_count"])
        if kwargs.get("learning_rate") is not None:
            meta.learning_rate = float(kwargs["learning_rate"])
        if kwargs.get("actual_rank") is not None:
            meta.actual_rank = int(kwargs["actual_rank"])
        self._current_session = session_id
        return {"status": "ok", "backend": "bumblebee"}

    def delete_session(self, session_id: str, *, traceparent: str | None = None) -> dict[str, Any]:
        del traceparent
        _, handle = self._require_runtime()
        store = handle._extras.setdefault("adapter_train_states", {})
        store.pop(session_id, None)
        self._session_meta.pop(session_id, None)
        if self._current_session == session_id:
            self._current_session = None
        return {"status": "ok", "backend": "bumblebee", "session_id": session_id}


@ray.remote(num_gpus=0, num_cpus=0)
class BumblebeeWorkerGroup:
    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        distributed_config: DistributedConfig | None = None,
        observability_base_model: str | None = None,
        traceparent: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.base_model = str(base_model)
        self.observability_base_model = str(observability_base_model or base_model or "unknown")
        self.lora_rank = int(lora_rank)
        self.learning_rate = float(learning_rate)
        self.config = distributed_config or DistributedConfig()
        self.traceparent = traceparent
        self.request_id = str(request_id or "") or None
        self.workers: list[ray.actor.ActorHandle] = []
        self.placement_group = None
        self._step_count = 0
        self._current_session: str | None = None
        self._initialize()

    def __ray_ready__(self) -> bool:
        return True

    def heartbeat(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "bumblebee",
            "base_model": self.observability_base_model,
            "world_size": int(self.config.world_size),
            "session_id": self._current_session,
            "step": self._step_count,
        }

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "bumblebee",
            "base_model": self.base_model,
            "observability_base_model": self.observability_base_model,
            "lora_rank": int(self.lora_rank),
            "world_size": int(self.config.world_size),
            "step": int(self._step_count),
        }

    def get_observability_binding(self) -> dict[str, Any]:
        return {
            "backend": "bumblebee",
            "base_model": self.observability_base_model,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }

    def get_tokenizer_info(self) -> dict[str, Any]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, local_files_only=True)
        return {
            "vocab_size": len(tokenizer),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "bos_token": getattr(tokenizer, "bos_token", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "unk_token": getattr(tokenizer, "unk_token", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        }

    def _initialize(self) -> None:
        world_size = int(self.config.world_size)
        bundles: list[dict[str, float | int]] = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
        placement = _model_gpu_placement_for_model(self.base_model)
        if placement is not None:
            if placement.total_gpus != world_size:
                raise ValueError(
                    f"Bumblebee placement GPU count mismatch for base_model={self.base_model!r}: "
                    f"need {world_size}, got {placement.total_gpus}"
                )
            assert_node_ip_capacity(
                required_gpus_by_node_ip=placement.required_gpus_by_node_ip(),
                context=f"[BumblebeeWorkerGroup] node pinning base_model={self.base_model}",
                ignore_placement_group_names={_make_bumblebee_pg_name(self.base_model)},
                ignore_placement_group_namespace=PERSISTENT_NAMESPACE,
            )
            bundles = placement.pg_bundles(cpu_per_gpu=1)

        self.placement_group = _get_or_create_megatron_placement_group(
            pg_name=_make_bumblebee_pg_name(self.base_model),
            bundles=bundles,
        )
        ray.get(self.placement_group.ready())
        bundle_ips = [_bundle_node_ip(bundle) for bundle in bundles]

        runtime_pythonpath = PFS_PYTHONPATH
        repo = _bumblebee_repo_path()
        if repo not in runtime_pythonpath.split(":"):
            runtime_pythonpath = repo + ":" + runtime_pythonpath
        runtime_env = {
            "env_vars": actor_runtime_env_vars(
                pythonpath=runtime_pythonpath,
                extra={
                    "USE_TORCH": "1",
                    "USE_TF": "0",
                    "USE_FLAX": "0",
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    **otel_env_vars(),
                },
            )
        }
        for key in (
            "MINT_BUMBLEBEE_REPO_PATH",
            "MINT_BUMBLEBEE_IMPL",
            "MINT_BUMBLEBEE_OPTIMIZER",
            "MINT_BUMBLEBEE_SKIP_HF_LOAD",
            "MINT_BUMBLEBEE_LORA_ALPHA",
            "MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON",
            "MINT_MODEL_PLACEMENT_JSON",
            "BUMBLEBEE_BUILD_TRACE",
            "BUMBLEBEE_RL_DEBUG_METRICS",
            "BUMBLEBEE_Q3MOE_GQA_PROBE",
            "BUMBLEBEE_Q3MOE_GQA_PROBE_ALL_RANKS",
            "MINT_BENCH_RECORD_LOGPROBS",
            "MINT_BENCH_RECORD_LOGITS",
            "MINT_BENCH_RECORD_TOPK",
            "MINT_BENCH_RECORD_INPUTS",
            "MINT_BENCH_RECORD_MODEL_STATE",
            "CUDA_LAUNCH_BLOCKING",
            "TORCH_DISTRIBUTED_DEBUG",
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NVTE_DEBUG",
            "NVTE_DEBUG_LEVEL",
        ):
            value = os.environ.get(key)
            if value is not None:
                runtime_env["env_vars"][key] = value

        master_addr, master_port = ray.get(
            get_node_ip_and_free_port.options(
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=0,
                ),
                resources=_node_affinity_resources(bundle_ips[0]),
                runtime_env=runtime_env,
            ).remote()
        )

        for rank in range(world_size):
            worker = BumblebeeRankWorker.options(
                num_gpus=1,
                num_cpus=0,
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=rank,
                ),
                resources=_node_affinity_resources(bundle_ips[rank]),
                runtime_env=runtime_env,
            ).remote(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                base_model=self.base_model,
                lora_rank=self.lora_rank,
                learning_rate=self.learning_rate,
                distributed_config=self.config,
            )
            self.workers.append(worker)
        ray.get([worker.__ray_ready__.remote() for worker in self.workers])
        ray.get([worker.initialize.remote() for worker in self.workers])

    def _ray_get_group_results(self, refs: list[ray.ObjectRef], *, op: str) -> list[Any]:
        timeout_s = float(server_config.training_remote_call_timeout_s or 0.0)
        if timeout_s <= 0:
            timeout_s = 3600.0 if op in {"train_step", "save_lora_weights", "save_training_state", "load_training_state"} else 1800.0
        return ray.get(refs, timeout=timeout_s)

    def forward_backward(
        self,
        data_items: list[dict[str, Any]],
        loss_fn: str,
        loss_fn_config: dict[str, Any],
        rollout_correction_config: dict[str, Any] | None,
        session_id: str,
        actual_rank: int | None,
        *,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del traceparent
        refs = [
            worker.forward_backward.remote(
                data_items,
                loss_fn,
                loss_fn_config,
                rollout_correction_config,
                session_id,
                actual_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="forward_backward")
        self._current_session = session_id
        return self._merge_rank_payloads(results)

    def forward(
        self,
        data_items: list[dict[str, Any]],
        session_id: str,
        actual_rank: int | None = None,
        *,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del traceparent
        refs = [
            worker.forward.remote(
                data_items,
                session_id,
                actual_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="forward")
        self._current_session = session_id
        return self._merge_rank_payloads(results)

    def optim_step(
        self,
        learning_rate: float | None,
        session_id: str,
        actual_rank: int | None,
        *,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del traceparent, train_attn, train_mlp, train_unembed
        refs = [
            worker.optimizer_step.remote(learning_rate, session_id, actual_rank)
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="optim_step")
        self._step_count += 1
        merged = self._merge_rank_payloads(results)
        merged.setdefault("metrics", {})["step"] = self._step_count
        return merged

    def train_step(
        self,
        data_items: list[dict[str, Any]],
        loss_fn: str,
        loss_fn_config: dict[str, Any],
        rollout_correction_config: dict[str, Any] | None,
        learning_rate: float | None,
        session_id: str,
        actual_rank: int | None,
        *,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        fb = self.forward_backward(
            data_items,
            loss_fn,
            loss_fn_config,
            rollout_correction_config,
            session_id,
            actual_rank,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        opt = self.optim_step(
            learning_rate,
            session_id,
            actual_rank,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        fb.setdefault("metrics", {}).update(opt.get("metrics", {}))
        return fb

    def save_lora_weights(
        self,
        save_path: str,
        *,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict[str, Any]:
        del traceparent, train_attn, train_mlp, train_unembed
        sid = session_id or self._current_session
        if not sid:
            raise RuntimeError("save_lora_weights requires session_id")
        refs = [
            worker.save_lora_weights.remote(save_path, session_id=sid, actual_rank=actual_rank)
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="save_lora_weights")
        for result in results:
            if isinstance(result, dict) and result.get("adapter_config"):
                return result
        return {"checkpoint_path": str(Path(save_path).resolve()), "backend": "bumblebee"}

    def save_checkpoint(self, save_path: str, **kwargs: Any) -> dict[str, Any]:
        return self.save_training_state(save_path, **kwargs)

    def save_training_state(
        self,
        save_path: str,
        *,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
        include_optimizer: bool = True,
    ) -> dict[str, Any]:
        del traceparent, train_attn, train_mlp, train_unembed
        sid = session_id or self._current_session
        if not sid:
            raise RuntimeError("save_training_state requires session_id")
        refs = [
            worker.save_training_state.remote(
                save_path,
                session_id=sid,
                actual_rank=actual_rank,
                include_optimizer=include_optimizer,
            )
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="save_training_state")
        return self._merge_rank_payloads(results)

    def load_checkpoint(self, load_path: str, load_optimizer: bool = True, **kwargs: Any) -> dict[str, Any]:
        return self.load_training_state(load_path, load_optimizer=load_optimizer, **kwargs)

    def load_training_state(self, load_path: str, load_optimizer: bool = True, **kwargs: Any) -> dict[str, Any]:
        sid = kwargs.get("session_id") or self._current_session
        if not sid:
            raise RuntimeError("load_training_state requires session_id")
        actual_rank = kwargs.get("actual_rank")
        refs = [
            worker.load_training_state.remote(
                load_path,
                load_optimizer,
                session_id=sid,
                actual_rank=actual_rank,
            )
            for worker in self.workers
        ]
        results = self._ray_get_group_results(refs, op="load_training_state")
        self._current_session = str(sid)
        return self._merge_rank_payloads(results)

    def mark_session_loaded(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        refs = [
            worker.mark_session_loaded.remote(session_id, **kwargs)
            for worker in self.workers
        ]
        self._ray_get_group_results(refs, op="mark_session_loaded")
        self._current_session = session_id
        return {"status": "ok", "backend": "bumblebee"}

    def delete_session(self, session_id: str, *, traceparent: str | None = None) -> dict[str, Any]:
        refs = [
            worker.delete_session.remote(session_id, traceparent=traceparent)
            for worker in self.workers
        ]
        self._ray_get_group_results(refs, op="delete_session")
        if self._current_session == session_id:
            self._current_session = None
        return {"status": "ok", "backend": "bumblebee", "session_id": session_id}

    def reset_expert_bias(self, *, traceparent: str | None = None) -> dict[str, Any]:
        del traceparent
        return {"modules_reset": 0, "backend": "bumblebee"}

    def _merge_rank_payloads(self, results: list[Any]) -> dict[str, Any]:
        primary = next((item for item in results if isinstance(item, dict)), {})
        payload = dict(primary)
        metrics = dict(payload.get("metrics") or {})
        rank_metrics = [dict(item.get("metrics") or {}) for item in results if isinstance(item, dict)]
        metrics["backend"] = "bumblebee"
        metrics["world_size"] = int(self.config.world_size)
        metrics["rank_metrics"] = rank_metrics
        payload["metrics"] = metrics
        return payload


def _model_gpu_placement_for_model(base_model: str):
    model_key = _model_key_from_base_model(base_model)
    lookup_keys = [model_key, model_key.lower(), base_model, base_model.lower()]
    return parse_model_gpu_placement(
        raw_json=os.environ.get("MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON")
        or os.environ.get("MINT_MODEL_PLACEMENT_JSON"),
        lookup_keys=lookup_keys,
        env_var_name="MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON",
        context=f"[BumblebeeWorkerGroup] node pinning model={model_key}",
    )


def get_or_create_bumblebee_worker_group(
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    distributed_config: DistributedConfig | None = None,
    session_id: str | None = None,
    actual_rank: int | None = None,
    observability_base_model: str | None = None,
    traceparent: str | None = None,
    request_id: str | None = None,
) -> ray.actor.ActorHandle:
    from mint_server.backend.model_actor_inventory import ActorType
    from mint_server.backend.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor

    del session_id
    if not ray.is_initialized():
        init_ray(namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    config = distributed_config or DistributedConfig()
    actor_name = _make_bumblebee_actor_name(base_model)
    observability_model = str(observability_base_model or base_model or "unknown")
    rank_metadata = {
        "backend": "bumblebee",
        "max_lora_rank": int(lora_rank),
        "actual_rank": int(actual_rank if actual_rank is not None else lora_rank),
    }

    with _get_bumblebee_create_lock(actor_name):
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            diagnostics = ray.get(actor.get_diagnostics.remote(), timeout=10)
            observed_rank = diagnostics.get("lora_rank") if isinstance(diagnostics, dict) else None
            if int(observed_rank) != int(lora_rank):
                ray_kill.kill(
                    actor,
                    reason="bumblebee_actor_lora_rank_mismatch",
                    actor_name=actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    verify_absent=True,
                )
                raise ValueError("Bumblebee actor lora_rank mismatch, will recreate")
            publish_backend_model_actor(
                BackendModelActorLaunch(
                    actor_name=actor_name,
                    actor_type=ActorType.MEGATRON,
                    num_gpus=int(config.world_size),
                    actor_handle=actor,
                    namespace=PERSISTENT_NAMESPACE,
                    base_model=observability_model,
                    protected=is_topology_desired_model(base_model),
                    metadata=rank_metadata,
                )
            )
            return actor
        except ValueError:
            logger.info("Creating new detached Bumblebee actor: %s for %s", actor_name, base_model)

        runtime_pythonpath = PFS_PYTHONPATH
        repo = _bumblebee_repo_path()
        if repo not in runtime_pythonpath.split(":"):
            runtime_pythonpath = repo + ":" + runtime_pythonpath
        runtime_env = {
            "env_vars": actor_runtime_env_vars(
                pythonpath=runtime_pythonpath,
                extra={
                    "USE_TORCH": "1",
                    "USE_TF": "0",
                    "USE_FLAX": "0",
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    **otel_env_vars(),
                },
            )
        }
        for key in (
            "MINT_BUMBLEBEE_REPO_PATH",
            "MINT_BUMBLEBEE_LORA_ALPHA",
            "MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON",
            "MINT_MODEL_PLACEMENT_JSON",
            "BUMBLEBEE_RL_DEBUG_METRICS",
            "BUMBLEBEE_Q3MOE_GQA_PROBE",
            "BUMBLEBEE_Q3MOE_GQA_PROBE_ALL_RANKS",
            "MINT_BENCH_RECORD_LOGPROBS",
            "MINT_BENCH_RECORD_LOGITS",
            "MINT_BENCH_RECORD_TOPK",
            "MINT_BENCH_RECORD_INPUTS",
            "MINT_BENCH_RECORD_MODEL_STATE",
            "NVTE_DEBUG",
            "NVTE_DEBUG_LEVEL",
        ):
            value = os.environ.get(key)
            if value is not None:
                runtime_env["env_vars"][key] = value

        manager_options: dict[str, Any] = {
            "name": actor_name,
            "namespace": PERSISTENT_NAMESPACE,
            "lifetime": "detached",
            "runtime_env": runtime_env,
        }
        placement = _model_gpu_placement_for_model(base_model)
        if placement is not None and placement.node_ips:
            manager_options["resources"] = _node_affinity_resources(placement.node_ips[0])

        actor = BumblebeeWorkerGroup.options(
            **manager_options,
        ).remote(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            distributed_config=config,
            observability_base_model=observability_model,
            traceparent=traceparent,
            request_id=request_id,
        )
        publish_backend_model_actor(
            BackendModelActorLaunch(
                actor_name=actor_name,
                actor_type=ActorType.MEGATRON,
                num_gpus=int(config.world_size),
                actor_handle=actor,
                namespace=PERSISTENT_NAMESPACE,
                base_model=observability_model,
                protected=is_topology_desired_model(base_model),
                metadata=rank_metadata,
            ),
            ready=False,
        )
        return actor


async def async_get_or_create_bumblebee_worker_group(
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    distributed_config: DistributedConfig | None = None,
    session_id: str | None = None,
    actual_rank: int | None = None,
    observability_base_model: str | None = None,
    traceparent: str | None = None,
    request_id: str | None = None,
) -> ray.actor.ActorHandle:
    return await asyncio.to_thread(
        get_or_create_bumblebee_worker_group,
        base_model,
        lora_rank,
        learning_rate,
        distributed_config,
        session_id,
        actual_rank,
        observability_base_model,
        traceparent or get_current_traceparent(),
        request_id or get_request_id(),
    )
