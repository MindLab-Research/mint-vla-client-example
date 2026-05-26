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
    return os.environ.get("MINT_BUMBLEBEE_REPO_PATH", "/root/code/bumblebee")


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
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            for name in ("lora_b", "lora_B"):
                param = getattr(module, name, None)
                if param is not None:
                    torch.nn.init.zeros_(param)

    def _reset_optimizer_state(self) -> None:
        _, handle = self._require_runtime()
        optimizer = handle._optimizer
        if optimizer is None:
            return
        state = getattr(optimizer, "state", None)
        if isinstance(state, dict):
            state.clear()

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
            runtime_batch["rollout_logprobs"] = batch.rollout_logprobs.reshape_as(thd.loss_mask)
            runtime_batch["advantages"] = batch.advantages.reshape_as(thd.loss_mask)
            runtime_batch["return_log_probs"] = True
        return runtime_batch

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
            mint_datums_to_packed_batch,
        )

        device = "cuda"
        if loss_fn in {"ppo", "importance_sampling", "grpo"}:
            batch = mint_datums_to_packed_batch(data_items, loss_fn=loss_fn, device=device)
            runtime_batch = self._mint_batch_to_runtime_dict(batch)
            result = rt.forward_backward(
                handle,
                [runtime_batch],
                make_mint_actor_loss_fn(loss_fn, loss_cfg),
                num_microbatches=1,
            )
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
            result = rt.forward_backward(handle, [runtime_batch], None, num_microbatches=1)
            loss_value = _coerce_scalar(result.metrics.get("loss"))
            payload = {
                "loss_fn_output_type": f"{loss_fn}_loss",
                "loss_fn_outputs": [
                    {"loss": {"data": [loss_value], "shape": [1], "dtype": "float32"}}
                    for _ in data_items
                ],
                "metrics": {"loss": loss_value},
            }

        payload.setdefault("metrics", {})
        payload["metrics"].update(
            {
                "backend": "bumblebee",
                "rank": self.rank,
                "session_state": switch["session_state"],
            }
        )
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

        from bumblebee.runtime.adapters.mint import mint_datums_to_packed_batch

        batch = mint_datums_to_packed_batch(data_items, loss_fn="cross_entropy", device="cuda")
        runtime_batch = self._mint_batch_to_runtime_dict(batch)
        result = rt.forward_backward(handle, [runtime_batch], None, num_microbatches=1, forward_only=True)
        payload = {
            "loss_fn_output_type": "cross_entropy_loss",
            "loss_fn_outputs": [],
            "metrics": {
                "backend": "bumblebee",
                "loss": _coerce_scalar(result.metrics.get("loss")),
                "rank": self.rank,
                "session_state": switch["session_state"],
            },
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
