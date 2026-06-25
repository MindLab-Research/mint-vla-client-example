from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import time
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref
from mint_server.backend.core.model_registry import get_model_config
from mint_server.config import (
    MINT_CODE_ROOT,
    PFS_HF_MODULES_PATH,
    PFS_PYTHONPATH,
    PFS_RUNTIME_ENV_ROOT,
    RAY_NAMESPACE,
    actor_runtime_env_vars,
    otel_env_vars,
)
from mint_server.observability.logging_context import run_async_with_otel_span
from mint_server.ray.ray_utils import init_ray
from mint_server.ray.runtime_env import TIER_GPU_RL, join_pythonpath, runtime_env_layout
from mint_server.config import config as server_config

import mint_server.backend.ray_cluster.ray_kill as ray_kill
from .actors.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor
from .actors.model_actor_inventory import ActorType
from .inference.multi_lora_engine import GenerateResult, _float_or_none, _resolve_model_path
from .actors.node_placement import assert_node_ip_capacity, parse_model_gpu_placement
from .ray_cluster.ray_placement_groups import remove_named_placement_group
from .actors.ray_keepalive import ray_get_with_model_actor_supervisor_keepalive
from .sampling_backend import actor_name_for_sampling_base_model
logger = logging.getLogger(__name__)

PERSISTENT_NAMESPACE = RAY_NAMESPACE

_MANAGERS: dict[str, "SGLangInferenceEngine"] = {}
_MANAGER_LOCKS: dict[str, asyncio.Lock] = {}
_MANAGER_LOCKS_GUARD = asyncio.Lock()


@dataclass(frozen=True)
class _SGLangPlacementPlan:
    mode: str
    total_gpus: int
    node_gpus: tuple[tuple[str, int], ...] = ()

    @property
    def is_multinode(self) -> bool:
        return self.mode == "multinode"

    @property
    def node_ips(self) -> list[str]:
        return [node_ip for node_ip, _gpu_count in self.node_gpus]

    def required_gpus_by_node_ip(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for node_ip, gpu_count in self.node_gpus:
            out[node_ip] = out.get(node_ip, 0) + int(gpu_count)
        return out


def _invalidate_model_session_loras(model_name: str) -> None:
    """Force SGLang LoRA sessions for a model to reload after actor recreate."""
    try:
        managers = []

        from .sessions.session_manager import session_manager as backend_session_manager

        if backend_session_manager is not None:
            managers.append(backend_session_manager)

        from ..routes import sampling as sampling_routes

        route_session_manager = getattr(sampling_routes, "session_manager", None)
        if route_session_manager is not None and route_session_manager not in managers:
            managers.append(route_session_manager)

        invalidated = 0
        for manager in managers:
            invalidated += manager.mark_model_lora_sessions_unloaded(model_name)
        if invalidated:
            logger.info(
                "Invalidated cached SGLang LoRA load state for %s session(s) on model=%s after actor recreate",
                invalidated,
                model_name,
            )
    except Exception as e:
        logger.warning(
            "Failed to invalidate cached SGLang LoRA load state for model=%s: %s: %s",
            model_name,
            type(e).__name__,
            e,
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _positive_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _sglang_remote_timeout_config(op: str) -> tuple[float | None, str | None]:
    op_key = {
        "generate": "MINT_SGLANG_GENERATE_TIMEOUT_S",
        "compute_prompt_logprobs": "MINT_SGLANG_LOGPROBS_TIMEOUT_S",
        "compute_prompt_topk": "MINT_SGLANG_TOPK_TIMEOUT_S",
    }.get(op)
    if op_key is not None:
        timeout_s = _positive_env_float(op_key)
        if timeout_s is not None:
            return timeout_s, op_key
    timeout_s = _positive_env_float("MINT_SGLANG_REQUEST_TIMEOUT_S")
    if timeout_s is not None:
        return timeout_s, "MINT_SGLANG_REQUEST_TIMEOUT_S"
    return None, None


def _sglang_remote_timeout_s(op: str) -> float | None:
    return _sglang_remote_timeout_config(op)[0]


def _default_sglang_engine_ready_wait_s(config: Any, *, total_gpus: int) -> float:
    try:
        num_parameters = float(getattr(config, "num_parameters", 0.0) or 0.0)
    except (TypeError, ValueError):
        num_parameters = 0.0
    try:
        gpu_count = int(total_gpus)
    except (TypeError, ValueError):
        gpu_count = 1
    is_moe = bool(getattr(config, "is_moe", False))

    if gpu_count >= 16 or num_parameters >= 200.0:
        return 3600.0
    if gpu_count >= 4 or num_parameters >= 25.0 or is_moe:
        return 1800.0
    return 900.0


def _sglang_slow_request_log_s() -> float | None:
    raw = os.environ.get("MINT_SGLANG_SLOW_REQUEST_LOG_S")
    if raw is not None and str(raw).strip() != "":
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return 30.0
        return value if value > 0 else None
    return 30.0


def _sglang_pythonpath() -> str:
    extra = os.environ.get("MINT_SGLANG_PYTHONPATH", "").strip()
    base = _sglang_base_pythonpath()
    if not extra:
        return base
    return join_pythonpath(extra, base)


def _sglang_base_pythonpath() -> str:
    """Use GPU site-packages without training source trees for SGLang actors."""
    env_root = str(PFS_RUNTIME_ENV_ROOT or "").strip()
    code_root = str(MINT_CODE_ROOT or "").strip()
    hf_modules = str(PFS_HF_MODULES_PATH or "").strip()
    if env_root and code_root and hf_modules:
        try:
            layout = runtime_env_layout(env_root, tier=TIER_GPU_RL)
            return join_pythonpath(
                layout.site_packages,
                code_root,
                hf_modules,
            )
        except Exception as exc:
            logger.warning("Failed to build narrow SGLang PYTHONPATH, falling back to GPU path: %s", exc)
    return PFS_PYTHONPATH


def _sglang_py_executable() -> str | None:
    raw = os.environ.get("MINT_SGLANG_PY_EXECUTABLE", "").strip()
    return raw or None


def _sglang_pycache_prefix() -> str | None:
    raw = os.environ.get("MINT_SGLANG_PYCACHE_PREFIX")
    if raw is not None:
        value = raw.strip()
        return value or None
    try:
        suffix = str(os.getuid())
    except AttributeError:
        suffix = "default"
    return os.path.join("/tmp", "mint_sglang_pycache", suffix)


def _worker_extra_env() -> dict[str, str]:
    out: dict[str, str] = {
        "MINT_SGLANG_PYTHONPATH": os.environ.get("MINT_SGLANG_PYTHONPATH", "").strip(),
        "MINT_SGLANG_PY_EXECUTABLE": os.environ.get("MINT_SGLANG_PY_EXECUTABLE", "").strip(),
        "HF_HOME": os.environ.get("HF_HOME", "/vePFS-Mindverse/share/huggingface"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
    }
    pycache_prefix = _sglang_pycache_prefix()
    if pycache_prefix:
        out["MINT_SGLANG_PYCACHE_PREFIX"] = pycache_prefix
        out["PYTHONPYCACHEPREFIX"] = pycache_prefix
    for key in (
        "MINT_SGLANG_MEM_FRACTION_STATIC",
        "MINT_SGLANG_MAX_RUNNING_REQUESTS",
        "MINT_SGLANG_MAX_TOTAL_TOKENS",
        "MINT_SGLANG_MAX_PREFILL_TOKENS",
        "MINT_SGLANG_CHUNKED_PREFILL_SIZE",
        "MINT_SGLANG_DISABLE_CUDA_GRAPH",
        "MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH",
        "MINT_SGLANG_DISABLE_FLASHINFER_KERNELS",
        "MINT_SGLANG_DISABLE_FLASHINFER_AUTOTUNE",
        "MINT_SGLANG_DISABLE_CUSTOM_ALL_REDUCE",
        "MINT_SGLANG_SKIP_SERVER_WARMUP",
        "MINT_SGLANG_TRUST_REMOTE_CODE",
        "MINT_SGLANG_LOG_LEVEL",
        "MINT_SGLANG_SHOW_TIME_COST",
        "MINT_SGLANG_WATCHDOG_TIMEOUT_S",
        "MINT_SGLANG_MODEL_PLACEMENT_JSON",
        "MINT_SGLANG_REQUEST_TIMEOUT_S",
        "MINT_SGLANG_GENERATE_TIMEOUT_S",
        "MINT_SGLANG_LOGPROBS_TIMEOUT_S",
        "MINT_SGLANG_TOPK_TIMEOUT_S",
        "MINT_SGLANG_SLOW_REQUEST_LOG_S",
        "MINT_SGLANG_PYCACHE_PREFIX",
        "MINT_SGLANG_ENABLE_LORA",
        "MINT_SGLANG_ENABLE_LORA_OVERLAP_LOADING",
        "MINT_SGLANG_MAX_LORA_RANK",
        "MINT_SGLANG_LORA_TARGET_MODULES",
        "MINT_SGLANG_MAX_LOADED_LORAS",
        "MINT_SGLANG_MAX_LORAS_PER_BATCH",
        "MINT_SGLANG_LORA_BACKEND",
    ):
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
            if key == "MINT_SGLANG_PYCACHE_PREFIX":
                stripped = str(value).strip()
                if stripped:
                    out["PYTHONPYCACHEPREFIX"] = stripped
                else:
                    out.pop("PYTHONPYCACHEPREFIX", None)
    return {k: v for k, v in out.items() if v is not None}


def _runtime_env() -> dict[str, Any]:
    runtime_env: dict[str, Any] = {
        "env_vars": actor_runtime_env_vars(
            pythonpath=_sglang_pythonpath(),
            extra={
                **_worker_extra_env(),
                **otel_env_vars(),
            },
        )
    }
    py_executable = _sglang_py_executable()
    if py_executable:
        runtime_env["py_executable"] = py_executable
    return runtime_env


def _get_actor_node_id(actor_handle: ray.actor.ActorHandle) -> str | None:
    try:
        actor_id_hex = actor_handle._actor_id.hex()
        from ray._private.state import actors as state_actors

        actor_info = state_actors(actor_id_hex)
        if actor_info:
            address = actor_info.get("Address", {})
            return address.get("NodeID")
    except Exception as e:
        logger.debug("Could not get SGLang actor node_id: %s", e)
    return None


def _single_node_pin_for_model(model_name: str, actor_name: str, total_gpus: int) -> str | None:
    raw_sglang_placement = os.environ.get("MINT_SGLANG_MODEL_PLACEMENT_JSON", "")
    raw_model_placement = os.environ.get("MINT_MODEL_PLACEMENT_JSON", "")
    lookup_keys = [
        str(model_name).strip(),
        str(model_name).strip().lower(),
        str(actor_name).strip(),
        str(actor_name).strip().lower(),
    ]
    context = f"single_node_sglang_pin model={model_name!r} actor={actor_name!r}"
    placement = parse_model_gpu_placement(
        raw_json=raw_sglang_placement,
        lookup_keys=lookup_keys,
        env_var_name="MINT_SGLANG_MODEL_PLACEMENT_JSON",
        context=context,
        replica=0,
    )
    if placement is None:
        placement = parse_model_gpu_placement(
            raw_json=raw_model_placement,
            lookup_keys=lookup_keys,
            env_var_name="MINT_MODEL_PLACEMENT_JSON",
            context=context,
            replica=0,
        )
    if placement is None:
        logger.info(
            "single_node_sglang_pin_lookup model=%r actor=%r pinned_node_ip=None lookup_keys=%s raw_sglang_placement_present=%s raw_model_placement_present=%s",
            model_name,
            actor_name,
            lookup_keys,
            bool(raw_sglang_placement),
            bool(raw_model_placement),
        )
        return None
    if len(placement.slices) != 1:
        raise RuntimeError(f"{context}: single-node SGLang requires exactly 1 placement slice, got {len(placement.slices)}")
    if placement.total_gpus != int(total_gpus):
        raise RuntimeError(f"{context}: placement GPU count mismatch, need {total_gpus} GPUs, got {placement.total_gpus}")
    pinned_node_ip = placement.slices[0].node_ip
    logger.info(
        "single_node_sglang_pin_lookup model=%r actor=%r pinned_node_ip=%s lookup_keys=%s raw_sglang_placement_present=%s raw_model_placement_present=%s",
        model_name,
        actor_name,
        pinned_node_ip,
        lookup_keys,
        bool(raw_sglang_placement),
        bool(raw_model_placement),
    )
    return pinned_node_ip


def _placement_for_model(
    *,
    model_name: str,
    actor_name: str,
    context: str,
) -> Any | None:
    raw_sglang_placement = os.environ.get("MINT_SGLANG_MODEL_PLACEMENT_JSON", "")
    raw_model_placement = os.environ.get("MINT_MODEL_PLACEMENT_JSON", "")
    lookup_keys = [
        str(model_name).strip(),
        str(model_name).strip().lower(),
        str(actor_name).strip(),
        str(actor_name).strip().lower(),
    ]
    placement = parse_model_gpu_placement(
        raw_json=raw_sglang_placement,
        lookup_keys=lookup_keys,
        env_var_name="MINT_SGLANG_MODEL_PLACEMENT_JSON",
        context=context,
        replica=0,
    )
    if placement is None:
        placement = parse_model_gpu_placement(
            raw_json=raw_model_placement,
            lookup_keys=lookup_keys,
            env_var_name="MINT_MODEL_PLACEMENT_JSON",
            context=context,
            replica=0,
        )
    return placement


def _sglang_placement_plan_for_model(
    *,
    model_name: str,
    actor_name: str,
    total_gpus: int,
) -> _SGLangPlacementPlan:
    total = int(total_gpus)
    context = f"sglang_placement model={model_name!r} actor={actor_name!r}"
    placement = _placement_for_model(model_name=model_name, actor_name=actor_name, context=context)
    if placement is None:
        return _SGLangPlacementPlan(mode="single_auto", total_gpus=total)
    if placement.total_gpus != total:
        raise RuntimeError(f"{context}: placement GPU count mismatch, need {total} GPUs, got {placement.total_gpus}")
    node_gpus = tuple((str(slice_.node_ip), int(slice_.gpu_count)) for slice_ in placement.slices)
    if len(node_gpus) == 1:
        return _SGLangPlacementPlan(mode="single_pinned", total_gpus=total, node_gpus=node_gpus)
    return _SGLangPlacementPlan(mode="multinode", total_gpus=total, node_gpus=node_gpus)


def _alive_ray_node_id_by_ip() -> dict[str, str]:
    return {
        str(n.get("NodeManagerAddress") or ""): str(n.get("NodeID") or "")
        for n in ray.nodes()
        if n.get("Alive") and n.get("NodeManagerAddress") and n.get("NodeID")
    }


def _node_affinity_for_ip(node_ip: str, *, actor_name: str) -> NodeAffinitySchedulingStrategy:
    node_map = _alive_ray_node_id_by_ip()
    node_id = node_map.get(str(node_ip))
    if not node_id:
        raise RuntimeError(
            f"Requested pinned_node_ip={node_ip} for actor={actor_name} "
            "but no alive Ray node has that IP"
        )
    return NodeAffinitySchedulingStrategy(node_id, soft=False)


def _rank_actor_name(base_actor_name: str, rank: int) -> str:
    if int(rank) == 0:
        return str(base_actor_name)
    return f"{base_actor_name}_rank{int(rank)}"


def _sglang_actor_lora_defaults_for_config(config: Any) -> dict[str, int | bool | tuple[str, ...]]:
    enable_lora = bool(server_config.enable_multi_lora)
    configured_max_loras = getattr(config, "max_loras", None)
    if configured_max_loras is not None and int(configured_max_loras) <= 0:
        enable_lora = False
    max_lora_rank = (
        getattr(config, "sglang_max_lora_rank", None)
        or getattr(config, "max_lora_rank", None)
        or server_config.max_lora_rank
    )
    configured_sglang_max_loaded_loras = getattr(config, "sglang_max_loaded_loras", None)
    max_loaded_loras = configured_sglang_max_loaded_loras
    if max_loaded_loras is None:
        max_loaded_loras = configured_max_loras
    if max_loaded_loras is None:
        max_loaded_loras = min(8, max(1, int(server_config.max_loras)))
    elif int(max_loaded_loras) <= 0:
        max_loaded_loras = 1
    lora_target_modules = getattr(config, "sglang_lora_target_modules", None)
    out: dict[str, int | bool | tuple[str, ...]] = {
        "default_enable_lora": enable_lora,
        "default_max_lora_rank": max(1, int(max_lora_rank)),
        "default_max_loaded_loras": max(1, int(max_loaded_loras)),
    }
    if configured_sglang_max_loaded_loras is not None:
        out["default_max_loras_per_batch"] = max(1, int(max_loaded_loras))
    if lora_target_modules:
        out["default_lora_target_modules"] = tuple(str(item) for item in lora_target_modules if str(item).strip())
        if not out["default_lora_target_modules"]:
            out.pop("default_lora_target_modules", None)
    return out


def _sglang_actor_memory_defaults_for_config(config: Any) -> dict[str, float]:
    gpu_memory_utilization = getattr(config, "gpu_memory_utilization", None)
    if gpu_memory_utilization is None:
        return {"default_mem_fraction_static": 0.4}
    try:
        value = float(gpu_memory_utilization)
    except (TypeError, ValueError):
        return {"default_mem_fraction_static": 0.4}
    if value <= 0:
        return {"default_mem_fraction_static": 0.4}
    return {"default_mem_fraction_static": min(value, 1.0)}


def _is_capacity_block_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "pinned node capacity check failed" in msg
        or "insufficient gpu" in msg
        or "insufficient pinned gpu" in msg
    )


def _sglang_reclaim_training_placement_enabled() -> bool:
    raw = os.environ.get("MINT_SGLANG_RECLAIM_TRAINING_PLACEMENT_ON_LAUNCH")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _training_actor_and_pg_names_for_model(
    model_name: str,
    *,
    namespace: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    actor_to_pg_names: dict[str, list[str]] = {}

    def add_entry(actor_name: str | None, *pg_names: str | None) -> None:
        if not actor_name:
            return
        existing = actor_to_pg_names.setdefault(actor_name, [])
        for pg_name in pg_names:
            if pg_name and pg_name not in existing:
                existing.append(pg_name)

    try:
        from .training.megatron.megatron_distributed import (
            _make_megatron_actor_name,
            _make_megatron_pg_name,
        )

        actor_name = _make_megatron_actor_name(model_name)
        add_entry(
            actor_name,
            f"{actor_name}_pg",
            _make_megatron_pg_name(model_name, namespace=namespace),
        )
    except Exception:
        pass
    try:
        from .training.bumblebee.bumblebee_distributed import (
            _make_bumblebee_actor_name,
            _make_bumblebee_pg_name,
        )

        actor_name = _make_bumblebee_actor_name(model_name)
        add_entry(
            actor_name,
            f"{actor_name}_pg",
            _make_bumblebee_pg_name(model_name, namespace=namespace),
        )
    except Exception:
        pass
    return tuple((actor_name, tuple(pg_names)) for actor_name, pg_names in actor_to_pg_names.items())


def _reclaim_same_model_training_resources_for_sglang(
    *,
    model_name: str,
    context: str,
    namespace: str = PERSISTENT_NAMESPACE,
) -> dict[str, list[str]]:
    if not _sglang_reclaim_training_placement_enabled():
        return {"killed_actor_names": [], "removed_placement_group_names": []}

    killed_actor_names: list[str] = []
    removed_pg_names: list[str] = []
    for actor_name, pg_names in _training_actor_and_pg_names_for_model(model_name, namespace=namespace):
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            actor = None
        except Exception as exc:
            logger.warning(
                "SGLang launch could not inspect same-model training actor actor=%s namespace=%s context=%s error_type=%s error=%s",
                actor_name,
                namespace,
                context,
                type(exc).__name__,
                exc,
            )
            actor = None
        if actor is not None:
            try:
                ray_kill.kill(
                    actor,
                    reason="sglang_launch_same_model_training_preempt",
                    actor_name=actor_name,
                    namespace=namespace,
                    no_restart=True,
                    verify_absent=True,
                )
                killed_actor_names.append(actor_name)
            except Exception as exc:
                logger.warning(
                    "SGLang launch failed to preempt same-model training actor actor=%s namespace=%s context=%s error_type=%s error=%s",
                    actor_name,
                    namespace,
                    context,
                    type(exc).__name__,
                    exc,
                )

        for pg_name in pg_names:
            try:
                if remove_named_placement_group(pg_name, namespace=namespace):
                    removed_pg_names.append(pg_name)
            except Exception as exc:
                logger.warning(
                    "SGLang launch failed to remove same-model training placement group actor=%s pg=%s namespace=%s context=%s error_type=%s error=%s",
                    actor_name,
                    pg_name,
                    namespace,
                    context,
                    type(exc).__name__,
                    exc,
                )

    if killed_actor_names or removed_pg_names:
        logger.warning(
            "SGLang launch reclaimed same-model training resources model=%s context=%s actors=%s placement_groups=%s",
            model_name,
            context,
            sorted(set(killed_actor_names)),
            sorted(set(removed_pg_names)),
        )
    return {
        "killed_actor_names": sorted(set(killed_actor_names)),
        "removed_placement_group_names": sorted(set(removed_pg_names)),
    }


def _assert_sglang_node_ip_capacity_with_training_reclaim(
    *,
    model_name: str,
    required_gpus_by_node_ip: dict[str, int],
    context: str,
) -> None:
    try:
        assert_node_ip_capacity(
            required_gpus_by_node_ip=required_gpus_by_node_ip,
            context=context,
        )
        return
    except Exception as exc:
        if not _is_capacity_block_error(exc):
            raise
        first_error = exc

    reclaimed = _reclaim_same_model_training_resources_for_sglang(
        model_name=model_name,
        context=context,
        namespace=PERSISTENT_NAMESPACE,
    )
    if not (reclaimed["killed_actor_names"] or reclaimed["removed_placement_group_names"]):
        raise first_error
    assert_node_ip_capacity(
        required_gpus_by_node_ip=required_gpus_by_node_ip,
        context=context,
    )


def _assert_single_node_sglang_schedulable(*, required_gpus: int, context: str) -> None:
    required = int(required_gpus)
    if required <= 0:
        raise RuntimeError(f"{context}: SGLang required_gpus must be >= 1, got {required_gpus!r}")
    alive_gpu_nodes: list[tuple[str, int]] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        total = int(float(resources.get("GPU", 0) or 0))
        if total <= 0:
            continue
        alive_gpu_nodes.append((str(node.get("NodeManagerAddress") or ""), total))
    if not alive_gpu_nodes:
        raise RuntimeError(f"{context}: no alive Ray GPU nodes found for single-node SGLang actor")
    max_node_gpus = max(total for _ip, total in alive_gpu_nodes)
    if required > max_node_gpus:
        raise RuntimeError(
            f"{context}: current SGLang backend launches one Ray actor and supports single-node TP only; "
            f"required_gpus={required} exceeds max_alive_node_gpus={max_node_gpus} "
            f"alive_gpu_nodes={alive_gpu_nodes}"
        )


class SGLangInferenceEngine:
    """Route-facing facade over a detached SGLang offline Engine actor."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str | None = None,
        actor_name: str | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ) -> None:
        self.model_name = str(model_name)
        self.model_path = str(model_path or _resolve_model_path(self.model_name))
        self.actor_name = str(actor_name or actor_name_for_sampling_base_model(self.model_name, backend="sglang"))
        self.config = get_model_config(self.model_name)
        self.tensor_parallel_size = int(tensor_parallel_size or self.config.inference_tp or 1)
        self.max_model_len = int(max_model_len or self.config.max_model_len)
        self.dtype = str(dtype)
        self.server: Any | None = None
        self._rank_servers: list[tuple[str, Any]] = []
        self._placement_plan: _SGLangPlacementPlan | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    def _actor_constructor_kwargs(
        self,
        *,
        actor_name: str,
        tp_size: int,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "actor_name": actor_name,
            "max_model_len": self.max_model_len,
            "tp_size": int(tp_size),
            "dtype": self.dtype,
            **_sglang_actor_lora_defaults_for_config(self.config),
            **_sglang_actor_memory_defaults_for_config(self.config),
            "engine_kwargs": dict(engine_kwargs or {}),
        }

    def _remote_actor_class(self, *, num_gpus: int, max_concurrency: int) -> Any:
        from .sglang_actor import SGLangEngineActor

        return ray.remote(num_gpus=int(num_gpus), max_concurrency=int(max_concurrency), max_restarts=0)(SGLangEngineActor)

    async def _launch_single_node_actor(self, *, plan: _SGLangPlacementPlan, max_concurrency: int) -> Any:
        total_gpus = int(plan.total_gpus)
        if plan.mode == "single_pinned":
            _assert_sglang_node_ip_capacity_with_training_reclaim(
                model_name=self.model_name,
                required_gpus_by_node_ip=plan.required_gpus_by_node_ip(),
                context=f"single_node_sglang model={self.model_name!r} actor={self.actor_name!r}_pin",
            )
        else:
            _assert_single_node_sglang_schedulable(
                required_gpus=total_gpus,
                context=f"single_node_sglang model={self.model_name!r} actor={self.actor_name!r}",
            )

        remote_cls = self._remote_actor_class(num_gpus=total_gpus, max_concurrency=max_concurrency)
        options: dict[str, Any] = {
            "name": self.actor_name,
            "namespace": PERSISTENT_NAMESPACE,
            "lifetime": "detached",
            "get_if_exists": True,
            "runtime_env": _runtime_env(),
        }
        if plan.mode == "single_pinned":
            pinned_node_ip = plan.node_ips[0]
            options["resources"] = {f"node:{pinned_node_ip}": 0.001}
            options["scheduling_strategy"] = _node_affinity_for_ip(pinned_node_ip, actor_name=self.actor_name)
        actor = remote_cls.options(**options).remote(
            **self._actor_constructor_kwargs(
                actor_name=self.actor_name,
                tp_size=self.tensor_parallel_size,
            )
        )
        timeout_s = _env_float(
            "MINT_SGLANG_ENGINE_READY_WAIT_S",
            _default_sglang_engine_ready_wait_s(self.config, total_gpus=total_gpus),
        )
        try:
            await async_get_ray_ref(actor.initialize.remote(), timeout_s=timeout_s)
        except Exception:
            try:
                ray_kill.kill(
                    actor,
                    reason="sglang_init_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    verify_absent=True,
                )
            except Exception:
                pass
            raise
        self._rank_servers = []
        return actor

    async def _launch_multinode_actor_group(self, *, plan: _SGLangPlacementPlan, max_concurrency: int) -> Any:
        if int(getattr(self.config, "inference_pp", 1) or 1) != 1 or int(getattr(self.config, "inference_dp", 1) or 1) != 1:
            raise RuntimeError(
                "Current SGLang multi-node backend supports TP-only execution; "
                f"got inference_tp={self.config.inference_tp}, "
                f"inference_pp={self.config.inference_pp}, inference_dp={self.config.inference_dp} "
                f"for model {self.model_name}"
            )
        if not plan.node_gpus:
            raise RuntimeError(f"SGLang multi-node placement is empty for model {self.model_name}")
        _assert_sglang_node_ip_capacity_with_training_reclaim(
            model_name=self.model_name,
            required_gpus_by_node_ip=plan.required_gpus_by_node_ip(),
            context=f"multinode_sglang model={self.model_name!r} actor={self.actor_name!r}",
        )
        rank_count = len(plan.node_gpus)
        if int(self.tensor_parallel_size) % rank_count != 0:
            raise RuntimeError(
                f"SGLang tensor_parallel_size={self.tensor_parallel_size} must be divisible by nnodes={rank_count}"
            )
        dist_port = _env_int("MINT_SGLANG_DIST_PORT", 29673)
        dist_init_addr = f"{plan.node_ips[0]}:{dist_port}"
        env_runtime = _runtime_env()
        env_vars = dict(env_runtime.get("env_vars") or {})
        env_vars["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        runtime_env = {**env_runtime, "env_vars": env_vars}

        launched: list[tuple[str, Any]] = []
        init_refs: list[Any] = []
        try:
            for rank, (node_ip, gpu_count) in enumerate(plan.node_gpus):
                rank_actor_name = _rank_actor_name(self.actor_name, rank)
                remote_cls = self._remote_actor_class(num_gpus=int(gpu_count), max_concurrency=max_concurrency)
                options = {
                    "name": rank_actor_name,
                    "namespace": PERSISTENT_NAMESPACE,
                    "lifetime": "detached",
                    "get_if_exists": True,
                    "runtime_env": runtime_env,
                    "resources": {f"node:{node_ip}": 0.001},
                    "scheduling_strategy": _node_affinity_for_ip(node_ip, actor_name=rank_actor_name),
                }
                actor = remote_cls.options(**options).remote(
                    **self._actor_constructor_kwargs(
                        actor_name=rank_actor_name,
                        tp_size=self.tensor_parallel_size,
                        engine_kwargs={
                            "nnodes": rank_count,
                            "node_rank": int(rank),
                            "dist_init_addr": dist_init_addr,
                        },
                    )
                )
                launched.append((rank_actor_name, actor))
                init_refs.append(actor.initialize.remote())

            timeout_s = _env_float(
                "MINT_SGLANG_ENGINE_READY_WAIT_S",
                _default_sglang_engine_ready_wait_s(self.config, total_gpus=self.tensor_parallel_size),
            )
            await asyncio.gather(
                *(async_get_ray_ref(ref, timeout_s=timeout_s) for ref in init_refs)
            )
        except Exception:
            for rank_actor_name, actor in launched:
                try:
                    ray_kill.kill(
                        actor,
                        reason="sglang_multinode_init_failed",
                        actor_name=rank_actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        verify_absent=True,
                    )
                except Exception:
                    pass
            self._rank_servers = []
            raise

        self._rank_servers = launched[1:]
        return launched[0][1]

    async def _try_reuse_existing_actor_group(self, *, plan: _SGLangPlacementPlan) -> bool | None:
        if not plan.is_multinode:
            try:
                existing = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                ready = await async_get_ray_ref(existing.is_ready.remote(), timeout_s=10)
                if ready:
                    self.server = existing
                    self._rank_servers = []
                    self._initialized = True
                    self._publish_inventory(existing)
                    return True
                logger.warning("SGLang actor %s exists but is not ready; recreating", self.actor_name)
                _invalidate_model_session_loras(self.model_name)
                ray_kill.kill(
                    existing,
                    reason="sglang_engine_not_ready",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    verify_absent=True,
                )
                return False
            except ValueError:
                return None
            except ray.exceptions.GetTimeoutError:
                logger.warning("SGLang actor %s readiness timed out; assuming busy and reusing", self.actor_name)
                self.server = existing
                self._rank_servers = []
                self._initialized = True
                self._publish_inventory(existing)
                return True
            except ray.exceptions.RayActorError:
                logger.warning("SGLang actor %s is dead; recreating", self.actor_name)
                _invalidate_model_session_loras(self.model_name)
                return False
            except Exception as e:
                logger.warning("SGLang actor %s readiness probe failed: %s: %s", self.actor_name, type(e).__name__, e)
                return False

        existing_group: list[tuple[str, Any]] = []
        missing = False
        for rank in range(len(plan.node_gpus)):
            rank_actor_name = _rank_actor_name(self.actor_name, rank)
            try:
                actor = ray.get_actor(rank_actor_name, namespace=PERSISTENT_NAMESPACE)
            except ValueError:
                missing = True
                continue
            existing_group.append((rank_actor_name, actor))
        if missing:
            for rank_actor_name, actor in existing_group:
                try:
                    ray_kill.kill(
                        actor,
                        reason="sglang_multinode_partial_group",
                        actor_name=rank_actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        verify_absent=True,
                    )
                except Exception:
                    pass
            return None
        try:
            ready = await asyncio.gather(
                *(async_get_ray_ref(actor.is_ready.remote(), timeout_s=10) for _name, actor in existing_group)
            )
        except ray.exceptions.GetTimeoutError:
            logger.warning("SGLang multi-node actor group %s readiness timed out; assuming busy and reusing", self.actor_name)
            self.server = existing_group[0][1]
            self._rank_servers = existing_group[1:]
            self._initialized = True
            self._publish_inventory(self.server)
            return True
        except ray.exceptions.RayActorError:
            _invalidate_model_session_loras(self.model_name)
            ready = []
        if ready and all(bool(item) for item in ready):
            self.server = existing_group[0][1]
            self._rank_servers = existing_group[1:]
            self._initialized = True
            self._publish_inventory(self.server)
            return True
        logger.warning("SGLang multi-node actor group %s exists but is not fully ready; recreating", self.actor_name)
        _invalidate_model_session_loras(self.model_name)
        for rank_actor_name, actor in existing_group:
            try:
                ray_kill.kill(
                    actor,
                    reason="sglang_multinode_group_not_ready",
                    actor_name=rank_actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    verify_absent=True,
                )
            except Exception:
                pass
        return False

    async def initialize(self) -> None:
        if self._initialized and self.server is not None:
            return
        async with self._init_lock:
            if self._initialized and self.server is not None:
                return
            if not ray.is_initialized():
                init_ray(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

            if int(getattr(self.config, "inference_pp", 1) or 1) != 1 or int(getattr(self.config, "inference_dp", 1) or 1) != 1:
                raise RuntimeError(
                    "Current SGLang backend supports single-node TP only; "
                    f"got inference_tp={self.config.inference_tp}, "
                    f"inference_pp={self.config.inference_pp}, inference_dp={self.config.inference_dp} "
                    f"for model {self.model_name}"
                )
            total_gpus = int(self.tensor_parallel_size)
            if total_gpus < 1:
                raise RuntimeError(f"SGLang tensor_parallel_size must be >= 1 for model {self.model_name}")
            plan = _sglang_placement_plan_for_model(
                model_name=self.model_name,
                actor_name=self.actor_name,
                total_gpus=total_gpus,
            )
            self._placement_plan = plan
            reused = await self._try_reuse_existing_actor_group(plan=plan)
            if reused:
                return

            # Default >1 so the actor serves concurrent sampling requests and
            # SGLang batches them internally. The actor guards adapter mutations
            # with a writer-preferring RW lock, so concurrent generation is safe
            # (generation takes the read lock; add/remove/init/shutdown the write
            # lock). Override with MINT_SGLANG_ACTOR_MAX_CONCURRENCY.
            max_concurrency = max(1, _env_int("MINT_SGLANG_ACTOR_MAX_CONCURRENCY", 64))
            try:
                if plan.is_multinode:
                    self.server = await self._launch_multinode_actor_group(plan=plan, max_concurrency=max_concurrency)
                else:
                    self.server = await self._launch_single_node_actor(plan=plan, max_concurrency=max_concurrency)
            except Exception:
                self.server = None
                raise
            self._initialized = True
            self._publish_inventory(self.server)

    def _publish_inventory(self, actor_handle: Any) -> None:
        plan = self._placement_plan
        metadata: dict[str, Any] = {
            "serving_backend": "sglang",
            "model_path": self.model_path,
            "max_model_len": self.max_model_len,
        }
        if plan is not None:
            metadata.update(
                {
                    "placement_mode": plan.mode,
                    "nnodes": len(plan.node_gpus) if plan.is_multinode else 1,
                    "node_ips": plan.node_ips,
                    "rank_actor_names": [_rank_actor_name(self.actor_name, rank) for rank in range(len(plan.node_gpus))]
                    if plan.is_multinode
                    else [self.actor_name],
                }
            )
        publish_backend_model_actor(
            BackendModelActorLaunch(
                actor_name=self.actor_name,
                actor_type=ActorType.SGLANG,
                num_gpus=int(self.tensor_parallel_size),
                actor_handle=actor_handle,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_name,
                node_id=_get_actor_node_id(actor_handle),
                metadata=metadata,
            )
        )

    async def add_lora_for_session_from_path(
        self,
        sampling_session_id: str,
        lora_path: str,
    ) -> int:
        self.validate_lora_adapter_supported(lora_path)
        await self.initialize()
        if self.server is None:
            raise RuntimeError(f"SGLang actor {self.actor_name} is not available")

        server = self.server
        async def _dispatch_load() -> dict[str, Any]:
            try:
                ref = server.add_lora_for_session_from_path.remote(
                    sampling_session_id=str(sampling_session_id),
                    lora_path=str(lora_path),
                )
                return await ray_get_with_model_actor_supervisor_keepalive(
                    ref,
                    actor_name=self.actor_name,
                    request_id=str(sampling_session_id),
                )
            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                _invalidate_model_session_loras(self.model_name)
                self._initialized = False
                self.server = None
                await self.initialize()
                if self.server is None:
                    raise RuntimeError(f"SGLang actor {self.actor_name} is not available after reinitialization")
                ref = self.server.add_lora_for_session_from_path.remote(
                    sampling_session_id=str(sampling_session_id),
                    lora_path=str(lora_path),
                )
                return await ray_get_with_model_actor_supervisor_keepalive(
                    ref,
                    actor_name=self.actor_name,
                    request_id=str(sampling_session_id),
                )

        result = await run_async_with_otel_span(
            "sampling.sglang.add_lora_from_path",
            _dispatch_load,
            component="sglang_engine",
            op="sampling.add_lora_from_path",
            request_id=str(sampling_session_id),
            attributes={
                "actor_name": self.actor_name,
                "base_model": self.model_name,
                "sampling_session_id": str(sampling_session_id),
                "adapter_path": str(lora_path),
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"SGLang add_lora_for_session_from_path returned {type(result).__name__}, expected dict")
        lora_int_id = result.get("lora_int_id")
        if lora_int_id is None:
            raise RuntimeError(f"SGLang add_lora_for_session_from_path response missing lora_int_id: {result!r}")
        return int(lora_int_id)

    def validate_lora_adapter_supported(self, lora_path: str) -> None:
        from .sglang_capabilities import validate_sglang_lora_adapter_supported

        validate_sglang_lora_adapter_supported(
            model_name=self.model_name,
            model_path=self.model_path,
            adapter_path=str(lora_path),
        )

    async def _best_effort_abort(self, *, request_id: str, op: str, reason: str) -> None:
        try:
            aborted = await self.abort(request_id)
            logger.warning(
                "SGLang abort_request backend=sglang actor=%s model=%s op=%s request_id=%s reason=%s aborted=%s",
                self.actor_name,
                self.model_name,
                op,
                request_id,
                reason,
                bool(aborted),
            )
        except Exception as e:
            logger.warning(
                "SGLang abort_request failed backend=sglang actor=%s model=%s op=%s request_id=%s reason=%s "
                "error_type=%s error=%s",
                self.actor_name,
                self.model_name,
                op,
                request_id,
                reason,
                type(e).__name__,
                e,
            )

    async def _await_actor_ref(
        self,
        ref: Any,
        *,
        op: str,
        request_id: str,
        prompt_tokens: int | None = None,
        max_tokens: int | None = None,
        abort_on_timeout: bool = True,
    ) -> Any:
        timeout_s, timeout_config = _sglang_remote_timeout_config(op)
        started = time.perf_counter()
        try:
            result = await ray_get_with_model_actor_supervisor_keepalive(
                ref,
                actor_name=self.actor_name,
                request_id=request_id,
                timeout_s=timeout_s,
            )
        except (asyncio.TimeoutError, ray.exceptions.GetTimeoutError) as e:
            elapsed_s = max(0.0, time.perf_counter() - started)
            if abort_on_timeout:
                await self._best_effort_abort(request_id=request_id, op=op, reason="timeout")
            logger.warning(
                "SGLang request timed out backend=sglang actor=%s model=%s op=%s request_id=%s "
                "elapsed_s=%.3f timeout_s=%s timeout_config=%s prompt_tokens=%s max_tokens=%s",
                self.actor_name,
                self.model_name,
                op,
                request_id,
                elapsed_s,
                timeout_s,
                timeout_config,
                prompt_tokens,
                max_tokens,
            )
            if timeout_s is not None:
                raise RuntimeError(
                    f"SGLang {op} timed out after {timeout_s:.1f}s "
                    f"for request_id={request_id} actor={self.actor_name}"
                ) from e
            raise
        except asyncio.CancelledError:
            await self._best_effort_abort(request_id=request_id, op=op, reason="cancelled")
            raise

        elapsed_s = max(0.0, time.perf_counter() - started)
        slow_s = _sglang_slow_request_log_s()
        if slow_s is not None and elapsed_s >= slow_s:
            logger.warning(
                "SGLang request slow backend=sglang actor=%s model=%s op=%s request_id=%s "
                "elapsed_s=%.3f slow_threshold_s=%.3f prompt_tokens=%s max_tokens=%s",
                self.actor_name,
                self.model_name,
                op,
                request_id,
                elapsed_s,
                slow_s,
                prompt_tokens,
                max_tokens,
            )
        return result

    async def generate(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
        prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> GenerateResult:
        if len(prompt_ids) + int(max_tokens) > self.max_model_len:
            raise ValueError(
                f"Prompt+max_tokens length {len(prompt_ids) + int(max_tokens)} exceeds "
                f"max_model_len {self.max_model_len} for model {self.model_name}"
            )
        await self.initialize()
        if self.server is None:
            raise RuntimeError(f"SGLang actor {self.actor_name} is not available")
        server = self.server
        async def _dispatch_generate() -> dict[str, Any]:
            ref = server.generate_base.remote(
                prompt_ids=[int(x) for x in prompt_ids],
                request_id=str(request_id),
                max_tokens=int(max_tokens),
                stop=stop,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                logprobs=bool(logprobs),
                sampling_session_id=sampling_session_id,
                prompt_logprobs=bool(prompt_logprobs),
                topk_prompt_logprobs=int(topk_prompt_logprobs),
            )
            return await self._await_actor_ref(
                ref,
                op="generate",
                request_id=request_id,
                prompt_tokens=len(prompt_ids),
                max_tokens=int(max_tokens),
            )

        started = time.perf_counter()
        result = await run_async_with_otel_span(
            "sampling.sglang.generate",
            _dispatch_generate,
            component="sglang_engine",
            op="sampling.generate",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "base_model": self.model_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
                "max_tokens": int(max_tokens),
            },
        )
        return GenerateResult(
            token_ids=[int(x) for x in result["token_ids"]],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
            routed_experts=None,
            timing_total_s=_float_or_none(result.get("_timing_total_s")) or max(0.0, time.perf_counter() - started),
            timing_first_tok_s=_float_or_none(result.get("_timing_first_tok_s")),
        )

    async def generate_many(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        num_samples: int,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> list[GenerateResult]:
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1 (got {num_samples})")
        # Dispatch all samples concurrently. The SGLang actor accepts concurrent
        # generate calls (max_concurrency > 1 + RW lock) and batches them
        # internally, so this issues N overlapping Ray calls rather than N
        # sequential round-trips. asyncio.gather preserves input order.
        results: list[GenerateResult] = await asyncio.gather(
            *(
                self.generate(
                    sampling_session_id=sampling_session_id,
                    prompt_ids=prompt_ids,
                    request_id=f"{request_id}_{i}",
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    logprobs=logprobs,
                )
                for i in range(int(num_samples))
            )
        )
        return results

    async def compute_logprobs(self, *args: Any, **kwargs: Any) -> list[float | None]:
        sampling_session_id = kwargs.get("sampling_session_id")
        prompt_ids = [int(x) for x in kwargs.get("prompt_ids", [])]
        request_id = str(kwargs.get("request_id", "sglang_prompt_logprobs"))
        if len(prompt_ids) + 1 > self.max_model_len:
            raise ValueError(
                f"Prompt+max_tokens length {len(prompt_ids) + 1} exceeds "
                f"max_model_len {self.max_model_len} for model {self.model_name}"
            )
        await self.initialize()
        if self.server is None:
            raise RuntimeError(f"SGLang actor {self.actor_name} is not available")

        server = self.server
        async def _dispatch_compute_logprobs() -> list[float | None]:
            ref = server.compute_prompt_logprobs.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                sampling_session_id=sampling_session_id,
            )
            result = await self._await_actor_ref(
                ref,
                op="compute_prompt_logprobs",
                request_id=request_id,
                prompt_tokens=len(prompt_ids),
                max_tokens=1,
            )
            return list(result)

        return await run_async_with_otel_span(
            "sampling.sglang.compute_prompt_logprobs",
            _dispatch_compute_logprobs,
            component="sglang_engine",
            op="sampling.compute_prompt_logprobs",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "base_model": self.model_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
            },
        )

    async def compute_topk(self, *args: Any, **kwargs: Any) -> list[list[tuple[int, float]] | None]:
        sampling_session_id = kwargs.get("sampling_session_id")
        prompt_ids = [int(x) for x in kwargs.get("prompt_ids", [])]
        request_id = str(kwargs.get("request_id", "sglang_prompt_topk"))
        kk = int(kwargs.get("k", 10))
        if kk <= 0:
            return [None] * len(prompt_ids)
        if len(prompt_ids) + 1 > self.max_model_len:
            raise ValueError(
                f"Prompt+max_tokens length {len(prompt_ids) + 1} exceeds "
                f"max_model_len {self.max_model_len} for model {self.model_name}"
            )
        await self.initialize()
        if self.server is None:
            raise RuntimeError(f"SGLang actor {self.actor_name} is not available")

        server = self.server
        async def _dispatch_compute_topk() -> list[list[tuple[int, float]] | None]:
            ref = server.compute_prompt_topk.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                k=kk,
                sampling_session_id=sampling_session_id,
            )
            result = await self._await_actor_ref(
                ref,
                op="compute_prompt_topk",
                request_id=request_id,
                prompt_tokens=len(prompt_ids),
                max_tokens=1,
            )
            return list(result)

        return await run_async_with_otel_span(
            "sampling.sglang.compute_prompt_topk",
            _dispatch_compute_topk,
            component="sglang_engine",
            op="sampling.compute_prompt_topk",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "base_model": self.model_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
                "topk": kk,
            },
        )

    async def abort(self, request_id: str) -> bool:
        if self.server is None:
            return False
        try:
            return bool(await async_get_ray_ref(self.server.abort.remote(str(request_id)), timeout_s=5))
        except Exception:
            return False

    async def abort_request(self, request_id: str) -> bool:
        return await self.abort(request_id)

    async def remove_session(self, sampling_session_id: str) -> bool:
        if self.server is None and not self._initialized:
            return False
        await self.initialize()
        if self.server is None:
            return False
        result = await async_get_ray_ref(
            self.server.remove_session.remote(str(sampling_session_id)),
            timeout_s=30,
        )
        return bool(isinstance(result, dict) and result.get("removed"))

    def invalidate_cached_handles(self) -> None:
        """Drop cached actor handles without killing the actors.

        For callers that have already terminated the underlying SGLang actors
        out-of-band (e.g. Megatron reclaiming GPU capacity by force-killing the
        same-model SGLang actor group) and only need this cached engine to forget
        its now-dead handles so the next use re-launches. Unlike shutdown(), this
        does NOT issue actor shutdown/kill RPCs.
        """
        self.server = None
        self._rank_servers = []
        self._initialized = False

    async def shutdown(self) -> None:
        rank_servers = list(self._rank_servers)
        if self.server is not None:
            try:
                await async_get_ray_ref(self.server.shutdown.remote(), timeout_s=30)
            finally:
                try:
                    ray_kill.kill(
                        self.server,
                        reason="sglang_shutdown",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
        for rank_actor_name, actor in rank_servers:
            try:
                await async_get_ray_ref(actor.shutdown.remote(), timeout_s=30)
            except Exception:
                pass
            try:
                ray_kill.kill(
                    actor,
                    reason="sglang_shutdown",
                    actor_name=rank_actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                )
            except Exception:
                pass
        self.server = None
        self._rank_servers = []
        self._initialized = False


async def _manager_lock(model_name: str) -> asyncio.Lock:
    async with _MANAGER_LOCKS_GUARD:
        lock = _MANAGER_LOCKS.get(model_name)
        if lock is None:
            lock = asyncio.Lock()
            _MANAGER_LOCKS[model_name] = lock
        return lock


async def get_sglang_engine_for_model(model_name: str) -> SGLangInferenceEngine:
    model = str(model_name).strip()
    if not model:
        raise ValueError("model_name is required")
    lock = await _manager_lock(model)
    async with lock:
        engine = _MANAGERS.get(model)
        if engine is None:
            engine = SGLangInferenceEngine(model_name=model)
            _MANAGERS[model] = engine
        return engine


def get_cached_sglang_engine_for_model(model_name: str) -> SGLangInferenceEngine | None:
    return _MANAGERS.get(str(model_name).strip())


async def remove_sglang_session_from_existing_actor(
    model_name: str,
    sampling_session_id: str,
    *,
    timeout_s: float = 30.0,
) -> bool:
    """Best-effort session cleanup for a detached SGLang actor without creating one."""
    model = str(model_name).strip()
    session_id = str(sampling_session_id).strip()
    if not model or not session_id or not ray.is_initialized():
        return False

    actor_name = actor_name_for_sampling_base_model(model, backend="sglang")
    try:
        actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
    except Exception:
        return False

    try:
        result = await async_get_ray_ref(
            actor.remove_session.remote(session_id),
            timeout_s=float(timeout_s),
        )
    except Exception as e:
        logger.warning(
            "Failed to remove SGLang session from existing actor model=%s session=%s actor=%s: %s: %s",
            model,
            session_id,
            actor_name,
            type(e).__name__,
            e,
        )
        return False
    return bool(isinstance(result, dict) and result.get("removed"))
