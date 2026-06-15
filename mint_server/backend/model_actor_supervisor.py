from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from ..config import (
    PFS_PYTHONPATH,
    actor_runtime_env,
    config as server_config,
    otel_env_vars,
    preferred_control_plane_resources,
)
from ..runtime_env import env_nonempty
from ..server_info import _git_sha
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref
from .cluster_placement_controller import (
    ClusterPlacementController,
    PlacementGroupCreateStatus,
    PlacementReconcileRequest,
    PlacementReconcileResult,
)
from .engine_liveness import EngineLivenessPush
from .model_actor_inventory import (
    ActorEntry,
    ActorType,
    ModelActorInventory,
    ModelActorSupervisorStaleError,
    _ModelActorInventoryState,
    actor_observability_metadata,
    async_actor_observability_metadata,
)
from .model_actor_launchers import (
    ModelActorLauncherRegistry,
    default_model_actor_launcher_registry,
)
from .model_actor_placement import (
    _default_gpu_actor_lister,
    _is_mint_gpu_actor_name,
    model_actor_placement_reconciler,
)
from .model_work_scheduler import ModelReplicaRegistration, ModelWorkSchedulerClient, model_work_scheduler
from .node_metrics_daemon import (
    NodeMetricsDaemonSpec,
    get_or_create_node_metrics_collector_actor,
    node_metrics_actor_name,
)
from .supervisor_state_store import (
    SupervisorMemoryStateStore,
    SupervisorSQLiteStateStore,
    SupervisorStateOwnerConflictError,
    create_supervisor_state_store,
)
from .topology import TopologyManager, is_ip_address, load_topology_config_from_env

logger = logging.getLogger(__name__)

__all__ = [
    "ActorEntry",
    "ActorType",
    "ModelActorSpec",
    "ModelActorSupervisor",
    "ModelActorSupervisorClient",
    "ModelActorSupervisorCore",
    "ModelActorSupervisorStaleError",
    "ModelActorSupervisorUnavailableError",
    "_ModelActorInventoryState",
    "actor_observability_metadata",
    "async_actor_observability_metadata",
    "default_model_actor_name",
    "domain_key_for_internal_runtime",
    "domain_key_for_training_base_model",
    "domain_key_for_vllm_base_model",
    "get_model_actor_supervisor",
    "ensure_started",
    "async_ensure_started",
    "model_actor_supervisor",
]

MODEL_ACTOR_SUPERVISOR_ACTOR_NAME = "mint_model_actor_supervisor"
CURRENT_CODE_IDENTITY = os.environ.get("MINT_GIT_SHA") or _git_sha()


class ModelActorSupervisorUnavailableError(RuntimeError):
    pass


class ModelActorSupervisorCodeIdentityMismatchError(RuntimeError):
    pass


def _ray_namespace() -> str:
    value = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if value:
        return value
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _node_metrics_enabled_by_default() -> bool:
    raw = str(os.environ.get("MINT_NODE_METRICS_DAEMON_ENABLED") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _reconcile_interval_s_from_env() -> float:
    raw = str(os.environ.get("MINT_ACTOR_RECONCILE_INTERVAL_S") or "5").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "[model_actor_supervisor] invalid MINT_ACTOR_RECONCILE_INTERVAL_S=%r; using 5s",
            raw,
        )
        return 5.0
    return max(0.1, value)


def _adopt_surviving_gpu_actors_enabled() -> bool:
    raw = str(os.environ.get("MINT_SUPERVISOR_ADOPT_SURVIVING_GPU_ACTORS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _otel_metric_attrs() -> dict[str, str]:
    attrs = {
        "deployment.env": os.getenv("MINT_DEPLOYMENT_ENV", "").strip(),
        "mint.cluster_id": os.getenv("MINT_CLUSTER_ID", "").strip(),
        "ray_namespace": _ray_namespace(),
    }
    return {key: value for key, value in attrs.items() if value}


def _prom_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _actor_workload(actor_type: object) -> str:
    return "sample" if str(actor_type or "").strip().lower() == "vllm" else "train"


def _model_actor_inventory_gpu_bindings(rec: dict[str, object]) -> list[dict[str, str]]:
    metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    actor_name = str(rec.get("actor_name") or "unknown")
    workload = _actor_workload(rec.get("actor_type"))

    bindings = metadata.get("gpu_bindings") if isinstance(metadata, dict) else None
    if isinstance(bindings, list):
        out: list[dict[str, str]] = []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            gpu_uuid = binding.get("gpu_uuid")
            if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
                continue
            out.append(
                {
                    "actor_name": actor_name,
                    "workload": workload,
                    "hostname": str(binding.get("hostname") or metadata.get("hostname") or "unknown"),
                    "gpu_uuid": gpu_uuid.strip(),
                }
            )
        if out:
            return out

    return []


def _model_actor_inventory_gpu_binding_missing_uuid(rec: dict[str, object]) -> list[dict[str, str]]:
    metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    actor_name = str(rec.get("actor_name") or "unknown")
    workload = _actor_workload(rec.get("actor_type"))
    hostname = str(metadata.get("hostname") or "unknown") if isinstance(metadata, dict) else "unknown"

    missing = 0
    bindings = metadata.get("gpu_bindings") if isinstance(metadata, dict) else None
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            gpu_uuid = binding.get("gpu_uuid")
            if isinstance(gpu_uuid, str) and gpu_uuid.strip():
                continue
            if binding.get("gpu_index") is not None:
                missing += 1

    gpu_indices = metadata.get("gpu_indices") if isinstance(metadata, dict) else None
    if not missing and isinstance(gpu_indices, list):
        missing = len([value for value in gpu_indices if value is not None])

    return [
        {
            "actor_name": actor_name,
            "workload": workload,
            "hostname": hostname,
            "missing_count": str(missing),
        }
    ] if missing > 0 else []


def _rss_state_for_record(rec: dict[str, object]) -> str:
    rss_state = str(rec.get("rss_cache_state") or "").strip().lower()
    if rss_state in {"fresh", "stale", "unknown"}:
        return rss_state
    if _prom_number(rec.get("rss_bytes")) is not None:
        return "fresh"
    if rec.get("rss_sample_age_s") is not None or rec.get("rss_sample_source") is not None:
        return "stale"
    return "unknown"


def _inventory_otel_gauge_callbacks(supervisor: "ModelActorSupervisorCore", observation_cls, attrs_fn) -> Iterable[tuple[str, Callable, str | None]]:
    def _inventory_records() -> list[dict[str, Any]]:
        try:
            return list(supervisor.cached_snapshot(refresh_metadata=False))
        except Exception:
            return []

    def _actor_metric(field: str):
        def _callback(_options):
            observations = []
            for rec in _inventory_records():
                value = _prom_number(rec.get(field))
                if value is None:
                    continue
                observations.append(
                    observation_cls(
                        value,
                        attrs_fn(
                            actor_type=rec.get("actor_type") or "unknown",
                            model=rec.get("base_model") or "unknown",
                            actor_name=rec.get("actor_name") or "unknown",
                        ),
                    )
                )
            return observations

        return _callback

    def _actor_rss_cache_state(_options):
        return [
            observation_cls(
                1.0,
                attrs_fn(
                    actor_type=rec.get("actor_type") or "unknown",
                    model=rec.get("base_model") or "unknown",
                    actor_name=rec.get("actor_name") or "unknown",
                    state=_rss_state_for_record(rec),
                ),
            )
            for rec in _inventory_records()
        ]

    def _actor_gpu_binding(_options):
        observations = []
        for rec in _inventory_records():
            for binding in _model_actor_inventory_gpu_bindings(rec):
                observations.append(observation_cls(1.0, attrs_fn(**binding)))
        return observations

    def _actor_gpu_binding_missing_uuid(_options):
        observations = []
        for rec in _inventory_records():
            for binding in _model_actor_inventory_gpu_binding_missing_uuid(rec):
                missing_count = float(binding.pop("missing_count"))
                observations.append(observation_cls(missing_count, attrs_fn(**binding)))
        return observations

    def _grouped_records() -> dict[tuple[str, str], dict[str, float]]:
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for rec in _inventory_records():
            actor_type = str(rec.get("actor_type") or "unknown")
            model = str(rec.get("base_model") or "unknown")
            rss_state = _rss_state_for_record(rec)
            bucket = grouped.setdefault(
                (actor_type, model),
                {
                    "count": 0.0,
                    "rss_sum": 0.0,
                    "rss_count": 0.0,
                    "rss_fresh": 0.0,
                    "rss_stale": 0.0,
                    "rss_unknown": 0.0,
                    "max_idle": 0.0,
                    "max_age": 0.0,
                },
            )
            bucket["count"] += 1.0
            bucket[f"rss_{rss_state}"] += 1.0
            idle = _prom_number(rec.get("idle_time"))
            if idle is not None and idle > bucket["max_idle"]:
                bucket["max_idle"] = idle
            age = _prom_number(rec.get("age"))
            if age is not None and age > bucket["max_age"]:
                bucket["max_age"] = age
            rss = _prom_number(rec.get("rss_bytes"))
            if rss is not None:
                bucket["rss_sum"] += rss
                bucket["rss_count"] += 1.0
        return grouped

    def _group_metric(field: str, *, require_rss_count: bool = False):
        def _callback(_options):
            observations = []
            for (actor_type, model), rec in sorted(_grouped_records().items()):
                if require_rss_count and rec.get("rss_count", 0.0) <= 0.0:
                    continue
                value = _prom_number(rec.get(field))
                if value is None:
                    continue
                observations.append(observation_cls(value, attrs_fn(actor_type=actor_type, model=model)))
            return observations

        return _callback

    def _group_rss_cache_samples(_options):
        observations = []
        for (actor_type, model), rec in sorted(_grouped_records().items()):
            for state in ("fresh", "stale", "unknown"):
                value = _prom_number(rec.get(f"rss_{state}"))
                if value is None or value <= 0.0:
                    continue
                observations.append(
                    observation_cls(value, attrs_fn(actor_type=actor_type, model=model, state=state))
                )
        return observations

    def _metadata_cache_metric(field: str):
        def _callback(_options):
            observations = []
            for row in supervisor.metadata_cache_metrics_snapshot():
                value = _prom_number(row.get(field))
                if value is None:
                    continue
                observations.append(
                    observation_cls(value, attrs_fn(actor_type=row.get("actor_type") or "unknown"))
                )
            return observations

        return _callback

    return (
        ("mint_model_actor_inventory_actor_idle_time_s", _actor_metric("idle_time"), "s"),
        ("mint_model_actor_inventory_actor_age_s", _actor_metric("age"), "s"),
        ("mint_model_actor_inventory_actor_rss_bytes", _actor_metric("rss_bytes"), "By"),
        ("mint_model_actor_inventory_actor_rss_sample_age_s", _actor_metric("rss_sample_age_s"), "s"),
        ("mint_model_actor_inventory_actor_rss_cache_state", _actor_rss_cache_state, None),
        ("mint_model_actor_inventory_actor_gpu_binding", _actor_gpu_binding, None),
        ("mint_model_actor_inventory_actor_gpu_binding_missing_uuid", _actor_gpu_binding_missing_uuid, None),
        ("mint_model_actor_inventory_actors", _group_metric("count"), None),
        ("mint_model_actor_inventory_group_oldest_idle_time_s", _group_metric("max_idle"), "s"),
        ("mint_model_actor_inventory_group_oldest_age_s", _group_metric("max_age"), "s"),
        ("mint_model_actor_inventory_group_rss_bytes", _group_metric("rss_sum", require_rss_count=True), "By"),
        ("mint_model_actor_inventory_group_rss_cache_samples", _group_rss_cache_samples, None),
        (
            "mint_model_actor_inventory_observability_cache_hits_total",
            _metadata_cache_metric("cache_hits_total"),
            None,
        ),
        (
            "mint_model_actor_inventory_observability_cache_stale_total",
            _metadata_cache_metric("cache_stale_total"),
            None,
        ),
        (
            "mint_model_actor_inventory_observability_refresh_success_total",
            _metadata_cache_metric("refresh_success_total"),
            None,
        ),
        (
            "mint_model_actor_inventory_observability_refresh_failures_total",
            _metadata_cache_metric("refresh_failures_total"),
            None,
        ),
    )


@dataclass(frozen=True)
class ModelActorSpec:
    domain_key: str
    replica_id: str = "replica-0"
    base_model: str | None = None
    actor_name: str | None = None
    launcher_key: str = "vllm"
    node_pin: str | None = None
    node_pins: tuple[str, ...] = ()
    placement_slices: tuple[tuple[str, str, int], ...] = ()
    worker_alias: str | None = None
    worker_aliases: tuple[str, ...] = ()
    placement_alias_slices: tuple[tuple[str, str, int], ...] = ()
    gpu_count: int | None = None
    enabled: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return str(self.domain_key), str(self.replica_id)

    def normalized_actor_name(self) -> str:
        if self.actor_name:
            return str(self.actor_name)
        return default_model_actor_name(self.domain_key, self.replica_id)

    def normalized_node_pins(self) -> list[str]:
        pins = [str(node_ip) for _replica_id, node_ip, _gpu_count in self.placement_slices if str(node_ip).strip()]
        pins.extend(str(pin) for pin in self.node_pins if str(pin).strip())
        if self.node_pin and str(self.node_pin).strip() and str(self.node_pin) not in pins:
            pins.append(str(self.node_pin))
        return list(dict.fromkeys(pins))

    def normalized_worker_aliases(self) -> list[str]:
        aliases = [
            str(alias)
            for _replica_id, alias, _gpu_count in self.placement_alias_slices
            if str(alias).strip()
        ]
        aliases.extend(str(alias) for alias in self.worker_aliases if str(alias).strip())
        if self.worker_alias and str(self.worker_alias).strip() and str(self.worker_alias) not in aliases:
            aliases.append(str(self.worker_alias))
        return list(dict.fromkeys(aliases))


RuntimeFactory = Callable[[ModelActorSpec, int], Any | Awaitable[Any]]
NodeInventory = Callable[[], set[str] | None | Awaitable[set[str] | None]]
SchedulerSync = Callable[[list[ModelReplicaRegistration]], Any | Awaitable[Any]]
SchedulerStats = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]
OrphanPlacementGroupCleaner = Callable[[dict[tuple[str, str], ModelActorSpec]], Any | Awaitable[Any]]
PlacementReconciler = Callable[[dict[tuple[str, str], ModelActorSpec]], Any | Awaitable[dict[str, Any]]]
TopologyResolver = Callable[[dict[tuple[str, str], ModelActorSpec]], Any | Awaitable[dict[str, Any]]]
NodeMetricsFactory = Callable[[NodeMetricsDaemonSpec], Any | Awaitable[Any]]
PlacementController = ClusterPlacementController
SupervisorStateStore = SupervisorMemoryStateStore | SupervisorSQLiteStateStore


@dataclass(frozen=True)
class ControlPlaneDependency:
    name: str
    ensure: Callable[[], Awaitable[Any]]
    ping: Callable[[], Awaitable[Any]]


def domain_key_for_vllm_base_model(base_model: str) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    return f"vllm:{model}"


def _normalize_megatron_domain_key(base_model: str) -> str:
    model_name = str(base_model or "").split("/")[-1]
    model_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return f"mint_megatron_{model_name}" if model_name else "mint_megatron_model"


def _is_moe_training_model(model: str) -> bool:
    try:
        from .verl_training import _uses_distributed_training_backend

        return _uses_distributed_training_backend(model)
    except Exception:
        logger.debug("training domain model config lookup failed for %s", model, exc_info=True)
        return False


def _selected_moe_training_backend(model: str, backend: str | None = None) -> str:
    raw_backend = str(backend or "").strip().lower()
    if raw_backend:
        if raw_backend not in {"bumblebee", "megatron"}:
            raise ValueError(f"unsupported MoE training backend for topology domain: {backend!r}")
        return raw_backend
    try:
        from .verl_training import _select_moe_training_backend

        return str(_select_moe_training_backend(model)).strip().lower()
    except Exception:
        logger.debug("MoE training backend lookup failed for %s", model, exc_info=True)
        return "megatron"


def domain_key_for_training_base_model(base_model: str, *, backend: str | None = None) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    if _is_moe_training_model(model):
        moe_backend = _selected_moe_training_backend(model, backend=backend)
        return f"{moe_backend}:{_normalize_megatron_domain_key(model)}"
    return f"training:{model}"


def domain_key_for_internal_runtime() -> str:
    return "internal:runtime"


def default_model_actor_name(domain_key: str, replica_id: str) -> str:
    raw = f"mint_model_runtime_{domain_key}_{replica_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")


def _replica_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"replica-{int(value or 0)}"


def _spec_from_obj(obj: Any) -> ModelActorSpec:
    if isinstance(obj, str):
        return ModelActorSpec(domain_key=domain_key_for_vllm_base_model(obj), base_model=obj)
    if not isinstance(obj, dict):
        raise TypeError(f"model actor spec must be dict or str, got {type(obj)}")

    base_model = obj.get("base_model") or obj.get("model") or obj.get("model_id")
    domain_key = obj.get("domain_key")
    if domain_key is None:
        if base_model is None:
            raise ValueError(f"model actor spec missing domain_key/base_model: {obj!r}")
        domain_key = domain_key_for_vllm_base_model(str(base_model))
    raw_node_pins = obj.get("node_pins")
    if raw_node_pins is None:
        raw_node_pins = obj.get("node_pin")
    if isinstance(raw_node_pins, str):
        node_pins = tuple(pin.strip() for pin in raw_node_pins.split(",") if pin.strip())
    else:
        node_pins = tuple(str(pin) for pin in (raw_node_pins or []) if str(pin).strip())
    raw_worker_aliases = obj.get("worker_aliases")
    if raw_worker_aliases is None:
        raw_worker_aliases = obj.get("worker_alias")
    if isinstance(raw_worker_aliases, str):
        worker_aliases = tuple(alias.strip() for alias in raw_worker_aliases.split(",") if alias.strip())
    else:
        worker_aliases = tuple(str(alias) for alias in (raw_worker_aliases or []) if str(alias).strip())
    if obj.get("worker_index") is not None or obj.get("worker_idx") is not None:
        raise ValueError("model actor spec worker_index is no longer supported; use node_pin/node_pins")
    return ModelActorSpec(
        domain_key=str(domain_key),
        replica_id=_replica_id(obj.get("replica_id", obj.get("replica", 0))),
        base_model=None if base_model is None else str(base_model),
        actor_name=None if obj.get("actor_name") is None else str(obj["actor_name"]),
        launcher_key=str(obj.get("launcher_key") or "vllm"),
        node_pin=None if obj.get("node_pin") is None else str(obj["node_pin"]),
        node_pins=node_pins,
        worker_alias=None if obj.get("worker_alias") is None else str(obj["worker_alias"]),
        worker_aliases=worker_aliases,
        gpu_count=None if obj.get("gpu_count") is None else int(obj["gpu_count"]),
        enabled=bool(obj.get("enabled", True)),
    )


def _topology_model_specs_from_config_models(models: dict[str, Any]) -> list[ModelActorSpec]:
    specs: list[ModelActorSpec] = []
    for model, raw_cfg in sorted(models.items()):
        if not isinstance(raw_cfg, dict):
            continue
        base_model = str(model).strip()
        if not base_model:
            continue
        for launcher_key in ("vllm", "training", "megatron", "bumblebee"):
            raw_launcher_cfg = raw_cfg.get(launcher_key)
            if raw_launcher_cfg is None:
                continue
            launcher_cfg = raw_launcher_cfg if isinstance(raw_launcher_cfg, dict) else {}
            backend_override = launcher_cfg.get("backend", launcher_cfg.get("training_backend"))
            if launcher_key == "vllm":
                domain_key = domain_key_for_vllm_base_model(base_model)
            elif launcher_key == "bumblebee":
                domain_key = domain_key_for_training_base_model(base_model, backend="bumblebee")
            elif launcher_key == "megatron":
                # Legacy topology files used `megatron` to mean distributed MoE
                # training placement. Keep the key as a compatibility alias for
                # the selected MoE backend so backend flips rebuild the runtime
                # actor domain instead of pinning the old Megatron domain.
                domain_key = domain_key_for_training_base_model(
                    base_model,
                    backend=None if backend_override is None else str(backend_override),
                )
            else:
                domain_key = domain_key_for_training_base_model(
                    base_model,
                    backend=None if backend_override is None else str(backend_override),
                )
            if launcher_key in {"megatron", "bumblebee"} and domain_key.startswith("training:"):
                continue
            placement_items = launcher_cfg.get("placement") or []
            if isinstance(placement_items, dict):
                placement_items = [placement_items]
            if placement_items and not isinstance(placement_items, list):
                raise ValueError(f"topology model placement for {base_model!r}/{launcher_key} must be a list")
            default_replica_id = _replica_id(launcher_cfg.get("replica_id", launcher_cfg.get("replica", 0)))
            default_gpu_count = launcher_cfg.get("gpu_count")
            by_replica: dict[str, dict[str, Any]] = {}
            for item in placement_items:
                if not isinstance(item, dict):
                    raise ValueError(f"topology model placement item for {base_model!r}/{launcher_key} must be an object")
                item_replica = _replica_id(item.get("replica_id", item.get("replica", default_replica_id)))
                bucket = by_replica.setdefault(
                    item_replica,
                    {
                        "node_pins": [],
                        "worker_aliases": [],
                        "placement_slices": [],
                        "placement_alias_slices": [],
                        "gpu_count": default_gpu_count,
                    },
                )
                item_gpu_count = item.get("gpu_count", bucket.get("gpu_count"))
                raw_worker = item.get("worker_alias", item.get("worker"))
                raw_node_ip = item.get("node_ip", item.get("node_pin"))
                has_node_ip = raw_node_ip is not None and bool(str(raw_node_ip).strip())
                has_worker = raw_worker is not None and bool(str(raw_worker).strip())
                if not has_node_ip and not has_worker:
                    raise ValueError(
                        f"topology model placement item for {base_model!r}/{launcher_key} "
                        "must include worker/worker_alias or node_ip/node_pin"
                    )
                if item_gpu_count is None:
                    raise ValueError(
                        f"topology model placement item for {base_model!r}/{launcher_key} must include gpu_count"
                    )
                item_gpu_count_int = int(item_gpu_count)
                if item_gpu_count is not None:
                    bucket["gpu_count"] = item_gpu_count_int
                if raw_node_ip is not None:
                    node_ip = str(raw_node_ip).strip()
                    if node_ip:
                        bucket["node_pins"].append(node_ip)
                        bucket["placement_slices"].append((item_replica, node_ip, item_gpu_count_int))
                    continue
                if raw_worker is not None:
                    worker = str(raw_worker).strip()
                    if not worker:
                        continue
                    if is_ip_address(worker):
                        bucket["node_pins"].append(worker)
                        bucket["placement_slices"].append((item_replica, worker, item_gpu_count_int))
                    else:
                        bucket["worker_aliases"].append(worker)
                        bucket["placement_alias_slices"].append((item_replica, worker, item_gpu_count_int))
            if not by_replica:
                by_replica[default_replica_id] = {
                    "node_pins": [],
                    "worker_aliases": [],
                    "placement_slices": [],
                    "placement_alias_slices": [],
                    "gpu_count": default_gpu_count,
                }
            spec_launcher = "training" if launcher_key in {"megatron", "bumblebee"} else launcher_key
            for replica_id, bucket in sorted(by_replica.items()):
                gpu_count = bucket.get("gpu_count")
                specs.append(
                    ModelActorSpec(
                        domain_key=domain_key,
                        replica_id=replica_id,
                        base_model=base_model,
                        launcher_key=spec_launcher,
                        node_pins=tuple(dict.fromkeys(bucket["node_pins"])),
                        placement_slices=tuple(bucket["placement_slices"]),
                        worker_aliases=tuple(dict.fromkeys(bucket["worker_aliases"])),
                        placement_alias_slices=tuple(bucket["placement_alias_slices"]),
                        gpu_count=None if gpu_count is None else int(gpu_count),
                        enabled=bool(launcher_cfg.get("enabled", True)),
                    )
                )
    return specs


def _topology_model_specs_from_env() -> list[ModelActorSpec]:
    config = load_topology_config_from_env()
    if config is None:
        return []
    return _topology_model_specs_from_config_models(config.models)


def _supported_model_specs_from_env() -> dict[str, ModelActorSpec]:
    specs = _topology_model_specs_from_env()
    if not specs:
        supported = os.environ.get("MINT_SUPPORTED_MODELS", "").strip()
        if not supported:
            return {}

        specs = []
        for model in (item.strip() for item in supported.split(",")):
            if not model:
                continue
            specs.append(
                ModelActorSpec(
                    domain_key=domain_key_for_vllm_base_model(model),
                    base_model=model,
                    launcher_key="vllm",
                )
            )
            specs.append(
                ModelActorSpec(
                    domain_key=domain_key_for_training_base_model(model),
                    base_model=model,
                    launcher_key="training",
                )
            )
    return {spec.domain_key: spec for spec in specs}


def _spec_for_scheduler_domain_from_env(domain_key: str) -> ModelActorSpec | None:
    domain = str(domain_key).strip()
    if not domain or domain == domain_key_for_internal_runtime():
        return None

    supported = _supported_model_specs_from_env()
    if domain in supported:
        return supported[domain]

    if domain.startswith("vllm:"):
        base_model = domain.removeprefix("vllm:").strip()
        if not base_model:
            return None
        return ModelActorSpec(
            domain_key=domain,
            base_model=base_model,
            launcher_key="vllm",
        )
    if domain.startswith("training:"):
        base_model = domain.removeprefix("training:").strip()
        if not base_model:
            return None
        return ModelActorSpec(
            domain_key=domain,
            base_model=base_model,
            launcher_key="training",
        )
    if domain.startswith(("megatron:", "bumblebee:")):
        for spec in supported.values():
            if spec.domain_key == domain:
                return spec
        normalized = domain.split(":", 1)[1].strip()
        for spec in supported.values():
            if (
                spec.base_model
                and _normalize_megatron_domain_key(spec.base_model) == normalized
                and spec.launcher_key == "training"
            ):
                return ModelActorSpec(
                    domain_key=domain,
                    replica_id=spec.replica_id,
                    base_model=spec.base_model,
                    launcher_key="training",
                    node_pin=spec.node_pin,
                    node_pins=spec.node_pins,
                    placement_slices=spec.placement_slices,
                    worker_alias=spec.worker_alias,
                    worker_aliases=spec.worker_aliases,
                    placement_alias_slices=spec.placement_alias_slices,
                    gpu_count=spec.gpu_count,
                    enabled=spec.enabled,
                )
    return None


def _active_scheduler_domains(stats: dict[str, Any]) -> set[str]:
    domains: set[str] = set()
    backlog = stats.get("backlog_depth_by_domain")
    if isinstance(backlog, dict):
        for domain, depth in backlog.items():
            try:
                if int(depth) > 0:
                    domains.add(str(domain))
            except Exception:
                continue

    replica_queues = stats.get("replica_queues")
    if isinstance(replica_queues, dict):
        for queue in replica_queues.values():
            if not isinstance(queue, dict):
                continue
            try:
                if int(queue.get("depth") or 0) > 0 and queue.get("domain_key"):
                    domains.add(str(queue["domain_key"]))
            except Exception:
                continue

    leases = stats.get("leases")
    if isinstance(leases, list):
        for lease in leases:
            if isinstance(lease, dict) and lease.get("domain_key"):
                domains.add(str(lease["domain_key"]))
    return domains


def desired_specs_from_env() -> list[ModelActorSpec]:
    def _with_internal_runtime(specs: list[ModelActorSpec]) -> list[ModelActorSpec]:
        enabled = str(os.environ.get("MINT_MODEL_ACTOR_INTERNAL_RUNTIME", "1")).strip().lower()
        if enabled in ("0", "false", "no", "n", "off"):
            return specs
        domain_key = domain_key_for_internal_runtime()
        if any(spec.domain_key == domain_key for spec in specs):
            return specs
        return [
            *specs,
            ModelActorSpec(
                domain_key=domain_key,
                launcher_key="cpu_runtime",
                gpu_count=0,
            ),
        ]

    specs = _topology_model_specs_from_env()
    if not specs:
        return _with_internal_runtime([])
    return _with_internal_runtime(specs)


def default_control_plane_dependencies() -> list[ControlPlaneDependency]:
    async def _ensure_task_state_store() -> Any:
        from .task_state_store import task_state_store

        out = await task_state_store.async_ensure_ready(timeout_s=5.0, create_if_missing=True)
        await task_state_store.async_future_ping(timeout_s=5.0)
        return out

    async def _ping_task_state_store() -> Any:
        from .task_state_store import task_state_store

        out = await task_state_store.async_ping(timeout_s=5.0)
        await task_state_store.async_future_ping(timeout_s=5.0)
        return out

    async def _ensure_model_work_scheduler() -> Any:
        from .model_work_scheduler import model_work_scheduler

        return await model_work_scheduler.stats(timeout_s=5.0, create_if_missing=True)

    async def _ping_model_work_scheduler() -> Any:
        from .model_work_scheduler import model_work_scheduler

        return await model_work_scheduler.async_ping(timeout_s=5.0)

    async def _ensure_maintenance_cron() -> Any:
        from .maintenance_cron_actor import maintenance_cron_actor

        return await maintenance_cron_actor.async_ensure_started(timeout_s=15.0)

    async def _ping_maintenance_cron() -> Any:
        from .maintenance_cron_actor import maintenance_cron_actor

        return await maintenance_cron_actor.async_ping(timeout_s=5.0)

    return [
        ControlPlaneDependency(
            name="task_state_store",
            ensure=_ensure_task_state_store,
            ping=_ping_task_state_store,
        ),
        ControlPlaneDependency(
            name="model_work_scheduler",
            ensure=_ensure_model_work_scheduler,
            ping=_ping_model_work_scheduler,
        ),
        ControlPlaneDependency(
            name="maintenance_cron_actor",
            ensure=_ensure_maintenance_cron,
            ping=_ping_maintenance_cron,
        ),
    ]


async def _maybe_await(value: Any) -> Any:
    timeout_s = 10.0
    raw_timeout = os.environ.get("MINT_MODEL_ACTOR_SUPERVISOR_REMOTE_CALL_TIMEOUT_S")
    if raw_timeout:
        try:
            timeout_s = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout_s = 10.0
    try:
        return await async_get_ray_ref(value, timeout_s=timeout_s)
    except TypeError:
        if inspect.isawaitable(value):
            return await value
        return value


async def _invoke_actor(actor: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return await _maybe_await(remote(*args, **kwargs))
    return await _maybe_await(method(*args, **kwargs))


def _is_ray_get_timeout_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ in {"GetTimeoutError", "_GetTimeoutError"}:
        return True
    return False


def _observed_free_gpus_by_node() -> dict[str, int]:
    try:
        from .node_placement import _list_alive_gpu_nodes

        return {
            str(node.node_ip): max(0, int(node.available_gpus))
            for node in _list_alive_gpu_nodes()
            if str(node.node_ip).strip()
        }
    except Exception:
        logger.debug("[model_actor_supervisor] observed GPU lookup failed", exc_info=True)
        return {}


def _placement_group_table() -> dict[str, Any]:
    try:
        import ray

        table = ray.util.placement_group_table()
        return table if isinstance(table, dict) else {}
    except Exception:
        logger.debug("[model_actor_supervisor] placement group table lookup failed", exc_info=True)
        return {}


class ModelActorSupervisorCore:
    def __init__(
        self,
        *,
        specs: list[ModelActorSpec] | None = None,
        runtime_factory: RuntimeFactory | None = None,
        node_inventory: NodeInventory | None = None,
        scheduler: ModelWorkSchedulerClient | None = None,
        scheduler_sync: SchedulerSync | None = None,
        scheduler_stats: SchedulerStats | None = None,
        orphan_pg_cleaner: OrphanPlacementGroupCleaner | None = None,
        placement_controller: PlacementController | None = None,
        placement_reconciler: PlacementReconciler | None = None,
        topology_resolver: TopologyResolver | None = None,
        topology_manager: TopologyManager | None = None,
        node_metrics_factory: NodeMetricsFactory | None = None,
        node_metrics_enabled: bool | None = None,
        control_plane_dependencies: list[ControlPlaneDependency] | None = None,
        control_plane_enabled: bool | None = None,
        reconcile_interval_s: float | None = None,
        launcher_registry: ModelActorLauncherRegistry | None = None,
        state_store: SupervisorStateStore | None = None,
        state_backend: str | None = None,
        state_db_path: str | None = None,
        owner_id: str | None = None,
        owner_ttl_s: float | None = None,
        state_event_limit: int | None = None,
        ray_address: str | None = None,
    ) -> None:
        # Detached control-plane actors already run inside a Ray worker context.
        # Do not write a driver/direct Ray address into the actor process env:
        # downstream control-plane calls must reuse that worker context instead
        # of attempting a nested ray.init()/direct attach from inside Ray.
        self._ray_address = str(ray_address or "").strip() or None
        try:
            from ..logging_context import init_actor_observability

            init_actor_observability()
        except Exception:
            logger.debug("[model_actor_supervisor] actor observability init skipped", exc_info=True)
        self._desired: dict[tuple[str, str], ModelActorSpec] = {}
        self._launcher_registry = launcher_registry or default_model_actor_launcher_registry()
        self._runtime_factory = runtime_factory
        self._node_inventory = node_inventory
        self._scheduler = scheduler or model_work_scheduler
        self._scheduler_sync = scheduler_sync
        self._scheduler_stats = scheduler_stats
        self._orphan_pg_cleaner = orphan_pg_cleaner
        self._topology_manager = topology_manager if topology_manager is not None else TopologyManager()
        self._topology_resolver = topology_resolver or self._resolve_topology_placements
        if placement_controller is not None:
            self._placement_controller = placement_controller
            self._placement_reconciler = None
        elif placement_reconciler is None and runtime_factory is None:
            self._placement_controller = self._default_placement_controller()
            self._placement_reconciler = None
        else:
            self._placement_controller = None
            self._placement_reconciler = placement_reconciler or model_actor_placement_reconciler
        self._node_metrics_factory = node_metrics_factory
        self._node_metrics_enabled = (
            _node_metrics_enabled_by_default()
            if node_metrics_enabled is None
            else bool(node_metrics_enabled)
        )
        if self._node_metrics_factory is None and self._node_metrics_enabled:
            self._node_metrics_factory = get_or_create_node_metrics_collector_actor
        self._node_metric_actors: dict[str, Any] = {}
        self._node_metric_states: dict[str, dict[str, Any]] = {}
        self._node_metrics_created_total = 0
        self._node_metrics_reconcile_failures_total = 0
        self._control_plane_enabled = True if control_plane_enabled is None else bool(control_plane_enabled)
        self._control_plane_dependencies = (
            list(control_plane_dependencies)
            if control_plane_dependencies is not None
            else default_control_plane_dependencies()
        )
        self._control_plane_states: dict[str, dict[str, Any]] = {}
        self._control_plane_ensure_failures_total = 0
        self._actors: dict[tuple[str, str], Any] = {}
        self._actor_generations: dict[tuple[str, str], int] = {}
        self._generations: dict[tuple[str, str], int] = {}
        self._states: dict[tuple[str, str], dict[str, Any]] = {}
        self._latest_push: dict[tuple[str, str], EngineLivenessPush] = {}
        self._reconcile_total = 0
        self._created_total = 0
        self._restarted_total = 0
        self._blocked_total = 0
        self._busy_recycle_skipped_total = 0
        self._health_timeout_preserved_total = 0
        self._scheduler_sync_failures_total = 0
        self._placement_reconcile_failures_total = 0
        self._topology_reconcile_failures_total = 0
        self._placement_reclaimed_total = 0
        self._placement_groups_created_total = 0
        self._last_reconcile_at: float | None = None
        self._last_scheduler_sync_at: float | None = None
        self._last_placement_reconcile: dict[str, Any] | None = None
        self._last_topology_reconcile: dict[str, Any] | None = None
        self._inventory = ModelActorInventory()
        self._state_store: SupervisorStateStore = state_store or create_supervisor_state_store(
            backend=state_backend or server_config.supervisor_state_backend,
            db_path=state_db_path or server_config.supervisor_state_db_path,
            event_limit=(
                int(state_event_limit)
                if state_event_limit is not None
                else int(server_config.supervisor_state_event_limit)
            ),
        )
        self._owner_name = "model_actor_supervisor"
        self._owner_id = owner_id or self._default_owner_id()
        self._owner_ttl_s = float(
            owner_ttl_s
            if owner_ttl_s is not None
            else server_config.supervisor_state_owner_ttl_s
        )
        self._state_owner: dict[str, Any] | None = None
        self._state_store_failures_total = 0
        self._ensure_state_owner()
        self._reconcile_interval_s = (
            _reconcile_interval_s_from_env()
            if reconcile_interval_s is None
            else float(reconcile_interval_s)
        )
        self._reconcile_task: asyncio.Task | None = None
        self._reconcile_loop_starting = False
        self._reconcile_inflight = False
        self._reconcile_inflight_started_at: float | None = None
        self._last_reconcile_loop_error: str | None = None
        for spec in specs or []:
            self.set_desired(spec)
        self._otel_enabled = False
        self._otel_error: str | None = None
        self._init_otel_metrics()

    def _default_placement_controller(self) -> ClusterPlacementController:
        return ClusterPlacementController(
            observed_free_gpus_by_node=_observed_free_gpus_by_node,
            placement_group_table=_placement_group_table,
            placement_reconciler=model_actor_placement_reconciler,
        )

    @staticmethod
    def _default_owner_id() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"

    def _ensure_state_owner(self) -> dict[str, Any] | None:
        if self._state_owner is not None:
            return self._state_owner
        try:
            self._state_owner = self._state_store.acquire_owner(
                name=self._owner_name,
                owner_id=self._owner_id,
                ttl_s=self._owner_ttl_s,
            )
            self._state_store.append_event(
                "owner_acquired",
                {"backend": self._state_store.backend},
                owner=self._state_owner,
            )
        except SupervisorStateOwnerConflictError:
            raise
        except Exception as e:
            self._state_store_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] state owner acquire failed: %s: %s",
                type(e).__name__,
                e,
            )
            self._state_owner = None
        return self._state_owner

    def _heartbeat_state_owner(self) -> None:
        owner = self._state_owner or self._ensure_state_owner()
        if owner is None:
            return
        try:
            self._state_owner = self._state_store.heartbeat_owner(
                name=self._owner_name,
                owner_id=str(owner["owner_id"]),
                epoch=int(owner["epoch"]),
                ttl_s=self._owner_ttl_s,
            )
        except Exception as e:
            self._state_store_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] state owner heartbeat failed: %s: %s",
                type(e).__name__,
                e,
            )

    def _state_generation_key(self, key: tuple[str, str]) -> str:
        return f"generation:{_label(key)}"

    async def _sync_active_scheduler_domains(self) -> None:
        stats: dict[str, Any] | None = None
        try:
            if self._scheduler_stats is not None:
                candidate = await _maybe_await(self._scheduler_stats())
            elif self._scheduler_sync is None:
                candidate = await self._scheduler.stats(timeout_s=2.0)
            else:
                candidate = None
            if isinstance(candidate, dict):
                stats = candidate
        except Exception as e:
            logger.debug(
                "[model_actor_supervisor] active scheduler domain sync skipped: %s: %s",
                type(e).__name__,
                e,
            )
        if not stats:
            return
        for domain_key in sorted(_active_scheduler_domains(stats)):
            spec = _spec_for_scheduler_domain_from_env(domain_key)
            if spec is None:
                logger.warning(
                    "[model_actor_supervisor] scheduler has active domain without launch spec: %s",
                    domain_key,
                )
                continue
            if spec.key not in self._desired:
                logger.info(
                    "[model_actor_supervisor] adding active scheduler domain to desired runtimes domain=%s replica=%s",
                    spec.domain_key,
                    spec.replica_id,
                )
                self.set_desired(spec)

    # Explicit inventory/launcher contract for backend-created GPU actors.
    # Backend-specific vLLM/Megatron/dense launchers own their Ray actor
    # creation, then publish lifecycle through ModelActorSupervisor.
    def register(
        self,
        *,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        actor_handle: Any | None = None,
        namespace: str = "mint",
        base_model: str = "",
        session_id: str | None = None,
        node_id: str | None = None,
        protected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ActorEntry:
        metadata = {
            "launcher_contract": "model_actor_supervisor",
            **dict(metadata or {}),
        }
        return self._inventory.register(
            actor_name=actor_name,
            actor_type=actor_type,
            num_gpus=num_gpus,
            actor_handle=actor_handle,
            namespace=namespace,
            base_model=base_model,
            session_id=session_id,
            node_id=node_id,
            protected=protected,
            metadata=metadata,
        )

    def unregister(self, actor_name: str) -> bool:
        return self._inventory.unregister(actor_name)

    def get(self, actor_name: str) -> ActorEntry | None:
        return self._inventory.get(actor_name)

    def set_session(self, actor_name: str, session_id: str | None) -> None:
        self._inventory.set_session(actor_name, session_id)

    async def async_set_session(self, actor_name: str, session_id: str | None) -> None:
        await self._inventory.async_set_session(actor_name, session_id)

    def set_protected(self, actor_name: str, protected: bool = True) -> bool:
        return self._inventory.set_protected(actor_name, protected)

    def is_protected(self, actor_name: str) -> bool:
        return self._inventory.is_protected(actor_name)

    def touch(self, actor_name: str) -> bool:
        return self._inventory.touch(actor_name)

    async def async_touch(self, actor_name: str) -> bool:
        return await self._inventory.async_touch(actor_name)

    def mark_inflight(self, actor_name: str, delta: int) -> None:
        self._inventory.mark_inflight(actor_name, delta)

    def mark_ready(self, actor_name: str) -> None:
        self._inventory.mark_ready(actor_name)

    def update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return self._inventory.update_metadata(
            actor_name,
            metadata=metadata,
            sample_time=sample_time,
            sample_source=sample_source,
        )

    async def async_update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return await self._inventory.async_update_metadata(
            actor_name,
            metadata=metadata,
            sample_time=sample_time,
            sample_source=sample_source,
        )

    def list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._inventory.list_actors(
            refresh_metadata=refresh_metadata,
            actor_type=actor_type,
            model_name=model_name,
        )

    async def async_list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._inventory.async_list_actors(
            refresh_metadata=refresh_metadata,
            actor_type=actor_type,
            model_name=model_name,
        )

    def metadata_cache_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return self._inventory.metadata_cache_metrics_snapshot()

    def lifecycle_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return self._inventory.lifecycle_metrics_snapshot()

    def cached_snapshot(self, *, refresh_metadata: bool = True) -> list[dict[str, Any]]:
        return self._inventory.cached_snapshot(refresh_metadata=refresh_metadata)

    def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
        return self._inventory.rss_snapshot(timeout_s=timeout_s)

    def _init_otel_metrics(self) -> None:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation

            meter = metrics.get_meter("mint.model_actor_supervisor")

            def _gauge(name: str, callback, *, unit: str | None = None, description: str = "") -> None:
                kwargs: dict[str, Any] = {"callbacks": [callback]}
                if unit:
                    kwargs["unit"] = unit
                if description:
                    kwargs["description"] = description
                meter.create_observable_gauge(name, **kwargs)

            def _attrs(**extra: object) -> dict[str, str]:
                attrs = _otel_metric_attrs()
                for key, value in extra.items():
                    text = str(value if value is not None else "").strip()
                    if text:
                        attrs[key] = text
                return attrs

            def _scalar(field: str):
                def _callback(_options):
                    value = _prom_number(self.snapshot().get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            for key in (
                "desired_total",
                "managed_total",
                "domain_total",
                "reconcile_total",
                "created_total",
                "restarted_total",
                "blocked_total",
                "busy_recycle_skipped_total",
                "scheduler_sync_failures_total",
                "topology_reconcile_failures_total",
                "node_metrics_created_total",
                "node_metrics_reconcile_failures_total",
            ):
                _gauge(f"mint_model_actor_supervisor_{key}", _scalar(key))

            def _topology_node_state(_options):
                snapshot = self.snapshot()
                topology = snapshot.get("topology")
                nodes = topology.get("nodes") if isinstance(topology, dict) else None
                if not isinstance(nodes, dict):
                    return []
                observations = []
                for alias, rec in sorted(nodes.items()):
                    if not isinstance(rec, dict):
                        continue
                    observations.append(
                        Observation(
                            1.0,
                            _attrs(
                                worker_alias=alias,
                                state=rec.get("state") or "unknown",
                                provider=rec.get("provider") or "unknown",
                            ),
                        )
                    )
                return observations

            def _topology_node_gpus(_options):
                snapshot = self.snapshot()
                topology = snapshot.get("topology")
                nodes = topology.get("nodes") if isinstance(topology, dict) else None
                if not isinstance(nodes, dict):
                    return []
                observations = []
                for alias, rec in sorted(nodes.items()):
                    if not isinstance(rec, dict):
                        continue
                    value = _prom_number(rec.get("gpu_count"))
                    if value is None:
                        continue
                    observations.append(
                        Observation(
                            value,
                            _attrs(
                                worker_alias=alias,
                                state=rec.get("state") or "unknown",
                                provider=rec.get("provider") or "unknown",
                            ),
                        )
                    )
                return observations

            _gauge("mint_topology_node_state", _topology_node_state)
            _gauge("mint_topology_node_gpus", _topology_node_gpus)

            def _node_metrics_scalar(field: str):
                def _callback(_options):
                    snapshot = self.snapshot()
                    daemons = snapshot.get("daemons")
                    node_metrics = daemons.get("node_metrics") if isinstance(daemons, dict) else None
                    if not isinstance(node_metrics, dict):
                        return []
                    value = _prom_number(node_metrics.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            _gauge("mint_node_metrics_daemon_enabled", _node_metrics_scalar("enabled"))
            _gauge("mint_node_metrics_daemon_desired_total", _node_metrics_scalar("desired_total"))
            _gauge("mint_node_metrics_daemon_managed_total", _node_metrics_scalar("managed_total"))

            def _node_metrics_state(_options):
                snapshot = self.snapshot()
                daemons = snapshot.get("daemons")
                node_metrics = daemons.get("node_metrics") if isinstance(daemons, dict) else None
                nodes = node_metrics.get("nodes") if isinstance(node_metrics, dict) else None
                if not isinstance(nodes, dict):
                    return []
                observations = []
                for alias, rec in sorted(nodes.items()):
                    if not isinstance(rec, dict):
                        continue
                    observations.append(
                        Observation(
                            1.0,
                            _attrs(worker_alias=alias, state=rec.get("state") or "unknown"),
                        )
                    )
                return observations

            def _node_metrics_health(field: str):
                def _callback(_options):
                    snapshot = self.snapshot()
                    daemons = snapshot.get("daemons")
                    node_metrics = daemons.get("node_metrics") if isinstance(daemons, dict) else None
                    nodes = node_metrics.get("nodes") if isinstance(node_metrics, dict) else None
                    if not isinstance(nodes, dict):
                        return []
                    observations = []
                    for alias, rec in sorted(nodes.items()):
                        if not isinstance(rec, dict):
                            continue
                        health = rec.get("health")
                        if not isinstance(health, dict):
                            continue
                        value = _prom_number(health.get(field))
                        if value is None:
                            continue
                        observations.append(
                            Observation(
                                value,
                                _attrs(worker_alias=alias, state=rec.get("state") or "unknown"),
                            )
                        )
                    return observations

                return _callback

            _gauge("mint_node_metrics_daemon_state", _node_metrics_state)
            _gauge("mint_node_metrics_daemon_sample_count", _node_metrics_health("sample_count"))
            _gauge("mint_node_metrics_daemon_error_count", _node_metrics_health("error_count"))

            def _domain_metric(field: str):
                def _callback(_options):
                    snapshot = self.snapshot()
                    domains = snapshot.get("domains")
                    if not isinstance(domains, dict):
                        return []
                    observations = []
                    for domain_key, rec in sorted(domains.items()):
                        if not isinstance(rec, dict):
                            continue
                        value = _prom_number(rec.get(field))
                        if value is None:
                            continue
                        observations.append(
                            Observation(
                                value,
                                _attrs(domain_key=domain_key),
                            )
                        )
                    return observations

                return _callback

            _gauge("mint_model_actor_supervisor_domain_replicas", _domain_metric("replicas"))
            _gauge("mint_model_actor_supervisor_domain_healthy", _domain_metric("healthy"))
            _gauge("mint_model_actor_supervisor_domain_unhealthy", _domain_metric("unhealthy"))

            def _replica_state(_options):
                snapshot = self.snapshot()
                replicas = snapshot.get("replicas")
                if not isinstance(replicas, dict):
                    return []
                observations = []
                for rec in replicas.values():
                    if not isinstance(rec, dict):
                        continue
                    observations.append(
                        Observation(
                            1.0,
                            _attrs(
                                domain_key=rec.get("domain_key") or "unknown",
                                replica_id=rec.get("replica_id") or "unknown",
                                actor_name=rec.get("actor_name") or rec.get("desired_actor_name") or "unknown",
                                state=rec.get("state") or "unknown",
                            ),
                        )
                    )
                return observations

            def _replica_generation(_options):
                snapshot = self.snapshot()
                replicas = snapshot.get("replicas")
                if not isinstance(replicas, dict):
                    return []
                observations = []
                for rec in replicas.values():
                    if not isinstance(rec, dict):
                        continue
                    value = _prom_number(rec.get("generation"))
                    if value is None:
                        continue
                    observations.append(
                        Observation(
                            value,
                            _attrs(
                                domain_key=rec.get("domain_key") or "unknown",
                                replica_id=rec.get("replica_id") or "unknown",
                                actor_name=rec.get("actor_name") or rec.get("desired_actor_name") or "unknown",
                                state=rec.get("state") or "unknown",
                            ),
                        )
                    )
                return observations

            _gauge("mint_model_actor_supervisor_replica_state", _replica_state)
            _gauge("mint_model_actor_supervisor_replica_generation", _replica_generation)

            for name, callback, unit in _inventory_otel_gauge_callbacks(self, Observation, _attrs):
                _gauge(name, callback, unit=unit)
            self._otel_enabled = True
        except Exception as e:
            self._otel_error = f"{type(e).__name__}: {e}"

    def iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        return self._inventory.iter_entries(prune_stale=prune_stale)

    async def async_iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        return await self._inventory.async_iter_entries(prune_stale=prune_stale)

    def clear_session(self, session_id: str, *, actor_type: ActorType | None = None) -> int:
        return self._inventory.clear_session(session_id, actor_type=actor_type)

    def total_gpus_used(self) -> int:
        return self._inventory.total_gpus_used()

    async def async_total_gpus_used(self) -> int:
        return await self._inventory.async_total_gpus_used()

    def gpus_used_by_node(self) -> dict[str, int]:
        return self._inventory.gpus_used_by_node()

    def clear(self, kill_actors: bool = True) -> int:
        return self._inventory.clear(kill_actors=kill_actors)

    def _reconcile_protected_actor_names(self, desired: dict[tuple[str, str], ModelActorSpec]) -> set[str]:
        protected = {
            str(spec.normalized_actor_name())
            for spec in desired.values()
            if bool(getattr(spec, "enabled", True)) and str(spec.normalized_actor_name()).strip()
        }
        protected.update(
            str(entry.actor_name)
            for entry in self._inventory.iter_entries(prune_stale=False)
            if str(entry.actor_name).strip()
        )
        protected.update(
            str(state.get("actor_name") or "")
            for state in self._node_metric_states.values()
            if str(state.get("actor_name") or "").strip()
        )
        return protected

    def set_desired(self, spec: ModelActorSpec) -> None:
        if not spec.domain_key:
            raise ValueError("domain_key is required")
        if not spec.replica_id:
            raise ValueError("replica_id is required")
        key = spec.key
        self._desired[key] = spec
        self._states.setdefault(
            key,
            {
                "domain_key": spec.domain_key,
                "replica_id": spec.replica_id,
                "state": "desired",
                "actor_name": spec.normalized_actor_name(),
                "generation": 0,
                "crash_count": 0,
                "last_error": None,
                "last_action": "set_desired",
                "last_action_at": time.time(),
            },
        )

    def remove_desired(self, *, domain_key: str, replica_id: str) -> None:
        self._desired.pop(_key(domain_key, replica_id), None)

    async def _available_nodes(self) -> set[str] | None:
        if self._node_inventory is None:
            return None
        nodes = await _maybe_await(self._node_inventory())
        if nodes is None:
            return None
        return {str(node) for node in nodes}

    async def _node_pins_possible(
        self,
        spec: ModelActorSpec,
        *,
        resolved_node_pins: list[str] | None = None,
    ) -> bool:
        pins = list(resolved_node_pins) if resolved_node_pins is not None else spec.normalized_node_pins()
        if not pins:
            return True
        nodes = await self._available_nodes()
        if nodes is None:
            return True
        return all(pin in nodes for pin in pins)

    async def _ensure_control_plane_dependencies(self) -> None:
        if not self._control_plane_enabled:
            self._control_plane_states = {}
            return
        now = time.time()
        for dependency in self._control_plane_dependencies:
            previous = dict(self._control_plane_states.get(dependency.name, {}))
            try:
                result = await _maybe_await(dependency.ensure())
                self._control_plane_states[dependency.name] = {
                    "name": dependency.name,
                    "state": "ready",
                    "last_error": None,
                    "last_checked_at": now,
                    "last_ready_at": now,
                    "result": result if isinstance(result, dict) else {"value": repr(result)},
                }
            except Exception as e:
                self._control_plane_ensure_failures_total += 1
                self._control_plane_states[dependency.name] = {
                    **previous,
                    "name": dependency.name,
                    "state": "unhealthy",
                    "last_error": f"{type(e).__name__}: {e}",
                    "last_checked_at": now,
                }
                logger.warning(
                    "[model_actor_supervisor] control-plane ensure failed name=%s error_type=%s error=%s",
                    dependency.name,
                    type(e).__name__,
                    e,
                )

    async def _resolve_topology_placements(
        self,
        desired: dict[tuple[str, str], ModelActorSpec],
    ) -> dict[str, Any]:
        manager = self._topology_manager
        if manager is None or not manager.enabled:
            blocked: dict[str, str] = {}
            node_pins: dict[str, list[str]] = {}
            placement_slices: dict[str, tuple[tuple[str, str, int], ...]] = {}
            for key, spec in sorted(desired.items()):
                label = _label(key)
                resolved_pins = spec.normalized_node_pins()
                resolved_slices = list(spec.placement_slices)
                for replica_id, alias, gpu_count in spec.placement_alias_slices:
                    if is_ip_address(alias):
                        resolved_slices.append((replica_id, alias, int(gpu_count)))
                        resolved_pins.append(alias)
                    else:
                        blocked[label] = "topology config is not enabled"
                        break
                if label in blocked:
                    continue
                for alias in spec.normalized_worker_aliases():
                    if is_ip_address(alias):
                        resolved_pins.append(alias)
                    else:
                        blocked[label] = "topology config is not enabled"
                        break
                if label in blocked:
                    continue
                resolved_pins = list(dict.fromkeys(pin for pin in resolved_pins if str(pin).strip()))
                if resolved_pins:
                    node_pins[label] = resolved_pins
                if resolved_slices:
                    placement_slices[label] = tuple(resolved_slices)
            return {
                "ok": not blocked,
                "blocked": blocked,
                "node_pins": node_pins,
                "placement_slices": placement_slices,
            }
        state = manager.reconcile_once()
        blocked: dict[str, str] = {}
        node_pins: dict[str, list[str]] = {}
        placement_slices: dict[str, tuple[tuple[str, str, int], ...]] = {}
        for key, spec in sorted(desired.items()):
            label = _label(key)
            resolved_pins = spec.normalized_node_pins()
            resolved_slices = list(spec.placement_slices)
            for replica_id, alias, gpu_count in spec.placement_alias_slices:
                node_ip, error = manager.resolve_alias(alias)
                if error:
                    blocked[label] = error
                    break
                if node_ip:
                    resolved_slices.append((replica_id, node_ip, int(gpu_count)))
                    resolved_pins.append(node_ip)
            if label in blocked:
                continue
            for alias in spec.normalized_worker_aliases():
                node_ip, error = manager.resolve_alias(alias)
                if error:
                    blocked[label] = error
                    break
                if node_ip:
                    resolved_pins.append(node_ip)
            if label in blocked:
                continue
            resolved_pins = list(dict.fromkeys(pin for pin in resolved_pins if str(pin).strip()))
            if resolved_pins:
                node_pins[label] = resolved_pins
            if resolved_slices:
                placement_slices[label] = tuple(resolved_slices)
        return {
            "ok": not blocked,
            "blocked": blocked,
            "node_pins": node_pins,
            "placement_slices": placement_slices,
            "state": manager.snapshot(),
            "observed_at": None if state is None else state.observed_at,
        }

    @staticmethod
    def _spec_with_resolved_topology(
        spec: ModelActorSpec,
        *,
        node_pins: list[str] | None = None,
        placement_slices: tuple[tuple[str, str, int], ...] | None = None,
    ) -> ModelActorSpec:
        if node_pins is None and placement_slices is None:
            return spec
        resolved_slices = spec.placement_slices if placement_slices is None else placement_slices
        resolved_pins = tuple(node_pins) if node_pins is not None else spec.node_pins
        return ModelActorSpec(
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            base_model=spec.base_model,
            actor_name=spec.actor_name,
            launcher_key=spec.launcher_key,
            node_pin=spec.node_pin,
            node_pins=resolved_pins,
            placement_slices=resolved_slices,
            worker_alias=spec.worker_alias,
            worker_aliases=spec.worker_aliases,
            placement_alias_slices=spec.placement_alias_slices,
            gpu_count=spec.gpu_count,
            enabled=spec.enabled,
        )

    def _node_metric_spec_from_runtime_node(self, alias: str, node: dict[str, Any]) -> NodeMetricsDaemonSpec | None:
        node_ip = str(node.get("node_ip") or "").strip()
        if str(node.get("state") or "") != "ready" or not node_ip:
            return None
        if node.get("enabled") is False:
            return None
        role = str(node.get("role") or "gpu")
        if role not in {"gpu", "head"}:
            return None
        gpu_count = node.get("gpu_count")
        if role == "gpu" and gpu_count is not None and int(gpu_count) <= 0:
            return None
        if node.get("mount_ok") is False or node.get("runtime_env_ok") is False:
            return None
        topology = self._topology_manager.snapshot() if self._topology_manager is not None else {}
        deployment_env = str(topology.get("deployment_env") or os.environ.get("MINT_DEPLOYMENT_ENV") or "").strip()
        cluster_id = str(topology.get("cluster_id") or os.environ.get("MINT_CLUSTER_ID") or "").strip()
        return NodeMetricsDaemonSpec(
            worker_alias=str(alias),
            node_ip=node_ip,
            ray_node_id=None if node.get("ray_node_id") is None else str(node.get("ray_node_id")),
            gpu_count=None if gpu_count is None else int(gpu_count),
            deployment_env=deployment_env or None,
            cluster_id=cluster_id or None,
            actor_name=node_metrics_actor_name(str(alias)),
            is_head_node=bool(node.get("is_head_node")),
        )

    async def _reconcile_node_metrics_daemons(self) -> None:
        if not self._node_metrics_enabled or self._node_metrics_factory is None:
            self._node_metric_states = {}
            return
        topology = self._topology_manager.snapshot() if self._topology_manager is not None else {}
        topology_nodes = topology.get("nodes") if isinstance(topology, dict) else {}
        if not isinstance(topology_nodes, dict):
            topology_nodes = {}
        desired_specs: dict[str, NodeMetricsDaemonSpec] = {}
        for alias, raw_node in sorted(topology_nodes.items()):
            if not isinstance(raw_node, dict):
                continue
            spec = self._node_metric_spec_from_runtime_node(str(alias), raw_node)
            if spec is not None:
                desired_specs[str(alias)] = spec

        for alias in sorted(set(self._node_metric_actors) - set(desired_specs)):
            actor = self._node_metric_actors.pop(alias, None)
            previous = dict(self._node_metric_states.get(alias, {}))
            try:
                if actor is not None:
                    await _invoke_actor(actor, "shutdown")
            except Exception as e:
                previous["last_error"] = f"{type(e).__name__}: {e}"
            previous.update(
                {
                    "worker_alias": alias,
                    "state": "stale",
                    "last_action": "removed_from_topology",
                    "last_action_at": time.time(),
                }
            )
            self._node_metric_states[alias] = previous

        for alias, spec in sorted(desired_specs.items()):
            actor = self._node_metric_actors.get(alias)
            if actor is not None:
                try:
                    health = await _invoke_actor(actor, "health_snapshot")
                    if not isinstance(health, dict):
                        raise TypeError(f"node metrics health_snapshot returned {type(health)}")
                    spec_mismatch = (
                        str(health.get("node_ip") or "") != spec.node_ip
                        or (health.get("ray_node_id") or None) != spec.ray_node_id
                        or bool(health.get("is_head_node")) != bool(spec.is_head_node)
                    )
                    if spec_mismatch:
                        await _invoke_actor(actor, "shutdown")
                        self._node_metric_actors.pop(alias, None)
                        self._node_metric_states[alias] = {
                            **self._node_metric_states.get(alias, {}),
                            "worker_alias": alias,
                            "node_ip": spec.node_ip,
                            "ray_node_id": spec.ray_node_id,
                            "actor_name": spec.normalized_actor_name(),
                            "state": "stale",
                            "last_error": None,
                            "last_action": "spec_changed_recreate",
                            "last_action_at": time.time(),
                        }
                        actor = None
                except Exception as e:
                    self._node_metrics_reconcile_failures_total += 1
                    self._node_metric_actors.pop(alias, None)
                    self._node_metric_states[alias] = {
                        **self._node_metric_states.get(alias, {}),
                        "worker_alias": alias,
                        "node_ip": spec.node_ip,
                        "ray_node_id": spec.ray_node_id,
                        "actor_name": spec.normalized_actor_name(),
                        "state": "dead",
                        "last_error": f"{type(e).__name__}: {e}",
                        "last_action": "spec_check_failed",
                        "last_action_at": time.time(),
                    }
                    actor = None
            if actor is None:
                try:
                    actor = await _maybe_await(self._node_metrics_factory(spec))
                    self._node_metric_actors[alias] = actor
                    self._node_metrics_created_total += 1
                    self._node_metric_states[alias] = {
                        "worker_alias": alias,
                        "node_ip": spec.node_ip,
                        "ray_node_id": spec.ray_node_id,
                        "actor_name": spec.normalized_actor_name(),
                        "state": "starting",
                        "last_error": None,
                        "last_action": "create",
                        "last_action_at": time.time(),
                    }
                except Exception as e:
                    self._node_metrics_reconcile_failures_total += 1
                    self._node_metric_states[alias] = {
                        "worker_alias": alias,
                        "node_ip": spec.node_ip,
                        "ray_node_id": spec.ray_node_id,
                        "actor_name": spec.normalized_actor_name(),
                        "state": "failed",
                        "last_error": f"{type(e).__name__}: {e}",
                        "last_action": "create_failed",
                        "last_action_at": time.time(),
                    }
                    continue
            try:
                health = await _invoke_actor(actor, "health_snapshot")
                if not isinstance(health, dict):
                    raise TypeError(f"node metrics health_snapshot returned {type(health)}")
                sample = await _invoke_actor(actor, "sample_cached")
                if isinstance(sample, dict):
                    health = {**health, "last_sample": sample}
                running = bool(health.get("running", True))
                self._node_metric_states[alias] = {
                    **self._node_metric_states.get(alias, {}),
                    "worker_alias": alias,
                    "node_ip": spec.node_ip,
                    "ray_node_id": spec.ray_node_id,
                    "actor_name": spec.normalized_actor_name(),
                    "state": "healthy" if running else "unhealthy",
                    "health": health,
                    "last_error": None if running else "node metrics daemon not running",
                    "last_action": "health_check",
                    "last_action_at": time.time(),
                }
            except Exception as e:
                self._node_metrics_reconcile_failures_total += 1
                self._node_metric_actors.pop(alias, None)
                self._node_metric_states[alias] = {
                    **self._node_metric_states.get(alias, {}),
                    "worker_alias": alias,
                    "node_ip": spec.node_ip,
                    "ray_node_id": spec.ray_node_id,
                    "actor_name": spec.normalized_actor_name(),
                    "state": "dead",
                    "last_error": f"{type(e).__name__}: {e}",
                    "last_action": "health_failed",
                    "last_action_at": time.time(),
                }

    async def _actor_health(self, actor: Any) -> dict[str, Any]:
        out = await _invoke_actor(actor, "health_snapshot")
        if not isinstance(out, dict):
            raise TypeError(f"runtime actor health_snapshot returned non-dict: {type(out)}")
        return out

    async def push_liveness(self, payload: EngineLivenessPush | dict[str, Any]) -> dict[str, Any]:
        push = payload if isinstance(payload, EngineLivenessPush) else EngineLivenessPush.from_wire(dict(payload))
        key = push.key
        current_generation = int(self._generations.get(key, 0) or 0)
        if current_generation > 0 and int(push.actor_generation) < current_generation:
            return {
                "ok": False,
                "domain_key": push.domain_key,
                "replica_id": push.replica_id,
                "actor_generation": int(push.actor_generation),
                "reason": "stale_generation",
            }
        self._latest_push[key] = push
        return {
            "ok": True,
            "domain_key": push.domain_key,
            "replica_id": push.replica_id,
            "actor_generation": int(push.actor_generation),
        }

    def _actor_generation_matches_current(
        self,
        key: tuple[str, str],
        actor: Any,
        health: dict[str, Any] | None = None,
    ) -> bool:
        expected = int(self._generations.get(key, 0))
        if expected <= 0:
            return True
        candidate: Any = None
        if isinstance(health, dict):
            candidate = health.get("actor_generation")
        if candidate is None:
            candidate = self._actor_generations.get(key)
        if candidate is None:
            candidate = getattr(actor, "generation", None)
        if candidate is None:
            return True
        try:
            return int(candidate) == expected
        except (TypeError, ValueError):
            return False

    def _next_generation(self, key: tuple[str, str]) -> int:
        previous = int(self._generations.get(key, 0))
        now = int(time.time())
        floor = max(now, previous + 1)
        try:
            return int(self._state_store.reserve_generation(self._state_generation_key(key), floor=floor))
        except Exception as e:
            self._state_store_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] state generation reserve failed key=%s error_type=%s error=%s",
                _label(key),
                type(e).__name__,
                e,
            )
            return floor

    async def _create_runtime(self, spec: ModelActorSpec, *, reason: str) -> Any:
        key = spec.key
        generation = self._next_generation(key)
        self._generations[key] = generation
        started_at = time.time()
        self._states[key] = {
            **self._states.get(key, {}),
            "domain_key": spec.domain_key,
            "replica_id": spec.replica_id,
            "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
            "state": "starting",
            "actor_name": spec.normalized_actor_name(),
            "launcher_key": spec.launcher_key,
            "generation": generation,
            "consumer_id": consumer_id_for_replica(spec.domain_key, spec.replica_id, generation),
            "crash_count": int(self._states.get(key, {}).get("crash_count", 0)),
            "started_at": started_at,
            "last_error": None,
            "last_action": f"reserve:{reason}",
            "last_action_at": started_at,
            "node_pins": spec.normalized_node_pins(),
            "gpu_count": spec.gpu_count,
            "scheduler_status": "starting",
        }
        await self._sync_scheduler(raise_on_error=True)
        if self._runtime_factory is not None:
            actor = await _maybe_await(self._runtime_factory(spec, generation))
        else:
            actor = await self._launcher_registry.launch(
                spec,
                generation,
                launcher_key=spec.launcher_key,
                ray_address=self._ray_address,
            )
        try:
            start_result = await _invoke_actor(actor, "start")
        except Exception as e:
            if not _is_ray_get_timeout_error(e):
                raise
            self._actors[key] = actor
            self._actor_generations[key] = generation
            self._created_total += 1
            if reason != "missing":
                self._restarted_total += 1
            self._states[key] = {
                **self._states.get(key, {}),
                "domain_key": spec.domain_key,
                "replica_id": spec.replica_id,
                "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                "state": "starting",
                "actor_name": spec.normalized_actor_name(),
                "launcher_key": spec.launcher_key,
                "generation": generation,
                "consumer_id": consumer_id_for_replica(spec.domain_key, spec.replica_id, generation),
                "crash_count": int(self._states.get(key, {}).get("crash_count", 0)),
                "started_at": started_at,
                "last_error": f"{type(e).__name__}: {e}",
                "last_action": f"start_pending:{reason}",
                "last_action_at": time.time(),
                "node_pins": spec.normalized_node_pins(),
                "gpu_count": spec.gpu_count,
                "scheduler_status": "starting",
            }
            try:
                await self._sync_scheduler(raise_on_error=True)
            except Exception:
                try:
                    await _invoke_actor(actor, "shutdown")
                except Exception as shutdown_error:
                    logger.warning(
                        "[model_actor_supervisor] runtime shutdown after pending-start scheduler sync failure failed: %s: %s",
                        type(shutdown_error).__name__,
                        shutdown_error,
                    )
                self._actors.pop(key, None)
                self._actor_generations.pop(key, None)
                raise
            return actor
        if isinstance(start_result, dict) and start_result.get("running") is False:
            raise RuntimeError(f"runtime actor did not start: {start_result!r}")
        self._actors[key] = actor
        self._actor_generations[key] = generation
        self._generations[key] = generation
        self._created_total += 1
        if reason != "missing":
            self._restarted_total += 1
        try:
            self._state_store.append_event(
                "runtime_created",
                {
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "actor_name": spec.normalized_actor_name(),
                    "generation": generation,
                    "reason": reason,
                },
                owner=self._state_owner,
            )
        except Exception as e:
            self._state_store_failures_total += 1
            logger.debug("[model_actor_supervisor] state event append failed: %s: %s", type(e).__name__, e)
        self._states[key] = {
            "domain_key": spec.domain_key,
            "replica_id": spec.replica_id,
            "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
            "state": "starting",
            "actor_name": spec.normalized_actor_name(),
            "launcher_key": spec.launcher_key,
            "generation": generation,
            "consumer_id": consumer_id_for_replica(spec.domain_key, spec.replica_id, generation),
            "crash_count": int(self._states.get(key, {}).get("crash_count", 0)),
            "started_at": started_at,
            "last_error": None,
            "last_action": f"create_pending_liveness:{reason}",
            "last_action_at": time.time(),
            "node_pins": spec.normalized_node_pins(),
            "gpu_count": spec.gpu_count,
            "scheduler_status": "starting",
        }
        try:
            await self._sync_scheduler(raise_on_error=True)
        except Exception:
            try:
                await _invoke_actor(actor, "shutdown")
            except Exception as shutdown_error:
                logger.warning(
                    "[model_actor_supervisor] runtime shutdown after scheduler sync failure failed: %s: %s",
                    type(shutdown_error).__name__,
                    shutdown_error,
                )
            self._actors.pop(key, None)
            self._actor_generations.pop(key, None)
            raise
        return actor

    async def _ensure_runtime_placement_group(self, spec: ModelActorSpec) -> dict[str, Any] | None:
        controller = self._placement_controller
        if controller is None:
            return None
        try:
            from .cluster_placement_controller import placement_group_bundle_request_for_spec

            create_request = placement_group_bundle_request_for_spec(spec).to_create_request()
        except ValueError:
            return None
        result = await controller.create_pg(create_request)
        if result.status is PlacementGroupCreateStatus.READY:
            self._placement_groups_created_total += 1
            return {
                "ok": True,
                "placement_group_name": result.placement_group_name,
            }
        return {
            "ok": False,
            "placement_group_name": result.placement_group_name,
            "reason": None if result.reason is None else result.reason.value,
            "message": result.message,
            "retry_at": result.retry_at,
        }

    def _runtime_placement_blocked_state(
        self,
        spec: ModelActorSpec,
        *,
        original_spec: ModelActorSpec,
        resolved_node_pins: list[str],
        pg_result: dict[str, Any],
    ) -> dict[str, Any]:
        message = str(pg_result.get("message") or pg_result.get("reason") or "placement group create blocked")
        placement_group_name = str(pg_result.get("placement_group_name") or "")
        if placement_group_name:
            message = f"{placement_group_name}: {message}"
        return {
            **self._states.get(spec.key, {}),
            "domain_key": spec.domain_key,
            "replica_id": spec.replica_id,
            "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
            "state": "blocked",
            "actor_name": spec.normalized_actor_name(),
            "launcher_key": spec.launcher_key,
            "node_pins": resolved_node_pins,
            "worker_aliases": original_spec.normalized_worker_aliases(),
            "gpu_count": spec.gpu_count,
            "last_error": f"placement group blocked: {message}",
            "last_action": "blocked:placement_group",
            "last_action_at": time.time(),
            "placement_group_name": placement_group_name or None,
            "placement_retry_at": pg_result.get("retry_at"),
            "scheduler_status": "blocked",
        }

    def _replica_registration_for_state(
        self,
        spec: ModelActorSpec,
        state: dict[str, Any],
    ) -> ModelReplicaRegistration:
        generation = int(state.get("generation") or self._generations.get(spec.key, 0) or 0)
        status = str(state.get("state") or "starting")
        if status == "running":
            status = "healthy"
        if status in {"blocked", "disabled", "dead", "unhealthy"}:
            claim_status = status
        else:
            claim_status = "healthy" if status == "healthy" else "starting"
        return ModelReplicaRegistration(
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            consumer_id=str(
                state.get("consumer_id")
                or consumer_id_for_replica(spec.domain_key, spec.replica_id, generation)
            ),
            generation=generation,
            status=claim_status,
            queue_id=queue_id_for_replica(spec.domain_key, spec.replica_id),
            capacity=max(1, int(spec.gpu_count or 1)),
            actor_name=spec.normalized_actor_name(),
            node_pins=spec.normalized_node_pins(),
            updated_at=float(state.get("last_action_at") or time.time()),
        )

    async def _sync_scheduler(self, *, raise_on_error: bool = False) -> bool:
        registrations = [
            self._replica_registration_for_state(
                self._spec_with_resolved_topology(
                    spec,
                    node_pins=[
                        str(pin)
                        for pin in (self._states.get(key, {}).get("node_pins") or [])
                        if str(pin).strip()
                    ]
                    or None,
                ),
                self._states.get(key, {}),
            )
            for key, spec in sorted(self._desired.items())
        ]
        try:
            if self._scheduler_sync is not None:
                await _maybe_await(self._scheduler_sync(registrations))
            else:
                await self._scheduler.sync_replicas(
                    registrations,
                    hydrate_task_state=False,
                )
            self._last_scheduler_sync_at = time.time()
            return True
        except Exception as e:
            self._scheduler_sync_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] scheduler sync failed: %s: %s",
                type(e).__name__,
                e,
            )
            if raise_on_error:
                raise
            return False

    async def sync_replicas(self) -> dict[str, Any]:
        await self._sync_scheduler()
        return {"ok": True, "last_scheduler_sync_at": self._last_scheduler_sync_at}

    async def reconcile_once(self) -> dict[str, Any]:
        if self._reconcile_inflight:
            now = time.time()
            started_at = self._reconcile_inflight_started_at
            stale_after_s = max(30.0, float(self._reconcile_interval_s) * 3.0)
            if started_at is None or now - float(started_at) <= stale_after_s:
                return {
                    "ok": True,
                    "skipped": "reconcile_inflight",
                    "reconcile_inflight_started_at": started_at,
                    "snapshot": self.snapshot(),
                }
            self._reconcile_inflight = False
            self._reconcile_inflight_started_at = None
            self._last_reconcile_loop_error = (
                f"reconcile_inflight stale for {now - float(started_at):.1f}s; forcing new reconcile"
            )
        self._reconcile_inflight = True
        self._reconcile_inflight_started_at = time.time()
        try:
            return await self._reconcile_once_impl()
        finally:
            self._reconcile_inflight = False
            self._reconcile_inflight_started_at = None

    async def _reconcile_once_impl(self) -> dict[str, Any]:
        self._reconcile_total += 1
        self._last_reconcile_at = time.time()
        self._heartbeat_state_owner()
        await self._ensure_control_plane_dependencies()
        await self._sync_active_scheduler_domains()
        if self._orphan_pg_cleaner is not None:
            await _maybe_await(self._orphan_pg_cleaner(dict(self._desired)))
        topology_out: dict[str, Any] = {}
        resolved_desired: dict[tuple[str, str], ModelActorSpec] = dict(self._desired)
        if self._topology_resolver is not None:
            try:
                candidate = await _maybe_await(self._topology_resolver(dict(self._desired)))
                topology_out = candidate if isinstance(candidate, dict) else {"ok": True, "result": candidate}
                self._last_topology_reconcile = dict(topology_out)
            except Exception as e:
                self._topology_reconcile_failures_total += 1
                topology_out = {"ok": False, "error": f"{type(e).__name__}: {e}", "blocked": {}}
                self._last_topology_reconcile = dict(topology_out)
                logger.warning(
                    "[model_actor_supervisor] topology reconcile failed: %s: %s",
                    type(e).__name__,
                    e,
                )
        await self._reconcile_node_metrics_daemons()
        topology_blocked = topology_out.get("blocked") if isinstance(topology_out, dict) else {}
        if not isinstance(topology_blocked, dict):
            topology_blocked = {}
        topology_node_pins = topology_out.get("node_pins") if isinstance(topology_out, dict) else {}
        if not isinstance(topology_node_pins, dict):
            topology_node_pins = {}
        topology_placement_slices = topology_out.get("placement_slices") if isinstance(topology_out, dict) else {}
        if not isinstance(topology_placement_slices, dict):
            topology_placement_slices = {}
        for key, spec in list(resolved_desired.items()):
            label = _label(key)
            if label in topology_node_pins or label in topology_placement_slices:
                resolved_desired[key] = self._spec_with_resolved_topology(
                    spec,
                    node_pins=topology_node_pins.get(label),
                    placement_slices=topology_placement_slices.get(label),
                )
        placement_out: dict[str, Any] = {}
        if self._placement_controller is not None:
            try:
                protected_actor_names = self._reconcile_protected_actor_names(resolved_desired)
                candidate = await self._placement_controller.reconcile(
                    PlacementReconcileRequest(
                        desired=dict(resolved_desired),
                        protected_actor_names=frozenset(protected_actor_names),
                    )
                )
                if isinstance(candidate, PlacementReconcileResult):
                    placement_out = candidate.to_legacy_dict()
                elif isinstance(candidate, dict):
                    placement_out = candidate
                else:
                    placement_out = {"ok": True, "result": candidate}
                self._last_placement_reconcile = dict(placement_out)
            except Exception as e:
                self._placement_reconcile_failures_total += 1
                placement_out = {"ok": False, "error": f"{type(e).__name__}: {e}", "blocked": {}}
                self._last_placement_reconcile = dict(placement_out)
                logger.warning(
                    "[model_actor_supervisor] placement controller reconcile failed: %s: %s",
                    type(e).__name__,
                    e,
                )
        elif self._placement_reconciler is not None:
            try:
                protected_actor_names = self._reconcile_protected_actor_names(resolved_desired)
                try:
                    candidate = self._placement_reconciler(
                        dict(resolved_desired),
                        protected_actor_names=protected_actor_names,
                    )
                except TypeError as e:
                    if "protected_actor_names" not in str(e):
                        raise
                    candidate = self._placement_reconciler(dict(resolved_desired))
                candidate = await _maybe_await(candidate)
                placement_out = candidate if isinstance(candidate, dict) else {"ok": True, "result": candidate}
                self._last_placement_reconcile = dict(placement_out)
                self._placement_reclaimed_total += int(placement_out.get("reclaimed_total") or 0)
            except Exception as e:
                self._placement_reconcile_failures_total += 1
                placement_out = {"ok": False, "error": f"{type(e).__name__}: {e}", "blocked": {}}
                self._last_placement_reconcile = dict(placement_out)
                logger.warning(
                    "[model_actor_supervisor] placement reconcile failed: %s: %s",
                    type(e).__name__,
                    e,
                )
        placement_blocked = placement_out.get("blocked") if isinstance(placement_out, dict) else {}
        if not isinstance(placement_blocked, dict):
            placement_blocked = {}
        placement_node_pins = placement_out.get("node_pins") if isinstance(placement_out, dict) else {}
        if not isinstance(placement_node_pins, dict):
            placement_node_pins = {}

        if self._desired:
            await self._sync_scheduler()

        results: dict[str, Any] = {}
        for key, original_spec in sorted(self._desired.items()):
            spec = resolved_desired.get(key, original_spec)
            label = _label(key)
            resolved_node_pins = [
                str(pin)
                for pin in placement_node_pins.get(label, spec.normalized_node_pins())
                if str(pin).strip()
            ]
            if not spec.enabled:
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "disabled",
                    "actor_name": spec.normalized_actor_name(),
                    "last_action": "disabled",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            topology_error = topology_blocked.get(label)
            if topology_error:
                self._blocked_total += 1
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "blocked",
                    "actor_name": spec.normalized_actor_name(),
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                    "last_error": f"topology blocked: {topology_error}",
                    "last_action": "blocked:topology",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            placement_error = placement_blocked.get(label)
            if placement_error:
                self._blocked_total += 1
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "blocked",
                    "actor_name": spec.normalized_actor_name(),
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                    "last_error": f"placement blocked: {placement_error}",
                    "last_action": "blocked:placement",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            if not await self._node_pins_possible(spec, resolved_node_pins=resolved_node_pins):
                self._blocked_total += 1
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "blocked",
                    "actor_name": spec.normalized_actor_name(),
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                    "last_error": f"node pin unavailable: {','.join(resolved_node_pins)}",
                    "last_action": "blocked:node_pin_unavailable",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            actor = self._actors.get(key)
            if actor is None:
                pg_result = await self._ensure_runtime_placement_group(spec)
                if pg_result is not None and not bool(pg_result.get("ok")):
                    self._blocked_total += 1
                    self._states[key] = self._runtime_placement_blocked_state(
                        spec,
                        original_spec=original_spec,
                        resolved_node_pins=resolved_node_pins,
                        pg_result=pg_result,
                    )
                    results[label] = self._states[key]
                    continue
                try:
                    await self._create_runtime(spec, reason="missing")
                except Exception as e:
                    previous = self._states.get(key, {})
                    crash_count = int(previous.get("crash_count", 0)) + 1
                    self._states[key] = {
                        **previous,
                        "domain_key": spec.domain_key,
                        "replica_id": spec.replica_id,
                        "state": "dead",
                        "actor_name": spec.normalized_actor_name(),
                        "crash_count": crash_count,
                        "last_error": f"{type(e).__name__}: {e}",
                        "last_action": "create_failed:missing",
                        "last_action_at": time.time(),
                        "node_pins": resolved_node_pins,
                        "worker_aliases": original_spec.normalized_worker_aliases(),
                        "gpu_count": spec.gpu_count,
                    }
                    results[label] = self._states[key]
                    continue
                self._states[key]["worker_aliases"] = original_spec.normalized_worker_aliases()
                results[label] = self._states[key]
                continue

            now = time.time()
            latest_push = self._latest_push.get(key)
            if latest_push is not None and not self._actor_generation_matches_current(
                key,
                actor,
                latest_push.to_wire(),
            ):
                self._latest_push.pop(key, None)
                latest_push = None

            stale_after_s = _liveness_stale_after_s(self._reconcile_interval_s)
            push_stale = _liveness_push_is_stale(
                latest_push,
                now=now,
                stale_after_s=stale_after_s,
            )
            gpu_busy = _liveness_push_gpu_busy(
                latest_push,
                threshold_percent=_liveness_gpu_busy_threshold_percent(),
            )
            previous = self._states.get(key, {})
            started_at = float(previous.get("started_at") or previous.get("last_action_at") or now)
            starting_age_s = max(0.0, now - started_at)
            initial_grace_s = _liveness_initial_grace_s(stale_after_s)
            push_ready = latest_push is not None and bool(latest_push.running) and bool(latest_push.engine_ready)
            push_startup_error = _liveness_push_requires_recreate(latest_push)
            if (
                str(previous.get("state") or "") == "starting"
                and not push_ready
                and not push_startup_error
                and (latest_push is None or bool(latest_push.running))
                and starting_age_s <= initial_grace_s
            ):
                generation = int(
                    getattr(latest_push, "actor_generation", None)
                    or self._generations.get(key, 0)
                    or previous.get("generation")
                    or 0
                )
                self._states[key] = {
                    **previous,
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                    "state": "starting",
                    "actor_name": spec.normalized_actor_name(),
                    "launcher_key": spec.launcher_key,
                    "generation": generation,
                    "consumer_id": str(
                        getattr(latest_push, "consumer_id", None)
                        or previous.get("consumer_id")
                        or consumer_id_for_replica(
                            spec.domain_key,
                            spec.replica_id,
                            generation,
                        )
                    ),
                    "health": latest_push.to_wire() if latest_push is not None else previous.get("health"),
                    "started_at": started_at,
                    "last_error": None if latest_push is None else (latest_push.last_error or latest_push.engine_health.last_error),
                    "last_action": "awaiting_liveness" if latest_push is None else "awaiting_engine_ready",
                    "last_action_at": now,
                    "liveness_stale": bool(push_stale),
                    "liveness_stale_after_s": stale_after_s,
                    "liveness_startup_grace_s": initial_grace_s,
                    "liveness_startup_age_s": starting_age_s,
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                    "scheduler_status": "starting",
                }
                results[label] = self._states[key]
                continue
            health = latest_push.to_wire() if latest_push is not None else {}
            generation = int(
                getattr(latest_push, "actor_generation", None)
                or self._generations.get(key, 0)
                or 0
            )
            consumer_id = str(
                getattr(latest_push, "consumer_id", None)
                or consumer_id_for_replica(spec.domain_key, spec.replica_id, generation)
            )
            if push_stale and not gpu_busy:
                crash_count = int(previous.get("crash_count", 0)) + 1
                stale_age_s = None
                if latest_push is not None and latest_push.pushed_at is not None:
                    stale_age_s = max(0.0, now - float(latest_push.pushed_at))
                self._states[key] = {
                    **previous,
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "dead",
                    "actor_name": spec.normalized_actor_name(),
                    "crash_count": crash_count,
                    "last_error": (
                        "runtime liveness push missing"
                        if latest_push is None
                        else f"runtime liveness push stale age_s={stale_age_s:.1f}"
                    ),
                    "last_action": "liveness_stale_dead",
                    "last_action_at": now,
                    "liveness_stale_after_s": stale_after_s,
                    "liveness_stale_age_s": stale_age_s,
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                }
                self._actors.pop(key, None)
                self._actor_generations.pop(key, None)
                try:
                    pg_result = await self._ensure_runtime_placement_group(spec)
                    if pg_result is not None and not bool(pg_result.get("ok")):
                        self._blocked_total += 1
                        self._states[key] = self._runtime_placement_blocked_state(
                            spec,
                            original_spec=original_spec,
                            resolved_node_pins=resolved_node_pins,
                            pg_result=pg_result,
                        )
                        results[label] = self._states[key]
                        continue
                    await self._create_runtime(spec, reason="dead")
                    self._states[key]["crash_count"] = crash_count
                except Exception as create_error:
                    self._states[key] = {
                        **self._states.get(key, {}),
                        "domain_key": spec.domain_key,
                        "replica_id": spec.replica_id,
                        "state": "dead",
                        "actor_name": spec.normalized_actor_name(),
                        "crash_count": crash_count,
                        "last_error": f"{type(create_error).__name__}: {create_error}",
                        "last_action": "create_failed:dead",
                        "last_action_at": time.time(),
                        "node_pins": resolved_node_pins,
                        "worker_aliases": original_spec.normalized_worker_aliases(),
                        "gpu_count": spec.gpu_count,
                    }
                results[label] = self._states[key]
                continue

            if push_stale and gpu_busy:
                state = "healthy"
                last_action = "liveness_stale_gpu_busy"
                last_error = None
            elif latest_push is not None and bool(latest_push.running) and bool(latest_push.engine_ready):
                state = "healthy"
                last_action = "liveness_push"
                last_error = None
            else:
                state = "unhealthy"
                last_action = "liveness_unhealthy"
                last_error = (
                    "runtime actor not running"
                    if latest_push is None or not bool(latest_push.running)
                    else (latest_push.last_error or latest_push.engine_health.reason or "engine not ready")
                )

            crash_count = int(previous.get("crash_count", 0))
            if state == "unhealthy":
                crash_count += 1
            self._states[key] = {
                **previous,
                "domain_key": spec.domain_key,
                "replica_id": spec.replica_id,
                "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                "state": state,
                "actor_name": spec.normalized_actor_name(),
                "launcher_key": spec.launcher_key,
                "generation": generation,
                "consumer_id": consumer_id,
                "health": health,
                "crash_count": crash_count,
                "last_error": last_error,
                "last_action": last_action,
                "last_action_at": now,
                "liveness_stale": bool(push_stale),
                "liveness_gpu_busy": bool(gpu_busy),
                "liveness_stale_after_s": stale_after_s,
                "node_pins": resolved_node_pins,
                "worker_aliases": original_spec.normalized_worker_aliases(),
                "gpu_count": spec.gpu_count,
                "scheduler_status": state,
            }
            results[label] = self._states[key]

        await self._sync_scheduler()
        return {"ok": True, "replicas": results, "snapshot": self.snapshot()}

    async def ensure_reconcile_loop_started(self) -> dict[str, Any]:
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return self.snapshot()
        # Re-entrancy guard: ModelActorSupervisorCore has max_concurrency=128,
        # so two concurrent callers can both pass the task-done check above and
        # then both await adoption and both create_task → two orphaned reconcile
        # loops.  The flag collapses all concurrent late-comers to a no-op.
        if self._reconcile_loop_starting:
            return self.snapshot()
        self._reconcile_loop_starting = True
        try:
            # Fix D (#727): re-adopt still-alive mint GPU workers into inventory
            # BEFORE the first reconcile fires the reaper. The Ray state API can
            # crash large shared clusters when actor history is huge, so keep
            # startup adoption opt-in and leave the explicit method available
            # for controlled restart flows.
            if _adopt_surviving_gpu_actors_enabled():
                await asyncio.to_thread(self._adopt_surviving_gpu_actors)
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        finally:
            self._reconcile_loop_starting = False
        return self.snapshot()

    def _adopt_surviving_gpu_actors(self) -> None:
        """Re-register still-alive mint GPU actors into inventory after a restart.

        On supervisor control-plane restart the in-memory inventory is empty.
        Any still-alive dense (or vLLM/Megatron) worker listed by Ray would
        otherwise be reaped by _cleanup_undesired_mint_gpu_actors on the first
        reconcile.  Calling this before the reconcile loop starts ensures those
        actors appear in _reconcile_protected_actor_names and are left alone.

        The method is idempotent: it skips actors already in the inventory.
        Errors are caught and logged — adoption failure must never block startup.

        Protection semantics: adopted actors enter the inventory and are therefore
        shielded by _reconcile_protected_actor_names for their lifetime — the
        SAME protection a normally-created dense/vLLM actor receives via its
        registration.  This is parity with existing behavior, not a new
        permanent-protection path; if the actor later disappears from Ray it will
        be pruned from the inventory just like any other registered actor.
        """
        namespace = _ray_namespace()
        try:
            actors = list(_default_gpu_actor_lister())
        except Exception as exc:
            logger.warning(
                "[model_actor_supervisor] _adopt_surviving_gpu_actors: lister failed"
                " error_type=%s error=%s; skipping adoption",
                type(exc).__name__,
                exc,
            )
            return

        adopted = 0
        skipped_already_registered = 0
        skipped_namespace = 0
        skipped_non_mint = 0

        for actor_info in actors:
            if not isinstance(actor_info, dict):
                continue
            name = str(actor_info.get("name") or "").strip()
            if not name:
                continue
            if not _is_mint_gpu_actor_name(name):
                skipped_non_mint += 1
                continue
            actor_ns = str(actor_info.get("namespace") or namespace).strip() or namespace
            if actor_ns != namespace:
                skipped_namespace += 1
                continue
            if self.get(name) is not None:
                skipped_already_registered += 1
                continue

            # Derive ActorType from the name prefix.
            if name.startswith("mint_dense_"):
                actor_type = ActorType.DENSE
            elif name.startswith("mint_vllm_"):
                actor_type = ActorType.VLLM
            elif name.startswith("mint_megatron_"):
                actor_type = ActorType.MEGATRON
            elif name.startswith(("mint_openpi_shared_", "openpi_shared_runtime_", "mint_openpi_action_")):
                actor_type = ActorType.OPENPI
            else:
                # mint_model_runtime_* wrapper actors are supervisor-owned and
                # should not be adopted into the inventory directly.
                continue

            # num_gpus: use the lister record when available.
            try:
                num_gpus = max(1, int(float(actor_info.get("gpu") or 1)))
            except (TypeError, ValueError):
                num_gpus = 1

            node_id = str(actor_info.get("node_id") or "").strip() or None

            try:
                self.register(
                    actor_name=name,
                    actor_type=actor_type,
                    num_gpus=num_gpus,
                    namespace=actor_ns,
                    base_model="",
                    node_id=node_id,
                    protected=False,
                    metadata={"adopted_on_restart": True},
                )
                # The actor is already alive — mark it ready so it is not
                # treated as still-creating, which would suppress idle/reap logic.
                self.mark_ready(name)
                adopted += 1
                logger.info(
                    "[model_actor_supervisor] adopted surviving GPU actor"
                    " actor=%s type=%s num_gpus=%d namespace=%s node_id=%s",
                    name,
                    actor_type.value,
                    num_gpus,
                    actor_ns,
                    node_id,
                )
            except Exception as exc:
                logger.warning(
                    "[model_actor_supervisor] _adopt_surviving_gpu_actors:"
                    " failed to register actor=%s error_type=%s error=%s",
                    name,
                    type(exc).__name__,
                    exc,
                )

        logger.info(
            "[model_actor_supervisor] _adopt_surviving_gpu_actors done"
            " adopted=%d already_registered=%d skipped_namespace=%d skipped_non_mint=%d",
            adopted,
            skipped_already_registered,
            skipped_namespace,
            skipped_non_mint,
        )

    async def _reconcile_loop(self) -> None:
        interval_s = max(0.1, float(self._reconcile_interval_s))
        while True:
            try:
                await self.reconcile_once()
                self._last_reconcile_loop_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._last_reconcile_loop_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "[model_actor_supervisor] reconcile loop failed error_type=%s error=%s",
                    type(e).__name__,
                    e,
                )
            await asyncio.sleep(interval_s)

    async def recycle(self, *, domain_key: str, replica_id: str, force: bool = False) -> dict[str, Any]:
        key = _key(domain_key, replica_id)
        actor = self._actors.get(key)
        spec = self._desired.get(key)
        if spec is None:
            return {"ok": False, "domain_key": domain_key, "replica_id": replica_id, "reason": "unknown_replica"}
        if actor is None:
            return {"ok": True, "domain_key": domain_key, "replica_id": replica_id, "recycled": False, "reason": "missing"}
        health: dict[str, Any] = {}
        try:
            health = await self._actor_health(actor)
        except Exception:
            health = {}
        if not force and health.get("active_request_id"):
            self._busy_recycle_skipped_total += 1
            return {
                "ok": False,
                "domain_key": domain_key,
                "replica_id": replica_id,
                "reason": "busy",
                "active_request_id": health.get("active_request_id"),
            }
        try:
            await _invoke_actor(actor, "shutdown")
        except Exception as e:
            logger.warning(
                "[model_actor_supervisor] runtime shutdown failed domain=%s replica=%s error_type=%s error=%s",
                domain_key,
                replica_id,
                type(e).__name__,
                e,
            )
        self._actors.pop(key, None)
        self._actor_generations.pop(key, None)
        self._states[key] = {
            **self._states.get(key, {}),
            "domain_key": domain_key,
            "replica_id": replica_id,
            "state": "dead",
            "actor_name": spec.normalized_actor_name(),
            "last_action": "recycle",
            "last_action_at": time.time(),
        }
        await self._sync_scheduler()
        return {"ok": True, "domain_key": domain_key, "replica_id": replica_id, "recycled": True}

    def snapshot(self) -> dict[str, Any]:
        snapshot_generated_at = time.time()
        replicas: dict[str, dict[str, Any]] = {}
        domains: dict[str, dict[str, Any]] = {}
        for key in sorted(set(self._states) | set(self._desired)):
            spec = self._desired.get(key)
            state = dict(self._states.get(key, {}))
            domain_key, replica_id = key
            state.setdefault("domain_key", domain_key)
            state.setdefault("replica_id", replica_id)
            state.setdefault("queue_id", queue_id_for_replica(domain_key, replica_id))
            if spec is not None:
                if not state.get("node_pins"):
                    state["node_pins"] = spec.normalized_node_pins()
                if not state.get("worker_aliases"):
                    state["worker_aliases"] = spec.normalized_worker_aliases()
                state.update(
                    {
                        "base_model": spec.base_model,
                        "desired_actor_name": spec.normalized_actor_name(),
                        "desired_enabled": bool(spec.enabled),
                        "launcher_key": spec.launcher_key,
                        "gpu_count": spec.gpu_count,
                    }
                )
            replicas[_label(key)] = state
            domain = domains.setdefault(domain_key, {"replicas": 0, "healthy": 0, "unhealthy": 0})
            domain["replicas"] += 1
            if state.get("state") == "healthy":
                domain["healthy"] += 1
            elif state.get("state") in {"dead", "unhealthy", "blocked"}:
                domain["unhealthy"] += 1
        topology_snapshot = self._topology_manager.snapshot() if self._topology_manager is not None else {}
        topology_nodes = topology_snapshot.get("nodes", {}) if isinstance(topology_snapshot, dict) else {}
        if not isinstance(topology_nodes, dict):
            topology_nodes = {}
        node_metrics_desired_total = len(
            [
                alias
                for alias, node in topology_nodes.items()
                if isinstance(node, dict) and self._node_metric_spec_from_runtime_node(str(alias), node) is not None
            ]
        )
        return {
            "snapshot_generated_at": snapshot_generated_at,
            "observed_at": snapshot_generated_at,
            "actor_name": _ray_model_actor_supervisor_actor_name(),
            "namespace": _ray_namespace(),
            "code_identity": CURRENT_CODE_IDENTITY,
            "desired_total": int(len(self._desired)),
            "managed_total": int(len(self._actors)),
            "domain_total": int(len(domains)),
            "reconcile_total": int(self._reconcile_total),
            "created_total": int(self._created_total),
            "restarted_total": int(self._restarted_total),
            "blocked_total": int(self._blocked_total),
            "busy_recycle_skipped_total": int(self._busy_recycle_skipped_total),
            "health_timeout_preserved_total": int(self._health_timeout_preserved_total),
            "scheduler_sync_failures_total": int(self._scheduler_sync_failures_total),
            "placement_reconcile_failures_total": int(self._placement_reconcile_failures_total),
            "topology_reconcile_failures_total": int(self._topology_reconcile_failures_total),
            "node_metrics_created_total": int(self._node_metrics_created_total),
            "node_metrics_reconcile_failures_total": int(self._node_metrics_reconcile_failures_total),
            "control_plane_ensure_failures_total": int(self._control_plane_ensure_failures_total),
            "state_store_failures_total": int(self._state_store_failures_total),
            "placement_reclaimed_total": int(self._placement_reclaimed_total),
            "placement_groups_created_total": int(self._placement_groups_created_total),
            "last_reconcile_at": self._last_reconcile_at,
            "reconcile_inflight": bool(self._reconcile_inflight),
            "reconcile_inflight_started_at": self._reconcile_inflight_started_at,
            "reconcile_loop_running": self._reconcile_task is not None and not self._reconcile_task.done(),
            "reconcile_interval_s": float(self._reconcile_interval_s),
            "last_reconcile_loop_error": self._last_reconcile_loop_error,
            "last_scheduler_sync_at": self._last_scheduler_sync_at,
            "last_placement_reconcile": self._last_placement_reconcile,
            "last_topology_reconcile": self._last_topology_reconcile,
            "liveness_pushes": {
                _label(key): push.to_wire()
                for key, push in sorted(self._latest_push.items())
            },
            "topology": topology_snapshot,
            "state_store": {
                "backend": self._state_store.backend,
                "db_path": self._state_store.db_path,
                "owner": self._state_store.owner_snapshot(name=self._owner_name),
            },
            "daemons": {
                "node_metrics": {
                    "enabled": bool(self._node_metrics_enabled),
                    "desired_total": node_metrics_desired_total,
                    "managed_total": len(self._node_metric_actors),
                    "nodes": {alias: dict(state) for alias, state in sorted(self._node_metric_states.items())},
                }
            },
            "control_plane": {
                "enabled": bool(self._control_plane_enabled),
                "dependencies": {
                    name: dict(state)
                    for name, state in sorted(self._control_plane_states.items())
                },
            },
            "domains": domains,
            "replicas": replicas,
        }

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        _ = timeout_s
        return self.snapshot()


ModelActorSupervisor = ModelActorSupervisorCore


def _key(domain_key: str, replica_id: str) -> tuple[str, str]:
    return str(domain_key), str(replica_id)


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _runtime_error_requires_recreate(error: str) -> bool:
    text = str(error or "").lower()
    return any(
        needle in text
        for needle in (
            "engine core initialization failed",
            "engine startup failed",
            "enginecore failed to start",
            "consumer_id mismatch",
        )
    )


def _liveness_stale_after_s(reconcile_interval_s: float) -> float:
    raw = str(os.environ.get("MINT_MODEL_RUNTIME_LIVENESS_STALE_AFTER_S") or "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except (TypeError, ValueError):
            logger.warning(
                "[model_actor_supervisor] invalid MINT_MODEL_RUNTIME_LIVENESS_STALE_AFTER_S=%r; using default",
                raw,
            )
    return max(30.0, float(reconcile_interval_s) * 3.0)


def _liveness_initial_grace_s(stale_after_s: float) -> float:
    raw = str(os.environ.get("MINT_MODEL_RUNTIME_INITIAL_LIVENESS_GRACE_S") or "").strip()
    if raw:
        try:
            return max(float(stale_after_s), float(raw))
        except (TypeError, ValueError):
            logger.warning(
                "[model_actor_supervisor] invalid MINT_MODEL_RUNTIME_INITIAL_LIVENESS_GRACE_S=%r; using default",
                raw,
            )
    return max(300.0, float(stale_after_s))


def _liveness_push_requires_recreate(push: EngineLivenessPush | None) -> bool:
    if push is None:
        return False
    return any(
        value is not None and _runtime_error_requires_recreate(str(value))
        for value in (
            push.last_error,
            push.engine_health.last_error,
            push.engine_health.reason,
        )
    )


def _liveness_gpu_busy_threshold_percent() -> float:
    raw = str(os.environ.get("MINT_MODEL_RUNTIME_LIVENESS_GPU_BUSY_THRESHOLD_PERCENT") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            logger.warning(
                "[model_actor_supervisor] invalid MINT_MODEL_RUNTIME_LIVENESS_GPU_BUSY_THRESHOLD_PERCENT=%r; using default",
                raw,
            )
    return 5.0


def _liveness_push_is_stale(
    push: EngineLivenessPush | None,
    *,
    now: float,
    stale_after_s: float,
) -> bool:
    if push is None:
        return True
    if push.pushed_at is None:
        return True
    return now - float(push.pushed_at) > float(stale_after_s)


def _liveness_push_gpu_busy(push: EngineLivenessPush | None, *, threshold_percent: float) -> bool:
    if push is None:
        return False
    samples = push.observability.gpu_performance
    for sample in samples:
        util = sample.utilization_percent
        if util is not None and float(util) >= float(threshold_percent):
            return True
    return False


def queue_id_for_replica(domain_key: str, replica_id: str) -> str:
    return f"{domain_key}::{replica_id}"


def consumer_id_for_replica(domain_key: str, replica_id: str, generation: int) -> str:
    return f"{domain_key}::{replica_id}::generation::{int(generation)}"


def _ray_model_actor_supervisor_actor_name() -> str:
    return str(
        os.environ.get("MINT_MODEL_ACTOR_SUPERVISOR_ACTOR_NAME")
        or getattr(server_config, "model_actor_supervisor_actor_name", MODEL_ACTOR_SUPERVISOR_ACTOR_NAME)
    )


def _model_actor_supervisor_actor_resources() -> dict[str, float] | None:
    try:
        import ray

        return preferred_control_plane_resources(ray.cluster_resources())
    except Exception:
        return None


def _kill_supervisor_actor(actor: Any, *, reason: str) -> None:
    from . import ray_kill

    actor_name = _ray_model_actor_supervisor_actor_name()
    namespace = _ray_namespace()
    logger.warning(
        "[model_actor_supervisor] killing detached actor reason=%s actor_name=%s namespace=%s",
        reason,
        actor_name,
        namespace,
    )
    ray_kill.kill(
        actor,
        reason=reason,
        actor_name=actor_name,
        namespace=namespace,
        no_restart=True,
        verify_absent=True,
        verify_timeout_s=15.0,
    )


def _create_ray_actor(*, require_ready: bool = True):
    try:
        import ray
    except Exception as e:
        raise ModelActorSupervisorUnavailableError("Ray import failed") from e

    actor_name = _ray_model_actor_supervisor_actor_name()
    max_concurrency = int(os.environ.get("MINT_MODEL_ACTOR_SUPERVISOR_ACTOR_MAX_CONCURRENCY", "128"))

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=0)
    class _RayModelActorSupervisorActor(ModelActorSupervisorCore):
        pass

    extra_env = otel_env_vars()
    if CURRENT_CODE_IDENTITY:
        extra_env["MINT_GIT_SHA"] = str(CURRENT_CODE_IDENTITY)
    if "MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM" in os.environ:
        extra_env["MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM"] = os.environ["MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM"]
    from ..ray_utils import strict_ray_gcs_address

    ray_address = strict_ray_gcs_address()
    if ray_address is None:
        raise RuntimeError("MINT_RAY_GCS_ADDRESS is required")

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(
            pythonpath=PFS_PYTHONPATH,
            extra=extra_env,
            include_ray_attach_hints=False,
        ),
    }
    resources = _model_actor_supervisor_actor_resources()
    if resources:
        options["resources"] = resources

    actor = _RayModelActorSupervisorActor.options(**options).remote(
        specs=desired_specs_from_env(),
        ray_address=ray_address,
    )
    if require_ready:
        out = sync_get_ray_ref(actor.snapshot.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
    return actor


class ModelActorSupervisorClient:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_ray_actor(self) -> None:
        self._ray_actor = None

    def _validate_code_identity(self, snapshot: dict[str, Any]) -> None:
        if not CURRENT_CODE_IDENTITY:
            return
        actor_code_identity = snapshot.get("code_identity")
        if actor_code_identity == CURRENT_CODE_IDENTITY:
            return
        raise ModelActorSupervisorCodeIdentityMismatchError(
            "model actor supervisor code identity mismatch: "
            f"expected={CURRENT_CODE_IDENTITY!r} actual={actor_code_identity!r}"
        )

    def _kill_cached_actor_for_code_identity_mismatch(self, exc: BaseException) -> None:
        actor = self._ray_actor
        self._reset_ray_actor()
        if actor is None:
            raise exc
        _kill_supervisor_actor(actor, reason="model_actor_supervisor_code_mismatch")

    def _get_ray_actor_sync(self, *, require_ready: bool = False):
        try:
            import ray
        except Exception as e:
            raise ModelActorSupervisorUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise ModelActorSupervisorUnavailableError("Ray not initialized")
        if self._ray_actor is not None:
            if not require_ready:
                return self._ray_actor
            try:
                out = sync_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
                self._validate_code_identity(out)
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        actor_name = _ray_model_actor_supervisor_actor_name()
        try:
            self._ray_actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception as e:
            raise ModelActorSupervisorUnavailableError(
                f"Detached Ray ModelActorSupervisor actor not found: {actor_name}"
            ) from e
        if require_ready:
            out = sync_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=5.0)
            if not isinstance(out, dict):
                raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
            self._validate_code_identity(out)
        return self._ray_actor

    async def _get_ray_actor_async(self, *, require_ready: bool = False):
        try:
            import ray
        except Exception as e:
            raise ModelActorSupervisorUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise ModelActorSupervisorUnavailableError("Ray not initialized")
        if self._ray_actor is not None:
            if not require_ready:
                return self._ray_actor
            try:
                out = await async_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
                self._validate_code_identity(out)
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        actor_name = _ray_model_actor_supervisor_actor_name()
        try:
            self._ray_actor = await asyncio.to_thread(
                ray.get_actor,
                actor_name,
                namespace=_ray_namespace(),
            )
        except Exception as e:
            raise ModelActorSupervisorUnavailableError(
                f"Detached Ray ModelActorSupervisor actor not found: {actor_name}"
            ) from e
        if require_ready:
            out = await async_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=5.0)
            if not isinstance(out, dict):
                raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
            self._validate_code_identity(out)
        return self._ray_actor

    def ensure_started(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        self._ray_actor = _create_ray_actor(require_ready=False)
        out = sync_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=float(timeout_s))
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
        try:
            self._validate_code_identity(out)
        except ModelActorSupervisorCodeIdentityMismatchError as e:
            self._kill_cached_actor_for_code_identity_mismatch(e)
            self._ray_actor = _create_ray_actor(require_ready=False)
        out = sync_get_ray_ref(self._ray_actor.ensure_reconcile_loop_started.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.ensure_reconcile_loop_started returned non-dict: {type(out)}")
        self._validate_code_identity(out)
        return out

    async def async_ensure_started(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        self._ray_actor = _create_ray_actor(require_ready=False)
        out = await async_get_ray_ref(
            self._ray_actor.snapshot.remote(),
            timeout_s=float(timeout_s),
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
        try:
            self._validate_code_identity(out)
        except ModelActorSupervisorCodeIdentityMismatchError as e:
            await asyncio.to_thread(self._kill_cached_actor_for_code_identity_mismatch, e)
            self._ray_actor = _create_ray_actor(require_ready=False)
        out = await async_get_ray_ref(
            self._ray_actor.ensure_reconcile_loop_started.remote(),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.ensure_reconcile_loop_started returned non-dict: {type(out)}")
        self._validate_code_identity(out)
        return out

    def _call_sync(
        self,
        method: str,
        *args: Any,
        ray_timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        actor = self._get_ray_actor_sync(require_ready=False)
        try:
            remote = getattr(actor, method).remote
            return sync_get_ray_ref(remote(*args, **kwargs), timeout_s=ray_timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise

    async def _call_async(
        self,
        method: str,
        *args: Any,
        ray_timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        actor = await self._get_ray_actor_async(require_ready=False)
        try:
            remote = getattr(actor, method).remote
            return await async_get_ray_ref(remote(*args, **kwargs), timeout_s=ray_timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise

    def register(self, **kwargs: Any) -> ActorEntry:
        return self._call_sync("register", **kwargs)

    def unregister(self, actor_name: str) -> bool:
        return bool(self._call_sync("unregister", actor_name))

    def get(self, actor_name: str) -> ActorEntry | None:
        return self._call_sync("get", actor_name)

    def set_session(self, actor_name: str, session_id: str | None) -> None:
        self._call_sync("set_session", actor_name, session_id)

    async def async_set_session(self, actor_name: str, session_id: str | None) -> None:
        await self._call_async("async_set_session", actor_name, session_id)

    def set_protected(self, actor_name: str, protected: bool = True) -> bool:
        return bool(self._call_sync("set_protected", actor_name, protected))

    def is_protected(self, actor_name: str) -> bool:
        return bool(self._call_sync("is_protected", actor_name))

    def touch(self, actor_name: str) -> bool:
        return bool(self._call_sync("touch", actor_name))

    async def async_touch(self, actor_name: str) -> bool:
        return bool(await self._call_async("async_touch", actor_name))

    def mark_inflight(self, actor_name: str, delta: int) -> None:
        self._call_sync("mark_inflight", actor_name, int(delta))

    def mark_ready(self, actor_name: str) -> None:
        self._call_sync("mark_ready", actor_name)

    def update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return bool(
            self._call_sync(
                "update_metadata",
                actor_name,
                metadata=metadata,
                sample_time=sample_time,
                sample_source=sample_source,
            )
        )

    async def async_update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return bool(
            await self._call_async(
                "async_update_metadata",
                actor_name,
                metadata=metadata,
                sample_time=sample_time,
                sample_source=sample_source,
            )
        )

    def list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        out = self._call_sync(
            "list_actors",
            refresh_metadata=refresh_metadata,
            actor_type=actor_type,
            model_name=model_name,
        )
        return list(out)

    async def async_list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        out = await self._call_async(
            "async_list_actors",
            refresh_metadata=refresh_metadata,
            actor_type=actor_type,
            model_name=model_name,
        )
        return list(out)

    def metadata_cache_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return list(self._call_sync("metadata_cache_metrics_snapshot"))

    def lifecycle_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return list(self._call_sync("lifecycle_metrics_snapshot"))

    def cached_snapshot(self) -> list[dict[str, Any]]:
        return list(self._call_sync("cached_snapshot"))

    def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
        return list(self._call_sync("rss_snapshot", timeout_s=timeout_s))

    def iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        return list(self._call_sync("iter_entries", prune_stale=prune_stale))

    async def async_iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        return list(await self._call_async("async_iter_entries", prune_stale=prune_stale))

    def clear_session(self, session_id: str, *, actor_type: ActorType | None = None) -> int:
        return int(self._call_sync("clear_session", session_id, actor_type=actor_type))

    def total_gpus_used(self) -> int:
        return int(self._call_sync("total_gpus_used"))

    async def async_total_gpus_used(self) -> int:
        return int(await self._call_async("async_total_gpus_used"))

    def gpus_used_by_node(self) -> dict[str, int]:
        return dict(self._call_sync("gpus_used_by_node"))

    def clear(self, kill_actors: bool = True) -> int:
        return int(self._call_sync("clear", kill_actors=kill_actors))

    def set_desired(self, spec: ModelActorSpec) -> None:
        self._call_sync("set_desired", spec)

    def remove_desired(self, *, domain_key: str, replica_id: str) -> None:
        self._call_sync("remove_desired", domain_key=domain_key, replica_id=replica_id)

    async def sync_replicas(self) -> dict[str, Any]:
        out = await self._call_async("sync_replicas")
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.sync_replicas returned non-dict: {type(out)}")
        return out

    async def reconcile_once(self) -> dict[str, Any]:
        out = await self._call_async("reconcile_once")
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.reconcile_once returned non-dict: {type(out)}")
        return out

    async def recycle(self, *, domain_key: str, replica_id: str, force: bool = False) -> dict[str, Any]:
        out = await self._call_async(
            "recycle",
            domain_key=domain_key,
            replica_id=replica_id,
            force=force,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.recycle returned non-dict: {type(out)}")
        return out

    async def push_liveness(self, payload: EngineLivenessPush | dict[str, Any]) -> dict[str, Any]:
        wire = payload.to_wire() if isinstance(payload, EngineLivenessPush) else dict(payload)
        out = await self._call_async("push_liveness", wire)
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.push_liveness returned non-dict: {type(out)}")
        return out

    def snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        out = self._call_sync("snapshot", ray_timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.snapshot returned non-dict: {type(out)}")
        return out

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        out = await self._call_async(
            "async_snapshot",
            timeout_s=timeout_s,
            ray_timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelActorSupervisor.async_snapshot returned non-dict: {type(out)}")
        return out


model_actor_supervisor = ModelActorSupervisorClient()


def get_model_actor_supervisor() -> ModelActorSupervisorClient:
    return model_actor_supervisor


def ensure_started(*, timeout_s: float = 10.0) -> dict[str, Any]:
    return model_actor_supervisor.ensure_started(timeout_s=timeout_s)


async def async_ensure_started(*, timeout_s: float = 10.0) -> dict[str, Any]:
    return await model_actor_supervisor.async_ensure_started(timeout_s=timeout_s)
