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
from typing import Any, Awaitable, Callable

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
from .model_actor_placement import model_actor_placement_reconciler
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


def domain_key_for_training_base_model(base_model: str) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    try:
        from .model_registry import get_model_config

        if bool(getattr(get_model_config(model), "is_moe", False)):
            return f"megatron:{_normalize_megatron_domain_key(model)}"
    except Exception:
        logger.debug("training domain model config lookup failed for %s", model, exc_info=True)
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
        for launcher_key, domain_key in (
            ("vllm", domain_key_for_vllm_base_model(base_model)),
            ("training", domain_key_for_training_base_model(base_model)),
            ("megatron", domain_key_for_training_base_model(base_model)),
        ):
            raw_launcher_cfg = raw_cfg.get(launcher_key)
            if raw_launcher_cfg is None:
                continue
            launcher_cfg = raw_launcher_cfg if isinstance(raw_launcher_cfg, dict) else {}
            if launcher_key == "megatron" and not domain_key.startswith("megatron:"):
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
            spec_launcher = "training" if launcher_key == "megatron" else launcher_key
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
    if domain.startswith("megatron:"):
        for spec in supported.values():
            if spec.domain_key == domain:
                return spec
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

        return await task_state_store.async_ensure_ready(timeout_s=5.0, create_if_missing=True)

    async def _ping_task_state_store() -> Any:
        from .task_state_store import task_state_store

        return await task_state_store.async_ping(timeout_s=5.0)

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
    try:
        return await async_get_ray_ref(value, timeout_s=10.0)
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
    ) -> None:
        self._desired: dict[tuple[str, str], ModelActorSpec] = {}
        self._launcher_registry = launcher_registry or default_model_actor_launcher_registry()
        self._runtime_factory = runtime_factory
        self._node_inventory = node_inventory
        self._scheduler = scheduler or model_work_scheduler
        self._scheduler_sync = scheduler_sync
        self._scheduler_stats = scheduler_stats
        self._orphan_pg_cleaner = orphan_pg_cleaner
        self._placement_reconciler = placement_reconciler or model_actor_placement_reconciler
        self._topology_manager = topology_manager if topology_manager is not None else TopologyManager()
        self._topology_resolver = topology_resolver or self._resolve_topology_placements
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
        self._generations: dict[tuple[str, str], int] = {}
        self._states: dict[tuple[str, str], dict[str, Any]] = {}
        self._reconcile_total = 0
        self._created_total = 0
        self._restarted_total = 0
        self._blocked_total = 0
        self._busy_recycle_skipped_total = 0
        self._scheduler_sync_failures_total = 0
        self._placement_reconcile_failures_total = 0
        self._topology_reconcile_failures_total = 0
        self._placement_reclaimed_total = 0
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
        self._reconcile_inflight = False
        self._reconcile_inflight_started_at: float | None = None
        self._last_reconcile_loop_error: str | None = None
        for spec in specs or []:
            self.set_desired(spec)

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
        source: str | None = None,
    ) -> bool:
        return self._inventory.update_metadata(
            actor_name,
            metadata,
            sample_time=sample_time,
            source=source,
        )

    async def async_update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        source: str | None = None,
    ) -> bool:
        return await self._inventory.async_update_metadata(
            actor_name,
            metadata,
            sample_time=sample_time,
            source=source,
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

    def cached_snapshot(self) -> list[dict[str, Any]]:
        return self._inventory.cached_snapshot()

    def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
        return self._inventory.rss_snapshot(timeout_s=timeout_s)

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
        if self._runtime_factory is not None:
            actor = await _maybe_await(self._runtime_factory(spec, generation))
        else:
            actor = await self._launcher_registry.launch(spec, generation, launcher_key=spec.launcher_key)
        start_result = await _invoke_actor(actor, "start")
        if isinstance(start_result, dict) and start_result.get("running") is False:
            raise RuntimeError(f"runtime actor did not start: {start_result!r}")
        self._actors[key] = actor
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
            "state": "healthy",
            "actor_name": spec.normalized_actor_name(),
            "launcher_key": spec.launcher_key,
            "generation": generation,
            "consumer_id": consumer_id_for_replica(spec.domain_key, spec.replica_id, generation),
            "crash_count": int(self._states.get(key, {}).get("crash_count", 0)),
            "last_error": None,
            "last_action": f"create:{reason}",
            "last_action_at": time.time(),
            "node_pins": spec.normalized_node_pins(),
            "gpu_count": spec.gpu_count,
        }
        return actor

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

    async def _sync_scheduler(self) -> None:
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
                await self._scheduler.sync_replicas(registrations)
            self._last_scheduler_sync_at = time.time()
        except Exception as e:
            self._scheduler_sync_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] scheduler sync failed: %s: %s",
                type(e).__name__,
                e,
            )

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
        if self._placement_reconciler is not None:
            try:
                candidate = await _maybe_await(self._placement_reconciler(dict(resolved_desired)))
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
                await self._create_runtime(spec, reason="missing")
                self._states[key]["worker_aliases"] = original_spec.normalized_worker_aliases()
                results[label] = self._states[key]
                continue

            try:
                health = await self._actor_health(actor)
                runtime_last_error = str(health.get("last_error") or "").strip()
                runtime_failed_total = int(health.get("failed_total") or 0)
                runtime_completed_total = int(health.get("completed_total") or 0)
                runtime_processed_total = int(health.get("processed_total") or 0)
                health_error_unrecovered = bool(
                    _runtime_error_requires_recreate(runtime_last_error)
                    and runtime_failed_total > 0
                    and runtime_completed_total <= 0
                    and runtime_processed_total <= runtime_failed_total
                )
                state = (
                    "healthy"
                    if bool(health.get("running", True)) and not health_error_unrecovered
                    else "unhealthy"
                )
                if state == "unhealthy":
                    previous = self._states.get(key, {})
                    crash_count = int(previous.get("crash_count", 0)) + 1
                    last_error = (
                        runtime_last_error
                        if health_error_unrecovered
                        else "runtime actor not running"
                    )
                    self._states[key] = {
                        **previous,
                        "domain_key": spec.domain_key,
                        "replica_id": spec.replica_id,
                        "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                        "state": "unhealthy",
                        "actor_name": spec.normalized_actor_name(),
                        "launcher_key": spec.launcher_key,
                        "generation": int(self._generations.get(key, 0)),
                        "consumer_id": consumer_id_for_replica(
                            spec.domain_key,
                            spec.replica_id,
                            int(self._generations.get(key, 0)),
                        ),
                        "health": health,
                        "crash_count": crash_count,
                        "last_error": last_error,
                        "last_action": "health_unhealthy",
                        "last_action_at": time.time(),
                        "node_pins": resolved_node_pins,
                        "worker_aliases": original_spec.normalized_worker_aliases(),
                        "gpu_count": spec.gpu_count,
                    }
                    self._actors.pop(key, None)
                    await self._create_runtime(spec, reason="unhealthy")
                    self._states[key]["crash_count"] = crash_count
                    results[label] = self._states[key]
                    continue
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                    "state": state,
                    "actor_name": spec.normalized_actor_name(),
                    "launcher_key": spec.launcher_key,
                    "generation": int(self._generations.get(key, 0)),
                    "consumer_id": consumer_id_for_replica(
                        spec.domain_key,
                        spec.replica_id,
                        int(self._generations.get(key, 0)),
                    ),
                    "health": health,
                    "last_error": None if state == "healthy" else "runtime actor not running",
                    "last_action": "health_check",
                    "last_action_at": time.time(),
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                }
                results[label] = self._states[key]
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
                    "last_action": "health_failed",
                    "last_action_at": time.time(),
                    "node_pins": resolved_node_pins,
                    "worker_aliases": original_spec.normalized_worker_aliases(),
                    "gpu_count": spec.gpu_count,
                }
                self._actors.pop(key, None)
                await self._create_runtime(spec, reason="dead")
                self._states[key]["crash_count"] = crash_count
                results[label] = self._states[key]

        await self._sync_scheduler()
        return {"ok": True, "replicas": results, "snapshot": self.snapshot()}

    async def ensure_reconcile_loop_started(self) -> dict[str, Any]:
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return self.snapshot()
        await self.reconcile_once()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        return self.snapshot()

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
            "scheduler_sync_failures_total": int(self._scheduler_sync_failures_total),
            "placement_reconcile_failures_total": int(self._placement_reconcile_failures_total),
            "topology_reconcile_failures_total": int(self._topology_reconcile_failures_total),
            "node_metrics_created_total": int(self._node_metrics_created_total),
            "node_metrics_reconcile_failures_total": int(self._node_metrics_reconcile_failures_total),
            "control_plane_ensure_failures_total": int(self._control_plane_ensure_failures_total),
            "state_store_failures_total": int(self._state_store_failures_total),
            "placement_reclaimed_total": int(self._placement_reclaimed_total),
            "last_reconcile_at": self._last_reconcile_at,
            "reconcile_inflight": bool(self._reconcile_inflight),
            "reconcile_inflight_started_at": self._reconcile_inflight_started_at,
            "reconcile_loop_running": self._reconcile_task is not None and not self._reconcile_task.done(),
            "reconcile_interval_s": float(self._reconcile_interval_s),
            "last_reconcile_loop_error": self._last_reconcile_loop_error,
            "last_scheduler_sync_at": self._last_scheduler_sync_at,
            "last_placement_reconcile": self._last_placement_reconcile,
            "last_topology_reconcile": self._last_topology_reconcile,
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

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=extra_env),
    }
    resources = _model_actor_supervisor_actor_resources()
    if resources:
        options["resources"] = resources

    actor = _RayModelActorSupervisorActor.options(**options).remote(specs=desired_specs_from_env())
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
        out = sync_get_ray_ref(self._ray_actor.snapshot.remote(), timeout_s=min(float(timeout_s), 10.0))
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
            timeout_s=min(float(timeout_s), 10.0),
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
        source: str | None = None,
    ) -> bool:
        return bool(
            self._call_sync(
                "update_metadata",
                actor_name,
                metadata,
                sample_time=sample_time,
                source=source,
            )
        )

    async def async_update_metadata(
        self,
        actor_name: str,
        metadata: dict[str, Any],
        *,
        sample_time: float | None = None,
        source: str | None = None,
    ) -> bool:
        return bool(
            await self._call_async(
                "async_update_metadata",
                actor_name,
                metadata,
                sample_time=sample_time,
                source=source,
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
