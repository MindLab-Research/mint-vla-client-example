from __future__ import annotations

import asyncio
import structlog
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from mint_server.config import (
    TIER_CPU,
    actor_runtime_env,
    apply_detached_actor_resources,
    config as server_config,
    otel_env_vars,
    preferred_control_plane_resources,
)
from mint_server.ray.runtime_env import env_nonempty
from mint_server.server_info import _git_sha
from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref, sync_get_ray_ref
from mint_server.backend.contracts.control_plane_contracts import (
    AppendWorkResult,
    AssignPendingResult,
    CancelTaskResult,
    ConflictReason,
    ClaimResult,
    ContainsResult,
    CreateTaskResult,
    ExpireResult,
    FailLeaseResult,
    FinishResult,
    LeaseResult,
    LeaseToken,
    OwnerLeaseResult,
    RenewResult,
    SyncReplicasResult,
    TaskMutationResult,
    TaskRecord,
    ValidateLeaseResult,
    as_task_ledger,
)
from mint_server.backend.stores.task_state_store import TERMINAL_TASK_STATUSES, TaskStateConflictError, TaskStateConflictReason, TaskStateNotFoundError
from mint_server.backend.scheduling.scheduler_admission import (
    AdmissionAccounting,
    sampling_inflight_admission_mode,
    sampling_inflight_limit,
)
from mint_server.backend.scheduling.scheduler_counters import SchedulerCounters
from mint_server.backend.scheduling.scheduler_loops import BackgroundLoopSupervisor
from mint_server.backend.scheduling.scheduler_metrics import (
    CLAIMABLE_REPLICA_STATUSES,
    SchedulerMetrics,
    metric_number,
    otel_metric_attrs,
    scheduler_domain_base_model,
)
from mint_server.backend.scheduling.scheduler_queue_projection import QueueProjection

logger = structlog.get_logger(__name__)

CURRENT_CODE_IDENTITY = os.environ.get("MINT_GIT_SHA") or _git_sha()

TRAINING_SAME_AFFINITY_MULTI_CLAIM_DOMAINS = ("bumblebee:", "megatron:")
_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "n", "off", "disabled", "disable"})


class ModelWorkSchedulerUnavailableError(RuntimeError):
    pass


class ModelWorkSchedulerConflictError(RuntimeError):
    pass


class ModelWorkSchedulerCodeIdentityMismatchError(RuntimeError):
    pass


def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _ray_model_work_scheduler_actor_name() -> str:
    env_value = os.environ.get("MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME")
    if env_value:
        return str(env_value)
    return str(getattr(server_config, "model_work_scheduler_actor_name", "mint_model_work_scheduler"))


def _otel_metric_attrs() -> dict[str, str]:
    return otel_metric_attrs(ray_namespace=_ray_namespace())


def _metric_number(value: object) -> float | None:
    return metric_number(value)


def _same_affinity_multi_claim_domains_from_env() -> tuple[str, ...]:
    raw = os.environ.get("MINT_MODEL_WORK_CLAIM_SAME_AFFINITY_DOMAINS")
    if raw is None:
        return TRAINING_SAME_AFFINITY_MULTI_CLAIM_DOMAINS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _model_work_scheduler_use_task_state_store_from_env() -> bool:
    raw = os.environ.get("MINT_MODEL_WORK_SCHEDULER_USE_TASK_STATE_STORE")
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSE_ENV_VALUES


def _sampling_inflight_admission_mode() -> str:
    return sampling_inflight_admission_mode()


def _sampling_inflight_limit(name: str, config_attr: str, default: int) -> int:
    return sampling_inflight_limit(name, config_attr, default)


def _scheduler_domain_base_model(domain_key: object) -> str | None:
    return scheduler_domain_base_model(domain_key)


def _model_work_scheduler_actor_resources() -> dict[str, float] | None:
    try:
        import ray

        return preferred_control_plane_resources(ray.cluster_resources())
    except Exception:
        return None


def _model_work_scheduler_debug_log_path() -> str:
    return os.environ.get(
        "MINT_MODEL_WORK_SCHEDULER_DEBUG_LOG_PATH",
        "/tmp/mint_model_work_scheduler.debug.jsonl",
    )


def _append_model_work_scheduler_debug(event: str, **fields: Any) -> None:
    import json

    record = {
        "ts": round(time.time(), 6),
        "pid": os.getpid(),
        "event": event,
        "actor_name": _ray_model_work_scheduler_actor_name(),
        "namespace": _ray_namespace(),
        **fields,
    }
    try:
        with open(_model_work_scheduler_debug_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True, default=str))
            fh.write("\n")
    except Exception:
        logger.debug("model work scheduler debug log write failed", exc_info=True)


@dataclass(frozen=True)
class ModelWorkItem:
    request_id: str
    op: str
    request_json: bytes
    user_id: str | None
    apikey_id: str | None
    throttle_principal: str | None
    webhook_url: str | None
    extra: dict[str, Any]
    created_at: float
    domain_key: str
    affinity_group: str | None = None
    ordering_key: str | None = None
    token_cost: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelWorkItem":
        return cls(
            request_id=str(data["request_id"]),
            op=str(data["op"]),
            request_json=bytes(data.get("request_json") or b""),
            user_id=None if data.get("user_id") is None else str(data.get("user_id")),
            apikey_id=None if data.get("apikey_id") is None else str(data.get("apikey_id")),
            throttle_principal=(
                None
                if data.get("throttle_principal") is None
                else str(data.get("throttle_principal"))
            ),
            webhook_url=None if data.get("webhook_url") is None else str(data.get("webhook_url")),
            extra=dict(data.get("extra") or {}),
            created_at=float(data.get("created_at") or time.time()),
            domain_key=str(data["domain_key"]),
            affinity_group=(
                None if data.get("affinity_group") is None else str(data.get("affinity_group"))
            ),
            ordering_key=None if data.get("ordering_key") is None else str(data.get("ordering_key")),
            token_cost=max(1, int(data.get("token_cost") or 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["request_json"] = bytes(self.request_json)
        out["extra"] = dict(self.extra)
        return out


@dataclass(frozen=True)
class ModelReplicaRegistration:
    domain_key: str
    replica_id: str
    consumer_id: str
    generation: int
    status: str
    queue_id: str | None = None
    capacity: int = 1
    actor_name: str | None = None
    node_pins: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelReplicaRegistration":
        return cls(
            domain_key=str(data["domain_key"]),
            replica_id=str(data["replica_id"]),
            consumer_id=str(data["consumer_id"]),
            generation=int(data["generation"]),
            status=str(data.get("status") or "starting"),
            queue_id=None if data.get("queue_id") is None else str(data.get("queue_id")),
            capacity=max(1, int(data.get("capacity") or 1)),
            actor_name=None if data.get("actor_name") is None else str(data.get("actor_name")),
            node_pins=[str(v) for v in data.get("node_pins") or []],
            updated_at=float(data.get("updated_at") or time.time()),
        )

    @property
    def effective_queue_id(self) -> str:
        return self.queue_id or f"{self.domain_key}::{self.replica_id}"

    @property
    def claimable(self) -> bool:
        return self.status in CLAIMABLE_REPLICA_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "queue_id": self.effective_queue_id,
            "claimable": self.claimable,
        }


@dataclass
class _AssignedWork:
    item: ModelWorkItem
    replica_id: str
    queue_id: str
    assigned_at: float
    assignment_generation: int
    assignment_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "replica_id": self.replica_id,
            "queue_id": self.queue_id,
            "assigned_at": self.assigned_at,
            "assignment_generation": self.assignment_generation,
            "assignment_reason": self.assignment_reason,
        }


@dataclass
class ModelWorkLease:
    lease_id: str
    item: ModelWorkItem
    domain_key: str
    replica_id: str
    queue_id: str
    attempt_id: str
    scheduler_epoch: int | None
    consumer_id: str
    consumer_generation: int
    leased_at: float
    lease_expires_at: float
    claim_attempt: int
    last_requeue_reason: str | None = None
    finalizing_until: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "item": self.item.to_dict(),
            "domain_key": self.domain_key,
            "replica_id": self.replica_id,
            "queue_id": self.queue_id,
            "attempt_id": self.attempt_id,
            "scheduler_epoch": self.scheduler_epoch,
            "consumer_id": self.consumer_id,
            "consumer_generation": self.consumer_generation,
            "leased_at": self.leased_at,
            "lease_expires_at": self.lease_expires_at,
            "claim_attempt": self.claim_attempt,
            "last_requeue_reason": self.last_requeue_reason,
            "finalizing_until": self.finalizing_until,
        }


@dataclass
class _PendingRequeue:
    assigned: _AssignedWork
    lease: ModelWorkLease | None = None


@dataclass(frozen=True)
class _TerminalTaskState:
    ok: bool
    status: str | None = None
    record: dict[str, Any] | None = None
    reason: ConflictReason | None = None


def _queue_key(domain_key: str, replica_id: str) -> tuple[str, str]:
    return str(domain_key), str(replica_id)


class _ModelWorkSchedulerActor:
    def __init__(
        self,
        *,
        use_task_state_store: bool = False,
        task_state_store: Any | None = None,
        owner_id: str | None = None,
        same_affinity_multi_claim_domains: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        try:
            from mint_server.observability.logging_context import init_actor_observability

            init_actor_observability()
        except Exception:
            logger.debug("actor observability init skipped", exc_info=True)
        self._cv = asyncio.Lock()
        self._instance_id = uuid.uuid4().hex
        self._owner_id = owner_id or f"{_ray_model_work_scheduler_actor_name()}:{self._instance_id}"
        self._use_task_state_store = bool(use_task_state_store)
        self._task_state_store = as_task_ledger(task_state_store) if task_state_store is not None else None
        self._same_affinity_multi_claim_domains = tuple(
            str(part).strip()
            for part in (
                _same_affinity_multi_claim_domains_from_env()
                if same_affinity_multi_claim_domains is None
                else same_affinity_multi_claim_domains
            )
            if str(part).strip()
        )
        self._scheduler_epoch: int | None = None
        self._task_state_hydrated = False
        self._last_owner_ensure_at = 0.0
        self._domain_backlog: dict[str, deque[ModelWorkItem]] = {}
        self._replicas: dict[tuple[str, str], ModelReplicaRegistration] = {}
        self._replica_queues: dict[tuple[str, str], deque[_AssignedWork]] = {}
        self._leases_by_id: dict[str, ModelWorkLease] = {}
        self._lease_id_by_request_id: dict[str, str] = {}
        self._request_locations: dict[str, str] = {}
        self._affinity_replica: dict[tuple[str, str], str] = {}
        self._counters = SchedulerCounters()
        self._admission = AdmissionAccounting()
        self._queue_projection = QueueProjection(self)
        owner_ttl_s = float(getattr(server_config, "task_state_store_owner_ttl_s", 30.0))
        self._task_state_owner_lock = asyncio.Lock()
        self._loops = BackgroundLoopSupervisor(
            assignment_loop=self._assignment_loop,
            owner_heartbeat_loop=self._owner_heartbeat_loop,
            reaper_loop=self._reaper_loop,
            use_task_state_store=self._use_task_state_store,
            assignment_interval_s=float(os.environ.get("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "1.0")),
            owner_heartbeat_interval_s=float(
                os.environ.get(
                    "MINT_MODEL_WORK_SCHEDULER_OWNER_HEARTBEAT_INTERVAL_S",
                    str(max(1.0, min(10.0, owner_ttl_s / 3.0))),
                )
            ),
            reaper_interval_s=float(os.environ.get("MINT_MODEL_WORK_SCHEDULER_REAPER_INTERVAL_S", "10.0")),
        )
        self._metrics = SchedulerMetrics(
            ray_namespace=_ray_namespace,
            stats_snapshot=self._stats_snapshot,
        )
        self._init_otel_metrics()
        self._ensure_background_loops_started()

    @property
    def _assigned_work_type(self) -> type[_AssignedWork]:
        return _AssignedWork

    @staticmethod
    def _queue_key(domain_key: str, replica_id: str) -> tuple[str, str]:
        return _queue_key(domain_key, replica_id)

    @property
    def _completed(self) -> int:
        return self._counters.completed

    @_completed.setter
    def _completed(self, value: int) -> None:
        self._counters.completed = int(value)

    @property
    def _failed(self) -> int:
        return self._counters.failed

    @_failed.setter
    def _failed(self, value: int) -> None:
        self._counters.failed = int(value)

    @property
    def _requeued(self) -> int:
        return self._counters.requeued

    @_requeued.setter
    def _requeued(self, value: int) -> None:
        self._counters.requeued = int(value)

    @property
    def _stale_dropped(self) -> int:
        return self._counters.stale_dropped

    @_stale_dropped.setter
    def _stale_dropped(self, value: int) -> None:
        self._counters.stale_dropped = int(value)

    @property
    def _appended(self) -> int:
        return self._counters.appended

    @_appended.setter
    def _appended(self, value: int) -> None:
        self._counters.appended = int(value)

    @property
    def _assigned(self) -> int:
        return self._counters.assigned

    @_assigned.setter
    def _assigned(self, value: int) -> None:
        self._counters.assigned = int(value)

    @property
    def _claimed(self) -> int:
        return self._counters.claimed

    @_claimed.setter
    def _claimed(self, value: int) -> None:
        self._counters.claimed = int(value)

    @property
    def _reaper_recovered(self) -> int:
        return self._counters.reaper_recovered

    @_reaper_recovered.setter
    def _reaper_recovered(self, value: int) -> None:
        self._counters.reaper_recovered = int(value)

    @property
    def _reaper_scanned(self) -> int:
        return self._counters.reaper_scanned

    @_reaper_scanned.setter
    def _reaper_scanned(self, value: int) -> None:
        self._counters.reaper_scanned = int(value)

    @property
    def _background_loop_manager_task(self) -> asyncio.Task | None:
        return self._loops.manager_task

    @_background_loop_manager_task.setter
    def _background_loop_manager_task(self, value: asyncio.Task | None) -> None:
        self._loops.manager_task = value

    @property
    def _background_loop_tasks(self) -> dict[str, asyncio.Task]:
        return self._loops.tasks

    @property
    def _background_loop_start_deferred(self) -> set[str]:
        return self._loops.start_deferred

    @_background_loop_start_deferred.setter
    def _background_loop_start_deferred(self, value: set[str]) -> None:
        self._loops.start_deferred = value

    @property
    def _background_loops_shutdown(self) -> bool:
        return self._loops.shutdown

    @_background_loops_shutdown.setter
    def _background_loops_shutdown(self, value: bool) -> None:
        self._loops.shutdown = value

    @property
    def _assignment_loop_task(self) -> asyncio.Task | None:
        return self._loops.assignment_task

    @_assignment_loop_task.setter
    def _assignment_loop_task(self, value: asyncio.Task | None) -> None:
        self._loops.assignment_task = value

    @property
    def _assignment_loop_interval_s(self) -> float:
        return self._loops.assignment_interval_s

    @_assignment_loop_interval_s.setter
    def _assignment_loop_interval_s(self, value: float) -> None:
        self._loops.assignment_interval_s = float(value)

    @property
    def _owner_heartbeat_task(self) -> asyncio.Task | None:
        return self._loops.owner_heartbeat_task

    @_owner_heartbeat_task.setter
    def _owner_heartbeat_task(self, value: asyncio.Task | None) -> None:
        self._loops.owner_heartbeat_task = value

    @property
    def _owner_heartbeat_interval_s(self) -> float:
        return self._loops.owner_heartbeat_interval_s

    @_owner_heartbeat_interval_s.setter
    def _owner_heartbeat_interval_s(self, value: float) -> None:
        self._loops.owner_heartbeat_interval_s = float(value)

    @property
    def _reaper_loop_task(self) -> asyncio.Task | None:
        return self._loops.reaper_task

    @_reaper_loop_task.setter
    def _reaper_loop_task(self, value: asyncio.Task | None) -> None:
        self._loops.reaper_task = value

    @property
    def _reaper_loop_interval_s(self) -> float:
        return self._loops.reaper_interval_s

    @_reaper_loop_interval_s.setter
    def _reaper_loop_interval_s(self, value: float) -> None:
        self._loops.reaper_interval_s = float(value)

    @property
    def _otel_enabled(self) -> bool:
        return self._metrics.enabled

    @property
    def _otel_error(self) -> str | None:
        return self._metrics.error

    @property
    def _sampling_inflight_by_domain(self) -> dict[str, int]:
        return self._admission.inflight_by_domain

    @_sampling_inflight_by_domain.setter
    def _sampling_inflight_by_domain(self, value: dict[str, int]) -> None:
        self._admission.inflight_by_domain = value

    @property
    def _sampling_inflight_by_principal_domain(self) -> dict[tuple[str, str], int]:
        return self._admission.inflight_by_principal_domain

    @_sampling_inflight_by_principal_domain.setter
    def _sampling_inflight_by_principal_domain(self, value: dict[tuple[str, str], int]) -> None:
        self._admission.inflight_by_principal_domain = value

    @property
    def _sampling_inflight_tokens_by_domain(self) -> dict[str, int]:
        return self._admission.inflight_tokens_by_domain

    @_sampling_inflight_tokens_by_domain.setter
    def _sampling_inflight_tokens_by_domain(self, value: dict[str, int]) -> None:
        self._admission.inflight_tokens_by_domain = value

    @property
    def _sampling_inflight_tokens_by_principal_domain(self) -> dict[tuple[str, str], int]:
        return self._admission.inflight_tokens_by_principal_domain

    @_sampling_inflight_tokens_by_principal_domain.setter
    def _sampling_inflight_tokens_by_principal_domain(
        self,
        value: dict[tuple[str, str], int],
    ) -> None:
        self._admission.inflight_tokens_by_principal_domain = value

    @property
    def _sampling_principal_domain_by_request_id(self) -> dict[str, tuple[str, str]]:
        return self._admission.principal_domain_by_request_id

    @_sampling_principal_domain_by_request_id.setter
    def _sampling_principal_domain_by_request_id(self, value: dict[str, tuple[str, str]]) -> None:
        self._admission.principal_domain_by_request_id = value

    @property
    def _sampling_token_cost_by_request_id(self) -> dict[str, int]:
        return self._admission.token_cost_by_request_id

    @_sampling_token_cost_by_request_id.setter
    def _sampling_token_cost_by_request_id(self, value: dict[str, int]) -> None:
        self._admission.token_cost_by_request_id = value

    @property
    def _sampling_admission_would_reject(self) -> dict[tuple[str, str], int]:
        return self._admission.would_reject

    @_sampling_admission_would_reject.setter
    def _sampling_admission_would_reject(self, value: dict[tuple[str, str], int]) -> None:
        self._admission.would_reject = value

    @property
    def _sampling_admission_reject(self) -> dict[tuple[str, str], int]:
        return self._admission.reject

    @_sampling_admission_reject.setter
    def _sampling_admission_reject(self, value: dict[tuple[str, str], int]) -> None:
        self._admission.reject = value

    def _init_otel_metrics(self) -> None:
        self._metrics.init_otel_metrics()

    def _all_request_ids(self) -> set[str]:
        return set(self._request_locations)

    async def _task_state_call(self, method: str, **kwargs: Any) -> Any:
        if self._task_state_store is None:
            from mint_server.backend.stores.task_state_store import task_state_store

            self._task_state_store = as_task_ledger(task_state_store)
        async_method = getattr(self._task_state_store, method)
        return await async_method(**kwargs)

    def _task_record_data(self, record: TaskRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, TaskRecord):
            return dict(record.data)
        if isinstance(record, dict):
            return dict(record)
        raise TypeError(f"task ledger returned non-TaskRecord: {type(record)}")

    async def _ensure_task_state_owner(self) -> int | None:
        if not self._use_task_state_store:
            return None
        now = time.monotonic()
        ttl_s = float(getattr(server_config, "task_state_store_owner_ttl_s", 30.0))
        while True:
            async with self._task_state_owner_lock:
                current_epoch = self._scheduler_epoch

            if current_epoch is not None:
                renewed = await self._task_state_call(
                    "renew_owner",
                    owner_id=self._owner_id,
                    epoch=int(current_epoch),
                    ttl_s=ttl_s,
                )
                if isinstance(renewed, OwnerLeaseResult) and renewed.ok:
                    async with self._task_state_owner_lock:
                        if self._scheduler_epoch in {None, int(current_epoch)}:
                            self._scheduler_epoch = int(current_epoch)
                            self._last_owner_ensure_at = now
                            return int(current_epoch)
                    continue

                async with self._task_state_owner_lock:
                    if self._scheduler_epoch == int(current_epoch):
                        self._scheduler_epoch = None
                    if self._scheduler_epoch is not None:
                        continue

            acquired = await self._task_state_call(
                "acquire_owner",
                owner_id=self._owner_id,
                ttl_s=ttl_s,
            )
            if not isinstance(acquired, OwnerLeaseResult) or not acquired.ok:
                raise ModelWorkSchedulerConflictError(f"failed to acquire scheduler owner: {acquired}")
            acquired_epoch = int(acquired.epoch or 0)
            async with self._task_state_owner_lock:
                self._scheduler_epoch = acquired_epoch
                self._last_owner_ensure_at = time.monotonic()
            return acquired_epoch

    async def _owner_heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._owner_heartbeat_interval_s)
            try:
                await self._ensure_task_state_owner()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[model_work_scheduler] owner heartbeat failed error_type=%s error=%s",
                    type(e).__name__,
                    e,
                )

    def _desired_background_loop_names(self) -> list[str]:
        return self._loops.desired_names()

    def _background_loop_running(self, name: str) -> bool:
        return self._loops.running(name)

    def _ensure_background_loops_started(self) -> None:
        self._loops.ensure_started()

    async def shutdown_background_loops(self) -> dict[str, Any]:
        return await self._loops.shutdown_loops()

    def _ensure_owner_heartbeat_started(self) -> None:
        self._ensure_background_loops_started()

    async def _assignment_loop(self) -> None:
        while True:
            await asyncio.sleep(self._assignment_loop_interval_s)
            try:
                await self._assign_pending_typed()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[model_work_scheduler] assignment loop failed error_type=%s error=%s",
                    type(e).__name__,
                    e,
                )

    def _ensure_assignment_loop_started(self) -> None:
        self._ensure_background_loops_started()

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reaper_loop_interval_s)
            try:
                await self.reap_lost_pending_tasks(reason="scheduler_reaper_requeue")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[model_work_scheduler] reaper loop failed error_type=%s error=%s",
                    type(e).__name__,
                    e,
                )

    def _ensure_reaper_loop_started(self) -> None:
        self._ensure_background_loops_started()

    @staticmethod
    def _is_scheduler_owned_record(record: dict[str, Any]) -> bool:
        """True only for tasks this scheduler created via ``create_task``.

        The shared TaskStateStore also holds plain futures created directly by
        route handlers (e.g. ``async_ensure_pending`` on the OpenPI pi0.5
        ``DIRECT_RUNTIME`` action path). Those live under ``future:default`` and
        are resolved by their own request coroutine, never by the scheduler.
        Only records that carry the scheduler's own append marker are model
        work the scheduler is responsible for hydrating/reaping. Adopting a
        plain future would let the scheduler assign/claim/terminalize it out
        from under the in-flight request, which fails the request's later
        ``async_resolve`` with a terminal task-state conflict.
        """
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        assert metadata is not None
        return bool(metadata.get("model_work_scheduler_append_attempt_id"))

    def _work_item_from_task_record(self, record: dict[str, Any]) -> ModelWorkItem:
        metadata = dict(record.get("metadata") or {})
        extra = dict(metadata)
        return ModelWorkItem(
            request_id=str(record["request_id"]),
            op=str(record["op"]),
            request_json=bytes(record.get("request_json") or b""),
            user_id=None if metadata.get("user_id") is None else str(metadata.get("user_id")),
            apikey_id=None if metadata.get("apikey_id") is None else str(metadata.get("apikey_id")),
            throttle_principal=(
                None
                if metadata.get("throttle_principal") is None
                else str(metadata.get("throttle_principal"))
            ),
            webhook_url=None if metadata.get("webhook_url") is None else str(metadata.get("webhook_url")),
            extra=extra,
            created_at=float(record.get("created_at") or time.time()),
            domain_key=str(record["domain_key"]),
            affinity_group=(
                None if metadata.get("affinity_group") is None else str(metadata.get("affinity_group"))
            ),
            ordering_key=None if metadata.get("ordering_key") is None else str(metadata.get("ordering_key")),
            token_cost=max(1, int(metadata.get("token_cost") or 1)),
        )

    def _is_sampling_inflight_work(self, item: ModelWorkItem) -> bool:
        return self._admission.is_sampling_inflight_work(item)

    def _sampling_principal(self, item: ModelWorkItem) -> str:
        return self._admission.principal(item)

    def _track_sampling_inflight_locked(self, item: ModelWorkItem) -> None:
        self._admission.track_locked(item)

    def _untrack_sampling_inflight_locked(self, request_id: str) -> None:
        self._admission.untrack_locked(request_id)

    def _sampling_inflight_limit_decision_locked(self, item: ModelWorkItem) -> dict[str, Any]:
        return self._admission.limit_decision_locked(item)

    async def _persist_requeue_task(self, request_id: str, *, reason: str) -> bool:
        if not self._use_task_state_store:
            return True
        try:
            out = await self._task_state_call(
                "requeue_task",
                request_id=str(request_id),
                scheduler_epoch=int(self._scheduler_epoch or 0),
                reason=str(reason),
            )
        except Exception as exc:
            if self._task_not_found_cause(exc) is not None:
                return False
            raise
        if isinstance(out, TaskMutationResult) and out.ok:
            return True
        if isinstance(out, TaskMutationResult) and out.reason == "terminal":
            return False
        raise ModelWorkSchedulerConflictError(f"failed to requeue task {request_id}: {out!r}")

    async def _ensure_task_state_ready(self) -> int | None:
        self._ensure_assignment_loop_started()
        self._ensure_owner_heartbeat_started()
        self._ensure_reaper_loop_started()
        epoch = await self._ensure_task_state_owner()
        if not self._use_task_state_store or self._task_state_hydrated:
            return epoch
        active = await self._task_state_call("list_active_tasks")
        if not isinstance(active, list):
            raise TypeError(f"TaskStateStore.list_active_tasks returned non-list: {type(active)}")
        to_requeue: list[tuple[ModelWorkItem, str]] = []
        pending_items: list[ModelWorkItem] = []
        for task_record in active:
            if not isinstance(task_record, TaskRecord):
                raise TypeError(f"TaskStateStore.list_active_tasks item returned non-TaskRecord: {type(task_record)}")
            record = self._task_record_data(task_record)
            status = str(record.get("status") or "")
            if not self._is_scheduler_owned_record(record):
                continue
            item = self._work_item_from_task_record(record)
            if status != "pending":
                to_requeue.append((item, "scheduler_hydrate_requeue"))
            else:
                pending_items.append(item)
        for item, reason in to_requeue:
            should_requeue = await self._persist_requeue_task(item.request_id, reason=reason)
            if should_requeue:
                pending_items.append(item)
        async with self._cv:
            if self._task_state_hydrated:
                return epoch
            for item in pending_items:
                if item.request_id in self._all_request_ids():
                    continue
                self._backlog(item.domain_key).append(item)
                self._request_locations[item.request_id] = "backlog"
                self._track_sampling_inflight_locked(item)
            self._task_state_hydrated = True
        return epoch

    async def reap_lost_pending_tasks(
        self,
        *,
        limit: int | None = None,
        reason: str = "scheduler_reaper_requeue",
    ) -> dict[str, Any]:
        epoch = await self._ensure_task_state_ready()
        if not self._use_task_state_store:
            return {
                "ok": True,
                "scanned": 0,
                "recovered": 0,
                "scheduler_epoch": epoch,
            }
        active = await self._task_state_call("list_active_tasks", limit=limit)
        if not isinstance(active, list):
            raise TypeError(f"TaskStateStore.list_active_tasks returned non-list: {type(active)}")
        pending_items: list[ModelWorkItem] = []
        scanned = 0
        for task_record in active:
            if not isinstance(task_record, TaskRecord):
                raise TypeError(f"TaskStateStore.list_active_tasks item returned non-TaskRecord: {type(task_record)}")
            record = self._task_record_data(task_record)
            scanned += 1
            if str(record.get("status") or "") != "pending":
                continue
            if not self._is_scheduler_owned_record(record):
                continue
            item = self._work_item_from_task_record(record)
            async with self._cv:
                if item.request_id in self._all_request_ids():
                    continue
            pending_items.append(item)
        recovered = 0
        async with self._cv:
            for item in pending_items:
                if item.request_id in self._all_request_ids():
                    continue
                self._backlog(item.domain_key).append(item)
                self._request_locations[item.request_id] = "backlog"
                self._track_sampling_inflight_locked(item)
                recovered += 1
            self._reaper_scanned += scanned
            self._reaper_recovered += recovered
            if recovered:
                self._requeued += recovered
        if recovered:
            logger.warning(
                "[model_work_scheduler] recovered lost pending tasks scanned=%s recovered=%s reason=%s",
                scanned,
                recovered,
                reason,
            )
        return {
            "ok": True,
            "scanned": scanned,
            "recovered": recovered,
            "scheduler_epoch": epoch,
            "reason": str(reason),
        }

    def _task_record_matches_work_item(self, record: dict[str, Any], item: ModelWorkItem) -> bool:
        if str(record.get("request_id") or "") != item.request_id:
            return False
        if str(record.get("op") or "") != item.op:
            return False
        if str(record.get("domain_key") or "") != item.domain_key:
            return False
        record_json = record.get("request_json")
        if record_json is not None and bytes(record_json) != item.request_json:
            return False
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return True
        record_hash = metadata.get("payload_hash") or record.get("payload_hash")
        item_hash = item.extra.get("payload_hash")
        if record_hash is not None and item_hash is not None and str(record_hash) != str(item_hash):
            return False
        return True

    def _hot_projection_matches_work_item_locked(self, item: ModelWorkItem) -> bool:
        return self._queue_projection.hot_projection_matches_work_item_locked(item)

    async def _duplicate_append_matches_hydrated_pending_task(self, item: ModelWorkItem) -> bool:
        if not self._use_task_state_store:
            return False
        if item.request_id not in self._all_request_ids():
            return False
        try:
            task_record = await self._task_state_call("get_task", request_id=item.request_id)
        except Exception:
            return False
        if not isinstance(task_record, (TaskRecord, dict)):
            return False
        record = self._task_record_data(task_record)
        if not self._task_record_matches_work_item(record, item):
            return False
        return str(record.get("status") or "") in {"pending", "queued", "assigned"}

    async def _drop_recreated_orphan_projection_before_append(self, item: ModelWorkItem) -> bool:
        if not self._use_task_state_store:
            return False
        durable_missing = False
        try:
            task_record = await self._task_state_call("get_task", request_id=item.request_id)
        except Exception as exc:
            if self._task_not_found_cause(exc) is None:
                return False
            durable_missing = True
        if not durable_missing:
            if not isinstance(task_record, (TaskRecord, dict)):
                return False
            record = self._task_record_data(task_record)
            if not self._task_record_matches_work_item(record, item):
                return False
            if str(record.get("status") or "") not in {"pending", "queued"}:
                return False
        async with self._cv:
            if self._request_locations.get(item.request_id) not in {"leased", "finalizing"}:
                return False
            if not self._hot_projection_matches_work_item_locked(item):
                return False
            return self._remove_request_from_memory_locked(item.request_id)

    async def _forget_created_task_after_append_cancel(
        self,
        item: ModelWorkItem,
        *,
        append_attempt_id: str,
        _durable: dict[str, object] | None = None,
    ) -> None:
        if not self._use_task_state_store:
            return
        if not append_attempt_id:
            return
        task_record: object | None = _durable
        if task_record is None:
            try:
                task_record = await self._task_state_call("get_task", request_id=item.request_id)
            except Exception as exc:
                if self._task_not_found_cause(exc) is not None:
                    return
                logger.exception(
                    "[model_work_scheduler] failed to inspect task before append cancellation rollback request_id=%s",
                    item.request_id,
                )
                return
        if not isinstance(task_record, (TaskRecord, dict)):
            return
        record = self._task_record_data(task_record)
        if not self._task_record_matches_work_item(record, item):
            return
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or str(
            metadata.get("model_work_scheduler_append_attempt_id") or ""
        ) != str(append_attempt_id):
            return
        if str(record.get("status") or "") not in {"pending", "queued"}:
            return
        try:
            await self._task_state_call("forget_task", request_id=item.request_id)
        except Exception:
            logger.exception(
                "[model_work_scheduler] failed to roll back task after append cancellation request_id=%s",
                item.request_id,
            )

    def _backlog(self, domain_key: str) -> deque[ModelWorkItem]:
        return self._queue_projection.backlog(domain_key)

    def _queue(self, domain_key: str, replica_id: str) -> deque[_AssignedWork]:
        return self._queue_projection.queue(domain_key, replica_id)

    def _claim_requires_same_affinity(self, domain_key: str) -> bool:
        prefixes = self._same_affinity_multi_claim_domains
        if not prefixes:
            return False
        return str(domain_key).startswith(prefixes)

    def _ordering_key_has_active_lease_locked(self, ordering_key: str | None) -> bool:
        return self._queue_projection.ordering_key_has_active_lease_locked(ordering_key)

    def _cluster_queue_head_affinity(self, queue: deque[_AssignedWork]) -> None:
        self._queue_projection.cluster_queue_head_affinity(queue)

    def _drop_empty_backlog(self, domain_key: str) -> None:
        self._queue_projection.drop_empty_backlog(domain_key)

    def _has_inflight_scheduler_transition_locked(self) -> bool:
        return self._queue_projection.has_inflight_scheduler_transition_locked()

    def _assigned_matches_locked(self, assigned: _AssignedWork, *, location: str = "assigning") -> bool:
        return self._queue_projection.assigned_matches_locked(assigned, location=location)

    def _commit_assigned_locked(self, assigned: _AssignedWork) -> None:
        self._queue_projection.commit_assigned_locked(assigned)

    def _restore_assigned_to_queue_locked(self, assigned: _AssignedWork) -> None:
        self._queue_projection.restore_assigned_to_queue_locked(assigned)

    def _restore_assigning_to_backlog_locked(self, assigned: _AssignedWork) -> bool:
        return self._queue_projection.restore_assigning_to_backlog_locked(assigned)

    async def _restore_or_commit_assigning_after_cancel(
        self,
        pending: list[_AssignedWork],
        *,
        _durable_results: dict[str, dict[str, object]] | None = None,
    ) -> None:
        async with self._cv:
            if not self._use_task_state_store:
                for assigned in reversed(pending):
                    self._restore_assigning_to_backlog_locked(assigned)
                return
        durable_map = _durable_results or {}
        for assigned in pending:
            durable_assigned = False
            rid = assigned.item.request_id
            if rid in durable_map:
                record = durable_map[rid]
                durable_assigned = (
                    str(record.get("status") or "") == "assigned"
                    and str(record.get("subqueue_id") or "") == str(assigned.queue_id)
                    and int(record.get("scheduler_epoch") or 0) == int(self._scheduler_epoch or 0)
                )
            else:
                try:
                    task_record = await self._task_state_call("get_task", request_id=rid)
                    record = self._task_record_data(task_record)
                    durable_assigned = (
                        str(record.get("status") or "") == "assigned"
                        and str(record.get("subqueue_id") or "") == str(assigned.queue_id)
                        and int(record.get("scheduler_epoch") or 0) == int(self._scheduler_epoch or 0)
                    )
                except Exception:
                    durable_assigned = False
            async with self._cv:
                if self._request_locations.get(rid) != "assigning":
                    continue
                if durable_assigned:
                    self._commit_assigned_locked(assigned)
                else:
                    self._restore_assigning_to_backlog_locked(assigned)

    def _prepare_assignments_locked(self, *, max_items: int | None = None) -> tuple[list[_AssignedWork], list[str]]:
        return self._queue_projection.prepare_assignments_locked(max_items=max_items)

    async def _assign_pending_core_unlocked(self, *, max_items: int | None = None) -> AssignPendingResult:
        async with self._cv:
            pending, skipped_domains = self._prepare_assignments_locked(max_items=max_items)
            if not pending:
                return AssignPendingResult(ok=True, assigned=0, skipped_domains=skipped_domains)
        assigned_count = 0
        assign_results: dict[str, dict[str, object]] = {}
        for index, assigned in enumerate(pending):
            try:
                if self._use_task_state_store:
                    result = await self._task_state_call(
                        "assign_task",
                        request_id=assigned.item.request_id,
                        subqueue_id=assigned.queue_id,
                        scheduler_epoch=int(self._scheduler_epoch or 0),
                    )
                    if isinstance(result, TaskMutationResult) and result.record:
                        assign_results[assigned.item.request_id] = result.record
            except asyncio.CancelledError:
                await self._restore_or_commit_assigning_after_cancel(
                    pending[index:],
                    _durable_results=assign_results,
                )
                raise
            except Exception as exc:
                conflict = self._claim_conflict_cause(exc)
                if conflict is not None:
                    reconciled = await self._reconcile_pending_assign_conflict(
                        assigned.item,
                        conflict=conflict,
                    )
                    if reconciled is not None:
                        continue
                async with self._cv:
                    for unprocessed in reversed(pending[index:]):
                        self._restore_assigning_to_backlog_locked(unprocessed)
                raise
            async with self._cv:
                if self._assigned_matches_locked(assigned, location="assigning"):
                    self._commit_assigned_locked(assigned)
                    assigned_count += 1
        return AssignPendingResult(ok=True, assigned=assigned_count, skipped_domains=skipped_domains)

    async def _assign_pending_unlocked(self, *, max_items: int | None = None) -> AssignPendingResult:
        return await self._assign_pending_core_unlocked(max_items=max_items)

    async def _assign_pending_if_no_inflight_unlocked(
        self,
        *,
        hydrate_task_state: bool = True,
    ) -> AssignPendingResult:
        async with self._cv:
            if self._has_inflight_scheduler_transition_locked():
                return AssignPendingResult(
                    ok=True,
                    assigned=0,
                    skipped_domains=[],
                    extra={"deferred": "inflight_scheduler_transition"},
                )
        return await self._assign_pending_typed(hydrate_task_state=hydrate_task_state)

    def _remove_request_from_memory_locked(self, request_id: str) -> bool:
        return self._queue_projection.remove_request_from_memory_locked(request_id)

    def _claimable_replicas(self, domain_key: str) -> list[ModelReplicaRegistration]:
        return self._queue_projection.claimable_replicas(domain_key)

    def _choose_replica(self, item: ModelWorkItem) -> ModelReplicaRegistration | None:
        return self._queue_projection.choose_replica(item)

    def _requeue_assigned(self, assigned: _AssignedWork, *, reason: str) -> None:
        self._queue_projection.requeue_assigned(assigned, reason=reason)

    def _remove_request_location(self, request_id: str) -> None:
        self._queue_projection.remove_request_location(request_id)

    def _claim_conflict_cause(self, exc: BaseException) -> TaskStateConflictError | None:
        if isinstance(exc, TaskStateConflictError):
            return exc
        as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
        if callable(as_instanceof_cause):
            try:
                cause = as_instanceof_cause()
            except Exception:
                cause = None
            if isinstance(cause, TaskStateConflictError):
                return cause
        cause = getattr(exc, "cause", None)
        if isinstance(cause, TaskStateConflictError):
            return cause
        return None

    def _task_not_found_cause(self, exc: BaseException) -> TaskStateNotFoundError | None:
        if isinstance(exc, TaskStateNotFoundError):
            return exc
        as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
        if callable(as_instanceof_cause):
            try:
                cause = as_instanceof_cause()
            except Exception:
                cause = None
            if isinstance(cause, TaskStateNotFoundError):
                return cause
        cause = getattr(exc, "cause", None)
        if isinstance(cause, TaskStateNotFoundError):
            return cause
        return None

    def _drop_claiming_request_locked(self, assigned: _AssignedWork) -> bool:
        return self._queue_projection.drop_claiming_request_locked(assigned)

    async def _restore_or_commit_claiming_after_cancel(
        self,
        assigned: _AssignedWork,
        *,
        lease_id: str,
        attempt_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float,
        _durable: dict[str, object] | None = None,
    ) -> None:
        durable_record: dict[str, Any] | None = _durable  # type: ignore[assignment]
        if _durable is None and self._use_task_state_store:
            try:
                task_record = await self._task_state_call(
                    "get_task",
                    request_id=assigned.item.request_id,
                )
                record = self._task_record_data(task_record)
                if (
                    str(record.get("status") or "") in {"leased", "running"}
                    and str(record.get("lease_id") or "") == str(lease_id)
                    and str(record.get("attempt_id") or "") == str(attempt_id)
                    and int(record.get("scheduler_epoch") or 0) == int(self._scheduler_epoch or 0)
                    and int(record.get("runtime_generation") or 0) == int(consumer_generation)
                ):
                    durable_record = record
            except Exception:
                durable_record = None
        async with self._cv:
            if self._request_locations.get(assigned.item.request_id) != "claiming":
                return
            if durable_record is None:
                self._restore_assigned_to_queue_locked(assigned)
                return
            now = time.time()
            lease_expires_at = durable_record.get("lease_expires_at")
            lease = ModelWorkLease(
                lease_id=str(lease_id),
                item=assigned.item,
                domain_key=assigned.item.domain_key,
                replica_id=assigned.replica_id,
                queue_id=assigned.queue_id,
                attempt_id=str(attempt_id),
                scheduler_epoch=self._scheduler_epoch,
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
                leased_at=float(durable_record.get("leased_at") or now),
                lease_expires_at=(
                    float(lease_expires_at)
                    if lease_expires_at is not None
                    else now + max(1.0, float(lease_ttl_s))
                ),
                claim_attempt=int(assigned.item.extra.get("claim_attempt") or 0) + 1,
                last_requeue_reason=assigned.item.extra.get("last_requeue_reason"),
            )
            self._leases_by_id[str(lease_id)] = lease
            self._lease_id_by_request_id[assigned.item.request_id] = str(lease_id)
            self._request_locations[assigned.item.request_id] = "leased"

    async def _reconcile_assigned_claim_conflict(
        self,
        assigned: _AssignedWork,
        *,
        conflict: TaskStateConflictError,
    ) -> str | None:
        # Structural matching with backward compat for old-style errors.
        reason = getattr(conflict, "reason", None)
        if reason == TaskStateConflictReason.STALE_SCHEDULER_OWNER:
            # Transient: restore to queue without raising; heartbeat will fix epoch.
            async with self._cv:
                if self._request_locations.get(assigned.item.request_id) == "claiming":
                    self._restore_assigned_to_queue_locked(assigned)
            logger.warning(
                "skipped_claim_transient_stale_owner",
                request_id=assigned.item.request_id,
                queue_id=assigned.queue_id,
            )
            return "transient_stale_owner"
        if reason == TaskStateConflictReason.CANNOT_CLAIM_ASSIGNED:
            pass  # proceed to reconciliation below
        elif reason is None and "cannot claim assigned task" in str(conflict):
            pass  # backward compat: old-style error without reason code
        else:
            return None
        try:
            task_record = await self._task_state_call("get_task", request_id=assigned.item.request_id)
            record = self._task_record_data(task_record)
        except TaskStateNotFoundError:
            async with self._cv:
                if self._drop_claiming_request_locked(assigned):
                    logger.warning(
                        "[model_work_scheduler] dropped stale assigned queue item with missing task state request_id=%s queue_id=%s",
                        assigned.item.request_id,
                        assigned.queue_id,
                    )
            return "missing"
        status = str(record.get("status") or "")
        scheduler_epoch = int(self._scheduler_epoch or 0)
        same_assignment = (
            status == "assigned"
            and str(record.get("subqueue_id") or "") == str(assigned.queue_id)
            and int(record.get("scheduler_epoch") or 0) == scheduler_epoch
        )
        if same_assignment:
            return None
        async with self._cv:
            if self._request_locations.get(assigned.item.request_id) != "claiming":
                return "changed"
            if status == "pending":
                self._backlog(assigned.item.domain_key).appendleft(assigned.item)
                self._request_locations[assigned.item.request_id] = "backlog"
                self._requeued += 1
                logger.warning(
                    "[model_work_scheduler] requeued stale assigned queue item whose task state is pending request_id=%s queue_id=%s",
                    assigned.item.request_id,
                    assigned.queue_id,
                )
                return "pending_requeued"
            self._remove_request_location(assigned.item.request_id)
            self._stale_dropped += 1
        logger.warning(
            "[model_work_scheduler] dropped stale assigned queue item after task-state claim conflict request_id=%s queue_id=%s status=%s terminal=%s",
            assigned.item.request_id,
            assigned.queue_id,
            status or "unknown",
            status in TERMINAL_TASK_STATUSES,
        )
        return status or "unknown"

    async def _reconcile_pending_assign_conflict(
        self,
        item: ModelWorkItem,
        *,
        conflict: TaskStateConflictError,
    ) -> str | None:
        # Structural matching with backward compat for old-style errors.
        reason = getattr(conflict, "reason", None)
        if reason is None and "cannot assign from pending" in str(conflict):
            pass  # backward compat
        elif reason != TaskStateConflictReason.CANNOT_ASSIGN_FROM_PENDING:
            return None
        try:
            task_record = await self._task_state_call("get_task", request_id=item.request_id)
            record = self._task_record_data(task_record)
        except TaskStateNotFoundError:
            async with self._cv:
                if self._request_locations.get(item.request_id) == "assigning":
                    self._remove_request_from_memory_locked(item.request_id)
                    self._stale_dropped += 1
            logger.warning(
                "[model_work_scheduler] dropped stale backlog item with missing task state request_id=%s domain_key=%s",
                item.request_id,
                item.domain_key,
            )
            return "missing"
        status = str(record.get("status") or "")
        if status == "pending":
            return None
        async with self._cv:
            if self._request_locations.get(item.request_id) == "assigning":
                self._remove_request_from_memory_locked(item.request_id)
                self._stale_dropped += 1
        logger.warning(
            "[model_work_scheduler] dropped stale backlog item after task-state assign conflict request_id=%s domain_key=%s status=%s terminal=%s",
            item.request_id,
            item.domain_key,
            status or "unknown",
            status in TERMINAL_TASK_STATUSES,
        )
        return status or "unknown"

    def _prepare_expired_leases_locked(self, *, now: float) -> list[_PendingRequeue]:
        expired: list[_PendingRequeue] = []
        lease_items = sorted(
            list(self._leases_by_id.items()),
            key=lambda item: (float(item[1].leased_at), str(item[0])),
            reverse=True,
        )
        for lease_id, lease in lease_items:
            if self._request_locations.get(lease.item.request_id) == "finalizing":
                continue
            expires_at = (
                max(float(lease.lease_expires_at), float(lease.finalizing_until))
                if lease.finalizing_until is not None
                else float(lease.lease_expires_at)
            )
            if expires_at > now:
                continue
            self._leases_by_id.pop(lease_id, None)
            self._lease_id_by_request_id.pop(lease.item.request_id, None)
            assigned = _AssignedWork(
                item=lease.item,
                replica_id=lease.replica_id,
                queue_id=lease.queue_id,
                assigned_at=now,
                assignment_generation=lease.consumer_generation,
                assignment_reason="lease_expired_requeue",
            )
            self._request_locations[lease.item.request_id] = "requeueing"
            expired.append(_PendingRequeue(assigned=assigned, lease=lease))
        return expired

    def _restore_pending_requeue_locked(self, pending: _PendingRequeue) -> None:
        assigned = pending.assigned
        if self._request_locations.get(assigned.item.request_id) != "requeueing":
            return
        if pending.lease is not None:
            lease = pending.lease
            self._leases_by_id[lease.lease_id] = lease
            self._lease_id_by_request_id[lease.item.request_id] = lease.lease_id
            self._request_locations[lease.item.request_id] = "leased"
            return
        self._restore_assigned_to_queue_locked(assigned)

    async def _restore_or_commit_requeues_after_cancel(self, requeue_items: list[_PendingRequeue]) -> set[str]:
        committed_request_ids: set[str] = set()
        for pending in requeue_items:
            assigned = pending.assigned
            durable_pending = False
            durable_terminal = False
            if self._use_task_state_store:
                try:
                    task_record = await self._task_state_call("get_task", request_id=assigned.item.request_id)
                    record = self._task_record_data(task_record)
                    status = str(record.get("status") or "")
                    durable_pending = status == "pending"
                    durable_terminal = status in TERMINAL_TASK_STATUSES
                except Exception:
                    durable_pending = False
                    durable_terminal = False
            async with self._cv:
                if self._request_locations.get(assigned.item.request_id) != "requeueing":
                    continue
                if durable_pending:
                    self._requeue_assigned(assigned, reason="cancelled_after_requeue_commit")
                    committed_request_ids.add(assigned.item.request_id)
                elif durable_terminal:
                    self._remove_request_location(assigned.item.request_id)
                    committed_request_ids.add(assigned.item.request_id)
                else:
                    self._restore_pending_requeue_locked(pending)
        return committed_request_ids

    async def _commit_requeues_unlocked(self, requeue_items: list[_PendingRequeue], *, reason: str) -> int:
        requeued = 0
        committed_request_ids: set[str] = set()
        for index, pending in enumerate(requeue_items):
            assigned = pending.assigned
            try:
                should_requeue = await self._persist_requeue_task(
                    assigned.item.request_id,
                    reason=reason,
                )
            except asyncio.CancelledError as exc:
                committed_request_ids.update(
                    await self._restore_or_commit_requeues_after_cancel(requeue_items[index:])
                )
                setattr(exc, "_model_work_requeue_committed_request_ids", set(committed_request_ids))
                raise
            except Exception as exc:
                async with self._cv:
                    for unprocessed in requeue_items[index:]:
                        self._restore_pending_requeue_locked(unprocessed)
                setattr(exc, "_model_work_requeue_committed_request_ids", set(committed_request_ids))
                raise
            async with self._cv:
                if self._request_locations.get(assigned.item.request_id) != "requeueing":
                    continue
                if should_requeue:
                    self._requeue_assigned(assigned, reason=reason)
                    requeued += 1
                else:
                    self._remove_request_location(assigned.item.request_id)
                committed_request_ids.add(assigned.item.request_id)
        return requeued

    async def _expire_leases_unlocked(self, *, now: float) -> int:
        async with self._cv:
            expired = self._prepare_expired_leases_locked(now=now)
            if not expired:
                return 0
        return await self._commit_requeues_unlocked(expired, reason="lease_expired")

    async def append(
        self,
        item: dict[str, Any],
        *,
        assign: bool = False,
        assign_max_items: int | None = None,
    ) -> AppendWorkResult:
        self._ensure_assignment_loop_started()
        work = ModelWorkItem.from_dict(item)
        async with self._cv:
            duplicate_in_memory = work.request_id in self._all_request_ids()
        if duplicate_in_memory and await self._drop_recreated_orphan_projection_before_append(work):
            duplicate_in_memory = False
        if duplicate_in_memory:
            if not await self._duplicate_append_matches_hydrated_pending_task(work):
                return AppendWorkResult(
                    ok=False,
                    reason=ConflictReason.DUPLICATE_REQUEST_ID,
                    request_id=work.request_id,
                )
            assigned = (
                (await self._assign_pending_typed(max_items=assign_max_items)).to_wire()
                if bool(assign)
                else {"ok": True, "assigned": 0, "skipped_domains": []}
            )
            async with self._cv:
                backlog_depth = len(self._backlog(work.domain_key))
            return AppendWorkResult(
                ok=True,
                request_id=work.request_id,
                domain_key=work.domain_key,
                scheduler_instance_id=self._instance_id,
                backlog_depth=backlog_depth,
                assigned=assigned,
                idempotent=True,
            )
        created: CreateTaskResult | None = None
        async with self._cv:
            admission = self._sampling_inflight_limit_decision_locked(work)
            if not bool(admission.get("ok")):
                return AppendWorkResult.from_wire(
                    {
                        **admission,
                        "request_id": work.request_id,
                        "scheduler_instance_id": self._instance_id,
                    }
                )
        append_attempt_id = uuid.uuid4().hex
        if self._use_task_state_store:
            try:
                created = await self._task_state_call(
                    "create_task",
                    request_id=work.request_id,
                    op=work.op,
                    domain_key=work.domain_key,
                    request_json=work.request_json,
                    payload_hash=work.extra.get("payload_hash"),
                    metadata={
                        **dict(work.extra),
                        "user_id": work.user_id,
                        "apikey_id": work.apikey_id,
                        "throttle_principal": work.throttle_principal,
                        "webhook_url": work.webhook_url,
                        "affinity_group": work.affinity_group,
                        "ordering_key": work.ordering_key,
                        "token_cost": work.token_cost,
                        "model_work_scheduler_instance_id": self._instance_id,
                        "model_work_scheduler_append_attempt_id": append_attempt_id,
                    },
                )
            except asyncio.CancelledError:
                await self._forget_created_task_after_append_cancel(
                    work,
                    append_attempt_id=append_attempt_id,
                    _durable=created.record if isinstance(created, CreateTaskResult) else None,
                )
                raise
            if isinstance(created, CreateTaskResult) and not created.created:
                record = created.record
                if not self._task_record_matches_work_item(record, work):
                    return AppendWorkResult(
                        ok=False,
                        reason=ConflictReason.DUPLICATE_REQUEST_ID,
                        request_id=work.request_id,
                    )
                if str(record.get("status") or "") not in {"pending", "queued"}:
                    return AppendWorkResult(
                        ok=False,
                        reason=ConflictReason.DUPLICATE_REQUEST_ID,
                        request_id=work.request_id,
                    )
        try:
            async with self._cv:
                if work.request_id in self._all_request_ids():
                    return AppendWorkResult(
                        ok=False,
                        reason=ConflictReason.DUPLICATE_REQUEST_ID,
                        request_id=work.request_id,
                    )
                self._backlog(work.domain_key).append(work)
                self._request_locations[work.request_id] = "backlog"
                self._appended += 1
                self._track_sampling_inflight_locked(work)
                backlog_depth = len(self._backlog(work.domain_key))
            assigned = (
                (await self._assign_pending_typed(max_items=assign_max_items)).to_wire()
                if bool(assign)
                else {"ok": True, "assigned": 0, "skipped_domains": []}
            )
        except asyncio.CancelledError:
            async with self._cv:
                location = self._request_locations.get(work.request_id)
                durable_transition_committed = location in {"assigned", "leased", "finalizing"}
                if not durable_transition_committed:
                    self._remove_request_from_memory_locked(work.request_id)
            if self._use_task_state_store and isinstance(created, CreateTaskResult) and created.created:
                await self._forget_created_task_after_append_cancel(
                    work,
                    append_attempt_id=append_attempt_id,
                    _durable=created.record,
                )
            raise
        except Exception:
            async with self._cv:
                self._remove_request_from_memory_locked(work.request_id)
            if self._use_task_state_store and isinstance(created, CreateTaskResult) and created.created:
                await self._forget_created_task_after_append_cancel(
                    work,
                    append_attempt_id=append_attempt_id,
                    _durable=created.record,
                )
            raise
        return AppendWorkResult(
            ok=True,
            request_id=work.request_id,
            domain_key=work.domain_key,
            scheduler_instance_id=self._instance_id,
            backlog_depth=backlog_depth,
            assigned=assigned,
            extra={"sampling_inflight_admission": admission},
        )

    async def cancel_request(self, *, request_id: str, reason: str = "cancelled") -> CancelTaskResult:
        request_id = str(request_id)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
            try:
                await self._task_state_call(
                    "complete_task_failure",
                    request_id=request_id,
                    error=f"Model work request cancelled: {reason}",
                    metadata={
                        "terminal_status": "cancelled",
                        "cancelled_at": time.time(),
                        "cancel_reason": str(reason),
                    },
                )
            except TaskStateConflictError:
                # Task is already in a conflicting state (e.g. already terminal).
                # Treat as success — no need for a second get_task call.
                pass
            except Exception as exc:
                if self._task_not_found_cause(exc) is None:
                    raise
        async with self._cv:
            removed = self._remove_request_from_memory_locked(request_id)
            if removed:
                self._failed += 1
            return CancelTaskResult(
                ok=True,
                request_id=request_id,
                cancelled=removed,
                was_terminal=not removed,
                reason=ConflictReason.CANCELLED,
            )

    async def contains_request(
        self,
        *,
        request_id: str,
        hydrate_task_state: bool = True,
    ) -> ContainsResult:
        request_id = str(request_id)
        if self._use_task_state_store and hydrate_task_state:
            await self._ensure_task_state_ready()
        async with self._cv:
            location = self._request_locations.get(request_id)
            lease_id = self._lease_id_by_request_id.get(request_id)
            return ContainsResult(
                ok=True,
                request_id=request_id,
                present=location is not None,
                location=location,
                lease_id=lease_id,
                scheduler_instance_id=self._instance_id,
            )

    async def is_empty(self) -> bool:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            return (
                not any(self._domain_backlog.values())
                and not any(self._replica_queues.values())
                and not self._leases_by_id
            )

    async def sync_replicas(
        self,
        replicas: list[dict[str, Any]],
        *,
        hydrate_task_state: bool = True,
    ) -> SyncReplicasResult:
        self._ensure_assignment_loop_started()
        now = time.time()
        if self._use_task_state_store:
            if hydrate_task_state:
                await self._ensure_task_state_ready()
            else:
                await self._ensure_task_state_owner()
        incoming = {
            _queue_key(reg.domain_key, reg.replica_id): reg
            for reg in (ModelReplicaRegistration.from_dict(replica) for replica in replicas)
        }
        requeued = 0
        async with self._cv:
            if self._has_inflight_scheduler_transition_locked():
                assigned = AssignPendingResult(
                    ok=True,
                    assigned=0,
                    skipped_domains=[],
                    extra={"deferred": "inflight_scheduler_transition"},
                )
                return SyncReplicasResult(
                    ok=True,
                    replicas=len(self._replicas),
                    removed=0,
                    requeued=0,
                    expired=0,
                    assigned=assigned.to_wire(),
                    extra={"deferred": "inflight_scheduler_transition"},
                )
            previous_replicas = dict(self._replicas)
            previous_queues = {
                key: deque(queue)
                for key, queue in self._replica_queues.items()
            }
            removed = set(self._replicas) - set(incoming)
            changed_unclaimable: set[tuple[str, str]] = set()
            for key, old in self._replicas.items():
                new = incoming.get(key)
                if new is None:
                    changed_unclaimable.add(key)
                elif (
                    not new.claimable
                    or new.consumer_id != old.consumer_id
                    or int(new.generation) != int(old.generation)
                ):
                    changed_unclaimable.add(key)

            self._replicas = incoming
            for key in incoming:
                self._replica_queues.setdefault(key, deque())

            requeue_items: list[_PendingRequeue] = []
            for key in changed_unclaimable:
                queue = self._replica_queues.get(key)
                while queue:
                    assigned = queue.pop()
                    self._request_locations[assigned.item.request_id] = "requeueing"
                    requeue_items.append(_PendingRequeue(assigned=assigned))
                lease_items = sorted(
                    list(self._leases_by_id.items()),
                    key=lambda item: (float(item[1].leased_at), str(item[0])),
                    reverse=True,
                )
                for lease_id, lease in lease_items:
                    if _queue_key(lease.domain_key, lease.replica_id) == key:
                        if lease.finalizing_until is not None and lease.finalizing_until > now:
                            continue
                        self._leases_by_id.pop(lease_id, None)
                        self._lease_id_by_request_id.pop(lease.item.request_id, None)
                        assigned = _AssignedWork(
                            item=lease.item,
                            replica_id=lease.replica_id,
                            queue_id=lease.queue_id,
                            assigned_at=now,
                            assignment_generation=lease.consumer_generation,
                            assignment_reason="lease_requeued_replica_unclaimable",
                        )
                        self._request_locations[lease.item.request_id] = "requeueing"
                        requeue_items.append(_PendingRequeue(assigned=assigned, lease=lease))

            for key in removed:
                self._replica_queues.pop(key, None)

            expired_items = self._prepare_expired_leases_locked(now=now)
            replica_count = len(self._replicas)
            removed_count = len(removed)
        try:
            requeued += await self._commit_requeues_unlocked(requeue_items, reason="replica_unclaimable")
        except asyncio.CancelledError as exc:
            committed_request_ids = set(getattr(exc, "_model_work_requeue_committed_request_ids", set()))
            if committed_request_ids:
                previous_queues = {
                    key: deque(
                        assigned
                        for assigned in queue
                        if assigned.item.request_id not in committed_request_ids
                    )
                    for key, queue in previous_queues.items()
                }
            async with self._cv:
                for expired_pending in expired_items:
                    self._restore_pending_requeue_locked(expired_pending)
                self._replicas = previous_replicas
                self._replica_queues = previous_queues
            raise
        except Exception as exc:
            committed_request_ids = set(getattr(exc, "_model_work_requeue_committed_request_ids", set()))
            if committed_request_ids:
                previous_queues = {
                    key: deque(
                        assigned
                        for assigned in queue
                        if assigned.item.request_id not in committed_request_ids
                    )
                    for key, queue in previous_queues.items()
                }
            async with self._cv:
                for expired_pending in expired_items:
                    self._restore_pending_requeue_locked(expired_pending)
                self._replicas = previous_replicas
                self._replica_queues = previous_queues
            raise
        expired = await self._commit_requeues_unlocked(expired_items, reason="lease_expired")
        assigned_pending = await self._assign_pending_if_no_inflight_unlocked(
            hydrate_task_state=hydrate_task_state,
        )
        return SyncReplicasResult(
            ok=True,
            replicas=replica_count,
            removed=removed_count,
            requeued=requeued + expired,
            expired=expired,
            assigned=assigned_pending.to_wire(),
        )

    async def _assign_pending_typed(
        self,
        *,
        max_items: int | None = None,
        hydrate_task_state: bool = True,
    ) -> AssignPendingResult:
        self._ensure_assignment_loop_started()
        if self._use_task_state_store:
            if hydrate_task_state:
                await self._ensure_task_state_ready()
            else:
                await self._ensure_task_state_owner()
        expired = await self._expire_leases_unlocked(now=time.time())
        out = await self._assign_pending_core_unlocked(max_items=max_items)
        return AssignPendingResult(
            ok=out.ok,
            assigned=out.assigned,
            skipped_domains=list(out.skipped_domains),
            reason=out.reason,
            extra={**dict(out.extra), "expired": expired},
        )

    async def assign_pending(
        self,
        *,
        max_items: int | None = None,
        hydrate_task_state: bool = True,
    ) -> AssignPendingResult:
        return await self._assign_pending_typed(
            max_items=max_items,
            hydrate_task_state=hydrate_task_state,
        )

    def _validate_claimer(
        self,
        *,
        domain_key: str,
        replica_id: str,
        consumer_id: str,
        consumer_generation: int,
    ) -> ModelReplicaRegistration:
        key = _queue_key(domain_key, replica_id)
        replica = self._replicas.get(key)
        if replica is None:
            raise ModelWorkSchedulerConflictError(
                f"unknown replica domain={domain_key!r} replica_id={replica_id!r}"
            )
        if not replica.claimable:
            raise ModelWorkSchedulerConflictError(
                f"replica {replica_id!r} is not claimable: status={replica.status!r}"
            )
        if replica.consumer_id != consumer_id:
            raise ModelWorkSchedulerConflictError(
                f"consumer_id mismatch for replica {replica_id!r}: "
                f"expected {replica.consumer_id!r}, got {consumer_id!r}"
            )
        if int(replica.generation) != int(consumer_generation):
            raise ModelWorkSchedulerConflictError(
                f"generation mismatch for replica {replica_id!r}: "
                f"expected {replica.generation}, got {consumer_generation}"
            )
        return replica

    async def claim_from_replica_queue(
        self,
        *,
        domain_key: str,
        replica_id: str,
        consumer_id: str,
        consumer_generation: int,
        max_items: int = 1,
        token_budget: int | None = None,
        lease_ttl_s: float = 30.0,
    ) -> ClaimResult:
        self._ensure_assignment_loop_started()
        now = time.time()
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        claimed: list[dict[str, Any]] = []
        spent = 0
        claim_affinity_group: str | None = None
        while len(claimed) < max(1, int(max_items)):
            async with self._cv:
                self._validate_claimer(
                    domain_key=domain_key,
                    replica_id=replica_id,
                    consumer_id=consumer_id,
                    consumer_generation=consumer_generation,
                )
                queue = self._queue(domain_key, replica_id)
                same_affinity_only = self._claim_requires_same_affinity(domain_key)
                if same_affinity_only and max(1, int(max_items)) > 1:
                    self._cluster_queue_head_affinity(queue)
                if not queue:
                    break
                assigned = queue[0]
                if self._request_locations.get(assigned.item.request_id) == "assigning":
                    break
                if self._ordering_key_has_active_lease_locked(assigned.item.ordering_key):
                    break
                assigned_affinity_group = assigned.item.affinity_group
                if same_affinity_only and claimed and assigned_affinity_group != claim_affinity_group:
                    break
                cost = max(1, int(assigned.item.token_cost))
                if token_budget is not None and claimed and spent + cost > int(token_budget):
                    break
                lease_id = uuid.uuid4().hex
                claim_attempt = int(assigned.item.extra.get("claim_attempt") or 0) + 1
                attempt_id = str(assigned.item.extra.get("model_work_attempt_id") or uuid.uuid4().hex)
                item = assigned.item
                if item.extra.get("model_work_attempt_id") != attempt_id:
                    item = ModelWorkItem(
                        **{
                            **asdict(item),
                            "request_json": item.request_json,
                            "extra": {**dict(item.extra), "model_work_attempt_id": attempt_id},
                        }
                    )
                    assigned.item = item
                queue.popleft()
                self._request_locations[assigned.item.request_id] = "claiming"
            if self._use_task_state_store:
                claim_durable: dict[str, object] | None = None
                try:
                    claim_result = await self._task_state_call(
                        "claim_task",
                        request_id=item.request_id,
                        subqueue_id=assigned.queue_id,
                        lease_id=lease_id,
                        attempt_id=attempt_id,
                        consumer_id=consumer_id,
                        scheduler_epoch=int(self._scheduler_epoch or 0),
                        runtime_generation=int(consumer_generation),
                        lease_ttl_s=max(1.0, float(lease_ttl_s)),
                    )
                    if isinstance(claim_result, TaskMutationResult) and claim_result.record:
                        claim_durable = claim_result.record
                except asyncio.CancelledError:
                    await self._restore_or_commit_claiming_after_cancel(
                        assigned,
                        lease_id=lease_id,
                        attempt_id=attempt_id,
                        consumer_id=consumer_id,
                        consumer_generation=int(consumer_generation),
                        lease_ttl_s=lease_ttl_s,
                        _durable=claim_durable,
                    )
                    raise
                except Exception as e:
                    if self._task_not_found_cause(e) is not None:
                        async with self._cv:
                            if self._drop_claiming_request_locked(assigned):
                                logger.warning(
                                    "[model_work_scheduler] dropped stale assigned queue item with missing task state during claim request_id=%s queue_id=%s",
                                    assigned.item.request_id,
                                    assigned.queue_id,
                                )
                        continue
                    conflict = self._claim_conflict_cause(e)
                    if conflict is None:
                        async with self._cv:
                            if self._request_locations.get(assigned.item.request_id) == "claiming":
                                self._restore_assigned_to_queue_locked(assigned)
                        raise
                    reconciled = await self._reconcile_assigned_claim_conflict(
                        assigned,
                        conflict=conflict,
                    )
                    async with self._cv:
                        if reconciled is None:
                            self._restore_assigned_to_queue_locked(assigned)
                            raise
                    continue
            async with self._cv:
                if self._request_locations.get(assigned.item.request_id) != "claiming":
                    continue
                lease = ModelWorkLease(
                    lease_id=lease_id,
                    item=item,
                    domain_key=domain_key,
                    replica_id=replica_id,
                    queue_id=assigned.queue_id,
                    attempt_id=attempt_id,
                    scheduler_epoch=self._scheduler_epoch,
                    consumer_id=consumer_id,
                    consumer_generation=int(consumer_generation),
                    leased_at=now,
                    lease_expires_at=now + max(1.0, float(lease_ttl_s)),
                    claim_attempt=claim_attempt,
                    last_requeue_reason=assigned.item.extra.get("last_requeue_reason"),
                )
                self._leases_by_id[lease_id] = lease
                self._lease_id_by_request_id[assigned.item.request_id] = lease_id
                self._request_locations[assigned.item.request_id] = "leased"
                claimed.append(lease.to_dict())
                if claim_affinity_group is None:
                    claim_affinity_group = assigned_affinity_group
                spent += cost
        async with self._cv:
            self._claimed += 1
            remaining_queue_depth = len(self._queue(domain_key, replica_id))
        return ClaimResult(ok=True, leases=claimed, remaining_queue_depth=remaining_queue_depth)

    async def begin_finalize_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
    ) -> LeaseResult:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return LeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return LeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            now = time.time()
            request_id = lease.item.request_id
            attempt_id = lease.attempt_id
            scheduler_epoch = int(self._scheduler_epoch or 0)
            self._request_locations[request_id] = "finalizing"
        if self._use_task_state_store:
            finalize_result: Any = None
            try:
                finalize_result = await self._task_state_call(
                    "begin_finalize",
                    request_id=request_id,
                    lease_id=str(lease_id),
                    attempt_id=attempt_id,
                    scheduler_epoch=scheduler_epoch,
                    runtime_generation=int(consumer_generation),
                    finalize_ttl_s=max(1.0, float(finalize_ttl_s)),
                    staged_payload_path=staged_payload_path,
                )
            except asyncio.CancelledError:
                durable_finalizing = False
                _fr: dict[str, object] | None = (
                    finalize_result.record
                    if isinstance(finalize_result, TaskMutationResult)
                    else None
                )
                if _fr is not None:
                    durable_finalizing = (
                        str(_fr.get("status") or "") == "finalizing"
                        and str(_fr.get("lease_id") or "") == str(lease_id)
                        and str(_fr.get("attempt_id") or "") == str(attempt_id)
                        and int(_fr.get("scheduler_epoch") or 0) == scheduler_epoch
                        and int(_fr.get("runtime_generation") or 0) == int(consumer_generation)
                    )
                else:
                    try:
                        task_record = await self._task_state_call("get_task", request_id=request_id)
                        record = self._task_record_data(task_record)
                        durable_finalizing = (
                            str(record.get("status") or "") == "finalizing"
                            and str(record.get("lease_id") or "") == str(lease_id)
                            and str(record.get("attempt_id") or "") == str(attempt_id)
                            and int(record.get("scheduler_epoch") or 0) == scheduler_epoch
                            and int(record.get("runtime_generation") or 0) == int(consumer_generation)
                        )
                    except Exception:
                        durable_finalizing = False
                async with self._cv:
                    lease = self._leases_by_id.get(str(lease_id))
                    if (
                        lease is not None
                        and lease.item.request_id == request_id
                        and lease.attempt_id == attempt_id
                        and lease.consumer_id == consumer_id
                        and int(lease.consumer_generation) == int(consumer_generation)
                        and self._request_locations.get(request_id) == "finalizing"
                    ):
                        if durable_finalizing:
                            lease.finalizing_until = time.time() + max(1.0, float(finalize_ttl_s))
                            lease.lease_expires_at = max(float(lease.lease_expires_at), lease.finalizing_until)
                        self._request_locations[request_id] = "leased"
                    raise
            except Exception:
                async with self._cv:
                    lease = self._leases_by_id.get(str(lease_id))
                    if (
                        lease is not None
                        and lease.item.request_id == request_id
                        and lease.attempt_id == attempt_id
                        and lease.consumer_id == consumer_id
                        and int(lease.consumer_generation) == int(consumer_generation)
                        and self._request_locations.get(request_id) == "finalizing"
                    ):
                        self._request_locations[request_id] = "leased"
                raise
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return LeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if (
                lease.item.request_id != request_id
                or lease.attempt_id != attempt_id
                or lease.consumer_id != consumer_id
                or int(lease.consumer_generation) != int(consumer_generation)
            ):
                return LeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            lease.finalizing_until = now + max(1.0, float(finalize_ttl_s))
            lease.lease_expires_at = max(float(lease.lease_expires_at), lease.finalizing_until)
            self._request_locations[request_id] = "leased"
            return LeaseResult(ok=True, lease=lease.to_dict())

    async def renew_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float = 30.0,
    ) -> LeaseResult:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return LeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return LeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            request_id = lease.item.request_id
            attempt_id = lease.attempt_id
            scheduler_epoch = int(self._scheduler_epoch or 0)
            new_expires_at = time.time() + max(1.0, float(lease_ttl_s))
        renew_rejected: str | None = None
        if self._use_task_state_store:
            try:
                renewed = await self._task_state_call(
                    "renew_lease",
                    request_id=request_id,
                    lease_id=str(lease_id),
                    attempt_id=attempt_id,
                    scheduler_epoch=scheduler_epoch,
                    runtime_generation=int(consumer_generation),
                    lease_ttl_s=max(1.0, float(lease_ttl_s)),
                )
            except Exception as exc:
                if self._task_not_found_cause(exc) is not None:
                    async with self._cv:
                        current = self._leases_by_id.get(str(lease_id))
                        if current is not None and current.item.request_id == request_id:
                            self._remove_request_from_memory_locked(request_id)
                    return LeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
                raise
            if isinstance(renewed, TaskMutationResult):
                if not renewed.ok:
                    renew_rejected = renewed.reason or ConflictReason.RENEW_REJECTED
                record = renewed.record
                if isinstance(record, dict) and record.get("lease_expires_at") is not None:
                    new_expires_at = float(record["lease_expires_at"])
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return LeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if (
                lease.item.request_id != request_id
                or lease.attempt_id != attempt_id
                or lease.consumer_id != consumer_id
                or int(lease.consumer_generation) != int(consumer_generation)
            ):
                return LeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            if renew_rejected is not None:
                if renew_rejected == ConflictReason.TERMINAL:
                    self._remove_request_from_memory_locked(request_id)
                return LeaseResult(ok=False, reason=renew_rejected)
            lease.lease_expires_at = new_expires_at
            return LeaseResult(ok=True, lease=lease.to_dict())

    async def batch_renew_leases(
        self,
        *,
        items: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        """Batch-renew multiple leases in a single call.

        Each item must contain: lease_id, consumer_id, consumer_generation,
        and lease_ttl_s.
        """
        results: list[dict[str, Any]] = []
        for item in items:
            try:
                result = await self.renew_lease(
                    lease_id=str(item["lease_id"]),
                    consumer_id=str(item["consumer_id"]),
                    consumer_generation=int(item["consumer_generation"]),
                    lease_ttl_s=float(item["lease_ttl_s"]),
                )
                results.append(result.to_wire())
            except Exception as exc:
                results.append({
                    "ok": False,
                    "reason": "unknown",
                    "lease_id": item.get("lease_id"),
                    "error": str(exc),
                })
        return results

    async def _terminal_task_state_for_lease(self, lease: ModelWorkLease) -> _TerminalTaskState:
        if not self._use_task_state_store:
            return _TerminalTaskState(ok=True, status="unknown")
        try:
            task_record = await self._task_state_call("get_task", request_id=lease.item.request_id)
        except Exception as exc:
            if self._task_not_found_cause(exc) is not None:
                return _TerminalTaskState(ok=True, status="missing")
            raise
        if not isinstance(task_record, TaskRecord):
            return _TerminalTaskState(ok=False, reason=ConflictReason.TASK_STATE_INVALID)
        record = self._task_record_data(task_record)
        status = str(record.get("status") or "")
        if status not in TERMINAL_TASK_STATUSES:
            return _TerminalTaskState(ok=False, reason=ConflictReason.NOT_TERMINAL)
        if str(record.get("lease_id") or "") != str(lease.lease_id):
            return _TerminalTaskState(ok=False, reason=ConflictReason.STALE_CONSUMER)
        if str(record.get("attempt_id") or "") != str(lease.attempt_id):
            return _TerminalTaskState(ok=False, reason=ConflictReason.STALE_CONSUMER)
        if int(record.get("scheduler_epoch") or 0) != int(lease.scheduler_epoch or 0):
            return _TerminalTaskState(ok=False, reason=ConflictReason.STALE_CONSUMER)
        if int(record.get("runtime_generation") or 0) != int(lease.consumer_generation):
            return _TerminalTaskState(ok=False, reason=ConflictReason.STALE_CONSUMER)
        return _TerminalTaskState(ok=True, status=status, record=record)

    def _current_lease_matches_locked(
        self,
        current: ModelWorkLease,
        *,
        expected: ModelWorkLease,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
    ) -> bool:
        request_id = expected.item.request_id
        return (
            str(current.lease_id) == str(lease_id)
            and current.item.request_id == request_id
            and str(current.attempt_id) == str(expected.attempt_id)
            and int(current.scheduler_epoch or 0) == int(expected.scheduler_epoch or 0)
            and current.consumer_id == consumer_id
            and int(current.consumer_generation) == int(consumer_generation)
            and self._lease_id_by_request_id.get(request_id) == str(lease_id)
        )

    async def _finish_lease_terminal(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        consumer_id: str,
        consumer_generation: int,
        success: bool,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        error: str | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
    ) -> FinishResult:
        request_id = str(request_id)
        lease_id = str(lease_id)
        attempt_id = str(attempt_id)
        scheduler_epoch = int(scheduler_epoch)
        consumer_generation = int(consumer_generation)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(lease_id)
            if lease is not None:
                if lease.consumer_id != consumer_id or int(lease.consumer_generation) != consumer_generation:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if (
                    lease.item.request_id != request_id
                    or str(lease.attempt_id) != attempt_id
                    or int(lease.scheduler_epoch or 0) != scheduler_epoch
                ):
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if self._request_locations.get(request_id) == "finalizing":
                    return FinishResult(ok=False, reason=ConflictReason.FINALIZE_INFLIGHT)
            elif self._request_locations.get(request_id) is not None:
                return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
        if self._use_task_state_store:
            try:
                if success:
                    committed = await self._task_state_call(
                        "commit_finalize_success",
                        request_id=request_id,
                        lease_id=lease_id,
                        attempt_id=attempt_id,
                        scheduler_epoch=scheduler_epoch,
                        runtime_generation=consumer_generation,
                        result_path=str(result_path),
                        result_checksum=result_checksum,
                        result_size_bytes=result_size_bytes,
                        billing_observations=billing_observations,
                    )
                else:
                    committed = await self._task_state_call(
                        "commit_finalize_failure",
                        request_id=request_id,
                        lease_id=lease_id,
                        attempt_id=attempt_id,
                        scheduler_epoch=scheduler_epoch,
                        runtime_generation=consumer_generation,
                        error=str(error or "failed"),
                        result_path=result_path,
                        result_checksum=result_checksum,
                        result_size_bytes=result_size_bytes,
                    )
            except TaskStateConflictError:
                async with self._cv:
                    current = self._leases_by_id.get(lease_id)
                    if current is not None and not self._current_lease_matches_locked(
                        current,
                        expected=lease,
                        lease_id=lease_id,
                        consumer_id=consumer_id,
                        consumer_generation=consumer_generation,
                    ):
                        return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                    if current is None and self._request_locations.get(request_id) is not None:
                        return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                return FinishResult(ok=False, reason=ConflictReason.TASK_STATE_INVALID)
            except asyncio.CancelledError:
                if lease is not None:
                    terminal = await self._terminal_task_state_for_lease(lease)
                    if terminal.ok:
                        await self._release_finished_lease_projection(
                            lease,
                            lease_id=lease_id,
                            consumer_id=consumer_id,
                            consumer_generation=consumer_generation,
                            success=str(terminal.status or "") == "done",
                        )
                raise
            except Exception as exc:
                if self._task_not_found_cause(exc) is not None:
                    return FinishResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
                raise
            if isinstance(committed, TaskMutationResult):
                record = committed.record
                if not isinstance(record, dict):
                    return FinishResult(ok=False, reason=ConflictReason.TASK_STATE_INVALID)
                status = str(record.get("status") or "")
                expected_status = "done" if success else "failed"
                if status != expected_status:
                    return FinishResult(ok=False, reason=ConflictReason.TASK_STATE_INVALID)
                if str(record.get("lease_id") or "") != lease_id:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if str(record.get("attempt_id") or "") != attempt_id:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if int(record.get("scheduler_epoch") or 0) != scheduler_epoch:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if int(record.get("runtime_generation") or 0) != consumer_generation:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                if str(record.get("consumer_id") or "") != str(consumer_id):
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                idempotent = bool(committed.idempotent)
            else:
                idempotent = False
        else:
            idempotent = False
        if lease is None:
            return FinishResult(
                ok=True,
                request_id=request_id,
                status="done" if success else "failed",
                idempotent=True,
            )
        released = await self._release_finished_lease_projection(
            lease,
            lease_id=lease_id,
            consumer_id=consumer_id,
            consumer_generation=consumer_generation,
            success=success,
        )
        if released.ok and idempotent:
            released = FinishResult(
                ok=released.ok,
                request_id=released.request_id,
                status=released.status,
                reason=released.reason,
                idempotent=True,
                extra=dict(released.extra),
            )
        return released

    async def _release_finished_lease_projection(
        self,
        lease: ModelWorkLease,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        success: bool,
    ) -> FinishResult:
        request_id = lease.item.request_id
        async with self._cv:
            current = self._leases_by_id.get(str(lease_id))
            if current is None:
                if self._request_locations.get(request_id) is not None:
                    return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                return FinishResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if not self._current_lease_matches_locked(
                current,
                expected=lease,
                lease_id=str(lease_id),
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
            ):
                return FinishResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            self._leases_by_id.pop(current.lease_id, None)
            self._lease_id_by_request_id.pop(request_id, None)
            self._remove_request_location(request_id)
            if success:
                self._completed += 1
            else:
                self._failed += 1
            return FinishResult(
                ok=True,
                request_id=request_id,
                status="done" if success else "failed",
            )

    async def finish_lease_success(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        consumer_id: str,
        consumer_generation: int,
        result_path: str,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
    ) -> FinishResult:
        return await self._finish_lease_terminal(
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            consumer_id=consumer_id,
            consumer_generation=consumer_generation,
            success=True,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            billing_observations=billing_observations,
        )

    async def finish_lease_failure(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        consumer_id: str,
        consumer_generation: int,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
    ) -> FinishResult:
        return await self._finish_lease_terminal(
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            consumer_id=consumer_id,
            consumer_generation=consumer_generation,
            success=False,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            error=error,
        )

    async def validate_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
    ) -> ValidateLeaseResult:
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return ValidateLeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return ValidateLeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            return ValidateLeaseResult(ok=True, request_id=lease.item.request_id)

    async def fail_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        requeue: bool = True,
        reason: str = "failed",
        abort_finalize: bool = False,
    ) -> FailLeaseResult:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return FailLeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return FailLeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            if self._request_locations.get(lease.item.request_id) == "finalizing":
                return FailLeaseResult(ok=False, reason=ConflictReason.FINALIZE_INFLIGHT)
            request_id = lease.item.request_id
        if not requeue:
            terminal = await self._terminal_task_state_for_lease(lease)
            if not terminal.ok:
                return FailLeaseResult(ok=False, reason=terminal.reason or ConflictReason.NOT_TERMINAL)
        elif lease.finalizing_until is not None:
            if self._use_task_state_store:
                try:
                    task_record = await self._task_state_call("get_task", request_id=request_id)
                    record = self._task_record_data(task_record)
                except Exception as exc:
                    if self._task_not_found_cause(exc) is not None:
                        async with self._cv:
                            current = self._leases_by_id.get(str(lease_id))
                            if current is not None and current.item.request_id == request_id:
                                self._remove_request_from_memory_locked(request_id)
                        return FailLeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
                    raise
                status = str(record.get("status") or "")
                if status == "finalizing" and not abort_finalize:
                    try:
                        durable_finalizing_until = float(record.get("finalizing_until") or 0.0)
                    except Exception:
                        durable_finalizing_until = 0.0
                    if durable_finalizing_until > time.time():
                        return FailLeaseResult(ok=False, reason=ConflictReason.FINALIZE_IN_PROGRESS)
                if status == "finalizing":
                    pass
                elif status in TERMINAL_TASK_STATUSES:
                    terminal = await self._terminal_task_state_for_lease(lease)
                    if terminal.ok:
                        requeue = False
        async with self._cv:
            current = self._leases_by_id.get(str(lease_id))
            if current is None:
                if self._request_locations.get(request_id) is not None:
                    return FailLeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
                return FailLeaseResult(ok=False, reason=ConflictReason.UNKNOWN_LEASE)
            if not self._current_lease_matches_locked(
                current,
                expected=lease,
                lease_id=str(lease_id),
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
            ):
                return FailLeaseResult(ok=False, reason=ConflictReason.STALE_CONSUMER)
            if self._request_locations.get(request_id) == "finalizing":
                return FailLeaseResult(ok=False, reason=ConflictReason.FINALIZE_INFLIGHT)
            self._leases_by_id.pop(lease.lease_id, None)
            self._lease_id_by_request_id.pop(lease.item.request_id, None)
            requeued_out = False
            if requeue:
                assigned = _AssignedWork(
                    item=lease.item,
                    replica_id=lease.replica_id,
                    queue_id=lease.queue_id,
                    assigned_at=time.time(),
                    assignment_generation=lease.consumer_generation,
                    assignment_reason="lease_failed_requeue",
                )
                self._request_locations[request_id] = "requeueing"
            else:
                self._remove_request_location(request_id)
                self._failed += 1
        if requeue:
            requeued_out = bool(
                await self._commit_requeues_unlocked(
                    [_PendingRequeue(assigned=assigned, lease=lease)],
                    reason=reason,
                )
            )
        return FailLeaseResult(ok=True, request_id=request_id, requeued=requeued_out)

    async def expire_leases(self, *, now: float | None = None) -> ExpireResult:
        ts = time.time() if now is None else float(now)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        expired = await self._expire_leases_unlocked(now=ts)
        return ExpireResult(ok=True, expired=expired)

    def _stats_snapshot(self) -> dict[str, Any]:
        now = time.time()
        backlog_depth_by_domain = {
            domain: len(backlog) for domain, backlog in sorted(self._domain_backlog.items())
        }

        replica_queues = {}
        for key, queue in sorted(self._replica_queues.items()):
            replica = self._replicas.get(key)
            domain_key, replica_id = key
            queue_id = replica.effective_queue_id if replica is not None else f"{domain_key}::{replica_id}"
            replica_queues[queue_id] = {
                "domain_key": domain_key,
                "replica_id": replica_id,
                "depth": len(queue),
                "status": replica.status if replica is not None else "missing",
                "generation": replica.generation if replica is not None else None,
                "consumer_id": replica.consumer_id if replica is not None else None,
            }
        return {
            "actor_name": _ray_model_work_scheduler_actor_name(),
            "namespace": _ray_namespace(),
            "code_identity": CURRENT_CODE_IDENTITY,
            "scheduler_instance_id": self._instance_id,
            "scheduler_epoch": self._scheduler_epoch,
            "task_state_store_enabled": self._use_task_state_store,
            **self._loops.stats_snapshot(),
            "now": now,
            "depth": sum(backlog_depth_by_domain.values())
            + sum(len(queue) for queue in self._replica_queues.values())
            + len(self._leases_by_id),
            "backlog_depth": sum(backlog_depth_by_domain.values()),
            "backlog_depth_by_domain": backlog_depth_by_domain,
            "replicas": [replica.to_dict() for _, replica in sorted(self._replicas.items())],
            "replica_queues": replica_queues,
            "leases": [lease.to_dict() for lease in self._leases_by_id.values()],
            "sampling_inflight": self._admission.inflight_snapshot(),
            "sampling_admission_counters": self._admission.admission_counters_snapshot(),
            "counters": self._counters.snapshot(),
        }

    def stats(self) -> dict[str, Any]:
        return self._stats_snapshot()

    def ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "actor_name": _ray_model_work_scheduler_actor_name(),
            "namespace": _ray_namespace(),
            "scheduler_instance_id": self._instance_id,
            "code_identity": CURRENT_CODE_IDENTITY,
        }


def _await_ray_ref_sync(ref: Any, *, timeout_s: float | None = None) -> Any:
    return sync_get_ray_ref(ref, timeout_s=timeout_s)


def _create_ray_actor_handle():
    import ray

    actor_name = _ray_model_work_scheduler_actor_name()
    max_concurrency = int(os.environ.get("MINT_MODEL_WORK_SCHEDULER_ACTOR_MAX_CONCURRENCY", "64"))
    extra_env = otel_env_vars()
    if CURRENT_CODE_IDENTITY:
        extra_env["MINT_GIT_SHA"] = str(CURRENT_CODE_IDENTITY)
    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(
            extra=extra_env,
            include_ray_attach_hints=False,
            include_config_snapshot=False,
            tier=TIER_CPU,
        ),
    }
    resources = _model_work_scheduler_actor_resources()
    if resources:
        options["resources"] = resources
    else:
        apply_detached_actor_resources(options, ray)

    @ray.remote(
        num_cpus=0,
        max_concurrency=max_concurrency,
        max_restarts=0,
        concurrency_groups={"health": 8, "lookup": 16},
    )
    class _RayModelWorkSchedulerActor(_ModelWorkSchedulerActor):
        @ray.method(concurrency_group="health")
        def ping(self) -> dict[str, Any]:
            return super().ping()

        @ray.method(concurrency_group="lookup")
        async def contains_request(
            self,
            *,
            request_id: str,
            hydrate_task_state: bool = True,
        ) -> dict[str, Any]:
            out = await super().contains_request(
                request_id=request_id,
                hydrate_task_state=hydrate_task_state,
            )
            return out.to_wire()

        async def append(
            self,
            item: dict[str, Any],
            *,
            assign: bool = False,
            assign_max_items: int | None = None,
        ) -> dict[str, Any]:
            out = await super().append(
                item,
                assign=assign,
                assign_max_items=assign_max_items,
            )
            return out.to_wire()

        async def sync_replicas(
            self,
            replicas: list[dict[str, Any]],
            *,
            hydrate_task_state: bool = True,
        ) -> dict[str, Any]:
            out = await super().sync_replicas(
                replicas,
                hydrate_task_state=hydrate_task_state,
            )
            return out.to_wire()

        async def assign_pending(
            self,
            *,
            max_items: int | None = None,
            hydrate_task_state: bool = True,
        ) -> dict[str, Any]:
            out = await super().assign_pending(
                max_items=max_items,
                hydrate_task_state=hydrate_task_state,
            )
            return out.to_wire()

        async def reap_lost_pending_tasks(
            self,
            *,
            limit: int | None = None,
            reason: str = "scheduler_reaper_requeue",
        ) -> dict[str, Any]:
            return await super().reap_lost_pending_tasks(
                limit=limit,
                reason=reason,
            )

        async def validate_lease(
            self,
            *,
            lease_id: str,
            consumer_id: str,
            consumer_generation: int,
        ) -> dict[str, Any]:
            out = await super().validate_lease(
                lease_id=lease_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
            )
            return out.to_wire()

        async def renew_lease(
            self,
            *,
            lease_id: str,
            consumer_id: str,
            consumer_generation: int,
            lease_ttl_s: float = 30.0,
        ) -> dict[str, Any]:
            out = await super().renew_lease(
                lease_id=lease_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                lease_ttl_s=lease_ttl_s,
            )
            return out.to_wire()

        async def batch_renew_leases(
            self,
            *,
            items: list[dict[str, object]],
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            return await super().batch_renew_leases(items=items, **kwargs)

        async def begin_finalize_lease(
            self,
            *,
            lease_id: str,
            consumer_id: str,
            consumer_generation: int,
            finalize_ttl_s: float = 30.0,
            staged_payload_path: str | None = None,
        ) -> dict[str, Any]:
            out = await super().begin_finalize_lease(
                lease_id=lease_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                finalize_ttl_s=finalize_ttl_s,
                staged_payload_path=staged_payload_path,
            )
            return out.to_wire()

        async def claim_from_replica_queue(
            self,
            *,
            domain_key: str,
            replica_id: str,
            consumer_id: str,
            consumer_generation: int,
            max_items: int = 1,
            token_budget: int | None = None,
            lease_ttl_s: float = 30.0,
        ) -> dict[str, Any]:
            out = await super().claim_from_replica_queue(
                domain_key=domain_key,
                replica_id=replica_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                max_items=max_items,
                token_budget=token_budget,
                lease_ttl_s=lease_ttl_s,
            )
            return out.to_wire()

        async def finish_lease_success(
            self,
            *,
            request_id: str,
            lease_id: str,
            attempt_id: str,
            scheduler_epoch: int,
            consumer_id: str,
            consumer_generation: int,
            result_path: str,
            result_checksum: str | None = None,
            result_size_bytes: int | None = None,
            billing_observations: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            out = await super().finish_lease_success(
                request_id=request_id,
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=scheduler_epoch,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                result_path=result_path,
                result_checksum=result_checksum,
                result_size_bytes=result_size_bytes,
                billing_observations=billing_observations,
            )
            return out.to_wire()

        async def finish_lease_failure(
            self,
            *,
            request_id: str,
            lease_id: str,
            attempt_id: str,
            scheduler_epoch: int,
            consumer_id: str,
            consumer_generation: int,
            error: str,
            result_path: str | None = None,
            result_checksum: str | None = None,
            result_size_bytes: int | None = None,
        ) -> dict[str, Any]:
            out = await super().finish_lease_failure(
                request_id=request_id,
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=scheduler_epoch,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                error=error,
                result_path=result_path,
                result_checksum=result_checksum,
                result_size_bytes=result_size_bytes,
            )
            return out.to_wire()

        async def fail_lease(
            self,
            *,
            lease_id: str,
            consumer_id: str,
            consumer_generation: int,
            requeue: bool = True,
            reason: str = "failed",
            abort_finalize: bool = False,
        ) -> dict[str, Any]:
            out = await super().fail_lease(
                lease_id=lease_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
                requeue=requeue,
                reason=reason,
                abort_finalize=abort_finalize,
            )
            return out.to_wire()

        async def expire_leases(self, *, now: float | None = None) -> dict[str, Any]:
            out = await super().expire_leases(now=now)
            return out.to_wire()

        async def shutdown_background_loops(self) -> dict[str, Any]:
            return await super().shutdown_background_loops()

    actor = _RayModelWorkSchedulerActor.options(**options).remote(
        use_task_state_store=_model_work_scheduler_use_task_state_store_from_env(),
        same_affinity_multi_claim_domains=_same_affinity_multi_claim_domains_from_env(),
    )
    return actor


def _create_ray_actor(*, require_ready: bool = True):
    actor = _create_ray_actor_handle()
    if require_ready:
        out = _await_ray_ref_sync(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
    return actor


async def _create_ray_actor_async(*, require_ready: bool = True):
    actor = await asyncio.to_thread(_create_ray_actor_handle)
    if require_ready:
        out = await async_get_ray_ref(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
    return actor


class ModelWorkSchedulerClient:
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
        raise ModelWorkSchedulerCodeIdentityMismatchError(
            "model work scheduler code identity mismatch: "
            f"expected={CURRENT_CODE_IDENTITY!r} actual={actor_code_identity!r}"
        )

    def _kill_cached_actor_for_code_identity_mismatch(self, exc: BaseException) -> None:
        actor = self._ray_actor
        self._reset_ray_actor()
        if actor is None:
            raise exc
        _append_model_work_scheduler_debug(
            "kill_actor_code_identity_mismatch",
            expected_code_identity=CURRENT_CODE_IDENTITY,
            error=f"{type(exc).__name__}: {exc}",
        )
        try:
            import ray
        except Exception as e:
            raise ModelWorkSchedulerUnavailableError("Ray import failed") from e
        ray.kill(actor, no_restart=True)

    async def _get_ray_actor_async(
        self,
        *,
        require_ready: bool = True,
        create_if_missing: bool = False,
    ):
        _append_model_work_scheduler_debug("get_ray_actor_async_begin", require_ready=require_ready)
        try:
            import ray
        except Exception as e:
            _append_model_work_scheduler_debug(
                "get_ray_actor_async_import_error",
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(),
            )
            raise ModelWorkSchedulerUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            raise ModelWorkSchedulerUnavailableError("Ray not initialized")
        if self._ray_actor is not None:
            try:
                out = await self._await_ray_ref(self._ray_actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
                self._validate_code_identity(out)
                return self._ray_actor
            except ModelWorkSchedulerCodeIdentityMismatchError:
                self._reset_ray_actor()
                if not create_if_missing:
                    raise
            except Exception:
                self._reset_ray_actor()

        actor_name = _ray_model_work_scheduler_actor_name()
        try:
            self._ray_actor = await asyncio.to_thread(
                ray.get_actor,
                actor_name,
                namespace=_ray_namespace(),
            )
            out = await self._await_ray_ref(
                self._ray_actor.ping.remote(),
                timeout_s=5.0 if require_ready else 1.0,
            )
            if not isinstance(out, dict):
                raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
            self._validate_code_identity(out)
            return self._ray_actor
        except ModelWorkSchedulerCodeIdentityMismatchError as e:
            if not create_if_missing:
                raise
            await asyncio.to_thread(self._kill_cached_actor_for_code_identity_mismatch, e)
        except ValueError:
            if not create_if_missing:
                raise ModelWorkSchedulerUnavailableError(
                    f"Detached Ray ModelWorkScheduler actor unavailable actor_name={actor_name!r}"
                )
            logger.info("actor__s_not_found__creating")
        except Exception:
            if not create_if_missing:
                raise ModelWorkSchedulerUnavailableError(
                    f"Detached Ray ModelWorkScheduler actor unavailable actor_name={actor_name!r}"
                )
            logger.info("failed_to_fetch_actor__s__creating")

        try:
            self._ray_actor = await _create_ray_actor_async(require_ready=require_ready)
        except Exception as e:
            raise ModelWorkSchedulerUnavailableError(
                "Failed to get/create detached Ray ModelWorkScheduler actor"
            ) from e
        return self._ray_actor

    async def _await_ray_ref(self, ref: Any, *, timeout_s: float | None = None) -> Any:
        return await async_get_ray_ref(ref, timeout_s=None if timeout_s is None else float(timeout_s))

    async def append(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        user_id: str | None,
        apikey_id: str | None,
        throttle_principal: str | None,
        webhook_url: str | None,
        domain_key: str,
        affinity_group: str | None = None,
        ordering_key: str | None = None,
        token_cost: int = 1,
        extra: dict[str, Any] | None = None,
        assign: bool = False,
        assign_max_items: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        item = ModelWorkItem(
            request_id=str(request_id),
            op=str(op),
            request_json=bytes(request_json),
            user_id=user_id,
            apikey_id=apikey_id,
            throttle_principal=throttle_principal,
            webhook_url=webhook_url,
            extra={} if extra is None else dict(extra),
            created_at=time.time(),
            domain_key=str(domain_key),
            affinity_group=affinity_group,
            ordering_key=ordering_key,
            token_cost=int(token_cost),
        )
        out = await self._await_ray_ref(
            actor.append.remote(
                item.to_dict(),
                assign=bool(assign),
                assign_max_items=assign_max_items,
            ),
            timeout_s=10.0 if timeout_s is None else float(timeout_s),
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.append returned non-dict: {type(out)}")
        if not bool(out.get("ok")) and out.get("reason") == "duplicate_request_id":
            raise ModelWorkSchedulerConflictError(f"duplicate request_id: {request_id}")
        return AppendWorkResult.from_wire(out)

    async def append_work(self, **kwargs: Any) -> AppendWorkResult:
        return await self.append(**kwargs)

    async def sync_replicas(
        self,
        replicas: list[ModelReplicaRegistration | dict[str, Any]],
        *,
        hydrate_task_state: bool = True,
        timeout_s: float = 10.0,
    ) -> SyncReplicasResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        payload = [
            replica.to_dict() if isinstance(replica, ModelReplicaRegistration) else dict(replica)
            for replica in replicas
        ]
        out = await self._await_ray_ref(
            actor.sync_replicas.remote(
                payload,
                hydrate_task_state=bool(hydrate_task_state),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.sync_replicas returned non-dict: {type(out)}")
        return SyncReplicasResult.from_wire(out)

    async def assign_pending(
        self,
        *,
        max_items: int | None = None,
        timeout_s: float = 10.0,
    ) -> AssignPendingResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.assign_pending.remote(max_items=max_items),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.assign_pending returned non-dict: {type(out)}")
        return AssignPendingResult.from_wire(out)

    async def reap_lost_pending_tasks(
        self,
        *,
        limit: int | None = None,
        reason: str = "scheduler_reaper_requeue",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.reap_lost_pending_tasks.remote(limit=limit, reason=str(reason)),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.reap_lost_pending_tasks returned non-dict: {type(out)}")
        return out

    async def cancel_request(
        self,
        *,
        request_id: str,
        reason: str = "cancelled",
        timeout_s: float = 10.0,
    ) -> CancelTaskResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.cancel_request.remote(request_id=str(request_id), reason=str(reason)),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.cancel_request returned non-dict: {type(out)}")
        if "was_terminal" not in out:
            out = {**out, "was_terminal": not bool(out.get("cancelled"))}
        return CancelTaskResult.from_wire(out)

    async def contains_request(
        self,
        *,
        request_id: str,
        hydrate_task_state: bool = True,
        timeout_s: float = 10.0,
    ) -> ContainsResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.contains_request.remote(
                request_id=str(request_id),
                hydrate_task_state=bool(hydrate_task_state),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.contains_request returned non-dict: {type(out)}")
        return ContainsResult.from_wire(out)

    async def contains(self, **kwargs: Any) -> ContainsResult:
        return await self.contains_request(**kwargs)

    async def is_empty(self, *, timeout_s: float = 10.0) -> bool:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(actor.is_empty.remote(), timeout_s=timeout_s)
        return bool(out)

    async def claim_from_replica_queue(
        self,
        *,
        domain_key: str,
        replica_id: str,
        consumer_id: str,
        consumer_generation: int,
        max_items: int = 1,
        token_budget: int | None = None,
        lease_ttl_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> ClaimResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.claim_from_replica_queue.remote(
                domain_key=str(domain_key),
                replica_id=str(replica_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
                max_items=int(max_items),
                token_budget=token_budget,
                lease_ttl_s=float(lease_ttl_s),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(
                f"ModelWorkScheduler.claim_from_replica_queue returned non-dict: {type(out)}"
            )
        return ClaimResult.from_wire(out)

    async def claim(self, **kwargs: Any) -> ClaimResult:
        return await self.claim_from_replica_queue(**kwargs)

    async def renew_lease(
        self,
        *,
        lease: LeaseToken,
        lease_ttl_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> RenewResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.renew_lease.remote(
                lease_id=str(lease.lease_id),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
                lease_ttl_s=float(lease_ttl_s),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.renew_lease returned non-dict: {type(out)}")
        return RenewResult.from_wire(out)

    async def renew(
        self,
        *,
        lease: LeaseToken,
        lease_ttl_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> RenewResult:
        return await self.renew_lease(
            lease=lease,
            lease_ttl_s=lease_ttl_s,
            timeout_s=timeout_s,
        )

    async def batch_renew_leases(
        self,
        *,
        items: list[dict[str, object]],
        timeout_s: float = 10.0,
    ) -> list[dict[str, Any]]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.batch_renew_leases.remote(items=items),
            timeout_s=timeout_s,
        )
        if not isinstance(out, list):
            raise TypeError(f"ModelWorkScheduler.batch_renew_leases returned non-list: {type(out)}")
        return out

    async def finish_lease_success(
        self,
        *,
        lease: LeaseToken,
        result_path: str,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
        timeout_s: float = 10.0,
    ) -> FinishResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.finish_lease_success.remote(
                request_id=str(lease.request_id),
                lease_id=str(lease.lease_id),
                attempt_id=str(lease.attempt_id),
                scheduler_epoch=int(lease.scheduler_epoch),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
                result_path=str(result_path),
                result_checksum=result_checksum,
                result_size_bytes=result_size_bytes,
                billing_observations=billing_observations,
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.finish_lease_success returned non-dict: {type(out)}")
        return FinishResult.from_wire(out)

    async def finish_lease_failure(
        self,
        *,
        lease: LeaseToken,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        timeout_s: float = 10.0,
    ) -> FinishResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.finish_lease_failure.remote(
                request_id=str(lease.request_id),
                lease_id=str(lease.lease_id),
                attempt_id=str(lease.attempt_id),
                scheduler_epoch=int(lease.scheduler_epoch),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
                error=str(error),
                result_path=result_path,
                result_checksum=result_checksum,
                result_size_bytes=result_size_bytes,
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.finish_lease_failure returned non-dict: {type(out)}")
        return FinishResult.from_wire(out)

    async def finish_success(
        self,
        *,
        lease: LeaseToken,
        result_path: str,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
        timeout_s: float = 10.0,
    ) -> FinishResult:
        return await self.finish_lease_success(
            lease=lease,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            billing_observations=billing_observations,
            timeout_s=timeout_s,
        )

    async def finish_failure(
        self,
        *,
        lease: LeaseToken,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        timeout_s: float = 10.0,
    ) -> FinishResult:
        return await self.finish_lease_failure(
            lease=lease,
            error=error,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            timeout_s=timeout_s,
        )

    async def begin_finalize_lease(
        self,
        *,
        lease: LeaseToken,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
        timeout_s: float = 10.0,
    ) -> LeaseResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.begin_finalize_lease.remote(
                lease_id=str(lease.lease_id),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
                finalize_ttl_s=float(finalize_ttl_s),
                staged_payload_path=staged_payload_path,
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.begin_finalize_lease returned non-dict: {type(out)}")
        return LeaseResult.from_wire(out)

    async def begin_finalize(
        self,
        *,
        lease: LeaseToken,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
        timeout_s: float = 10.0,
    ) -> LeaseResult:
        return await self.begin_finalize_lease(
            lease=lease,
            finalize_ttl_s=finalize_ttl_s,
            staged_payload_path=staged_payload_path,
            timeout_s=timeout_s,
        )

    async def validate_lease(
        self,
        *,
        lease: LeaseToken,
        timeout_s: float = 10.0,
    ) -> ValidateLeaseResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.validate_lease.remote(
                lease_id=str(lease.lease_id),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.validate_lease returned non-dict: {type(out)}")
        return ValidateLeaseResult.from_wire(out)

    async def validate(
        self,
        *,
        lease: LeaseToken,
        timeout_s: float = 10.0,
    ) -> ValidateLeaseResult:
        return await self.validate_lease(lease=lease, timeout_s=timeout_s)

    async def fail_lease(
        self,
        *,
        lease: LeaseToken,
        requeue: bool = True,
        reason: str = "failed",
        abort_finalize: bool = False,
        timeout_s: float = 10.0,
    ) -> FailLeaseResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.fail_lease.remote(
                lease_id=str(lease.lease_id),
                consumer_id=str(lease.consumer_id),
                consumer_generation=int(lease.consumer_generation),
                requeue=bool(requeue),
                reason=str(reason),
                abort_finalize=bool(abort_finalize),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.fail_lease returned non-dict: {type(out)}")
        return FailLeaseResult.from_wire(out)

    async def fail(
        self,
        *,
        lease: LeaseToken,
        requeue: bool = True,
        reason: str = "failed",
        abort_finalize: bool = False,
        timeout_s: float = 10.0,
    ) -> FailLeaseResult:
        return await self.fail_lease(
            lease=lease,
            requeue=requeue,
            reason=reason,
            abort_finalize=abort_finalize,
            timeout_s=timeout_s,
        )

    async def expire_leases(
        self,
        *,
        now: float | None = None,
        timeout_s: float = 10.0,
    ) -> ExpireResult:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(actor.expire_leases.remote(now=now), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.expire_leases returned non-dict: {type(out)}")
        return ExpireResult.from_wire(out)

    async def expire(self, **kwargs: Any) -> ExpireResult:
        return await self.expire_leases(**kwargs)

    async def stats(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=create_if_missing)
        out = await self._await_ray_ref(actor.stats.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.stats returned non-dict: {type(out)}")
        try:
            self._validate_code_identity(out)
        except ModelWorkSchedulerCodeIdentityMismatchError as e:
            if not create_if_missing:
                raise
            await asyncio.to_thread(self._kill_cached_actor_for_code_identity_mismatch, e)
            actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=True)
            out = await self._await_ray_ref(actor.stats.remote(), timeout_s=timeout_s)
            if not isinstance(out, dict):
                raise TypeError(f"ModelWorkScheduler.stats returned non-dict: {type(out)}")
            self._validate_code_identity(out)
        return out

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        try:
            out = await self._await_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise ModelWorkSchedulerUnavailableError(f"ModelWorkScheduler ping failed: {out!r}")
        self._validate_code_identity(out)
        return out

    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self.async_ping(timeout_s=timeout_s)


model_work_scheduler = ModelWorkSchedulerClient()
