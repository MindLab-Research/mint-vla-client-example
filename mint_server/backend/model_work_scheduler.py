from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import (
    PFS_PYTHONPATH,
    actor_runtime_env,
    apply_detached_actor_resources,
    config as server_config,
    otel_env_vars,
    preferred_control_plane_resources,
)
from ..runtime_env import env_nonempty
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref

logger = logging.getLogger(__name__)

CLAIMABLE_REPLICA_STATUSES = frozenset({"healthy", "ready"})


class ModelWorkSchedulerUnavailableError(RuntimeError):
    pass


class ModelWorkSchedulerConflictError(RuntimeError):
    pass


def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _ray_model_work_scheduler_actor_name() -> str:
    env_value = os.environ.get("MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME")
    if env_value:
        return str(env_value)
    return str(getattr(server_config, "model_work_scheduler_actor_name", "mint_model_work_scheduler"))


def _otel_metric_attrs() -> dict[str, str]:
    attrs = {
        "deployment.env": os.getenv("MINT_DEPLOYMENT_ENV", "").strip(),
        "mint.cluster_id": os.getenv("MINT_CLUSTER_ID", "").strip(),
        "ray_namespace": _ray_namespace(),
    }
    return {key: value for key, value in attrs.items() if value}


def _metric_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scheduler_domain_base_model(domain_key: object) -> str | None:
    domain = str(domain_key or "").strip()
    if not domain or ":" not in domain:
        return None
    backend, model = domain.split(":", 1)
    if backend.strip().lower() != "vllm":
        return None
    model = model.strip()
    return model or None


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


def _queue_key(domain_key: str, replica_id: str) -> tuple[str, str]:
    return str(domain_key), str(replica_id)


class _ModelWorkSchedulerActor:
    def __init__(
        self,
        *,
        use_task_state_store: bool = False,
        task_state_store: Any | None = None,
        owner_id: str | None = None,
    ) -> None:
        try:
            from ..logging_context import init_actor_observability

            init_actor_observability()
        except Exception:
            logger.debug("[model_work_scheduler] actor observability init skipped", exc_info=True)
        self._cv = asyncio.Condition()
        self._instance_id = uuid.uuid4().hex
        self._owner_id = owner_id or f"{_ray_model_work_scheduler_actor_name()}:{self._instance_id}"
        self._use_task_state_store = bool(use_task_state_store)
        self._task_state_store = task_state_store
        self._scheduler_epoch: int | None = None
        self._task_state_hydrated = False
        self._domain_backlog: dict[str, deque[ModelWorkItem]] = {}
        self._replicas: dict[tuple[str, str], ModelReplicaRegistration] = {}
        self._replica_queues: dict[tuple[str, str], deque[_AssignedWork]] = {}
        self._leases_by_id: dict[str, ModelWorkLease] = {}
        self._lease_id_by_request_id: dict[str, str] = {}
        self._request_locations: dict[str, str] = {}
        self._affinity_replica: dict[tuple[str, str], str] = {}
        self._completed = 0
        self._failed = 0
        self._requeued = 0
        self._appended = 0
        self._assigned = 0
        self._claimed = 0
        self._assignment_loop_task: asyncio.Task | None = None
        self._assignment_loop_interval_s = float(
            os.environ.get("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "1.0")
        )
        self._otel_enabled = False
        self._otel_error: str | None = None
        self._init_otel_metrics()
        self._ensure_assignment_loop_started()

    def _init_otel_metrics(self) -> None:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation

            meter = metrics.get_meter("mint.model_work_scheduler")

            def _attrs(**extra: object) -> dict[str, str]:
                attrs = _otel_metric_attrs()
                for key, value in extra.items():
                    text = str(value if value is not None else "").strip()
                    if text:
                        attrs[key] = text
                return attrs

            def _gauge(name: str, callback, *, unit: str | None = None) -> None:
                kwargs: dict[str, Any] = {"callbacks": [callback]}
                if unit:
                    kwargs["unit"] = unit
                meter.create_observable_gauge(name, **kwargs)

            def _scalar(field: str):
                def _callback(_options):
                    value = _metric_number(self._stats_snapshot().get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            _gauge("mint_model_work_scheduler_depth", _scalar("depth"))
            _gauge("mint_model_work_scheduler_backlog_depth", _scalar("backlog_depth"))

            def _counter(field: str):
                def _callback(_options):
                    counters = self._stats_snapshot().get("counters")
                    if not isinstance(counters, dict):
                        return []
                    value = _metric_number(counters.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            for key in ("appended", "assigned", "claimed", "completed", "failed", "requeued"):
                _gauge(f"mint_model_work_scheduler_{key}_total", _counter(key))

            def _domain_backlog(_options):
                backlog_by_domain = self._stats_snapshot().get("backlog_depth_by_domain")
                if not isinstance(backlog_by_domain, dict):
                    return []
                observations = []
                for domain_key, depth in sorted(backlog_by_domain.items()):
                    value = _metric_number(depth)
                    if value is None:
                        continue
                    observations.append(Observation(value, _attrs(domain_key=domain_key)))
                return observations

            _gauge("mint_model_work_scheduler_domain_backlog_depth", _domain_backlog)

            def _replica_queue_depth(_options):
                replica_queues = self._stats_snapshot().get("replica_queues")
                if not isinstance(replica_queues, dict):
                    return []
                observations = []
                for queue_id, rec in sorted(replica_queues.items()):
                    if not isinstance(rec, dict):
                        continue
                    value = _metric_number(rec.get("depth"))
                    if value is None:
                        continue
                    observations.append(
                        Observation(
                            value,
                            _attrs(
                                domain_key=rec.get("domain_key") or "unknown",
                                replica_id=rec.get("replica_id") or "unknown",
                                queue_id=queue_id,
                                status=rec.get("status") or "unknown",
                            ),
                        )
                    )
                return observations

            _gauge("mint_model_work_scheduler_replica_queue_depth", _replica_queue_depth)

            def _leases(_options):
                leases = self._stats_snapshot().get("leases")
                if not isinstance(leases, list):
                    return []
                return [Observation(float(len(leases)), _attrs())]

            _gauge("mint_model_work_scheduler_leases", _leases)

            def _sample_model_load(metric: str):
                def _callback(_options):
                    stats = self._stats_snapshot()
                    load: dict[str, dict[str, float]] = {}
                    replica_queues = stats.get("replica_queues")
                    if isinstance(replica_queues, dict):
                        for rec in replica_queues.values():
                            if not isinstance(rec, dict):
                                continue
                            base_model = _scheduler_domain_base_model(rec.get("domain_key"))
                            if not base_model:
                                continue
                            bucket = load.setdefault(
                                base_model,
                                {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                            )
                            bucket["pending_requests"] += float(_metric_number(rec.get("depth")) or 0.0)
                            if str(rec.get("status") or "").lower() in CLAIMABLE_REPLICA_STATUSES:
                                bucket["capacity_workers"] += 1.0
                    leases = stats.get("leases")
                    if isinstance(leases, list):
                        for lease in leases:
                            if not isinstance(lease, dict):
                                continue
                            item = lease.get("item") if isinstance(lease.get("item"), dict) else {}
                            base_model = _scheduler_domain_base_model(
                                item.get("domain_key") or lease.get("domain_key")
                            )
                            if not base_model:
                                continue
                            bucket = load.setdefault(
                                base_model,
                                {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                            )
                            bucket["inflight_workers"] += 1.0
                    observations = []
                    for base_model, bucket in sorted(load.items()):
                        capacity = float(bucket.get("capacity_workers", 0.0))
                        values = {
                            "pending_requests": float(bucket.get("pending_requests", 0.0)),
                            "inflight_workers": float(bucket.get("inflight_workers", 0.0)),
                            "capacity_workers": capacity,
                            "load_pct": 0.0
                            if capacity <= 0.0
                            else 100.0 * float(bucket.get("inflight_workers", 0.0)) / capacity,
                        }
                        observations.append(
                            Observation(
                                values[metric],
                                _attrs(base_model=base_model, workload="sample"),
                            )
                        )
                    return observations

                return _callback

            _gauge("mint_model_load_pct", _sample_model_load("load_pct"))
            _gauge("mint_model_pending_requests", _sample_model_load("pending_requests"))
            _gauge("mint_model_inflight_workers", _sample_model_load("inflight_workers"))
            _gauge("mint_model_capacity_workers", _sample_model_load("capacity_workers"))

            self._otel_enabled = True
        except Exception as e:
            self._otel_error = f"{type(e).__name__}: {e}"

    def _all_request_ids(self) -> set[str]:
        return set(self._request_locations)

    async def _task_state_call(self, method: str, **kwargs: Any) -> Any:
        if self._task_state_store is None:
            from .task_state_store import task_state_store

            self._task_state_store = task_state_store
        async_method = getattr(self._task_state_store, f"async_{method}", None)
        if callable(async_method):
            return await async_method(**kwargs)
        sync_method = getattr(self._task_state_store, method)
        return sync_method(**kwargs)

    async def _ensure_task_state_owner(self) -> int | None:
        if not self._use_task_state_store:
            return None
        if self._scheduler_epoch is not None:
            renewed = await self._task_state_call(
                "renew_scheduler_owner",
                owner_id=self._owner_id,
                epoch=int(self._scheduler_epoch),
                ttl_s=float(getattr(server_config, "task_state_store_owner_ttl_s", 30.0)),
            )
            if isinstance(renewed, dict) and bool(renewed.get("ok")):
                return int(self._scheduler_epoch)
            self._scheduler_epoch = None
        acquired = await self._task_state_call(
            "acquire_scheduler_owner",
            owner_id=self._owner_id,
            ttl_s=float(getattr(server_config, "task_state_store_owner_ttl_s", 30.0)),
        )
        if not isinstance(acquired, dict) or not bool(acquired.get("ok")):
            raise ModelWorkSchedulerConflictError(f"failed to acquire scheduler owner: {acquired}")
        self._scheduler_epoch = int(acquired["epoch"])
        return int(self._scheduler_epoch)

    async def _assignment_loop(self) -> None:
        while True:
            await asyncio.sleep(self._assignment_loop_interval_s)
            try:
                await self.assign_pending()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[model_work_scheduler] assignment loop failed error_type=%s error=%s",
                    type(e).__name__,
                    e,
                )

    def _ensure_assignment_loop_started(self) -> None:
        if self._assignment_loop_interval_s <= 0:
            return
        if self._assignment_loop_task is not None and not self._assignment_loop_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._assignment_loop_task = loop.create_task(self._assignment_loop())

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

    async def _persist_requeue_task(self, request_id: str, *, reason: str) -> bool:
        if not self._use_task_state_store:
            return True
        out = await self._task_state_call(
            "requeue_task",
            request_id=str(request_id),
            scheduler_epoch=int(self._scheduler_epoch or 0),
            reason=str(reason),
        )
        if isinstance(out, dict) and bool(out.get("ok")):
            return True
        if isinstance(out, dict) and out.get("reason") == "terminal":
            return False
        raise ModelWorkSchedulerConflictError(f"failed to requeue task {request_id}: {out!r}")

    async def _ensure_task_state_ready(self) -> int | None:
        self._ensure_assignment_loop_started()
        epoch = await self._ensure_task_state_owner()
        if not self._use_task_state_store or self._task_state_hydrated:
            return epoch
        active = await self._task_state_call("list_active_tasks")
        if not isinstance(active, list):
            raise TypeError(f"TaskStateStore.list_active_tasks returned non-list: {type(active)}")
        async with self._cv:
            if self._task_state_hydrated:
                return epoch
            for record in active:
                if not isinstance(record, dict):
                    continue
                status = str(record.get("status") or "")
                item = self._work_item_from_task_record(record)
                if status != "pending":
                    should_requeue = await self._persist_requeue_task(
                        item.request_id,
                        reason="scheduler_hydrate_requeue",
                    )
                    if not should_requeue:
                        continue
                self._backlog(item.domain_key).append(item)
                self._request_locations[item.request_id] = "backlog"
            self._task_state_hydrated = True
        return epoch

    def _backlog(self, domain_key: str) -> deque[ModelWorkItem]:
        return self._domain_backlog.setdefault(str(domain_key), deque())

    def _queue(self, domain_key: str, replica_id: str) -> deque[_AssignedWork]:
        return self._replica_queues.setdefault(_queue_key(domain_key, replica_id), deque())

    def _drop_empty_backlog(self, domain_key: str) -> None:
        backlog = self._domain_backlog.get(domain_key)
        if backlog is not None and not backlog:
            self._domain_backlog.pop(domain_key, None)

    def _claimable_replicas(self, domain_key: str) -> list[ModelReplicaRegistration]:
        replicas = [
            replica
            for (replica_domain, _), replica in self._replicas.items()
            if replica_domain == domain_key and replica.claimable
        ]
        active_by_replica: dict[str, int] = {}
        for lease in self._leases_by_id.values():
            if lease.domain_key != domain_key:
                continue
            active_by_replica[lease.replica_id] = active_by_replica.get(lease.replica_id, 0) + 1
        replicas.sort(
            key=lambda r: (
                active_by_replica.get(r.replica_id, 0) + len(self._queue(r.domain_key, r.replica_id)),
                active_by_replica.get(r.replica_id, 0),
                len(self._queue(r.domain_key, r.replica_id)),
                r.replica_id,
            )
        )
        return replicas

    def _choose_replica(self, item: ModelWorkItem) -> ModelReplicaRegistration | None:
        replicas = self._claimable_replicas(item.domain_key)
        if not replicas:
            return None
        if item.affinity_group:
            affinity_key = (item.domain_key, item.affinity_group)
            sticky = self._affinity_replica.get(affinity_key)
            if sticky is not None:
                for replica in replicas:
                    if replica.replica_id == sticky:
                        return replica
        replica = replicas[0]
        if item.affinity_group:
            self._affinity_replica[(item.domain_key, item.affinity_group)] = replica.replica_id
        return replica

    def _requeue_assigned(self, assigned: _AssignedWork, *, reason: str) -> None:
        item = assigned.item
        updated_extra = dict(item.extra)
        updated_extra["last_requeue_reason"] = str(reason)
        updated = ModelWorkItem(
            **{
                **asdict(item),
                "request_json": item.request_json,
                "extra": updated_extra,
            }
        )
        self._backlog(updated.domain_key).appendleft(updated)
        self._request_locations[updated.request_id] = "backlog"
        self._requeued += 1

    def _remove_request_location(self, request_id: str) -> None:
        self._request_locations.pop(str(request_id), None)
        self._lease_id_by_request_id.pop(str(request_id), None)

    async def _expire_leases_locked(self, *, now: float) -> int:
        expired = 0
        for lease_id, lease in list(self._leases_by_id.items()):
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
            should_requeue = await self._persist_requeue_task(
                lease.item.request_id,
                reason="lease_expired",
            )
            if should_requeue:
                self._requeue_assigned(assigned, reason="lease_expired")
                expired += 1
            else:
                self._remove_request_location(lease.item.request_id)
        return expired

    async def _assign_pending_locked(self, *, max_items: int | None = None) -> dict[str, Any]:
        assigned = 0
        skipped_domains: list[str] = []
        limit = None if max_items is None else max(0, int(max_items))
        for domain_key in sorted(list(self._domain_backlog)):
            backlog = self._domain_backlog.get(domain_key)
            while backlog:
                if limit is not None and assigned >= limit:
                    return {"ok": True, "assigned": assigned, "skipped_domains": skipped_domains}
                item = backlog[0]
                replica = self._choose_replica(item)
                if replica is None:
                    skipped_domains.append(domain_key)
                    break
                if self._use_task_state_store:
                    await self._task_state_call(
                        "assign_task",
                        request_id=item.request_id,
                        subqueue_id=replica.effective_queue_id,
                        scheduler_epoch=int(self._scheduler_epoch or 0),
                    )
                backlog.popleft()
                queue = self._queue(replica.domain_key, replica.replica_id)
                queue.append(
                    _AssignedWork(
                        item=item,
                        replica_id=replica.replica_id,
                        queue_id=replica.effective_queue_id,
                        assigned_at=time.time(),
                        assignment_generation=replica.generation,
                        assignment_reason="least_loaded_affinity",
                    )
                )
                self._request_locations[item.request_id] = "assigned"
                assigned += 1
                self._assigned += 1
            self._drop_empty_backlog(domain_key)
        return {"ok": True, "assigned": assigned, "skipped_domains": skipped_domains}

    async def append(
        self,
        item: dict[str, Any],
        *,
        assign: bool = False,
        assign_max_items: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_assignment_loop_started()
        work = ModelWorkItem.from_dict(item)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            if work.request_id in self._all_request_ids():
                return {
                    "ok": False,
                    "reason": "duplicate_request_id",
                    "request_id": work.request_id,
                }
            if self._use_task_state_store:
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
                    },
                )
                if isinstance(created, dict) and not bool(created.get("created", True)):
                    return {
                        "ok": False,
                        "reason": "duplicate_request_id",
                        "request_id": work.request_id,
                    }
            self._backlog(work.domain_key).append(work)
            self._request_locations[work.request_id] = "backlog"
            self._appended += 1
            assigned = (
                await self._assign_pending_locked(max_items=assign_max_items)
                if bool(assign)
                else {"ok": True, "assigned": 0, "skipped_domains": []}
            )
            self._cv.notify_all()
            return {
                "ok": True,
                "request_id": work.request_id,
                "domain_key": work.domain_key,
                "scheduler_instance_id": self._instance_id,
                "backlog_depth": len(self._backlog(work.domain_key)),
                "assigned": assigned,
            }

    async def cancel_request(self, *, request_id: str, reason: str = "cancelled") -> dict[str, Any]:
        request_id = str(request_id)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        removed = False
        async with self._cv:
            for domain_key, backlog in list(self._domain_backlog.items()):
                kept = deque(item for item in backlog if item.request_id != request_id)
                if len(kept) != len(backlog):
                    removed = True
                    self._domain_backlog[domain_key] = kept
                    self._drop_empty_backlog(domain_key)
            for key, queue in list(self._replica_queues.items()):
                kept = deque(assigned for assigned in queue if assigned.item.request_id != request_id)
                if len(kept) != len(queue):
                    removed = True
                    self._replica_queues[key] = kept
            lease_id = self._lease_id_by_request_id.pop(request_id, None)
            if lease_id is not None:
                removed = self._leases_by_id.pop(lease_id, None) is not None or removed
            if removed:
                self._request_locations.pop(request_id, None)
                self._failed += 1
                self._cv.notify_all()
            return {"ok": True, "request_id": request_id, "cancelled": removed, "reason": str(reason)}

    async def contains_request(self, *, request_id: str) -> dict[str, Any]:
        request_id = str(request_id)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            location = self._request_locations.get(request_id)
            lease_id = self._lease_id_by_request_id.get(request_id)
            return {
                "ok": True,
                "request_id": request_id,
                "present": location is not None,
                "location": location,
                "lease_id": lease_id,
                "scheduler_instance_id": self._instance_id,
            }

    async def is_empty(self) -> bool:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            return (
                not any(self._domain_backlog.values())
                and not any(self._replica_queues.values())
                and not self._leases_by_id
            )

    async def sync_replicas(self, replicas: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_assignment_loop_started()
        now = time.time()
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        incoming = {
            _queue_key(reg.domain_key, reg.replica_id): reg
            for reg in (ModelReplicaRegistration.from_dict(replica) for replica in replicas)
        }
        requeued = 0
        async with self._cv:
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

            for key in changed_unclaimable:
                queue = self._replica_queues.get(key)
                while queue:
                    assigned = queue.pop()
                    should_requeue = await self._persist_requeue_task(
                        assigned.item.request_id,
                        reason="replica_unclaimable",
                    )
                    if should_requeue:
                        self._requeue_assigned(assigned, reason="replica_unclaimable")
                        requeued += 1
                    else:
                        self._remove_request_location(assigned.item.request_id)
                for lease_id, lease in list(self._leases_by_id.items()):
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
                        should_requeue = await self._persist_requeue_task(
                            lease.item.request_id,
                            reason="replica_unclaimable",
                        )
                        if should_requeue:
                            self._requeue_assigned(assigned, reason="replica_unclaimable")
                            requeued += 1
                        else:
                            self._remove_request_location(lease.item.request_id)

            for key in removed:
                self._replica_queues.pop(key, None)
            self._replicas = incoming
            for key in incoming:
                self._replica_queues.setdefault(key, deque())
            expired = await self._expire_leases_locked(now=now)
            assigned_pending = await self._assign_pending_locked()
            self._cv.notify_all()
            return {
                "ok": True,
                "replicas": len(self._replicas),
                "removed": len(removed),
                "requeued": requeued + expired,
                "expired": expired,
                "assigned": assigned_pending,
            }

    async def assign_pending(self, *, max_items: int | None = None) -> dict[str, Any]:
        self._ensure_assignment_loop_started()
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            expired = await self._expire_leases_locked(now=time.time())
            out = await self._assign_pending_locked(max_items=max_items)
            out["expired"] = expired
            self._cv.notify_all()
            return out

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
    ) -> dict[str, Any]:
        self._ensure_assignment_loop_started()
        now = time.time()
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            self._validate_claimer(
                domain_key=domain_key,
                replica_id=replica_id,
                consumer_id=consumer_id,
                consumer_generation=consumer_generation,
            )
            queue = self._queue(domain_key, replica_id)
            claimed: list[dict[str, Any]] = []
            spent = 0
            while queue and len(claimed) < max(1, int(max_items)):
                assigned = queue[0]
                cost = max(1, int(assigned.item.token_cost))
                if token_budget is not None and claimed and spent + cost > int(token_budget):
                    break
                if token_budget is not None and not claimed and cost > int(token_budget):
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
                if self._use_task_state_store:
                    await self._task_state_call(
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
                queue.popleft()
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
                spent += cost
            self._claimed += 1
            return {"ok": True, "leases": claimed, "remaining_queue_depth": len(queue)}

    async def begin_finalize_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
    ) -> dict[str, Any]:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return {"ok": False, "reason": "stale_consumer"}
            now = time.time()
            if self._use_task_state_store:
                await self._task_state_call(
                    "begin_finalize",
                    request_id=lease.item.request_id,
                    lease_id=lease.lease_id,
                    attempt_id=lease.attempt_id,
                    scheduler_epoch=int(self._scheduler_epoch or 0),
                    runtime_generation=int(consumer_generation),
                    finalize_ttl_s=max(1.0, float(finalize_ttl_s)),
                    staged_payload_path=staged_payload_path,
                )
            lease.finalizing_until = now + max(1.0, float(finalize_ttl_s))
            lease.lease_expires_at = max(float(lease.lease_expires_at), lease.finalizing_until)
            return {"ok": True, "lease": lease.to_dict()}

    async def renew_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float = 30.0,
    ) -> dict[str, Any]:
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return {"ok": False, "reason": "stale_consumer"}
            lease.lease_expires_at = time.time() + max(1.0, float(lease_ttl_s))
            return {"ok": True, "lease": lease.to_dict()}

    async def complete_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
    ) -> dict[str, Any]:
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return {"ok": False, "reason": "stale_consumer"}
            self._leases_by_id.pop(lease.lease_id, None)
            self._remove_request_location(lease.item.request_id)
            self._completed += 1
            return {"ok": True, "request_id": lease.item.request_id}

    async def validate_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
    ) -> dict[str, Any]:
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return {"ok": False, "reason": "stale_consumer"}
            return {"ok": True, "request_id": lease.item.request_id}

    async def fail_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        requeue: bool = True,
        reason: str = "failed",
    ) -> dict[str, Any]:
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            lease = self._leases_by_id.get(str(lease_id))
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease.consumer_id != consumer_id or int(lease.consumer_generation) != int(
                consumer_generation
            ):
                return {"ok": False, "reason": "stale_consumer"}
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
                should_requeue = await self._persist_requeue_task(
                    lease.item.request_id,
                    reason=reason,
                )
                if should_requeue:
                    self._requeue_assigned(assigned, reason=reason)
                    requeued_out = True
                else:
                    self._remove_request_location(lease.item.request_id)
            else:
                self._remove_request_location(lease.item.request_id)
                self._failed += 1
            self._cv.notify_all()
            return {"ok": True, "request_id": lease.item.request_id, "requeued": requeued_out}

    async def expire_leases(self, *, now: float | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else float(now)
        if self._use_task_state_store:
            await self._ensure_task_state_ready()
        async with self._cv:
            expired = await self._expire_leases_locked(now=ts)
            if expired:
                self._cv.notify_all()
            return {"ok": True, "expired": expired}

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
            "scheduler_instance_id": self._instance_id,
            "scheduler_epoch": self._scheduler_epoch,
            "task_state_store_enabled": self._use_task_state_store,
            "assignment_loop_interval_s": self._assignment_loop_interval_s,
            "assignment_loop_running": self._assignment_loop_task is not None and not self._assignment_loop_task.done(),
            "now": now,
            "depth": sum(backlog_depth_by_domain.values())
            + sum(len(queue) for queue in self._replica_queues.values())
            + len(self._leases_by_id),
            "backlog_depth": sum(backlog_depth_by_domain.values()),
            "backlog_depth_by_domain": backlog_depth_by_domain,
            "replicas": [replica.to_dict() for _, replica in sorted(self._replicas.items())],
            "replica_queues": replica_queues,
            "leases": [lease.to_dict() for lease in self._leases_by_id.values()],
            "counters": {
                "appended": self._appended,
                "assigned": self._assigned,
                "claimed": self._claimed,
                "completed": self._completed,
                "failed": self._failed,
                "requeued": self._requeued,
            },
        }

    def stats(self) -> dict[str, Any]:
        self._ensure_assignment_loop_started()
        return self._stats_snapshot()

    def ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "actor_name": _ray_model_work_scheduler_actor_name(),
            "namespace": _ray_namespace(),
            "scheduler_instance_id": self._instance_id,
        }


def _await_ray_ref_sync(ref: Any, *, timeout_s: float | None = None) -> Any:
    return sync_get_ray_ref(ref, timeout_s=timeout_s)


def _create_ray_actor(*, require_ready: bool = True):
    import ray

    actor_name = _ray_model_work_scheduler_actor_name()
    max_concurrency = int(os.environ.get("MINT_MODEL_WORK_SCHEDULER_ACTOR_MAX_CONCURRENCY", "256"))
    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars()),
    }
    resources = _model_work_scheduler_actor_resources()
    if resources:
        options["resources"] = resources
    else:
        apply_detached_actor_resources(options, ray)

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=0)
    class _RayModelWorkSchedulerActor(_ModelWorkSchedulerActor):
        pass

    actor = _RayModelWorkSchedulerActor.options(**options).remote(use_task_state_store=True)
    if require_ready:
        out = _await_ray_ref_sync(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
    return actor


class ModelWorkSchedulerClient:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_ray_actor(self) -> None:
        self._ray_actor = None

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
            if not require_ready:
                return self._ray_actor
            try:
                out = await self._await_ray_ref(self._ray_actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"ModelWorkScheduler.ping returned non-dict: {type(out)}")
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()

        actor_name = _ray_model_work_scheduler_actor_name()
        try:
            self._ray_actor = await asyncio.to_thread(
                ray.get_actor,
                actor_name,
                namespace=_ray_namespace(),
            )
            return self._ray_actor
        except ValueError:
            if not create_if_missing:
                raise ModelWorkSchedulerUnavailableError(
                    f"Detached Ray ModelWorkScheduler actor unavailable actor_name={actor_name!r}"
                )
            logger.info("[model_work_scheduler] actor %s not found; creating", actor_name)
        except Exception:
            if not create_if_missing:
                raise ModelWorkSchedulerUnavailableError(
                    f"Detached Ray ModelWorkScheduler actor unavailable actor_name={actor_name!r}"
                )
            logger.info("[model_work_scheduler] failed to fetch actor %s; creating", actor_name)

        try:
            self._ray_actor = _create_ray_actor(require_ready=require_ready)
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
            timeout_s=10.0,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.append returned non-dict: {type(out)}")
        if not bool(out.get("ok")) and out.get("reason") == "duplicate_request_id":
            raise ModelWorkSchedulerConflictError(f"duplicate request_id: {request_id}")
        return out

    async def sync_replicas(
        self,
        replicas: list[ModelReplicaRegistration | dict[str, Any]],
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        payload = [
            replica.to_dict() if isinstance(replica, ModelReplicaRegistration) else dict(replica)
            for replica in replicas
        ]
        out = await self._await_ray_ref(actor.sync_replicas.remote(payload), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.sync_replicas returned non-dict: {type(out)}")
        return out

    async def assign_pending(
        self,
        *,
        max_items: int | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.assign_pending.remote(max_items=max_items),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.assign_pending returned non-dict: {type(out)}")
        return out

    async def cancel_request(
        self,
        *,
        request_id: str,
        reason: str = "cancelled",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.cancel_request.remote(request_id=str(request_id), reason=str(reason)),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.cancel_request returned non-dict: {type(out)}")
        return out

    async def contains_request(
        self,
        *,
        request_id: str,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.contains_request.remote(request_id=str(request_id)),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.contains_request returned non-dict: {type(out)}")
        return out

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
    ) -> dict[str, Any]:
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
        return out

    async def renew_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.renew_lease.remote(
                lease_id=str(lease_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
                lease_ttl_s=float(lease_ttl_s),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.renew_lease returned non-dict: {type(out)}")
        return out

    async def complete_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.complete_lease.remote(
                lease_id=str(lease_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.complete_lease returned non-dict: {type(out)}")
        return out

    async def begin_finalize_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.begin_finalize_lease.remote(
                lease_id=str(lease_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
                finalize_ttl_s=float(finalize_ttl_s),
                staged_payload_path=staged_payload_path,
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.begin_finalize_lease returned non-dict: {type(out)}")
        return out

    async def validate_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.validate_lease.remote(
                lease_id=str(lease_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.validate_lease returned non-dict: {type(out)}")
        return out

    async def fail_lease(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        requeue: bool = True,
        reason: str = "failed",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(
            actor.fail_lease.remote(
                lease_id=str(lease_id),
                consumer_id=str(consumer_id),
                consumer_generation=int(consumer_generation),
                requeue=bool(requeue),
                reason=str(reason),
            ),
            timeout_s=timeout_s,
        )
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.fail_lease returned non-dict: {type(out)}")
        return out

    async def expire_leases(
        self,
        *,
        now: float | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(create_if_missing=False)
        out = await self._await_ray_ref(actor.expire_leases.remote(now=now), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"ModelWorkScheduler.expire_leases returned non-dict: {type(out)}")
        return out

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
        return out

    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self.async_ping(timeout_s=timeout_s)


model_work_scheduler = ModelWorkSchedulerClient()
