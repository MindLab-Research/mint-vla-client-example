from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..config import config as server_config, otel_env_vars, preferred_control_plane_resources
from ..logging_context import (
    classify_failure_reason,
    ensure_trace_id,
    extract_trace_id_from_traceparent,
    get_trace_id,
    get_otel_tracer,
    init_actor_observability,
    log_with_bound_context,
    record_scheduler_decision_otel,
    set_request_id,
    set_trace_id,
)
from ..queue_priority import QUEUE_PRIORITY_AGING_S, effective_queue_priority, normalize_queue_priority
from .work_classification import infer_scheduler_capacity_owner

logger = logging.getLogger(__name__)


def _api_work_queue_debug_log_path() -> str:
    raw = os.environ.get("MINT_API_WORK_QUEUE_DEBUG_LOG_PATH", "").strip()
    if raw:
        return raw
    fallback = os.environ.get("MINT_QUEUE_EXECUTION_RUNTIME_DEBUG_LOG_PATH", "").strip()
    if fallback:
        return fallback
    return "/tmp/tinker_api_work_queue.debug.jsonl"


def _summarize_debug_runtime_env(runtime_env: Any) -> Any:
    if not isinstance(runtime_env, dict):
        return runtime_env
    summary = dict(runtime_env)
    env_vars = summary.get("env_vars")
    if isinstance(env_vars, dict):
        summary["env_var_keys"] = sorted(str(key) for key in env_vars)
        summary.pop("env_vars", None)
    return summary


def _append_api_work_queue_debug(event: str, **fields: Any) -> None:
    record = {
        "ts": round(time.time(), 6),
        "pid": os.getpid(),
        "event": event,
        "actor_name": _ray_api_work_queue_actor_name(),
        "namespace": _ray_namespace(),
        **fields,
    }
    try:
        with open(_api_work_queue_debug_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True, default=str))
            fh.write("\n")
    except Exception:
        logger.debug("api work queue debug log write failed", exc_info=True)


class ApiWorkQueueUnavailableError(RuntimeError):
    pass


class StaleConsumerError(RuntimeError):
    pass


class ApiWorkQueueThrottleError(RuntimeError):
    def __init__(self, *, scope: str, limit: int, pending: int):
        self.detail = {
            "code": "sampling_principal_backpressure",
            "scope": str(scope),
            "limit": int(limit),
            "pending": int(pending),
            "message": "Sampling backpressure: principal budget exhausted",
        }
        super().__init__(self.detail["message"])

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> "ApiWorkQueueThrottleError":
        return cls(
            scope=str(detail.get("scope") or "api_key"),
            limit=int(detail.get("limit") or 0),
            pending=int(detail.get("pending") or 0),
        )


def _unwrap_queue_throttle_error(exc: Exception) -> ApiWorkQueueThrottleError | Exception | None:
    candidate: Exception | None = exc
    as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            candidate = as_instanceof_cause()
        except Exception:
            candidate = exc
    if isinstance(candidate, ApiWorkQueueThrottleError):
        return candidate
    detail = getattr(candidate, "detail", None)
    if isinstance(detail, dict) and detail.get("code") == "sampling_principal_backpressure":
        return candidate
    return None


def _ray_namespace() -> str:
    v = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _ray_api_work_queue_actor_name() -> str:
    env_value = (
        os.environ.get("TINKER_API_WORK_QUEUE_ACTOR_NAME")
        or os.environ.get("MINT_API_WORK_QUEUE_ACTOR_NAME")
    )
    if env_value:
        return str(env_value)
    return str(getattr(server_config, "api_work_queue_actor_name", "tinker_api_work_queue"))


def _api_work_queue_actor_resources() -> dict[str, float] | None:
    pinned_ip = str(os.environ.get("MINT_API_WORK_QUEUE_PINNED_NODE_IP") or "").strip()
    if pinned_ip:
        return {f"node:{pinned_ip}": 0.001}
    try:
        import ray

        return preferred_control_plane_resources(ray.cluster_resources())
    except Exception:
        return None


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    op: str
    request_json: bytes
    user_id: str | None
    apikey_id: str | None
    throttle_principal: str | None
    webhook_url: str | None
    extra: dict[str, Any]
    created_at: float


@dataclass
class _ExecutionSerialState:
    cond: asyncio.Condition
    current_epoch: str | None = None
    next_seq: int = 1
    active_epoch: str | None = None
    active_seq: int | None = None
    pending_seqs_by_epoch: dict[str, set[int]] = field(default_factory=dict)
    refs: int = 0



def _create_ray_actor(*, require_ready: bool = True):
    import ray

    actor_name = _ray_api_work_queue_actor_name()
    max_concurrency = int(os.environ.get("MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY", "256"))
    max_restarts = int(os.environ.get("MINT_API_WORK_QUEUE_MAX_RESTARTS", "3"))

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=max_restarts)
    class _RayApiWorkQueueActor:
        def __init__(self) -> None:
            _append_api_work_queue_debug("actor_init_begin")
            try:
                from collections import deque

                init_actor_observability()
                logger.info(
                    "[api_work_queue] actor (re)initializing (max_restarts=%d)",
                    max_restarts,
                )
                self._items = deque()
                self._cv = asyncio.Condition()
                self._enqueued = 0
                self._dequeued = 0
                debug_max = int(os.environ.get("MINT_API_WORK_QUEUE_DEBUG_MAX", "50"))
                self._recent_dequeues = deque(maxlen=debug_max)
                self._recent_enqueues = deque(maxlen=debug_max)
                self._recent_scheduler_decisions = deque(
                    maxlen=int(os.environ.get("MINT_API_WORK_QUEUE_SCHED_DEBUG_MAX", str(debug_max)))
                )
                self._scheduler_decision_seq = 0
                self._active_job_id: str | None = None
                self._ema_exec_s_by_op: dict[str, float] = {}
                self._last_exec_s_by_op: dict[str, float] = {}
                self._sum_exec_s_by_op: dict[str, float] = {}
                self._count_exec_by_op: dict[str, int] = {}
                self._max_exec_s_by_op: dict[str, float] = {}
                self._ema_alpha = float(os.environ.get("MINT_API_WORK_QUEUE_ETA_ALPHA", "0.1"))
                self._max_pending_asample_per_apikey = int(
                    getattr(server_config, "sampling_max_pending_asample_per_apikey", 64)
                )
                self._queued_asample_by_principal: dict[str, int] = {}
                self._queued_asample_by_apikey: dict[str, int] = {}
                self._queued_asample_request_state: dict[str, tuple[str | None, str | None, str | None]] = {}
                self._scheduler_request_meta: dict[str, tuple[str, str, str]] = {}
                self._scheduler_lease_consumer: dict[str, str] = {}
                self._scheduler_enabled = self._to_bool(os.environ.get("MINT_SCHEDULER_ENABLE", "1"))
                self._scheduler_max_consecutive = max(
                    1,
                    int(os.environ.get("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")),
                )
                fairness = str(os.environ.get("MINT_SCHEDULER_FAIRNESS", "oldest")).strip().lower()
                if fairness not in ("oldest", "rr"):
                    fairness = "oldest"
                self._scheduler_fairness = fairness
                self._scheduler_starvation_s = max(
                    0.0,
                    float(os.environ.get("MINT_SCHEDULER_STARVATION_S", "30")),
                )
                self._scheduler_coalesce_ms = max(
                    0.0,
                    float(os.environ.get("MINT_SCHEDULER_COALESCE_MS", "20")),
                )
                self._sched_domains: dict[str, dict[str, Any]] = {}
                self._sched_stats: dict[str, Any] = {
                    "picks_total": 0,
                    "switches_total": 0,
                    "starvation_picks_total": 0,
                    "wait_s_sum": 0.0,
                    "switch_reasons": {},
                }
                self._scheduler_arbitration_total = 0
                self._scheduler_arbitration_by_winner: dict[str, int] = {}
                self._scheduler_arbitration_by_reason: dict[str, int] = {}
                self._scheduled_dequeue_stats: dict[tuple[str, str, str], int] = {}
                self._legacy_dequeue_stats: dict[tuple[str, str], int] = {}
                self._execution_serial_seq_by_key: dict[str, int] = {}
                self._execution_serial_epoch = uuid.uuid4().hex
                _append_api_work_queue_debug(
                    "actor_init_ok",
                    cwd=os.getcwd(),
                    debug_log_path=_api_work_queue_debug_log_path(),
                    pythonpath=os.environ.get("PYTHONPATH", ""),
                )
            except Exception as e:
                _append_api_work_queue_debug(
                    "actor_init_error",
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(),
                )
                raise

        def _asample_throttle_identity(self, item: dict[str, Any]) -> tuple[str | None, str | None]:
            if str(item.get("op")) != "sampling.asample":
                return None, None
            principal = item.get("throttle_principal")
            apikey_id = item.get("apikey_id")
            principal_str = None if principal is None else str(principal).strip()
            apikey_str = None if apikey_id is None else str(apikey_id).strip()
            if principal_str == "":
                principal_str = None
            if apikey_str == "":
                apikey_str = None
            return principal_str, apikey_str

        def _reserve_asample_slot(self, item: dict[str, Any]) -> None:
            principal, apikey_id = self._asample_throttle_identity(item)
            if principal is None:
                return
            request_id = str(item.get("request_id"))
            if request_id in self._queued_asample_request_state:
                return
            pending = int(self._queued_asample_by_principal.get(principal, 0))
            if pending >= self._max_pending_asample_per_apikey:
                raise ApiWorkQueueThrottleError(
                    scope="api_key" if apikey_id is not None else "user",
                    limit=self._max_pending_asample_per_apikey,
                    pending=pending,
                )
            self._queued_asample_by_principal[principal] = pending + 1
            if apikey_id is not None:
                self._queued_asample_by_apikey[apikey_id] = int(self._queued_asample_by_apikey.get(apikey_id, 0)) + 1
            self._queued_asample_request_state[request_id] = (principal, apikey_id, None)

        def _release_asample_slot(self, request_id: str) -> None:
            principal, apikey_id, _consumer_job_id = self._queued_asample_request_state.pop(
                str(request_id), (None, None, None)
            )
            if principal is not None:
                pending = int(self._queued_asample_by_principal.get(principal, 0)) - 1
                if pending > 0:
                    self._queued_asample_by_principal[principal] = pending
                else:
                    self._queued_asample_by_principal.pop(principal, None)
            if apikey_id is not None:
                pending = int(self._queued_asample_by_apikey.get(apikey_id, 0)) - 1
                if pending > 0:
                    self._queued_asample_by_apikey[apikey_id] = pending
                else:
                    self._queued_asample_by_apikey.pop(apikey_id, None)

        async def finalize_request(self, request_id: str) -> None:
            rid = str(request_id)
            self._release_asample_slot(rid)
            async with self._cv:
                meta = self._scheduler_request_meta.pop(rid, None)
                self._scheduler_lease_consumer.pop(rid, None)
                if meta is None:
                    return
                domain, session_id, op = meta
                state = self._sched_domains.get(domain)
                if state is None:
                    return
                if state.get("leased_request_id") == rid:
                    state["leased_request_id"] = None
                    state["leased_session"] = None
                self._cv.notify_all()

        def _mark_asample_leased_to_consumer(self, request_id: str, consumer_job_id: str | None) -> None:
            request_id = str(request_id)
            state = self._queued_asample_request_state.get(request_id)
            if state is None:
                return
            principal, apikey_id, _previous_consumer = state
            self._queued_asample_request_state[request_id] = (
                principal,
                apikey_id,
                None if consumer_job_id is None else str(consumer_job_id),
            )

        def _release_slots_for_consumer(self, consumer_job_id: str | None) -> int:
            if consumer_job_id is None:
                return 0
            doomed = [
                request_id
                for request_id, (_principal, _apikey_id, leased_consumer) in self._queued_asample_request_state.items()
                if leased_consumer == str(consumer_job_id)
            ]
            for request_id in doomed:
                self._release_asample_slot(request_id)
            return len(doomed)

        def release_scheduler_leases(self, request_ids: list[str]) -> int:
            doomed_ids = {str(request_id) for request_id in request_ids if str(request_id)}
            if not doomed_ids:
                return 0
            released = 0
            for state in self._sched_domains.values():
                leased_request_id = state.get("leased_request_id")
                if leased_request_id in doomed_ids:
                    state["leased_request_id"] = None
                    state["leased_session"] = None
                    released += 1
            for request_id in doomed_ids:
                self._scheduler_lease_consumer.pop(request_id, None)
            return released

        def release_scheduler_leases_for_consumer(self, consumer_job_id: str | None) -> list[str]:
            if consumer_job_id is None:
                return []
            doomed_ids = [
                request_id
                for request_id, leased_consumer in self._scheduler_lease_consumer.items()
                if leased_consumer == str(consumer_job_id)
            ]
            self.release_scheduler_leases(doomed_ids)
            for request_id in doomed_ids:
                self._scheduler_request_meta.pop(str(request_id), None)
            return [str(request_id) for request_id in doomed_ids]

        def release_stale_scheduler_leases(self, active_consumer_job_id: str | None) -> list[str]:
            active = None if active_consumer_job_id is None else str(active_consumer_job_id)
            doomed_ids = [
                request_id
                for request_id, leased_consumer in self._scheduler_lease_consumer.items()
                if leased_consumer and leased_consumer != active
            ]
            self.release_scheduler_leases(doomed_ids)
            for request_id in doomed_ids:
                self._scheduler_request_meta.pop(str(request_id), None)
            return [str(request_id) for request_id in doomed_ids]
        def _to_bool(self, v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if v is None:
                return False
            s = str(v).strip().lower()
            return s in ("1", "true", "yes", "y", "on")

        def _item_created_at(self, item: dict[str, Any], now: float | None = None) -> float:
            fallback = time.time() if now is None else float(now)
            try:
                return float(item.get("created_at", fallback))
            except Exception:
                return fallback

        def _item_raw_priority(self, item: dict[str, Any]) -> int:
            extra = item.get("extra")
            if not isinstance(extra, dict):
                return 0
            return normalize_queue_priority(extra.get("queue_priority", 0))

        def _item_effective_priority(self, item: dict[str, Any], *, now: float) -> int:
            return effective_queue_priority(
                raw_priority=self._item_raw_priority(item),
                created_at=self._item_created_at(item, now=now),
                now=now,
                aging_s=QUEUE_PRIORITY_AGING_S,
            )

        def _annotate_queue_priority(self, item: dict[str, Any], *, now: float, kind: str) -> None:
            extra = item.get("extra")
            if not isinstance(extra, dict):
                extra = {}
                item["extra"] = extra
            extra["_queue_priority_raw"] = self._item_raw_priority(item)
            extra["_queue_priority_effective"] = self._item_effective_priority(item, now=now)
            extra["_queue_kind"] = str(kind)

        def _item_log_context(self, item: dict[str, Any]) -> tuple[str, str]:
            request_id = str(item.get("request_id") or "-")
            trace_id: str | None = None
            extra = item.get("extra")
            if isinstance(extra, dict):
                raw_trace_id = extra.get("_trace_id")
                if isinstance(raw_trace_id, str) and raw_trace_id.strip():
                    trace_id = raw_trace_id.strip()
                if trace_id is None:
                    raw_traceparent = extra.get("_traceparent")
                    if isinstance(raw_traceparent, str) and raw_traceparent.strip():
                        trace_id = extract_trace_id_from_traceparent(raw_traceparent)
            if not isinstance(trace_id, str) or not trace_id:
                trace_id = "-"
            return request_id, trace_id

        def _scheduler_item_info(self, item: dict[str, Any]) -> tuple[bool, str, str]:
            extra = item.get("extra")
            if not isinstance(extra, dict):
                return False, "", ""
            enabled = bool(self._scheduler_enabled)
            if "scheduler_enabled" in extra:
                enabled = self._to_bool(extra.get("scheduler_enabled"))
            if not enabled:
                return False, "", ""
            domain = str(extra.get("scheduler_domain") or "").strip()
            # Preferred key is scheduler_session_key; keep legacy session_id fallback
            # so in-flight enqueued items (or older clients) remain schedulable.
            session_id = str(extra.get("scheduler_session_key") or extra.get("session_id") or "").strip()
            if not domain or not session_id:
                return False, "", ""
            return True, domain, session_id

        def _scheduler_domain_policy(self, item: dict[str, Any]) -> tuple[str | None, int | None]:
            extra = item.get("extra")
            if not isinstance(extra, dict):
                return None, None
            fairness_raw = extra.get("scheduler_fairness")
            fairness: str | None = None
            if fairness_raw is not None:
                fairness = str(fairness_raw).strip().lower()
                if fairness not in ("oldest", "rr"):
                    raise ValueError(f"invalid scheduler_fairness override: {fairness_raw!r}")
            max_consecutive_raw = extra.get("scheduler_max_consecutive")
            max_consecutive: int | None = None
            if max_consecutive_raw is not None:
                max_consecutive = int(max_consecutive_raw)
                if max_consecutive < 1:
                    raise ValueError(
                        f"scheduler_max_consecutive override must be >= 1, got {max_consecutive_raw!r}"
                    )
            return fairness, max_consecutive

        def _get_domain_state(self, domain: str) -> dict[str, Any]:
            from collections import deque

            state = self._sched_domains.get(domain)
            if state is not None:
                return state
            state = {
                "queues_by_session": {},
                "ready_rr": deque(),
                "ready_set": set(),
                "current_session": None,
                "last_session": None,
                "last_pick_ts": 0.0,
                "consecutive_count": 0,
                "scheduler_fairness_override": None,
                "scheduler_max_consecutive_override": None,
                "capacity_owner": None,
                "leased_request_id": None,
                "leased_session": None,
                "stats": {
                    "picks": 0,
                    "switches": 0,
                    "starvation_picks": 0,
                    "wait_s_sum": 0.0,
                },
            }
            self._sched_domains[domain] = state
            return state

        def _compact_domain_ready(self, state: dict[str, Any]) -> None:
            from collections import deque

            queues_by_session = state["queues_by_session"]
            new_rr = deque()
            new_set = set()
            for sid in list(state["ready_rr"]):
                q = queues_by_session.get(sid)
                if q:
                    if sid not in new_set:
                        new_rr.append(sid)
                        new_set.add(sid)
                else:
                    queues_by_session.pop(sid, None)
            state["ready_rr"] = new_rr
            state["ready_set"] = new_set
            current = state.get("current_session")
            if current is not None and not queues_by_session.get(current):
                raise RuntimeError(
                    "scheduler invariant violated: "
                    f"current_session={current!r} has no queue during compaction"
                )

        def _remove_session_from_ready(self, state: dict[str, Any], session_id: str) -> None:
            from collections import deque

            state["ready_set"].discard(session_id)
            state["ready_rr"] = deque([sid for sid in state["ready_rr"] if sid != session_id])

        def _scheduled_depth(self) -> int:
            depth = 0
            for state in self._sched_domains.values():
                for q in state.get("queues_by_session", {}).values():
                    depth += len(q)
            return int(depth)

        def _oldest_scheduled_created_at(self) -> float | None:
            oldest: float | None = None
            for state in self._sched_domains.values():
                for q in state.get("queues_by_session", {}).values():
                    if not q:
                        continue
                    ts = self._item_created_at(q[0])
                    if oldest is None or ts < oldest:
                        oldest = ts
            return oldest

        def _domain_capacity_workers(self, domain: str, state: dict[str, Any]) -> int | None:
            owner = state.get("capacity_owner")
            if owner is None:
                owner = infer_scheduler_capacity_owner(domain)
            owner = None if owner is None else str(owner).strip()
            if owner in {"single_worker", "vllm_replica_single_worker"}:
                return 1
            return None

        def _scheduler_domain_snapshot(self, *, domain: str, state: dict[str, Any], now: float) -> dict[str, Any] | None:
            queues_by_session = state.get("queues_by_session", {}) or {}
            pending_requests = 0
            active_sessions = 0
            oldest_created_at: float | None = None
            for q in queues_by_session.values():
                if not q:
                    continue
                active_sessions += 1
                pending_requests += int(len(q))
                ts = self._item_created_at(q[0], now=now)
                if oldest_created_at is None or ts < oldest_created_at:
                    oldest_created_at = ts
            if pending_requests <= 0 and int((state.get("stats") or {}).get("picks", 0)) == 0:
                return None
            inflight_workers = 1 if state.get("leased_request_id") else 0
            capacity_workers = self._domain_capacity_workers(domain, state)
            last_pick_ts = float(state.get("last_pick_ts", 0.0) or 0.0)
            oldest_queued_s = 0.0 if oldest_created_at is None else max(0.0, now - oldest_created_at)
            service_gap_s = oldest_queued_s if last_pick_ts <= 0 else max(0.0, now - last_pick_ts)
            return {
                "backend": self._scheduler_backend(domain),
                "pending_requests": int(pending_requests),
                "active_sessions": int(active_sessions),
                "oldest_queued_s": float(oldest_queued_s),
                "inflight_workers": int(inflight_workers),
                "capacity_owner": state.get("capacity_owner"),
                "capacity_workers": capacity_workers,
                "admissible": False if capacity_workers is None else bool(inflight_workers < capacity_workers),
                "service_gap_s": float(service_gap_s),
                "stats": {
                    "picks": int((state.get("stats") or {}).get("picks", 0)),
                    "starvation_picks": int((state.get("stats") or {}).get("starvation_picks", 0)),
                },
            }

        def _scheduler_metrics_snapshot(self, *, now: float | None = None) -> dict[str, Any]:
            ts = time.time() if now is None else float(now)
            scheduler_domains: dict[str, Any] = {}
            for domain, state in self._sched_domains.items():
                snapshot = self._scheduler_domain_snapshot(domain=str(domain), state=state, now=ts)
                if snapshot is not None:
                    scheduler_domains[str(domain)] = snapshot
            scheduled_dequeue_stats = [
                {
                    "scheduler_domain": scheduler_domain,
                    "reason": reason,
                    "op": op,
                    "total": int(total),
                }
                for (scheduler_domain, reason, op), total in sorted(self._scheduled_dequeue_stats.items())
            ]
            legacy_dequeue_stats = [
                {
                    "reason": reason,
                    "op": op,
                    "total": int(total),
                }
                for (reason, op), total in sorted(self._legacy_dequeue_stats.items())
            ]
            return {
                "scheduler_arbitration_total": int(self._scheduler_arbitration_total),
                "scheduler_arbitration_by_winner": dict(sorted(self._scheduler_arbitration_by_winner.items())),
                "scheduler_arbitration_by_reason": dict(sorted(self._scheduler_arbitration_by_reason.items())),
                "scheduled_dequeue_stats": scheduled_dequeue_stats,
                "legacy_dequeue_stats": legacy_dequeue_stats,
                "scheduler_domains": scheduler_domains,
            }

        def _record_scheduler_arbitration(self, *, winner_bucket: str, reason: str) -> None:
            bucket = str(winner_bucket).strip() or "legacy"
            why = str(reason).strip() or "unknown"
            self._scheduler_arbitration_total += 1
            self._scheduler_arbitration_by_winner[bucket] = int(self._scheduler_arbitration_by_winner.get(bucket, 0)) + 1
            self._scheduler_arbitration_by_reason[why] = int(self._scheduler_arbitration_by_reason.get(why, 0)) + 1

        def _record_dequeue_stat(self, *, scheduler_domain: str | None, reason: str, op: str) -> None:
            op_key = str(op).strip() or "unknown"
            reason_key = str(reason).strip() or "unknown"
            if scheduler_domain is None:
                key = (reason_key, op_key)
                self._legacy_dequeue_stats[key] = int(self._legacy_dequeue_stats.get(key, 0)) + 1
                return
            key = (str(scheduler_domain), reason_key, op_key)
            self._scheduled_dequeue_stats[key] = int(self._scheduled_dequeue_stats.get(key, 0)) + 1

        def _enqueue_scheduled(self, item: dict[str, Any], *, domain: str, session_id: str) -> None:
            from collections import deque

            state = self._get_domain_state(domain)
            fairness, max_consecutive = self._scheduler_domain_policy(item)
            current_fairness = state.get("scheduler_fairness_override")
            if fairness is not None:
                if current_fairness is None:
                    state["scheduler_fairness_override"] = fairness
                elif current_fairness != fairness:
                    raise RuntimeError(
                        "scheduler fairness override conflict: "
                        f"domain={domain!r} existing={current_fairness!r} incoming={fairness!r}"
                    )
            current_max_consecutive = state.get("scheduler_max_consecutive_override")
            if max_consecutive is not None:
                if current_max_consecutive is None:
                    state["scheduler_max_consecutive_override"] = max_consecutive
                elif int(current_max_consecutive) != int(max_consecutive):
                    raise RuntimeError(
                        "scheduler max_consecutive override conflict: "
                        f"domain={domain!r} existing={current_max_consecutive!r} incoming={max_consecutive!r}"
                    )
            extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
            owner = extra.get("scheduler_capacity_owner") if isinstance(extra, dict) else None
            if owner is None:
                owner = infer_scheduler_capacity_owner(domain)
            if owner is not None:
                current_owner = state.get("capacity_owner")
                owner = str(owner)
                if current_owner is None:
                    state["capacity_owner"] = owner
                elif str(current_owner) != owner:
                    raise RuntimeError(
                        "scheduler capacity_owner conflict: "
                        f"domain={domain!r} existing={current_owner!r} incoming={owner!r}"
                    )
            queues_by_session = state["queues_by_session"]
            q = queues_by_session.get(session_id)
            if q is None:
                q = deque()
                queues_by_session[session_id] = q
            was_empty = len(q) == 0
            q.append(item)
            if was_empty and session_id not in state["ready_set"]:
                state["ready_rr"].append(session_id)
                state["ready_set"].add(session_id)

        def _pick_round_robin_session(self, state: dict[str, Any], *, avoid: str | None = None) -> str | None:
            self._compact_domain_ready(state)
            rr = state["ready_rr"]
            queues_by_session = state["queues_by_session"]
            if not rr:
                return None

            fallback: str | None = None
            n = len(rr)
            for _ in range(n):
                sid = rr.popleft()
                q = queues_by_session.get(sid)
                if not q:
                    state["ready_set"].discard(sid)
                    queues_by_session.pop(sid, None)
                    continue
                rr.append(sid)
                if fallback is None:
                    fallback = sid
                if avoid is not None and sid == avoid:
                    continue
                return sid
            return fallback

        def _pick_oldest_session(self, state: dict[str, Any], *, now: float, avoid: str | None = None) -> str | None:
            self._compact_domain_ready(state)
            queues_by_session = state["queues_by_session"]
            rr = [sid for sid in state["ready_rr"] if queues_by_session.get(sid)]
            if not rr:
                return None

            def _select(candidates: list[str]) -> str | None:
                best_sid: str | None = None
                best_created_at: float | None = None
                for sid in candidates:
                    q = queues_by_session.get(sid)
                    if not q:
                        continue
                    created_at = self._item_created_at(q[0], now=now)
                    if best_created_at is None or created_at < best_created_at:
                        best_created_at = created_at
                        best_sid = sid
                return best_sid

            if avoid is not None and len(rr) > 1:
                sid = _select([sid for sid in rr if sid != avoid])
                if sid is not None:
                    return sid
            return _select(rr)

        def _choose_session_for_domain(self, domain: str, state: dict[str, Any], *, now: float) -> tuple[str, str] | None:
            self._compact_domain_ready(state)
            queues_by_session = state["queues_by_session"]
            rr_order = [sid for sid in state["ready_rr"] if queues_by_session.get(sid)]
            if not rr_order:
                return None

            fairness = str(state.get("scheduler_fairness_override") or self._scheduler_fairness)
            max_consecutive = int(
                state.get("scheduler_max_consecutive_override") or self._scheduler_max_consecutive
            )

            if self._scheduler_starvation_s > 0:
                chosen_starved_sid: str | None = None
                max_wait = self._scheduler_starvation_s
                for sid in rr_order:
                    q = queues_by_session.get(sid)
                    if not q:
                        continue
                    wait_s = max(0.0, now - self._item_created_at(q[0], now=now))
                    if wait_s >= self._scheduler_starvation_s and wait_s >= max_wait:
                        chosen_starved_sid = sid
                        max_wait = wait_s
                if chosen_starved_sid is not None:
                    return chosen_starved_sid, "starvation"

            current = state.get("current_session")
            if (
                isinstance(current, str)
                and current
                and queues_by_session.get(current)
                and int(state.get("consecutive_count", 0)) < int(max_consecutive)
            ):
                return current, "sticky"

            avoid = current if isinstance(current, str) and current and len(rr_order) > 1 else None
            if fairness == "oldest":
                sid = self._pick_oldest_session(state, now=now, avoid=avoid)
                reason = "fairness_oldest"
            else:
                sid = self._pick_round_robin_session(state, avoid=avoid)
                reason = "fairness_rr"
            if sid is None:
                return None
            return sid, reason

        def _pick_scheduled_candidate(self, *, now: float) -> tuple[str, str, str] | None:
            best: tuple[int, float, str, str, str] | None = None
            for domain, state in self._sched_domains.items():
                if state.get("leased_request_id") is not None:
                    continue
                chosen = self._choose_session_for_domain(domain, state, now=now)
                if chosen is None:
                    continue
                sid, reason = chosen
                q = state["queues_by_session"].get(sid)
                if not q:
                    continue
                head = q[0]
                priority = self._item_effective_priority(head, now=now)
                created_at = self._item_created_at(head, now=now)
                candidate = (-priority, created_at, domain, sid, reason)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                return None
            return best[2], best[3], best[4]

        def _record_switch_reason(self, reason: str) -> None:
            reasons = self._sched_stats.get("switch_reasons")
            if not isinstance(reasons, dict):
                reasons = {}
                self._sched_stats["switch_reasons"] = reasons
            reasons[reason] = int(reasons.get(reason, 0)) + 1

        def _scheduler_backend(self, scheduler_domain: str | None) -> str:
            domain = str(scheduler_domain or "").strip()
            if not domain or ":" not in domain:
                return "legacy"
            return domain.split(":", 1)[0].strip() or "legacy"

        def _scheduler_decision_context(
            self,
            *,
            domain: str,
            session_id: str,
            reason: str,
            coalesce_applied: bool,
        ) -> dict[str, Any]:
            state = self._sched_domains.get(domain)
            if state is None:
                return {
                    "previous_session_id": None,
                    "ready_sessions": 0,
                    "chosen_queue_depth": 0,
                    "switch_happened": False,
                    "starvation_triggered": str(reason) == "starvation",
                    "coalesce_applied": bool(coalesce_applied),
                }
            queues_by_session = state.get("queues_by_session", {}) or {}
            chosen_queue_depth = len(queues_by_session.get(session_id) or [])
            previous_session_id = state.get("current_session")
            previous = str(previous_session_id) if isinstance(previous_session_id, str) and previous_session_id else None
            return {
                "previous_session_id": previous,
                "ready_sessions": int(len(queues_by_session)),
                "chosen_queue_depth": int(chosen_queue_depth),
                "switch_happened": previous is not None and previous != str(session_id),
                "starvation_triggered": str(reason) == "starvation",
                "coalesce_applied": bool(coalesce_applied),
            }

        def _append_scheduler_decision(
            self,
            *,
            ts: float,
            item: dict[str, Any],
            scheduler_domain: str,
            scheduler_session_id: str,
            dequeue_reason: str,
            wait_s: float,
            decision_ctx: dict[str, Any],
        ) -> None:
            self._scheduler_decision_seq += 1
            self._recent_scheduler_decisions.append(
                {
                    "seq": int(self._scheduler_decision_seq),
                    "ts": float(ts),
                    "request_id": str(item.get("request_id") or ""),
                    "op": str(item.get("op") or "unknown"),
                    "scheduler_domain": str(scheduler_domain),
                    "decision_reason": str(dequeue_reason),
                    "chosen_session_id": str(scheduler_session_id),
                    "previous_session_id": decision_ctx.get("previous_session_id"),
                    "wait_s": float(wait_s),
                    "ready_sessions": int(decision_ctx.get("ready_sessions") or 0),
                    "chosen_queue_depth": int(decision_ctx.get("chosen_queue_depth") or 0),
                    "switch_happened": bool(decision_ctx.get("switch_happened")),
                    "starvation_triggered": bool(decision_ctx.get("starvation_triggered")),
                    "coalesce_applied": bool(decision_ctx.get("coalesce_applied")),
                }
            )

        def _record_scheduler_metrics(
            self,
            *,
            item: dict[str, Any],
            scheduler_domain: str | None,
            dequeue_reason: str,
            wait_s: float,
            decision_ctx: dict[str, Any] | None,
        ) -> None:
            record_scheduler_decision_otel(
                op=str(item.get("op") or "unknown"),
                backend=self._scheduler_backend(scheduler_domain),
                queue_kind="scheduled" if scheduler_domain is not None else "legacy",
                reason=str(dequeue_reason or "fifo"),
                queue_wait_s=max(0.0, float(wait_s)),
                switched=bool((decision_ctx or {}).get("switch_happened")),
                ready_sessions=(decision_ctx or {}).get("ready_sessions"),
                chosen_queue_depth=(decision_ctx or {}).get("chosen_queue_depth"),
            )

        def _pop_scheduled(self, *, domain: str, session_id: str, reason: str, now: float) -> dict[str, Any]:
            state = self._sched_domains.get(domain)
            if state is None:
                raise KeyError(f"scheduler domain missing during pop: {domain!r}")
            queues_by_session = state["queues_by_session"]
            q = queues_by_session.get(session_id)
            if not q:
                raise KeyError(f"scheduler session queue empty during pop: domain={domain!r} session={session_id!r}")

            previous_current = state.get("current_session")
            item = q.popleft()
            if not q:
                queues_by_session.pop(session_id, None)
                self._remove_session_from_ready(state, session_id)

            next_consecutive = 1
            if previous_current == session_id:
                next_consecutive = int(state.get("consecutive_count", 0)) + 1
            else:
                if previous_current is not None:
                    state["stats"]["switches"] = int(state["stats"].get("switches", 0)) + 1
                    self._sched_stats["switches_total"] = int(self._sched_stats.get("switches_total", 0)) + 1
                    self._record_switch_reason(reason)
                next_consecutive = 1

            wait_s = max(0.0, now - self._item_created_at(item, now=now))
            state["stats"]["wait_s_sum"] = float(state["stats"].get("wait_s_sum", 0.0)) + float(wait_s)
            state["stats"]["picks"] = int(state["stats"].get("picks", 0)) + 1
            self._sched_stats["wait_s_sum"] = float(self._sched_stats.get("wait_s_sum", 0.0)) + float(wait_s)
            self._sched_stats["picks_total"] = int(self._sched_stats.get("picks_total", 0)) + 1
            if reason == "starvation":
                state["stats"]["starvation_picks"] = int(state["stats"].get("starvation_picks", 0)) + 1
                self._sched_stats["starvation_picks_total"] = int(self._sched_stats.get("starvation_picks_total", 0)) + 1

            # Invariant: current_session must always reference a non-empty queue.
            if queues_by_session.get(session_id):
                state["current_session"] = session_id
                state["consecutive_count"] = int(next_consecutive)
            else:
                state["current_session"] = None
                state["consecutive_count"] = 0
            state["last_session"] = session_id
            state["last_pick_ts"] = float(now)
            leased_request_id = str(item.get("request_id") or "")
            state["leased_request_id"] = leased_request_id
            state["leased_session"] = session_id

            return item

        def _scheduler_debug(self) -> dict[str, Any]:
            domains: dict[str, Any] = {}
            for domain, state in self._sched_domains.items():
                queues_by_session = state.get("queues_by_session", {})
                queue_depths = {
                    str(sid): int(len(q))
                    for sid, q in queues_by_session.items()
                    if len(q) > 0
                }
                domain_stats = state.get("stats", {})
                if not queue_depths and int(domain_stats.get("picks", 0)) == 0:
                    continue
                domains[str(domain)] = {
                    "current_session": state.get("current_session"),
                    "consecutive_count": int(state.get("consecutive_count", 0)),
                    "scheduler_fairness": str(
                        state.get("scheduler_fairness_override") or self._scheduler_fairness
                    ),
                    "scheduler_max_consecutive": int(
                        state.get("scheduler_max_consecutive_override") or self._scheduler_max_consecutive
                    ),
                    "queue_depths": queue_depths,
                    "ready_rr": [str(x) for x in list(state.get("ready_rr", []))],
                    "stats": {
                        "picks": int(domain_stats.get("picks", 0)),
                        "switches": int(domain_stats.get("switches", 0)),
                        "starvation_picks": int(domain_stats.get("starvation_picks", 0)),
                        "wait_s_sum": float(domain_stats.get("wait_s_sum", 0.0)),
                    },
                }
            return {
                "enabled": bool(self._scheduler_enabled),
                "max_consecutive": int(self._scheduler_max_consecutive),
                "fairness": str(self._scheduler_fairness),
                "starvation_s": float(self._scheduler_starvation_s),
                "coalesce_ms": float(self._scheduler_coalesce_ms),
                "domains": domains,
                "stats": {
                    "picks_total": int(self._sched_stats.get("picks_total", 0)),
                    "switches_total": int(self._sched_stats.get("switches_total", 0)),
                    "starvation_picks_total": int(self._sched_stats.get("starvation_picks_total", 0)),
                    "wait_s_sum": float(self._sched_stats.get("wait_s_sum", 0.0)),
                    "switch_reasons": dict(self._sched_stats.get("switch_reasons", {})),
                },
            }

        def set_active_job_id(self, job_id: str) -> None:
            next_job_id = None if not job_id else str(job_id)
            previous_job_id = self._active_job_id
            if previous_job_id is not None and previous_job_id != next_job_id:
                self._release_slots_for_consumer(previous_job_id)
            self._active_job_id = next_job_id

        def clear_active_job_id_if_matches(self, job_id: str) -> bool:
            expected = None if not job_id else str(job_id)
            if self._active_job_id != expected:
                return False
            self._release_slots_for_consumer(expected)
            self._active_job_id = None
            return True

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        async def enqueue(self, item: dict[str, Any], producer_job_id: str | None = None) -> dict[str, Any]:
            async with self._cv:
                packed = dict(item)
                try:
                    self._reserve_asample_slot(packed)
                except ApiWorkQueueThrottleError as e:
                    return {"ok": False, "detail": dict(e.detail)}
                request_id, trace_id = self._item_log_context(packed)

                # Log enqueue with request_id context
                log_with_bound_context(
                    logger,
                    request_id=request_id,
                    trace_id=trace_id,
                    message=(
                        f"[Queue] enqueue: request_id={request_id}, op={packed.get('op')}, "
                        f"user_id={packed.get('user_id')} apikey_id={packed.get('apikey_id')}"
                    ),
                )

                is_sched, domain, session_id = self._scheduler_item_info(packed)
                if is_sched:
                    self._enqueue_scheduled(packed, domain=domain, session_id=session_id)
                    self._scheduler_request_meta[str(packed.get("request_id"))] = (
                        str(domain),
                        str(session_id),
                        str(packed.get("op")),
                    )
                else:
                    self._items.append(packed)
                self._enqueued += 1
                try:
                    import ray

                    ctx = ray.get_runtime_context()
                    extra = packed.get("extra")
                    scheduler_domain = None
                    scheduler_session_id = None
                    scheduler_enabled = None
                    if isinstance(extra, dict):
                        scheduler_domain = extra.get("scheduler_domain")
                        scheduler_session_id = extra.get("scheduler_session_key")
                        if scheduler_session_id is None:
                            scheduler_session_id = extra.get("session_id")
                        scheduler_enabled = self._to_bool(extra.get("scheduler_enabled"))
                    self._recent_enqueues.append(
                        {
                            "ts": time.time(),
                            "job_id": None if producer_job_id is None else str(producer_job_id),
                            "task_id": str(ctx.get_task_id()),
                            "request_id": str(packed.get("request_id")),
                            "op": str(packed.get("op")),
                            "apikey_id": None if packed.get("apikey_id") is None else str(packed.get("apikey_id")),
                            "throttle_principal": (
                                None
                                if packed.get("throttle_principal") is None
                                else str(packed.get("throttle_principal"))
                            ),
                            "scheduler_domain": None if scheduler_domain is None else str(scheduler_domain),
                            "scheduler_session_id": None if scheduler_session_id is None else str(scheduler_session_id),
                            "scheduler_enabled": scheduler_enabled,
                            "scheduler_accepted": bool(is_sched),
                        }
                    )
                except Exception:
                    pass
                self._cv.notify(1)
                return {"ok": True}

        async def dequeue(self, consumer_job_id: str) -> dict[str, Any]:

            dequeue_poll_s = max(
                0.05,
                float(os.environ.get("MINT_API_WORK_QUEUE_DEQUEUE_POLL_S", "1.0")),
            )

            async with self._cv:
                coalesce_applied = False
                decision_ctx: dict[str, Any] | None = None
                while True:
                    if self._active_job_id is not None and str(consumer_job_id) != self._active_job_id:
                        return {
                            "request_id": "",
                            "op": "__stale_consumer__",
                            "request_json": b"",
                            "user_id": None,
                            "apikey_id": None,
                            "throttle_principal": None,
                            "webhook_url": None,
                            "extra": {
                                "consumer_job_id": str(consumer_job_id),
                                "active_job_id": self._active_job_id,
                            },
                            "created_at": time.time(),
                        }

                    has_legacy = bool(self._items)
                    now = time.time()
                    sched_choice = self._pick_scheduled_candidate(now=now)

                    if not has_legacy and sched_choice is None:
                        try:
                            await asyncio.wait_for(self._cv.wait(), timeout=dequeue_poll_s)
                        except asyncio.TimeoutError:
                            pass
                        continue

                    # Short coalescing wait: if we are about to switch sessions in a
                    # scheduled domain right after the previous pick, give the current
                    # session a tiny window to enqueue the next chunk.
                    if (
                        not has_legacy
                        and sched_choice is not None
                        and self._scheduler_coalesce_ms > 0.0
                    ):
                        sched_domain, sched_session_id, sched_reason = sched_choice
                        if str(sched_reason).startswith("fairness"):
                            state = self._sched_domains.get(sched_domain)
                            if isinstance(state, dict):
                                queues = state.get("queues_by_session", {}) or {}
                                current = state.get("current_session")
                                if isinstance(current, str) and current and not queues.get(current):
                                    # current_session should always point to a non-empty queue.
                                    raise RuntimeError(
                                        "scheduler invariant violated: "
                                        f"current_session={current!r} has no queue in domain={sched_domain!r}"
                                    )
                                last_session = state.get("last_session")
                                if (
                                    isinstance(last_session, str)
                                    and last_session
                                    and last_session != sched_session_id
                                    and not queues.get(last_session)
                                ):
                                    last_pick_ts = float(state.get("last_pick_ts", 0.0) or 0.0)
                                    coalesce_window_s = float(self._scheduler_coalesce_ms) / 1000.0
                                    remaining_s = coalesce_window_s - max(0.0, now - last_pick_ts)
                                    if remaining_s > 0.0:
                                        try:
                                            await asyncio.wait_for(self._cv.wait(), timeout=remaining_s)
                                        except asyncio.TimeoutError:
                                            pass
                                        coalesce_applied = True
                                        continue

                    item: dict[str, Any]
                    dequeue_reason = "fifo"
                    scheduler_domain = None
                    scheduler_session_id = None
                    decision_ctx = None
                    arbitration_winner = "legacy"
                    arbitration_reason = "legacy_only"

                    if has_legacy and sched_choice is not None:
                        legacy_head = self._items[0]
                        legacy_priority = self._item_effective_priority(legacy_head, now=now)
                        legacy_ts = self._item_created_at(legacy_head, now=now)
                        sched_domain, sched_session_id, sched_reason = sched_choice
                        sched_state = self._sched_domains.get(sched_domain)
                        sched_queue = (
                            None
                            if sched_state is None
                            else (sched_state.get("queues_by_session", {}) or {}).get(sched_session_id)
                        )
                        if not sched_queue:
                            item = self._items.popleft()
                            arbitration_winner = "legacy"
                            arbitration_reason = "legacy_only"
                        else:
                            sched_head = sched_queue[0]
                            sched_priority = self._item_effective_priority(sched_head, now=now)
                            sched_head_ts = self._item_created_at(sched_head, now=now)
                            if (legacy_priority, -legacy_ts) >= (sched_priority, -sched_head_ts):
                                item = self._items.popleft()
                                arbitration_winner = "legacy"
                                arbitration_reason = "legacy_head_older"
                            else:
                                decision_ctx = self._scheduler_decision_context(
                                    domain=str(sched_domain),
                                    session_id=str(sched_session_id),
                                    reason=str(sched_reason),
                                    coalesce_applied=coalesce_applied,
                                )
                                item = self._pop_scheduled(
                                    domain=sched_domain,
                                    session_id=sched_session_id,
                                    reason=sched_reason,
                                    now=now,
                                )
                                dequeue_reason = str(sched_reason)
                                scheduler_domain = str(sched_domain)
                                scheduler_session_id = str(sched_session_id)
                                arbitration_winner = "scheduled"
                                arbitration_reason = f"scheduled_{sched_reason}"
                    elif has_legacy:
                        arbitration_winner = "legacy"
                        arbitration_reason = "legacy_only"
                        best_idx = 0
                        best_priority = self._item_effective_priority(self._items[0], now=now)
                        best_ts = self._item_created_at(self._items[0], now=now)
                        for idx, candidate in enumerate(list(self._items)[1:], start=1):
                            candidate_priority = self._item_effective_priority(candidate, now=now)
                            candidate_ts = self._item_created_at(candidate, now=now)
                            if (candidate_priority, -candidate_ts) > (best_priority, -best_ts):
                                best_idx = idx
                                best_priority = candidate_priority
                                best_ts = candidate_ts
                        if best_idx == 0:
                            item = self._items.popleft()
                        else:
                            item = self._items[best_idx]
                            del self._items[best_idx]
                    else:
                        if sched_choice is None:
                            await self._cv.wait()
                            continue
                        sched_domain, sched_session_id, sched_reason = sched_choice
                        decision_ctx = self._scheduler_decision_context(
                            domain=str(sched_domain),
                            session_id=str(sched_session_id),
                            reason=str(sched_reason),
                            coalesce_applied=coalesce_applied,
                        )
                        item = self._pop_scheduled(
                            domain=sched_domain,
                            session_id=sched_session_id,
                            reason=sched_reason,
                            now=now,
                        )
                        dequeue_reason = str(sched_reason)
                        scheduler_domain = str(sched_domain)
                        scheduler_session_id = str(sched_session_id)
                        arbitration_winner = "scheduled"
                        arbitration_reason = f"scheduled_{sched_reason}"
                    serial_key = self._execution_serial_key(item)
                    if serial_key is not None:
                        extra = item.get("extra")
                        if not isinstance(extra, dict):
                            extra = {}
                            item["extra"] = extra
                        next_seq = int(self._execution_serial_seq_by_key.get(serial_key, 0)) + 1
                        self._execution_serial_seq_by_key[serial_key] = next_seq
                        extra["execution_serial_epoch"] = self._execution_serial_epoch
                        extra["execution_serial_seq"] = next_seq
                    self._mark_asample_leased_to_consumer(item.get("request_id", ""), consumer_job_id)
                    if scheduler_domain is not None:
                        leased_request_id = str(item.get("request_id") or "")
                        if leased_request_id:
                            self._scheduler_lease_consumer[leased_request_id] = str(consumer_job_id)
                    break

                self._dequeued += 1
                dequeue_ts = time.time()
                wait_s = max(0.0, dequeue_ts - self._item_created_at(item, now=dequeue_ts))
                self._record_scheduler_arbitration(
                    winner_bucket=arbitration_winner,
                    reason=arbitration_reason,
                )
                self._record_dequeue_stat(
                    scheduler_domain=scheduler_domain,
                    reason=str(dequeue_reason),
                    op=str(item.get("op") or "unknown"),
                )
                self._annotate_queue_priority(
                    item,
                    now=dequeue_ts,
                    kind="scheduled" if scheduler_domain is not None else "legacy",
                )
                self._record_scheduler_metrics(
                    item=item,
                    scheduler_domain=scheduler_domain,
                    dequeue_reason=str(dequeue_reason),
                    wait_s=wait_s,
                    decision_ctx=decision_ctx,
                )
                if scheduler_domain is not None and scheduler_session_id is not None:
                    self._append_scheduler_decision(
                        ts=dequeue_ts,
                        item=item,
                        scheduler_domain=str(scheduler_domain),
                        scheduler_session_id=str(scheduler_session_id),
                        dequeue_reason=str(dequeue_reason),
                        wait_s=wait_s,
                        decision_ctx=decision_ctx or {},
                    )

                # Log dequeue with request_id context
                request_id, trace_id = self._item_log_context(item)
                log_with_bound_context(
                    logger,
                    request_id=request_id,
                    trace_id=trace_id,
                    message=(
                        f"[Queue] dequeue: request_id={request_id}, op={item.get('op')}, "
                        f"apikey_id={item.get('apikey_id')} reason={dequeue_reason}, scheduler_domain={scheduler_domain}, "
                        f"scheduler_session_id={scheduler_session_id}"
                    ),
                )

                try:
                    import ray

                    ctx = ray.get_runtime_context()
                    self._recent_dequeues.append(
                        {
                            "ts": dequeue_ts,
                            "job_id": str(consumer_job_id),
                            "task_id": str(ctx.get_task_id()),
                            "request_id": str(item.get("request_id")),
                            "op": str(item.get("op")),
                            "apikey_id": None if item.get("apikey_id") is None else str(item.get("apikey_id")),
                            "throttle_principal": (
                                None
                                if item.get("throttle_principal") is None
                                else str(item.get("throttle_principal"))
                            ),
                            "dequeue_reason": str(dequeue_reason),
                            "scheduler_domain": None if scheduler_domain is None else str(scheduler_domain),
                            "scheduler_session_id": None if scheduler_session_id is None else str(scheduler_session_id),
                            "wait_s": float(wait_s),
                        }
                    )
                except Exception:
                    pass
                return item

        def _item_executor(self, item: dict[str, Any]) -> str:
            executor = item.get("executor")
            if isinstance(executor, str) and executor.strip():
                return executor.strip()
            op = item.get("op")
            if isinstance(op, str) and op.strip():
                return op.strip()
            return "unknown"

        def _execution_serial_key(self, item: dict[str, Any]) -> str | None:
            extra = item.get("extra")
            if not isinstance(extra, dict):
                return None
            raw = extra.get("execution_serial_key")
            if not isinstance(raw, str):
                return None
            key = raw.strip()
            return key or None

        def _iter_all_queued_items(self):
            for item in self._items:
                yield item
            for state in self._sched_domains.values():
                queues_by_session = state.get("queues_by_session", {})
                for q in queues_by_session.values():
                    for item in q:
                        yield item

        def _queued_age_stats(self) -> dict[str, float]:
            now = time.time()
            ages: list[float] = []
            for item in self._iter_all_queued_items():
                try:
                    ts = float(item.get("created_at", 0.0))
                except Exception:
                    continue
                if ts <= 0:
                    continue
                ages.append(max(0.0, now - ts))

            return {
                "oldest_queued_s": max(ages) if ages else 0.0,
                "avg_queued_s": (sum(ages) / len(ages)) if ages else 0.0,
            }

        def stats(self) -> dict[str, Any]:
            now = time.time()
            depth_legacy = int(len(self._items))
            depth_scheduled = int(self._scheduled_depth())
            by_executor: dict[str, int] = {}
            for item in self._iter_all_queued_items():
                executor = self._item_executor(item)
                by_executor[executor] = int(by_executor.get(executor, 0)) + 1
            scheduler_metrics_ready = bool(self._active_job_id)
            return {
                "depth": int(depth_legacy + depth_scheduled),
                "depth_legacy": int(depth_legacy),
                "depth_scheduled": int(depth_scheduled),
                "enqueued": int(self._enqueued),
                "dequeued": int(self._dequeued),
                "by_executor": by_executor,
                "by_apikey_id": dict(self._queued_asample_by_apikey),
                "by_throttle_principal": dict(self._queued_asample_by_principal),
                "age_stats": self._queued_age_stats(),
                "execution_time_s_by_op": {
                    op: {
                        "last": float(self._last_exec_s_by_op.get(op, 0.0)),
                        "ema": float(self._ema_exec_s_by_op.get(op, 0.0)),
                        "sum": float(self._sum_exec_s_by_op.get(op, 0.0)),
                        "count": int(self._count_exec_by_op.get(op, 0)),
                        "max": float(self._max_exec_s_by_op.get(op, 0.0)),
                    }
                    for op in sorted(
                        set(self._last_exec_s_by_op)
                        | set(self._ema_exec_s_by_op)
                        | set(self._sum_exec_s_by_op)
                        | set(self._count_exec_by_op)
                        | set(self._max_exec_s_by_op)
                    )
                },
                "scheduler_metrics_ready": scheduler_metrics_ready,
                "scheduler_enabled": bool(self._scheduler_enabled),
                "scheduler_picks_total": int(self._sched_stats.get("picks_total", 0)),
                "scheduler_switches_total": int(self._sched_stats.get("switches_total", 0)),
                "scheduler_starvation_picks_total": int(self._sched_stats.get("starvation_picks_total", 0)),
                "scheduler_wait_s_sum": float(self._sched_stats.get("wait_s_sum", 0.0)),
                "scheduler_domains_total": int(len(self._sched_domains)),
                **self._scheduler_metrics_snapshot(now=now),
            }

        def metrics_seed_snapshot(self) -> dict[str, Any]:
            queued_items: list[dict[str, Any]] = []
            for item in self._iter_all_queued_items():
                queued_items.append(
                    {
                        "request_id": str(item.get("request_id") or ""),
                        "executor": self._item_executor(item),
                        "created_at": item.get("created_at"),
                        "op": item.get("op"),
                        "throttle_principal": item.get("throttle_principal"),
                        "apikey_id": item.get("apikey_id"),
                    }
                )
            return {
                "stats": self.stats(),
                "queued_items": queued_items,
            }

        def _scheduler_summary(self) -> dict[str, Any]:
            return {
                "enabled": bool(self._scheduler_enabled),
                "max_consecutive": int(self._scheduler_max_consecutive),
                "fairness": str(self._scheduler_fairness),
                "starvation_s": float(self._scheduler_starvation_s),
                "coalesce_ms": float(self._scheduler_coalesce_ms),
                "stats": {
                    "picks_total": int(self._sched_stats.get("picks_total", 0)),
                    "switches_total": int(self._sched_stats.get("switches_total", 0)),
                    "starvation_picks_total": int(self._sched_stats.get("starvation_picks_total", 0)),
                    "wait_s_sum": float(self._sched_stats.get("wait_s_sum", 0.0)),
                    "switch_reasons": dict(self._sched_stats.get("switch_reasons", {})),
                },
            }

        def debug_state(self) -> dict[str, Any]:
            return {
                "stats": self.stats(),
                "recent_enqueues": list(self._recent_enqueues),
                "recent_dequeues": list(self._recent_dequeues),
                "recent_scheduler_decisions": list(self._recent_scheduler_decisions),
                "active_job_id": self._active_job_id,
                "scheduler": self._scheduler_debug(),
            }

        def scheduler_decisions(
            self,
            *,
            limit: int = 100,
            scheduler_domain: str | None = None,
            reason: str | None = None,
            since_seq: int | None = None,
        ) -> dict[str, Any]:
            items = list(self._recent_scheduler_decisions)
            if scheduler_domain is not None:
                target_domain = str(scheduler_domain).strip()
                items = [item for item in items if str(item.get("scheduler_domain") or "") == target_domain]
            if reason is not None:
                target_reason = str(reason).strip()
                items = [item for item in items if str(item.get("decision_reason") or "") == target_reason]
            if since_seq is not None:
                min_seq = int(since_seq)
                items = [item for item in items if int(item.get("seq") or 0) > min_seq]
            max_items = max(1, int(limit))
            items = items[-max_items:]
            return {
                "actor_name": _ray_api_work_queue_actor_name(),
                "last_seq": int(self._scheduler_decision_seq),
                "items": items,
                "scheduler": self._scheduler_summary(),
            }

        def find_position(self, request_id: str) -> dict[str, Any]:
            rid = str(request_id)
            pos = None
            depth = len(self._items)
            for idx, item in enumerate(self._items):
                if str(item.get("request_id")) == rid:
                    pos = idx
                    break
            return {"found": pos is not None, "position": pos, "depth": depth}

        def describe_pending_request(self, request_id: str, op: str | None = None) -> dict[str, Any]:
            out = self.find_position(request_id)
            pos = out.get("position")
            if pos is None or op is None:
                out["ema_exec_s"] = None
                return out
            key = str(op).strip() or "unknown"
            v = self._ema_exec_s_by_op.get(key)
            out["ema_exec_s"] = None if v is None else float(v)
            return out

        def record_execution_time(self, op: str, duration_s: float) -> None:
            key = str(op).strip() or "unknown"
            try:
                d = float(duration_s)
            except Exception:
                return
            if d <= 0:
                return
            self._last_exec_s_by_op[key] = d
            self._sum_exec_s_by_op[key] = float(self._sum_exec_s_by_op.get(key, 0.0)) + d
            self._count_exec_by_op[key] = int(self._count_exec_by_op.get(key, 0)) + 1
            self._max_exec_s_by_op[key] = max(float(self._max_exec_s_by_op.get(key, 0.0)), d)
            alpha = self._ema_alpha
            prev = self._ema_exec_s_by_op.get(key)
            if prev is None:
                self._ema_exec_s_by_op[key] = d
            else:
                self._ema_exec_s_by_op[key] = (alpha * d) + ((1.0 - alpha) * float(prev))

        def get_eta_state(self, op: str | None = None) -> dict[str, Any]:
            if op is None:
                return {"ema_exec_s": None}
            key = str(op).strip() or "unknown"
            v = self._ema_exec_s_by_op.get(key)
            return {"ema_exec_s": None if v is None else float(v)}

    # Keep the detached queue actor on a stable control-plane node. By default
    # this is the head, but MINT_DETACHED_ACTOR_NODE_IP can move it elsewhere.
    # The API work queue still has its own higher-priority pin: if
    # MINT_API_WORK_QUEUE_PINNED_NODE_IP is set we honor that first, otherwise we
    # fall back to the general detached/control-plane placement rules.
    resources = _api_work_queue_actor_resources()

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_restarts": -1,
        "max_task_retries": -1,
    }
    actor_otel_env = otel_env_vars()
    actor_otel_env.setdefault("MINT_API_WORK_QUEUE_DEBUG_LOG_PATH", _api_work_queue_debug_log_path())
    from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources
    if resources is not None:
        options["resources"] = resources
    else:
        apply_detached_actor_resources(options, ray)
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )
    _append_api_work_queue_debug(
        "driver_create_attempt",
        runtime_env=_summarize_debug_runtime_env(options.get("runtime_env")),
        resources=options.get("resources"),
    )

    created = _RayApiWorkQueueActor.options(  # type: ignore[attr-defined]
        **options
    ).remote()
    _append_api_work_queue_debug("driver_create_remote_returned", require_ready=bool(require_ready))
    if not require_ready:
        return created
    try:
        ray.get(created.stats.remote(), timeout=1.0)
        _append_api_work_queue_debug("driver_create_ready_ok")
        return created
    except Exception as e:
        _append_api_work_queue_debug(
            "driver_create_ready_probe_failed",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
        return ray.get_actor(actor_name, namespace=_ray_namespace())


def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_api_work_queue_actor_name()
    probe_timeout_s = float(os.environ.get("MINT_API_WORK_QUEUE_PROBE_TIMEOUT_S", "1.0"))
    fail_fast_on_probe_timeout = (
        os.environ.get("MINT_API_WORK_QUEUE_FAIL_FAST_ON_PROBE_TIMEOUT", "").strip().lower()
        in ("1", "true", "yes", "y", "on")
    )
    actor = None
    try:
        actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        # Quick liveness probe: if the actor is mid-restart, stats() will hang
        # until Ray finishes re-initializing it. Use a short timeout so the
        # request path fails fast with 503 instead of blocking.
        ray.get(actor.stats.remote(), timeout=probe_timeout_s)
        return actor
    except ValueError:
        logger.info("[api_work_queue] actor %s not found; creating", actor_name)
    except ray.exceptions.GetTimeoutError:
        if fail_fast_on_probe_timeout:
            logger.warning(
                "[api_work_queue] actor %s alive but unresponsive (probe_timeout_s=%.2f); failing fast",
                actor_name,
                probe_timeout_s,
            )
            raise ApiWorkQueueUnavailableError(
                f"queue actor {actor_name} unresponsive (restarting?)"
            )
        logger.warning(
            "[api_work_queue] actor %s probe timed out (probe_timeout_s=%.2f); reusing existing actor",
            actor_name,
            probe_timeout_s,
        )
        if actor is not None:
            return actor
    except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
        logger.warning(
            "[api_work_queue] actor %s dead (%s: %s); Ray auto-restart will recover",
            actor_name,
            type(e).__name__,
            e,
        )
        raise ApiWorkQueueUnavailableError(
            f"queue actor {actor_name} restarting ({type(e).__name__})"
        ) from e
    except Exception as e:
        logger.warning(
            "[api_work_queue] failed to fetch detached actor %s (%s: %s); creating",
            actor_name,
            type(e).__name__,
            e,
        )

    return _create_ray_actor()


Executor = Callable[[WorkItem], Awaitable[None]]


class ApiWorkQueueClient:
    def __init__(self) -> None:
        self._ray_actor = None
        self._executors: dict[str, Executor] = {}
        self._worker_tasks: list[Any] = []
        self._queue_supervisor_task: asyncio.Task | None = None
        self._running = False
        self._desired_num_workers = 1
        self._consumer_job_id: str | None = None
        self._consumer_generation_id: int | None = None
        self._execution_ready_event = asyncio.Event()
        self._execution_ready_generation_id: int | None = None
        self._execution_ready_at: float | None = None
        self._execution_serial_states: dict[str, _ExecutionSerialState] = {}
        self._execution_serial_states_guard = asyncio.Lock()

        # Process-local queue snapshot for cheap /metrics reads.
        self._snapshot_lock = threading.Lock()
        self._snapshot_enqueued = 0
        self._snapshot_dequeued = 0
        self._snapshot_items_by_request_id: dict[str, dict[str, Any]] = {}
        self._snapshot_by_executor: dict[str, int] = {}
        self._snapshot_by_apikey_id: dict[str, int] = {}
        self._snapshot_by_throttle_principal: dict[str, int] = {}
        self._snapshot_scheduler_view: dict[str, Any] = {}
        self._snapshot_hydrated = False
        self._snapshot_hydrate_last_attempt_s = 0.0
        self._snapshot_hydrate_min_interval_s = float(
            os.environ.get("MINT_API_WORK_QUEUE_SNAPSHOT_HYDRATE_MIN_INTERVAL_S", "30.0")
        )
        from ..ray_utils import register_ray_reconnect_invalidator

        register_ray_reconnect_invalidator(self._reset_ray_actor)

    def _reset_ray_actor(self) -> None:
        self._ray_actor = None

    @staticmethod
    def _scheduler_metrics_view(snapshot: dict[str, Any]) -> dict[str, Any]:
        trimmed = ApiWorkQueueClient._trim_unready_scheduler_metrics(snapshot)
        out: dict[str, Any] = {}
        for key in (
            "scheduler_metrics_ready",
            "depth_scheduled",
            "scheduler_enabled",
            "scheduler_picks_total",
            "scheduler_switches_total",
            "scheduler_starvation_picks_total",
            "scheduler_wait_s_sum",
            "scheduler_domains_total",
            "scheduler_arbitration_total",
            "scheduler_arbitration_by_winner",
            "scheduler_arbitration_by_reason",
            "scheduled_dequeue_stats",
            "legacy_dequeue_stats",
            "scheduler_domains",
        ):
            if key in trimmed:
                out[key] = trimmed.get(key)
        return out

    @staticmethod
    def _trim_unready_scheduler_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
        if bool(snapshot.get("scheduler_metrics_ready", True)):
            return snapshot
        out = dict(snapshot)
        out.pop("depth_scheduled", None)
        out.pop("scheduled_depth_by_priority", None)
        out.pop("scheduled_dequeue_stats", None)
        out.pop("legacy_dequeue_stats", None)
        for key in list(out):
            if key.startswith("scheduler_") and key != "scheduler_metrics_ready":
                out.pop(key, None)
        return out

    @staticmethod
    def _snapshot_bump(bucket: dict[str, int], key: str | None, delta: int) -> None:
        if key is None:
            return
        next_value = int(bucket.get(key, 0)) + int(delta)
        if next_value > 0:
            bucket[key] = next_value
        else:
            bucket.pop(key, None)

    @staticmethod
    def _snapshot_executor(item: dict[str, Any]) -> str:
        executor = item.get("executor")
        if isinstance(executor, str) and executor.strip():
            return executor.strip()
        op = item.get("op")
        if isinstance(op, str) and op.strip():
            return op.strip()
        return "unknown"

    @staticmethod
    def _snapshot_created_at(item: dict[str, Any]) -> float:
        now = time.time()
        try:
            ts = float(item.get("created_at", now))
        except Exception:
            return now
        if ts <= 0:
            return now
        return ts

    @staticmethod
    def _snapshot_asample_identity(item: dict[str, Any]) -> tuple[str | None, str | None]:
        if str(item.get("op")) != "sampling.asample":
            return None, None
        principal = item.get("throttle_principal")
        apikey_id = item.get("apikey_id")
        principal_str = None if principal is None else str(principal).strip()
        apikey_str = None if apikey_id is None else str(apikey_id).strip()
        if principal_str == "":
            principal_str = None
        if apikey_str == "":
            apikey_str = None
        return principal_str, apikey_str

    def _snapshot_drop_record(self, rec: dict[str, Any]) -> None:
        self._snapshot_bump(self._snapshot_by_executor, rec.get("executor"), -1)
        principal = rec.get("throttle_principal")
        apikey_id = rec.get("apikey_id")
        if principal is not None:
            self._snapshot_bump(self._snapshot_by_throttle_principal, principal, -1)
            if apikey_id is not None:
                self._snapshot_bump(self._snapshot_by_apikey_id, apikey_id, -1)

    def _snapshot_on_enqueue(self, item: dict[str, Any]) -> None:
        request_id = str(item.get("request_id") or "")
        if not request_id:
            return
        executor = self._snapshot_executor(item)
        created_at = self._snapshot_created_at(item)
        principal, apikey_id = self._snapshot_asample_identity(item)
        rec = {
            "executor": executor,
            "created_at": created_at,
            "throttle_principal": principal,
            "apikey_id": apikey_id,
        }
        with self._snapshot_lock:
            previous = self._snapshot_items_by_request_id.pop(request_id, None)
            if isinstance(previous, dict):
                self._snapshot_drop_record(previous)
            self._snapshot_items_by_request_id[request_id] = rec
            self._snapshot_bump(self._snapshot_by_executor, executor, 1)
            if principal is not None:
                self._snapshot_bump(self._snapshot_by_throttle_principal, principal, 1)
                if apikey_id is not None:
                    self._snapshot_bump(self._snapshot_by_apikey_id, apikey_id, 1)
            self._snapshot_enqueued += 1

    def _snapshot_on_dequeue(self, item: dict[str, Any]) -> None:
        request_id = str(item.get("request_id") or "")
        with self._snapshot_lock:
            if request_id:
                rec = self._snapshot_items_by_request_id.pop(request_id, None)
                if isinstance(rec, dict):
                    self._snapshot_drop_record(rec)
            self._snapshot_dequeued += 1

    def _get_ray_actor(self, *, require_ready: bool = True):
        try:
            import ray
        except Exception as e:
            raise ApiWorkQueueUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            raise ApiWorkQueueUnavailableError("Ray not initialized")

        actor_name = _ray_api_work_queue_actor_name()
        probe_timeout_s = float(os.environ.get("MINT_API_WORK_QUEUE_PROBE_TIMEOUT_S", "1.0"))
        fail_fast_on_probe_timeout = (
            os.environ.get("MINT_API_WORK_QUEUE_FAIL_FAST_ON_PROBE_TIMEOUT", "").strip().lower()
            in ("1", "true", "yes", "y", "on")
        )

        actor = self._ray_actor
        if actor is not None:
            if not require_ready:
                return actor
            try:
                ray.get(actor.stats.remote(), timeout=1.0)
                return actor
            except Exception:
                self._ray_actor = None

        actor = None
        try:
            actor = ray.get_actor(actor_name, namespace=_ray_namespace())
            if not require_ready:
                self._ray_actor = actor
                return actor
            ray.get(actor.stats.remote(), timeout=probe_timeout_s)
            self._ray_actor = actor
            return actor
        except ValueError:
            logger.info("[api_work_queue] actor %s not found; creating", actor_name)
        except ray.exceptions.GetTimeoutError:
            if fail_fast_on_probe_timeout:
                logger.warning(
                    "[api_work_queue] actor %s alive but unresponsive (probe_timeout_s=%.2f); failing fast",
                    actor_name,
                    probe_timeout_s,
                )
                raise ApiWorkQueueUnavailableError(
                    f"queue actor {actor_name} unresponsive (restarting?)"
                )
            logger.warning(
                "[api_work_queue] actor %s probe timed out (probe_timeout_s=%.2f); reusing existing actor",
                actor_name,
                probe_timeout_s,
            )
            if actor is not None:
                self._ray_actor = actor
                return actor
        except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
            logger.warning(
                "[api_work_queue] actor %s dead (%s: %s); Ray auto-restart will recover",
                actor_name,
                type(e).__name__,
                e,
            )
            raise ApiWorkQueueUnavailableError(
                f"queue actor {actor_name} restarting ({type(e).__name__})"
            ) from e
        except Exception as e:
            logger.warning(
                "[api_work_queue] failed to fetch detached actor %s (%s: %s); creating",
                actor_name,
                type(e).__name__,
                e,
            )

        try:
            self._ray_actor = _create_ray_actor(require_ready=require_ready)
        except Exception as e:
            raise ApiWorkQueueUnavailableError("Failed to get/create detached Ray ApiWorkQueue actor") from e
        return self._ray_actor

    def hydrate_metrics_snapshot(self, *, timeout_s: float = 10.0, force: bool = False) -> bool:
        now = time.time()
        with self._snapshot_lock:
            if not force and (now - float(self._snapshot_hydrate_last_attempt_s)) < float(
                self._snapshot_hydrate_min_interval_s
            ):
                return bool(self._snapshot_hydrated)
            self._snapshot_hydrate_last_attempt_s = now

        actor = self._get_ray_actor()
        try:
            import ray

            payload = ray.get(actor.metrics_seed_snapshot.remote(), timeout=float(timeout_s))
        except Exception as e:
            logger.warning(
                "[api_work_queue] metrics snapshot hydration failed: %s: %s",
                type(e).__name__,
                e,
            )
            return False

        if not isinstance(payload, dict):
            logger.warning(
                "[api_work_queue] metrics snapshot hydration ignored non-dict payload: %s",
                type(payload),
            )
            return False

        stats = payload.get("stats")
        queued_items = payload.get("queued_items")
        if not isinstance(stats, dict) or not isinstance(queued_items, list):
            logger.warning("[api_work_queue] metrics snapshot hydration ignored malformed payload")
            return False

        next_items: dict[str, dict[str, Any]] = {}
        next_by_executor: dict[str, int] = {}
        next_by_apikey_id: dict[str, int] = {}
        next_by_principal: dict[str, int] = {}

        for item in queued_items:
            if not isinstance(item, dict):
                continue
            request_id = str(item.get("request_id") or "")
            if not request_id:
                continue
            executor = self._snapshot_executor(item)
            created_at = self._snapshot_created_at(item)
            principal, apikey_id = self._snapshot_asample_identity(item)
            rec = {
                "executor": executor,
                "created_at": created_at,
                "throttle_principal": principal,
                "apikey_id": apikey_id,
            }
            next_items[request_id] = rec
            self._snapshot_bump(next_by_executor, executor, 1)
            if principal is not None:
                self._snapshot_bump(next_by_principal, principal, 1)
                if apikey_id is not None:
                    self._snapshot_bump(next_by_apikey_id, apikey_id, 1)

        try:
            enqueued = max(0, int(stats.get("enqueued", len(next_items))))
        except Exception:
            enqueued = int(len(next_items))
        try:
            dequeued = max(0, int(stats.get("dequeued", 0)))
        except Exception:
            dequeued = 0
        scheduler_view = self._scheduler_metrics_view(stats)

        with self._snapshot_lock:
            self._snapshot_items_by_request_id = next_items
            self._snapshot_by_executor = next_by_executor
            self._snapshot_by_apikey_id = next_by_apikey_id
            self._snapshot_by_throttle_principal = next_by_principal
            self._snapshot_enqueued = int(enqueued)
            self._snapshot_dequeued = int(dequeued)
            self._snapshot_scheduler_view = scheduler_view
            self._snapshot_hydrated = True
        return True

    def metrics_snapshot(self) -> dict[str, Any]:
        try:
            self.hydrate_metrics_snapshot(timeout_s=1.0, force=False)
        except Exception:
            pass
        with self._snapshot_lock:
            now = time.time()
            ages = [
                max(0.0, now - float(rec.get("created_at", now)))
                for rec in self._snapshot_items_by_request_id.values()
            ]
            depth = int(len(self._snapshot_items_by_request_id))
            out = {
                "depth": depth,
                "depth_legacy": depth,
                "enqueued": int(self._snapshot_enqueued),
                "dequeued": int(self._snapshot_dequeued),
                "by_executor": dict(self._snapshot_by_executor),
                "by_apikey_id": dict(self._snapshot_by_apikey_id),
                "by_throttle_principal": dict(self._snapshot_by_throttle_principal),
                "age_stats": {
                    "oldest_queued_s": max(ages) if ages else 0.0,
                    "avg_queued_s": (sum(ages) / len(ages)) if ages else 0.0,
                },
                "scheduler_metrics_ready": False,
            }
            scheduler_view = dict(self._snapshot_scheduler_view)
            if scheduler_view:
                out.update(scheduler_view)
                try:
                    depth_scheduled = int(scheduler_view.get("depth_scheduled", 0) or 0)
                except Exception:
                    depth_scheduled = 0
                out["depth_legacy"] = max(0, depth - depth_scheduled)
            return out

    def _get_cached_ray_actor_for_async_request_path(self):
        return self._get_ray_actor(require_ready=False)

    async def _get_ray_actor_async(self, *, require_ready: bool = True):
        _append_api_work_queue_debug("get_ray_actor_async_begin", require_ready=bool(require_ready))
        try:
            import ray
        except Exception as e:
            _append_api_work_queue_debug(
                "get_ray_actor_async_import_error",
                require_ready=bool(require_ready),
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(),
            )
            raise ApiWorkQueueUnavailableError("Ray import failed") from e

        async def _ensure_active_job_binding(actor: Any, *, timeout_s: float) -> None:
            if self._consumer_job_id is None:
                return
            state = await self._await_ray_ref(actor.debug_state.remote(), timeout_s=timeout_s)
            if not isinstance(state, dict):
                raise TypeError(f"ApiWorkQueue.debug_state returned non-dict: {type(state)}")
            if state.get("active_job_id") == self._consumer_job_id:
                return
            ref = actor.set_active_job_id.remote(self._consumer_job_id)
            await self._await_ray_ref(ref, timeout_s=timeout_s)

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray

                init_ray(namespace=_ray_namespace(), ignore_reinit_error=True)
                _append_api_work_queue_debug("get_ray_actor_async_after_init_ray", require_ready=bool(require_ready))
            except Exception as e:
                _append_api_work_queue_debug(
                    "get_ray_actor_async_init_ray_error",
                    require_ready=bool(require_ready),
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(),
                )
                raise ApiWorkQueueUnavailableError("Ray not initialized (init_ray failed)") from e
        else:
            _append_api_work_queue_debug("get_ray_actor_async_using_existing_ray", require_ready=bool(require_ready))
        if not ray.is_initialized():
            raise ApiWorkQueueUnavailableError("Ray not initialized")

        if self._ray_actor is not None:
            if not require_ready:
                _append_api_work_queue_debug("get_ray_actor_async_return_cached", require_ready=False)
                return self._ray_actor
            try:
                await self._await_ray_ref(self._ray_actor.stats.remote(), timeout_s=1.0)
                await _ensure_active_job_binding(self._ray_actor, timeout_s=5.0)
                return self._ray_actor
            except Exception:
                self._ray_actor = None

        actor_name = _ray_api_work_queue_actor_name()
        probe_timeout_s = float(os.environ.get("MINT_API_WORK_QUEUE_PROBE_TIMEOUT_S", "1.0"))
        fail_fast_on_probe_timeout = (
            os.environ.get("MINT_API_WORK_QUEUE_FAIL_FAST_ON_PROBE_TIMEOUT", "").strip().lower()
            in ("1", "true", "yes", "y", "on")
        )

        actor = None
        try:
            actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=_ray_namespace())
            if not require_ready:
                self._ray_actor = actor
                _append_api_work_queue_debug("get_ray_actor_async_found_existing", require_ready=False)
                return actor
            await self._await_ray_ref(actor.stats.remote(), timeout_s=probe_timeout_s)
            await _ensure_active_job_binding(actor, timeout_s=max(5.0, probe_timeout_s))
            self._ray_actor = actor
            return actor
        except ValueError:
            logger.info("[api_work_queue] actor %s not found; creating", actor_name)
        except ray.exceptions.GetTimeoutError:
            if fail_fast_on_probe_timeout:
                logger.info(
                    "[api_work_queue] actor %s alive but unresponsive (probe_timeout_s=%.2f); failing fast",
                    actor_name,
                    probe_timeout_s,
                )
                raise ApiWorkQueueUnavailableError(
                    f"queue actor {actor_name} unresponsive (restarting?)"
                )
            logger.info(
                "[api_work_queue] actor %s probe timed out (probe_timeout_s=%.2f); reusing existing actor",
                actor_name,
                probe_timeout_s,
            )
            if actor is not None:
                self._ray_actor = actor
                return actor
        except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
            logger.info(
                "[api_work_queue] actor %s dead (%s: %s); Ray auto-restart will recover",
                actor_name,
                type(e).__name__,
                e,
            )
            raise ApiWorkQueueUnavailableError(
                f"queue actor {actor_name} restarting ({type(e).__name__})"
            ) from e
        except Exception as e:
            logger.info(
                "[api_work_queue] failed to fetch detached actor %s (%s: %s); creating",
                actor_name,
                type(e).__name__,
                e,
            )

        try:
            self._ray_actor = _create_ray_actor(require_ready=require_ready)
            _append_api_work_queue_debug(
                "get_ray_actor_async_created",
                require_ready=bool(require_ready),
            )
            if require_ready:
                await _ensure_active_job_binding(self._ray_actor, timeout_s=max(5.0, probe_timeout_s))
        except Exception as e:
            _append_api_work_queue_debug(
                "get_ray_actor_async_error",
                require_ready=bool(require_ready),
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(),
            )
            raise ApiWorkQueueUnavailableError("Failed to get/create detached Ray ApiWorkQueue actor") from e
        return self._ray_actor

    async def _await_ray_ref(self, ref: Any, *, timeout_s: float | None = None) -> Any:
        """Await a Ray ObjectRef without threadpool bridges.

        Prefer Ray's asyncio-compatible future() bridge so we can apply asyncio
        timeout semantics without run_in_executor/to_thread.
        """
        awaitable: Any = ref
        ref_future = getattr(ref, "future", None)
        if callable(ref_future):
            try:
                awaitable = asyncio.wrap_future(ref_future())
            except Exception:
                awaitable = ref
        try:
            if timeout_s is None:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=float(timeout_s))
        except asyncio.TimeoutError as e:
            # Preserve the previous surface where timeout on ray.get(...) raised
            # a Ray GetTimeoutError instead of asyncio.TimeoutError.
            import ray

            raise ray.exceptions.GetTimeoutError(f"timed out after {float(timeout_s):.3f}s") from e

    async def async_ensure_started(self) -> None:
        await self._get_ray_actor_async(require_ready=False)

    async def async_ensure_ready(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False)
        import ray

        try:
            out = await self._await_ray_ref(actor.stats.remote(), timeout_s=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise ApiWorkQueueUnavailableError("Detached Ray ApiWorkQueue actor died") from e
        if not isinstance(out, dict):
            raise TypeError(f"ApiWorkQueue.stats returned non-dict: {type(out)}")
        return self._trim_unready_scheduler_metrics(out)

    def set_executor(self, op: str, executor: Executor) -> None:
        self._executors[str(op)] = executor

    def _execution_serial_info(self, item: WorkItem) -> tuple[str | None, str | None, int | None]:
        extra = item.extra if isinstance(item.extra, dict) else {}
        raw_key = extra.get("execution_serial_key")
        key = raw_key.strip() if isinstance(raw_key, str) else ""
        raw_epoch = extra.get("execution_serial_epoch")
        epoch = raw_epoch.strip() if isinstance(raw_epoch, str) else "legacy"
        raw_seq = extra.get("execution_serial_seq")
        seq = raw_seq if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else None
        if not key or seq is None or seq < 1:
            return None, None, None
        return key, epoch, seq

    async def _acquire_execution_serial_state(self, key: str) -> _ExecutionSerialState:
        async with self._execution_serial_states_guard:
            state = self._execution_serial_states.get(key)
            if state is None:
                state = _ExecutionSerialState(cond=asyncio.Condition())
                self._execution_serial_states[key] = state
            state.refs += 1
            return state

    async def _release_execution_serial_state(self, key: str, state: _ExecutionSerialState) -> None:
        async with self._execution_serial_states_guard:
            current = self._execution_serial_states.get(key)
            if current is not state:
                return
            state.refs = max(0, int(state.refs) - 1)
            # Keep the per-key sequence cursor even when there are no active waiters.
            # Dequeue assigns monotonically increasing execution_serial_seq values per key,
            # so dropping state here would reset next_seq back to 1 and deadlock the next
            # non-overlapping item for the same key.

    @asynccontextmanager
    async def _execution_serialized(self, item: WorkItem):
        key, epoch, seq = self._execution_serial_info(item)
        if key is None or epoch is None or seq is None:
            yield
            return

        state = await self._acquire_execution_serial_state(key)
        try:
            async with state.cond:
                state.pending_seqs_by_epoch.setdefault(epoch, set()).add(int(seq))
                if state.current_epoch is None:
                    state.current_epoch = epoch
                    state.next_seq = int(seq)
                while True:
                    current_epoch = state.current_epoch
                    if current_epoch != epoch:
                        current_pending = (
                            set()
                            if current_epoch is None
                            else state.pending_seqs_by_epoch.get(current_epoch, set())
                        )
                        if state.active_seq is None and not current_pending:
                            state.current_epoch = epoch
                            state.next_seq = min(state.pending_seqs_by_epoch[epoch])
                            current_epoch = epoch
                        else:
                            await state.cond.wait()
                            continue
                    if state.active_seq is None and int(state.next_seq) == int(seq):
                        state.active_epoch = epoch
                        state.active_seq = int(seq)
                        break
                    await state.cond.wait()
            try:
                yield
            finally:
                async with state.cond:
                    pending = state.pending_seqs_by_epoch.get(epoch)
                    if pending is not None:
                        pending.discard(int(seq))
                        if not pending:
                            state.pending_seqs_by_epoch.pop(epoch, None)
                    if state.active_epoch == epoch and state.active_seq == int(seq):
                        state.active_epoch = None
                        state.active_seq = None
                    if state.current_epoch == epoch and int(state.next_seq) == int(seq):
                        state.next_seq += 1
                    state.cond.notify_all()
        finally:
            await self._release_execution_serial_state(key, state)

    async def enqueue(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        user_id: str | None,
        webhook_url: str | None,
        apikey_id: str | None = None,
        throttle_principal: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        import ray

        actor = await self._get_ray_actor_async()
        tracer = get_otel_tracer()
        producer_job_id = None
        try:
            producer_job_id = str(ray.get_runtime_context().get_job_id())
        except Exception:
            producer_job_id = None
        item = {
            # Keep external request_id semantics intact; this is the future key/API identifier.
            "request_id": str(request_id),
            "op": str(op),
            "request_json": bytes(request_json),
            "user_id": None if user_id is None else str(user_id),
            "apikey_id": None if apikey_id is None else str(apikey_id),
            "throttle_principal": None if throttle_principal is None else str(throttle_principal),
            "webhook_url": None if webhook_url is None else str(webhook_url),
            "extra": {},
            "created_at": time.time(),
        }
        item_extra = {} if extra is None else dict(extra)
        if not isinstance(item_extra.get("_trace_id"), str) or not item_extra.get("_trace_id"):
            trace_id = get_trace_id() or ensure_trace_id()
            item_extra["_trace_id"] = trace_id
        if not isinstance(item_extra.get("_traceparent"), str) or not item_extra.get("_traceparent"):
            try:
                from opentelemetry import trace as otel_trace

                span_ctx = otel_trace.get_current_span().get_span_context()
                if span_ctx is not None and bool(getattr(span_ctx, "is_valid", False)):
                    flags = int(getattr(span_ctx, "trace_flags", 1))
                    item_extra["_traceparent"] = (
                        f"00-{int(span_ctx.trace_id):032x}-{int(span_ctx.span_id):016x}-{flags:02x}"
                    )
            except Exception:
                pass
        item["extra"] = item_extra

        async def _do_enqueue() -> None:
            # Ensure enqueue succeeds, otherwise the request can remain pending forever
            # while capacity stays reserved.
            ref = actor.enqueue.remote(item, producer_job_id)
            enqueue_timeout_s = max(
                10.0,
                float(os.environ.get("MINT_API_WORK_QUEUE_ENQUEUE_TIMEOUT_S", "60.0")),
            )
            try:
                result = await self._await_ray_ref(ref, timeout_s=enqueue_timeout_s)
            except Exception as e:
                throttle_error = _unwrap_queue_throttle_error(e)
                if throttle_error is not None:
                    raise throttle_error
                raise
            if isinstance(result, dict) and not bool(result.get("ok")):
                detail = result.get("detail")
                if isinstance(detail, dict):
                    raise ApiWorkQueueThrottleError.from_detail(detail)
                raise RuntimeError(f"ApiWorkQueue.enqueue rejected item without detail: {result!r}")
            self._snapshot_on_enqueue(item)

        if tracer is None:
            await _do_enqueue()
            return

        try:
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except Exception:
            await _do_enqueue()
            return

        with tracer.start_as_current_span("queue.enqueue", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("component", "api_work_queue")
            span.set_attribute("op", str(op))
            span.set_attribute("request_id", str(request_id))
            try:
                await _do_enqueue()
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "[queue.enqueue] failed request_id=%s op=%s failure_reason=%s error_type=%s next_action=%s",
                    str(request_id),
                    str(op),
                    classify_failure_reason(e),
                    type(e).__name__,
                    "check_api_work_queue_actor",
                )
                raise

    async def _dequeue(self) -> WorkItem:
        if self._consumer_job_id is None:
            raise RuntimeError("ApiWorkQueueClient not started (consumer job id missing)")

        actor = await self._get_ray_actor_async()
        ref = actor.dequeue.remote(self._consumer_job_id)
        item = await self._await_ray_ref(ref)
        if not isinstance(item, dict):
            raise TypeError(f"ApiWorkQueue.dequeue returned non-dict: {type(item)}")
        if str(item.get("op")) == "__stale_consumer__":
            extra = dict(item.get("extra") or {})
            raise StaleConsumerError(
                f"stale consumer job_id={extra.get('consumer_job_id')!r} active_job_id={extra.get('active_job_id')!r}"
            )
        self._snapshot_on_dequeue(item)
        return WorkItem(
            request_id=str(item["request_id"]),
            op=str(item["op"]),
            request_json=bytes(item["request_json"]),
            user_id=None if item.get("user_id") is None else str(item["user_id"]),
            apikey_id=None if item.get("apikey_id") is None else str(item["apikey_id"]),
            throttle_principal=(
                None if item.get("throttle_principal") is None else str(item["throttle_principal"])
            ),
            webhook_url=None if item.get("webhook_url") is None else str(item["webhook_url"]),
            extra=dict(item.get("extra") or {}),
            created_at=float(item.get("created_at", 0.0)),
        )

    async def find_position(self, request_id: str) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False)
        ref = actor.find_position.remote(request_id=str(request_id))
        result = await self._await_ray_ref(ref, timeout_s=5.0)
        if not isinstance(result, dict):
            raise TypeError(f"ApiWorkQueue.find_position returned non-dict: {type(result)}")
        return result

    async def describe_pending_request(self, request_id: str, op: str | None) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False)
        rid = str(request_id)
        op_key = None if op is None else str(op)
        try:
            ref = actor.describe_pending_request.remote(rid, op_key)
            result = await self._await_ray_ref(ref, timeout_s=5.0)
            if not isinstance(result, dict):
                raise TypeError(f"ApiWorkQueue.describe_pending_request returned non-dict: {type(result)}")
            return result
        except AttributeError:
            ref = actor.find_position.remote(request_id=rid)
            result = await self._await_ray_ref(ref, timeout_s=5.0)
            if not isinstance(result, dict):
                raise TypeError(f"ApiWorkQueue.find_position returned non-dict: {type(result)}")
            pos = result.get("position")
            if pos is None or op_key is None:
                result["ema_exec_s"] = None
                return result
            eta_ref = actor.get_eta_state.remote(op_key)
            eta_state = await self._await_ray_ref(eta_ref, timeout_s=5.0)
            if not isinstance(eta_state, dict):
                raise TypeError(f"ApiWorkQueue.get_eta_state returned non-dict: {type(eta_state)}")
            result["ema_exec_s"] = eta_state.get("ema_exec_s")
            return result

    async def record_execution_time(self, op: str, duration_s: float) -> None:
        actor = await self._get_ray_actor_async()
        ref = actor.record_execution_time.remote(str(op), float(duration_s))
        await self._await_ray_ref(ref, timeout_s=5.0)

    async def get_eta_state(self, op: str | None) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False)
        ref = actor.get_eta_state.remote(None if op is None else str(op))
        result = await self._await_ray_ref(ref, timeout_s=5.0)
        if not isinstance(result, dict):
            raise TypeError(f"ApiWorkQueue.get_eta_state returned non-dict: {type(result)}")
        return result

    async def _reconcile_stale_running_requests(self, consumer_job_id: str) -> int:
        from .capacity_manager import capacity_manager
        from .future_store import FutureStoreUnavailableError, future_store

        stale_leased_request_ids: list[str] = []
        try:
            actor = await self._get_ray_actor_async()
            ref = actor.release_stale_scheduler_leases.remote(str(consumer_job_id))
            stale_leased_request_ids = await self._await_ray_ref(ref, timeout_s=30.0)
            if not isinstance(stale_leased_request_ids, list):
                stale_leased_request_ids = []
        except Exception as e:
            logger.warning(
                "[api_work_queue] release_stale_scheduler_leases failed consumer_job_id=%s: %s: %s",
                str(consumer_job_id),
                type(e).__name__,
                e,
            )

        try:
            stale_request_ids = await future_store.async_fail_stale_running_requests(
                str(consumer_job_id),
                "api server restarted while request was running",
            )
        except FutureStoreUnavailableError as e:
            logger.warning(
                "[api_work_queue] stale-running reconciliation unavailable consumer_job_id=%s: %s: %s",
                str(consumer_job_id),
                type(e).__name__,
                e,
            )
            return 0
        except Exception as e:
            logger.warning(
                "[api_work_queue] stale-running reconciliation failed consumer_job_id=%s: %s: %s",
                str(consumer_job_id),
                type(e).__name__,
                e,
            )
            return 0

        stale_leased_request_ids = [str(request_id) for request_id in stale_leased_request_ids]
        pending_leased_request_ids = [
            request_id
            for request_id in stale_leased_request_ids
            if request_id not in set(stale_request_ids)
        ]
        for request_id in pending_leased_request_ids:
            try:
                await future_store.async_fail(
                    request_id,
                    "api server restarted while request was dequeued before execution began",
                )
            except Exception as e:
                logger.warning(
                    "[api_work_queue] future_store.async_fail failed for stale leased request_id=%s consumer_job_id=%s: %s: %s",
                    str(request_id),
                    str(consumer_job_id),
                    type(e).__name__,
                    e,
                )

        all_stale_request_ids = [*stale_request_ids, *pending_leased_request_ids]
        if not all_stale_request_ids:
            return 0

        for request_id in all_stale_request_ids:
            try:
                await capacity_manager.async_release_all(request_id)
            except Exception as e:
                logger.warning(
                    "[api_work_queue] release_all failed for stale running request_id=%s consumer_job_id=%s: %s: %s",
                    str(request_id),
                    str(consumer_job_id),
                    type(e).__name__,
                    e,
                )
        logger.warning(
            "[api_work_queue] failed %d stale request(s) for previous consumer generation: %s",
            len(all_stale_request_ids),
            all_stale_request_ids,
        )
        return len(all_stale_request_ids)

    def _clear_execution_ready(self) -> None:
        self._execution_ready_event.clear()
        self._execution_ready_generation_id = None
        self._execution_ready_at = None

    def _mark_execution_ready(self, generation_id: int) -> None:
        self._execution_ready_generation_id = int(generation_id)
        self._execution_ready_at = time.time()
        self._execution_ready_event.set()

    async def wait_until_execution_ready(self, *, timeout_s: float = 60.0) -> dict[str, Any]:
        await asyncio.wait_for(self._execution_ready_event.wait(), timeout=float(timeout_s))
        return {
            "execution_ready": True,
            "generation_id": self._execution_ready_generation_id,
            "ready_at": self._execution_ready_at,
        }

    async def _ensure_local_workers_running(self, num_workers: int) -> None:
        alive = [task for task in self._worker_tasks if not task.done()]
        self._worker_tasks = alive
        if self._worker_tasks:
            return
        n = int(num_workers)
        if n < 1:
            n = 1
        self._worker_tasks = [asyncio.create_task(self._worker_loop(i)) for i in range(n)]

    async def _stop_local_workers(self) -> None:
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        self._clear_execution_ready()
        actor = None
        consumer_job_id = self._consumer_job_id
        if consumer_job_id is not None:
            try:
                actor = await self._get_ray_actor_async()
            except Exception:
                actor = None
        if actor is not None and consumer_job_id is not None:
            try:
                ref = actor.clear_active_job_id_if_matches.remote(consumer_job_id)
                await self._await_ray_ref(ref, timeout_s=5.0)
            except Exception:
                pass

    async def _queue_supervisor_loop(self) -> None:
        from .queue_supervisor import queue_supervisor

        start_timeout_s = max(
            10.0,
            float(os.environ.get("MINT_API_WORK_QUEUE_START_TIMEOUT_S", "60.0")),
        )

        self._clear_execution_ready()
        while self._running:
            try:
                snapshot = await queue_supervisor.async_claim_generation(timeout_s=start_timeout_s)
                generation_id = int(snapshot.get("generation_id") or 0)
                owner_id = snapshot.get("owner_id")
                generation_state = str(snapshot.get("state") or "")
                owns_generation = bool(owner_id) and str(owner_id) == queue_supervisor.owner_id() and generation_id > 0
                if owns_generation:
                    consumer_job_id = f"{queue_supervisor.owner_id()}:{generation_id}"
                    generation_changed = (
                        self._consumer_generation_id != generation_id
                        or self._consumer_job_id != consumer_job_id
                    )
                    needs_reconcile = generation_changed or generation_state != "active"
                    # This loop is the mechanism that transitions the queue path
                    # into the execution-ready state. Do not recurse through the
                    # "require_ready" probe path here or startup can deadlock on
                    # repeated ready checks against an actor that is waiting for
                    # this very loop to advance.
                    actor = await self._get_ray_actor_async(require_ready=False)
                    if generation_changed:
                        self._clear_execution_ready()
                        ref = actor.set_active_job_id.remote(consumer_job_id)
                        await self._await_ray_ref(ref, timeout_s=start_timeout_s)
                        self._consumer_job_id = consumer_job_id
                        self._consumer_generation_id = generation_id
                    if needs_reconcile:
                        await queue_supervisor.async_begin_reconcile(generation_id=generation_id)
                        stale_reconciled = await self._reconcile_stale_running_requests(consumer_job_id)
                        await queue_supervisor.async_finish_reconcile(
                            generation_id=generation_id,
                            stale_reconciled=stale_reconciled,
                        )
                    await self._ensure_local_workers_running(self._desired_num_workers)
                    ok = await queue_supervisor.async_heartbeat(generation_id=generation_id)
                    if ok:
                        self._mark_execution_ready(generation_id)
                    else:
                        await self._stop_local_workers()
                        self._consumer_job_id = None
                        self._consumer_generation_id = None
                else:
                    self._clear_execution_ready()
                    if self._worker_tasks:
                        await self._stop_local_workers()
                    self._consumer_job_id = None
                    self._consumer_generation_id = None
            except Exception as e:
                self._clear_execution_ready()
                logger.error(
                    "[api_work_queue] queue supervisor loop failed: %s: %s",
                    type(e).__name__,
                    e,
                )
            await asyncio.sleep(queue_supervisor.poll_s())

    async def start_workers(self, *, num_workers: int) -> None:
        self._desired_num_workers = max(1, int(num_workers))
        if self._running:
            alive = [task for task in self._worker_tasks if not task.done()]
            self._worker_tasks = alive
            if not self._worker_tasks or self._queue_supervisor_task is None or self._queue_supervisor_task.done():
                self._clear_execution_ready()
                if self._queue_supervisor_task is None or self._queue_supervisor_task.done():
                    self._queue_supervisor_task = asyncio.create_task(self._queue_supervisor_loop())
            return

        hydrate_retries = max(
            1,
            int(os.environ.get("MINT_API_WORK_QUEUE_METRICS_HYDRATE_STARTUP_RETRIES", "3")),
        )
        hydrate_retry_delay_s = max(
            0.0,
            float(os.environ.get("MINT_API_WORK_QUEUE_METRICS_HYDRATE_RETRY_DELAY_S", "0.2")),
        )
        hydrate_timeout_s = max(
            10.0,
            float(os.environ.get("MINT_API_WORK_QUEUE_START_TIMEOUT_S", "60.0")),
        )
        hydrated = False
        hydrate_error: Exception | None = None
        for attempt in range(1, hydrate_retries + 1):
            try:
                hydrated = bool(self.hydrate_metrics_snapshot(timeout_s=hydrate_timeout_s, force=True))
                hydrate_error = None
            except Exception as e:
                hydrated = False
                hydrate_error = e
                logger.warning(
                    "[api_work_queue] metrics snapshot hydration startup attempt=%s failed: %s: %s",
                    attempt,
                    type(e).__name__,
                    e,
                )
            if hydrated:
                break
            if attempt < hydrate_retries and hydrate_retry_delay_s > 0.0:
                await asyncio.sleep(hydrate_retry_delay_s)

        if not hydrated:
            if hydrate_error is not None:
                logger.warning(
                    "[api_work_queue] metrics snapshot hydration unavailable at startup; continuing without scheduler metrics: %s: %s",
                    type(hydrate_error).__name__,
                    hydrate_error,
                )
            else:
                logger.warning(
                    "[api_work_queue] metrics snapshot hydration unavailable at startup; continuing without scheduler metrics"
                )

        self._clear_execution_ready()
        self._running = True
        self._queue_supervisor_task = asyncio.create_task(self._queue_supervisor_loop())

    async def shutdown(self) -> None:
        self._running = False
        if self._queue_supervisor_task is not None:
            self._queue_supervisor_task.cancel()
            await asyncio.gather(self._queue_supervisor_task, return_exceptions=True)
            self._queue_supervisor_task = None
        await self._stop_local_workers()
        self._clear_execution_ready()
        self._consumer_job_id = None
        self._consumer_generation_id = None

    async def _worker_loop(self, worker_idx: int) -> None:

        from .capacity_manager import capacity_manager
        from .future_store import FutureStatus, future_store
        from .queue_execution_context import queue_execution_context
        from .queue_supervisor import queue_supervisor

        async def _finalize_request_slot(request_id: str) -> None:
            try:
                await self.finalize_request(request_id)
            except Exception as e:
                logger.error(
                    "[api_work_queue] finalize_request failed (worker_idx=%s, request_id=%s): %s: %s",
                    int(worker_idx),
                    str(request_id),
                    type(e).__name__,
                    e,
                )

        while self._running:
            try:
                item = await self._dequeue()
            except StaleConsumerError as e:
                logger.info(
                    "[api_work_queue] stale consumer exiting (worker_idx=%s): %s",
                    int(worker_idx),
                    e,
                )
                break
            except Exception as e:
                # Never let a dequeue failure permanently kill the background workers.
                # If the detached Ray queue actor dies (or Ray connectivity blips),
                # keep the server alive and retry.
                try:
                    import ray

                    if isinstance(e, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                        self._ray_actor = None
                except Exception:
                    logger.error(
                        "[api_work_queue] failed to classify dequeue exception as Ray error (worker_idx=%s): %s: %s",
                        int(worker_idx),
                        type(e).__name__,
                        e,
                    )

                logger.error(
                    "[api_work_queue] dequeue failed (worker_idx=%s): %s: %s failure_reason=%s next_action=%s",
                    int(worker_idx),
                    type(e).__name__,
                    e,
                    classify_failure_reason(e),
                    "retry_dequeue",
                )
                await asyncio.sleep(1.0)
                continue

            # Restore trace_id/request_id context for logging in worker.
            item_trace_id = item.extra.get("_trace_id")
            item_traceparent = item.extra.get("_traceparent")
            trace_id = item_trace_id if isinstance(item_trace_id, str) else None
            if trace_id is None and isinstance(item_traceparent, str):
                trace_id = extract_trace_id_from_traceparent(item_traceparent)
            set_trace_id(trace_id)
            set_request_id(item.request_id)

            current_generation_id = self._consumer_generation_id
            is_current = False
            if current_generation_id is not None:
                try:
                    is_current = await queue_supervisor.async_is_generation_current(
                        generation_id=int(current_generation_id)
                    )
                except Exception:
                    is_current = False
            if current_generation_id is None or not is_current:
                try:
                    await future_store.async_fail(
                        item.request_id,
                        "queue generation fenced before execution",
                    )
                except Exception:
                    pass
                try:
                    await capacity_manager.async_release_all(item.request_id)
                except Exception:
                    pass
                try:
                    await queue_supervisor.async_record_fenced_worker(generation_id=int(current_generation_id or -1))
                except Exception:
                    pass
                await _finalize_request_slot(item.request_id)
                logger.warning(
                    "[api_work_queue] fenced stale generation before execution request_id=%s worker_idx=%s generation_id=%s",
                    str(item.request_id),
                    int(worker_idx),
                    current_generation_id,
                )
                break

            if str(item.op) == "training.create_model":
                try:
                    age_s = max(0.0, time.time() - float(item.created_at))
                except Exception:
                    age_s = -1.0
                logger.info(
                    "[api_work_queue] dequeued request_id=%s op=%s worker_idx=%s age_s=%.3f",
                    str(item.request_id),
                    str(item.op),
                    int(worker_idx),
                    float(age_s),
                )
            try:
                await capacity_manager.async_release_queue(item.request_id)
            except Exception as e:
                # Do not fail open: the reservation leak will force 429 and surface via stats.
                logger.error(
                    "[api_work_queue] release_queue failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                    int(worker_idx),
                    str(item.request_id),
                    str(item.op),
                    type(e).__name__,
                    e,
                )

            async with self._execution_serialized(item):
                # If the future has already transitioned to a terminal state (for example due to
                # queue-timeout), do not run the executor. This prevents a timed-out future from
                # later being overwritten by a "successful" resolve.
                try:
                    status = await future_store.async_get_status(item.request_id)
                except KeyError:
                    await _finalize_request_slot(item.request_id)
                    try:
                        await capacity_manager.async_release_all(item.request_id)
                    except Exception as e:
                        logger.error(
                            "[api_work_queue] release_all failed after unknown future (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e).__name__,
                            e,
                        )
                    continue
                except Exception as e:
                    logger.error(
                        "[api_work_queue] get_status failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                    )
                    try:
                        await future_store.async_fail(
                            item.request_id,
                            f"internal error: future_store.get_status failed: {type(e).__name__}: {e}",
                        )
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] future_store.fail failed after get_status error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    try:
                        await capacity_manager.async_release_all(item.request_id)
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] release_all failed after get_status error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    await _finalize_request_slot(item.request_id)
                    continue

                if status is not None and status != FutureStatus.PENDING:
                    if str(item.op) == "training.create_model":
                        logger.info(
                            "[api_work_queue] skip_non_pending request_id=%s op=%s worker_idx=%s status=%s",
                            str(item.request_id),
                            str(item.op),
                            int(worker_idx),
                            str(status),
                        )
                    try:
                        await capacity_manager.async_release_all(item.request_id)
                    except Exception as e:
                        logger.error(
                            "[api_work_queue] release_all failed after skip_non_pending (worker_idx=%s, request_id=%s, op=%s, status=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            str(status),
                            type(e).__name__,
                            e,
                        )
                    await _finalize_request_slot(item.request_id)
                    continue

                try:
                    running_stage = "prefill" if str(item.op).startswith("sampling.") else "running"
                    await future_store.async_mark_running(
                        item.request_id,
                        meta={
                            "worker_idx": int(worker_idx),
                            "consumer_job_id": str(self._consumer_job_id),
                            "generation_id": None if self._consumer_generation_id is None else int(self._consumer_generation_id),
                            "op": item.op,
                            "queue_state": "running",
                            "stage": running_stage,
                            "running_at": time.time(),
                        },
                    )
                    logger.debug(
                        "[api_work_queue] mark_running request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )
                except Exception as e:
                    logger.error(
                        "[api_work_queue] mark_running failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                    )
                    try:
                        await future_store.async_fail(
                            item.request_id,
                            f"internal error: future_store.mark_running failed: {type(e).__name__}: {e}",
                        )
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] future_store.fail failed after mark_running error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    try:
                        await capacity_manager.async_release_all(item.request_id)
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] release_all failed after mark_running error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    await _finalize_request_slot(item.request_id)
                    continue

                if str(item.op) == "training.create_model":
                    logger.info(
                        "[api_work_queue] running request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )

                ex = self._executors.get(item.op)
                if ex is None:
                    logger.error(
                        "[api_work_queue] unknown op request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )
                    try:
                        await future_store.async_fail(item.request_id, f"unknown op: {item.op!r}")
                    except Exception as e:
                        logger.error(
                            "[api_work_queue] future_store.fail failed for unknown op (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e).__name__,
                            e,
                        )
                    try:
                        await capacity_manager.async_release_object_store(item.request_id)
                    except Exception as e:
                        logger.error(
                            "[api_work_queue] release_object_store failed for unknown op (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e).__name__,
                            e,
                        )
                    await _finalize_request_slot(item.request_id)
                    continue

                try:
                    exec_start = time.perf_counter()
                    logger.debug(
                        "[api_work_queue] executing request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )
                    tracer = get_otel_tracer()
                    with queue_execution_context(
                        consumer_id=self._consumer_job_id,
                        generation_id=self._consumer_generation_id,
                    ):
                        if tracer is None:
                            await ex(item)
                        else:
                            try:
                                from opentelemetry.propagate import extract
                                from opentelemetry.trace import SpanKind, Status, StatusCode
                            except Exception:
                                # Best-effort: never block execution if OTel deps are unavailable.
                                await ex(item)
                            else:
                                span_context = None
                                traceparent = None
                                if isinstance(item.extra, dict):
                                    traceparent = item.extra.get("_traceparent")
                                if isinstance(traceparent, str) and traceparent:
                                    try:
                                        span_context = extract({"traceparent": traceparent})
                                    except Exception:
                                        span_context = None
                                with tracer.start_as_current_span(
                                    "queue.execute",
                                    kind=SpanKind.INTERNAL,
                                    context=span_context,
                                ) as span:
                                    span.set_attribute("component", "api_work_queue")
                                    span.set_attribute("op", str(item.op))
                                    span.set_attribute("request_id", str(item.request_id))
                                    span.set_attribute("worker_idx", int(worker_idx))
                                    try:
                                        await ex(item)
                                    except Exception as e:
                                        span.record_exception(e)
                                        span.set_status(Status(StatusCode.ERROR, str(e)))
                                        raise
                    logger.debug(
                        "[api_work_queue] executor completed request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )
                    exec_elapsed = time.perf_counter() - exec_start
                    try:
                        actor = await self._get_ray_actor_async()
                        actor.record_execution_time.remote(str(item.op), float(exec_elapsed))
                    except Exception as e:
                        logger.warning(
                            "[api_work_queue] record_execution_time failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e).__name__,
                            e,
                        )
                    if str(item.op) == "training.create_model":
                        logger.info(
                            "[api_work_queue] done request_id=%s op=%s worker_idx=%s",
                            str(item.request_id),
                            str(item.op),
                            int(worker_idx),
                        )
                    await _finalize_request_slot(item.request_id)
                except Exception as e:
                    logger.error(
                        "[api_work_queue] executor failed (worker_idx=%s, request_id=%s, op=%s): %s: %s failure_reason=%s next_action=%s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                        classify_failure_reason(e),
                        "check_executor_and_future_store",
                    )
                    # Ensure the future does not remain pending forever.
                    try:
                        await future_store.async_fail(item.request_id, f"executor failed: {e}")
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] future_store.fail failed after executor exception (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    try:
                        await capacity_manager.async_release_object_store(item.request_id)
                    except Exception as e2:
                        logger.error(
                            "[api_work_queue] release_object_store failed after executor exception (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                            int(worker_idx),
                            str(item.request_id),
                            str(item.op),
                            type(e2).__name__,
                            e2,
                        )
                    await _finalize_request_slot(item.request_id)

    async def stats(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        try:
            actor = self._get_cached_ray_actor_for_async_request_path()
        except ApiWorkQueueUnavailableError:
            if not self._snapshot_hydrated:
                self.hydrate_metrics_snapshot(timeout_s=float(timeout_s), force=True)
            return self.metrics_snapshot()
        ref = actor.stats.remote()
        out = await self._await_ray_ref(ref, timeout_s=float(timeout_s))
        if not isinstance(out, dict):
            raise TypeError(f"ApiWorkQueue.stats returned non-dict: {type(out)}")
        return self._trim_unready_scheduler_metrics(out)

    async def rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        actor = self._get_cached_ray_actor_for_async_request_path()
        ref = actor.get_rss_bytes.remote()
        v = await self._await_ray_ref(ref, timeout_s=float(timeout_s))
        return int(v)

    async def finalize_request(self, request_id: str, *, timeout_s: float = 10.0) -> None:
        actor = await self._get_ray_actor_async()
        ref = actor.finalize_request.remote(str(request_id))
        await self._await_ray_ref(ref, timeout_s=float(timeout_s))

    async def debug_state(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async()
        ref = actor.debug_state.remote()
        v = await self._await_ray_ref(ref, timeout_s=float(timeout_s))
        if not isinstance(v, dict):
            raise TypeError(f"ApiWorkQueue.debug_state returned non-dict: {type(v)}")
        return v

    async def scheduler_decisions(
        self,
        *,
        limit: int = 100,
        scheduler_domain: str | None = None,
        reason: str | None = None,
        since_seq: int | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async()
        ref = actor.scheduler_decisions.remote(
            limit=int(limit),
            scheduler_domain=None if scheduler_domain is None else str(scheduler_domain),
            reason=None if reason is None else str(reason),
            since_seq=None if since_seq is None else int(since_seq),
        )
        v = await self._await_ray_ref(ref, timeout_s=float(timeout_s))
        if not isinstance(v, dict):
            raise TypeError(f"ApiWorkQueue.scheduler_decisions returned non-dict: {type(v)}")
        return v


api_work_queue = ApiWorkQueueClient()
