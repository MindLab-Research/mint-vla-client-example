from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..config import config as server_config, otel_env_vars
from ..logging_context import (
    classify_failure_reason,
    ensure_trace_id,
    extract_trace_id_from_traceparent,
    get_trace_id,
    get_otel_tracer,
    init_actor_observability,
    log_with_bound_context,
    set_request_id,
    set_trace_id,
)

logger = logging.getLogger(__name__)


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


def _unwrap_queue_throttle_error(exc: Exception) -> ApiWorkQueueThrottleError | None:
    candidate: Exception | None = exc
    as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            candidate = as_instanceof_cause()
        except Exception:
            candidate = exc
    if isinstance(candidate, ApiWorkQueueThrottleError):
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
    return str(getattr(server_config, "api_work_queue_actor_name", "tinker_api_work_queue"))


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



def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_api_work_queue_actor_name()
    probe_timeout_s = float(os.environ.get("MINT_API_WORK_QUEUE_PROBE_TIMEOUT_S", "1.0"))
    fail_fast_on_probe_timeout = (
        os.environ.get("MINT_API_WORK_QUEUE_FAIL_FAST_ON_PROBE_TIMEOUT", "").strip().lower()
        in ("1", "true", "yes", "y", "on")
    )
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
            "[api_work_queue] actor %s probe timed out (probe_timeout_s=%.2f); killing stale actor and recreating",
            actor_name,
            probe_timeout_s,
        )
        try:
            ray.kill(actor, no_restart=True)
        except Exception as kill_e:
            logger.warning(
                "[api_work_queue] failed to kill stale actor %s after probe timeout (%s: %s); will try recreate",
                actor_name,
                type(kill_e).__name__,
                kill_e,
            )
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

    max_concurrency = int(os.environ.get("MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY", "256"))
    max_restarts = int(os.environ.get("MINT_API_WORK_QUEUE_MAX_RESTARTS", "3"))

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=max_restarts)
    class _RayApiWorkQueueActor:
        def __init__(self) -> None:
            from collections import deque

            init_actor_observability()
            logger.warning(
                "[api_work_queue] actor (re)initializing (max_restarts=%d)",
                max_restarts,
            )
            self._items = deque()
            self._cv = asyncio.Condition()
            self._enqueued = 0
            self._dequeued = 0
            self._recent_dequeues = deque(maxlen=int(os.environ.get("MINT_API_WORK_QUEUE_DEBUG_MAX", "50")))
            self._recent_enqueues = deque(maxlen=int(os.environ.get("MINT_API_WORK_QUEUE_DEBUG_MAX", "50")))
            self._active_job_id: str | None = None
            self._ema_exec_s_by_op: dict[str, float] = {}
            self._ema_alpha = float(os.environ.get("MINT_API_WORK_QUEUE_ETA_ALPHA", "0.1"))
            self._max_pending_asample_per_apikey = int(
                getattr(server_config, "sampling_max_pending_asample_per_apikey", 64)
            )
            self._queued_asample_by_principal: dict[str, int] = {}
            self._queued_asample_by_apikey: dict[str, int] = {}
            self._queued_asample_request_state: dict[str, tuple[str | None, str | None, str | None]] = {}
            self._scheduler_request_meta: dict[str, tuple[str, str, str]] = {}
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
            self._scheduler_train_followup_hold_s = max(
                0.0,
                float(os.environ.get("MINT_SCHEDULER_TRAIN_FOLLOWUP_HOLD_S", "60.0")),
            )
            self._sched_domains: dict[str, dict[str, Any]] = {}
            self._sched_stats: dict[str, Any] = {
                "picks_total": 0,
                "switches_total": 0,
                "starvation_picks_total": 0,
                "wait_s_sum": 0.0,
                "switch_reasons": {},
            }

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
                if meta is None:
                    return
                domain, session_id, op = meta
                state = self._sched_domains.get(domain)
                if state is None:
                    return
                if state.get("leased_request_id") == rid:
                    state["leased_request_id"] = None
                    state["leased_session"] = None
                if op == "training.forward_backward" and self._scheduler_train_followup_hold_s > 0:
                    state["followup_session"] = session_id
                    state["followup_deadline"] = time.time() + float(self._scheduler_train_followup_hold_s)
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
                "consecutive_count": 0,
                "followup_session": None,
                "followup_deadline": 0.0,
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

        def _enqueue_scheduled(self, item: dict[str, Any], *, domain: str, session_id: str) -> None:
            from collections import deque

            state = self._get_domain_state(domain)
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

            followup_session = state.get("followup_session")
            followup_deadline = float(state.get("followup_deadline", 0.0) or 0.0)
            if (
                isinstance(followup_session, str)
                and followup_session
                and followup_deadline > now
                and queues_by_session.get(followup_session)
            ):
                return followup_session, "followup_hold"

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
                and int(state.get("consecutive_count", 0)) < int(self._scheduler_max_consecutive)
            ):
                return current, "sticky"

            avoid = current if isinstance(current, str) and current and len(rr_order) > 1 else None
            if self._scheduler_fairness == "oldest":
                sid = self._pick_oldest_session(state, now=now, avoid=avoid)
                reason = "fairness_oldest"
            else:
                sid = self._pick_round_robin_session(state, avoid=avoid)
                reason = "fairness_rr"
            if sid is None:
                return None
            return sid, reason

        def _pick_scheduled_candidate(self, *, now: float) -> tuple[str, str, str] | None:
            best: tuple[float, str, str, str] | None = None
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
                created_at = self._item_created_at(q[0], now=now)
                if best is None or created_at < best[0]:
                    best = (created_at, domain, sid, reason)
            if best is None:
                return None
            return best[1], best[2], best[3]

        def _record_switch_reason(self, reason: str) -> None:
            reasons = self._sched_stats.get("switch_reasons")
            if not isinstance(reasons, dict):
                reasons = {}
                self._sched_stats["switch_reasons"] = reasons
            reasons[reason] = int(reasons.get(reason, 0)) + 1

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

            if state.get("followup_session") == session_id:
                state["followup_session"] = None
                state["followup_deadline"] = 0.0

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
            state["leased_request_id"] = str(item.get("request_id") or "")
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
                                        continue

                    item: dict[str, Any]
                    dequeue_reason = "fifo"
                    scheduler_domain = None
                    scheduler_session_id = None

                    if has_legacy and sched_choice is not None:
                        legacy_head = self._items[0]
                        legacy_ts = self._item_created_at(legacy_head, now=now)
                        sched_domain, sched_session_id, sched_reason = sched_choice
                        sched_state = self._sched_domains.get(sched_domain)
                        sched_queue = (
                            None
                            if sched_state is None
                            else (sched_state.get("queues_by_session", {}) or {}).get(sched_session_id)
                        )
                        sched_head_ts = (
                            legacy_ts
                            if not sched_queue
                            else self._item_created_at(sched_queue[0], now=now)
                        )
                        if legacy_ts <= sched_head_ts:
                            item = self._items.popleft()
                        else:
                            item = self._pop_scheduled(
                                domain=sched_domain,
                                session_id=sched_session_id,
                                reason=sched_reason,
                                now=now,
                            )
                            dequeue_reason = str(sched_reason)
                            scheduler_domain = str(sched_domain)
                            scheduler_session_id = str(sched_session_id)
                    elif has_legacy:
                        item = self._items.popleft()
                    else:
                        if sched_choice is None:
                            await self._cv.wait()
                            continue
                        sched_domain, sched_session_id, sched_reason = sched_choice
                        item = self._pop_scheduled(
                            domain=sched_domain,
                            session_id=sched_session_id,
                            reason=sched_reason,
                            now=now,
                        )
                        dequeue_reason = str(sched_reason)
                        scheduler_domain = str(sched_domain)
                        scheduler_session_id = str(sched_session_id)
                    self._mark_asample_leased_to_consumer(item.get("request_id", ""), consumer_job_id)
                    break

                self._dequeued += 1

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
                            "ts": time.time(),
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
            depth_legacy = int(len(self._items))
            depth_scheduled = int(self._scheduled_depth())
            by_executor: dict[str, int] = {}
            for item in self._iter_all_queued_items():
                executor = self._item_executor(item)
                by_executor[executor] = int(by_executor.get(executor, 0)) + 1
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
                "scheduler_enabled": bool(self._scheduler_enabled),
                "scheduler_picks_total": int(self._sched_stats.get("picks_total", 0)),
                "scheduler_switches_total": int(self._sched_stats.get("switches_total", 0)),
                "scheduler_starvation_picks_total": int(self._sched_stats.get("starvation_picks_total", 0)),
                "scheduler_wait_s_sum": float(self._sched_stats.get("wait_s_sum", 0.0)),
                "scheduler_domains_total": int(len(self._sched_domains)),
            }

        def debug_state(self) -> dict[str, Any]:
            return {
                "stats": self.stats(),
                "recent_enqueues": list(self._recent_enqueues),
                "recent_dequeues": list(self._recent_dequeues),
                "active_job_id": self._active_job_id,
                "scheduler": self._scheduler_debug(),
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

        def record_execution_time(self, op: str, duration_s: float) -> None:
            key = str(op).strip() or "unknown"
            try:
                d = float(duration_s)
            except Exception:
                return
            if d <= 0:
                return
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

    # Keep the detached queue actor on the head node when possible. Losing this
    # actor drops all queued items (in-memory queue), which can leave futures
    # pending forever.
    resources = None
    try:
        if "node:__internal_head__" in ray.cluster_resources():
            resources = {"node:__internal_head__": 0.001}
    except Exception:
        resources = None

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_restarts": -1,
        "max_task_retries": -1,
    }
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env_vars
    options["runtime_env"] = {
        "env_vars": actor_runtime_env_vars(
            pythonpath=PFS_PYTHONPATH,
            extra=actor_otel_env,
        )
    }
    if resources is not None:
        options["resources"] = resources

    return _RayApiWorkQueueActor.options(  # type: ignore[attr-defined]
        **options
    ).remote()


Executor = Callable[[WorkItem], Awaitable[None]]


class ApiWorkQueueClient:
    def __init__(self) -> None:
        self._ray_actor = None
        self._executors: dict[str, Executor] = {}
        self._worker_tasks: list[Any] = []
        self._running = False
        self._dequeue_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._consumer_job_id: str | None = None

    def _get_ray_actor(self):
        try:
            import ray
        except Exception as e:
            raise ApiWorkQueueUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray

                init_ray(namespace=_ray_namespace(), ignore_reinit_error=True)
            except Exception as e:
                raise ApiWorkQueueUnavailableError("Ray not initialized (init_ray failed)") from e

        if not ray.is_initialized():
            raise ApiWorkQueueUnavailableError("Ray not initialized")

        if self._ray_actor is not None:
            try:
                ray.get(self._ray_actor.stats.remote(), timeout=1.0)
            except Exception:
                self._ray_actor = None

        if self._ray_actor is None:
            try:
                self._ray_actor = _get_or_create_ray_actor()
            except Exception as e:
                raise ApiWorkQueueUnavailableError("Failed to get/create detached Ray ApiWorkQueue actor") from e
        return self._ray_actor

    def set_executor(self, op: str, executor: Executor) -> None:
        self._executors[str(op)] = executor

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

        actor = self._get_ray_actor()
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
            loop = asyncio.get_running_loop()
            enqueue_timeout_s = max(
                10.0,
                float(os.environ.get("MINT_API_WORK_QUEUE_ENQUEUE_TIMEOUT_S", "60.0")),
            )
            try:
                result = await loop.run_in_executor(None, lambda: ray.get(ref, timeout=enqueue_timeout_s))
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
        import ray

        if self._dequeue_executor is None:
            raise RuntimeError("ApiWorkQueueClient not started (dequeue executor missing)")
        if self._consumer_job_id is None:
            raise RuntimeError("ApiWorkQueueClient not started (consumer job id missing)")

        actor = self._get_ray_actor()
        ref = actor.dequeue.remote(self._consumer_job_id)
        loop = asyncio.get_running_loop()
        item = await loop.run_in_executor(self._dequeue_executor, ray.get, ref)
        if not isinstance(item, dict):
            raise TypeError(f"ApiWorkQueue.dequeue returned non-dict: {type(item)}")
        if str(item.get("op")) == "__stale_consumer__":
            extra = dict(item.get("extra") or {})
            raise StaleConsumerError(
                f"stale consumer job_id={extra.get('consumer_job_id')!r} active_job_id={extra.get('active_job_id')!r}"
            )
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
        import ray

        actor = self._get_ray_actor()
        ref = actor.find_position.remote(request_id=str(request_id))
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: ray.get(ref, timeout=5.0))
        if not isinstance(result, dict):
            raise TypeError(f"ApiWorkQueue.find_position returned non-dict: {type(result)}")
        return result

    async def record_execution_time(self, op: str, duration_s: float) -> None:
        import ray

        actor = self._get_ray_actor()
        ref = actor.record_execution_time.remote(str(op), float(duration_s))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: ray.get(ref, timeout=5.0))

    async def get_eta_state(self, op: str | None) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        ref = actor.get_eta_state.remote(None if op is None else str(op))
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: ray.get(ref, timeout=5.0))
        if not isinstance(result, dict):
            raise TypeError(f"ApiWorkQueue.get_eta_state returned non-dict: {type(result)}")
        return result

    async def _reconcile_stale_running_requests(self, consumer_job_id: str) -> None:
        from .capacity_manager import capacity_manager
        from .future_store import FutureStoreUnavailableError, future_store

        try:
            stale_request_ids = future_store.fail_stale_running_requests(
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
            return
        except Exception as e:
            logger.warning(
                "[api_work_queue] stale-running reconciliation failed consumer_job_id=%s: %s: %s",
                str(consumer_job_id),
                type(e).__name__,
                e,
            )
            return

        if not stale_request_ids:
            return

        for request_id in stale_request_ids:
            try:
                capacity_manager.release_all(request_id)
            except Exception as e:
                logger.warning(
                    "[api_work_queue] release_all failed for stale running request_id=%s consumer_job_id=%s: %s: %s",
                    str(request_id),
                    str(consumer_job_id),
                    type(e).__name__,
                    e,
                )
        logger.warning(
            "[api_work_queue] failed %d stale running requests for previous consumer generation: %s",
            len(stale_request_ids),
            stale_request_ids,
        )

    async def start_workers(self, *, num_workers: int) -> None:
        import ray

        if self._running:
            return
        self._running = True

        try:
            job_id = str(ray.get_runtime_context().get_job_id())
            self._consumer_job_id = job_id
            loop = asyncio.get_running_loop()
            start_timeout_s = max(
                10.0,
                float(os.environ.get("MINT_API_WORK_QUEUE_START_TIMEOUT_S", "60.0")),
            )
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    actor = self._get_ray_actor()
                    ref = actor.set_active_job_id.remote(job_id)
                    await loop.run_in_executor(None, lambda: ray.get(ref, timeout=start_timeout_s))
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "[api_work_queue] set_active_job_id attempt=%s failed (%s: %s); recreating actor",
                        attempt,
                        type(e).__name__,
                        e,
                    )
                    self._ray_actor = None
            if last_error is not None:
                raise last_error
            await self._reconcile_stale_running_requests(job_id)
        except Exception as e:
            self._running = False
            self._consumer_job_id = None
            raise RuntimeError(f"Failed to set ApiWorkQueue active job id: {type(e).__name__}: {e}") from e

        n = int(num_workers)
        if n < 1:
            n = 1
        self._dequeue_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=n,
            thread_name_prefix="api_work_queue_dequeue",
        )
        self._worker_tasks = [asyncio.create_task(self._worker_loop(i)) for i in range(n)]

    async def shutdown(self) -> None:
        import ray

        self._running = False
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        actor = None
        consumer_job_id = self._consumer_job_id
        if consumer_job_id is not None:
            try:
                actor = self._get_ray_actor()
            except Exception:
                actor = None
        if actor is not None and consumer_job_id is not None:
            try:
                ref = actor.clear_active_job_id_if_matches.remote(consumer_job_id)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: ray.get(ref, timeout=5.0))
            except Exception:
                pass
        if self._dequeue_executor is not None:
            self._dequeue_executor.shutdown(wait=False, cancel_futures=True)
            self._dequeue_executor = None
        self._consumer_job_id = None

    async def _worker_loop(self, worker_idx: int) -> None:

        from .capacity_manager import capacity_manager
        from .future_store import FutureStatus, future_store

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
                capacity_manager.release_queue(item.request_id)
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

            # If the future has already transitioned to a terminal state (for example due to
            # queue-timeout), do not run the executor. This prevents a timed-out future from
            # later being overwritten by a "successful" resolve.
            try:
                status = future_store.get_status(item.request_id)
            except KeyError:
                await _finalize_request_slot(item.request_id)
                try:
                    capacity_manager.release_all(item.request_id)
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
                    future_store.fail(item.request_id, f"internal error: future_store.get_status failed: {type(e).__name__}: {e}")
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
                    capacity_manager.release_all(item.request_id)
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
                    capacity_manager.release_all(item.request_id)
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
                future_store.mark_running(
                    item.request_id,
                    meta={
                        "worker_idx": int(worker_idx),
                        "consumer_job_id": str(self._consumer_job_id),
                        "op": item.op,
                        "queue_state": "running",
                        "stage": "prefill",
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
                    future_store.fail(item.request_id, f"internal error: future_store.mark_running failed: {type(e).__name__}: {e}")
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
                    capacity_manager.release_all(item.request_id)
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
                    future_store.fail(item.request_id, f"unknown op: {item.op!r}")
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
                    capacity_manager.release_object_store(item.request_id)
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
                if tracer is None:
                    await ex(item)
                else:
                    try:
                        from opentelemetry.trace import SpanKind, Status, StatusCode
                        from opentelemetry.propagate import extract
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
                    actor = self._get_ray_actor()
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
                    future_store.fail(item.request_id, f"executor failed: {e}")
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
                    capacity_manager.release_object_store(item.request_id)
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
        import ray

        actor = self._get_ray_actor()
        ref = actor.stats.remote()
        return ray.get(ref, timeout=float(timeout_s))

    async def rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        import ray

        actor = self._get_ray_actor()
        ref = actor.get_rss_bytes.remote()
        v = ray.get(ref, timeout=float(timeout_s))
        return int(v)

    async def finalize_request(self, request_id: str, *, timeout_s: float = 10.0) -> None:
        import ray

        actor = self._get_ray_actor()
        ref = actor.finalize_request.remote(str(request_id))
        ray.get(ref, timeout=float(timeout_s))

    async def debug_state(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        ref = actor.debug_state.remote()
        v = ray.get(ref, timeout=float(timeout_s))
        if not isinstance(v, dict):
            raise TypeError(f"ApiWorkQueue.debug_state returned non-dict: {type(v)}")
        return v


api_work_queue = ApiWorkQueueClient()
