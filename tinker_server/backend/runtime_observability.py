from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..logging_context import (
    record_megatron_session_switch_otel,
    record_training_operation_latency_otel,
    record_vllm_actor_latency_otel,
)


@dataclass
class _MegatronSwitchAggregate:
    count: int = 0
    save_s_total: float = 0.0
    save_s_max: float = 0.0
    swap_s_total: float = 0.0
    swap_s_max: float = 0.0
    load_s_total: float = 0.0
    load_s_max: float = 0.0
    reset_bias_s_total: float = 0.0
    reset_bias_s_max: float = 0.0
    total_s_total: float = 0.0
    total_s_max: float = 0.0


@dataclass
class _VllmAggregate:
    requests_total: int = 0
    prompt_tokens_total: int = 0
    generated_tokens_total: int = 0
    duration_s_total: float = 0.0
    duration_s_max: float = 0.0
    ttft_s_total: float = 0.0
    ttft_s_max: float = 0.0
    ttft_s_count: int = 0
    tpot_s_total: float = 0.0
    tpot_s_max: float = 0.0
    tpot_s_count: int = 0


@dataclass
class _OperationAggregate:
    count: int = 0
    duration_s_total: float = 0.0
    duration_s_max: float = 0.0


@dataclass
class _RecentTrainingIncident:
    ts: float
    kind: str
    base_model: str
    backend: str
    op: str
    status: str
    failure_class: str
    actor_name: str | None = None
    node_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    detail: str | None = None
    context: dict[str, object] | None = None


class RuntimeObservability:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._megatron_session_switch: dict[tuple[str, str], _MegatronSwitchAggregate] = {}
        self._megatron_session_switch_pending: dict[tuple[str, str], _MegatronSwitchAggregate] = {}
        self._megatron_session_switch_failures: dict[tuple[str, str], int] = {}
        self._megatron_actor_lifecycle: dict[tuple[str, str], int] = {}
        self._vllm_workload: dict[tuple[str, str, str, str], _VllmAggregate] = {}
        self._vllm_active_requests: dict[tuple[str, str, str], int] = {}
        self._training_operation_latency: dict[tuple[str, str, str, str, str], _OperationAggregate] = {}
        self._dense_actor_bind_decision: dict[tuple[str, str], int] = {}
        self._dense_actor_fatal: dict[tuple[str, str, str], int] = {}
        self._dense_actor_retire: dict[tuple[str, str], int] = {}
        self._recent_training_incidents: list[_RecentTrainingIncident] = []
        self._recent_training_incidents_limit = 64

    def record_megatron_session_switch(
        self,
        *,
        base_model: str,
        session_state: str,
        save_s: float,
        swap_s: float,
        load_s: float,
        reset_bias_s: float,
        total_s: float,
    ) -> None:
        key = (str(base_model or "unknown"), str(session_state or "unknown"))
        with self._lock:
            for store in (self._megatron_session_switch, self._megatron_session_switch_pending):
                agg = store.setdefault(key, _MegatronSwitchAggregate())
                agg.count += 1
                agg.save_s_total += float(save_s)
                agg.save_s_max = max(agg.save_s_max, float(save_s))
                agg.swap_s_total += float(swap_s)
                agg.swap_s_max = max(agg.swap_s_max, float(swap_s))
                agg.load_s_total += float(load_s)
                agg.load_s_max = max(agg.load_s_max, float(load_s))
                agg.reset_bias_s_total += float(reset_bias_s)
                agg.reset_bias_s_max = max(agg.reset_bias_s_max, float(reset_bias_s))
                agg.total_s_total += float(total_s)
                agg.total_s_max = max(agg.total_s_max, float(total_s))

    def record_megatron_session_switch_failure(self, *, base_model: str, reason: str) -> None:
        key = (str(base_model or "unknown"), str(reason or "unknown"))
        with self._lock:
            self._megatron_session_switch_failures[key] = int(self._megatron_session_switch_failures.get(key, 0)) + 1

    def record_megatron_actor_lifecycle(self, *, base_model: str, event: str) -> None:
        key = (str(base_model or "unknown"), str(event or "unknown"))
        with self._lock:
            self._megatron_actor_lifecycle[key] = int(self._megatron_actor_lifecycle.get(key, 0)) + 1

    def begin_vllm_request(self, *, actor_name: str | None, base_model: str, op: str) -> None:
        key = (str(actor_name or "unknown"), str(base_model or "unknown"), str(op or "unknown"))
        with self._lock:
            self._vllm_active_requests[key] = int(self._vllm_active_requests.get(key, 0)) + 1

    def finish_vllm_request(
        self,
        *,
        actor_name: str | None,
        base_model: str,
        op: str,
        status: str,
        prompt_tokens: int,
        generated_tokens: int,
        duration_s: float,
        ttft_s: float | None = None,
        tpot_s: float | None = None,
    ) -> None:
        actor = str(actor_name or "unknown")
        model = str(base_model or "unknown")
        op_name = str(op or "unknown")
        active_key = (actor, model, op_name)
        workload_key = (actor, model, op_name, str(status or "unknown"))
        duration = max(0.0, float(duration_s))
        with self._lock:
            current = int(self._vllm_active_requests.get(active_key, 0))
            self._vllm_active_requests[active_key] = max(0, current - 1)
            agg = self._vllm_workload.setdefault(workload_key, _VllmAggregate())
            agg.requests_total += 1
            agg.prompt_tokens_total += max(0, int(prompt_tokens))
            agg.generated_tokens_total += max(0, int(generated_tokens))
            agg.duration_s_total += duration
            agg.duration_s_max = max(agg.duration_s_max, duration)
            if ttft_s is not None:
                ttft = max(0.0, float(ttft_s))
                agg.ttft_s_total += ttft
                agg.ttft_s_max = max(agg.ttft_s_max, ttft)
                agg.ttft_s_count += 1
            if tpot_s is not None:
                tpot = max(0.0, float(tpot_s))
                agg.tpot_s_total += tpot
                agg.tpot_s_max = max(agg.tpot_s_max, tpot)
                agg.tpot_s_count += 1
        record_vllm_actor_latency_otel(
            actor_name=actor_name,
            base_model=model,
            op=op_name,
            status=str(status or "unknown"),
            duration_s=duration,
        )

    def record_training_operation(
        self,
        *,
        base_model: str,
        backend: str,
        op: str,
        status: str,
        failure_class: str | None,
        duration_s: float,
    ) -> None:
        key = (
            str(base_model or "unknown"),
            str(backend or "unknown"),
            str(op or "unknown"),
            str(status or "unknown"),
            str(failure_class or "none"),
        )
        duration = max(0.0, float(duration_s))
        with self._lock:
            agg = self._training_operation_latency.setdefault(key, _OperationAggregate())
            agg.count += 1
            agg.duration_s_total += duration
            agg.duration_s_max = max(agg.duration_s_max, duration)
        record_training_operation_latency_otel(
            base_model=str(base_model or "unknown"),
            backend=str(backend or "unknown"),
            op=str(op or "unknown"),
            status=str(status or "unknown"),
            failure_class=str(failure_class or "none"),
            duration_s=duration,
        )

    def record_dense_actor_bind_decision(self, *, base_model: str, decision: str) -> None:
        key = (str(base_model or "unknown"), str(decision or "unknown"))
        with self._lock:
            self._dense_actor_bind_decision[key] = int(self._dense_actor_bind_decision.get(key, 0)) + 1

    def record_dense_actor_fatal(self, *, base_model: str, op: str, failure_class: str) -> None:
        key = (
            str(base_model or "unknown"),
            str(op or "unknown"),
            str(failure_class or "unknown"),
        )
        with self._lock:
            self._dense_actor_fatal[key] = int(self._dense_actor_fatal.get(key, 0)) + 1

    def record_dense_actor_retire(self, *, base_model: str, outcome: str) -> None:
        key = (str(base_model or "unknown"), str(outcome or "unknown"))
        with self._lock:
            self._dense_actor_retire[key] = int(self._dense_actor_retire.get(key, 0)) + 1

    def record_training_incident(
        self,
        *,
        kind: str,
        base_model: str,
        backend: str,
        op: str,
        status: str,
        failure_class: str,
        actor_name: str | None = None,
        node_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        detail: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        incident = _RecentTrainingIncident(
            ts=time.time(),
            kind=str(kind or "unknown"),
            base_model=str(base_model or "unknown"),
            backend=str(backend or "unknown"),
            op=str(op or "unknown"),
            status=str(status or "unknown"),
            failure_class=str(failure_class or "unknown"),
            actor_name=None if actor_name is None else str(actor_name),
            node_id=None if node_id is None else str(node_id),
            request_id=None if request_id is None else str(request_id),
            session_id=None if session_id is None else str(session_id),
            detail=None if detail is None else str(detail),
            context=dict(context or {}) if context else None,
        )
        with self._lock:
            self._recent_training_incidents.append(incident)
            if len(self._recent_training_incidents) > self._recent_training_incidents_limit:
                del self._recent_training_incidents[: len(self._recent_training_incidents) - self._recent_training_incidents_limit]

    def flush_otel(self) -> None:
        with self._lock:
            pending = self._megatron_session_switch_pending
            self._megatron_session_switch_pending = {}
        for (base_model, session_state), agg in sorted(pending.items()):
            record_megatron_session_switch_otel(
                base_model=base_model,
                session_state=session_state,
                count=int(agg.count),
                durations_s={
                    "save": float(agg.save_s_total),
                    "swap": float(agg.swap_s_total),
                    "load": float(agg.load_s_total),
                    "reset_bias": float(agg.reset_bias_s_total),
                    "total": float(agg.total_s_total),
                },
            )

    def snapshot(self) -> dict:
        with self._lock:
            megatron = []
            for (base_model, session_state), agg in sorted(self._megatron_session_switch.items()):
                megatron.append(
                    {
                        "base_model": base_model,
                        "session_state": session_state,
                        "count": int(agg.count),
                        "save_s_total": float(agg.save_s_total),
                        "save_s_max": float(agg.save_s_max),
                        "swap_s_total": float(agg.swap_s_total),
                        "swap_s_max": float(agg.swap_s_max),
                        "load_s_total": float(agg.load_s_total),
                        "load_s_max": float(agg.load_s_max),
                        "reset_bias_s_total": float(agg.reset_bias_s_total),
                        "reset_bias_s_max": float(agg.reset_bias_s_max),
                        "total_s_total": float(agg.total_s_total),
                        "total_s_max": float(agg.total_s_max),
                    }
                )

            megatron_session_switch_failures = [
                {
                    "base_model": base_model,
                    "reason": reason,
                    "count": int(count),
                }
                for (base_model, reason), count in sorted(self._megatron_session_switch_failures.items())
            ]

            megatron_actor_lifecycle = [
                {
                    "base_model": base_model,
                    "event": event,
                    "count": int(count),
                }
                for (base_model, event), count in sorted(self._megatron_actor_lifecycle.items())
            ]

            vllm = []
            for (actor_name, base_model, op, status), agg in sorted(self._vllm_workload.items()):
                vllm.append(
                    {
                        "actor_name": actor_name,
                        "base_model": base_model,
                        "op": op,
                        "status": status,
                        "requests_total": int(agg.requests_total),
                        "prompt_tokens_total": int(agg.prompt_tokens_total),
                        "generated_tokens_total": int(agg.generated_tokens_total),
                        "duration_s_total": float(agg.duration_s_total),
                        "duration_s_max": float(agg.duration_s_max),
                        "ttft_s_total": float(agg.ttft_s_total),
                        "ttft_s_max": float(agg.ttft_s_max),
                        "ttft_s_count": int(agg.ttft_s_count),
                        "tpot_s_total": float(agg.tpot_s_total),
                        "tpot_s_max": float(agg.tpot_s_max),
                        "tpot_s_count": int(agg.tpot_s_count),
                    }
                )

            active = []
            for (actor_name, base_model, op), active_requests in sorted(self._vllm_active_requests.items()):
                active.append(
                    {
                        "actor_name": actor_name,
                        "base_model": base_model,
                        "op": op,
                        "active_requests": int(active_requests),
                    }
                )

            training_operation_latency = []
            for (base_model, backend, op, status, failure_class), agg in sorted(self._training_operation_latency.items()):
                training_operation_latency.append(
                    {
                        "base_model": base_model,
                        "backend": backend,
                        "op": op,
                        "status": status,
                        "failure_class": failure_class,
                        "count": int(agg.count),
                        "duration_s_total": float(agg.duration_s_total),
                        "duration_s_max": float(agg.duration_s_max),
                    }
                )

            dense_actor_bind_decision = [
                {
                    "base_model": base_model,
                    "decision": decision,
                    "count": int(count),
                }
                for (base_model, decision), count in sorted(self._dense_actor_bind_decision.items())
            ]

            dense_actor_fatal = [
                {
                    "base_model": base_model,
                    "op": op,
                    "failure_class": failure_class,
                    "count": int(count),
                }
                for (base_model, op, failure_class), count in sorted(self._dense_actor_fatal.items())
            ]

            dense_actor_retire = [
                {
                    "base_model": base_model,
                    "outcome": outcome,
                    "count": int(count),
                }
                for (base_model, outcome), count in sorted(self._dense_actor_retire.items())
            ]

            recent_training_incidents = [
                {
                    "ts": float(incident.ts),
                    "kind": incident.kind,
                    "base_model": incident.base_model,
                    "backend": incident.backend,
                    "op": incident.op,
                    "status": incident.status,
                    "failure_class": incident.failure_class,
                    "actor_name": incident.actor_name,
                    "node_id": incident.node_id,
                    "request_id": incident.request_id,
                    "session_id": incident.session_id,
                    "detail": incident.detail,
                    "context": None if incident.context is None else dict(incident.context),
                }
                for incident in self._recent_training_incidents
            ]

        return {
            "megatron_session_switch": megatron,
            "megatron_session_switch_failures": megatron_session_switch_failures,
            "megatron_actor_lifecycle": megatron_actor_lifecycle,
            "vllm_workload": vllm,
            "vllm_active_requests": active,
            "training_operation_latency": training_operation_latency,
            "dense_actor_bind_decision": dense_actor_bind_decision,
            "dense_actor_fatal": dense_actor_fatal,
            "dense_actor_retire": dense_actor_retire,
            "recent_training_incidents": recent_training_incidents,
        }


runtime_observability = RuntimeObservability()
