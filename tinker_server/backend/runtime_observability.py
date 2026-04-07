from __future__ import annotations

import threading
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


class RuntimeObservability:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._megatron_session_switch: dict[tuple[str, str], _MegatronSwitchAggregate] = {}
        self._megatron_session_switch_pending: dict[tuple[str, str], _MegatronSwitchAggregate] = {}
        self._vllm_workload: dict[tuple[str, str, str, str], _VllmAggregate] = {}
        self._vllm_active_requests: dict[tuple[str, str, str], int] = {}

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
        duration_s: float,
    ) -> None:
        record_training_operation_latency_otel(
            base_model=str(base_model or "unknown"),
            backend=str(backend or "unknown"),
            op=str(op or "unknown"),
            status=str(status or "unknown"),
            duration_s=max(0.0, float(duration_s)),
        )

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

        return {
            "megatron_session_switch": megatron,
            "vllm_workload": vllm,
            "vllm_active_requests": active,
        }


runtime_observability = RuntimeObservability()
