from __future__ import annotations

import threading
from dataclasses import dataclass


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


class RuntimeObservability:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._megatron_session_switch: dict[tuple[str, str], _MegatronSwitchAggregate] = {}
        self._vllm_workload: dict[tuple[str, str, str], _VllmAggregate] = {}
        self._vllm_active_requests: dict[tuple[str, str], int] = {}

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
            agg = self._megatron_session_switch.setdefault(key, _MegatronSwitchAggregate())
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

    def begin_vllm_request(self, *, base_model: str, op: str) -> None:
        key = (str(base_model or "unknown"), str(op or "unknown"))
        with self._lock:
            self._vllm_active_requests[key] = int(self._vllm_active_requests.get(key, 0)) + 1

    def finish_vllm_request(
        self,
        *,
        base_model: str,
        op: str,
        status: str,
        prompt_tokens: int,
        generated_tokens: int,
        duration_s: float,
    ) -> None:
        active_key = (str(base_model or "unknown"), str(op or "unknown"))
        workload_key = (active_key[0], active_key[1], str(status or "unknown"))
        with self._lock:
            current = int(self._vllm_active_requests.get(active_key, 0))
            self._vllm_active_requests[active_key] = max(0, current - 1)
            agg = self._vllm_workload.setdefault(workload_key, _VllmAggregate())
            agg.requests_total += 1
            agg.prompt_tokens_total += max(0, int(prompt_tokens))
            agg.generated_tokens_total += max(0, int(generated_tokens))
            agg.duration_s_total += max(0.0, float(duration_s))
            agg.duration_s_max = max(agg.duration_s_max, max(0.0, float(duration_s)))

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
            for (base_model, op, status), agg in sorted(self._vllm_workload.items()):
                vllm.append(
                    {
                        "base_model": base_model,
                        "op": op,
                        "status": status,
                        "requests_total": int(agg.requests_total),
                        "prompt_tokens_total": int(agg.prompt_tokens_total),
                        "generated_tokens_total": int(agg.generated_tokens_total),
                        "duration_s_total": float(agg.duration_s_total),
                        "duration_s_max": float(agg.duration_s_max),
                    }
                )

            active = []
            for (base_model, op), active_requests in sorted(self._vllm_active_requests.items()):
                active.append(
                    {
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
