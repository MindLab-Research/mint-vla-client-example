from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _LatencyAgg:
    total: float = 0.0
    count: int = 0
    max: float = 0.0

    def observe(self, value: object) -> None:
        try:
            x = max(0.0, float(value))
        except (TypeError, ValueError):
            return
        self.total += x
        self.count += 1
        self.max = max(self.max, x)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "total": float(self.total),
            "count": int(self.count),
            "max": float(self.max),
        }


class VllmStatsObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting = 0
        self._running = 0
        self._kv_cache_usage_ratio = 0.0
        self._prefix_cache_queries_total = 0
        self._prefix_cache_hits_total = 0
        self._preemptions_total = 0
        self._queue_time = _LatencyAgg()
        self._prefill_time = _LatencyAgg()
        self._decode_time = _LatencyAgg()
        self._output_token_time = _LatencyAgg()

    def record(self, scheduler_stats: Any, iteration_stats: Any) -> None:
        with self._lock:
            if scheduler_stats is not None:
                self._waiting = max(0, int(getattr(scheduler_stats, "num_waiting_reqs", 0) or 0))
                self._running = max(0, int(getattr(scheduler_stats, "num_running_reqs", 0) or 0))
                try:
                    self._kv_cache_usage_ratio = max(
                        0.0,
                        min(1.0, float(getattr(scheduler_stats, "kv_cache_usage", 0.0) or 0.0)),
                    )
                except (TypeError, ValueError):
                    pass
                pcs = getattr(scheduler_stats, "prefix_cache_stats", None)
                if pcs is not None:
                    self._prefix_cache_queries_total += max(0, int(getattr(pcs, "queries", 0) or 0))
                    self._prefix_cache_hits_total += max(0, int(getattr(pcs, "hits", 0) or 0))

            if iteration_stats is None:
                return

            self._preemptions_total += max(0, int(getattr(iteration_stats, "num_preempted_reqs", 0) or 0))
            for req in getattr(iteration_stats, "finished_requests", []) or []:
                self._queue_time.observe(getattr(req, "queued_time", None))
                self._prefill_time.observe(getattr(req, "prefill_time", None))
                self._decode_time.observe(getattr(req, "decode_time", None))
                self._output_token_time.observe(getattr(req, "mean_time_per_output_token", None))

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            queries = int(self._prefix_cache_queries_total)
            hits = int(self._prefix_cache_hits_total)
            hit_ratio = 0.0 if queries <= 0 else float(hits) / float(queries)
            return {
                "scheduler_waiting_requests": int(self._waiting),
                "scheduler_running_requests": int(self._running),
                "scheduler_kv_cache_usage_ratio": float(self._kv_cache_usage_ratio),
                "prefix_cache_queries_total": queries,
                "prefix_cache_hits_total": hits,
                "prefix_cache_hit_ratio": float(hit_ratio),
                "preemptions_total": int(self._preemptions_total),
                "queue_time_s_total": float(self._queue_time.total),
                "queue_time_s_count": int(self._queue_time.count),
                "queue_time_s_max": float(self._queue_time.max),
                "prefill_time_s_total": float(self._prefill_time.total),
                "prefill_time_s_count": int(self._prefill_time.count),
                "prefill_time_s_max": float(self._prefill_time.max),
                "decode_time_s_total": float(self._decode_time.total),
                "decode_time_s_count": int(self._decode_time.count),
                "decode_time_s_max": float(self._decode_time.max),
                "time_per_output_token_s_total": float(self._output_token_time.total),
                "time_per_output_token_s_count": int(self._output_token_time.count),
                "time_per_output_token_s_max": float(self._output_token_time.max),
            }


class VllmStatsLogger:
    def __init__(self, observer: VllmStatsObserver, vllm_config: Any, engine_index: int = 0):
        self._observer = observer
        self.engine_index = engine_index

    def record(self, scheduler_stats: Any, iteration_stats: Any, engine_idx: int = 0):
        self._observer.record(scheduler_stats, iteration_stats)

    def log_engine_initialized(self):
        return None

    def log(self):
        return None


def make_vllm_stats_logger_factory(observer: VllmStatsObserver):
    def _factory(vllm_config: Any, engine_index: int = 0):
        return VllmStatsLogger(observer, vllm_config, engine_index)

    return _factory


def attach_vllm_stats_logger(engine: Any, observer: VllmStatsObserver) -> None:
    logger_manager = getattr(engine, "logger_manager", None)
    vllm_config = getattr(engine, "vllm_config", None)
    if logger_manager is None:
        return
    for engine_idx, loggers in logger_manager.per_engine_logger_dict.items():
        loggers.append(VllmStatsLogger(observer, vllm_config, engine_idx))
