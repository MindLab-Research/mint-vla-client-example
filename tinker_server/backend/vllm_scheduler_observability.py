from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _LatencyAgg:
    total: float = 0.0
    count: int = 0
    max: float = 0.0
    # Keep a bounded recent window for percentile estimation.
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=512))

    def observe(self, value: object) -> None:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(x):
            return
        x = max(0.0, x)
        self.total += x
        self.count += 1
        self.max = max(self.max, x)
        self.recent.append(x)

    @staticmethod
    def _percentile(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        if len(xs) == 1:
            return float(xs[0])
        pos = max(0.0, min(1.0, float(q))) * float(len(xs) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(xs[lo])
        frac = pos - float(lo)
        return float(xs[lo] + (xs[hi] - xs[lo]) * frac)

    def snapshot(self) -> dict[str, float | int]:
        xs = sorted(self.recent)
        return {
            "total": float(self.total),
            "count": int(self.count),
            "max": float(self.max),
            "p50_recent": self._percentile(xs, 0.50),
            "p95_recent": self._percentile(xs, 0.95),
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
        self._seq_slot_wait_time = _LatencyAgg()
        self._generate_lock_wait_time = _LatencyAgg()
        self._engine_read_lock_wait_time = _LatencyAgg()
        self._add_request_wait_time = _LatencyAgg()
        self._add_request_exec_time = _LatencyAgg()
        self._first_token_observed_time = _LatencyAgg()

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

    def observe_actor_timing(
        self,
        *,
        seq_slot_wait_s: float | None = None,
        generate_lock_wait_s: float | None = None,
        engine_read_lock_wait_s: float | None = None,
        add_request_wait_s: float | None = None,
        add_request_exec_s: float | None = None,
        first_token_observed_s: float | None = None,
    ) -> None:
        with self._lock:
            self._seq_slot_wait_time.observe(seq_slot_wait_s)
            self._generate_lock_wait_time.observe(generate_lock_wait_s)
            self._engine_read_lock_wait_time.observe(engine_read_lock_wait_s)
            self._add_request_wait_time.observe(add_request_wait_s)
            self._add_request_exec_time.observe(add_request_exec_s)
            self._first_token_observed_time.observe(first_token_observed_s)

    @staticmethod
    def _snapshot_with_stem(stem: str, agg: _LatencyAgg) -> dict[str, float | int]:
        snap = agg.snapshot()
        return {
            f"{stem}_total": float(snap["total"]),
            f"{stem}_count": int(snap["count"]),
            f"{stem}_max": float(snap["max"]),
            f"{stem}_p50_recent": float(snap["p50_recent"]),
            f"{stem}_p95_recent": float(snap["p95_recent"]),
        }

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            queries = int(self._prefix_cache_queries_total)
            hits = int(self._prefix_cache_hits_total)
            hit_ratio = 0.0 if queries <= 0 else float(hits) / float(queries)
            out: dict[str, float | int] = {
                "scheduler_waiting_requests": int(self._waiting),
                "scheduler_running_requests": int(self._running),
                "scheduler_kv_cache_usage_ratio": float(self._kv_cache_usage_ratio),
                "prefix_cache_queries_total": queries,
                "prefix_cache_hits_total": hits,
                "prefix_cache_hit_ratio": float(hit_ratio),
                "preemptions_total": int(self._preemptions_total),
            }
            for stem, agg in (
                ("queue_time_s", self._queue_time),
                ("prefill_time_s", self._prefill_time),
                ("decode_time_s", self._decode_time),
                ("time_per_output_token_s", self._output_token_time),
                ("seq_slot_wait_s", self._seq_slot_wait_time),
                ("generate_lock_wait_s", self._generate_lock_wait_time),
                ("engine_read_lock_wait_s", self._engine_read_lock_wait_time),
                ("add_request_wait_s", self._add_request_wait_time),
                ("add_request_exec_s", self._add_request_exec_time),
                ("first_token_observed_s", self._first_token_observed_time),
            ):
                out.update(self._snapshot_with_stem(stem, agg))
            return out


class VllmStatsLogger:
    def __init__(self, observer: VllmStatsObserver, vllm_config: Any, engine_index: int = 0):
        self._observer = observer
        self.engine_index = engine_index

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        engine_idx: int = 0,
        **_kwargs: Any,
    ):
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

    per_engine = getattr(logger_manager, "per_engine_logger_dict", None)
    if isinstance(per_engine, dict):
        for engine_idx, loggers in per_engine.items():
            if isinstance(loggers, list):
                loggers.append(VllmStatsLogger(observer, vllm_config, engine_idx))
        return

    # Newer vLLM / wrapper variants may expose a different logger-manager shape.
    # Observability must not be startup-critical.
    return
