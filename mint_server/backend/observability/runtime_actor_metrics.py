from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


VLLM_SCALAR_FIELDS = (
    "scheduler_waiting_requests",
    "scheduler_running_requests",
    "scheduler_kv_cache_usage_ratio",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "prefix_cache_hit_ratio",
    "preemptions_total",
)

VLLM_STEMS = (
    "queue_time_s",
    "prefill_time_s",
    "decode_time_s",
    "time_per_output_token_s",
    "scheduled_tokens_iter",
    "scheduled_new_requests_iter",
    "scheduled_cached_requests_iter",
    "prefill_requests_iter",
    "decode_requests_iter",
    "prompt_tokens_iter",
    "generation_tokens_iter",
    "time_to_first_token_s",
    "inter_token_latency_s",
    "executor_execute_model_s",
    "worker_execute_model_s",
    "seq_slot_wait_s",
    "generate_lock_wait_s",
    "engine_read_lock_wait_s",
    "add_request_wait_s",
    "add_request_exec_s",
    "first_token_observed_s",
)

VLLM_STEM_SUFFIXES = ("sum", "count", "max", "p50_recent", "p95_recent")
VLLM_SNAPSHOT_SUFFIX_FIELDS = {
    "sum": "total",
    "count": "count",
    "max": "max",
    "p50_recent": "p50_recent",
    "p95_recent": "p95_recent",
}


def metric_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def runtime_metric_attrs(**extra: object) -> dict[str, str]:
    attrs = {
        "deployment.env": os.getenv("MINT_DEPLOYMENT_ENV", "").strip(),
        "mint.cluster_id": os.getenv("MINT_CLUSTER_ID", "").strip(),
        "ray_namespace": os.getenv("MINT_RAY_NAMESPACE", "").strip(),
    }
    for key, value in extra.items():
        text = str(value if value is not None else "").strip()
        if text:
            attrs[key] = text
    return {key: value for key, value in attrs.items() if value}


def current_ray_actor_name(default: str = "unknown") -> str:
    try:
        import ray

        name = ray.get_runtime_context().get_actor_name()
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return default


def _register_observable_gauge(meter: Any, name: str, callback: Callable, *, unit: str | None = None) -> None:
    kwargs: dict[str, Any] = {"callbacks": [callback]}
    if unit:
        kwargs["unit"] = unit
    meter.create_observable_gauge(name, **kwargs)


def _snapshot_value(snapshot_fn: Callable[[], dict[str, Any]], field: str) -> float | None:
    try:
        snapshot = snapshot_fn()
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    return metric_number(snapshot.get(field))


def init_vllm_runtime_otel_metrics(
    *,
    snapshot_fn: Callable[[], dict[str, Any]],
    actor_name: str,
    base_model: str,
) -> bool:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return False
    label_base_model = (os.getenv("MINT_VLLM_BASE_MODEL_NAME") or "").strip() or str(base_model)
    try:
        from opentelemetry import metrics
        from opentelemetry.metrics import Observation

        meter = metrics.get_meter("mint.vllm_runtime_actor")

        def _attrs() -> dict[str, str]:
            return runtime_metric_attrs(actor_name=actor_name, base_model=label_base_model)

        def _observe(field: str):
            def _callback(_options):
                value = _snapshot_value(snapshot_fn, field)
                if value is None:
                    return []
                return [Observation(value, _attrs())]

            return _callback

        for field in VLLM_SCALAR_FIELDS:
            _register_observable_gauge(meter, f"mint_vllm_{field}", _observe(field))
        for stem in VLLM_STEMS:
            for suffix in VLLM_STEM_SUFFIXES:
                field = f"{stem}_{VLLM_SNAPSHOT_SUFFIX_FIELDS[suffix]}"
                _register_observable_gauge(
                    meter,
                    f"mint_vllm_{stem}_{suffix}",
                    _observe(field),
                    unit="s" if stem.endswith("_s") and suffix not in {"count"} else None,
                )
        return True
    except Exception:
        return False


def init_megatron_group_runtime_otel_metrics(
    *,
    snapshot_fn: Callable[[], dict[str, Any]],
    actor_name: str,
    base_model: str,
) -> bool:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return False
    try:
        from opentelemetry import metrics
        from opentelemetry.metrics import Observation

        meter = metrics.get_meter("mint.megatron_runtime_actor")

        def _attrs() -> dict[str, str]:
            return runtime_metric_attrs(actor_name=actor_name, base_model=base_model)

        def _observe(field: str):
            def _callback(_options):
                value = _snapshot_value(snapshot_fn, field)
                if value is None:
                    return []
                return [Observation(value, _attrs())]

            return _callback

        for field in (
            "active_sessions",
            "session_unknown",
            "session_step",
            "learning_rate",
            "training_requests_total",
            "input_tokens_total",
            "output_tokens_total",
        ):
            _register_observable_gauge(
                meter,
                f"mint_megatron_{field}",
                _observe(field),
            )
        return True
    except Exception:
        return False


def init_megatron_rank_runtime_otel_metrics(
    *,
    snapshot_fn: Callable[[], dict[str, Any]],
    actor_name: str,
    base_model: str,
    rank: int,
) -> bool:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return False
    try:
        from opentelemetry import metrics
        from opentelemetry.metrics import Observation

        meter = metrics.get_meter("mint.megatron_rank_worker")

        def _attrs() -> dict[str, str]:
            return runtime_metric_attrs(actor_name=actor_name, base_model=base_model, rank=rank)

        def _observe(field: str):
            def _callback(_options):
                value = _snapshot_value(snapshot_fn, field)
                if value is None:
                    return []
                return [Observation(value, _attrs())]

            return _callback

        for field in (
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "gpu_memory_fragmentation_bytes",
        ):
            _register_observable_gauge(
                meter,
                f"mint_megatron_{field}",
                _observe(field),
                unit="By",
            )
        return True
    except Exception:
        return False
