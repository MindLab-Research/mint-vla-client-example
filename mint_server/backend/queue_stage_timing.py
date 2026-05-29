from __future__ import annotations

from typing import Any


QUEUE_STAGE_TIMING_FIELDS = (
    "scheduler_wait_s",
    "executor_wait_s",
    "lora_s",
    "vllm_generate_s",
    "finalization_s",
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out < 0.0:
        return 0.0
    return out


def _first_number(meta: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(meta.get(key))
        if value is not None:
            return value
    return None


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def build_queue_stage_timing(
    meta: dict[str, Any] | None,
    *,
    now: float | None = None,
) -> dict[str, float | int | None]:
    """Normalize queue/runtime metadata into the stable Issue 648 timing buckets."""
    data = dict(meta or {})
    now_s = _number(now)

    queued_at = _first_number(data, "queued_at", "created_at")
    dequeue_at = _first_number(data, "dequeue_at", "running_at", "leased_at")
    executor_started_at = _first_number(data, "executor_started_at", "executor_start_at")
    executor_done_at = _first_number(data, "executor_done_at")
    finalization_started_at = _first_number(data, "finalization_started_at")
    finalization_done_at = _first_number(data, "finalization_done_at", "done_at", "failed_at")

    if dequeue_at is None and str(data.get("queue_state") or "") == "queued":
        dequeue_at = now_s
    if executor_started_at is None and str(data.get("queue_state") or "") == "running":
        executor_started_at = now_s
    if finalization_done_at is None and str(data.get("stage") or "") == "finalizing":
        finalization_done_at = now_s

    scheduler_wait_s = _first_number(data, "scheduler_wait_s", "queue_wait_s")
    if scheduler_wait_s is None:
        scheduler_wait_s = _duration(queued_at, dequeue_at)

    executor_wait_s = _first_number(data, "executor_wait_s")
    if executor_wait_s is None:
        executor_wait_s = _duration(dequeue_at, executor_started_at)

    lora_s = _first_number(data, "lora_s", "lora_load_s")
    vllm_generate_s = _first_number(data, "vllm_generate_s", "generate_s")

    finalization_s = _first_number(data, "finalization_s")
    if finalization_s is None:
        finalization_s = _duration(finalization_started_at or executor_done_at, finalization_done_at)

    terminal_at = finalization_done_at or executor_done_at or now_s
    total_observed_s = _duration(queued_at, terminal_at)

    return {
        "schema_version": 1,
        "scheduler_wait_s": _round(scheduler_wait_s),
        "executor_wait_s": _round(executor_wait_s),
        "lora_s": _round(lora_s),
        "vllm_generate_s": _round(vllm_generate_s),
        "finalization_s": _round(finalization_s),
        "total_observed_s": _round(total_observed_s),
    }


def attach_queue_stage_timing(payload: object, timing: dict[str, Any] | None) -> object:
    if not isinstance(payload, dict) or not isinstance(timing, dict):
        return payload
    return {**payload, "queue_stage_timing": dict(timing)}
