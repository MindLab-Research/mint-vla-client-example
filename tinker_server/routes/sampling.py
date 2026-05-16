"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

import asyncio
import copy
import array
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, Response

from ..config import config as server_config
from ..backend.task_state_store import FutureStatus, TaskStateStoreUnavailableError, task_state_futures
from ..backend.model_work_admission import enqueue_model_work
from ..gateway_auth import GatewayAuthContext, build_billing_auth_context
from ..logging_context import (
    classify_failure_reason,
    get_otel_tracer,
    record_sampling_admission_metric,
    run_async_with_otel_span,
    set_request_id,
    start_as_current_span,
)
from ..model_access_control import can_access_model, get_access_denied_error
from ..queue_priority import merge_queue_priority_extra
from ..models.types import (
    ComputeLogprobsRequest,
    ComputeLogprobsResponse,
    ModelInput,
    SampleRequest,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    UntypedAPIFuture,
)
from ..sampling_utils import normalize_prompt_logprobs_for_tinker, sampled_sequence_from_result
from ..usage_store import UsageEvent, schedule_usage_events

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None

_SAMPLING_BACKPRESSURE_HEADER = "X-Tinker-Sampling-Backpressure"
_MAX_INFLIGHT_SAMPLE_TASKS = int(server_config.sampling_max_inflight_sample_tasks)
_MAX_CONCURRENT_SAMPLES_PER_REQUEST = int(server_config.sampling_max_concurrent_samples_per_request)
_inflight_sample_tasks = 0

_SAMPLE_COALESCE = bool(server_config.sampling_sample_coalesce)
_SAMPLE_COALESCE_WINDOW_MS = float(server_config.sampling_sample_coalesce_window_ms)
_SAMPLE_COALESCE_MAX_BATCH = int(server_config.sampling_sample_coalesce_max_batch)
_SAMPLE_COALESCE_MAX_SAMPLES = int(server_config.sampling_sample_coalesce_max_samples)
_sample_coalesce_lock = asyncio.Lock()
_sample_coalesce_groups: dict[tuple, dict] = {}
_coalesced_abort_aliases_guard = asyncio.Lock()
_coalesced_abort_aliases: dict[str, str] = {}

_lora_load_locks_guard = asyncio.Lock()
_lora_load_locks: dict[str, asyncio.Lock] = {}

_ASAMPLE_ROUTE = "/api/v1/asample"
_COMPUTE_LOGPROBS_ROUTE = "/api/v1/compute_logprobs"
_SAMPLE_ONCE_ROUTE = "sample_once"


def _active_session_manager() -> SessionManager | None:
    if session_manager is not None:
        return session_manager
    try:
        from . import service as service_route
    except Exception:
        return None
    manager = getattr(service_route, "session_manager", None)
    if manager is None:
        return None
    return manager


@dataclass(frozen=True)
class SamplingSessionSnapshot:
    """Request-scope immutable sampling metadata."""

    session_id: str
    uses_multi_lora: bool
    uses_base_model: bool
    base_model: str | None
    lora_rank: int
    adapter_path: str | None
    lora_loaded: bool
    lora_int_id: int | None
    metadata_version: int


async def _get_lora_load_lock(session_id: str) -> asyncio.Lock:
    async with _lora_load_locks_guard:
        lock = _lora_load_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _lora_load_locks[session_id] = lock
        return lock


async def _drop_lora_load_lock(session_id: str) -> None:
    async with _lora_load_locks_guard:
        _lora_load_locks.pop(session_id, None)


async def _lora_load_lock_count() -> int:
    async with _lora_load_locks_guard:
        return len(_lora_load_locks)


def _resolve_billing_model(session_id: str) -> str:
    snapshot = _get_sampling_snapshot(session_id)
    if snapshot is None:
        return session_id
    return snapshot.base_model or session_id


def _build_sampling_usage_label(*, model: str, route: str, dimension: str) -> str:
    return f"model={model},route={route},dimension={dimension}"


def build_sample_once_usage_events(
    *,
    session_id: str,
    token_ids: list[int],
    sequence,
    http_request: Request,
    request_id: str,
) -> list[UsageEvent]:
    billing_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
    if billing_auth is None:
        return []
    label_model = _resolve_billing_model(session_id)
    return [
        UsageEvent(
            account_id=billing_auth.account_id,
            apikey_id=billing_auth.apikey_id,
            charge_item="sampling",
            quantity=len(token_ids),
            request_id=billing_auth.request_id,
            label=_build_sampling_usage_label(
                model=label_model,
                route="sampling.sample_once",
                dimension="prefill",
            ),
        ),
        UsageEvent(
            account_id=billing_auth.account_id,
            apikey_id=billing_auth.apikey_id,
            charge_item="sampling",
            quantity=len(sequence.tokens),
            request_id=billing_auth.request_id,
            label=_build_sampling_usage_label(
                model=label_model,
                route="sampling.sample_once",
                dimension="sample",
            ),
        ),
    ]


def _record_vllm_workload_start(*, actor_name: str | None, base_model: str, op: str) -> None:
    from ..backend.runtime_observability import runtime_observability

    runtime_observability.begin_vllm_request(actor_name=actor_name, base_model=base_model, op=op)


def _vllm_request_observation(results: list[object], generated_tokens: int) -> dict[str, float | None]:
    if not results:
        return {
            "ttft_s": None,
            "tpot_s": None,
        }
    first = results[0]
    ttft_s = getattr(first, "timing_first_tok_s", None)
    total_s = getattr(first, "timing_total_s", None)
    tpot_s = None
    if ttft_s is not None and total_s is not None and int(generated_tokens) > 1:
        decode_s = max(0.0, float(total_s) - float(ttft_s))
        tpot_s = decode_s / float(max(1, int(generated_tokens) - 1))
    return {
        "ttft_s": float(ttft_s) if ttft_s is not None else None,
        "tpot_s": float(tpot_s) if tpot_s is not None else None,
    }


def _record_vllm_workload_finish(
    *,
    actor_name: str | None,
    base_model: str,
    op: str,
    status: str,
    prompt_tokens: int,
    generated_tokens: int,
    started_at: float,
    ttft_s: float | None = None,
    tpot_s: float | None = None,
) -> None:
    from ..backend.runtime_observability import runtime_observability

    runtime_observability.finish_vllm_request(
        actor_name=actor_name,
        base_model=base_model,
        op=op,
        status=status,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        duration_s=max(0.0, time.perf_counter() - float(started_at)),
        ttft_s=ttft_s,
        tpot_s=tpot_s,
    )


def _snapshot_from_legacy_getters(session_id: str) -> SamplingSessionSnapshot | None:
    manager = _active_session_manager()
    if manager is None:
        return None
    is_multi_lora = bool(manager.is_multi_lora_session(session_id))
    get_engine = getattr(manager, "get_engine", None)
    if not is_multi_lora and callable(get_engine):
        if get_engine(session_id) is None:
            return None

    get_base_model = getattr(manager, "get_session_base_model", None)
    get_lora_rank = getattr(manager, "get_session_lora_rank", None)
    get_adapter_path = getattr(manager, "get_session_adapter_path", None)
    get_loaded = getattr(manager, "is_session_lora_loaded", None)
    get_lora_int_id = getattr(manager, "get_session_lora_int_id", None)
    is_base_model_session = getattr(manager, "is_base_model_session", None)
    get_metadata_version = getattr(manager, "get_session_metadata_version", None)

    base_model = get_base_model(session_id) if callable(get_base_model) else None
    lora_rank = int(get_lora_rank(session_id) or 0) if callable(get_lora_rank) else 0
    adapter_path = get_adapter_path(session_id) if callable(get_adapter_path) else None
    lora_loaded = bool(get_loaded(session_id)) if callable(get_loaded) else False
    lora_int_id = get_lora_int_id(session_id) if callable(get_lora_int_id) else None
    uses_base_model = bool(is_base_model_session(session_id)) if callable(is_base_model_session) else False
    metadata_version = int(get_metadata_version(session_id) or 1) if callable(get_metadata_version) else 1

    return SamplingSessionSnapshot(
        session_id=session_id,
        uses_multi_lora=is_multi_lora,
        uses_base_model=uses_base_model,
        base_model=base_model,
        lora_rank=lora_rank,
        adapter_path=adapter_path,
        lora_loaded=lora_loaded,
        lora_int_id=None if lora_int_id is None else int(lora_int_id),
        metadata_version=max(1, metadata_version),
    )


def _coerce_sampling_snapshot(raw: object, session_id: str) -> SamplingSessionSnapshot | None:
    if raw is None:
        return None
    return SamplingSessionSnapshot(
        session_id=str(getattr(raw, "session_id", session_id) or session_id),
        uses_multi_lora=bool(getattr(raw, "uses_multi_lora", False)),
        uses_base_model=bool(getattr(raw, "uses_base_model", False)),
        base_model=getattr(raw, "base_model", None),
        lora_rank=int(getattr(raw, "lora_rank", 0) or 0),
        adapter_path=getattr(raw, "adapter_path", None),
        lora_loaded=bool(getattr(raw, "lora_loaded", False)),
        lora_int_id=(
            None
            if getattr(raw, "lora_int_id", None) is None
            else int(getattr(raw, "lora_int_id"))
        ),
        metadata_version=max(1, int(getattr(raw, "metadata_version", 1) or 1)),
    )


def _get_sampling_snapshot(session_id: str) -> SamplingSessionSnapshot | None:
    manager = _active_session_manager()
    if manager is None:
        return None
    get_snapshot = getattr(manager, "get_sampling_session_snapshot", None)
    if callable(get_snapshot):
        snapshot = _coerce_sampling_snapshot(get_snapshot(session_id), session_id)
        if snapshot is not None:
            return snapshot
    return _snapshot_from_legacy_getters(session_id)


async def _async_get_detached_sampling_snapshot(session_id: str) -> SamplingSessionSnapshot | None:
    try:
        from ..backend.sampling_session_store import async_get_sampling_session_info

        info = await async_get_sampling_session_info(session_id)
    except Exception:
        manager = _active_session_manager()
        if manager is not None:
            return _get_sampling_snapshot(session_id)
        return None
    if not isinstance(info, dict):
        manager = _active_session_manager()
        if manager is not None:
            return _get_sampling_snapshot(session_id)
        return None
    return SamplingSessionSnapshot(
        session_id=str(info.get("session_id") or session_id),
        uses_multi_lora=True,
        uses_base_model=bool(info.get("uses_base_model")),
        base_model=info.get("base_model"),
        lora_rank=int(info.get("lora_rank") or 0),
        adapter_path=info.get("adapter_path"),
        lora_loaded=bool(info.get("lora_loaded")),
        lora_int_id=None if info.get("lora_int_id") is None else int(info.get("lora_int_id")),
        metadata_version=max(1, int(info.get("metadata_version") or 1)),
    )


def _has_local_sampling_session(session_id: str) -> bool:
    manager = _active_session_manager()
    if manager is None:
        return False
    if manager.is_multi_lora_session(session_id):
        return True
    return manager.get_engine(session_id) is not None


async def _drop_local_sampling_session(session_id: str) -> None:
    manager = _active_session_manager()
    if manager is None:
        return
    end_session = getattr(manager, "end_session", None)
    if not callable(end_session):
        return
    try:
        await end_session(session_id)
        logger.info("[sampling restore] dropped stale local sampler session_id=%s after detached miss", session_id)
    except Exception as e:
        logger.warning("Failed to drop stale local sampler session_id=%s: %s", session_id, e)


async def _restore_local_sampling_session_if_needed(session_id: str) -> bool:
    """Best-effort restore of detached sampler metadata on this API worker."""
    snapshot = _get_sampling_snapshot(session_id)
    if snapshot is not None:
        refreshed = await _refresh_sampling_session_if_stale(session_id, snapshot)
        return refreshed is not None
    manager = _active_session_manager()
    if manager is None:
        return False

    restore_session = getattr(manager, "restore_sampling_session", None)
    if not callable(restore_session):
        return _has_local_sampling_session(session_id)

    try:
        from ..backend.sampling_session_store import async_get_sampling_session_info

        info = await async_get_sampling_session_info(session_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable") from e

    if not isinstance(info, dict):
        return False

    try:
        restored = bool(restore_session(info))
    except ValueError:
        restored = _has_local_sampling_session(session_id)

    if restored:
        logger.info("[sampling restore] restored detached sampler session_id=%s", session_id)
    return restored or _has_local_sampling_session(session_id)


async def _refresh_sampling_session_if_stale(
    session_id: str,
    snapshot: SamplingSessionSnapshot,
) -> SamplingSessionSnapshot | None:
    manager = _active_session_manager()
    if manager is None or not snapshot.uses_multi_lora:
        return snapshot
    try:
        from ..backend.sampling_session_store import async_get_sampling_session_info

        info = await async_get_sampling_session_info(session_id)
    except Exception as e:
        logger.debug("Sampling session refresh skipped session_id=%s: %s", session_id, e)
        return snapshot

    if not isinstance(info, dict):
        await _drop_local_sampling_session(session_id)
        return None

    incoming_version = max(1, int(info.get("metadata_version") or 1))
    if incoming_version <= int(snapshot.metadata_version):
        return snapshot

    try:
        restored = bool(manager.restore_sampling_session(info))
    except ValueError:
        restored = False
    if not restored:
        return snapshot
    refreshed = _get_sampling_snapshot(session_id)
    return refreshed or snapshot


async def _enqueue_sampling_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
    session_id: str | None = None,
    base_model: str | None = None,
) -> object:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        return await enqueue_coro

    with tracer.start_as_current_span(f"{op}.enqueue") as span:
        span.set_attribute("component", "routes.sampling")
        span.set_attribute("op", str(op))
        span.set_attribute("request_id", str(request_id))
        if session_id:
            span.set_attribute("sampling_session_id", str(session_id))
        if base_model:
            span.set_attribute("base_model", str(base_model))
        span.add_event(
            "task_state_futures_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        out = await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round((time.perf_counter() - enqueue_start_s) * 1000.0, 3),
                "route_elapsed_ms": round((time.perf_counter() - route_start_s) * 1000.0, 3),
            },
        )
        return out


async def _ensure_session_lora_loaded(
    engine,
    session_id: str,
    *,
    snapshot: SamplingSessionSnapshot | None = None,
) -> None:
    manager = _active_session_manager()
    if manager is None:
        raise RuntimeError("Session manager not initialized")

    snap = snapshot or _get_sampling_snapshot(session_id)
    if snap is None:
        raise RuntimeError(f"No sampling session metadata found for session {session_id}")
    if int(snap.lora_rank) <= 0:
        return

    if snap.lora_loaded and snap.lora_int_id is not None:
        return
    if snap.lora_loaded and snap.lora_int_id is None:
        logger.warning(
            "Sampling session %s marked loaded without lora_int_id; reloading adapter from path",
            session_id,
        )

    adapter_path = snap.adapter_path
    if not adapter_path:
        raise RuntimeError(f"Session {session_id} has lora_rank={snap.lora_rank} but no adapter_path")

    lock = await _get_lora_load_lock(session_id)
    with start_as_current_span(
        "sampling.ensure_session_lora_loaded.lock_and_reload",
        component="routes.sampling",
        op="sampling.ensure_session_lora_loaded.lock_and_reload",
        attributes={
            "sampling_session_id": str(session_id),
            "base_model": str(snap.base_model) if snap.base_model else None,
            "lora_rank": int(snap.lora_rank),
            "lora_loaded_before": bool(snap.lora_loaded),
        },
    ):
        async with lock:
            refreshed = _get_sampling_snapshot(session_id)
            if refreshed is not None and refreshed.lora_loaded and refreshed.lora_int_id is not None:
                return
            if refreshed is not None and refreshed.adapter_path:
                adapter_path = refreshed.adapter_path

            # Prefer path-based loading to avoid sending large tensors through Ray.
            add_from_path = getattr(engine, "add_lora_for_session_from_path", None)
            if add_from_path is None:
                raise RuntimeError(f"Engine for session {session_id} does not support add_lora_for_session_from_path()")

            load_snapshot = refreshed or snap
            with start_as_current_span(
                "sampling.ensure_session_lora_loaded.add_from_path",
                component="routes.sampling",
                op="sampling.ensure_session_lora_loaded.add_from_path",
                attributes={
                    "sampling_session_id": str(session_id),
                    "base_model": str(load_snapshot.base_model) if load_snapshot.base_model else None,
                    "lora_rank": int(load_snapshot.lora_rank),
                    "adapter_path": str(adapter_path),
                    "lora_loaded_before": bool(load_snapshot.lora_loaded),
                },
            ):
                lora_int_id = await add_from_path(sampling_session_id=session_id, lora_path=adapter_path)
            manager.mark_session_lora_loaded(session_id, True, lora_int_id=lora_int_id)


async def _register_coalesced_abort_aliases(waiters: list[tuple], engine_request_id: str) -> None:
    async with _coalesced_abort_aliases_guard:
        for _fut, _ns, request_id in waiters:
            _coalesced_abort_aliases[request_id] = engine_request_id


async def _unregister_coalesced_abort_aliases(waiters: list[tuple], engine_request_id: str) -> None:
    async with _coalesced_abort_aliases_guard:
        for _fut, _ns, request_id in waiters:
            if _coalesced_abort_aliases.get(request_id) == engine_request_id:
                _coalesced_abort_aliases.pop(request_id, None)


async def _resolve_abort_request_id(request_id: str) -> str:
    async with _coalesced_abort_aliases_guard:
        return _coalesced_abort_aliases.get(request_id, request_id)


async def _abort_engine_request(engine, request_id: str) -> None:
    if engine is None:
        return
    abort = getattr(engine, "abort_request", None)
    if abort is None:
        return
    abort_request_id = await _resolve_abort_request_id(request_id)
    try:
        maybe_awaitable = abort(abort_request_id)
        if asyncio.isfuture(maybe_awaitable) or asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable
    except Exception as e:
        logger.warning(
            f"Best-effort abort failed: request_id={request_id} abort_request_id={abort_request_id} "
            f"{type(e).__name__}: {e}"
        )


async def _await_with_external_fail_abort(*, engine, request_id: str, awaitable):
    task = asyncio.create_task(awaitable)
    started = time.monotonic()
    last_log = started
    await_timeout_s = float(os.environ.get("MINT_SAMPLE_AWAIT_TIMEOUT_S", "0"))
    logger.info(f"[sample await start] request_id={request_id} await_timeout_s={await_timeout_s}")
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if done:
                elapsed = time.monotonic() - started
                logger.info(f"[sample await done] request_id={request_id} elapsed_s={elapsed:.1f}")
                return await task
            try:
                status = await task_state_futures.async_get_status(request_id)
            except KeyError:
                status = FutureStatus.PENDING
            now = time.monotonic()
            elapsed = now - started
            if now - last_log >= 30.0:
                logger.info(f"[sample await progress] request_id={request_id} elapsed_s={elapsed:.1f} future_status={status.value}")
                last_log = now
            if await_timeout_s > 0 and elapsed >= await_timeout_s:
                await _abort_engine_request(engine, request_id)
                task.cancel()
                try:
                    await task
                except Exception:
                    pass
                raise RuntimeError(
                    f"request_id={request_id} timed out in _await_with_external_fail_abort "
                    f"after {elapsed:.1f}s (MINT_SAMPLE_AWAIT_TIMEOUT_S={await_timeout_s})"
                )
            if status != FutureStatus.PENDING:
                await _abort_engine_request(engine, request_id)
                task.cancel()
                try:
                    await task
                except Exception:
                    pass
                raise RuntimeError(f"request_id={request_id} canceled due to future_status={status.value}")
    except asyncio.CancelledError:
        await _abort_engine_request(engine, request_id)
        task.cancel()
        try:
            await task
        except Exception:
            pass
        raise


def _prompt_fingerprint(token_ids: list[int]) -> bytes:
    # 32k token prompts are common; collisions must be negligible but hashing cost is amortized by prefill.
    a = array.array("I", token_ids)
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest()


def _payload_hash(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _model_work_domain_key(base_model: str) -> str:
    return f"vllm:{str(base_model).strip()}"


def _model_work_affinity_group(snapshot: SamplingSessionSnapshot) -> str:
    if snapshot.uses_base_model:
        return f"base:{snapshot.base_model}"
    return f"lora:{snapshot.session_id}:generation:{int(snapshot.metadata_version)}"


def _deterministic_request_id(session_id: str, seq_id: int) -> str:
    seed = f"{session_id}:{int(seq_id)}".encode("utf-8")
    digest = hashlib.blake2b(seed, digest_size=16).hexdigest()
    return f"sample_{digest}"


def _stop_key(stop: object | None) -> object:
    if stop is None:
        return None
    if isinstance(stop, str):
        return ("s", stop)
    if isinstance(stop, list):
        if not stop:
            return ("empty",)
        if all(isinstance(x, int) for x in stop):
            return ("ti", tuple(int(x) for x in stop))
        if all(isinstance(x, str) for x in stop):
            return ("ts", tuple(str(x) for x in stop))
        raise ValueError(f"stop must be list[int] or list[str], got mixed: {stop!r}")
    raise TypeError(f"stop must be None, str, list[str], or list[int]; got {type(stop)}")


async def _coalesced_generate(
    *,
    engine,
    sampling_session_id: str,
    prompt_ids: list[int],
    request_id: str,
    num_samples: int,
    max_tokens: int,
    stop: object | None,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1 (got {num_samples})")
    coalesce_identity: str = sampling_session_id
    try:
        registry = getattr(engine, "registry", None)
        if registry is not None:
            lora_id = await registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                coalesce_identity = f"lora:{lora_id}"
            elif sampling_session_id == "__base__":
                coalesce_identity = "__base__"
    except Exception:
        coalesce_identity = sampling_session_id

    key = (
        coalesce_identity,
        _prompt_fingerprint(prompt_ids),
        int(max_tokens),
        _stop_key(stop),
        float(temperature),
        int(top_k),
        float(top_p),
    )

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    delay_s: float | None = None
    do_flush_now = False
    async with _sample_coalesce_lock:
        g = _sample_coalesce_groups.get(key)
        need = int(num_samples)
        if need > _SAMPLE_COALESCE_MAX_SAMPLES:
            raise ValueError(
                f"coalesce: num_samples={need} exceeds TINKER_SAMPLE_COALESCE_MAX_SAMPLES={_SAMPLE_COALESCE_MAX_SAMPLES}"
            )
        if g is None:
            g = {
                "engine": engine,
                "sampling_session_id": sampling_session_id,
                "prompt_ids": prompt_ids,
                "max_tokens": max_tokens,
                "stop": stop,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "leader_request_id": request_id,
                "waiters": [],
                "total_samples": 0,
                "flush_task": None,
            }
            _sample_coalesce_groups[key] = g
        # If this request would exceed the per-group total sample cap, flush the existing group
        # immediately (without including this request), then start a fresh group.
        if int(g["total_samples"]) + need > _SAMPLE_COALESCE_MAX_SAMPLES and g["waiters"]:
            _sample_coalesce_groups.pop(key, None)
            flush_task = g.get("flush_task")
            if flush_task is not None:
                try:
                    flush_task.cancel()
                except Exception:
                    pass
            asyncio.create_task(_flush_group(g))
            g = {
                "engine": engine,
                "sampling_session_id": sampling_session_id,
                "prompt_ids": prompt_ids,
                "max_tokens": max_tokens,
                "stop": stop,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "leader_request_id": request_id,
                "waiters": [],
                "total_samples": 0,
                "flush_task": None,
            }
            _sample_coalesce_groups[key] = g

        g["waiters"].append((fut, need, request_id))
        g["total_samples"] = int(g["total_samples"]) + need
        do_flush_now = (len(g["waiters"]) >= _SAMPLE_COALESCE_MAX_BATCH) or (int(g["total_samples"]) >= _SAMPLE_COALESCE_MAX_SAMPLES)
        if g["flush_task"] is None:
            delay_s = 0.0 if do_flush_now else max(0.0, _SAMPLE_COALESCE_WINDOW_MS / 1000.0)
            g["flush_task"] = asyncio.create_task(_flush_coalesced_group(key, delay_s))
        logger.info(
            f"[coalesce queue] request_id={request_id} sampling_session_id={sampling_session_id} coalesce_identity={coalesce_identity} "
            f"waiters={len(g['waiters'])} total_samples={int(g['total_samples'])} "
            f"do_flush_now={do_flush_now} delay_s={delay_s if delay_s is not None else -1.0:.3f}"
        )

    res = await fut
    if not isinstance(res, list):
        raise RuntimeError(f"coalesce: expected list result, got {type(res)}")
    if len(res) != int(num_samples):
        raise RuntimeError(f"coalesce: expected {num_samples} samples, got {len(res)}")
    return res


async def _flush_group(g: dict) -> None:
    waiters = list(g.get("waiters") or [])
    if not waiters:
        return
    vllm_request_id: str | None = None
    try:
        total = sum(int(ns) for _fut, ns, _rid in waiters)
        if total == 1 and int(waiters[0][1]) == 1:
            res = await g["engine"].generate(
                sampling_session_id=g["sampling_session_id"],
                prompt_ids=g["prompt_ids"],
                request_id=g["leader_request_id"],
                max_tokens=g["max_tokens"],
                stop=g.get("stop"),
                temperature=g["temperature"],
                top_k=g["top_k"],
                top_p=g["top_p"],
                logprobs=True,
            )
            fut0, _ns0, _rid0 = waiters[0]
            if not fut0.done():
                fut0.set_result([res])
            return

        if float(g["temperature"]) == 0.0:
            # vLLM forces greedy sampling to n=1, so identical coalesced requests
            # must share one engine.generate() result and fan it back out locally.
            vllm_request_id = f"{g['leader_request_id']}_coalesced"
            rid_ns = ",".join(f"{rid}:{int(ns)}" for _fut, ns, rid in waiters)
            logger.info(
                f"[coalesce flush greedy] leader={g['leader_request_id']} vllm_req={vllm_request_id} "
                f"waiters={len(waiters)} total_samples={total} rid_ns={rid_ns}"
            )
            await _register_coalesced_abort_aliases(waiters, vllm_request_id)
            try:
                res = await g["engine"].generate(
                    sampling_session_id=g["sampling_session_id"],
                    prompt_ids=g["prompt_ids"],
                    request_id=vllm_request_id,
                    max_tokens=g["max_tokens"],
                    stop=g.get("stop"),
                    temperature=g["temperature"],
                    top_k=g["top_k"],
                    top_p=g["top_p"],
                    logprobs=True,
                )
            finally:
                await _unregister_coalesced_abort_aliases(waiters, vllm_request_id)
            for fut, ns, _rid in waiters:
                if not fut.done():
                    fut.set_result([copy.deepcopy(res) for _ in range(int(ns))])
            return

        vllm_request_id = f"{g['leader_request_id']}_coalesced"
        rid_ns = ",".join(f"{rid}:{int(ns)}" for _fut, ns, rid in waiters)
        logger.info(
            f"[coalesce flush] leader={g['leader_request_id']} vllm_req={vllm_request_id} "
            f"waiters={len(waiters)} total_samples={total} rid_ns={rid_ns}"
        )
        await _register_coalesced_abort_aliases(waiters, vllm_request_id)
        try:
            results = await g["engine"].generate_many(
                sampling_session_id=g["sampling_session_id"],
                prompt_ids=g["prompt_ids"],
                request_id=vllm_request_id,
                num_samples=total,
                max_tokens=g["max_tokens"],
                stop=g.get("stop"),
                temperature=g["temperature"],
                top_k=g["top_k"],
                top_p=g["top_p"],
                logprobs=True,
            )
        finally:
            await _unregister_coalesced_abort_aliases(waiters, vllm_request_id)
        if len(results) != total:
            raise RuntimeError(f"coalesce: got {len(results)} results for total_samples={total}")

        cur = 0
        for fut, ns, _rid in waiters:
            n = int(ns)
            chunk = results[cur : cur + n]
            cur += n
            if not fut.done():
                fut.set_result(chunk)
    except Exception as e:
        if vllm_request_id is not None:
            await _abort_engine_request(g.get("engine"), g["leader_request_id"])
        for fut, _ns, _rid in waiters:
            if not fut.done():
                fut.set_exception(e)


async def _flush_coalesced_group(key: tuple, delay_s: float) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    async with _sample_coalesce_lock:
        g = _sample_coalesce_groups.pop(key, None)
    if g is None:
        logger.warning(f"[coalesce flush task] missing group delay_s={delay_s:.3f}")
        return
    logger.info(
        f"[coalesce flush task] leader={g.get('leader_request_id')} waiters={len(g.get('waiters') or [])} "
        f"total_samples={int(g.get('total_samples') or 0)} delay_s={delay_s:.3f}"
    )
    await _flush_group(g)


def _normalize_topk_prompt_logprobs(
    raw: list[dict[int, float] | list[tuple[int, float]] | None],
    k: int,
) -> list[list[tuple[int, float]] | None]:
    out: list[list[tuple[int, float]] | None] = []
    for i, entry in enumerate(raw):
        if entry is None:
            out.append(None)
            continue
        if isinstance(entry, dict):
            pairs = [(int(tok), float(lp)) for tok, lp in entry.items()]
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            out.append(pairs[:k])
            continue
        if isinstance(entry, list):
            pairs: list[tuple[int, float]] = []
            for pair in entry:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"topk_prompt_logprobs[{i}] has invalid pair: {pair!r}")
                tok, lp = pair
                pairs.append((int(tok), float(lp)))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            out.append(pairs[:k])
            continue
        raise ValueError(f"topk_prompt_logprobs[{i}] has invalid entry type: {type(entry)}")
    return out


def _should_backpressure(http_request: Request) -> bool:
    if http_request.headers.get(_SAMPLING_BACKPRESSURE_HEADER) != "1":
        return False
    return _inflight_sample_tasks >= _MAX_INFLIGHT_SAMPLE_TASKS


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state (set by auth middleware)."""
    user_data = getattr(request.state, "user_data", None)
    if user_data:
        return user_data.get("user_id")
    return None


def _get_apikey_id(
    request: Request,
    *,
    billing_auth: GatewayAuthContext | None = None,
) -> str | None:
    if billing_auth is not None and billing_auth.apikey_id:
        return str(billing_auth.apikey_id)
    user_data = getattr(request.state, "user_data", None)
    if isinstance(user_data, dict):
        apikey_id = str(user_data.get("apikey_id") or user_data.get("key_id") or "").strip()
        if apikey_id:
            return apikey_id
    return None


def _get_asample_throttle_identity(
    request: Request,
    *,
    billing_auth: GatewayAuthContext | None = None,
) -> tuple[str | None, str | None, str]:
    apikey_id = _get_apikey_id(request, billing_auth=billing_auth)
    if apikey_id:
        return f"apikey:{apikey_id}", apikey_id, "api_key"
    user_id = _get_user_id(request)
    if user_id:
        return f"user:{user_id}", None, "user"
    return None, None, "anonymous"


async def _persist_usage_events(*, auth_ctx: GatewayAuthContext, events: list[UsageEvent]) -> None:
    _ = auth_ctx
    schedule_usage_events(events)


@router.post("/asample")
async def asample(
    request: SampleRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Submit an async sampling request.

    The request is processed in the background. Use /retrieve_future
    with the returned request_id to get results.
    """
    route_start_s = time.perf_counter()
    # Gateway forwarding: if this sampling_session_id was created upstream, proxy the
    # request and return a gateway-encoded request_id so /retrieve_future can route it.
    from ..gateway import (
        async_remote_sampling_session,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    if request.needs_session_creation():
        from .service import ensure_sampling_session

        selector = request.model_path or request.base_model
        if not isinstance(selector, str) or not selector:
            raise HTTPException(status_code=422, detail="base_model or model_path is required")
        session_id, _created_base_model = await ensure_sampling_session(model_path=selector, http_request=http_request)
        request = request.model_copy(
            update={
                "sampling_session_id": session_id,
                "model_id": None,
                "base_model": None,
                "model_path": None,
            }
        )

    uses_existing_session_selector = request.has_session_selector()
    if server_config.sampling_require_seq_id and uses_existing_session_selector and request.seq_id is None:
        raise HTTPException(
            status_code=422,
            detail="seq_id is required when sampling_session_id or model_id is provided",
        )
    session_id = request.get_session_id()
    manager = _active_session_manager()
    snapshot = await _async_get_detached_sampling_snapshot(session_id)
    remote = None
    if snapshot is None:
        try:
            remote = await async_remote_sampling_session(session_id)
        except Exception:
            remote = None
        if remote is None:
            try:
                from ..gateway import remote_sampling_session

                remote = remote_sampling_session(session_id)
            except Exception:
                remote = None
    if snapshot is None and remote is None and manager is None and request.sampling_session_id is not None:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable")
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = getattr(http_request.state, "user_data", None)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/asample",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=300.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream asample returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    # Preflight prompt length gate from detached sampling state before enqueuing work.
    if snapshot is not None and snapshot.uses_multi_lora:
        base_model = snapshot.base_model
        if not base_model:
            raise HTTPException(status_code=500, detail=f"Session {session_id!r} missing base_model")
        token_ids = request.prompt.to_token_ids()
        max_tokens = int(request.sampling_params.max_tokens)
        from ..backend.model_registry import get_model_config

        try:
            max_model_len = int(get_model_config(base_model).max_model_len)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Cannot determine max_model_len for base_model {base_model!r}: "
                    f"{type(e).__name__}: {e}"
                ),
            )
        total_len = len(token_ids) + max_tokens
        if total_len > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt+max_tokens length {total_len} exceeds max_model_len {max_model_len} "
                    f"for model {base_model}"
                ),
            )

    global _inflight_sample_tasks
    if _should_backpressure(http_request):
        record_sampling_admission_metric(
            route=_ASAMPLE_ROUTE,
            decision="rejected",
            reason="server_overloaded",
        )
        raise HTTPException(status_code=429, detail="Sampling backpressure: server overloaded")
    user_id = _get_user_id(http_request)
    request_json = request.model_dump_json().encode("utf-8")
    payload_hash = _payload_hash(request_json)
    if request.seq_id is not None:
        request_id = _deterministic_request_id(session_id, request.seq_id)
    else:
        request_id = uuid.uuid4().hex
    billing_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
    throttle_principal, apikey_id, throttle_scope = _get_asample_throttle_identity(
        http_request,
        billing_auth=billing_auth,
    )
    created_pending = False
    model_work_attempt_id = uuid.uuid4().hex
    base_model = snapshot.base_model if snapshot is not None else None
    if snapshot is None or not base_model:
        snapshot = _get_sampling_snapshot(session_id)
        base_model = snapshot.base_model if snapshot is not None else None
    if snapshot is None or not base_model:
        raise HTTPException(status_code=404, detail=f"Sampling session {session_id!r} not found")

    # Set request_id in context for logging
    set_request_id(request_id)
    logger.info(f"asample request received: session_id={session_id}, seq_id={request.seq_id}")

    if request.seq_id is not None:
        for attempt in range(2):
            try:
                ensure = await task_state_futures.async_ensure_pending(
                    request_id=request_id,
                    meta={
                        "payload_hash": payload_hash,
                        "model_work_attempt_id": model_work_attempt_id,
                    },
                )
            except TaskStateStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: TaskStateStore requires Ray")
            if bool(ensure.get("created")):
                created_pending = True
                break
            meta = ensure.get("meta")
            existing_hash = meta.get("payload_hash") if isinstance(meta, dict) else None
            if not isinstance(existing_hash, str) or not existing_hash:
                raise HTTPException(
                    status_code=409,
                    detail="Duplicate seq_id with existing request lacking payload hash",
                )
            if existing_hash != payload_hash:
                raise HTTPException(
                    status_code=409,
                    detail="Duplicate seq_id with different request payload",
                )
            try:
                await task_state_futures.async_get_status(request_id)
            except TaskStateStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: TaskStateStore requires Ray")
            except KeyError:
                if attempt == 0:
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="Duplicate seq_id lost while confirming pending request",
                )
            return UntypedAPIFuture(request_id=request_id)

    created_by_admission = False
    scheduler_append_confirmed = False
    try:
        domain_key = _model_work_domain_key(str(base_model))
        affinity_group = _model_work_affinity_group(snapshot)
        ordering_key = f"session:{session_id}"
        queued_meta = {
            "op": "sampling.asample",
            "sampling_session_id": str(session_id),
            "queue_state": "queued",
            "queued_at": time.time(),
            "stage": "queued",
            "queue_kind": "model_work_scheduler",
            "domain_key": domain_key,
            "affinity_group": affinity_group,
            "ordering_key": ordering_key,
            "model_work_attempt_id": model_work_attempt_id,
        }
        enqueue_extra = merge_queue_priority_extra(
            {"gateway_auth": billing_auth.__dict__} if billing_auth is not None else None,
            request=http_request,
        )
        admission = await enqueue_model_work(
            request_id=request_id,
            op="sampling.asample",
            request_json=request_json,
            user_id=user_id,
            apikey_id=apikey_id,
            throttle_principal=throttle_principal,
            webhook_url=None,
            domain_key=domain_key,
            affinity_group=affinity_group,
            ordering_key=ordering_key,
            token_cost=max(
                1,
                int(request.sampling_params.max_tokens) * int(request.num_samples),
            ),
            assign=True,
            assign_max_items=1,
            extra={
                **enqueue_extra,
                "model_work_attempt_id": model_work_attempt_id,
            },
            queued_meta=queued_meta,
            create_future=not created_pending,
            payload_hash=payload_hash,
            task_state_futures_client=task_state_futures,
            trace_enqueue=_enqueue_sampling_request_with_trace,
            trace_kwargs={
                "route_start_s": route_start_s,
                "session_id": session_id,
                "base_model": base_model,
            },
        )
        scheduler_result = admission.scheduler_result
        scheduler_append_confirmed = bool(scheduler_result.get("ok"))
        created_by_admission = not created_pending
        if scheduler_result.get("scheduler_instance_id"):
            await task_state_futures.async_update_meta(
                request_id,
                {
                    "model_work_scheduler_instance_id": str(
                        scheduler_result["scheduler_instance_id"]
                    ),
                    "model_work_attempt_id": model_work_attempt_id,
                },
            )
    except Exception as e:
        if scheduler_append_confirmed:
            try:
                from ..backend.model_work_scheduler import model_work_scheduler

                await model_work_scheduler.cancel_request(
                    request_id=request_id,
                    reason="asample_enqueue_failed",
                )
            except Exception:
                pass
        if created_pending:
            try:
                await task_state_futures.async_forget(request_id)
            except TaskStateStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: TaskStateStore requires Ray")
        elif created_by_admission:
            await task_state_futures.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue sampling request: {e}")

    record_sampling_admission_metric(
        route=_ASAMPLE_ROUTE,
        decision="accepted",
        reason="queued",
        scope=throttle_scope,
    )
    return UntypedAPIFuture(request_id=request_id)


async def sample_once(
    *,
    session_id: str,
    token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | list[int] | None,
    request_id: str,
    http_request: Request,
    user_id: str | None,
    bill_usage: bool = True,
) -> SampledSequence:
    """Synchronously execute one sampling request using the multi-LoRA path."""
    from ..gateway import async_remote_sampling_session, forward_json, upstream_for_alias

    if _should_backpressure(http_request):
        record_sampling_admission_metric(
            route=_SAMPLE_ONCE_ROUTE,
            decision="rejected",
            reason="server_overloaded",
        )
        raise HTTPException(status_code=429, detail="Sampling backpressure: server overloaded")

    manager = _active_session_manager()
    snapshot = await _async_get_detached_sampling_snapshot(session_id)
    remote = None
    if snapshot is None:
        try:
            remote = await async_remote_sampling_session(session_id)
        except Exception:
            remote = None
        if remote is None:
            try:
                from ..gateway import remote_sampling_session

                remote = remote_sampling_session(session_id)
            except Exception:
                remote = None
    if snapshot is None and remote is None and manager is None:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable")
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = getattr(http_request.state, "user_data", None)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        sample_request = SampleRequest(
            sampling_session_id=session_id,
            num_samples=1,
            prompt=ModelInput.from_ints(token_ids),
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            ),
        )
        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/asample",
            incoming_headers=dict(http_request.headers),
            json_body=sample_request.model_dump(),
            timeout_s=300.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream asample returned invalid request_id")

        poll_timeout_s = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
        poll_sleep_s = float(os.environ.get("TINKER_POLL_SLEEP_S", "0.2"))
        deadline = time.time() + poll_timeout_s
        while True:
            poll_resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/retrieve_future",
                incoming_headers=dict(http_request.headers),
                json_body={"request_id": upstream_request_id},
                timeout_s=30.0,
            )
            if poll_resp.status_code == 408:
                if time.time() > deadline:
                    raise HTTPException(
                        status_code=504,
                        detail=(
                            f"Upstream {upstream_alias!r} retrieve_future timed out after "
                            f"{poll_timeout_s:.1f}s for request_id={upstream_request_id!r}"
                        ),
                    )
                await asyncio.sleep(poll_sleep_s)
                continue
            if poll_resp.status_code >= 400:
                raise HTTPException(status_code=poll_resp.status_code, detail=poll_resp.text)
            try:
                poll_payload = poll_resp.json()
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream {upstream_alias!r} returned non-JSON retrieve_future payload",
                ) from e
            if isinstance(poll_payload, dict) and "error" in poll_payload:
                raise HTTPException(status_code=500, detail=poll_payload["error"])
            try:
                sample_response = SampleResponse.model_validate(poll_payload)
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream {upstream_alias!r} returned invalid sample payload: {type(e).__name__}: {e}",
                ) from e
            if len(sample_response.sequences) != 1:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Upstream {upstream_alias!r} returned {len(sample_response.sequences)} sequences "
                        "for sample_once(num_samples=1)"
                    ),
                )
            return sample_response.sequences[0]

    if manager is None:
        from ..models.types import FutureRetrieveRequest
        from .futures import retrieve_future

        future = await asample(
            SampleRequest(
                sampling_session_id=session_id,
                num_samples=1,
                prompt=ModelInput.from_ints(token_ids),
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                ),
            ),
            http_request,
        )
        poll_timeout_s = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
        poll_sleep_s = float(os.environ.get("TINKER_POLL_SLEEP_S", "0.2"))
        deadline = time.time() + poll_timeout_s
        while True:
            poll_response = Response()
            payload = await retrieve_future(
                FutureRetrieveRequest(request_id=future.request_id),
                http_request,
                poll_response,
            )
            if poll_response.status_code == 408:
                if time.time() > deadline:
                    raise HTTPException(
                        status_code=504,
                        detail=(
                            "Local retrieve_future timed out after "
                            f"{poll_timeout_s:.1f}s for request_id={future.request_id!r}"
                        ),
                    )
                await asyncio.sleep(poll_sleep_s)
                continue
            if poll_response.status_code >= 400:
                if isinstance(payload, dict) and "detail" in payload:
                    detail = payload["detail"]
                else:
                    detail = payload
                raise HTTPException(status_code=poll_response.status_code, detail=detail)
            if isinstance(payload, dict) and "error" in payload:
                raise HTTPException(status_code=500, detail=payload["error"])
            try:
                sample_response = SampleResponse.model_validate(payload)
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Local retrieve_future returned invalid sample payload: {type(e).__name__}: {e}",
                ) from e
            if len(sample_response.sequences) != 1:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Local retrieve_future returned {len(sample_response.sequences)} sequences "
                        "for sample_once(num_samples=1)"
                    ),
                )
            return sample_response.sequences[0]

    engine = None
    resource_pool = None
    resource_pool_actor_name: str | None = None
    manager.mark_session_inflight(session_id, +1)
    try:
        snapshot = _get_sampling_snapshot(session_id)
        if snapshot is None:
            await _restore_local_sampling_session_if_needed(session_id)
            snapshot = _get_sampling_snapshot(session_id)
        is_multi_lora = bool(snapshot.uses_multi_lora) if snapshot is not None else manager.is_multi_lora_session(session_id)
        if is_multi_lora:
            base_model = snapshot.base_model if snapshot is not None else manager.get_session_base_model(session_id)
            if not base_model:
                raise HTTPException(status_code=500, detail=f"Session {session_id!r} missing base_model")

            from ..backend.model_registry import get_model_config

            try:
                max_model_len = int(get_model_config(base_model).max_model_len)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Cannot determine max_model_len for base_model {base_model!r}: "
                        f"{type(e).__name__}: {e}"
                    ),
                ) from e

            total_len = len(token_ids) + int(max_tokens)
            if total_len > max_model_len:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt+max_tokens length {total_len} exceeds max_model_len {max_model_len} "
                        f"for model {base_model}"
                    ),
                )

            engine = await run_async_with_otel_span(
                "sampling.get_engine_for_session",
                lambda: manager.get_engine_for_session(session_id),
                component="sampling",
                op="sampling.get_engine_for_session",
                request_id=request_id,
                attributes={
                    "sampling_session_id": session_id,
                    "base_model": snapshot.base_model if snapshot is not None else None,
                    "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                    "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                },
            )
            if engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")

            from ..backend.resource_pool import get_resource_pool

            resource_pool = get_resource_pool()
            resource_pool_actor_name = getattr(engine, "actor_name", None)
            if not isinstance(resource_pool_actor_name, str) or not resource_pool_actor_name:
                raise RuntimeError(
                    f"Engine for session {session_id} missing actor_name; cannot protect from eviction"
                )
            resource_pool.mark_inflight(resource_pool_actor_name, +1)
            await run_async_with_otel_span(
                "sampling.ensure_lora_loaded",
                lambda: _ensure_session_lora_loaded(engine, session_id, snapshot=snapshot),
                component="sampling",
                op="sampling.ensure_lora_loaded",
                request_id=request_id,
                attributes={
                    "sampling_session_id": session_id,
                    "base_model": snapshot.base_model if snapshot is not None else None,
                    "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                    "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                    "has_adapter_path": bool(snapshot.adapter_path) if snapshot is not None else None,
                },
            )
            result = await _await_with_external_fail_abort(
                engine=engine,
                request_id=request_id,
                awaitable=engine.generate(
                    sampling_session_id=session_id,
                    prompt_ids=token_ids,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=-1,
                    top_p=top_p,
                    logprobs=False,
                ),
            )
        else:
            engine = manager.get_engine(session_id)
            if engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")
            result = await _await_with_external_fail_abort(
                engine=engine,
                request_id=request_id,
                awaitable=engine.generate(
                    prompt_ids=token_ids,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=-1,
                    top_p=top_p,
                    logprobs=False,
                ),
            )

        sequence = sampled_sequence_from_result(result)
        if bill_usage:
            schedule_usage_events(
                build_sample_once_usage_events(
                    session_id=session_id,
                    token_ids=token_ids,
                    sequence=sequence,
                    http_request=http_request,
                    request_id=request_id,
                )
            )
        return sequence
    except HTTPException:
        await _abort_engine_request(engine, request_id)
        raise
    except Exception:
        await _abort_engine_request(engine, request_id)
        raise
    finally:
        if resource_pool is not None and resource_pool_actor_name is not None:
            resource_pool.mark_inflight(resource_pool_actor_name, -1)
        manager.mark_session_inflight(session_id, -1)


async def _do_sample(
    request_id: str,
    request: SampleRequest,
    user_id: str | None,
    gateway_auth: dict | None = None,
) -> None:
    """Background task to perform sampling."""
    # Restore request_id context for logging
    set_request_id(request_id)

    global _inflight_sample_tasks
    _inflight_sample_tasks += 1
    session_id: str | None = None
    engine = None
    resource_pool = None
    resource_pool_actor_name: str | None = None
    workload_base_model = "unknown"
    workload_started_at = time.perf_counter()
    workload_started = False
    workload_status = "error"
    workload_generated_tokens = 0
    workload_obs = {
        "ttft_s": None,
        "tpot_s": None,
    }
    try:
        try:
            if session_manager is None:
                raise RuntimeError("Session manager not initialized")

            token_ids = request.prompt.to_token_ids()
            session_id = request.get_session_id()  # Supports both sampling_session_id and model_id
            await _restore_local_sampling_session_if_needed(session_id)
            workload_base_model = _resolve_billing_model(session_id)
            session_manager.mark_session_inflight(session_id, +1)
            snapshot = _get_sampling_snapshot(session_id)

            # Handle include_prompt_logprobs alias
            want_prompt_logprobs = request.prompt_logprobs or request.include_prompt_logprobs

            # Check if session uses multi-LoRA mode (includes base model sessions)
            is_multi_lora = bool(snapshot.uses_multi_lora) if snapshot is not None else session_manager.is_multi_lora_session(session_id)
            if is_multi_lora:
                base_model = snapshot.base_model if snapshot is not None else session_manager.get_session_base_model(session_id)
                if not base_model:
                    raise RuntimeError(f"Session {session_id!r} missing base_model")
                from ..backend.model_registry import get_model_config

                max_model_len = int(get_model_config(base_model).max_model_len)
                total_len = len(token_ids) + int(request.sampling_params.max_tokens)
                if total_len > max_model_len:
                    raise ValueError(
                        f"Prompt+max_tokens length {total_len} exceeds max_model_len {max_model_len} "
                        f"for model {base_model}"
                    )

            # Generate for each sample
            if request.num_samples < 1:
                raise ValueError(f"num_samples must be >= 1 (got {request.num_samples})")

            if is_multi_lora:
                # Multi-LoRA mode: handles both LoRA and base model sessions
                # Get engine once; it is shared across samples.
                logger.info(
                    f"[sample path] request_id={request_id} session_id={session_id} stage=before_get_engine"
                )
                engine = await run_async_with_otel_span(
                    "sampling.get_engine_for_session",
                    lambda: session_manager.get_engine_for_session(session_id),
                    component="sampling",
                    op="sampling.get_engine_for_session",
                    request_id=request_id,
                    attributes={
                        "sampling_session_id": session_id,
                        "base_model": snapshot.base_model if snapshot is not None else None,
                        "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                        "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                    },
                )
                logger.info(
                    f"[sample path] request_id={request_id} session_id={session_id} stage=after_get_engine"
                )
                if engine is None:
                    raise RuntimeError(f"No engine found for session {session_id}")
                from ..backend.resource_pool import get_resource_pool

                resource_pool = get_resource_pool()
                resource_pool_actor_name = getattr(engine, "actor_name", None)
                if not isinstance(resource_pool_actor_name, str) or not resource_pool_actor_name:
                    raise RuntimeError(
                        f"Engine for session {session_id} missing actor_name; cannot protect from eviction"
                    )
                resource_pool.mark_inflight(resource_pool_actor_name, +1)
                _record_vllm_workload_start(
                    actor_name=resource_pool_actor_name,
                    base_model=workload_base_model,
                    op="asample",
                )
                workload_started = True

                logger.info(
                    f"[sample path] session_id={session_id} "
                    f"prompt_tokens={len(token_ids)} num_samples={request.num_samples} stage=before_lora_load"
                )
                await run_async_with_otel_span(
                    "sampling.ensure_lora_loaded",
                    lambda: _ensure_session_lora_loaded(engine, session_id, snapshot=snapshot),
                    component="sampling",
                    op="sampling.ensure_lora_loaded",
                    request_id=request_id,
                    attributes={
                        "sampling_session_id": session_id,
                        "base_model": snapshot.base_model if snapshot is not None else None,
                        "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                        "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                        "has_adapter_path": bool(snapshot.adapter_path) if snapshot is not None else None,
                    },
                )
                logger.info(f"[sample path] session_id={session_id} stage=after_lora_load")

                gen_many = getattr(engine, "generate_many", None)
                can_coalesce = (
                    _SAMPLE_COALESCE
                    and (not want_prompt_logprobs)
                    and request.topk_prompt_logprobs == 0
                    and _SAMPLE_COALESCE_MAX_BATCH > 1
                    and gen_many is not None
                    and request.num_samples <= _SAMPLE_COALESCE_MAX_SAMPLES
                )
                if engine.__class__.__name__ == "MultiNodeInferenceEngine":
                    # Multi-node vLLM has shown severe hangs on the coalesced
                    # generate_many path even for a single waiter. Keep the
                    # native per-request generate path for these engines.
                    can_coalesce = False
                logger.info(
                    f"[sample path] session_id={session_id} "
                    f"can_coalesce={can_coalesce} sample_coalesce={_SAMPLE_COALESCE} "
                    f"want_prompt_logprobs={want_prompt_logprobs} topk_prompt_logprobs={request.topk_prompt_logprobs} "
                    f"has_generate_many={gen_many is not None} num_samples={request.num_samples}"
                )
                if can_coalesce:
                    logger.info("[sample path] branch=coalesced_generate")
                    results = await run_async_with_otel_span(
                        "sampling.generate",
                        lambda: _await_with_external_fail_abort(
                            engine=engine,
                            request_id=request_id,
                            awaitable=_coalesced_generate(
                                engine=engine,
                                sampling_session_id=session_id,
                                prompt_ids=token_ids,
                                request_id=request_id,
                                num_samples=request.num_samples,
                                max_tokens=request.sampling_params.max_tokens,
                                stop=request.sampling_params.stop,
                                temperature=request.sampling_params.temperature,
                                top_k=request.sampling_params.top_k,
                                top_p=request.sampling_params.top_p,
                            ),
                        ),
                        component="sampling",
                        op="sampling.generate",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id, "num_samples": request.num_samples},
                    )
                elif request.num_samples == 1:
                    logger.info("[sample path] branch=generate_single")
                    one_result = await run_async_with_otel_span(
                        "sampling.generate",
                        lambda: _await_with_external_fail_abort(
                            engine=engine,
                            request_id=request_id,
                            awaitable=engine.generate(
                                sampling_session_id=session_id,
                                prompt_ids=token_ids,
                                request_id=request_id,
                                max_tokens=request.sampling_params.max_tokens,
                                stop=request.sampling_params.stop,
                                temperature=request.sampling_params.temperature,
                                top_k=request.sampling_params.top_k,
                                top_p=request.sampling_params.top_p,
                                logprobs=True,
                            ),
                        ),
                        component="sampling",
                        op="sampling.generate",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id, "num_samples": 1},
                    )
                    results = [one_result]
                else:
                    if gen_many is None:
                        raise RuntimeError(f"Engine for session {session_id} does not support generate_many()")
                    logger.info("[sample path] branch=generate_many")
                    results = await run_async_with_otel_span(
                        "sampling.generate",
                        lambda: _await_with_external_fail_abort(
                            engine=engine,
                            request_id=request_id,
                            awaitable=gen_many(
                                sampling_session_id=session_id,
                                prompt_ids=token_ids,
                                request_id=request_id,
                                num_samples=request.num_samples,
                                max_tokens=request.sampling_params.max_tokens,
                                stop=request.sampling_params.stop,
                                temperature=request.sampling_params.temperature,
                                top_k=request.sampling_params.top_k,
                                top_p=request.sampling_params.top_p,
                                logprobs=True,
                            ),
                        ),
                        component="sampling",
                        op="sampling.generate_many",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id, "num_samples": request.num_samples},
                    )
            else:
                # Legacy mode: per-session engine
                engine = session_manager.get_engine(session_id)
                if engine is None:
                    raise RuntimeError(f"No engine found for session {session_id}")
                resource_pool_actor_name = getattr(engine, "actor_name", None)
                _record_vllm_workload_start(
                    actor_name=resource_pool_actor_name,
                    base_model=workload_base_model,
                    op="asample",
                )
                workload_started = True

                async def _generate_one(i: int):
                    return await engine.generate(
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_{i}",
                        max_tokens=request.sampling_params.max_tokens,
                        stop=request.sampling_params.stop,
                        temperature=request.sampling_params.temperature,
                        top_k=request.sampling_params.top_k,
                        top_p=request.sampling_params.top_p,
                        logprobs=True,
                    )

                max_concurrent = _MAX_CONCURRENT_SAMPLES_PER_REQUEST
                if max_concurrent <= 0:
                    max_concurrent = request.num_samples
                sem = asyncio.Semaphore(max_concurrent)

                async def _generate_limited(i: int):
                    async with sem:
                        return await _generate_one(i)

                results = await asyncio.gather(*(_generate_limited(i) for i in range(request.num_samples)))

            sequences = []
            for result in results:
                sequences.append(sampled_sequence_from_result(result))

            # Build response
            response = SampleResponse(sequences=sequences)

            # Handle prompt logprobs if requested
            if want_prompt_logprobs:
                if is_multi_lora:
                    engine_for_logprobs = engine
                    if engine_for_logprobs is None:
                        raise RuntimeError(f"No engine found for session {session_id}")
                    computed_logprobs = await run_async_with_otel_span(
                        "sampling.compute_prompt_logprobs",
                        lambda: engine_for_logprobs.compute_logprobs(
                            sampling_session_id=session_id,
                            prompt_ids=token_ids,
                            request_id=f"{request_id}_prompt_logprobs",
                        ),
                        component="sampling",
                        op="sampling.compute_prompt_logprobs",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id, "prompt_tokens": len(token_ids)},
                    )
                else:
                    computed_logprobs = await engine.compute_logprobs(
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_prompt_logprobs",
                    )
                response.prompt_logprobs = normalize_prompt_logprobs_for_tinker(
                    computed_logprobs, prompt_len=len(token_ids)
                )

            # Handle top-K prompt logprobs if requested
            if request.topk_prompt_logprobs > 0:
                if is_multi_lora:
                    engine_for_topk = engine
                    if engine_for_topk is None:
                        raise RuntimeError(f"No engine found for session {session_id}")
                    computed_topk = await run_async_with_otel_span(
                        "sampling.compute_prompt_topk",
                        lambda: engine_for_topk.compute_topk(
                            sampling_session_id=session_id,
                            prompt_ids=token_ids,
                            request_id=f"{request_id}_topk",
                            k=request.topk_prompt_logprobs,
                        ),
                        component="sampling",
                        op="sampling.compute_prompt_topk",
                        request_id=request_id,
                        attributes={
                            "sampling_session_id": session_id,
                            "prompt_tokens": len(token_ids),
                            "topk": request.topk_prompt_logprobs,
                        },
                    )
                else:
                    computed_topk = await engine.server.compute_prompt_topk.remote(
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_topk",
                        k=request.topk_prompt_logprobs,
                    )
                response.topk_prompt_logprobs = _normalize_topk_prompt_logprobs(
                    list(computed_topk),
                    request.topk_prompt_logprobs,
                )

            usage_events: list[UsageEvent] = []
            if gateway_auth:
                auth_ctx = GatewayAuthContext(**gateway_auth)
                prefill_tokens = len(token_ids)
                sampling_tokens = sum(len(seq.tokens) for seq in sequences)
                label_model = _resolve_billing_model(session_id)
                usage_events.extend(
                    [
                        UsageEvent(
                            account_id=auth_ctx.account_id,
                            apikey_id=auth_ctx.apikey_id,
                            charge_item="sampling",
                            quantity=prefill_tokens,
                            request_id=auth_ctx.request_id,
                            label=_build_sampling_usage_label(
                                model=label_model,
                                route="sampling.asample",
                                dimension="prefill",
                            ),
                        ),
                        UsageEvent(
                            account_id=auth_ctx.account_id,
                            apikey_id=auth_ctx.apikey_id,
                            charge_item="sampling",
                            quantity=sampling_tokens,
                            request_id=auth_ctx.request_id,
                            label=_build_sampling_usage_label(
                                model=label_model,
                                route="sampling.asample",
                                dimension="sample",
                            ),
                        ),
                    ]
                )
            # Compatibility: older tinker clients don't accept a top-level `type` field on SampleResponse.
            await task_state_futures.async_resolve(request_id, response.model_dump(exclude={"type"}))
            if usage_events:
                await _persist_usage_events(auth_ctx=auth_ctx, events=usage_events)
            workload_status = "ok"
            workload_generated_tokens = sum(len(seq.tokens) for seq in sequences)
            workload_obs = _vllm_request_observation(results, workload_generated_tokens)
            logger.info(f"Sampling completed: {len(sequences)} sequences generated")

        except asyncio.CancelledError:
            workload_status = "canceled"
            await _abort_engine_request(engine, request_id)
            await task_state_futures.async_fail(request_id, "sampling task cancelled")
            logger.warning(
                "[sampling.asample] canceled request_id=%s session_id=%s next_action=%s",
                str(request_id),
                str(session_id),
                "caller_can_retry",
            )
            raise
        except Exception as e:
            workload_status = "error"
            await _abort_engine_request(engine, request_id)
            logger.exception(
                "[sampling.asample] failed request_id=%s session_id=%s failure_reason=%s error_type=%s next_action=%s",
                str(request_id),
                str(session_id),
                classify_failure_reason(e),
                type(e).__name__,
                "check_sampling_session_and_vllm_actor",
            )
            await task_state_futures.async_fail(request_id, str(e))
    finally:
        if workload_started:
            _record_vllm_workload_finish(
                actor_name=resource_pool_actor_name,
                base_model=workload_base_model,
                op="asample",
                status=workload_status,
                prompt_tokens=len(token_ids) if 'token_ids' in locals() else 0,
                generated_tokens=workload_generated_tokens,
                started_at=workload_started_at,
                ttft_s=workload_obs["ttft_s"],
                tpot_s=workload_obs["tpot_s"],
            )
        if resource_pool is not None and resource_pool_actor_name is not None:
            resource_pool.mark_inflight(resource_pool_actor_name, -1)
        if session_manager is not None and session_id is not None:
            session_manager.mark_session_inflight(session_id, -1)
        _inflight_sample_tasks -= 1


@router.post("/compute_logprobs")
async def compute_logprobs(
    request: ComputeLogprobsRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Compute logprobs for a sequence.

    Returns a list of length len(sequence), where:
    - logprobs[0] is None (first token has no conditioning context)
    - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1
    """
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_sampling_session,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    snapshot = await _async_get_detached_sampling_snapshot(request.sampling_session_id)
    remote = None
    if snapshot is None:
        try:
            remote = await async_remote_sampling_session(request.sampling_session_id)
        except Exception:
            remote = None
        if remote is None:
            try:
                from ..gateway import remote_sampling_session

                remote = remote_sampling_session(request.sampling_session_id)
            except Exception:
                remote = None
    if snapshot is None and remote is None and session_manager is None:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable")
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = getattr(http_request.state, "user_data", None)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/compute_logprobs",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=300.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream compute_logprobs returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    # Preflight length gate for multi-LoRA sessions to fail fast on registry issues.
    if snapshot is not None and snapshot.uses_multi_lora:
        base_model = snapshot.base_model
        if not base_model:
            raise HTTPException(status_code=500, detail=f"Session {request.sampling_session_id!r} missing base_model")
        token_ids = request.sequence.to_token_ids()
        from ..backend.model_registry import get_model_config

        try:
            max_model_len = int(get_model_config(base_model).max_model_len)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Cannot determine max_model_len for base_model {base_model!r}: "
                    f"{type(e).__name__}: {e}"
                ),
            )
        total_len = len(token_ids) + 1
        if total_len > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt+max_tokens length {total_len} exceeds max_model_len {max_model_len} "
                    f"for model {base_model}"
                ),
            )

    user_id = _get_user_id(http_request)
    from ..backend.model_work_admission import enqueue_model_work
    from ..backend.model_work_scheduler import model_work_scheduler

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    billing_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
    base_model = snapshot.base_model if snapshot is not None else None
    if not base_model:
        raise HTTPException(status_code=500, detail=f"Session {request.sampling_session_id!r} missing base_model")
    domain_key = _model_work_domain_key(str(base_model))
    affinity_group = (
        _model_work_affinity_group(snapshot)
        if snapshot is not None
        else f"session:{request.sampling_session_id}"
    )

    try:
        await enqueue_model_work(
            request_id=request_id,
            op="sampling.compute_logprobs",
            request_json=request_json,
            user_id=user_id,
            apikey_id=_get_apikey_id(http_request, billing_auth=billing_auth),
            webhook_url=None,
            domain_key=domain_key,
            affinity_group=affinity_group,
            ordering_key=f"session:{request.sampling_session_id}",
            token_cost=max(1, len(request.sequence.to_token_ids())),
            extra=merge_queue_priority_extra(
                {"gateway_auth": billing_auth.__dict__} if billing_auth is not None else None,
                request=http_request,
            ),
            queued_meta={
                "op": "sampling.compute_logprobs",
                "sampling_session_id": str(request.sampling_session_id),
                "queue_state": "queued",
                "queued_at": time.time(),
                "stage": "queued",
                "queue_kind": "model_work_scheduler",
                "domain_key": domain_key,
                "affinity_group": affinity_group,
            },
            task_state_futures_client=task_state_futures,
            scheduler_client=model_work_scheduler,
            trace_enqueue=_enqueue_sampling_request_with_trace,
            trace_kwargs={
                "route_start_s": route_start_s,
                "session_id": request.sampling_session_id,
                "base_model": base_model,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to enqueue compute_logprobs request: {e}")

    record_sampling_admission_metric(
        route=_COMPUTE_LOGPROBS_ROUTE,
        decision="accepted",
        reason="queued",
    )
    return UntypedAPIFuture(request_id=request_id)


async def _do_compute_logprobs(
    request_id: str,
    request: ComputeLogprobsRequest,
    user_id: str | None,
    gateway_auth: dict | None = None,
) -> None:
    """Background task to compute logprobs."""
    session_id: str | None = None
    resource_pool = None
    resource_pool_actor_name: str | None = None
    workload_base_model = "unknown"
    workload_started_at = time.perf_counter()
    workload_started = False
    workload_status = "error"
    try:
        set_request_id(request_id)
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.sequence.to_token_ids()
        session_id = request.sampling_session_id
        await _restore_local_sampling_session_if_needed(session_id)
        workload_base_model = _resolve_billing_model(session_id)
        session_manager.mark_session_inflight(session_id, +1)
        snapshot = _get_sampling_snapshot(session_id)
        base_model = snapshot.base_model if snapshot is not None else session_manager.get_session_base_model(session_id)

        async def _compute_logprobs_action():
            nonlocal resource_pool, resource_pool_actor_name, workload_started

            # Check if session uses multi-LoRA mode (includes base model sessions)
            is_multi_lora = bool(snapshot.uses_multi_lora) if snapshot is not None else session_manager.is_multi_lora_session(session_id)
            if is_multi_lora:
                if not base_model:
                    raise RuntimeError(f"Session {session_id!r} missing base_model")
                from ..backend.model_registry import get_model_config

                max_model_len = int(get_model_config(base_model).max_model_len)
                total_len = len(token_ids) + 1
                if total_len > max_model_len:
                    raise ValueError(
                        f"Prompt+max_tokens length {total_len} exceeds max_model_len {max_model_len} "
                        f"for model {base_model}"
                    )

            if is_multi_lora:
                multi_lora_engine = await run_async_with_otel_span(
                    "sampling.get_engine_for_session",
                    lambda: session_manager.get_engine_for_session(session_id),
                    component="sampling",
                    op="sampling.get_engine_for_session",
                    request_id=request_id,
                    attributes={
                        "sampling_session_id": session_id,
                        "base_model": snapshot.base_model if snapshot is not None else None,
                        "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                        "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                    },
                )
                if multi_lora_engine is None:
                    raise RuntimeError(f"No engine found for session {session_id}")
                from ..backend.resource_pool import get_resource_pool

                resource_pool = get_resource_pool()
                resource_pool_actor_name = getattr(multi_lora_engine, "actor_name", None)
                if not isinstance(resource_pool_actor_name, str) or not resource_pool_actor_name:
                    raise RuntimeError(
                        f"Engine for session {session_id} missing actor_name; cannot protect from eviction"
                    )
                resource_pool.mark_inflight(resource_pool_actor_name, +1)
                _record_vllm_workload_start(
                    actor_name=resource_pool_actor_name,
                    base_model=workload_base_model,
                    op="compute_logprobs",
                )
                workload_started = True
                await run_async_with_otel_span(
                    "sampling.ensure_lora_loaded",
                    lambda: _ensure_session_lora_loaded(multi_lora_engine, session_id, snapshot=snapshot),
                    component="sampling",
                    op="sampling.ensure_lora_loaded",
                    request_id=request_id,
                    attributes={
                        "sampling_session_id": session_id,
                        "base_model": snapshot.base_model if snapshot is not None else None,
                        "lora_rank": int(snapshot.lora_rank) if snapshot is not None else None,
                        "lora_loaded_before": bool(snapshot.lora_loaded) if snapshot is not None else None,
                        "has_adapter_path": bool(snapshot.adapter_path) if snapshot is not None else None,
                    },
                )
                return await multi_lora_engine.compute_logprobs(
                    sampling_session_id=session_id,
                    prompt_ids=token_ids,
                    request_id=request_id,
                )

            engine = session_manager.get_engine(session_id)
            if engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")
            resource_pool_actor_name = getattr(engine, "actor_name", None)
            _record_vllm_workload_start(
                actor_name=resource_pool_actor_name,
                base_model=workload_base_model,
                op="compute_logprobs",
            )
            workload_started = True
            return await engine.compute_logprobs(
                prompt_ids=token_ids,
                request_id=request_id,
            )

        logprobs = await run_async_with_otel_span(
            "sampling.compute_logprobs.execute",
            _compute_logprobs_action,
            component="routes.sampling",
            op="sampling.compute_logprobs",
            request_id=str(request_id),
            attributes={
                "sampling_session_id": str(session_id),
                "base_model": str(base_model) if base_model else None,
                "prompt_tokens": int(len(token_ids)),
            },
        )

        logprobs = normalize_prompt_logprobs_for_tinker(logprobs, prompt_len=len(token_ids))
        response = ComputeLogprobsResponse(logprobs=logprobs)
        # Compatibility: older tinker clients don't accept a top-level `type` field on ComputeLogprobsResponse.
        await task_state_futures.async_resolve(request_id, response.model_dump(exclude={"type"}))
        if gateway_auth:
            auth_ctx = GatewayAuthContext(**gateway_auth)
            await _persist_usage_events(
                auth_ctx=auth_ctx,
                events=[
                    UsageEvent(
                        account_id=auth_ctx.account_id,
                        apikey_id=auth_ctx.apikey_id,
                        charge_item="sampling",
                        quantity=len(token_ids),
                        request_id=auth_ctx.request_id,
                        label=_build_sampling_usage_label(
                            model=_resolve_billing_model(session_id),
                            route="sampling.compute_logprobs",
                            dimension="prefill",
                        ),
                    )
                ],
            )
        workload_status = "ok"
        logger.debug(f"Request {request_id} computed {len(logprobs)} logprobs")

    except asyncio.CancelledError:
        workload_status = "canceled"
        await task_state_futures.async_fail(request_id, "compute_logprobs task cancelled")
        logger.warning(
            "[sampling.compute_logprobs] canceled request_id=%s session_id=%s next_action=%s",
            str(request_id),
            str(session_id or request.sampling_session_id),
            "caller_can_retry",
        )
        raise
    except Exception as e:
        workload_status = "error"
        logger.exception(
            "[sampling.compute_logprobs] failed request_id=%s session_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(session_id or request.sampling_session_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_sampling_session_and_token_length",
        )
        await task_state_futures.async_fail(request_id, str(e))
    finally:
        if workload_started:
            _record_vllm_workload_finish(
                actor_name=resource_pool_actor_name,
                base_model=workload_base_model,
                op="compute_logprobs",
                status=workload_status,
                prompt_tokens=len(token_ids) if 'token_ids' in locals() else 0,
                generated_tokens=0,
                started_at=workload_started_at,
            )
        if resource_pool is not None and resource_pool_actor_name is not None:
            resource_pool.mark_inflight(resource_pool_actor_name, -1)
        if session_manager is not None and session_id is not None:
            session_manager.mark_session_inflight(session_id, -1)
