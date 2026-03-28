"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

import asyncio
import array
import hashlib
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ..config import config as server_config
from ..backend.future_store import FutureStatus, FutureStoreUnavailableError, future_store
from ..gateway_auth import GatewayAuthContext, build_billing_auth_context
from ..logging_context import (
    classify_failure_reason,
    get_otel_tracer,
    record_sampling_admission_metric,
    run_async_with_otel_span,
    set_request_id,
)
from ..model_access_control import can_access_model, get_access_denied_error
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
from ..usage_store import UsageEvent, get_usage_store

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


async def _normalize_sampling_request_session(
    request: SampleRequest,
    http_request: Request,
) -> tuple[SampleRequest, str]:
    if not request.needs_session_creation():
        return request, request.get_session_id()

    from .service import ensure_sampling_session

    model_ref = request.model_path or request.base_model
    if not isinstance(model_ref, str) or not model_ref:
        raise HTTPException(
            status_code=422,
            detail="Exactly one selector must be provided: sampling_session_id/model_id, base_model, or model_path",
        )

    session_id, _base_model = await ensure_sampling_session(
        model_path=model_ref,
        http_request=http_request,
    )
    normalized = request.model_copy(
        update={
            "sampling_session_id": session_id,
            "model_id": None,
            "base_model": None,
            "model_path": None,
        }
    )
    return normalized, session_id


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
    if session_manager is None:
        return session_id
    return session_manager.get_session_base_model(session_id) or session_id


def _build_sampling_usage_label(*, model: str, route: str, dimension: str) -> str:
    return f"model={model},route={route},dimension={dimension}"


async def _enqueue_sampling_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
    session_id: str | None = None,
    base_model: str | None = None,
) -> None:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        await enqueue_coro
        return

    with tracer.start_as_current_span(f"{op}.enqueue") as span:
        span.set_attribute("component", "routes.sampling")
        span.set_attribute("op", str(op))
        span.set_attribute("request_id", str(request_id))
        if session_id:
            span.set_attribute("sampling_session_id", str(session_id))
        if base_model:
            span.set_attribute("base_model", str(base_model))
        span.add_event(
            "future_store_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round((time.perf_counter() - enqueue_start_s) * 1000.0, 3),
                "route_elapsed_ms": round((time.perf_counter() - route_start_s) * 1000.0, 3),
            },
        )


async def _ensure_session_lora_loaded(engine, session_id: str) -> None:
    if session_manager is None:
        raise RuntimeError("Session manager not initialized")

    lora_rank = session_manager.get_session_lora_rank(session_id)
    if not lora_rank or int(lora_rank) <= 0:
        return

    if session_manager.is_session_lora_loaded(session_id):
        return

    adapter_path = session_manager.get_session_adapter_path(session_id)
    if not adapter_path:
        raise RuntimeError(f"Session {session_id} has lora_rank={lora_rank} but no adapter_path")

    lock = await _get_lora_load_lock(session_id)
    async with lock:
        if session_manager.is_session_lora_loaded(session_id):
            return

        # Prefer path-based loading to avoid sending large tensors through Ray.
        add_from_path = getattr(engine, "add_lora_for_session_from_path", None)
        if add_from_path is None:
            raise RuntimeError(f"Engine for session {session_id} does not support add_lora_for_session_from_path()")

        await add_from_path(sampling_session_id=session_id, lora_path=adapter_path)
        session_manager.mark_session_lora_loaded(session_id, True)


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
                status = await future_store.async_get_status(request_id)
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
    usage_store = await get_usage_store()
    await usage_store.write_events(events)


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

    uses_existing_session_selector = request.has_session_selector()
    if server_config.sampling_require_seq_id and uses_existing_session_selector and request.seq_id is None:
        raise HTTPException(
            status_code=422,
            detail="seq_id is required when sampling_session_id or model_id is provided",
        )
    request, session_id = await _normalize_sampling_request_session(request, http_request)
    is_local = False
    if session_manager is not None:
        is_local = session_manager.is_multi_lora_session(session_id) or (session_manager.get_engine(session_id) is not None)
    remote = None if is_local else await async_remote_sampling_session(session_id)
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

    # Preflight prompt length gate for multi-LoRA sessions. Do this before enqueuing
    # work so misconfiguration is surfaced as an HTTP error rather than a latent
    # async failure.
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    if session_manager.is_multi_lora_session(session_id):
        base_model = session_manager.get_session_base_model(session_id)
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
    from ..backend.api_work_queue import ApiWorkQueueThrottleError, _unwrap_queue_throttle_error, api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_sampling_result_bytes

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

    # Set request_id in context for logging
    set_request_id(request_id)
    logger.info(f"asample request received: session_id={session_id}, seq_id={request.seq_id}")

    if request.seq_id is not None:
        for attempt in range(2):
            try:
                ensure = await future_store.async_ensure_pending(
                    request_id=request_id,
                    meta={"payload_hash": payload_hash},
                )
            except FutureStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
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
                await future_store.async_get_status(request_id)
            except FutureStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
            except KeyError:
                if attempt == 0:
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="Duplicate seq_id lost while confirming pending request",
                )
            return UntypedAPIFuture(request_id=request_id)

    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_sampling_result_bytes(request),
    )
    if not bool(reserve.get("ok")):
        if created_pending:
            try:
                await future_store.async_forget(request_id)
            except FutureStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
        record_sampling_admission_metric(
            route=_ASAMPLE_ROUTE,
            decision="rejected",
            reason="capacity_rejected",
        )
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        if not created_pending:
            await future_store.async_create_with_id(request_id)
            created = True
        await future_store.async_mark_queued(
            request_id,
            meta={
                "op": "sampling.asample",
                "queue_state": "queued",
                "queued_at": time.time(),
                "stage": "queued",
            },
        )
        get_session_base_model = getattr(session_manager, "get_session_base_model", None)
        base_model = get_session_base_model(session_id) if callable(get_session_base_model) else None
        await _enqueue_sampling_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="sampling.asample",
            session_id=session_id,
            base_model=base_model,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="sampling.asample",
                request_json=request_json,
                user_id=user_id,
                apikey_id=apikey_id,
                throttle_principal=throttle_principal,
                webhook_url=None,
                extra={"gateway_auth": billing_auth.__dict__} if billing_auth is not None else None,
            ),
        )
    except ApiWorkQueueThrottleError as e:
        await capacity_manager.async_release_all(request_id)
        if created_pending:
            try:
                await future_store.async_forget(request_id)
            except FutureStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
        elif created:
            await future_store.async_cleanup(request_id)
        detail = e.detail if isinstance(e.detail, dict) else {}
        record_sampling_admission_metric(
            route=_ASAMPLE_ROUTE,
            decision="rejected",
            reason="queue_throttled",
            scope=detail.get("scope") if isinstance(detail, dict) else None,
        )
        raise HTTPException(status_code=429, detail=e.detail) from e
    except Exception as e:
        throttle_error = _unwrap_queue_throttle_error(e)
        if throttle_error is not None:
            await capacity_manager.async_release_all(request_id)
            if created_pending:
                try:
                    await future_store.async_forget(request_id)
                except FutureStoreUnavailableError:
                    raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
            elif created:
                await future_store.async_cleanup(request_id)
            detail = throttle_error.detail if isinstance(throttle_error.detail, dict) else {}
            record_sampling_admission_metric(
                route=_ASAMPLE_ROUTE,
                decision="rejected",
                reason="queue_throttled",
                scope=detail.get("scope") if isinstance(detail, dict) else None,
            )
            raise HTTPException(status_code=429, detail=throttle_error.detail) from e
        await capacity_manager.async_release_all(request_id)
        if created_pending:
            try:
                await future_store.async_forget(request_id)
            except FutureStoreUnavailableError:
                raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
        elif created:
            await future_store.async_cleanup(request_id)
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

    is_local = False
    if session_manager is not None:
        is_local = session_manager.is_multi_lora_session(session_id) or (session_manager.get_engine(session_id) is not None)
    remote = None if is_local else await async_remote_sampling_session(session_id)
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

    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    engine = None
    resource_pool = None
    resource_pool_actor_name: str | None = None
    session_manager.mark_session_inflight(session_id, +1)
    try:
        if session_manager.is_multi_lora_session(session_id):
            base_model = session_manager.get_session_base_model(session_id)
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

            engine = await session_manager.get_engine_for_session(session_id)
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
            await _ensure_session_lora_loaded(engine, session_id)
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
            engine = session_manager.get_engine(session_id)
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
        billing_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
        if billing_auth is not None:
            label_model = _resolve_billing_model(session_id)
            await _persist_usage_events(
                auth_ctx=billing_auth,
                events=[
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
                ],
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
        session_manager.mark_session_inflight(session_id, -1)


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
    try:
        try:
            if session_manager is None:
                raise RuntimeError("Session manager not initialized")

            token_ids = request.prompt.to_token_ids()
            session_id = request.get_session_id()  # Supports both sampling_session_id and model_id
            session_manager.mark_session_inflight(session_id, +1)

            # Handle include_prompt_logprobs alias
            want_prompt_logprobs = request.prompt_logprobs or request.include_prompt_logprobs

            # Check if session uses multi-LoRA mode (includes base model sessions)
            is_multi_lora = session_manager.is_multi_lora_session(session_id)
            if is_multi_lora:
                base_model = session_manager.get_session_base_model(session_id)
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
                    attributes={"sampling_session_id": session_id},
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

                logger.info(
                    f"[sample path] session_id={session_id} "
                    f"prompt_tokens={len(token_ids)} num_samples={request.num_samples} stage=before_lora_load"
                )
                await run_async_with_otel_span(
                    "sampling.ensure_lora_loaded",
                    lambda: _ensure_session_lora_loaded(engine, session_id),
                    component="sampling",
                    op="sampling.ensure_lora_loaded",
                    request_id=request_id,
                    attributes={"sampling_session_id": session_id},
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
                    # Get engine for session (already fetched above, but refetch to ensure exists)
                    engine_for_logprobs = await run_async_with_otel_span(
                        "sampling.get_engine_for_prompt_logprobs",
                        lambda: session_manager.get_engine_for_session(session_id),
                        component="sampling",
                        op="sampling.get_engine_for_prompt_logprobs",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id},
                    )
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
                    engine_for_topk = await run_async_with_otel_span(
                        "sampling.get_engine_for_prompt_topk",
                        lambda: session_manager.get_engine_for_session(session_id),
                        component="sampling",
                        op="sampling.get_engine_for_prompt_topk",
                        request_id=request_id,
                        attributes={"sampling_session_id": session_id},
                    )
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
            if usage_events:
                await _persist_usage_events(auth_ctx=auth_ctx, events=usage_events)

            # Compatibility: older tinker clients don't accept a top-level `type` field on SampleResponse.
            future_store.resolve(request_id, response.model_dump(exclude={"type"}))
            logger.info(f"Sampling completed: {len(sequences)} sequences generated")

        except asyncio.CancelledError:
            await _abort_engine_request(engine, request_id)
            await future_store.async_fail(request_id, "sampling task cancelled")
            logger.warning(
                "[sampling.asample] canceled request_id=%s session_id=%s next_action=%s",
                str(request_id),
                str(session_id),
                "caller_can_retry",
            )
            raise
        except Exception as e:
            await _abort_engine_request(engine, request_id)
            logger.exception(
                "[sampling.asample] failed request_id=%s session_id=%s failure_reason=%s error_type=%s next_action=%s",
                str(request_id),
                str(session_id),
                classify_failure_reason(e),
                type(e).__name__,
                "check_sampling_session_and_vllm_actor",
            )
            await future_store.async_fail(request_id, str(e))
    finally:
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

    is_local = False
    if session_manager is not None:
        is_local = session_manager.is_multi_lora_session(request.sampling_session_id) or (
            session_manager.get_engine(request.sampling_session_id) is not None
        )
    remote = None if is_local else await async_remote_sampling_session(request.sampling_session_id)
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
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    if session_manager.is_multi_lora_session(request.sampling_session_id):
        base_model = session_manager.get_session_base_model(request.sampling_session_id)
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
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_compute_logprobs_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    billing_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_compute_logprobs_result_bytes(request),
    )
    if not bool(reserve.get("ok")):
        record_sampling_admission_metric(
            route=_COMPUTE_LOGPROBS_ROUTE,
            decision="rejected",
            reason="capacity_rejected",
        )
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": "sampling.compute_logprobs"})
        get_session_base_model = getattr(session_manager, "get_session_base_model", None)
        base_model = (
            get_session_base_model(request.sampling_session_id)
            if callable(get_session_base_model)
            else None
        )
        await _enqueue_sampling_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="sampling.compute_logprobs",
            session_id=request.sampling_session_id,
            base_model=base_model,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="sampling.compute_logprobs",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra={"gateway_auth": billing_auth.__dict__} if billing_auth is not None else None,
            ),
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
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
    try:
        set_request_id(request_id)
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.sequence.to_token_ids()
        session_id = request.sampling_session_id
        session_manager.mark_session_inflight(session_id, +1)
        base_model = session_manager.get_session_base_model(session_id)

        async def _compute_logprobs_action():
            # Check if session uses multi-LoRA mode (includes base model sessions)
            is_multi_lora = session_manager.is_multi_lora_session(session_id)
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
                multi_lora_engine = await session_manager.get_engine_for_session(session_id)
                if multi_lora_engine is None:
                    raise RuntimeError(f"No engine found for session {session_id}")
                await _ensure_session_lora_loaded(multi_lora_engine, session_id)
                return await multi_lora_engine.compute_logprobs(
                    sampling_session_id=session_id,
                    prompt_ids=token_ids,
                    request_id=request_id,
                )

            engine = session_manager.get_engine(session_id)
            if engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")
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
        # Compatibility: older tinker clients don't accept a top-level `type` field on ComputeLogprobsResponse.
        future_store.resolve(request_id, response.model_dump(exclude={"type"}))
        logger.debug(f"Request {request_id} computed {len(logprobs)} logprobs")

    except Exception as e:
        logger.exception(
            "[sampling.compute_logprobs] failed request_id=%s session_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(session_id or request.sampling_session_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_sampling_session_and_token_length",
        )
        await future_store.async_fail(request_id, str(e))
    finally:
        if session_manager is not None and session_id is not None:
            session_manager.mark_session_inflight(session_id, -1)
