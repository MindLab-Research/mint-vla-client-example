"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

import asyncio
import array
import hashlib
import os
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..backend.future_store import future_store
from ..models.types import (
    ComputeLogprobsRequest,
    ComputeLogprobsResponse,
    SampledSequence,
    SampleRequest,
    SampleResponse,
    UntypedAPIFuture,
)
from ..usage_logger import get_usage_logger

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None

_SAMPLING_BACKPRESSURE_HEADER = "X-Tinker-Sampling-Backpressure"
_MAX_INFLIGHT_SAMPLE_TASKS = int(os.environ.get("TINKER_MAX_INFLIGHT_SAMPLE_TASKS", "64"))
_MAX_CONCURRENT_SAMPLES_PER_REQUEST = int(os.environ.get("TINKER_MAX_CONCURRENT_SAMPLES_PER_REQUEST", "8"))
_inflight_sample_tasks = 0

_SAMPLE_COALESCE = os.environ.get("TINKER_SAMPLE_COALESCE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
_SAMPLE_COALESCE_WINDOW_MS = float(os.environ.get("TINKER_SAMPLE_COALESCE_WINDOW_MS", "2.0"))
_SAMPLE_COALESCE_MAX_BATCH = int(os.environ.get("TINKER_SAMPLE_COALESCE_MAX_BATCH", "32"))
_sample_coalesce_lock = asyncio.Lock()
_sample_coalesce_groups: dict[tuple, dict] = {}


def _prompt_fingerprint(token_ids: list[int]) -> bytes:
    # 32k token prompts are common; collisions must be negligible but hashing cost is amortized by prefill.
    a = array.array("I", token_ids)
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest()


async def _coalesced_generate_one(
    *,
    engine,
    sampling_session_id: str,
    prompt_ids: list[int],
    request_id: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    key = (
        sampling_session_id,
        _prompt_fingerprint(prompt_ids),
        int(max_tokens),
        float(temperature),
        int(top_k),
        float(top_p),
    )

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    do_flush_now = False
    async with _sample_coalesce_lock:
        g = _sample_coalesce_groups.get(key)
        if g is None:
            g = {
                "engine": engine,
                "sampling_session_id": sampling_session_id,
                "prompt_ids": prompt_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "leader_request_id": request_id,
                "waiters": [],
                "flush_task": None,
            }
            _sample_coalesce_groups[key] = g
        g["waiters"].append(fut)
        if len(g["waiters"]) >= _SAMPLE_COALESCE_MAX_BATCH:
            do_flush_now = True
        if g["flush_task"] is None:
            delay_s = 0.0 if do_flush_now else max(0.0, _SAMPLE_COALESCE_WINDOW_MS / 1000.0)
            g["flush_task"] = asyncio.create_task(_flush_coalesced_group(key, delay_s))

    return await fut


async def _flush_coalesced_group(key: tuple, delay_s: float) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    async with _sample_coalesce_lock:
        g = _sample_coalesce_groups.pop(key, None)
    if g is None:
        return

    waiters = list(g["waiters"])
    if not waiters:
        return

    try:
        if len(waiters) == 1:
            res = await g["engine"].generate(
                sampling_session_id=g["sampling_session_id"],
                prompt_ids=g["prompt_ids"],
                request_id=g["leader_request_id"],
                max_tokens=g["max_tokens"],
                temperature=g["temperature"],
                top_k=g["top_k"],
                top_p=g["top_p"],
                logprobs=True,
            )
            if not waiters[0].done():
                waiters[0].set_result(res)
        else:
            results = await g["engine"].generate_many(
                sampling_session_id=g["sampling_session_id"],
                prompt_ids=g["prompt_ids"],
                request_id=f"{g['leader_request_id']}_coalesced",
                num_samples=len(waiters),
                max_tokens=g["max_tokens"],
                temperature=g["temperature"],
                top_k=g["top_k"],
                top_p=g["top_p"],
                logprobs=True,
            )
            if len(results) != len(waiters):
                raise RuntimeError(f"coalesce: got {len(results)} results for {len(waiters)} waiters")
            for fut, res in zip(waiters, results):
                if not fut.done():
                    fut.set_result(res)
    except Exception as e:
        for fut in waiters:
            if not fut.done():
                fut.set_exception(e)


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


@router.post("/asample")
async def asample(
    request: SampleRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Submit an async sampling request.

    The request is processed in the background. Use /retrieve_future
    with the returned request_id to get results.
    """
    global _inflight_sample_tasks
    if _should_backpressure(http_request):
        raise HTTPException(status_code=429, detail="Sampling backpressure: server overloaded")
    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    _inflight_sample_tasks += 1
    background_tasks.add_task(_do_sample, request_id, request, user_id)
    return UntypedAPIFuture(request_id=request_id)


async def _do_sample(
    request_id: str, request: SampleRequest, user_id: str | None
) -> None:
    """Background task to perform sampling."""
    global _inflight_sample_tasks
    session_id: str | None = None
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
                if base_model:
                    try:
                        from ..backend.model_registry import get_model_config, normalize_model_name

                        cfg = get_model_config(normalize_model_name(base_model))
                        max_model_len = int(cfg.max_model_len)
                    except Exception:
                        max_model_len = None
                    if max_model_len is not None and len(token_ids) > max_model_len:
                        raise ValueError(
                            f"Prompt length {len(token_ids)} exceeds max_model_len {max_model_len} "
                            f"for model {base_model}"
                        )

            # Generate for each sample
            if request.num_samples < 1:
                raise ValueError(f"num_samples must be >= 1 (got {request.num_samples})")

            if is_multi_lora:
                # Multi-LoRA mode: handles both LoRA and base model sessions
                # Get engine once; it is shared across samples.
                engine = await session_manager.get_engine_for_session(session_id)
                if engine is None:
                    raise RuntimeError(f"No engine found for session {session_id}")

                if request.num_samples == 1:
                    if (
                        _SAMPLE_COALESCE
                        and (not want_prompt_logprobs)
                        and request.topk_prompt_logprobs == 0
                        and _SAMPLE_COALESCE_MAX_BATCH > 1
                    ):
                        one = await _coalesced_generate_one(
                            engine=engine,
                            sampling_session_id=session_id,
                            prompt_ids=token_ids,
                            request_id=request_id,
                            max_tokens=request.sampling_params.max_tokens,
                            temperature=request.sampling_params.temperature,
                            top_k=request.sampling_params.top_k,
                            top_p=request.sampling_params.top_p,
                        )
                        results = [one]
                    else:
                        results = [
                            await engine.generate(
                                sampling_session_id=session_id,
                                prompt_ids=token_ids,
                                request_id=request_id,
                                max_tokens=request.sampling_params.max_tokens,
                                temperature=request.sampling_params.temperature,
                                top_k=request.sampling_params.top_k,
                                top_p=request.sampling_params.top_p,
                                logprobs=True,
                            )
                        ]
                else:
                    gen_many = getattr(engine, "generate_many", None)
                    if gen_many is None:
                        raise RuntimeError(f"Engine for session {session_id} does not support generate_many()")
                    results = await gen_many(
                        sampling_session_id=session_id,
                        prompt_ids=token_ids,
                        request_id=request_id,
                        num_samples=request.num_samples,
                        max_tokens=request.sampling_params.max_tokens,
                        temperature=request.sampling_params.temperature,
                        top_k=request.sampling_params.top_k,
                        top_p=request.sampling_params.top_p,
                        logprobs=True,
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
            eos_tokens = {151645, 151643}
            for result in results:
                # Normalize logprobs attribute name (multi-LoRA uses 'logprobs', legacy uses 'log_probs')
                logprobs = getattr(result, "logprobs", None) or getattr(result, "log_probs", None)

                # Infer stop reason: check if EOS tokens are present
                # verl's TokenOutput doesn't include finish_reason, so we infer it
                if result.token_ids and result.token_ids[-1] in eos_tokens:
                    stop_reason = "stop"
                else:
                    stop_reason = "length"

                sequences.append(
                    SampledSequence(
                        tokens=result.token_ids,
                        logprobs=logprobs,
                        stop_reason=stop_reason,
                    )
                )

            # Build response
            response = SampleResponse(sequences=sequences)

            # Handle prompt logprobs if requested
            if want_prompt_logprobs:
                if is_multi_lora:
                    # Get engine for session (already fetched above, but refetch to ensure exists)
                    engine_for_logprobs = await session_manager.get_engine_for_session(session_id)
                    if engine_for_logprobs is None:
                        raise RuntimeError(f"No engine found for session {session_id}")
                    computed_logprobs = await engine_for_logprobs.compute_logprobs(
                        sampling_session_id=session_id,
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_prompt_logprobs",
                    )
                else:
                    computed_logprobs = await engine.compute_logprobs(
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_prompt_logprobs",
                    )
                response.prompt_logprobs = computed_logprobs

            # Handle top-K prompt logprobs if requested
            if request.topk_prompt_logprobs > 0:
                if is_multi_lora:
                    engine_for_topk = await session_manager.get_engine_for_session(session_id)
                    if engine_for_topk is None:
                        raise RuntimeError(f"No engine found for session {session_id}")
                    computed_topk = await engine_for_topk.compute_topk(
                        sampling_session_id=session_id,
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_topk",
                        k=request.topk_prompt_logprobs,
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

            future_store.resolve(request_id, response.model_dump())
            logger.debug(f"Request {request_id} completed with {len(sequences)} sequences")

            # Log usage - separate prefill and sampling tokens
            if user_id:
                prefill_tokens = len(token_ids)
                sampling_tokens = sum(len(seq.tokens) for seq in sequences)

                # Log prefill (input prompt) tokens
                get_usage_logger().log(
                    user_id=user_id,
                    operation_type="sample_prefill",
                    model_name=session_id,  # Use session_id as model identifier
                    token_count=prefill_tokens,
                    session_id=session_id,
                    request_id=request_id,
                )

                # Log sampling (generated) tokens
                get_usage_logger().log(
                    user_id=user_id,
                    operation_type="sample_generation",
                    model_name=session_id,
                    token_count=sampling_tokens,
                    session_id=session_id,
                    request_id=request_id,
                )

        except Exception as e:
            logger.exception(f"Request {request_id} failed: {e}")
            future_store.fail(request_id, str(e))
    finally:
        if session_manager is not None and session_id is not None:
            session_manager.mark_session_inflight(session_id, -1)
        _inflight_sample_tasks -= 1


@router.post("/compute_logprobs")
async def compute_logprobs(
    request: ComputeLogprobsRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Compute logprobs for a sequence.

    Returns a list of length len(sequence), where:
    - logprobs[0] is None (first token has no conditioning context)
    - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1
    """
    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    background_tasks.add_task(_do_compute_logprobs, request_id, request, user_id)
    return UntypedAPIFuture(request_id=request_id)


async def _do_compute_logprobs(
    request_id: str, request: ComputeLogprobsRequest, user_id: str | None
) -> None:
    """Background task to compute logprobs."""
    session_id: str | None = None
    try:
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.sequence.to_token_ids()
        session_id = request.sampling_session_id
        session_manager.mark_session_inflight(session_id, +1)

        # Check if session uses multi-LoRA mode (includes base model sessions)
        is_multi_lora = session_manager.is_multi_lora_session(session_id)
        if is_multi_lora:
            base_model = session_manager.get_session_base_model(session_id)
            if base_model:
                try:
                    from ..backend.model_registry import get_model_config, normalize_model_name

                    cfg = get_model_config(normalize_model_name(base_model))
                    max_model_len = int(cfg.max_model_len)
                except Exception:
                    max_model_len = None
                if max_model_len is not None and len(token_ids) > max_model_len:
                    raise ValueError(
                        f"Prompt length {len(token_ids)} exceeds max_model_len {max_model_len} "
                        f"for model {base_model}"
                    )

        if is_multi_lora:
            # Multi-LoRA mode: handles both LoRA and base model sessions
            # Get engine for this session's model (dynamically creates if needed)
            multi_lora_engine = await session_manager.get_engine_for_session(session_id)
            if multi_lora_engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")

            logprobs = await multi_lora_engine.compute_logprobs(
                sampling_session_id=session_id,
                prompt_ids=token_ids,
                request_id=request_id,
            )
        else:
            # Legacy mode: per-session engine
            engine = session_manager.get_engine(session_id)
            if engine is None:
                raise RuntimeError(f"No engine found for session {session_id}")

            logprobs = await engine.compute_logprobs(
                prompt_ids=token_ids,
                request_id=request_id,
            )

        response = ComputeLogprobsResponse(logprobs=logprobs)
        future_store.resolve(request_id, response.model_dump())
        logger.debug(
            f"Request {request_id} computed {len(logprobs)} logprobs"
        )

    except Exception as e:
        logger.exception(f"Request {request_id} failed: {e}")
        future_store.fail(request_id, str(e))
    finally:
        if session_manager is not None and session_id is not None:
            session_manager.mark_session_inflight(session_id, -1)
