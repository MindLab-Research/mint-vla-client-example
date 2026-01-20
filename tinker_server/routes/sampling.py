"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

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
_inflight_sample_tasks = 0


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
    try:
        try:
            if session_manager is None:
                raise RuntimeError("Session manager not initialized")

            token_ids = request.prompt.to_token_ids()
            session_id = request.get_session_id()  # Supports both sampling_session_id and model_id

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
            sequences = []
            for i in range(request.num_samples):
                if is_multi_lora:
                    # Multi-LoRA mode: handles both LoRA and base model sessions
                    # Get engine for this session's model (dynamically creates if needed)
                    multi_lora_engine = await session_manager.get_engine_for_session(session_id)
                    if multi_lora_engine is None:
                        raise RuntimeError(f"No engine found for session {session_id}")

                    result = await multi_lora_engine.generate(
                        sampling_session_id=session_id,
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_{i}",
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
                        raise RuntimeError(
                            f"No engine found for session {session_id}"
                        )

                    result = await engine.generate(
                        prompt_ids=token_ids,
                        request_id=f"{request_id}_{i}",
                        max_tokens=request.sampling_params.max_tokens,
                        temperature=request.sampling_params.temperature,
                        top_k=request.sampling_params.top_k,
                        top_p=request.sampling_params.top_p,
                        logprobs=True,
                    )

                # Normalize logprobs attribute name (multi-LoRA uses 'logprobs', legacy uses 'log_probs')
                logprobs = getattr(result, 'logprobs', None) or getattr(result, 'log_probs', None)

                # Infer stop reason: check if EOS tokens are present
                # verl's TokenOutput doesn't include finish_reason, so we infer it
                # Common EOS tokens: 151645 (<|im_end|>), 151643 (<|endoftext|>)
                eos_tokens = {151645, 151643}
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
                response.topk_prompt_logprobs = list(computed_topk)

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
    try:
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.sequence.to_token_ids()
        session_id = request.sampling_session_id

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
