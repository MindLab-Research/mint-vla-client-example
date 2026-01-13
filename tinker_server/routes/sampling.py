"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Request

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
    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    background_tasks.add_task(_do_sample, request_id, request, user_id)
    return UntypedAPIFuture(request_id=request_id)


async def _do_sample(
    request_id: str, request: SampleRequest, user_id: str | None
) -> None:
    """Background task to perform sampling."""
    try:
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.prompt.to_token_ids()
        session_id = request.get_session_id()  # Supports both sampling_session_id and model_id

        # Handle include_prompt_logprobs alias
        want_prompt_logprobs = request.prompt_logprobs or request.include_prompt_logprobs

        # Check if session uses multi-LoRA mode (includes base model sessions)
        is_multi_lora = session_manager.is_multi_lora_session(session_id)

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
            # Use compute_logprobs to get prompt log probabilities
            # compute_logprobs returns len(sequence)-1 values: logprobs[i] = log P(token[i+1] | token[0:i+1])
            # The API expects len(sequence) values with prompt_logprobs[0] as placeholder (no conditioning)
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
            # Prepend 0.0 for first token (no log probability for unconditional token)
            response.prompt_logprobs = [0.0] + computed_logprobs

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
            # Prepend empty dict for first token (no prior context)
            response.topk_prompt_logprobs = [{}] + list(computed_topk)

        future_store.resolve(request_id, response.model_dump())
        logger.debug(f"Request {request_id} completed with {len(sequences)} sequences")

        # Log usage
        if user_id:
            total_tokens = len(token_ids) + sum(len(seq.tokens) for seq in sequences)
            get_usage_logger().log(
                user_id=user_id,
                operation_type="sample",
                model_name=session_id,  # Use session_id as model identifier
                token_count=total_tokens,
                session_id=session_id,
                request_id=request_id,
            )

    except Exception as e:
        logger.exception(f"Request {request_id} failed: {e}")
        future_store.fail(request_id, str(e))


@router.post("/compute_logprobs")
async def compute_logprobs(
    request: ComputeLogprobsRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Compute logprobs for a sequence.

    Returns logprobs[i] = log P(token[i+1] | token[0:i+1]).
    Output length is len(sequence) - 1.
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

        # Log usage
        if user_id:
            get_usage_logger().log(
                user_id=user_id,
                operation_type="compute_logprobs",
                model_name=session_id,
                token_count=len(token_ids),
                session_id=session_id,
                request_id=request_id,
            )

    except Exception as e:
        logger.exception(f"Request {request_id} failed: {e}")
        future_store.fail(request_id, str(e))
