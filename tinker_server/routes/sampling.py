"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
- POST /compute_logprobs: Compute logprobs for a sequence (returns future)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks

from ..backend.future_store import future_store
from ..models.types import (
    ComputeLogprobsRequest,
    ComputeLogprobsResponse,
    SampledSequence,
    SampleRequest,
    SampleResponse,
    UntypedAPIFuture,
)

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None


@router.post("/asample")
async def asample(
    request: SampleRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Submit an async sampling request.

    The request is processed in the background. Use /retrieve_future
    with the returned request_id to get results.
    """
    request_id = future_store.create()
    background_tasks.add_task(_do_sample, request_id, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_sample(request_id: str, request: SampleRequest) -> None:
    """Background task to perform sampling."""
    try:
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.prompt.to_token_ids()
        session_id = request.sampling_session_id

        # Check if this session uses multi-LoRA mode
        is_multi_lora = session_manager.is_multi_lora_session(session_id)

        # Generate for each sample
        sequences = []
        for i in range(request.num_samples):
            if is_multi_lora:
                # Multi-LoRA mode: use shared engine with session-specific LoRA
                multi_lora_engine = session_manager.get_multi_lora_engine()
                if multi_lora_engine is None:
                    raise RuntimeError("Multi-LoRA engine not initialized")

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
        if request.prompt_logprobs:
            # For MVP, prompt logprobs require a separate forward pass
            # TODO: implement prompt logprobs using vLLM's prompt_logprobs feature
            response.prompt_logprobs = None

        future_store.resolve(request_id, response.model_dump())
        logger.debug(f"Request {request_id} completed with {len(sequences)} sequences")

    except Exception as e:
        logger.exception(f"Request {request_id} failed: {e}")
        future_store.fail(request_id, str(e))


@router.post("/compute_logprobs")
async def compute_logprobs(
    request: ComputeLogprobsRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Compute logprobs for a sequence.

    Returns logprobs[i] = log P(token[i+1] | token[0:i+1]).
    Output length is len(sequence) - 1.
    """
    request_id = future_store.create()
    background_tasks.add_task(_do_compute_logprobs, request_id, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_compute_logprobs(
    request_id: str, request: ComputeLogprobsRequest
) -> None:
    """Background task to compute logprobs."""
    try:
        if session_manager is None:
            raise RuntimeError("Session manager not initialized")

        token_ids = request.sequence.to_token_ids()
        session_id = request.sampling_session_id

        # Check if this session uses multi-LoRA mode
        is_multi_lora = session_manager.is_multi_lora_session(session_id)

        if is_multi_lora:
            # Multi-LoRA mode: use shared engine with session-specific LoRA
            multi_lora_engine = session_manager.get_multi_lora_engine()
            if multi_lora_engine is None:
                raise RuntimeError("Multi-LoRA engine not initialized")

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
