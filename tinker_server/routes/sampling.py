"""Sampling routes for text generation.

Endpoints:
- POST /asample: Async sample request (returns future)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks

from ..backend.future_store import future_store
from ..models.types import (
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

        # Get engine for this session
        engine = session_manager.get_engine(request.sampling_session_id)
        if engine is None:
            raise RuntimeError(
                f"No engine found for session {request.sampling_session_id}"
            )

        token_ids = request.prompt.to_token_ids()

        # Generate for each sample
        sequences = []
        for i in range(request.num_samples):
            # NOTE: max_tokens parameter is passed here but currently ignored by verl.
            # verl computes max_tokens as (max_model_len - prompt_len) internally.
            result = await engine.generate(
                prompt_ids=token_ids,
                request_id=f"{request_id}_{i}",
                max_tokens=request.sampling_params.max_tokens,
                temperature=request.sampling_params.temperature,
                top_k=request.sampling_params.top_k,
                top_p=request.sampling_params.top_p,
                logprobs=True,
            )

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
                    logprobs=result.log_probs,
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
