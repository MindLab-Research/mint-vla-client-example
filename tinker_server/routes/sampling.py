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
    from ..backend.verl_inference import VerlInferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Global engine reference (set by app lifespan)
verl_engine: VerlInferenceEngine | None = None


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
        if verl_engine is None:
            raise RuntimeError("Inference engine not initialized")

        token_ids = request.prompt.to_token_ids()

        # Generate for each sample
        sequences = []
        for i in range(request.num_samples):
            result = await verl_engine.generate(
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
