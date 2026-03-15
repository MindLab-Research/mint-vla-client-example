"""OpenAI-compatible routes backed by existing MinT sampling flows."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..models.types import (
    OAIChatCompletionChoice,
    OAIChatCompletionRequest,
    OAIChatCompletionResponse,
    OAIChatMessageResponse,
    OAICompletionChoice,
    OAICompletionRequest,
    OAICompletionResponse,
    OAIUsage,
)
from .sampling import sample_once
from .service import ensure_sampling_session

router = APIRouter()

_session_cache: dict[tuple[str | None, str], tuple[str, str]] = {}
_session_lock = asyncio.Lock()
_tokenizer_cache: dict[str, Any] = {}
_tokenizer_lock = asyncio.Lock()


def _get_user_id(request: Request) -> str | None:
    user_data = getattr(request.state, "user_data", None)
    if user_data:
        return user_data.get("user_id")
    return None


def _error_response(*, message: str, status_code: int, error_type: str = "invalid_request_error", code: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def _load_tokenizer_cpu(base_model: str):
    """Load tokenizer without importing torch.

    The API server runs on CPU and must not load GPU/NVSHMEM libraries.
    transformers supports tokenizer-only mode when torch is unavailable,
    which is all we need here (encode + apply_chat_template).
    """
    import sys

    _sentinel = object()
    _prev_torch = sys.modules.get("torch", _sentinel)
    # Block torch import so transformers falls back to tokenizer-only mode.
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    finally:
        if _prev_torch is _sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _prev_torch


async def _get_tokenizer(base_model: str):
    tokenizer = _tokenizer_cache.get(base_model)
    if tokenizer is not None:
        return tokenizer
    async with _tokenizer_lock:
        tokenizer = _tokenizer_cache.get(base_model)
        if tokenizer is None:
            tokenizer = await asyncio.to_thread(_load_tokenizer_cpu, base_model)
            _tokenizer_cache[base_model] = tokenizer
        return tokenizer


async def _get_or_create_cached_session(*, model_path: str, http_request: Request) -> tuple[str, str]:
    user_id = _get_user_id(http_request)
    cache_key = (user_id, model_path)

    cached = _session_cache.get(cache_key)
    if cached is not None:
        return cached

    async with _session_lock:
        cached = _session_cache.get(cache_key)
        if cached is not None:
            return cached
        session_id, base_model = await ensure_sampling_session(model_path=model_path, http_request=http_request)
        _session_cache[cache_key] = (session_id, base_model)
        return session_id, base_model


def _finish_reason(stop_reason: str) -> str:
    if stop_reason in ("stop", "eos"):
        return "stop"
    return "length"


def _usage(*, prompt_tokens: int, completion_tokens: int) -> OAIUsage:
    return OAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


@router.post("/completions", response_model=OAICompletionResponse)
async def completions(request: OAICompletionRequest, http_request: Request):
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=True is not supported")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")
        user_id = _get_user_id(http_request)
        cache_key = (user_id, request.model)
        sequence = None
        for attempt in range(2):
            sampling_session_id, base_model = await _get_or_create_cached_session(
                model_path=request.model,
                http_request=http_request,
            )
            tokenizer = await _get_tokenizer(base_model)
            prompt_token_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
            try:
                sequence = await sample_once(
                    session_id=sampling_session_id,
                    token_ids=prompt_token_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    request_id=f"oai_cmpl_{uuid.uuid4().hex}",
                    http_request=http_request,
                    user_id=user_id,
                )
                break
            except RuntimeError:
                _session_cache.pop(cache_key, None)
                if attempt == 1:
                    raise
        text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        return OAICompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                OAICompletionChoice(
                    text=text,
                    index=0,
                    finish_reason=_finish_reason(sequence.stop_reason),
                )
            ],
            usage=_usage(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(sequence.tokens),
            ),
        )
    except HTTPException as exc:
        return _error_response(message=str(exc.detail), status_code=exc.status_code)
    except Exception as exc:
        return _error_response(message=f"{type(exc).__name__}: {exc}", status_code=500)


@router.post("/chat/completions", response_model=OAIChatCompletionResponse)
async def chat_completions(request: OAIChatCompletionRequest, http_request: Request):
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=True is not supported")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")
        user_id = _get_user_id(http_request)
        cache_key = (user_id, request.model)
        sequence = None
        for attempt in range(2):
            sampling_session_id, base_model = await _get_or_create_cached_session(
                model_path=request.model,
                http_request=http_request,
            )
            tokenizer = await _get_tokenizer(base_model)
            prompt_token_ids = tokenizer.apply_chat_template(
                [{"role": message.role, "content": message.content} for message in request.messages],
                tokenize=True,
                add_generation_prompt=True,
            )
            if not isinstance(prompt_token_ids, list) or (prompt_token_ids and not isinstance(prompt_token_ids[0], int)):
                raise HTTPException(
                    status_code=500,
                    detail=f"Tokenizer.apply_chat_template returned unexpected type {type(prompt_token_ids).__name__}; expected list[int]",
                )
            try:
                sequence = await sample_once(
                    session_id=sampling_session_id,
                    token_ids=prompt_token_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    request_id=f"oai_chat_{uuid.uuid4().hex}",
                    http_request=http_request,
                    user_id=user_id,
                )
                break
            except RuntimeError:
                _session_cache.pop(cache_key, None)
                if attempt == 1:
                    raise
        text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        return OAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                OAIChatCompletionChoice(
                    index=0,
                    message=OAIChatMessageResponse(content=text),
                    finish_reason=_finish_reason(sequence.stop_reason),
                )
            ],
            usage=_usage(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(sequence.tokens),
            ),
        )
    except HTTPException as exc:
        return _error_response(message=str(exc.detail), status_code=exc.status_code)
    except Exception as exc:
        return _error_response(message=f"{type(exc).__name__}: {exc}", status_code=500)
