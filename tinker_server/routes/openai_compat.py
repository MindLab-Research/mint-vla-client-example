"""OpenAI-compatible routes backed by existing MinT sampling flows."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
    OAIFunctionCall,
    OAICompletionChoice,
    OAICompletionRequest,
    OAICompletionResponse,
    OAIToolCall,
    OAIUsage,
)
from .sampling import sample_once
from .service import ensure_sampling_session

router = APIRouter()
logger = logging.getLogger(__name__)

_session_cache: dict[tuple[str | None, str], tuple[str, str]] = {}
_session_lock = asyncio.Lock()
_tokenizer_cache: dict[str, Any] = {}
_tokenizer_lock = asyncio.Lock()
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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


def _tool_choice_mode(request: OAIChatCompletionRequest) -> tuple[str, str | None]:
    tool_choice = request.tool_choice
    if tool_choice is None or tool_choice == "auto":
        return "auto", None
    if tool_choice == "none":
        return "none", None
    if tool_choice == "required":
        return "required", None
    function = tool_choice.get("function") if isinstance(tool_choice, dict) else None
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            return "function", name
    raise HTTPException(status_code=400, detail="Invalid tool_choice")


def _tool_prompt_text(request: OAIChatCompletionRequest) -> str:
    tool_payload = [
        {
            "type": tool.type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            },
        }
        for tool in request.tools or []
    ]
    mode, function_name = _tool_choice_mode(request)
    lines = [
        "Available tools:",
        json.dumps(tool_payload, ensure_ascii=False, indent=2),
        'When calling a tool, respond with one or more XML blocks exactly like <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>.',
        "The JSON inside each <tool_call> block must be valid.",
        "The `name` field must exactly match one of the declared function names above.",
        "Do not invent tool names from skills, capabilities, or prior context.",
    ]
    if mode == "required":
        lines.append("You must call at least one tool before giving a final answer.")
    elif mode == "function" and function_name is not None:
        lines.append(f"You must call the function `{function_name}` before giving a final answer.")
    elif mode == "none":
        lines.append("Do not call any tools.")
    return "\n".join(lines)


def _build_chat_template_messages(
    request: OAIChatCompletionRequest,
    *,
    include_tool_prompt: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    mode, function_name = _tool_choice_mode(request)

    if include_tool_prompt and request.tools:
        messages.append({"role": "system", "content": _tool_prompt_text(request)})
    elif mode == "required":
        messages.append({
            "role": "system",
            "content": "You must call at least one tool before giving a final answer.",
        })
    elif mode == "function" and function_name is not None:
        messages.append({
            "role": "system",
            "content": f"You must call the function `{function_name}` before giving a final answer.",
        })

    for message in request.messages:
        item: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        elif message.role == "assistant" and message.tool_calls:
            item["content"] = ""

        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        messages.append(item)
    return messages


def _extract_tool_calls(text: str) -> tuple[str | None, list[OAIToolCall]]:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return stripped or None, []

    tool_calls: list[OAIToolCall] = []
    for match in matches:
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            stripped = text.strip()
            return stripped or None, []

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or not name:
            stripped = text.strip()
            return stripped or None, []

        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False)

        tool_calls.append(
            OAIToolCall(
                id=f"call_{uuid.uuid4().hex}",
                function=OAIFunctionCall(name=name, arguments=arguments_json),
            )
        )

    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return remaining or None, tool_calls


def _invalid_tool_call_names(
    request: OAIChatCompletionRequest,
    tool_calls: list[OAIToolCall],
) -> list[str]:
    allowed_names = {tool.function.name for tool in request.tools or []}
    return sorted({tool_call.function.name for tool_call in tool_calls if tool_call.function.name not in allowed_names})


def _validate_tool_calls(
    request: OAIChatCompletionRequest,
    *,
    tool_calls: list[OAIToolCall],
) -> None:
    mode, function_name = _tool_choice_mode(request)
    if mode == "none":
        if tool_calls:
            raise HTTPException(status_code=400, detail="tool_choice='none' forbids tool calls")
        return

    if not request.tools:
        return

    if request.parallel_tool_calls is False and len(tool_calls) > 1:
        raise HTTPException(
            status_code=400,
            detail="parallel_tool_calls=False but model returned multiple tool calls",
        )

    invalid_names = _invalid_tool_call_names(request, tool_calls)
    if invalid_names:
        raise HTTPException(
            status_code=400,
            detail=f"Model returned undeclared tool calls: {', '.join(invalid_names)}",
        )

    if mode == "required" and not tool_calls:
        raise HTTPException(status_code=400, detail="Model did not return a required tool call")

    if mode == "function" and function_name is not None:
        if not tool_calls:
            raise HTTPException(
                status_code=400,
                detail=f"Model did not return the required tool call {function_name!r}",
            )
        wrong_tool_names = sorted({tool_call.function.name for tool_call in tool_calls if tool_call.function.name != function_name})
        if wrong_tool_names:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model returned tool calls outside required function {function_name!r}: "
                    f"{', '.join(wrong_tool_names)}"
                ),
            )


def _render_chat_prompt_token_ids(
    *,
    tokenizer,
    request: OAIChatCompletionRequest,
    base_model: str,
    force_tool_prompt: bool = False,
) -> list[int]:
    if force_tool_prompt and request.tools:
        fallback_messages = _build_chat_template_messages(request, include_tool_prompt=True)
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    chat_messages = _build_chat_template_messages(request)
    effective_tools = None
    if request.tool_choice != "none" and request.tools:
        effective_tools = [tool.model_dump() for tool in request.tools]

    try:
        return tokenizer.apply_chat_template(
            chat_messages,
            tools=effective_tools,
            tokenize=True,
            add_generation_prompt=True,
        )
    except TypeError as exc:
        if effective_tools is None:
            raise
        try:
            fallback_messages = _build_chat_template_messages(request, include_tool_prompt=True)
            return tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Tool calling is not supported by tokenizer chat template for base_model "
                    f"{base_model}: {type(fallback_exc).__name__}: {fallback_exc}"
                ),
            ) from fallback_exc
    except Exception as exc:
        if effective_tools is not None:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Tool calling is not supported by tokenizer chat template for base_model "
                    f"{base_model}: {type(exc).__name__}: {exc}"
                ),
            ) from exc
        raise


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
        for prompt_attempt in range(2):
            force_tool_prompt = prompt_attempt == 1
            sequence = None
            for attempt in range(2):
                sampling_session_id, base_model = await _get_or_create_cached_session(
                    model_path=request.model,
                    http_request=http_request,
                )
                tokenizer = await _get_tokenizer(base_model)
                prompt_token_ids = _render_chat_prompt_token_ids(
                    tokenizer=tokenizer,
                    request=request,
                    base_model=base_model,
                    force_tool_prompt=force_tool_prompt,
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
            if request.tools and request.tool_choice != "none":
                content, tool_calls = _extract_tool_calls(text)
            else:
                content, tool_calls = text, []

            invalid_names = _invalid_tool_call_names(request, tool_calls)
            if invalid_names and request.tools and request.tool_choice != "none" and not force_tool_prompt:
                logger.warning(
                    "Model returned undeclared tool calls %s for model %s; retrying with explicit tool prompt",
                    invalid_names,
                    request.model,
                )
                continue

            _validate_tool_calls(request, tool_calls=tool_calls)
            finish_reason = "tool_calls" if tool_calls else _finish_reason(sequence.stop_reason)
            return OAIChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    OAIChatCompletionChoice(
                        index=0,
                        message=OAIChatMessageResponse(content=content, tool_calls=tool_calls or None),
                        finish_reason=finish_reason,
                    )
                ],
                usage=_usage(
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=len(sequence.tokens),
                ),
            )

        raise HTTPException(status_code=500, detail="tool calling retry exhausted unexpectedly")
    except HTTPException as exc:
        return _error_response(message=str(exc.detail), status_code=exc.status_code)
    except Exception as exc:
        return _error_response(message=f"{type(exc).__name__}: {exc}", status_code=500)
