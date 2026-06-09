"""OpenAI-compatible routes backed by existing MinT sampling flows."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
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
from ..backend.task_state_store import task_futures
from ..runtime_env import env_get
from .sampling import build_sample_once_billing_observations, sample_once
from .service import ensure_sampling_session

router = APIRouter()
logger = logging.getLogger(__name__)


async def _append_billing_observations(observations) -> None:
    await task_futures.async_append_billing_outbox(observations, source="sync_http")


@dataclass
class _SessionCacheEntry:
    session_id: str
    base_model: str
    created_at: float
    last_used: float


_MAX_SESSION_CACHE_SIZE = int(
    env_get(os.environ, "MINT_OAI_SESSION_CACHE_MAX_SIZE", "1024") or 1024
)
_SESSION_CACHE_TTL_S = int(
    env_get(os.environ, "MINT_OAI_SESSION_CACHE_TTL_S", "3600") or 3600
)
_session_cache: OrderedDict[tuple[str | None, str], _SessionCacheEntry] = OrderedDict()
_session_lock = asyncio.Lock()
_tokenizer_cache: dict[str, Any] = {}
_tokenizer_locks: dict[str, asyncio.Lock] = {}
_tokenizer_locks_guard = asyncio.Lock()
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Secondary: ```json {...} ``` / ```json [...] ``` code blocks for tool payloads.
# Handles models whose native chat templates emit JSON code blocks instead of <tool_call> XML.
_CODE_BLOCK_TOOL_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _get_user_id(request: Request) -> str | None:
    user_data = getattr(request.state, "user_data", None)
    if user_data:
        return user_data.get("user_id")
    return None


def _error_response(
    *,
    message: str,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
):
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


def _list_supported_models() -> list[str]:
    from ..backend.model_registry import list_supported_models

    return list_supported_models()


def _oai_model_payload(model_id: str, *, created: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "mint",
    }


def _load_tokenizer_cpu(base_model: str):
    """Load tokenizer on the API host.

    This uses transformers' normal optional-backend detection instead of
    mutating ``sys.modules["torch"]`` process-wide.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model, local_files_only=True)


def _tokenizer_max_workers() -> int:
    raw = env_get(os.environ, "MINT_OAI_TOKENIZER_MAX_WORKERS", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


@lru_cache(maxsize=1)
def _get_tokenizer_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_tokenizer_max_workers(),
        thread_name_prefix="mint-oai-tokenizer",
    )


def shutdown_tokenizer_executor() -> None:
    if _get_tokenizer_executor.cache_info().currsize == 0:
        return
    executor = _get_tokenizer_executor()
    executor.shutdown(wait=False, cancel_futures=True)
    _get_tokenizer_executor.cache_clear()


async def _get_tokenizer_model_lock(base_model: str) -> asyncio.Lock:
    lock = _tokenizer_locks.get(base_model)
    if lock is not None:
        return lock
    async with _tokenizer_locks_guard:
        lock = _tokenizer_locks.get(base_model)
        if lock is None:
            lock = asyncio.Lock()
            _tokenizer_locks[base_model] = lock
        return lock


def preload_supported_tokenizers() -> dict[str, str]:
    failures: dict[str, str] = {}
    for base_model in _list_supported_models():
        if base_model in _tokenizer_cache:
            continue
        try:
            _tokenizer_cache[base_model] = _load_tokenizer_cpu(base_model)
        except Exception as exc:
            failures[base_model] = f"{type(exc).__name__}: {exc}"
    return failures


def _get_running_loop():
    return asyncio.get_running_loop()


async def _get_tokenizer(base_model: str):
    tokenizer = _tokenizer_cache.get(base_model)
    if tokenizer is not None:
        return tokenizer
    model_lock = await _get_tokenizer_model_lock(base_model)
    async with model_lock:
        tokenizer = _tokenizer_cache.get(base_model)
        if tokenizer is None:
            loop = _get_running_loop()
            tokenizer = await loop.run_in_executor(
                _get_tokenizer_executor(),
                _load_tokenizer_cpu,
                base_model,
            )
            _tokenizer_cache[base_model] = tokenizer
        return tokenizer


def _is_session_cache_expired(entry: _SessionCacheEntry, now: float) -> bool:
    if _SESSION_CACHE_TTL_S <= 0:
        return False
    return now - entry.last_used > _SESSION_CACHE_TTL_S


def _prune_session_cache(now: float) -> None:
    if not _session_cache:
        return
    if _SESSION_CACHE_TTL_S > 0:
        expired = [
            key
            for key, entry in _session_cache.items()
            if _is_session_cache_expired(entry, now)
        ]
        for key in expired:
            _session_cache.pop(key, None)
    if _MAX_SESSION_CACHE_SIZE > 0:
        while len(_session_cache) > _MAX_SESSION_CACHE_SIZE:
            _session_cache.popitem(last=False)


async def _get_or_create_cached_session(
    *, model_path: str, http_request: Request
) -> tuple[str, str]:
    user_id = _get_user_id(http_request)
    cache_key = (user_id, model_path)
    now = time.time()

    cached = _session_cache.get(cache_key)
    if cached is not None:
        if _is_session_cache_expired(cached, now):
            _session_cache.pop(cache_key, None)
        else:
            cached.last_used = now
            _session_cache.move_to_end(cache_key)
            return cached.session_id, cached.base_model

    async with _session_lock:
        _prune_session_cache(now)
        cached = _session_cache.get(cache_key)
        if cached is not None:
            if _is_session_cache_expired(cached, now):
                _session_cache.pop(cache_key, None)
            else:
                cached.last_used = now
                _session_cache.move_to_end(cache_key)
                return cached.session_id, cached.base_model
        session_id, base_model = await ensure_sampling_session(
            model_path=model_path, http_request=http_request
        )
        _session_cache[cache_key] = _SessionCacheEntry(
            session_id=session_id,
            base_model=base_model,
            created_at=now,
            last_used=now,
        )
        _session_cache.move_to_end(cache_key)
        # Evict the oldest entry when the cache exceeds the size limit.
        _prune_session_cache(now)
        return session_id, base_model


def _invalidate_cached_session(*, user_id: str | None, model_path: str) -> None:
    _session_cache.pop((user_id, model_path), None)


async def _is_remote_sampling_session(session_id: str) -> bool:
    try:
        from ..gateway import async_remote_sampling_session

        remote = await async_remote_sampling_session(session_id)
    except Exception:
        remote = None
    if remote is not None:
        return True
    try:
        from ..gateway import remote_sampling_session

        return remote_sampling_session(session_id) is not None
    except Exception:
        return False


def _should_retry_with_fresh_session(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return True
    if not isinstance(exc, HTTPException):
        return False
    if exc.status_code == 404:
        return True
    if exc.status_code < 500:
        return False
    detail = str(exc.detail).lower()
    return (
        "sampling session" in detail
        or "session " in detail
        or "request_id" in detail
        or "no engine found" in detail
        or "unknown request_id" in detail
        or "retrieve_future" in detail
        or "asample" in detail
    )


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
        lines.append(
            f"You must call the function `{function_name}` before giving a final answer."
        )
    return "\n".join(lines)


def _message_to_fallback_dict(message) -> dict[str, Any]:
    """Serialize a message to a plain-text dict for templates without native tool support.

    - assistant with tool_calls: fold calls into content as <tool_call> XML
    - tool role: convert to user message with <tool_result> wrapper
    - everything else: pass through (drop structured tool fields)
    """
    if message.role == "assistant" and message.tool_calls:
        tc_parts = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, ValueError):
                args = tc.function.arguments
            payload = {"name": tc.function.name, "arguments": args}
            tc_parts.append(
                f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
            )
        tc_text = "\n".join(tc_parts)
        prefix = message.content or ""
        combined = (prefix + "\n" + tc_text).strip() if prefix else tc_text
        return {"role": "assistant", "content": combined}

    if message.role == "tool":
        return {
            "role": "user",
            "content": f"<tool_result>{message.content}</tool_result>",
        }

    item: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        item["content"] = message.content
    return item


def _build_chat_template_messages(
    request: OAIChatCompletionRequest,
    *,
    include_tool_prompt: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    # Text to inject as (or prepend to) the system message.
    # Only inject in the fallback (text-prompt) path; native path lets the chat template handle it.
    extra_system: str | None = None
    if include_tool_prompt and request.tools:
        extra_system = _tool_prompt_text(request)

    # If user already has a system message we merge into it to avoid duplicate system blocks.
    first_is_system = bool(request.messages and request.messages[0].role == "system")
    if extra_system and not first_is_system:
        messages.append({"role": "system", "content": extra_system})

    for i, message in enumerate(request.messages):
        # Merge extra_system prefix into the user-supplied system message.
        if i == 0 and message.role == "system" and extra_system:
            merged = extra_system + "\n\n" + (message.content or "")
            messages.append({"role": "system", "content": merged})
            continue

        if include_tool_prompt:
            messages.append(_message_to_fallback_dict(message))
            continue

        # Native path: pass structured tool data; the chat template handles formatting.
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


def _parse_tool_call_payload(payload: str) -> OAIToolCall | None:
    """Parse a raw JSON string as a tool call. Returns None if invalid or missing required keys."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed tool_call JSON: %r", payload[:200])
        return None

    name = parsed.get("name")
    arguments = parsed.get("arguments", {})
    if not isinstance(name, str) or not name:
        logger.warning("Skipping tool_call with missing/invalid name: %r", parsed)
        return None

    arguments_json = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    return OAIToolCall(
        id=f"call_{uuid.uuid4().hex}",
        function=OAIFunctionCall(name=name, arguments=arguments_json),
    )


def _coerce_tool_calls(items: list[dict[str, Any]]) -> list[OAIToolCall]:
    tool_calls: list[OAIToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = (
            item.get("function") if isinstance(item.get("function"), dict) else None
        )
        name = None
        arguments = None
        if function is not None:
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = item.get("name")
            arguments = item.get("arguments")

        if not isinstance(name, str) or not name:
            continue

        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments or {}, ensure_ascii=False)

        tool_calls.append(
            OAIToolCall(
                id=item.get("id") or f"call_{uuid.uuid4().hex}",
                function=OAIFunctionCall(name=name, arguments=arguments_json),
            )
        )
    return tool_calls


def _tool_calls_from_json_blob(blob: str) -> list[OAIToolCall]:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            return _coerce_tool_calls(parsed["tool_calls"])
        if "name" in parsed:
            return _coerce_tool_calls([parsed])
        if isinstance(parsed.get("function_call"), dict):
            fn = parsed["function_call"]
            return _coerce_tool_calls(
                [{"name": fn.get("name"), "arguments": fn.get("arguments")}]
            )
    elif isinstance(parsed, list):
        return _coerce_tool_calls(parsed)
    return []


def _extract_tool_calls(text: str) -> tuple[str | None, list[OAIToolCall]]:
    # Primary: <tool_call>...</tool_call> XML (Qwen3, DeepSeek, and our fallback prompt path).
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        tool_calls = [
            tc
            for m in matches
            if (tc := _parse_tool_call_payload(m.group(1).strip())) is not None
        ]
        if not tool_calls:
            # All blocks were malformed – treat entire output as plain text.
            return text.strip() or None, []
        remaining = _TOOL_CALL_RE.sub("", text).strip()
        return remaining or None, tool_calls

    # Secondary: ```json {...} ``` or ```json [...] ``` code blocks.
    # Some model templates emit tool calls as JSON code blocks rather than <tool_call> XML.
    code_matches = list(_CODE_BLOCK_TOOL_RE.finditer(text))
    if code_matches:
        tool_calls: list[OAIToolCall] = []
        for match in code_matches:
            tool_calls.extend(_tool_calls_from_json_blob(match.group(1).strip()))
        if tool_calls:
            remaining = _CODE_BLOCK_TOOL_RE.sub("", text).strip()
            return remaining or None, tool_calls

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        tool_calls = _tool_calls_from_json_blob(stripped)
        if tool_calls:
            return None, tool_calls

    return stripped or None, []


def _invalid_tool_call_names(
    request: OAIChatCompletionRequest,
    tool_calls: list[OAIToolCall],
) -> list[str]:
    allowed_names = {tool.function.name for tool in request.tools or []}
    return sorted(
        {
            tool_call.function.name
            for tool_call in tool_calls
            if tool_call.function.name not in allowed_names
        }
    )


def _validate_tool_calls(
    request: OAIChatCompletionRequest,
    *,
    tool_calls: list[OAIToolCall],
) -> None:
    # tool_choice="none" means extraction is skipped upstream (tool_calls is always []) so no check needed.
    mode, function_name = _tool_choice_mode(request)
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
        raise HTTPException(
            status_code=400, detail="Model did not return a required tool call"
        )

    if mode == "function" and function_name is not None:
        if not tool_calls:
            raise HTTPException(
                status_code=400,
                detail=f"Model did not return the required tool call {function_name!r}",
            )
        wrong_tool_names = sorted(
            {
                tool_call.function.name
                for tool_call in tool_calls
                if tool_call.function.name != function_name
            }
        )
        if wrong_tool_names:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model returned tool calls outside required function {function_name!r}: "
                    f"{', '.join(wrong_tool_names)}"
                ),
            )


def _is_template_tools_error(exc: BaseException) -> bool:
    """Return True if exc is a known error from apply_chat_template rejecting the tools= kwarg."""
    if isinstance(exc, (TypeError, ValueError)):
        return True
    # jinja2.TemplateError — check by module to avoid a hard import dependency.
    tp = type(exc)
    return tp.__module__ is not None and tp.__module__.startswith("jinja2")


def _render_chat_prompt_token_ids(
    *,
    tokenizer,
    request: OAIChatCompletionRequest,
    base_model: str,
    force_tool_prompt: bool = False,
) -> list[int]:
    if force_tool_prompt and request.tools:
        fallback_messages = _build_chat_template_messages(
            request, include_tool_prompt=True
        )
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
    except Exception as exc:
        if effective_tools is None or not _is_template_tools_error(exc):
            raise
        # Template doesn't support tools= (TypeError, ValueError, jinja2 TemplateError, …).
        # Fall back to injecting tool descriptions as a system message in plain text.
        logger.warning(
            "apply_chat_template with tools= failed for %s (%s: %s); falling back to text prompt injection",
            base_model,
            type(exc).__name__,
            exc,
        )
        try:
            fallback_messages = _build_chat_template_messages(
                request, include_tool_prompt=True
            )
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


@router.get("/models")
async def list_models():
    """Return supported models in OpenAI /v1/models format.

    Reads from MINT_SUPPORTED_MODELS (or the built-in default list).
    """
    try:
        models = _list_supported_models()
        now = int(time.time())
        return {
            "object": "list",
            "data": [_oai_model_payload(m, created=now) for m in models],
        }
    except HTTPException as exc:
        return _error_response(message=str(exc.detail), status_code=exc.status_code)
    except Exception as exc:
        logger.exception("list_supported_models() failed for /models")
        return _error_response(message=f"{type(exc).__name__}: {exc}", status_code=500)


@router.get("/models/{model_id:path}")
async def retrieve_model(model_id: str):
    """Return a single supported model in OpenAI /v1/models/{id} format."""
    try:
        models = _list_supported_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        return _oai_model_payload(model_id, created=int(time.time()))
    except HTTPException as exc:
        return _error_response(message=str(exc.detail), status_code=exc.status_code)
    except Exception as exc:
        logger.exception("list_supported_models() failed for /models/%s", model_id)
        return _error_response(message=f"{type(exc).__name__}: {exc}", status_code=500)


@router.post("/completions", response_model=OAICompletionResponse)
async def completions(request: OAICompletionRequest, http_request: Request):
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=True is not supported")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")
        user_id = _get_user_id(http_request)
        sequence = None
        for attempt in range(2):
            sampling_session_id, base_model = await _get_or_create_cached_session(
                model_path=request.model,
                http_request=http_request,
            )
            tokenizer = await _get_tokenizer(base_model)
            prompt_token_ids = tokenizer.encode(
                request.prompt, add_special_tokens=False
            )
            try:
                sampling_request_id = f"oai_cmpl_{uuid.uuid4().hex}"
                sequence = await sample_once(
                    session_id=sampling_session_id,
                    token_ids=prompt_token_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    request_id=sampling_request_id,
                    http_request=http_request,
                    user_id=user_id,
                    bill_usage=False,
                )
                break
            except Exception as exc:
                if not _should_retry_with_fresh_session(exc):
                    raise
                _invalidate_cached_session(user_id=user_id, model_path=request.model)
                if attempt == 1:
                    raise
        text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        if not await _is_remote_sampling_session(sampling_session_id):
            billing_observations = build_sample_once_billing_observations(
                session_id=sampling_session_id,
                token_ids=prompt_token_ids,
                sequence=sequence,
                http_request=http_request,
                request_id=sampling_request_id,
                model=base_model,
            )
            if billing_observations:
                await _append_billing_observations(billing_observations)
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
                if not isinstance(prompt_token_ids, list) or (
                    prompt_token_ids and not isinstance(prompt_token_ids[0], int)
                ):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Tokenizer.apply_chat_template returned unexpected type {type(prompt_token_ids).__name__}; expected list[int]",
                    )
                try:
                    sampling_request_id = f"oai_chat_{uuid.uuid4().hex}"
                    sequence = await sample_once(
                        session_id=sampling_session_id,
                        token_ids=prompt_token_ids,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop,
                        request_id=sampling_request_id,
                        http_request=http_request,
                        user_id=user_id,
                        bill_usage=False,
                    )
                    break
                except Exception as exc:
                    if not _should_retry_with_fresh_session(exc):
                        raise
                    _invalidate_cached_session(
                        user_id=user_id, model_path=request.model
                    )
                    if attempt == 1:
                        raise

            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            if request.tools and request.tool_choice != "none":
                content, tool_calls = _extract_tool_calls(text)
                if not tool_calls and not force_tool_prompt and text.strip():
                    # Native template path: model output was non-empty but contained no
                    # recognisable tool call blocks. Could mean the model legitimately chose
                    # not to call any tool (auto mode), or that it uses an unrecognised format.
                    logger.debug(
                        "Native template path yielded no tool_calls for %s "
                        "(tool_choice=%r). Model may use an unrecognised output format. "
                        "Output[:300]: %r",
                        request.model,
                        request.tool_choice,
                        text[:300],
                    )
            else:
                content, tool_calls = text, []

            invalid_names = _invalid_tool_call_names(request, tool_calls)
            mode, function_name = _tool_choice_mode(request)
            wrong_function = (
                mode == "function"
                and function_name is not None
                and tool_calls
                and any(tc.function.name != function_name for tc in tool_calls)
            )
            should_retry = (
                not force_tool_prompt
                and request.tools
                and request.tool_choice != "none"
                and (
                    bool(invalid_names)
                    or (mode == "required" and not tool_calls)
                    or (
                        mode == "function"
                        and function_name is not None
                        and not tool_calls
                    )
                    or wrong_function
                )
            )
            if should_retry:
                logger.warning(
                    "Tool constraint not satisfied for model %s (mode=%s, invalid_names=%s, "
                    "tool_calls=%d, wrong_function=%s); retrying with explicit tool prompt",
                    request.model,
                    mode,
                    invalid_names,
                    len(tool_calls),
                    wrong_function,
                )
                continue

            _validate_tool_calls(request, tool_calls=tool_calls)
            finish_reason = (
                "tool_calls" if tool_calls else _finish_reason(sequence.stop_reason)
            )
            if not await _is_remote_sampling_session(sampling_session_id):
                billing_observations = build_sample_once_billing_observations(
                    session_id=sampling_session_id,
                    token_ids=prompt_token_ids,
                    sequence=sequence,
                    http_request=http_request,
                    request_id=sampling_request_id,
                    model=base_model,
                )
                if billing_observations:
                    await _append_billing_observations(billing_observations)
            return OAIChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    OAIChatCompletionChoice(
                        index=0,
                        message=OAIChatMessageResponse(
                            content=content, tool_calls=tool_calls or None
                        ),
                        finish_reason=finish_reason,
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
