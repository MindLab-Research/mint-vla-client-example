from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from mint_server.observability.logging_context import get_current_traceparent, get_trace_id
from mint_server.ray.runtime_env import env_get

logger = logging.getLogger(__name__)

_GATEWAY_REQUEST_ID_PREFIX = "gw:"

_http_clients_lock = asyncio.Lock()
_http_clients: dict[str, httpx.AsyncClient] = {}


@dataclass(frozen=True)
class Upstream:
    alias: str
    base_url: str
    auth_mode: str  # "pass_through" | "static_api_key" | "none"
    api_key: str | None = None


@dataclass(frozen=True)
class GatewayConfig:
    model_to_upstream: dict[str, str]
    upstreams: dict[str, Upstream]


_gateway_config: GatewayConfig | None = None


def get_gateway_config() -> GatewayConfig | None:
    global _gateway_config
    if _gateway_config is not None:
        return _gateway_config

    raw = (env_get(os.environ, "MINT_GATEWAY_CONFIG_JSON", "") or "").strip()
    if not raw:
        return None

    data = json.loads(raw)
    model_to_upstream = dict(
        data.get("model_to_upstream")
        or data.get("model_to_deployment_target")
        or data.get("model_to_target")
        or {}
    )

    upstreams_raw = data.get("upstreams") or data.get("deployment_targets") or data.get("targets") or {}
    upstreams: dict[str, Upstream] = {}
    for alias, u in upstreams_raw.items():
        base_url = str(u.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError(f"MINT_GATEWAY_CONFIG_JSON: upstream {alias!r} missing base_url")
        auth_mode = str(u.get("auth_mode") or "pass_through").strip().lower()
        api_key = u.get("api_key")
        if auth_mode == "static_api_key" and not api_key:
            raise ValueError(
                f"MINT_GATEWAY_CONFIG_JSON: upstream {alias!r} auth_mode=static_api_key requires api_key"
            )
        upstreams[str(alias)] = Upstream(
            alias=str(alias),
            base_url=base_url,
            auth_mode=auth_mode,
            api_key=str(api_key) if api_key else None,
        )

    _gateway_config = GatewayConfig(model_to_upstream=model_to_upstream, upstreams=upstreams)
    return _gateway_config


def upstream_for_model(model_name: str) -> Upstream | None:
    cfg = get_gateway_config()
    if cfg is None:
        return None
    alias = cfg.model_to_upstream.get(model_name)
    if not alias:
        return None
    up = cfg.upstreams.get(alias)
    if up is None:
        raise ValueError(f"MINT_GATEWAY_CONFIG_JSON: model {model_name!r} maps to unknown upstream {alias!r}")
    return up


def upstream_for_alias(alias: str) -> Upstream | None:
    cfg = get_gateway_config()
    if cfg is None:
        return None
    return cfg.upstreams.get(alias)


def encode_request_id(*, upstream_alias: str, upstream_request_id: str) -> str:
    return f"{_GATEWAY_REQUEST_ID_PREFIX}{upstream_alias}:{upstream_request_id}"


def decode_request_id(request_id: str) -> tuple[str, str] | None:
    if not request_id.startswith(_GATEWAY_REQUEST_ID_PREFIX):
        return None
    rest = request_id[len(_GATEWAY_REQUEST_ID_PREFIX) :]
    if ":" not in rest:
        return None
    alias, upstream_request_id = rest.split(":", 1)
    if not alias or not upstream_request_id:
        return None
    return alias, upstream_request_id


def _pick_auth_headers(*, incoming_headers: dict[str, str], upstream: Upstream) -> dict[str, str]:
    incoming_lower = {k.lower(): v for k, v in incoming_headers.items()}
    forwarded: dict[str, str] = {}
    user_agent = (incoming_headers.get("User-Agent") or incoming_lower.get("user-agent") or "").strip()
    if user_agent:
        forwarded["User-Agent"] = user_agent
    if upstream.auth_mode == "none":
        return forwarded
    if upstream.auth_mode == "static_api_key":
        return {**forwarded, "X-API-Key": upstream.api_key or ""}
    if upstream.auth_mode == "pass_through":
        api_key = (incoming_headers.get("X-API-Key") or incoming_lower.get("x-api-key") or "").strip()
        if api_key:
            return {**forwarded, "X-API-Key": api_key}
        auth = (incoming_headers.get("Authorization") or incoming_lower.get("authorization") or "").strip()
        if auth:
            return {**forwarded, "Authorization": auth}
        return forwarded
    raise ValueError(
        f"MINT_GATEWAY_CONFIG_JSON: unsupported auth_mode={upstream.auth_mode!r} for {upstream.alias!r}"
    )


def _pick_trace_headers(*, incoming_headers: dict[str, str]) -> dict[str, str]:
    incoming_lower = {k.lower(): v for k, v in incoming_headers.items()}
    forwarded: dict[str, str] = {}

    traceparent = (incoming_headers.get("traceparent") or incoming_lower.get("traceparent") or "").strip()
    if not traceparent:
        traceparent = str(get_current_traceparent() or "").strip()
    if traceparent:
        forwarded["traceparent"] = traceparent

    trace_id = (incoming_headers.get("X-Trace-Id") or incoming_lower.get("x-trace-id") or "").strip()
    if not trace_id:
        trace_id = str(get_trace_id() or "").strip()
    if trace_id:
        forwarded["X-Trace-Id"] = trace_id

    return forwarded


async def forward_json(
    *,
    upstream: Upstream,
    method: str,
    path: str,
    incoming_headers: dict[str, str],
    json_body: dict[str, Any] | None,
    timeout_s: float = 30.0,
) -> httpx.Response:
    headers = {
        **_pick_auth_headers(incoming_headers=incoming_headers, upstream=upstream),
        **_pick_trace_headers(incoming_headers=incoming_headers),
    }
    key = upstream.base_url
    async with _http_clients_lock:
        client = _http_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(base_url=upstream.base_url)
            _http_clients[key] = client
    return await client.request(method, path, headers=headers, json=json_body, timeout=timeout_s)


async def forward_request(
    *,
    upstream: Upstream,
    method: str,
    path: str,
    incoming_headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
    stream: bool = False,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    headers = {
        **_pick_auth_headers(incoming_headers=incoming_headers, upstream=upstream),
        **_pick_trace_headers(incoming_headers=incoming_headers),
    }
    client = httpx.AsyncClient(base_url=upstream.base_url, timeout=timeout_s)
    request = client.build_request(method, path, headers=headers, params=params)
    response = await client.send(request, stream=stream, follow_redirects=False)
    return client, response


async def close_http_clients() -> None:
    async with _http_clients_lock:
        clients = list(_http_clients.values())
        _http_clients.clear()
    for c in clients:
        await c.aclose()


async def forward_file(
    *,
    upstream: Upstream,
    path: str,
    incoming_headers: dict[str, str],
    file_path: str,
    field_name: str = "file",
    media_type: str = "application/gzip",
    timeout_s: float = 600.0,
) -> httpx.Response:
    url = f"{upstream.base_url}{path}"
    headers = {
        **_pick_auth_headers(incoming_headers=incoming_headers, upstream=upstream),
        **_pick_trace_headers(incoming_headers=incoming_headers),
    }
    filename = os.path.basename(file_path)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        with open(file_path, "rb") as f:
            files = {field_name: (filename, f, media_type)}
            return await client.post(url, headers=headers, files=files)


_remote_sampling_sessions: dict[str, tuple[str, str]] = {}  # sampling_session_id -> (upstream_alias, base_model)
_remote_training_models: dict[str, dict[str, str | None]] = {}  # model_id -> routing metadata
_pending_save_weights_for_sampler: dict[tuple[str, str], tuple[float, str]] = {}  # (up_alias, up_req_id) -> (ts, base_model)


def register_pending_save_weights_for_sampler_future(
    *, upstream_alias: str, upstream_request_id: str, base_model: str
) -> None:
    """Record that a pending save_weights_for_sampler future will return a sampling_session_id."""
    # Best-effort TTL cleanup.
    now = time.time()
    for k, (ts, _) in list(_pending_save_weights_for_sampler.items()):
        if now - ts > 3600:
            _pending_save_weights_for_sampler.pop(k, None)
    _pending_save_weights_for_sampler[(upstream_alias, upstream_request_id)] = (now, base_model)


def maybe_register_sampling_session_from_retrieve_future(
    *, upstream_alias: str, upstream_request_id: str, payload: Any
) -> None:
    """If retrieve_future returns an ephemeral sampling_session_id, register it for /asample routing."""
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "save_weights_for_sampler":
        return
    sampling_session_id = payload.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        return
    key = (upstream_alias, upstream_request_id)
    entry = _pending_save_weights_for_sampler.get(key)
    if entry is None:
        return
    _, base_model = entry
    register_remote_sampling_session(
        sampling_session_id=sampling_session_id,
        upstream_alias=upstream_alias,
        base_model=base_model,
    )
    _pending_save_weights_for_sampler.pop(key, None)


async def async_maybe_register_sampling_session_from_retrieve_future(
    *, upstream_alias: str, upstream_request_id: str, payload: Any
) -> None:
    """Async variant for request paths that should avoid sync gateway store lookups."""
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "save_weights_for_sampler":
        return
    sampling_session_id = payload.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        return
    key = (upstream_alias, upstream_request_id)
    entry = _pending_save_weights_for_sampler.get(key)
    if entry is None:
        return
    _, base_model = entry
    await async_register_remote_sampling_session(
        sampling_session_id=sampling_session_id,
        upstream_alias=upstream_alias,
        base_model=base_model,
    )
    _pending_save_weights_for_sampler.pop(key, None)


def register_remote_sampling_session(*, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
    _remote_sampling_sessions[sampling_session_id] = (upstream_alias, base_model)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        gateway_session_store.upsert_sampling_session(
            sampling_session_id=sampling_session_id,
            upstream_alias=upstream_alias,
            base_model=base_model,
        )
    except Exception:
        _remote_sampling_sessions.pop(sampling_session_id, None)
        logger.exception("gateway_session_store.upsert_sampling_session failed")
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_register_remote_sampling_session(
    *, sampling_session_id: str, upstream_alias: str, base_model: str
) -> None:
    _remote_sampling_sessions[sampling_session_id] = (upstream_alias, base_model)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        await gateway_session_store.async_upsert_sampling_session(
            sampling_session_id=sampling_session_id,
            upstream_alias=upstream_alias,
            base_model=base_model,
        )
    except Exception:
        _remote_sampling_sessions.pop(sampling_session_id, None)
        logger.exception("gateway_session_store.async_upsert_sampling_session failed")
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


def remote_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    cached = _remote_sampling_sessions.get(sampling_session_id)
    if cached is not None:
        return cached
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return None
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        info = gateway_session_store.get_sampling_session(sampling_session_id)
        if info is not None:
            _remote_sampling_sessions[sampling_session_id] = info
        return info
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_remote_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    cached = _remote_sampling_sessions.get(sampling_session_id)
    if cached is not None:
        return cached
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return None
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        info = await gateway_session_store.async_get_sampling_session(sampling_session_id)
        if info is not None:
            _remote_sampling_sessions[sampling_session_id] = info
        return info
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


def unregister_remote_sampling_session(sampling_session_id: str) -> None:
    _remote_sampling_sessions.pop(sampling_session_id, None)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        gateway_session_store.delete_sampling_session(sampling_session_id)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_unregister_remote_sampling_session(sampling_session_id: str) -> None:
    _remote_sampling_sessions.pop(sampling_session_id, None)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        await gateway_session_store.async_delete_sampling_session(sampling_session_id)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


def register_remote_training_model(
    *,
    model_id: str,
    upstream_alias: str,
    base_model: str,
    owner_id: str | None = None,
) -> None:
    info: dict[str, str | None] = {
        "upstream_alias": upstream_alias,
        "base_model": base_model,
        "owner_id": owner_id,
    }
    _remote_training_models[model_id] = info
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        gateway_session_store.upsert_training_model(
            model_id=model_id,
            upstream_alias=upstream_alias,
            base_model=base_model,
            owner_id=owner_id,
        )
    except Exception:
        _remote_training_models.pop(model_id, None)
        logger.exception("gateway_session_store.upsert_training_model failed")
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_register_remote_training_model(
    *,
    model_id: str,
    upstream_alias: str,
    base_model: str,
    owner_id: str | None = None,
) -> None:
    info: dict[str, str | None] = {
        "upstream_alias": upstream_alias,
        "base_model": base_model,
        "owner_id": owner_id,
    }
    _remote_training_models[model_id] = info
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        await gateway_session_store.async_upsert_training_model(
            model_id=model_id,
            upstream_alias=upstream_alias,
            base_model=base_model,
            owner_id=owner_id,
        )
    except Exception:
        _remote_training_models.pop(model_id, None)
        logger.exception("gateway_session_store.async_upsert_training_model failed")
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


def remote_training_model_info(model_id: str) -> dict[str, str | None] | None:
    cached = _remote_training_models.get(model_id)
    if cached is not None:
        return dict(cached)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return None
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        info = gateway_session_store.get_training_model_info(model_id)
        if info is not None:
            _remote_training_models[model_id] = dict(info)
            return dict(info)
        return None
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_remote_training_model_info(model_id: str) -> dict[str, str | None] | None:
    cached = _remote_training_models.get(model_id)
    if cached is not None:
        return dict(cached)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return None
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        info = await gateway_session_store.async_get_training_model_info(model_id)
        if info is not None:
            _remote_training_models[model_id] = dict(info)
            return dict(info)
        return None
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


def remote_training_model(model_id: str) -> tuple[str, str] | None:
    info = remote_training_model_info(model_id)
    if info is None:
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return upstream_alias, base_model


async def async_remote_training_model(model_id: str) -> tuple[str, str] | None:
    info = await async_remote_training_model_info(model_id)
    if info is None:
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return upstream_alias, base_model


def unregister_remote_training_model(model_id: str) -> None:
    _remote_training_models.pop(model_id, None)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        gateway_session_store.delete_training_model(model_id)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


async def async_unregister_remote_training_model(model_id: str) -> None:
    _remote_training_models.pop(model_id, None)
    cfg = get_gateway_config()
    if cfg is None or not cfg.model_to_upstream:
        return
    try:
        from fastapi import HTTPException

        from mint_server.backend.stores import gateway_session_store

        await gateway_session_store.async_delete_training_model(model_id)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Gateway session store unavailable",
        )


_cap_cache: dict[str, tuple[float, dict[str, int]]] = {}  # upstream_alias -> (ts, model_name->max_len)


async def get_upstream_capabilities(
    *, upstream: Upstream, incoming_headers: dict[str, str], cache_ttl_s: float = 5.0
) -> dict[str, int]:
    now = time.time()
    cached = _cap_cache.get(upstream.alias)
    if cached is not None:
        ts, caps = cached
        if now - ts <= cache_ttl_s:
            return caps

    resp = await forward_json(
        upstream=upstream,
        method="GET",
        path="/api/v1/get_server_capabilities",
        incoming_headers=incoming_headers,
        json_body=None,
        timeout_s=10.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    models = payload.get("supported_models") or []

    out: dict[str, int] = {}
    for m in models:
        name = m.get("model_name")
        max_len = m.get("max_context_length")
        if isinstance(name, str) and isinstance(max_len, int):
            out[name] = max_len

    _cap_cache[upstream.alias] = (now, out)
    return out
