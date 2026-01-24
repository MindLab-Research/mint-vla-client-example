from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

_GATEWAY_REQUEST_ID_PREFIX = "gw:"


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

    raw = os.environ.get("TINKER_GATEWAY_CONFIG_JSON", "").strip()
    if not raw:
        return None

    data = json.loads(raw)
    model_to_upstream = dict(data.get("model_to_upstream") or {})

    upstreams_raw = data.get("upstreams") or {}
    upstreams: dict[str, Upstream] = {}
    for alias, u in upstreams_raw.items():
        base_url = str(u.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError(f"TINKER_GATEWAY_CONFIG_JSON: upstream {alias!r} missing base_url")
        auth_mode = str(u.get("auth_mode") or "pass_through").strip().lower()
        api_key = u.get("api_key")
        if auth_mode == "static_api_key" and not api_key:
            raise ValueError(
                f"TINKER_GATEWAY_CONFIG_JSON: upstream {alias!r} auth_mode=static_api_key requires api_key"
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
        raise ValueError(f"TINKER_GATEWAY_CONFIG_JSON: model {model_name!r} maps to unknown upstream {alias!r}")
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
    if upstream.auth_mode == "none":
        return {}
    if upstream.auth_mode == "static_api_key":
        return {"X-API-Key": upstream.api_key or ""}
    if upstream.auth_mode == "pass_through":
        if "X-API-Key" in incoming_headers and incoming_headers["X-API-Key"]:
            return {"X-API-Key": incoming_headers["X-API-Key"]}
        if "Authorization" in incoming_headers and incoming_headers["Authorization"]:
            return {"Authorization": incoming_headers["Authorization"]}
        return {}
    raise ValueError(
        f"TINKER_GATEWAY_CONFIG_JSON: unsupported auth_mode={upstream.auth_mode!r} for {upstream.alias!r}"
    )


async def forward_json(
    *,
    upstream: Upstream,
    method: str,
    path: str,
    incoming_headers: dict[str, str],
    json_body: dict[str, Any] | None,
    timeout_s: float = 30.0,
) -> httpx.Response:
    url = f"{upstream.base_url}{path}"
    headers = _pick_auth_headers(incoming_headers=incoming_headers, upstream=upstream)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await client.request(method, url, headers=headers, json=json_body)


_remote_sampling_sessions: dict[str, tuple[str, str]] = {}  # sampling_session_id -> (upstream_alias, base_model)
_remote_training_models: dict[str, tuple[str, str]] = {}  # model_id -> (upstream_alias, base_model)


def register_remote_sampling_session(*, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
    _remote_sampling_sessions[sampling_session_id] = (upstream_alias, base_model)


def remote_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    return _remote_sampling_sessions.get(sampling_session_id)


def register_remote_training_model(*, model_id: str, upstream_alias: str, base_model: str) -> None:
    _remote_training_models[model_id] = (upstream_alias, base_model)


def remote_training_model(model_id: str) -> tuple[str, str] | None:
    return _remote_training_models.get(model_id)


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
