from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from tinker_server import app as app_module
from tinker_server.gateway_auth import GatewayAuthContext
from tinker_server.gateway_auth import (
    build_billing_auth_context,
    extract_gateway_auth_context_from_headers,
    has_gateway_auth_headers,
)
from tinker_server.model_access_control import can_access_model


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_extract_gateway_auth_context_accepts_mint_headers_and_derived_account_id():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-User-Role": "admin",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.user_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert ctx.user_role == "admin"
    assert ctx.account_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert ctx.apikey_id == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert ctx.request_id == "req-123"


def test_extract_gateway_auth_context_requires_internal_token_configuration():
    with pytest.raises(HTTPException) as exc:
        extract_gateway_auth_context_from_headers(
            {
                "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "X-MinT-User-Role": "user",
                "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
                "X-MinT-Request-Id": "req-456",
                "X-Internal-Token": "secret",
            }
        )

    assert exc.value.status_code == 503


def test_extract_gateway_auth_context_requires_internal_token_when_configured():
    with pytest.raises(HTTPException) as exc:
        extract_gateway_auth_context_from_headers(
            {
                "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "X-MinT-User-Role": "user",
                "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
                "X-MinT-Request-Id": "req-123",
            },
            internal_api_token="secret",
        )

    assert exc.value.status_code == 403


def test_has_gateway_auth_headers_detects_forwarded_auth():
    assert has_gateway_auth_headers({"X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb"}) is True
    assert has_gateway_auth_headers({"X-Request-Id": "req-1"}) is False
    assert has_gateway_auth_headers({"Authorization": "Bearer sk-abc"}) is False


def test_can_access_model_treats_gateway_admin_as_privileged():
    assert can_access_model(
        "moonshotai/Kimi-K2-Instruct",
        {"user_id": "aaaaaaaaaaaaaaaaaaaaaaaa", "user_role": "admin", "is_admin": True},
    )


def test_build_billing_auth_context_from_legacy_user_data():
    request = SimpleNamespace(
        state=SimpleNamespace(
            user_data={
                "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "key_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            }
        ),
        headers={},
    )

    ctx = build_billing_auth_context(request, fallback_request_id="req-legacy")

    assert ctx is not None
    assert ctx.user_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert ctx.user_role == "user"
    assert ctx.account_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert ctx.apikey_id == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert ctx.request_id == "req-legacy"


def test_build_billing_auth_context_from_gateway_state_preserves_role():
    request = SimpleNamespace(
        state=SimpleNamespace(
            gateway_auth=GatewayAuthContext(
                user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                user_role="admin",
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                request_id="req-1",
            )
        ),
        headers={},
    )

    ctx = build_billing_auth_context(request, fallback_request_id="req-2")

    assert ctx is not None
    assert ctx.user_role == "admin"
    assert ctx.request_id == "req-1"


@pytest.mark.anyio
async def test_internal_route_requires_auth(monkeypatch, anyio_backend):
    monkeypatch.setattr(app_module.config, "api_key", "admin-key")
    monkeypatch.setattr(app_module.config, "token_secret_key", "")
    monkeypatch.setattr(app_module.config, "internal_api_token", "internal-secret")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/internal/usage_logs",
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)

    async def call_next(_request):
        return PlainTextResponse("ok")

    response = await app_module.api_key_auth_middleware(request, call_next)

    assert response.status_code == 401
