from types import SimpleNamespace
import logging
import sys
import types

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
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
    assert ctx.session_id == ""


def test_extract_gateway_auth_context_accepts_optional_session_id():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-User-Role": "admin",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-MinT-Session-Id": "sess-123",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.session_id == "sess-123"


def test_extract_gateway_auth_context_defaults_write_true_when_header_omitted():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-MinT-Cap-View-Internal-Errors": "false",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.caps_from_headers is True
    assert ctx.cap_write is True


def test_extract_gateway_auth_context_ignores_write_false_header():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-MinT-Cap-Write": "false",
            "X-MinT-Cap-View-Internal-Errors": "false",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.caps_from_headers is True
    assert ctx.cap_write is True


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


def test_can_access_model_allows_privileged_restricted_model_access():
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


@pytest.mark.anyio
async def test_gateway_auth_sets_response_apikey_header(monkeypatch, anyio_backend):
    monkeypatch.setattr(app_module.config, "api_key", "")
    monkeypatch.setattr(app_module.config, "token_secret_key", "")
    monkeypatch.setattr(app_module.config, "internal_api_token", "internal-secret")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/models",
        "query_string": b"",
        "headers": [
            (b"x-mint-user-id", b"aaaaaaaaaaaaaaaaaaaaaaaa"),
            (b"x-mint-user-role", b"user"),
            (b"x-mint-account-id", b"cccccccccccccccccccccccc"),
            (b"x-mint-apikey-id", b"bbbbbbbbbbbbbbbbbbbbbbbb"),
            (b"x-mint-request-id", b"req-123"),
            (b"x-mint-session-id", b"sess-123"),
            (b"x-internal-token", b"internal-secret"),
        ],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)

    async def call_next(passed_request):
        assert passed_request.state.user_data["apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
        assert passed_request.state.user_data["session_id"] == "sess-123"
        return JSONResponse({"ok": True})

    response = await app_module.api_key_auth_middleware(request, call_next)

    assert response.status_code == 200
    assert response.headers["X-MinT-Apikey-Id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"


class _DummySpanContext:
    trace_id = int("1" * 32, 16)


class _DummySpan:
    def __init__(self):
        self.attributes: dict[str, object] = {}
        self.status = None
        self.recorded: list[Exception] = []
        self.name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_span_context(self):
        return _DummySpanContext()

    def set_attribute(self, key, value):
        self.attributes[str(key)] = value

    def update_name(self, name):
        self.name = str(name)

    def record_exception(self, error, attributes=None):
        self.recorded.append(error)

    def set_status(self, status):
        self.status = status


class _DummyTracer:
    def __init__(self, span):
        self.span = span

    def start_as_current_span(self, *_args, **_kwargs):
        return self.span


@pytest.mark.anyio
async def test_http_observability_includes_gateway_identity(monkeypatch, caplog, anyio_backend):
    monkeypatch.setattr(app_module.config, "api_key", "")
    monkeypatch.setattr(app_module.config, "token_secret_key", "")
    monkeypatch.setattr(app_module.config, "internal_api_token", "internal-secret")

    propagate_mod = types.ModuleType("opentelemetry.propagate")
    propagate_mod.extract = lambda headers: None
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.SpanKind = types.SimpleNamespace(SERVER="server")
    trace_mod.StatusCode = types.SimpleNamespace(ERROR="error")

    class _DummyStatus:
        def __init__(self, code, description=None):
            self.code = code
            self.description = description

    trace_mod.Status = _DummyStatus
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", propagate_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)

    span = _DummySpan()
    monkeypatch.setattr(app_module, "get_otel_tracer", lambda: _DummyTracer(span))

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/models",
        "query_string": b"",
        "headers": [
            (b"x-mint-user-id", b"aaaaaaaaaaaaaaaaaaaaaaaa"),
            (b"x-mint-user-role", b"user"),
            (b"x-mint-account-id", b"cccccccccccccccccccccccc"),
            (b"x-mint-apikey-id", b"bbbbbbbbbbbbbbbbbbbbbbbb"),
            (b"x-mint-request-id", b"req-123"),
            (b"x-mint-session-id", b"sess-123"),
            (b"x-internal-token", b"internal-secret"),
        ],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)

    async def endpoint(_request):
        logging.getLogger("tinker_server.app").info("inside-endpoint")
        return JSONResponse({"ok": True})

    with caplog.at_level(logging.INFO, logger="tinker_server.app"):
        response = await app_module.api_key_auth_middleware(
            request,
            lambda req: app_module.otel_trace_metrics_middleware(req, endpoint),
        )

    assert response.status_code == 200
    assert span.attributes["mint.user_id"] == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert span.attributes["mint.user_role"] == "user"
    assert span.attributes["mint.account_id"] == "cccccccccccccccccccccccc"
    assert span.attributes["mint.apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert span.attributes["mint.gateway_request_id"] == "req-123"
    assert span.attributes["mint.gateway_session_id"] == "sess-123"
    http_logs = [rec.getMessage() for rec in caplog.records if "[http.request]" in rec.getMessage()]
    assert any("apikey_id=bbbbbbbbbbbbbbbbbbbbbbbb" in msg for msg in http_logs)
    assert any("gateway_request_id=req-123" in msg for msg in http_logs)
    assert any("gateway_session_id=sess-123" in msg for msg in http_logs)


@pytest.mark.anyio
async def test_legacy_user_data_sets_response_apikey_header(monkeypatch, anyio_backend):
    monkeypatch.setattr(app_module.config, "api_key", "")
    monkeypatch.setattr(app_module.config, "token_secret_key", "")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/models",
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)
    request.state.user_data = {"user_id": "aaaaaaaaaaaaaaaaaaaaaaaa", "key_id": "bbbbbbbbbbbbbbbbbbbbbbbb"}

    async def call_next(passed_request):
        return JSONResponse({"ok": True})

    response = await app_module.api_key_auth_middleware(request, call_next)

    assert response.status_code == 200
    assert response.headers["X-MinT-Apikey-Id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
