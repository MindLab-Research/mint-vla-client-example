"""Test gateway retrieve_future caching to prevent 'Future already retrieved' errors.

Issue #277: Gateway-routed requests should cache terminal responses to handle client retries.
"""
import json
from types import SimpleNamespace

import pytest

from mint_server.models.types import FutureRetrieveRequest
from mint_server.routes import futures as futures_route


def _reset_gateway_and_cache():
    """Reset gateway config and _recent cache."""
    import mint_server.gateway as gw
    gw._gateway_config = None
    gw._remote_sampling_sessions.clear()
    gw._remote_training_models.clear()

    # Clear _recent cache
    futures_route._RECENT.clear()
    futures_route._PENDING_HINTS.clear()


def _request_stub(*, admin: bool = True):
    user_data = {"user_id": "admin"} if admin else None
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def _mock_upstream_response(status_code: int, payload: dict):
    """Create a mock upstream response."""
    resp = SimpleNamespace()
    resp.status_code = status_code
    resp.headers = {}
    resp.json = lambda: payload
    return resp


@pytest.mark.anyio
async def test_gateway_retrieve_future_caches_terminal_response(monkeypatch):
    """First retrieve forwards to upstream and caches result, second retrieve returns from cache."""
    _reset_gateway_and_cache()

    # Setup gateway config
    cfg = {
        "model_to_upstream": {"test-model": "test-upstream"},
        "upstreams": {
            "test-upstream": {"base_url": "http://test:8000", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))

    # Mock forward_json to simulate upstream response
    forward_call_count = 0
    async def mock_forward_json(*args, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        return _mock_upstream_response(200, {"result": "success", "data": "test_data"})

    import mint_server.gateway as gw
    monkeypatch.setattr(gw, "forward_json", mock_forward_json)

    # First retrieve_future call
    body = FutureRetrieveRequest(request_id="gw:test-upstream:abc123")
    response = _response_stub()
    payload1 = await futures_route.retrieve_future(body, _request_stub(), response)

    assert payload1["result"] == "success"
    assert payload1["data"] == "test_data"
    assert forward_call_count == 1

    # Second retrieve_future call (should hit cache)
    response2 = _response_stub()
    payload2 = await futures_route.retrieve_future(body, _request_stub(), response2)

    assert payload2 == payload1
    assert forward_call_count == 1  # Should NOT forward again


@pytest.mark.anyio
async def test_gateway_retrieve_future_caches_error_response(monkeypatch):
    """Terminal error responses should also be cached."""
    _reset_gateway_and_cache()

    cfg = {
        "model_to_upstream": {"test-model": "test-upstream"},
        "upstreams": {
            "test-upstream": {"base_url": "http://test:8000", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))

    forward_call_count = 0
    async def mock_forward_json(*args, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        return _mock_upstream_response(200, {"error": "Future already retrieved", "category": "system"})

    import mint_server.gateway as gw
    monkeypatch.setattr(gw, "forward_json", mock_forward_json)

    body = FutureRetrieveRequest(request_id="gw:test-upstream:xyz789")
    response = _response_stub()
    payload1 = await futures_route.retrieve_future(body, _request_stub(), response)

    assert "error" in payload1
    assert forward_call_count == 1

    # Retry should hit cache
    response2 = _response_stub()
    payload2 = await futures_route.retrieve_future(body, _request_stub(), response2)

    assert payload2 == payload1
    assert forward_call_count == 1


@pytest.mark.anyio
async def test_gateway_retrieve_future_does_not_cache_pending(monkeypatch):
    """Pending responses (408) should NOT be cached."""
    _reset_gateway_and_cache()

    cfg = {
        "model_to_upstream": {"test-model": "test-upstream"},
        "upstreams": {
            "test-upstream": {"base_url": "http://test:8000", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))

    # Disable pending throttle to test actual forwarding behavior
    monkeypatch.setattr(futures_route, "_pending_hint_maybe_throttle", lambda req_id: None)

    forward_call_count = 0
    async def mock_forward_json(*args, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        return _mock_upstream_response(408, {})

    import mint_server.gateway as gw
    monkeypatch.setattr(gw, "forward_json", mock_forward_json)

    body = FutureRetrieveRequest(request_id="gw:test-upstream:pending123")
    response = _response_stub()

    # First call
    await futures_route.retrieve_future(body, _request_stub(), response)
    assert forward_call_count == 1

    # Second call should forward again (pending not cached)
    response2 = _response_stub()
    await futures_route.retrieve_future(body, _request_stub(), response2)
    assert forward_call_count == 2


@pytest.mark.anyio
async def test_gateway_retrieve_future_cached_response_preserves_public_error_and_gateway_request_id(monkeypatch):
    """Cached retries should preserve post-processed error masking and request_id rewriting."""
    _reset_gateway_and_cache()

    cfg = {
        "model_to_upstream": {"test-model": "test-upstream"},
        "upstreams": {
            "test-upstream": {"base_url": "http://test:8000", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))
    monkeypatch.setattr(futures_route, "_is_privileged", lambda _req: False)

    forward_call_count = 0

    async def mock_forward_json(*args, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        return _mock_upstream_response(
            200,
            {"error": "sensitive internal detail", "category": "system", "request_id": "upstream-123"},
        )

    import mint_server.gateway as gw
    monkeypatch.setattr(gw, "forward_json", mock_forward_json)

    body = FutureRetrieveRequest(request_id="gw:test-upstream:upstream-123")

    response1 = _response_stub()
    payload1 = await futures_route.retrieve_future(body, _request_stub(admin=False), response1)
    response2 = _response_stub()
    payload2 = await futures_route.retrieve_future(body, _request_stub(admin=False), response2)

    expected = {
        "error": futures_route.GENERIC_ERROR_MESSAGE,
        "category": "system",
        "request_id": body.request_id,
    }
    assert payload1 == expected
    assert payload2 == expected
    assert response2.status_code == 200
    assert forward_call_count == 1


@pytest.mark.anyio
async def test_gateway_retrieve_future_cached_response_preserves_terminal_status_and_public_detail(monkeypatch):
    """Cached retries should preserve non-200 terminal status and masked detail."""
    _reset_gateway_and_cache()

    cfg = {
        "model_to_upstream": {"test-model": "test-upstream"},
        "upstreams": {
            "test-upstream": {"base_url": "http://test:8000", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))
    monkeypatch.setattr(futures_route, "_is_privileged", lambda _req: False)

    forward_call_count = 0

    async def mock_forward_json(*args, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        return _mock_upstream_response(503, {"detail": "internal upstream detail"})

    import mint_server.gateway as gw
    monkeypatch.setattr(gw, "forward_json", mock_forward_json)

    body = FutureRetrieveRequest(request_id="gw:test-upstream:upstream-503")

    response1 = _response_stub()
    payload1 = await futures_route.retrieve_future(body, _request_stub(admin=False), response1)
    response2 = _response_stub()
    payload2 = await futures_route.retrieve_future(body, _request_stub(admin=False), response2)

    expected = {"detail": futures_route.GENERIC_ERROR_MESSAGE}
    assert response1.status_code == 503
    assert payload1 == expected
    assert response2.status_code == 503
    assert payload2 == expected
    assert forward_call_count == 1
