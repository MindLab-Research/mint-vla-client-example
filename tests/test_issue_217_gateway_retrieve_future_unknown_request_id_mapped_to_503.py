import anyio
import httpx
import pytest
from types import SimpleNamespace

from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


def _request_non_privileged():
    return SimpleNamespace(state=SimpleNamespace(user_data=None), headers={})


def _request_privileged():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_issue_217_gateway_unknown_request_id_maps_404_to_503_non_privileged(monkeypatch):
    import tinker_server.gateway as gw
    monkeypatch.setattr(futures_route, "_is_privileged", lambda _req: False)

    async def _forward_json(*, upstream, method, path, incoming_headers, json_body, timeout_s=30.0):
        assert path == "/api/v1/retrieve_future"
        return httpx.Response(404, json={"detail": "Unknown request_id: upstream-uuid"})

    monkeypatch.setattr(
        gw,
        "upstream_for_alias",
        lambda alias: gw.Upstream(alias=alias, base_url="http://example.com", auth_mode="none", api_key=None),
    )
    monkeypatch.setattr(gw, "forward_json", _forward_json)

    req_id = gw.encode_request_id(upstream_alias="u1", upstream_request_id="upstream-uuid")
    body = FutureRetrieveRequest(request_id=req_id)

    with pytest.raises(futures_route.HTTPException) as e:
        anyio.run(futures_route.retrieve_future, body, _request_non_privileged(), _response_stub())
    assert e.value.status_code == 503
    assert e.value.detail == futures_route.GENERIC_ERROR_MESSAGE


def test_issue_217_gateway_unknown_request_id_maps_404_to_503_privileged(monkeypatch):
    import tinker_server.gateway as gw
    monkeypatch.setattr(futures_route, "_is_privileged", lambda _req: True)

    async def _forward_json(*, upstream, method, path, incoming_headers, json_body, timeout_s=30.0):
        assert path == "/api/v1/retrieve_future"
        return httpx.Response(404, json={"detail": "Unknown request_id: upstream-uuid"})

    monkeypatch.setattr(
        gw,
        "upstream_for_alias",
        lambda alias: gw.Upstream(alias=alias, base_url="http://example.com", auth_mode="none", api_key=None),
    )
    monkeypatch.setattr(gw, "forward_json", _forward_json)

    req_id = gw.encode_request_id(upstream_alias="u1", upstream_request_id="upstream-uuid")
    body = FutureRetrieveRequest(request_id=req_id)

    with pytest.raises(futures_route.HTTPException) as e:
        anyio.run(futures_route.retrieve_future, body, _request_privileged(), _response_stub())
    assert e.value.status_code == 503
    assert isinstance(e.value.detail, dict)
    assert e.value.detail.get("upstream_alias") == "u1"
    assert e.value.detail.get("upstream_request_id") == "upstream-uuid"
