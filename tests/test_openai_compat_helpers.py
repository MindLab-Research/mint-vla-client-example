# Run:
#   cd /vePFS-Mindverse/share/code/leixiang/tinker-server
#   /root/tinker_project/tinker-server/.venv31213/bin/pytest tests/test_openai_compat_helpers.py -v

import uuid
from types import SimpleNamespace

import anyio
from fastapi import HTTPException

from tinker_server.models.types import CreateSamplingSessionResponse
from tinker_server.routes import service as service_route


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


class _StubSessionManager:
    def __init__(self, base_model: str | None):
        self.base_model = base_model

    def get_session_base_model(self, _session_id: str):
        return self.base_model


def test_ensure_sampling_session_generates_parent_session_id(monkeypatch):
    seen: dict[str, str] = {}

    async def _fake_create_sampling_session(request, _http_request):
        seen["parent_session_id"] = request.session_id
        seen["model_path"] = request.model_path
        return CreateSamplingSessionResponse(sampling_session_id="sample-1")

    monkeypatch.setattr(service_route, "create_sampling_session", _fake_create_sampling_session)
    monkeypatch.setattr(service_route, "session_manager", _StubSessionManager("Qwen/Qwen3-4B-Instruct-2507"))

    import tinker_server.gateway as gw

    monkeypatch.setattr(gw, "remote_sampling_session", lambda _sid: None)

    result = anyio.run(
        lambda: service_route.ensure_sampling_session(
            model_path="tinker://exp/sampler_weights/000001",
            http_request=_dummy_request("u1"),
        )
    )

    assert result == ("sample-1", "Qwen/Qwen3-4B-Instruct-2507")
    assert seen["model_path"] == "tinker://exp/sampler_weights/000001"
    assert str(uuid.UUID(seen["parent_session_id"])) == seen["parent_session_id"]


def test_ensure_sampling_session_uses_base_model_for_plain_model_names(monkeypatch):
    seen: dict[str, str] = {}

    async def _fake_create_sampling_session(request, _http_request):
        seen["parent_session_id"] = request.session_id
        seen["base_model"] = request.base_model
        seen["model_path"] = request.model_path
        return CreateSamplingSessionResponse(sampling_session_id="sample-base-1")

    monkeypatch.setattr(service_route, "create_sampling_session", _fake_create_sampling_session)
    monkeypatch.setattr(service_route, "session_manager", _StubSessionManager("Qwen/Qwen3-30B-A3B-Instruct-2507"))

    import tinker_server.gateway as gw

    monkeypatch.setattr(gw, "remote_sampling_session", lambda _sid: None)

    result = anyio.run(
        lambda: service_route.ensure_sampling_session(
            model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
            http_request=_dummy_request("admin"),
        )
    )

    assert result == ("sample-base-1", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert seen["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert seen["model_path"] is None
    assert str(uuid.UUID(seen["parent_session_id"])) == seen["parent_session_id"]


def test_ensure_sampling_session_rejects_gateway_session(monkeypatch):
    async def _fake_create_sampling_session(_request, _http_request):
        return CreateSamplingSessionResponse(sampling_session_id="remote-sample")

    monkeypatch.setattr(service_route, "create_sampling_session", _fake_create_sampling_session)
    monkeypatch.setattr(service_route, "session_manager", _StubSessionManager(None))

    import tinker_server.gateway as gw

    monkeypatch.setattr(
        gw,
        "remote_sampling_session",
        lambda _sid: ("aliyun", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    )

    try:
        anyio.run(
            lambda: service_route.ensure_sampling_session(
                model_path="tinker://exp/sampler_weights/000002",
                http_request=_dummy_request("u1"),
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 501
        assert "gateway" in str(exc.detail).lower()
    else:
        raise AssertionError("expected HTTPException")
