# Run:
#   cd /vePFS-Mindverse/share/code/leixiang/tinker-server
#   /root/tinker_project/tinker-server/.venv31213/bin/pytest tests/test_openai_compat_helpers.py -v

import uuid
import sys
import types
from types import SimpleNamespace

import anyio
from fastapi import HTTPException

from tinker_server.models.types import CreateSamplingSessionResponse
from tinker_server.routes import openai_compat
from tinker_server.routes import service as service_route


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


class _StubSessionManager:
    def __init__(self, base_model: str | None):
        self.base_model = base_model

    def get_session_base_model(self, _session_id: str):
        return self.base_model


def test_load_tokenizer_cpu_uses_transformers_directly(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(base_model: str, local_files_only: bool = True):
            seen["base_model"] = base_model
            seen["local_files_only"] = local_files_only
            return "tokenizer"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer = openai_compat._load_tokenizer_cpu("Qwen/Qwen3-0.6B")

    assert tokenizer == "tokenizer"
    assert seen == {
        "base_model": "Qwen/Qwen3-0.6B",
        "local_files_only": True,
    }


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


def test_ensure_sampling_session_returns_gateway_session_base_model(monkeypatch):
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

    result = anyio.run(
        lambda: service_route.ensure_sampling_session(
            model_path="tinker://exp/sampler_weights/000002",
            http_request=_dummy_request("u1"),
        )
    )

    assert result == ("remote-sample", "Qwen/Qwen3-235B-A22B-Instruct-2507")


def test_ensure_sampling_session_allows_routed_base_model_creation(monkeypatch):
    seen: dict[str, str | bool | None] = {"create_called": False, "base_model": None}

    async def _fake_create_sampling_session(request, _http_request):
        seen["create_called"] = True
        seen["base_model"] = request.base_model
        return CreateSamplingSessionResponse(sampling_session_id="remote-sample")

    monkeypatch.setattr(service_route, "create_sampling_session", _fake_create_sampling_session)
    monkeypatch.setattr(service_route, "session_manager", _StubSessionManager(None))

    import tinker_server.gateway as gw

    monkeypatch.setattr(
        gw,
        "remote_sampling_session",
        lambda _sid: ("mint-prod-aliyun", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    )

    result = anyio.run(
        lambda: service_route.ensure_sampling_session(
            model_path="Qwen/Qwen3-235B-A22B-Instruct-2507",
            http_request=_dummy_request("u1"),
        )
    )

    assert result == ("remote-sample", "Qwen/Qwen3-235B-A22B-Instruct-2507")
    assert seen["create_called"] is True
    assert seen["base_model"] == "Qwen/Qwen3-235B-A22B-Instruct-2507"
