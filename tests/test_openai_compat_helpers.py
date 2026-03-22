# Run:
#   cd /vePFS-Mindverse/share/code/leixiang/tinker-server
#   /root/tinker_project/tinker-server/.venv31213/bin/pytest tests/test_openai_compat_helpers.py -v

import asyncio
import uuid
import sys
import threading
import types
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import anyio

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


def test_preload_supported_tokenizers_populates_cache_and_reports_failures(monkeypatch):
    openai_compat._tokenizer_cache.clear()
    monkeypatch.setattr(openai_compat, "_list_supported_models", lambda: ["m-good", "m-bad"])

    calls: list[str] = []

    def _fake_load(base_model: str):
        calls.append(base_model)
        if base_model == "m-bad":
            raise RuntimeError("missing tokenizer")
        return f"tokenizer:{base_model}"

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _fake_load)

    failures = openai_compat.preload_supported_tokenizers()

    assert failures == {"m-bad": "RuntimeError: missing tokenizer"}
    assert calls == ["m-good", "m-bad"]
    assert openai_compat._tokenizer_cache["m-good"] == "tokenizer:m-good"


def test_tokenizer_max_workers_env(monkeypatch):
    monkeypatch.setenv("TINKER_OAI_TOKENIZER_MAX_WORKERS", "12")
    monkeypatch.delenv("MINT_OAI_TOKENIZER_MAX_WORKERS", raising=False)

    assert openai_compat._tokenizer_max_workers() == 12


def test_tokenizer_max_workers_env_alias_and_invalid_fallback(monkeypatch):
    monkeypatch.delenv("TINKER_OAI_TOKENIZER_MAX_WORKERS", raising=False)
    monkeypatch.setenv("MINT_OAI_TOKENIZER_MAX_WORKERS", "7")
    assert openai_compat._tokenizer_max_workers() == 7

    monkeypatch.setenv("MINT_OAI_TOKENIZER_MAX_WORKERS", "bad")
    assert openai_compat._tokenizer_max_workers() == 8


def test_shutdown_tokenizer_executor_clears_cached_executor(monkeypatch):
    created: list[object] = []

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            created.append(self)
            self.shutdown_calls: list[tuple[bool, bool]] = []

        def shutdown(self, wait: bool = True, cancel_futures: bool = False):
            self.shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(openai_compat, "ThreadPoolExecutor", _FakeExecutor)
    openai_compat._get_tokenizer_executor.cache_clear()

    executor = openai_compat._get_tokenizer_executor()
    openai_compat.shutdown_tokenizer_executor()

    assert created == [executor]
    assert executor.shutdown_calls == [(False, True)]
    assert openai_compat._get_tokenizer_executor.cache_info().currsize == 0


def test_get_tokenizer_lazy_loads_via_dedicated_executor_and_caches(monkeypatch):
    openai_compat._tokenizer_cache.clear()
    seen: dict[str, object] = {}
    sentinel_executor = object()

    def _fake_load(base_model: str):
        seen["base_model"] = base_model
        return f"tokenizer:{base_model}"

    class _FakeLoop:
        def run_in_executor(self, executor, fn, *args):
            seen["executor"] = executor

            async def _done():
                return fn(*args)

            return _done()

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _fake_load)
    monkeypatch.setattr(openai_compat, "_get_tokenizer_executor", lambda: sentinel_executor)
    monkeypatch.setattr(openai_compat.asyncio, "get_running_loop", lambda: _FakeLoop())

    first = anyio.run(openai_compat._get_tokenizer, "Qwen/Qwen3-0.6B")
    second = anyio.run(openai_compat._get_tokenizer, "Qwen/Qwen3-0.6B")

    assert first == "tokenizer:Qwen/Qwen3-0.6B"
    assert second == "tokenizer:Qwen/Qwen3-0.6B"
    assert seen == {
        "executor": sentinel_executor,
        "base_model": "Qwen/Qwen3-0.6B",
    }


def test_get_tokenizer_single_flight_on_concurrent_cache_miss(monkeypatch):
    openai_compat._tokenizer_cache.clear()
    openai_compat._tokenizer_locks.clear()
    calls: list[str] = []
    executor = ThreadPoolExecutor(max_workers=2)

    def _fake_load(base_model: str):
        calls.append(base_model)
        time.sleep(0.05)
        return f"tokenizer:{base_model}"

    async def _run():
        first, second = await asyncio.gather(
            openai_compat._get_tokenizer("Qwen/Qwen3-0.6B"),
            openai_compat._get_tokenizer("Qwen/Qwen3-0.6B"),
        )
        return first, second

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _fake_load)
    monkeypatch.setattr(openai_compat, "_get_tokenizer_executor", lambda: executor)

    try:
        first, second = anyio.run(_run)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert first == "tokenizer:Qwen/Qwen3-0.6B"
    assert second == "tokenizer:Qwen/Qwen3-0.6B"
    assert calls == ["Qwen/Qwen3-0.6B"]


def test_get_tokenizer_allows_concurrent_misses_for_different_models(monkeypatch):
    openai_compat._tokenizer_cache.clear()
    openai_compat._tokenizer_locks.clear()
    executor = ThreadPoolExecutor(max_workers=2)
    calls: list[str] = []
    barrier = threading.Barrier(2, timeout=5.0)

    def _fake_load(base_model: str):
        calls.append(base_model)
        barrier.wait()
        return f"tokenizer:{base_model}"

    async def _run():
        first, second = await asyncio.gather(
            openai_compat._get_tokenizer("model-a"),
            openai_compat._get_tokenizer("model-b"),
        )
        return first, second

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _fake_load)
    monkeypatch.setattr(openai_compat, "_get_tokenizer_executor", lambda: executor)

    try:
        first, second = anyio.run(_run)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert first == "tokenizer:model-a"
    assert second == "tokenizer:model-b"
    assert sorted(calls) == ["model-a", "model-b"]


def test_get_tokenizer_lazy_load_still_works_after_preload_failure(monkeypatch):
    openai_compat._tokenizer_cache.clear()
    attempts: list[tuple[str, str]] = []

    def _flaky_load(base_model: str):
        attempt = len(attempts)
        attempts.append((base_model, f"attempt-{attempt}"))
        if attempt == 0:
            raise RuntimeError("preload missing")
        return f"tokenizer:{base_model}"

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _flaky_load)
    monkeypatch.setattr(openai_compat, "_list_supported_models", lambda: ["Qwen/Qwen3-0.6B"])

    failures = openai_compat.preload_supported_tokenizers()
    tokenizer = anyio.run(openai_compat._get_tokenizer, "Qwen/Qwen3-0.6B")

    assert failures == {"Qwen/Qwen3-0.6B": "RuntimeError: preload missing"}
    assert tokenizer == "tokenizer:Qwen/Qwen3-0.6B"
    assert attempts == [
        ("Qwen/Qwen3-0.6B", "attempt-0"),
        ("Qwen/Qwen3-0.6B", "attempt-1"),
    ]


def test_get_tokenizer_surfaces_loader_exception(monkeypatch):
    openai_compat._tokenizer_cache.clear()

    def _boom(_base_model: str):
        raise RuntimeError("loader boom")

    monkeypatch.setattr(openai_compat, "_load_tokenizer_cpu", _boom)

    try:
        anyio.run(openai_compat._get_tokenizer, "Qwen/Qwen3-0.6B")
    except RuntimeError as exc:
        assert str(exc) == "loader boom"
    else:
        raise AssertionError("expected loader failure to propagate")


def test_ensure_sampling_session_generates_parent_session_id(monkeypatch):
    seen: dict[str, str] = {}

    async def _fake_create_sampling_session(request, _http_request):
        seen["parent_session_id"] = request.session_id
        seen["model_path"] = request.model_path
        return CreateSamplingSessionResponse(sampling_session_id="sample-1")

    monkeypatch.setattr(service_route, "create_sampling_session", _fake_create_sampling_session)
    monkeypatch.setattr(service_route, "session_manager", _StubSessionManager("Qwen/Qwen3-4B-Instruct-2507"))

    import tinker_server.gateway as gw

    async def _remote_sampling_session(_sid):
        return None

    monkeypatch.setattr(gw, "async_remote_sampling_session", _remote_sampling_session)

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

    async def _remote_sampling_session(_sid):
        return None

    monkeypatch.setattr(gw, "async_remote_sampling_session", _remote_sampling_session)

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

    async def _remote_sampling_session(_sid):
        return ("aliyun", "Qwen/Qwen3-235B-A22B-Instruct-2507")

    monkeypatch.setattr(gw, "async_remote_sampling_session", _remote_sampling_session)

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

    async def _remote_sampling_session(_sid):
        return ("mint-prod-aliyun", "Qwen/Qwen3-235B-A22B-Instruct-2507")

    monkeypatch.setattr(gw, "async_remote_sampling_session", _remote_sampling_session)

    result = anyio.run(
        lambda: service_route.ensure_sampling_session(
            model_path="Qwen/Qwen3-235B-A22B-Instruct-2507",
            http_request=_dummy_request("u1"),
        )
    )

    assert result == ("remote-sample", "Qwen/Qwen3-235B-A22B-Instruct-2507")
    assert seen["create_called"] is True
    assert seen["base_model"] == "Qwen/Qwen3-235B-A22B-Instruct-2507"
