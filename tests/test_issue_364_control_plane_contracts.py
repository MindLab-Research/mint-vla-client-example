from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _install_uninitialized_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    ray_module = types.ModuleType("ray")
    ray_module.is_initialized = lambda: False
    ray_module.get_actor = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("actor not found"))
    ray_module.get = lambda *args, **kwargs: None
    ray_module.init = lambda *args, **kwargs: None
    ray_module.shutdown = lambda: None
    ray_module.nodes = lambda: []
    ray_module.exceptions = SimpleNamespace(
        ActorDiedError=RuntimeError,
        RayActorError=RuntimeError,
        GetTimeoutError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.exceptions", ray_module.exceptions)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue364_request_path_helpers_fail_fast_without_init_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_uninitialized_ray(monkeypatch)
    init_ray_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    ray_utils = importlib.import_module("tinker_server.ray_utils")
    monkeypatch.setattr(
        ray_utils,
        "init_ray",
        lambda *args, **kwargs: init_ray_calls.append((args, kwargs)),
    )

    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    gateway_session_store_module = importlib.import_module("tinker_server.backend.gateway_session_store")

    with pytest.raises(capacity_manager_module.CapacityManagerUnavailableError, match="Ray not initialized"):
        capacity_manager_module.CapacityManager()._get_cached_ray_actor_for_async_request_path()
    with pytest.raises(future_store_module.FutureStoreUnavailableError, match="Ray not initialized"):
        future_store_module.FutureStore()._get_cached_ray_actor_for_async_request_path()
    with pytest.raises(api_work_queue_module.ApiWorkQueueUnavailableError, match="Ray not initialized"):
        await api_work_queue_module.ApiWorkQueueClient()._get_ray_actor_async(require_ready=False)
    with pytest.raises(RuntimeError, match="Ray not initialized"):
        gateway_session_store_module._ensure_ray_initialized()

    assert init_ray_calls == []


def test_issue364_session_index_write_fails_closed_without_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_uninitialized_ray(monkeypatch)
    session_index_store = importlib.import_module("tinker_server.backend.session_index_store")

    with pytest.raises(RuntimeError, match="Session index store write failed: upsert_session: Ray not initialized"):
        session_index_store.upsert_session_index({"session_id": "sess-364"})

    with pytest.raises(RuntimeError, match="Session index store write failed: add_training_run: Ray not initialized"):
        session_index_store.add_training_run_to_session("sess-364", "run-364")


def test_issue364_training_and_sampling_store_writes_fail_closed_without_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_uninitialized_ray(monkeypatch)
    training_store = importlib.import_module("tinker_server.backend.training_session_store")
    sampling_store = importlib.import_module("tinker_server.backend.sampling_session_store")

    with pytest.raises(RuntimeError, match="Training session store write failed: upsert: Ray not initialized"):
        training_store.upsert_training_session({"model_id": "run-364"})

    with pytest.raises(RuntimeError, match="Sampling session store write failed: upsert: Ray not initialized"):
        sampling_store.upsert_sampling_session({"session_id": "sample-364"})


@pytest.mark.anyio
async def test_issue364_create_session_returns_503_when_index_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_route = importlib.import_module("tinker_server.routes.service")
    models = importlib.import_module("tinker_server.models.types")
    session_index_store = importlib.import_module("tinker_server.backend.session_index_store")

    monkeypatch.setattr(service_route, "_get_user_id", lambda _request: "user-364")
    monkeypatch.setattr(
        session_index_store,
        "upsert_session_index",
        lambda _info: (_ for _ in ()).throw(RuntimeError("Ray not initialized")),
    )

    with pytest.raises(HTTPException, match="Session index store unavailable") as exc:
        await service_route.create_session(
            models.CreateSessionRequest(),
            SimpleNamespace(),
        )

    assert exc.value.status_code == 503
