from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_439_internal_noop_dispatch_uses_async_task_futures(monkeypatch) -> None:
    from mint_server.backend.scheduling import model_work_dispatch as dispatch
    import ray

    task_state_store_module = importlib.import_module("mint_server.backend.stores.task_state_store")
    calls: list[tuple[str, str]] = []

    class _AsyncOnlyTaskFutureService:
        async def async_resolve(self, request_id: str, result) -> None:
            calls.append((str(request_id), str(result.get("op"))))

        def resolve(self, request_id: str, result) -> None:
            raise AssertionError("sync resolve should not be used")

    async def _passthrough(_name, fn, **_kwargs):
        return await fn()

    monkeypatch.setattr(dispatch, "run_async_with_otel_span", _passthrough)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(task_state_store_module, "task_futures", _AsyncOnlyTaskFutureService())

    item = SimpleNamespace(
        op="internal.noop",
        request_id="rid-noop",
        request_json="{}",
        user_id=None,
        webhook_url=None,
        extra=None,
    )

    await dispatch.execute_model_work_item(item)

    assert calls == [("rid-noop", "internal.noop")]


@pytest.mark.anyio
async def test_model_work_dispatch_does_not_lazy_init_ray(monkeypatch) -> None:
    from mint_server.backend.scheduling import model_work_dispatch as dispatch
    from mint_server import ray_utils
    import ray

    init_ray_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *args, **kwargs: init_ray_calls.append((args, kwargs)))

    item = SimpleNamespace(
        op="internal.noop",
        request_id="rid-noop",
        request_json="{}",
        user_id=None,
        webhook_url=None,
        extra=None,
    )

    with pytest.raises(RuntimeError, match="Ray is not initialized"):
        await dispatch.execute_model_work_item(item)

    assert init_ray_calls == []


@pytest.mark.anyio
async def test_model_work_dispatch_classifies_dead_actor_as_executor_outcome(monkeypatch) -> None:
    from mint_server.backend.scheduling import model_work_dispatch as dispatch
    import ray

    class ActorDiedError(RuntimeError):
        pass

    async def _raise_dead_actor(_name, _fn, **_kwargs):
        raise ActorDiedError("backend actor died")

    monkeypatch.setattr(dispatch, "run_async_with_otel_span", _raise_dead_actor)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    item = SimpleNamespace(
        op="internal.noop",
        request_id="rid-noop",
        request_json="{}",
        user_id=None,
        webhook_url=None,
        extra=None,
    )

    outcome = await dispatch.execute_model_work_item(item)

    assert outcome.kind == "fatal_backend_death"
    assert outcome.error == "backend actor died"
