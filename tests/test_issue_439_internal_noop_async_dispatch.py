from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_439_internal_noop_dispatch_uses_async_future_store(monkeypatch) -> None:
    from tinker_server.backend import api_work_queue_dispatch as dispatch
    import ray

    task_state_store_module = importlib.import_module("tinker_server.backend.task_state_store")
    calls: list[tuple[str, str]] = []

    class _AsyncOnlyFutureStore:
        async def async_resolve(self, request_id: str, result) -> None:
            calls.append((str(request_id), str(result.get("op"))))

        def resolve(self, request_id: str, result) -> None:
            raise AssertionError("sync resolve should not be used")

    async def _passthrough(_name, fn, **_kwargs):
        return await fn()

    monkeypatch.setattr(dispatch, "run_async_with_otel_span", _passthrough)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(task_state_store_module, "task_state_futures", _AsyncOnlyFutureStore())

    item = SimpleNamespace(
        op="internal.noop",
        request_id="rid-noop",
        request_json="{}",
        user_id=None,
        webhook_url=None,
        extra=None,
    )

    await dispatch.execute_work_item(item)

    assert calls == [("rid-noop", "internal.noop")]


@pytest.mark.anyio
async def test_api_work_queue_dispatch_does_not_lazy_init_ray(monkeypatch) -> None:
    from tinker_server.backend import api_work_queue_dispatch as dispatch
    from tinker_server import ray_utils
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
        await dispatch.execute_work_item(item)

    assert init_ray_calls == []
