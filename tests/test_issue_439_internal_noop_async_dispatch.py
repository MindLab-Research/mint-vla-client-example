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

    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    calls: list[tuple[str, str]] = []

    class _AsyncOnlyFutureStore:
        async def async_resolve(self, request_id: str, result) -> None:
            calls.append((str(request_id), str(result.get("op"))))

        def resolve(self, request_id: str, result) -> None:
            raise AssertionError("sync resolve should not be used")

    async def _passthrough(_name, fn, **_kwargs):
        return await fn()

    monkeypatch.setattr(dispatch, "run_async_with_otel_span", _passthrough)
    monkeypatch.setattr(future_store_module, "future_store", _AsyncOnlyFutureStore())

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
