from __future__ import annotations

import asyncio
import concurrent.futures

import pytest


class _FutureRayRef:
    def __init__(self, fut: concurrent.futures.Future):
        self._future = fut

    def future(self):
        return self._future


def test_async_get_ray_ref_timeout_does_not_cancel_ray_future() -> None:
    import ray

    from mint_server.backend.async_ray_control import async_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()

    async def _run() -> None:
        with pytest.raises(ray.exceptions.GetTimeoutError):
            await async_get_ray_ref(_FutureRayRef(fut), timeout_s=0.01)
        assert not fut.cancelled()
        assert not fut.done()
        fut.set_result("late-ok")
        await asyncio.sleep(0)

    asyncio.run(_run())


def test_async_get_ray_ref_cancellation_silences_late_exception(monkeypatch) -> None:
    from mint_server.backend import async_ray_control

    discarded: list[str] = []

    def _record_late_result(fut: asyncio.Future) -> None:
        try:
            fut.result()
        except RuntimeError as exc:
            discarded.append(str(exc))
        except BaseException as exc:
            discarded.append(type(exc).__name__)

    class _AsyncFutureRayRef:
        def __init__(self, fut: asyncio.Future):
            self._future = fut

        def future(self):
            return self._future

    monkeypatch.setattr(async_ray_control, "_discard_late_result", _record_late_result)

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(async_ray_control.async_get_ray_ref(_AsyncFutureRayRef(fut), timeout_s=60.0))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not fut.cancelled()

        fut.set_exception(RuntimeError("late boom"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert discarded == ["late boom"]

    asyncio.run(_run())


def test_async_get_ray_ref_prefers_future_bridge_over_direct_await() -> None:
    from mint_server.backend.async_ray_control import async_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result("ok")

    class _DualRayRef(_FutureRayRef):
        def __await__(self):
            raise AssertionError("direct __await__ path should not be used")

    assert asyncio.run(async_get_ray_ref(_DualRayRef(fut), timeout_s=1.0)) == "ok"


def test_sync_get_ray_ref_timeout_does_not_cancel_ray_future() -> None:
    import ray

    from mint_server.backend.async_ray_control import sync_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()

    with pytest.raises(ray.exceptions.GetTimeoutError):
        sync_get_ray_ref(_FutureRayRef(fut), timeout_s=0.01)
    assert not fut.cancelled()
    assert not fut.done()
    fut.set_result("late-ok")


def test_sync_get_ray_ref_prefers_future_bridge_over_direct_await() -> None:
    from mint_server.backend.async_ray_control import sync_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result("ok")

    class _DualRayRef(_FutureRayRef):
        def __await__(self):
            raise AssertionError("direct __await__ path should not be used")

    assert sync_get_ray_ref(_DualRayRef(fut), timeout_s=1.0) == "ok"


def test_sync_get_ray_ref_awaitable_inside_running_loop() -> None:
    from mint_server.backend.async_ray_control import sync_get_ray_ref

    async def _value() -> str:
        await asyncio.sleep(0)
        return "ok"

    async def _run() -> str:
        return sync_get_ray_ref(_value(), timeout_s=1.0)

    assert asyncio.run(_run()) == "ok"


def test_sync_get_ray_ref_done_asyncio_future_inside_running_loop() -> None:
    from mint_server.backend.async_ray_control import sync_get_ray_ref

    async def _run() -> str:
        fut = asyncio.get_running_loop().create_future()
        fut.set_result("ok")
        return sync_get_ray_ref(fut, timeout_s=1.0)

    assert asyncio.run(_run()) == "ok"


def test_sync_get_ray_ref_pending_asyncio_future_inside_running_loop_fails_fast() -> None:
    from mint_server.backend.async_ray_control import sync_get_ray_ref

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        with pytest.raises(RuntimeError, match="use async_get_ray_ref"):
            sync_get_ray_ref(fut, timeout_s=1.0)
        assert not fut.done()
        fut.cancel()

    asyncio.run(_run())
