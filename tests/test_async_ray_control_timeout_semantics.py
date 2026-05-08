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

    from tinker_server.backend.async_ray_control import async_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()

    async def _run() -> None:
        with pytest.raises(ray.exceptions.GetTimeoutError):
            await async_get_ray_ref(_FutureRayRef(fut), timeout_s=0.01)
        assert not fut.cancelled()
        assert not fut.done()
        fut.set_result("late-ok")
        await asyncio.sleep(0)

    asyncio.run(_run())


def test_async_get_ray_ref_prefers_future_bridge_over_direct_await() -> None:
    from tinker_server.backend.async_ray_control import async_get_ray_ref

    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result("ok")

    class _DualRayRef(_FutureRayRef):
        def __await__(self):
            raise AssertionError("direct __await__ path should not be used")

    assert asyncio.run(async_get_ray_ref(_DualRayRef(fut), timeout_s=1.0)) == "ok"

