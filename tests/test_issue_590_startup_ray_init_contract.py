from __future__ import annotations

import asyncio

import pytest


def test_lifespan_fails_before_yield_when_ray_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from tinker_server import app as app_module

    def _fail_init_ray(*_args, **_kwargs) -> None:
        raise RuntimeError("ray startup unavailable")

    monkeypatch.setattr(app_module, "init_ray", _fail_init_ray)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="ray startup unavailable"):
            async with app_module.lifespan(app_module.app):
                raise AssertionError("lifespan should not yield when Ray startup fails")

    asyncio.run(_run())
