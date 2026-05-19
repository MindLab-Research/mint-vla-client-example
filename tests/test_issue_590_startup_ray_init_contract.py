from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest


def test_lifespan_fails_before_yield_when_ray_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import app as app_module

    def _fail_init_ray(*_args, **_kwargs) -> None:
        raise RuntimeError("ray startup unavailable")

    monkeypatch.setattr(app_module, "init_ray", _fail_init_ray)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="ray startup unavailable"):
            async with app_module.lifespan(app_module.app):
                raise AssertionError("lifespan should not yield when Ray startup fails")

    asyncio.run(_run())


def test_lifespan_checks_control_plane_without_creating_actors(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import app as app_module

    calls: list[str] = []

    class _UsageStore:
        async def health_check(self) -> bool:
            calls.append("usage.health_check")
            return True

    class _ActionSessionRouter:
        pass

    class _MaintenanceCron:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls.append("cron.ping")
            return {"ok": True}

        async def async_health_snapshot(
            self,
            *,
            timeout_s: float = 10.0,
            create_if_missing: bool = True,
        ) -> dict:
            calls.append(f"cron.snapshot.create={create_if_missing}")
            assert create_if_missing is False
            return {
                "actor_name": "mint_maintenance_cron",
                "namespace": "mint",
                "epoch_id": "epoch",
                "code_identity": app_module._git_sha(),
                "loops": {},
            }

        async def async_ensure_started(self, *_args, **_kwargs) -> dict:
            raise AssertionError("API lifespan must not create MaintenanceCronActor")

    class _Scheduler:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls.append("scheduler.ping")
            return {"ok": True}

        async def stats(self, *_args, **_kwargs) -> dict:
            raise AssertionError("API lifespan must not start scheduler through stats")

    async def _config_ping(*, timeout_s: float = 5.0) -> dict:
        calls.append("config.ping")
        return {"ok": True}

    async def _usage_store() -> _UsageStore:
        return _UsageStore()

    async def _close_usage_store() -> None:
        calls.append("usage.close")

    async def _close_http_clients() -> None:
        calls.append("http.close")

    monkeypatch.setattr(app_module, "init_ray", lambda *_args, **_kwargs: calls.append("init_ray"))
    monkeypatch.setattr(app_module, "ray_connection_epoch", lambda: 0)
    monkeypatch.setattr(app_module, "ray_reconnect_poll_s", lambda: 3600.0)
    monkeypatch.setattr(app_module, "_should_preload_openai_tokenizers", lambda: False)
    monkeypatch.setattr(app_module.openai_compat, "shutdown_tokenizer_executor", lambda: None)
    monkeypatch.setattr(app_module.openai_compat, "preload_supported_tokenizers", lambda: [])
    monkeypatch.setattr(app_module, "close_http_clients", _close_http_clients)

    import mint_server.backend.action_session_manager as action_manager_module
    import mint_server.backend.config_actor as config_actor_module
    import mint_server.backend.maintenance_cron_actor as cron_module
    import mint_server.backend.model_work_scheduler as scheduler_module
    import mint_server.backend.task_state_store as task_state_module
    import mint_server.usage_store as usage_store_module

    class _TaskStateStore:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls.append("task_state.ping")
            return {"ok": True}

        async def async_ensure_started(self) -> None:
            raise AssertionError("API lifespan must not create TaskStateStore")

    class _TaskFutures:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls.append("task_futures.ping")
            return {"ok": True}

        async def async_ensure_started(self) -> None:
            raise AssertionError("API lifespan must not create task futures")

    monkeypatch.setattr(action_manager_module, "ActionSessionRouter", _ActionSessionRouter)
    monkeypatch.setattr(config_actor_module, "async_ping", _config_ping)
    monkeypatch.setattr(cron_module, "maintenance_cron_actor", _MaintenanceCron())
    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _Scheduler())
    monkeypatch.setattr(task_state_module, "task_state_store", _TaskStateStore())
    monkeypatch.setattr(task_state_module, "task_futures", _TaskFutures())
    monkeypatch.setattr(usage_store_module, "get_usage_store", _usage_store)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _close_usage_store)

    app = SimpleNamespace()

    async def _run() -> None:
        async with app_module.lifespan(app):
            assert "config.ping" in calls
            assert "task_state.ping" in calls
            assert "task_futures.ping" in calls
            assert "scheduler.ping" in calls
            assert "cron.ping" in calls
            assert "cron.snapshot.create=False" in calls

    try:
        asyncio.run(_run())
    finally:
        with suppress(Exception):
            app_module.action_sampling.action_session_manager = None

    assert "usage.close" in calls
    assert "http.close" in calls
