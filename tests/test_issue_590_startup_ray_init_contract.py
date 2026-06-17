from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest


def test_lifespan_fails_before_yield_when_ray_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import app as app_module

    calls: list[str] = []

    def _fail_init_ray(*_args, **_kwargs) -> None:
        raise RuntimeError("ray startup unavailable")

    monkeypatch.setattr(app_module, "configure_logging", lambda: calls.append("configure_logging"))
    monkeypatch.setattr(app_module, "init_ray", _fail_init_ray)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="ray startup unavailable"):
            async with app_module.lifespan(app_module.app):
                raise AssertionError("lifespan should not yield when Ray startup fails")

    asyncio.run(_run())
    assert calls == ["configure_logging"]


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

    class _Supervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            calls.append("supervisor.snapshot")
            return {"ok": True, "desired_total": 0, "managed_total": 0}

    async def _config_ping(*, timeout_s: float = 5.0) -> dict:
        calls.append("config.ping")
        return {"ok": True}

    async def _usage_store() -> _UsageStore:
        return _UsageStore()

    async def _close_usage_store() -> None:
        calls.append("usage.close")

    async def _close_http_clients() -> None:
        calls.append("http.close")

    monkeypatch.setattr(app_module, "configure_logging", lambda: calls.append("configure_logging"))
    monkeypatch.setattr(app_module, "init_ray", lambda *_args, **_kwargs: calls.append("init_ray"))
    monkeypatch.setattr(app_module, "ray_connection_epoch", lambda: 0)
    monkeypatch.setattr(app_module, "ray_reconnect_poll_s", lambda: 3600.0)
    monkeypatch.setattr(app_module, "_should_preload_openai_tokenizers", lambda: False)
    monkeypatch.setattr(app_module.openai_compat, "shutdown_tokenizer_executor", lambda: None)
    monkeypatch.setattr(app_module.openai_compat, "preload_supported_tokenizers", lambda: [])
    monkeypatch.setattr(app_module, "close_http_clients", _close_http_clients)

    import mint_server.backend.openpi.action_session_manager as action_manager_module
    import mint_server.backend.core.config_actor as config_actor_module
    import mint_server.backend.ops.maintenance_cron_actor as cron_module
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.billing.usage_store as usage_store_module

    monkeypatch.setattr(action_manager_module, "ActionSessionRouter", _ActionSessionRouter)
    monkeypatch.setattr(config_actor_module, "async_ping", _config_ping)
    monkeypatch.setattr(cron_module, "maintenance_cron_actor", _MaintenanceCron())
    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", _Supervisor())
    monkeypatch.setattr(usage_store_module, "get_usage_store", _usage_store)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _close_usage_store)

    app = SimpleNamespace()

    async def _run() -> None:
        async with app_module.lifespan(app):
            assert "config.ping" in calls
            assert "supervisor.snapshot" in calls
            assert "cron.snapshot.create=False" in calls

    try:
        asyncio.run(_run())
    finally:
        with suppress(Exception):
            app_module.action_sampling.action_session_manager = None

    assert "usage.close" in calls
    assert "http.close" in calls
    assert calls.index("configure_logging") < calls.index("init_ray")


def test_lifespan_leaves_execution_route_globals_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import app as app_module
    from mint_server.routes import sampling, service, training, weights

    calls: list[str] = []

    class _UsageStore:
        async def health_check(self) -> bool:
            return True

    class _ActionSessionRouter:
        pass

    class _MaintenanceCron:
        async def async_health_snapshot(
            self,
            *,
            timeout_s: float = 10.0,
            create_if_missing: bool = True,
        ) -> dict:
            assert create_if_missing is False
            return {
                "actor_name": "mint_maintenance_cron",
                "namespace": "mint",
                "epoch_id": "epoch",
                "code_identity": app_module._git_sha(),
                "loops": {},
            }

    class _Supervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {"ok": True, "desired_total": 0, "managed_total": 0}

    async def _config_ping(*, timeout_s: float = 5.0) -> dict:
        return {"ok": True}

    async def _usage_store() -> _UsageStore:
        return _UsageStore()

    async def _close_usage_store() -> None:
        calls.append("usage.close")

    async def _close_http_clients() -> None:
        calls.append("http.close")

    stale_manager = object()
    stale_engine = object()
    monkeypatch.setattr(service, "session_manager", stale_manager)
    monkeypatch.setattr(sampling, "session_manager", stale_manager)
    monkeypatch.setattr(training, "training_manager", stale_manager)
    monkeypatch.setattr(training, "training_engine", stale_engine)
    monkeypatch.setattr(training, "inference_manager", stale_manager)
    monkeypatch.setattr(weights, "training_manager", stale_manager)
    monkeypatch.setattr(weights, "training_engine", stale_engine)
    monkeypatch.setattr(weights, "inference_manager", stale_manager)
    if app_module.mint is not None:
        monkeypatch.setattr(app_module.mint, "training_manager", stale_manager)
        monkeypatch.setattr(app_module.mint, "training_engine", stale_engine)

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(app_module, "init_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "ray_connection_epoch", lambda: 0)
    monkeypatch.setattr(app_module, "ray_reconnect_poll_s", lambda: 3600.0)
    monkeypatch.setattr(app_module, "_should_preload_openai_tokenizers", lambda: False)
    monkeypatch.setattr(app_module.openai_compat, "shutdown_tokenizer_executor", lambda: None)
    monkeypatch.setattr(app_module.openai_compat, "preload_supported_tokenizers", lambda: [])
    monkeypatch.setattr(app_module, "close_http_clients", _close_http_clients)

    import mint_server.backend.openpi.action_session_manager as action_manager_module
    import mint_server.backend.core.config_actor as config_actor_module
    import mint_server.backend.ops.maintenance_cron_actor as cron_module
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.billing.usage_store as usage_store_module

    monkeypatch.setattr(action_manager_module, "ActionSessionRouter", _ActionSessionRouter)
    monkeypatch.setattr(config_actor_module, "async_ping", _config_ping)
    monkeypatch.setattr(cron_module, "maintenance_cron_actor", _MaintenanceCron())
    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", _Supervisor())
    monkeypatch.setattr(usage_store_module, "get_usage_store", _usage_store)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _close_usage_store)

    async def _assert_route_globals_unbound() -> None:
        async with app_module.lifespan(SimpleNamespace()):
            assert service.session_manager is None
            assert sampling.session_manager is None
            assert training.training_manager is None
            assert training.training_engine is None
            assert training.inference_manager is None
            assert weights.training_manager is None
            assert weights.training_engine is None
            assert weights.inference_manager is None
            if app_module.mint is not None:
                assert app_module.mint.training_manager is None
                assert app_module.mint.training_engine is None

    try:
        asyncio.run(_assert_route_globals_unbound())
    finally:
        with suppress(Exception):
            app_module.action_sampling.action_session_manager = None

    assert calls == ["usage.close", "http.close"]


def test_lifespan_degrades_but_yields_when_usage_postgres_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import app as app_module
    from mint_server.health.health_state import clear_startup_degraded_state, get_startup_degraded_state

    calls: list[str] = []

    class _UsageStore:
        async def health_check(self) -> bool:
            calls.append("usage.health_check")
            return False

    class _ActionSessionRouter:
        pass

    class _MaintenanceCron:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            return {"actor_name": "mint_maintenance_cron", "epoch_id": "epoch", "loops": {}}

    class _Supervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {"ok": True, "desired_total": 0, "managed_total": 0}

    async def _config_ping(*, timeout_s: float = 5.0) -> dict:
        return {"ok": True}

    async def _usage_store() -> _UsageStore:
        return _UsageStore()

    async def _close_usage_store() -> None:
        pass

    async def _close_http_clients() -> None:
        pass

    monkeypatch.setattr(app_module, "configure_logging", lambda: calls.append("configure_logging"))
    monkeypatch.setattr(app_module, "init_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "ray_connection_epoch", lambda: 0)
    monkeypatch.setattr(app_module, "ray_reconnect_poll_s", lambda: 3600.0)
    monkeypatch.setattr(app_module, "_should_preload_openai_tokenizers", lambda: False)
    monkeypatch.setattr(app_module.openai_compat, "shutdown_tokenizer_executor", lambda: None)
    monkeypatch.setattr(app_module.openai_compat, "preload_supported_tokenizers", lambda: [])
    monkeypatch.setattr(app_module, "close_http_clients", _close_http_clients)

    import mint_server.backend.openpi.action_session_manager as action_manager_module
    import mint_server.backend.core.config_actor as config_actor_module
    import mint_server.backend.ops.maintenance_cron_actor as cron_module
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.billing.usage_store as usage_store_module

    monkeypatch.setattr(action_manager_module, "ActionSessionRouter", _ActionSessionRouter)
    monkeypatch.setattr(config_actor_module, "async_ping", _config_ping)
    monkeypatch.setattr(cron_module, "maintenance_cron_actor", _MaintenanceCron())
    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", _Supervisor())
    monkeypatch.setattr(usage_store_module, "get_usage_store", _usage_store)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _close_usage_store)

    clear_startup_degraded_state()

    async def _run() -> None:
        async with app_module.lifespan(SimpleNamespace()):
            degraded = get_startup_degraded_state()
            assert degraded is not None
            assert degraded["reason"] == "usage_billing_postgres_unavailable"

    try:
        asyncio.run(_run())
    finally:
        clear_startup_degraded_state()
        with suppress(Exception):
            app_module.action_sampling.action_session_manager = None
