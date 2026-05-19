from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_issue_364_future_reaper_once_reaps_task_state(monkeypatch) -> None:
    from mint_server.backend import maintenance_cron_actor as ors

    class _FakeTaskFutureService:
        async def async_ensure_started(self) -> dict:
            raise AssertionError("Cron future reaper must not ensure TaskStateStore")

        async def async_reap(self) -> dict:
            return {
                "expired": ["req-expired"],
                "timed_out": ["req-timeout"],
                "payload_evicted": ["req-payload"],
                "tombstones_deleted": ["req-tombstone"],
                "payload_evict_errors": [{"request_id": "req-error", "error": "boom"}],
            }

    import importlib

    task_state_store_module = importlib.import_module("mint_server.backend.task_state_store")

    monkeypatch.setattr(task_state_store_module, "task_futures", _FakeTaskFutureService())

    out = ors.run_future_reaper_once()

    assert out == {
        "expired": ["req-expired"],
        "timed_out": ["req-timeout"],
        "payload_evicted": ["req-payload"],
        "tombstones_deleted": ["req-tombstone"],
        "payload_evict_errors": [{"request_id": "req-error", "error": "boom"}],
    }


def test_issue_364_checkpoint_helpers_proxy_results(monkeypatch) -> None:
    from mint_server.backend import maintenance_cron_actor as ors

    monkeypatch.setattr(ors, "reap_runtime_checkpoints", lambda: {"ephemeral": ["a"], "persistent_cache": [], "persistent": []})
    monkeypatch.setattr(ors, "process_pending_checkpoint_mirrors", lambda: {"mirrored": ["m1"], "failed": ["f1"]})

    assert ors.run_checkpoint_reaper_once() == {
        "ephemeral": ["a"],
        "persistent_cache": [],
        "persistent": [],
    }
    assert ors.run_checkpoint_mirror_once() == {"mirrored": ["m1"], "failed": ["f1"]}


def test_issue_364_training_cleanup_runner_proxies_results(monkeypatch) -> None:
    from mint_server.backend import maintenance_cron_actor as ors

    async def _fake_cleanup(*, stale_after_s=None):
        return ["model-a", "model-b"]

    monkeypatch.setattr(
        "mint_server.backend.training_cleanup_executor.cleanup_stale_training_sessions_once_impl",
        _fake_cleanup,
    )

    assert ors.run_training_cleanup_once() == {"cleaned": ["model-a", "model-b"]}


def test_issue_364_training_cleanup_runner_respects_disable_env(monkeypatch) -> None:
    from mint_server.backend import maintenance_cron_actor as ors

    monkeypatch.setenv("MINT_TRAINING_HEARTBEAT_STALE_S", "0")

    assert ors.run_training_cleanup_once() == {"cleaned": []}


def test_issue_364_sampling_cleanup_runner_proxies_results(monkeypatch) -> None:
    from mint_server.backend import maintenance_cron_actor as ors

    async def _fake_cleanup(*, stale_after_s=None):
        return ["sess-a", "sess-b"]

    monkeypatch.setattr(
        "mint_server.backend.sampling_cleanup_executor.cleanup_stale_sampling_sessions_once_impl",
        _fake_cleanup,
    )

    assert ors.run_sampling_cleanup_once() == {"cleaned": ["sess-a", "sess-b"]}


@pytest.mark.anyio
async def test_issue_364_runtime_degraded_cron_is_internal_health_degraded(monkeypatch) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module
    from mint_server import health_checks
    from mint_server.health_state import clear_runtime_degraded_state, set_runtime_degraded_state

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "snapshot_generated_at": now,
                "desired_total": 1,
                "managed_total": 1,
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    clear_runtime_degraded_state()
    set_runtime_degraded_state(
        reason="maintenance_cron_actor_unavailable",
        error="boom",
        details={
            "x": 1,
            "snapshot": {
                "loops": {
                    "checkpoint_reaper": {
                        "last_error_traceback": "Traceback secret/path.py checkpoint boom",
                    },
                },
            },
        },
    )
    try:
        out = await health_checks.internal_lightweight_healthz_response()
        assert out["status"] == "degraded"
        assert out["maintenance_cron_actor"]["reason"] == "maintenance_cron_actor_unavailable"
    finally:
        clear_runtime_degraded_state()


@pytest.mark.anyio
async def test_issue_364_internal_maintenance_cron_actor_health(monkeypatch) -> None:
    from mint_server.routes import internal

    class _FakeMaintenanceCronActor:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0, create_if_missing: bool = True):
            assert create_if_missing is False
            return {
                "actor_name": "mint_maintenance_cron",
                "epoch_id": "epoch-1",
                "timeout_s": float(timeout_s),
            }

    monkeypatch.setattr("mint_server.backend.maintenance_cron_actor.maintenance_cron_actor", _FakeMaintenanceCronActor())

    out = await internal.maintenance_cron_actor_health()

    assert out["actor_name"] == "mint_maintenance_cron"
    assert out["epoch_id"] == "epoch-1"
    assert out["timeout_s"] == 10.0



def test_issue_364_maintenance_cron_loop_snapshot_includes_error_details(monkeypatch):
    from mint_server.backend import maintenance_cron_actor as ors

    actor_cls_box = {}

    class _FakeRemoteActorClass:
        def __init__(self, cls):
            actor_cls_box["cls"] = cls

        def options(self, **_kwargs):
            return self

        def remote(self):
            raise AssertionError("actor creation is not needed for this test")

    class _FakeRay:
        @staticmethod
        def remote(**_kwargs):
            def _wrap(cls):
                return _FakeRemoteActorClass(cls)
            return _wrap

        @staticmethod
        def get_actor(*_args, **_kwargs):
            raise ValueError("missing")

    monkeypatch.setitem(__import__("sys").modules, "ray", _FakeRay)
    monkeypatch.setattr(ors, "run_checkpoint_reaper_once", lambda: (_ for _ in ()).throw(RuntimeError("checkpoint boom")))
    monkeypatch.setattr(ors, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ors, "otel_env_vars", lambda: {})
    monkeypatch.setattr(ors, "actor_runtime_env", lambda **_kwargs: {})

    try:
        ors._get_or_create_actor()
    except (AssertionError, ValueError):
        pass

    actor = actor_cls_box["cls"]()
    out = __import__("asyncio").run(actor._run_loop_once("checkpoint_reaper"))
    snapshot = actor.health_snapshot()
    loop = snapshot["loops"]["checkpoint_reaper"]

    assert out["error_type"] == "RuntimeError"
    assert loop["last_error"] == "RuntimeError: checkpoint boom"
    assert loop["last_error_type"] == "RuntimeError"
    assert "last_error_traceback" not in loop


def test_issue_593_maintenance_cron_does_not_own_model_reconcile_loops(monkeypatch):
    from mint_server.backend import maintenance_cron_actor as ors

    actor_cls_box = {}

    class _FakeRemoteActorClass:
        def __init__(self, cls):
            actor_cls_box["cls"] = cls

        def options(self, **_kwargs):
            return self

        def remote(self):
            raise AssertionError("actor creation is not needed for this test")

    class _FakeRay:
        @staticmethod
        def remote(**_kwargs):
            def _wrap(cls):
                return _FakeRemoteActorClass(cls)
            return _wrap

        @staticmethod
        def get_actor(*_args, **_kwargs):
            raise ValueError("missing")

    monkeypatch.setitem(__import__("sys").modules, "ray", _FakeRay)
    monkeypatch.setattr(ors, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ors, "otel_env_vars", lambda: {})
    monkeypatch.setattr(ors, "actor_runtime_env", lambda **_kwargs: {})

    try:
        ors._get_or_create_actor()
    except (AssertionError, ValueError):
        pass

    actor = actor_cls_box["cls"]()
    loops = actor.health_snapshot()["loops"]

    assert "model_actor_supervisor" not in loops
    assert "actor_reconciliation" not in loops
