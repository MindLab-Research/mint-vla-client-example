from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_issue_364_future_reaper_once_reaps_task_state(monkeypatch) -> None:
    from tinker_server.backend import maintenance_cron_actor as ors

    class _FakeFutureStore:
        async def async_ensure_started(self) -> dict:
            return {"ok": True}

        async def async_reap(self) -> dict:
            return {"expired": ["req-expired"], "timed_out": ["req-timeout"]}

    import importlib

    task_state_store_module = importlib.import_module("tinker_server.backend.task_state_store")

    monkeypatch.setattr(task_state_store_module, "task_state_futures", _FakeFutureStore())

    out = ors.run_future_reaper_once()

    assert out == {
        "expired": ["req-expired"],
        "timed_out": ["req-timeout"],
    }


def test_issue_364_checkpoint_helpers_proxy_results(monkeypatch) -> None:
    from tinker_server.backend import maintenance_cron_actor as ors

    monkeypatch.setattr(ors, "reap_runtime_checkpoints", lambda: {"ephemeral": ["a"], "persistent_cache": [], "persistent": []})
    monkeypatch.setattr(ors, "process_pending_checkpoint_mirrors", lambda: {"mirrored": ["m1"], "failed": ["f1"]})

    assert ors.run_checkpoint_reaper_once() == {
        "ephemeral": ["a"],
        "persistent_cache": [],
        "persistent": [],
    }
    assert ors.run_checkpoint_mirror_once() == {"mirrored": ["m1"], "failed": ["f1"]}


def test_issue_364_training_cleanup_runner_proxies_results(monkeypatch) -> None:
    from tinker_server.backend import maintenance_cron_actor as ors

    class _FakeTrainingCleanupExecutor:
        async def async_cleanup_stale_sessions_once(self, *, stale_after_s=None):
            return ["model-a", "model-b"]

    monkeypatch.setattr(
        "tinker_server.backend.training_cleanup_executor.training_cleanup_executor",
        _FakeTrainingCleanupExecutor(),
    )

    assert ors.run_training_cleanup_once() == {"cleaned": ["model-a", "model-b"]}


def test_issue_364_training_cleanup_runner_respects_disable_env(monkeypatch) -> None:
    from tinker_server.backend import maintenance_cron_actor as ors

    monkeypatch.setenv("MINT_TRAINING_HEARTBEAT_STALE_S", "0")

    assert ors.run_training_cleanup_once() == {"cleaned": []}


def test_issue_364_sampling_cleanup_runner_proxies_results(monkeypatch) -> None:
    from tinker_server.backend import maintenance_cron_actor as ors

    class _FakeSamplingCleanupExecutor:
        async def async_cleanup_stale_sessions_once(self):
            return ["sess-a", "sess-b"]

    monkeypatch.setattr(
        "tinker_server.backend.sampling_cleanup_executor.sampling_cleanup_executor",
        _FakeSamplingCleanupExecutor(),
    )

    assert ors.run_sampling_cleanup_once() == {"cleaned": ["sess-a", "sess-b"]}


def test_issue_364_runtime_degraded_healthz() -> None:
    from fastapi.responses import JSONResponse

    from tinker_server.health_checks import public_healthz_response
    from tinker_server.health_state import clear_runtime_degraded_state, set_runtime_degraded_state

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
        out = public_healthz_response()
        assert isinstance(out, JSONResponse)
        assert out.status_code == 503
        assert b'maintenance_cron_actor_unavailable' in out.body
        assert b"last_error_traceback" not in out.body
        assert b"Traceback secret/path.py" not in out.body
    finally:
        clear_runtime_degraded_state()


@pytest.mark.anyio
async def test_issue_364_internal_maintenance_cron_actor_health(monkeypatch) -> None:
    from tinker_server.routes import internal

    class _FakeMaintenanceCronActor:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0):
            return {
                "actor_name": "mint_maintenance_cron",
                "epoch_id": "epoch-1",
                "timeout_s": float(timeout_s),
            }

    monkeypatch.setattr("tinker_server.backend.maintenance_cron_actor.maintenance_cron_actor", _FakeMaintenanceCronActor())

    out = await internal.maintenance_cron_actor_health()

    assert out["actor_name"] == "mint_maintenance_cron"
    assert out["epoch_id"] == "epoch-1"
    assert out["timeout_s"] == 10.0



def test_issue_364_maintenance_cron_loop_snapshot_includes_error_details(monkeypatch):
    from tinker_server.backend import maintenance_cron_actor as ors

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
