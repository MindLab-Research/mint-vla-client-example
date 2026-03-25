from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace


def _spec():
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    return OpenPIFastRuntimeSpec(
        python_executable=os.sys.executable,
        startup_timeout_s=30.0,
        create_session_timeout_s=300.0,
        request_timeout_s=300.0,
        save_weights_timeout_s=300.0,
        load_weights_timeout_s=300.0,
    )


def test_openpi_ray_runtime_env_vars_forward_mint_openpi_overrides(monkeypatch) -> None:
    from tinker_server.backend.openpi_ray_runtime import _openpi_runtime_env_vars

    monkeypatch.setenv("MINT_OPENPI_FAST_WEIGHTS_PATH", "/tmp/fast-weights")
    monkeypatch.setenv("MINT_OPENPI_PI05_ASSETS_BASE_DIR", "/tmp/pi05-assets")
    monkeypatch.setenv("UNRELATED_ENV", "ignore-me")

    env_vars = _openpi_runtime_env_vars()

    assert env_vars["MINT_OPENPI_FAST_WEIGHTS_PATH"] == "/tmp/fast-weights"
    assert env_vars["MINT_OPENPI_PI05_ASSETS_BASE_DIR"] == "/tmp/pi05-assets"
    assert "UNRELATED_ENV" not in env_vars
    assert env_vars["PYTHONPATH"]


def test_openpi_ray_actor_ready_timeout_defaults_to_extended_budget(monkeypatch) -> None:
    from tinker_server.backend.openpi_ray_runtime import _actor_ready_timeout_s

    monkeypatch.delenv("MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S", raising=False)

    assert _actor_ready_timeout_s(_spec()) == 300.0


def test_openpi_ray_actor_ready_timeout_respects_override(monkeypatch) -> None:
    from tinker_server.backend.openpi_ray_runtime import _actor_ready_timeout_s

    monkeypatch.setenv("MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S", "123")

    assert _actor_ready_timeout_s(_spec()) == 123.0


def test_start_openpi_ray_runtime_passes_model_and_training_session_ids(monkeypatch) -> None:
    from tinker_server.backend import openpi_ray_runtime

    actor_state: dict[str, object] = {}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            actor_state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            actor_state["remote"] = kwargs
            return "fake-actor-handle"

    class _FakeClient:
        def __init__(self, *, actor, spec, session_id, ready_timeout_s):
            actor_state["client_init"] = {
                "actor": actor,
                "spec": spec,
                "session_id": session_id,
                "ready_timeout_s": ready_timeout_s,
            }

        async def ready(self):
            actor_state["ready_called"] = True
            return {"actor_id": "abc"}

        async def close(self):
            actor_state["close_called"] = True

    monkeypatch.setattr(openpi_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_ray_runtime, "OpenPIRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_ray_runtime, "OpenPIRayRuntimeClient", _FakeClient)

    client = asyncio.run(
        openpi_ray_runtime.start_openpi_ray_runtime(
            session=SimpleNamespace(model_id="model-1", session_id="session-1"),
            spec=_spec(),
        )
    )

    assert isinstance(client, _FakeClient)
    assert actor_state["remote"] == {
        "model_id": "model-1",
        "training_session_id": "session-1",
        "spec": _spec(),
    }
    assert actor_state["client_init"]["ready_timeout_s"] == 300.0
    assert actor_state["ready_called"] is True
