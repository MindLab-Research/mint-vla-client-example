from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import time
from types import SimpleNamespace

import pytest


def _reload_openpi_ray_runtime(monkeypatch):
    import mint_server.config as config

    monkeypatch.setattr(config, "PFS_PYTHONPATH", "/runtime/site-packages:/repo:/hf")
    monkeypatch.setattr(config, "PFS_RUNTIME_ENV_ROOT", "/runtime")
    monkeypatch.setattr(config, "MINT_CODE_ROOT", "/repo")
    monkeypatch.setattr(config, "PFS_HF_MODULES_PATH", "/hf")

    def _fake_actor_runtime_env_vars(
        *,
        pythonpath: str,
        extra: dict[str, str] | None = None,
        include_ray_attach_hints: bool = True,
    ) -> dict[str, str]:
        del include_ray_attach_hints
        out = {
            "PFS_RUNTIME_ENV_ROOT": "/runtime",
            "MINT_CODE_ROOT": "/repo",
            "PFS_HF_MODULES_PATH": "/hf",
            "RAY_ADDRESS": "auto",
            "MINT_RAY_NAMESPACE": "mint",
            "PYTHONPATH": pythonpath,
        }
        if extra:
            out.update(extra)
        return out

    monkeypatch.setattr(config, "actor_runtime_env_vars", _fake_actor_runtime_env_vars)
    import mint_server.backend.openpi.openpi_ray_runtime as openpi_ray_runtime

    return importlib.reload(openpi_ray_runtime)


def _spec():
    from mint_server.backend.openpi.openpi_fast_runtime import OpenPIFastRuntimeSpec

    return OpenPIFastRuntimeSpec(
        startup_timeout_s=30.0,
        create_session_timeout_s=300.0,
        request_timeout_s=300.0,
        save_weights_timeout_s=300.0,
        load_weights_timeout_s=300.0,
    )


def test_openpi_ray_runtime_env_vars_forward_mint_openpi_overrides(monkeypatch) -> None:
    openpi_ray_runtime = _reload_openpi_ray_runtime(monkeypatch)

    monkeypatch.setenv("MINT_OPENPI_FAST_WEIGHTS_PATH", "/tmp/fast-weights")
    monkeypatch.setenv("MINT_OPENPI_PI05_ASSETS_BASE_DIR", "/tmp/pi05-assets")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-home")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OPENPI_DATA_HOME", "/tmp/openpi-data")
    monkeypatch.setenv("UNRELATED_ENV", "ignore-me")

    env_vars = openpi_ray_runtime._openpi_runtime_env_vars()

    assert env_vars["MINT_OPENPI_FAST_WEIGHTS_PATH"] == "/tmp/fast-weights"
    assert env_vars["MINT_OPENPI_PI05_ASSETS_BASE_DIR"] == "/tmp/pi05-assets"
    assert env_vars["HF_HOME"] == "/tmp/hf-home"
    assert env_vars["HF_HUB_OFFLINE"] == "1"
    assert env_vars["OPENPI_DATA_HOME"] == "/tmp/openpi-data"
    assert env_vars["PYTHONPATH"] == "/runtime/site-packages:/repo:/hf"
    assert env_vars["PFS_RUNTIME_ENV_ROOT"] == "/runtime"
    assert env_vars["MINT_CODE_ROOT"] == "/repo"
    assert env_vars["PFS_HF_MODULES_PATH"] == "/hf"
    assert "UNRELATED_ENV" not in env_vars


def test_openpi_ray_actor_ready_timeout_defaults_to_extended_budget(monkeypatch) -> None:
    from mint_server.backend.openpi.openpi_ray_runtime import _actor_ready_timeout_s

    monkeypatch.delenv("MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S", raising=False)

    assert _actor_ready_timeout_s(_spec()) == 300.0


def test_openpi_ray_actor_ready_timeout_respects_override(monkeypatch) -> None:
    from mint_server.backend.openpi.openpi_ray_runtime import _actor_ready_timeout_s

    monkeypatch.setenv("MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S", "123")

    assert _actor_ready_timeout_s(_spec()) == 123.0


def test_start_openpi_ray_runtime_passes_model_and_training_session_ids(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_ray_runtime

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


def test_openpi_ray_runtime_client_ready_uses_metadata_method(monkeypatch) -> None:
    from mint_server.backend.openpi.openpi_ray_runtime import OpenPIRayRuntimeClient

    class _Method:
        def __init__(self, value):
            self._value = value

        def remote(self):
            return self._value

    actor = SimpleNamespace(
        __ray_ready__=_Method(True),
        ready_metadata=_Method({"actor_id": "abc", "node_ip": "192.168.0.1"}),
    )
    client = OpenPIRayRuntimeClient(
        actor=actor,
        spec=_spec(),
        session_id="session-1",
        ready_timeout_s=30.0,
    )

    async def _fake_ray_get(ref, *, timeout_s):
        return ref

    monkeypatch.setattr(client, "_ray_get", _fake_ray_get)

    metadata = asyncio.run(client.ready())

    assert metadata == {"actor_id": "abc", "node_ip": "192.168.0.1"}
    assert client.metadata == metadata


def test_openpi_ray_runtime_client_ray_get_awaits_future_without_ray_get(monkeypatch) -> None:
    pytest.importorskip("ray")
    from mint_server.backend.openpi.openpi_ray_runtime import OpenPIRayRuntimeClient

    client = OpenPIRayRuntimeClient(
        actor=object(),
        spec=_spec(),
        session_id="session-1",
        ready_timeout_s=30.0,
    )
    fut: concurrent.futures.Future[dict[str, object]] = concurrent.futures.Future()
    fut.set_result({"ok": True})
    ref = SimpleNamespace(future=lambda: fut)

    monkeypatch.setattr(
        "mint_server.backend.openpi.openpi_ray_runtime.ray.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get should not be called")),
    )

    assert asyncio.run(client._ray_get(ref, timeout_s=1.0)) == {"ok": True}


def test_openpi_ray_runtime_client_ray_get_preserves_timeout_surface(monkeypatch) -> None:
    ray = pytest.importorskip("ray")
    from mint_server.backend.openpi.openpi_ray_runtime import OpenPIRayRuntimeClient
    from mint_server.backend.openpi.openpi_fast_runtime import OpenPIFastWorkerProtocolError

    client = OpenPIRayRuntimeClient(
        actor=object(),
        spec=_spec(),
        session_id="session-1",
        ready_timeout_s=30.0,
    )
    fut: concurrent.futures.Future[dict[str, object]] = concurrent.futures.Future()
    ref = SimpleNamespace(future=lambda: fut)

    monkeypatch.setattr(
        "mint_server.backend.openpi.openpi_ray_runtime.ray.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get should not be called")),
    )

    with pytest.raises(OpenPIFastWorkerProtocolError) as exc_info:
        asyncio.run(client._ray_get(ref, timeout_s=0.001))

    assert isinstance(exc_info.value.__cause__, ray.exceptions.GetTimeoutError)


def test_ray_keepalive_awaits_future_without_ray_get(monkeypatch) -> None:
    pytest.importorskip("ray")
    from mint_server.backend.actors import ray_keepalive

    fut: concurrent.futures.Future[dict[str, object]] = concurrent.futures.Future()
    fut.set_result({"ok": True})
    ref = SimpleNamespace(future=lambda: fut)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "mint_server.backend.actors.ray_keepalive.ray.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get should not be called")),
    )
    monkeypatch.setattr(
        ray_keepalive,
        "get_model_actor_supervisor",
        lambda: SimpleNamespace(
            mark_inflight=lambda actor_name, delta: calls.append(("mark_inflight", actor_name, delta)),
            touch=lambda actor_name: calls.append(("touch", actor_name)),
        ),
    )

    result = asyncio.run(
        ray_keepalive.ray_get_with_model_actor_supervisor_keepalive(
            ref,
            actor_name="actor-1",
            interval_s=1.0,
            timeout_s=5.0,
        )
    )

    assert result == {"ok": True}
    assert calls == [
        ("mark_inflight", "actor-1", 1),
        ("touch", "actor-1"),
        ("mark_inflight", "actor-1", -1),
    ]


def test_ray_keepalive_preserves_periodic_touch_while_waiting(monkeypatch) -> None:
    pytest.importorskip("ray")
    from mint_server.backend.actors import ray_keepalive

    future: concurrent.futures.Future[dict[str, object]] = concurrent.futures.Future()
    ref = SimpleNamespace(future=lambda: future)
    touches: list[str] = []

    monkeypatch.setattr(
        "mint_server.backend.actors.ray_keepalive.ray.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get should not be called")),
    )
    def _touch(actor_name: str) -> None:
        touches.append(actor_name)
        if len(touches) >= 2 and not future.done():
            future.set_result({"ok": True})

    monkeypatch.setattr(
        ray_keepalive,
        "get_model_actor_supervisor",
        lambda: SimpleNamespace(
            mark_inflight=lambda *_args: None,
            touch=_touch,
        ),
    )

    start = time.time()
    assert asyncio.run(
        ray_keepalive.ray_get_with_model_actor_supervisor_keepalive(
            ref,
            actor_name="actor-1",
            interval_s=0.01,
            timeout_s=1.0,
        )
    ) == {"ok": True}
    assert len(touches) >= 2
    assert time.time() - start < 1.0


def test_ray_keepalive_cancellation_silences_late_exception(monkeypatch) -> None:
    pytest.importorskip("ray")
    from mint_server.backend.ray_cluster import async_ray_control
    from mint_server.backend.actors import ray_keepalive

    discarded: list[str] = []

    def _record_late_result(fut: asyncio.Future) -> None:
        try:
            fut.result()
        except RuntimeError as exc:
            discarded.append(str(exc))
        except BaseException as exc:
            discarded.append(type(exc).__name__)

    monkeypatch.setattr(async_ray_control, "_discard_late_result", _record_late_result)
    monkeypatch.setattr(
        ray_keepalive,
        "get_model_actor_supervisor",
        lambda: SimpleNamespace(
            mark_inflight=lambda *_args: None,
            touch=lambda *_args: None,
        ),
    )

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        ref = SimpleNamespace(future=lambda: fut)
        task = asyncio.create_task(
            ray_keepalive.ray_get_with_model_actor_supervisor_keepalive(
                ref,
                actor_name="actor-1",
                interval_s=1.0,
                timeout_s=60.0,
            )
        )
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
