from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from types import SimpleNamespace

import pytest

from tinker_server.backend.training_session_manager import TrainingSession


def _spec(*, worker_module: str = "tinker_server.backend.openpi_fast_worker"):
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    return OpenPIFastRuntimeSpec(
        python_executable=os.sys.executable,
        worker_module=worker_module,
        startup_timeout_s=30.0,
        create_session_timeout_s=300.0,
        request_timeout_s=300.0,
        save_weights_timeout_s=300.0,
        load_weights_timeout_s=300.0,
    )


def _make_session(model_id: str, session_id: str) -> TrainingSession:
    return TrainingSession(
        model_id=model_id,
        session_id=session_id,
        model_seq_id=0,
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
    )


def _model_config():
    return SimpleNamespace(
        action_dim=32,
        action_horizon=10,
        action_token_budget=21,
        max_model_len=200,
    )


def _create_payload(session: TrainingSession, *, learning_rate: float = 1e-4) -> dict[str, object]:
    return {
        "model_id": session.model_id,
        "session_id": session.session_id,
        "base_model": session.base_model,
        "config_name": "pi0_fast_libero_low_mem_finetune",
        "learning_rate": learning_rate,
        "action_dim": 32,
        "action_horizon": 10,
        "action_token_budget": 21,
        "max_token_len": 200,
        "camera_layout": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    }


def _completed_ref(value: object) -> SimpleNamespace:
    fut: concurrent.futures.Future[object] = concurrent.futures.Future()
    fut.set_result(value)
    return SimpleNamespace(future=lambda: fut)


def _failed_ref(exc: BaseException) -> SimpleNamespace:
    fut: concurrent.futures.Future[object] = concurrent.futures.Future()
    fut.set_exception(exc)
    return SimpleNamespace(future=lambda: fut)


def _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime) -> None:
    monkeypatch.setenv("PFS_TINKER_PATH", "/repo")
    monkeypatch.setattr(
        openpi_shared_ray_runtime,
        "get_resource_pool",
        lambda: SimpleNamespace(
            unregister=lambda *_args, **_kwargs: None,
            register=lambda **_kwargs: None,
            mark_ready=lambda *_args, **_kwargs: None,
            touch=lambda *_args, **_kwargs: None,
            mark_inflight=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        openpi_shared_ray_runtime,
        "_openpi_runtime_env_vars",
        lambda: {
            "PYTHONPATH": "/runtime/site-packages:/repo:/hf",
            "PFS_RUNTIME_ENV_ROOT": "/runtime",
            "PFS_TINKER_PATH": "/repo",
            "PFS_HF_MODULES_PATH": "/hf",
        },
    )
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: False)
    openpi_shared_ray_runtime.clear_openpi_shared_runtime_pool()


def test_start_openpi_shared_ray_runtime_reuses_actor_for_same_pool_key(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {"remote_calls": [], "client_inits": []}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote_calls"].append(kwargs)
            return f"actor-{len(state['remote_calls'])}"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            state["client_inits"].append(
                {
                    "actor": actor,
                    "actor_name": actor_name,
                    "session_id": session_id,
                    "ready_timeout_s": ready_timeout_s,
                    "worker_module": spec.worker_module,
                    "owns_started_actor": owns_started_actor,
                }
            )

        async def ready(self):
            return {"actor_id": "abc"}

        async def close(self):
            return None

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FakeClient)

    session_a = _make_session("model-a", "session-a")
    session_b = _make_session("model-b", "session-b")

    client_a = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session_a,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )
    client_b = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session_b,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    assert isinstance(client_a, _FakeClient)
    assert isinstance(client_b, _FakeClient)
    assert len(state["remote_calls"]) == 1
    actor_name = state["remote_calls"][0]["actor_name"]
    assert state["options"]["runtime_env"]["env_vars"]["MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT"] == (
        f"/repo/checkpoints/openpi_action_session_state/tinker/{actor_name}"
    )
    assert state["client_inits"] == [
        {
            "actor": "actor-1",
            "actor_name": state["client_inits"][0]["actor_name"],
            "session_id": "model-a",
            "ready_timeout_s": 300.0,
            "worker_module": "tinker_server.backend.openpi_fast_worker",
            "owns_started_actor": True,
        },
        {
            "actor": "actor-1",
            "actor_name": state["client_inits"][0]["actor_name"],
            "session_id": "model-b",
            "ready_timeout_s": 300.0,
            "worker_module": "tinker_server.backend.openpi_fast_worker",
            "owns_started_actor": False,
        },
    ]


def test_start_openpi_shared_ray_runtime_registers_actor_metadata_in_resource_pool(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            _ = actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor

        async def ready(self):
            return {
                "actor_id": "actor-123",
                "node_id": "node-456",
                "node_ip": "192.168.0.8",
                "pid": 999,
                "cuda_visible_devices": "0",
                "current_session_id": None,
            }

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs
            return None

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"] = actor_name

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(
        openpi_shared_ray_runtime,
        "_openpi_runtime_env_vars",
        lambda: {"PYTHONPATH": "/runtime/site-packages:/repo:/hf", "PFS_TINKER_PATH": "/repo"},
    )

    session = _make_session("model-a", "session-a")
    asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    register = state["register"]
    actor_name = state["remote"]["actor_name"]
    assert state["options"]["runtime_env"]["env_vars"]["MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT"] == (
        f"/repo/checkpoints/openpi_action_session_state/tinker/{actor_name}"
    )
    assert register["actor_type"].value == "openpi"
    assert register["node_id"] == "node-456"
    assert register["metadata"]["worker_module"] == "tinker_server.backend.openpi_fast_worker"
    assert register["metadata"]["actor_id"] == "actor-123"
    assert register["metadata"]["node_ip"] == "192.168.0.8"
    assert register["metadata"]["pid"] == 999
    assert register["metadata"]["cuda_visible_devices"] == "0"


def test_start_openpi_shared_ray_runtime_refreshes_stale_cached_actor_handle(monkeypatch) -> None:
    pytest.importorskip("ray")
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {"client_inits": []}

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            state["client_inits"].append(
                {
                    "actor": actor,
                    "actor_name": actor_name,
                    "session_id": session_id,
                    "ready_timeout_s": ready_timeout_s,
                    "worker_module": spec.worker_module,
                    "owns_started_actor": owns_started_actor,
                }
            )

        async def ready(self):
            return {"actor_id": "fresh-123"}

        async def close(self):
            return None

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", lambda actor_name, namespace=None: "fresh-actor")

    session = _make_session("model-a", "session-a")
    actor_name = openpi_shared_ray_runtime._shared_actor_name(
        openpi_shared_ray_runtime._normalize_pool_key(
            spec=_spec(),
            session=session,
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )
    openpi_shared_ray_runtime._SHARED_ACTORS[actor_name] = openpi_shared_ray_runtime._SharedActorEntry(
        actor_name=actor_name,
        actor="stale-actor",
        pool_key={},
    )

    client = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    assert isinstance(client, _FakeClient)
    assert state["client_inits"] == [
        {
            "actor": "fresh-actor",
            "actor_name": actor_name,
            "session_id": "model-a",
            "ready_timeout_s": 300.0,
            "worker_module": "tinker_server.backend.openpi_fast_worker",
            "owns_started_actor": False,
        }
    ]
    assert openpi_shared_ray_runtime._SHARED_ACTORS[actor_name].actor == "fresh-actor"


def test_openpi_shared_runtime_client_refreshes_named_actor_before_request(monkeypatch) -> None:
    pytest.importorskip("ray")
    from tinker_server.backend import openpi_shared_ray_runtime

    calls: list[tuple[str, object]] = []

    class _FreshActor:
        class register_session:
            @staticmethod
            def remote(session_id, payload):
                calls.append(("register_session", session_id, payload))
                return "fresh-register-ref"

        class request_for_session:
            @staticmethod
            def remote(session_id, op, payload, timeout_s=None):
                calls.append(("request_for_session", session_id, op, payload, timeout_s))
                return "fresh-request-ref"

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", lambda actor_name, namespace=None: _FreshActor())

    client = openpi_shared_ray_runtime.OpenPISharedRayRuntimeClient(
        actor="stale-actor",
        actor_name="openpi_shared_runtime_deadbeef",
        spec=_spec(),
        session_id="model-a",
        ready_timeout_s=300.0,
    )

    async def _fake_ray_get(ref, *, timeout_s):
        calls.append(("ray_get", ref, timeout_s))
        return {"ok": True}

    monkeypatch.setattr(client, "_ray_get", _fake_ray_get)

    result = asyncio.run(client.request("create_session", {"foo": "bar"}))

    assert result == {"ok": True}
    assert calls == [
        ("register_session", "model-a", {"foo": "bar"}),
        ("ray_get", "fresh-register-ref", None),
    ]


def test_openpi_shared_ray_runtime_client_ray_get_awaits_future_without_ray_get(monkeypatch) -> None:
    pytest.importorskip("ray")
    from tinker_server.backend.openpi_shared_ray_runtime import OpenPISharedRayRuntimeClient

    client = OpenPISharedRayRuntimeClient(
        actor=object(),
        actor_name="openpi-shared-actor",
        spec=_spec(),
        session_id="model-a",
        ready_timeout_s=30.0,
    )
    fut: concurrent.futures.Future[dict[str, object]] = concurrent.futures.Future()
    fut.set_result({"ok": True})
    ref = SimpleNamespace(future=lambda: fut)

    monkeypatch.setattr(
        "tinker_server.backend.openpi_shared_ray_runtime.ray.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ray.get should not be called")),
    )

    assert asyncio.run(client._ray_get(ref, timeout_s=1.0)) == {"ok": True}


def test_start_openpi_shared_ray_runtime_applies_single_node_pin(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {}
    node_id = "a" * 56

    monkeypatch.setenv("PFS_TINKER_PATH", "/repo")

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            _ = actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor

        async def ready(self):
            return {"actor_id": "actor-123", "node_id": node_id, "node_ip": "192.168.38.176"}

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"] = actor_name

    openpi_shared_ray_runtime.clear_openpi_shared_runtime_pool()
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_shared_ray_runtime, "_openpi_runtime_env_vars", lambda: {"PYTHONPATH": "/runtime/site-packages:/repo:/hf"})
    monkeypatch.setattr(
        openpi_shared_ray_runtime,
        "parse_model_node_ip_list",
        lambda **_kwargs: ["192.168.38.176"],
    )
    capacity_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        openpi_shared_ray_runtime,
        "assert_node_ip_capacity",
        lambda **kwargs: capacity_calls.append(kwargs),
    )
    monkeypatch.setattr(
        openpi_shared_ray_runtime.ray,
        "nodes",
        lambda: [{"Alive": True, "NodeManagerAddress": "192.168.38.176", "NodeID": node_id}],
    )

    session = _make_session("model-a", "session-a")
    asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    options = state["options"]
    assert options["resources"] == {"node:192.168.38.176": 0.001}
    assert options["scheduling_strategy"].node_id == node_id
    assert options["scheduling_strategy"].soft is False
    assert capacity_calls == [
        {
            "required_gpus_by_node_ip": {"192.168.38.176": 1},
            "context": "[OpenPISharedRuntime] node pinning model='openpi/pi0-fast-libero-low-mem-finetune' actor="
            f"{state['register']['actor_name']!r}",
        }
    ]


def test_shared_client_close_cleans_up_new_actor_after_failed_initial_create_session(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {
        "unregister": [],
        "kill_calls": [],
        "shutdown_refs": [],
        "inflight": [],
        "sessions": [],
        "touch": [],
    }

    class _FakeActorHandle:
        class ready_metadata:
            @staticmethod
            def remote():
                return _completed_ref({"actor_id": "actor-123", "current_session_id": None})

        class register_session:
            @staticmethod
            def remote(*_args, **_kwargs):
                return _failed_ref(RuntimeError("create failed"))

        class shutdown:
            @staticmethod
            def remote():
                return "shutdown-ref"

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            actor = _FakeActorHandle()
            state["actor"] = actor
            return actor

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"].append(actor_name)

        def mark_inflight(self, actor_name, delta):
            state["inflight"].append((actor_name, delta))

        def set_session(self, actor_name, session_id):
            state["sessions"].append((actor_name, session_id))

        def unregister(self, actor_name):
            state["unregister"].append(actor_name)

    def _raise_missing_actor(*_args, **_kwargs):
        raise ValueError("actor not found")

    def _fake_ray_get(ref, timeout=None):
        if ref == "shutdown-ref":
            state["shutdown_refs"].append((ref, timeout))
            return None
        raise AssertionError(f"unexpected ref {ref!r}")

    def _fake_ray_kill(actor, *, no_restart=True):
        state["kill_calls"].append((actor, no_restart))

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", _raise_missing_actor)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get", _fake_ray_get)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "kill", _fake_ray_kill)

    session = _make_session("model-a", "session-a")
    client = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    with pytest.raises(RuntimeError, match="create failed"):
        asyncio.run(client.request("create_session", _create_payload(session)))

    asyncio.run(client.close())

    actor_name = state["register"]["actor_name"]
    assert actor_name in state["unregister"]
    assert state["shutdown_refs"] == [("shutdown-ref", 5.0)]
    assert state["kill_calls"] == [(state["actor"], True)]
    assert openpi_shared_ray_runtime._SHARED_ACTORS == {}


def test_shared_client_close_does_not_kill_actor_after_successful_create_session(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {
        "unregister": [],
        "kill_calls": [],
        "shutdown_refs": [],
        "inflight": [],
        "sessions": [],
        "touch": [],
    }

    class _FakeActorHandle:
        class ready_metadata:
            @staticmethod
            def remote():
                return _completed_ref({"actor_id": "actor-123", "current_session_id": None})

        class register_session:
            @staticmethod
            def remote(*_args, **_kwargs):
                return _completed_ref(
                    {"backend": "openpi_fast", "config_name": "pi0_fast_libero_low_mem_finetune"}
                )

        class shutdown:
            @staticmethod
            def remote():
                return "shutdown-ref"

    class _FakeActorBuilder:
        def options(self, **kwargs):
            _ = kwargs
            return self

        def remote(self, **kwargs):
            _ = kwargs
            actor = _FakeActorHandle()
            state["actor"] = actor
            return actor

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"].append(actor_name)

        def mark_inflight(self, actor_name, delta):
            state["inflight"].append((actor_name, delta))

        def set_session(self, actor_name, session_id):
            state["sessions"].append((actor_name, session_id))

        def unregister(self, actor_name):
            state["unregister"].append(actor_name)

    def _raise_missing_actor(*_args, **_kwargs):
        raise ValueError("actor not found")

    def _fake_ray_get(ref, timeout=None):
        if ref == "shutdown-ref":
            state["shutdown_refs"].append((ref, timeout))
            return None
        raise AssertionError(f"unexpected ref {ref!r}")

    def _fake_ray_kill(actor, *, no_restart=True):
        state["kill_calls"].append((actor, no_restart))

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", _raise_missing_actor)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get", _fake_ray_get)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "kill", _fake_ray_kill)

    session = _make_session("model-a", "session-a")
    client = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    result = asyncio.run(client.request("create_session", _create_payload(session)))
    assert result["backend"] == "openpi_fast"

    asyncio.run(client.close())

    assert state["unregister"] == []
    assert state["shutdown_refs"] == []
    assert state["kill_calls"] == []


def test_shared_client_shutdown_reclaims_actor_when_last_session_exits(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {
        "unregister": [],
        "kill_calls": [],
        "shutdown_refs": [],
        "inflight": [],
        "sessions": [],
        "touch": [],
    }

    class _FakeActorHandle:
        class ready_metadata:
            @staticmethod
            def remote():
                return _completed_ref({"actor_id": "actor-123", "current_session_id": None})

        class register_session:
            @staticmethod
            def remote(*_args, **_kwargs):
                return _completed_ref(
                    {"backend": "openpi_fast", "config_name": "pi0_fast_libero_low_mem_finetune"}
                )

        class request_for_session:
            @staticmethod
            def remote(*_args, **_kwargs):
                return _completed_ref({"stopped": True, "known_session_ids": []})

        class shutdown:
            @staticmethod
            def remote():
                return "shutdown-ref"

    class _FakeActorBuilder:
        def options(self, **kwargs):
            _ = kwargs
            return self

        def remote(self, **kwargs):
            _ = kwargs
            actor = _FakeActorHandle()
            state["actor"] = actor
            return actor

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"].append(actor_name)

        def mark_inflight(self, actor_name, delta):
            state["inflight"].append((actor_name, delta))

        def set_session(self, actor_name, session_id):
            state["sessions"].append((actor_name, session_id))

        def unregister(self, actor_name):
            state["unregister"].append(actor_name)

    def _raise_missing_actor(*_args, **_kwargs):
        raise ValueError("actor not found")

    def _fake_ray_get(ref, timeout=None):
        if ref == "shutdown-ref":
            state["shutdown_refs"].append((ref, timeout))
            return None
        raise AssertionError(f"unexpected ref {ref!r}")

    def _fake_ray_kill(actor, *, no_restart=True):
        state["kill_calls"].append((actor, no_restart))

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", _raise_missing_actor)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get", _fake_ray_get)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "kill", _fake_ray_kill)

    session = _make_session("model-a", "session-a")
    client = asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    asyncio.run(client.request("create_session", _create_payload(session)))
    result = asyncio.run(client.request("shutdown", {"model_id": session.model_id}))

    actor_name = state["register"]["actor_name"]
    assert result["stopped"] is True
    assert state["sessions"][-1] == (actor_name, None)
    assert actor_name in state["unregister"]
    assert state["shutdown_refs"] == [("shutdown-ref", 5.0)]
    assert state["kill_calls"] == [(state["actor"], True)]
    assert openpi_shared_ray_runtime._SHARED_ACTORS == {}


def test_start_openpi_shared_ray_runtime_uses_model_id_as_runtime_session_key(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {"client_inits": []}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            _ = kwargs
            return self

        def remote(self, **kwargs):
            _ = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            _ = actor, actor_name, spec, ready_timeout_s, owns_started_actor
            state["client_inits"].append(session_id)

        async def ready(self):
            return {"actor_id": "actor-123"}

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            _ = kwargs
            return None

        def mark_ready(self, actor_name):
            _ = actor_name

        def touch(self, actor_name):
            _ = actor_name

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())

    session_a = _make_session("model-a", "shared-http-session")
    session_b = _make_session("model-b", "shared-http-session")

    asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session_a,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )
    asyncio.run(
        openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
            session=session_b,
            spec=_spec(),
            config_name="pi0_fast_libero_low_mem_finetune",
            model_config=_model_config(),
        )
    )

    assert state["client_inits"] == ["model-a", "model-b"]


def test_start_openpi_shared_ray_runtime_cleans_up_detached_actor_when_ready_fails(monkeypatch) -> None:
    from tinker_server.backend import openpi_shared_ray_runtime

    state: dict[str, object] = {"shutdown_refs": [], "kill_calls": [], "unregister": []}

    class _FakeActorHandle:
        def __init__(self) -> None:
            self.shutdown = SimpleNamespace(remote=lambda: "shutdown-ref")

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            actor = _FakeActorHandle()
            state["actor"] = actor
            return actor

    class _FailingClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s, owns_started_actor=False):
            _ = actor, spec, session_id, ready_timeout_s, owns_started_actor
            state["actor_name"] = actor_name

        async def ready(self):
            raise RuntimeError("ready failed")

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs
            return None

        def unregister(self, actor_name):
            state["unregister"].append(actor_name)

    def _raise_missing_actor(*_args, **_kwargs):
        raise ValueError("actor not found")

    def _fake_ray_get(ref, timeout=None):
        state["shutdown_refs"].append((ref, timeout))
        return None

    def _fake_ray_kill(actor, *, no_restart=True):
        state["kill_calls"].append((actor, no_restart))

    _reset_shared_runtime_test_state(monkeypatch, openpi_shared_ray_runtime)
    monkeypatch.setattr(openpi_shared_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_shared_ray_runtime, "OpenPISharedRayRuntimeClient", _FailingClient)
    monkeypatch.setattr(openpi_shared_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get_actor", _raise_missing_actor)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "get", _fake_ray_get)
    monkeypatch.setattr(openpi_shared_ray_runtime.ray, "kill", _fake_ray_kill)

    session = _make_session("model-a", "session-a")
    with pytest.raises(RuntimeError, match="ready failed"):
        asyncio.run(
            openpi_shared_ray_runtime.start_openpi_shared_ray_runtime(
                session=session,
                spec=_spec(),
                config_name="pi0_fast_libero_low_mem_finetune",
                model_config=_model_config(),
            )
        )

    actor_name = state["actor_name"]
    assert actor_name in state["unregister"]
    assert state["shutdown_refs"] == [("shutdown-ref", 5.0)]
    assert state["kill_calls"] == [(state["actor"], True)]
    assert openpi_shared_ray_runtime._SHARED_ACTORS == {}
    assert "register" not in state


def test_openpi_shared_runtime_core_swaps_sessions_on_a_b_a() -> None:
    from tinker_server.backend.openpi_shared_ray_runtime import (
        OpenPISharedRuntimeCore,
        _template_session_id,
    )

    class _FakeWorkerRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object] | None]] = []
            self._saved_sessions: set[str] = set()

        async def request(self, op: str, payload: dict[str, object] | None = None, *, timeout_s=None) -> dict:
            _ = timeout_s
            self.calls.append((op, payload))
            if op == "create_session":
                return {"backend": "openpi_fast", "config_name": payload["config_name"]}
            if op == "save_session_state":
                self._saved_sessions.add(str(payload["session_id"]))
                return {"path": f"/tmp/{payload['session_id']}"}
            if op == "load_session_state":
                session_id = str(payload["session_id"])
                if session_id not in self._saved_sessions:
                    raise FileNotFoundError(session_id)
                return {"current_step": 0, "learning_rate": 1e-4}
            if op == "forward_backward":
                return {"loss_fn_output_type": "cross_entropy_loss", "loss_fn_outputs": [], "metrics": {}}
            raise AssertionError(f"unexpected op {op}")

    runtime = _FakeWorkerRuntime()
    session_a = _make_session("model-a", "session-a")
    session_b = _make_session("model-b", "session-b")

    core = OpenPISharedRuntimeCore(
        spec=_spec(),
        runtime_factory=lambda spec: runtime,
        actor_metadata={"actor_name": "openpi_shared_runtime_fast"},
    )
    template_session_id = _template_session_id({"actor_name": "openpi_shared_runtime_fast"})

    asyncio.run(core.register_session(session_a.session_id, _create_payload(session_a)))
    asyncio.run(core.request_for_session(session_a.session_id, "forward_backward", {"batch": []}))
    asyncio.run(core.register_session(session_b.session_id, _create_payload(session_b, learning_rate=5e-4)))
    asyncio.run(core.request_for_session(session_b.session_id, "forward_backward", {"batch": []}))
    asyncio.run(core.request_for_session(session_a.session_id, "forward_backward", {"batch": []}))

    assert runtime.calls == [
        ("create_session", _create_payload(session_a)),
        ("save_session_state", {"session_id": template_session_id}),
        ("save_session_state", {"session_id": "session-a"}),
        ("forward_backward", {"batch": []}),
        ("save_session_state", {"session_id": "session-a"}),
        ("load_session_state", {"session_id": template_session_id}),
        ("forward_backward", {"batch": []}),
        ("save_session_state", {"session_id": "session-b"}),
        ("load_session_state", {"session_id": "session-a"}),
        ("forward_backward", {"batch": []}),
    ]


def test_openpi_shared_runtime_core_bootstraps_distinct_sessions_when_template_not_reusable() -> None:
    from tinker_server.backend.openpi_shared_ray_runtime import OpenPISharedRuntimeCore, _template_session_id

    class _FakeWorkerRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object] | None]] = []
            self._saved_sessions: set[str] = set()

        async def request(self, op: str, payload: dict[str, object] | None = None, *, timeout_s=None) -> dict:
            _ = timeout_s
            self.calls.append((op, payload))
            if op == "create_session":
                return {"backend": "openpi_fast", "config_name": payload["config_name"]}
            if op == "save_session_state":
                self._saved_sessions.add(str(payload["session_id"]))
                return {"path": f"/tmp/{payload['session_id']}"}
            if op == "load_session_state":
                session_id = str(payload["session_id"])
                if session_id not in self._saved_sessions:
                    raise FileNotFoundError(session_id)
                return {"current_step": 0, "learning_rate": 1e-4}
            if op == "forward_backward":
                return {"loss_fn_output_type": "cross_entropy_loss", "loss_fn_outputs": [], "metrics": {}}
            raise AssertionError(f"unexpected op {op}")

    runtime = _FakeWorkerRuntime()
    session_a = _make_session("model-a", "session-a")
    session_b = _make_session("model-b", "session-b")

    actor_metadata = {"actor_name": "openpi_shared_runtime_action_fast"}
    core = OpenPISharedRuntimeCore(
        spec=_spec(worker_module="tinker_server.backend.openpi_fast_action_worker"),
        runtime_factory=lambda spec: runtime,
        actor_metadata=actor_metadata,
        template_reusable=False,
    )
    template_session_id = _template_session_id(actor_metadata)

    asyncio.run(core.register_session(session_a.session_id, _create_payload(session_a)))
    asyncio.run(core.request_for_session(session_a.session_id, "forward_backward", {"batch": []}))
    asyncio.run(core.register_session(session_b.session_id, _create_payload(session_b, learning_rate=5e-4)))
    asyncio.run(core.request_for_session(session_b.session_id, "forward_backward", {"batch": []}))
    asyncio.run(core.request_for_session(session_a.session_id, "forward_backward", {"batch": []}))

    assert runtime.calls == [
        ("create_session", _create_payload(session_a)),
        ("save_session_state", {"session_id": template_session_id}),
        ("save_session_state", {"session_id": "session-a"}),
        ("forward_backward", {"batch": []}),
        ("save_session_state", {"session_id": "session-a"}),
        ("create_session", _create_payload(session_b, learning_rate=5e-4)),
        ("save_session_state", {"session_id": "session-b"}),
        ("forward_backward", {"batch": []}),
        ("save_session_state", {"session_id": "session-b"}),
        ("load_session_state", {"session_id": "session-a"}),
        ("forward_backward", {"batch": []}),
    ]


def test_openpi_shared_runtime_core_surfaces_restore_failures_without_fallback() -> None:
    from tinker_server.backend.openpi_shared_ray_runtime import OpenPISharedRuntimeCore

    class _FakeWorkerRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object] | None]] = []
            self.create_calls = 0

        async def request(self, op: str, payload: dict[str, object] | None = None, *, timeout_s=None) -> dict:
            _ = timeout_s
            self.calls.append((op, payload))
            if op == "create_session":
                self.create_calls += 1
                return {"backend": "openpi_fast", "config_name": payload["config_name"]}
            if op == "save_session_state":
                return {"path": f"/tmp/{payload['session_id']}"}
            if op == "load_session_state":
                raise RuntimeError("restore failed")
            if op == "forward_backward":
                return {"loss_fn_output_type": "cross_entropy_loss", "loss_fn_outputs": [], "metrics": {}}
            raise AssertionError(f"unexpected op {op}")

    runtime = _FakeWorkerRuntime()
    session_a = _make_session("model-a", "session-a")
    session_b = _make_session("model-b", "session-b")

    core = OpenPISharedRuntimeCore(
        spec=_spec(),
        runtime_factory=lambda spec: runtime,
        actor_metadata={"actor_id": "actor-1", "node_ip": "127.0.0.1"},
    )

    asyncio.run(core.register_session(session_a.session_id, _create_payload(session_a)))
    asyncio.run(core.register_session(session_b.session_id, _create_payload(session_b)))

    with pytest.raises(RuntimeError, match="restore failed"):
        asyncio.run(core.request_for_session(session_b.session_id, "forward_backward", {"batch": []}))

    assert runtime.create_calls == 1


def test_openpi_shared_runtime_core_isolates_template_state_per_actor(tmp_path) -> None:
    from tinker_server.backend.openpi_session_state import OpenPISessionStateManager
    from tinker_server.backend.openpi_shared_ray_runtime import OpenPISharedRuntimeCore

    class _StateBackedRuntime:
        def __init__(self, *, worker_module: str, state_root) -> None:
            self._worker_module = worker_module
            self._state_store = OpenPISessionStateManager(state_root)
            self._runtime_signature = {"worker_module": worker_module}
            self._current_payload: dict[str, object] | None = None

        async def request(self, op: str, payload: dict[str, object] | None = None, *, timeout_s=None) -> dict:
            _ = timeout_s
            if op == "create_session":
                self._current_payload = dict(payload or {})
                return {
                    "backend": "openpi_shared",
                    "config_name": self._current_payload["config_name"],
                }
            if op == "save_session_state":
                session_id = str((payload or {})["session_id"])
                current_payload = dict(self._current_payload or {})

                def _save_train_state(path, state) -> None:
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "state.json").write_text(json.dumps(state), encoding="utf-8")

                self._state_store.save_state(
                    session_id,
                    worker_module=self._worker_module,
                    runtime_signature=self._runtime_signature,
                    state=current_payload,
                    rng={"worker_module": self._worker_module},
                    pending_grads=None,
                    learning_rate=1e-4,
                    current_step=0,
                    save_train_state_fn=_save_train_state,
                )
                return {"path": str(self._state_store.get_session_path(session_id))}
            if op == "load_session_state":
                session_id = str((payload or {})["session_id"])

                def _load_train_state(path):
                    return json.loads((path / "state.json").read_text(encoding="utf-8"))

                loaded = self._state_store.load_state(
                    session_id,
                    expected_worker_module=self._worker_module,
                    expected_runtime_signature=self._runtime_signature,
                    load_train_state_fn=_load_train_state,
                )
                self._current_payload = dict(loaded["state"])
                return {"current_step": loaded["current_step"], "learning_rate": loaded["learning_rate"]}
            if op == "forward_backward":
                return {"loss_fn_output_type": "cross_entropy_loss", "loss_fn_outputs": [], "metrics": {}}
            raise AssertionError(f"unexpected op {op}")

    state_root = tmp_path / "_mint_session_state"
    fast_runtime = _StateBackedRuntime(
        worker_module="tinker_server.backend.openpi_fast_worker",
        state_root=state_root,
    )
    pi05_runtime = _StateBackedRuntime(
        worker_module="tinker_server.backend.openpi_pi05_worker",
        state_root=state_root,
    )

    fast_core = OpenPISharedRuntimeCore(
        spec=_spec(),
        runtime_factory=lambda spec: fast_runtime,
        actor_metadata={"actor_name": "openpi_shared_runtime_fast"},
    )
    pi05_core = OpenPISharedRuntimeCore(
        spec=_spec(worker_module="tinker_server.backend.openpi_pi05_worker"),
        runtime_factory=lambda spec: pi05_runtime,
        actor_metadata={"actor_name": "openpi_shared_runtime_pi05"},
    )

    fast_session_a = _make_session("fast-model-a", "fast-session-a")
    fast_session_b = _make_session("fast-model-b", "fast-session-b")
    pi05_session = _make_session("pi05-model-a", "pi05-session-a")

    asyncio.run(fast_core.register_session(fast_session_a.session_id, _create_payload(fast_session_a)))
    asyncio.run(pi05_core.register_session(pi05_session.session_id, _create_payload(pi05_session)))
    asyncio.run(fast_core.register_session(fast_session_b.session_id, _create_payload(fast_session_b)))

    result = asyncio.run(
        fast_core.request_for_session(fast_session_b.session_id, "forward_backward", {"batch": []})
    )

    assert result["loss_fn_output_type"] == "cross_entropy_loss"


def test_openpi_shared_runtime_core_resets_after_initial_create_session_failure() -> None:
    from tinker_server.backend.openpi_shared_ray_runtime import OpenPISharedRuntimeCore

    class _FailingRuntime:
        def __init__(self, *, should_fail: bool) -> None:
            self.should_fail = should_fail
            self.close_calls = 0

        async def request(self, op: str, payload: dict[str, object] | None = None, *, timeout_s=None) -> dict:
            _ = payload, timeout_s
            if op == "create_session" and self.should_fail:
                raise RuntimeError("create failed")
            if op == "create_session":
                return {"backend": "openpi_fast", "config_name": "pi0_fast_libero_low_mem_finetune"}
            if op == "save_session_state":
                return {"path": "/tmp/state"}
            raise AssertionError(f"unexpected op {op}")

        async def close(self) -> None:
            self.close_calls += 1

    runtimes = [_FailingRuntime(should_fail=True), _FailingRuntime(should_fail=False)]
    runtime_factory_calls = 0

    def _runtime_factory(spec):
        nonlocal runtime_factory_calls
        _ = spec
        runtime = runtimes[runtime_factory_calls]
        runtime_factory_calls += 1
        return runtime

    session = _make_session("model-a", "session-a")
    core = OpenPISharedRuntimeCore(
        spec=_spec(),
        runtime_factory=_runtime_factory,
        actor_metadata={"actor_id": "actor-1", "node_ip": "127.0.0.1"},
    )

    with pytest.raises(RuntimeError, match="create failed"):
        asyncio.run(core.register_session(session.session_id, _create_payload(session)))

    result = asyncio.run(core.register_session(session.session_id, _create_payload(session)))

    assert result == {"backend": "openpi_fast", "config_name": "pi0_fast_libero_low_mem_finetune"}
    assert runtime_factory_calls == 2
    assert runtimes[0].close_calls == 1
