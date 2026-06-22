from __future__ import annotations

import asyncio
import sys
import types

import pytest

from mint_server.backend.openpi.openpi_direct_runtime import OpenPIDirectWorkerClient
from mint_server.backend.openpi.openpi_fast_runtime import (
    OpenPIFastRuntimeSpec,
    OpenPIFastWorkerProtocolError,
    OpenPIFastWorkerRemoteError,
)
from mint_server.backend.openpi.openpi_shared_ray_runtime import OpenPISharedRuntimeCore


def _spec(worker_module: str) -> OpenPIFastRuntimeSpec:
    return OpenPIFastRuntimeSpec(
        worker_module=worker_module,
        startup_timeout_s=1.0,
        create_session_timeout_s=1.0,
        request_timeout_s=1.0,
        save_weights_timeout_s=1.0,
        load_weights_timeout_s=1.0,
    )


def test_direct_runtime_calls_training_session_in_actor_process(monkeypatch) -> None:
    module_name = "tests.fake_openpi_training_worker"
    calls: list[tuple[str, object]] = []
    module = types.ModuleType(module_name)

    class OpenPIFastWorkerSession:
        def __init__(self, payload):
            calls.append(("init", dict(payload)))
            self.payload = dict(payload)

        def create_session(self):
            calls.append(("create_session", self.payload["model_id"]))
            return {"ready": True, "model_id": self.payload["model_id"]}

        def forward_backward(self, payload):
            calls.append(("forward_backward", dict(payload)))
            return {"loss": 1.25}

        def shutdown(self):
            calls.append(("shutdown", self.payload["model_id"]))
            return {"stopped": True}

    def _dispatch(session, op, payload):
        if session is None:
            raise RuntimeError("missing session")
        if op == "forward_backward":
            return session.forward_backward(payload), False
        if op == "shutdown":
            return session.shutdown(), True
        raise ValueError(op)

    module.OpenPIFastWorkerSession = OpenPIFastWorkerSession
    module._dispatch = _dispatch
    monkeypatch.setitem(sys.modules, module_name, module)

    client = asyncio.run(OpenPIDirectWorkerClient.start(_spec(module_name)))

    assert asyncio.run(client.request("create_session", {"model_id": "model-a"})) == {
        "ready": True,
        "model_id": "model-a",
    }
    assert asyncio.run(client.request("forward_backward", {"batch": 3})) == {"loss": 1.25}
    assert asyncio.run(client.request("shutdown", {})) == {"stopped": True}
    with pytest.raises(OpenPIFastWorkerProtocolError, match="closed"):
        asyncio.run(client.request("forward_backward", {}))
    assert calls == [
        ("init", {"model_id": "model-a"}),
        ("create_session", "model-a"),
        ("forward_backward", {"batch": 3}),
        ("shutdown", "model-a"),
    ]


def test_direct_runtime_calls_action_dispatch_create_session(monkeypatch) -> None:
    module_name = "tests.fake_openpi_action_worker"
    calls: list[tuple[str, object]] = []
    module = types.ModuleType(module_name)

    class OpenPIFastActionSession:
        def __init__(self, payload):
            calls.append(("init", dict(payload)))
            self.payload = dict(payload)

        def act(self, payload):
            calls.append(("act", dict(payload)))
            return {"actions": [1, 2, 3]}

        def shutdown(self):
            calls.append(("shutdown", self.payload["action_session_id"]))
            return {"stopped": True}

    def _dispatch(session, op, payload):
        if op == "create_session":
            if session is not None:
                session.shutdown()
            return {"ready": True}, OpenPIFastActionSession(payload)
        if session is None:
            raise RuntimeError("missing action session")
        if op == "act":
            return session.act(payload), session
        if op == "shutdown":
            return session.shutdown(), None
        raise ValueError(op)

    module.OpenPIFastActionSession = OpenPIFastActionSession
    module._dispatch = _dispatch
    monkeypatch.setitem(sys.modules, module_name, module)

    client = asyncio.run(OpenPIDirectWorkerClient.start(_spec(module_name)))

    assert asyncio.run(
        client.request("create_session", {"action_session_id": "session-a:action:1"})
    ) == {"ready": True}
    assert asyncio.run(client.request("act", {"observation": {}})) == {"actions": [1, 2, 3]}
    assert asyncio.run(client.request("shutdown", {})) == {"stopped": True}
    assert calls == [
        ("init", {"action_session_id": "session-a:action:1"}),
        ("act", {"observation": {}}),
        ("shutdown", "session-a:action:1"),
    ]


def test_direct_runtime_wraps_worker_exceptions(monkeypatch) -> None:
    module_name = "tests.fake_openpi_failing_worker"
    module = types.ModuleType(module_name)

    class OpenPIFastWorkerSession:
        def __init__(self, payload):
            _ = payload

        def create_session(self):
            return {"ready": True}

    def _dispatch(session, op, payload):
        _ = session, op, payload
        raise ValueError("worker exploded")

    module.OpenPIFastWorkerSession = OpenPIFastWorkerSession
    module._dispatch = _dispatch
    monkeypatch.setitem(sys.modules, module_name, module)

    client = asyncio.run(OpenPIDirectWorkerClient.start(_spec(module_name)))
    asyncio.run(client.request("create_session", {}))

    with pytest.raises(OpenPIFastWorkerRemoteError, match="worker exploded"):
        asyncio.run(client.request("forward_backward", {}))


def test_shared_runtime_core_defaults_to_direct_worker_client(monkeypatch) -> None:
    module_name = "tests.fake_openpi_shared_worker"
    calls: list[str] = []
    module = types.ModuleType(module_name)

    class OpenPIFastWorkerSession:
        def __init__(self, payload):
            calls.append(f"init:{payload['model_id']}")

        def create_session(self):
            calls.append("create")
            return {"ready": True}

        def save_session_state(self, payload):
            calls.append(f"save:{payload['session_id']}")
            return {"path": payload["session_id"]}

        def load_session_state(self, payload):
            calls.append(f"load:{payload['session_id']}")
            return {"path": payload["session_id"]}

    def _dispatch(session, op, payload):
        if op == "save_session_state":
            return session.save_session_state(payload), False
        if op == "load_session_state":
            return session.load_session_state(payload), False
        if op == "shutdown":
            return {"stopped": True}, True
        raise ValueError(op)

    module.OpenPIFastWorkerSession = OpenPIFastWorkerSession
    module._dispatch = _dispatch
    monkeypatch.setitem(sys.modules, module_name, module)

    core = OpenPISharedRuntimeCore(
        spec=_spec(module_name),
        actor_metadata={"actor_name": "mint_openpi_shared_test"},
    )

    assert asyncio.run(
        core.register_session("model-a", {"model_id": "model-a"})
    ) == {"ready": True}
    assert calls[:2] == ["init:model-a", "create"]
    assert calls[2].startswith("save:__mint_initial__:")
    assert calls[3] == "save:model-a"
