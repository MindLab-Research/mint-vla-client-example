from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.models.types import EncodedTextChunk, ImageChunk, ModelInput, TensorData


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _make_session() -> TrainingSession:
    return TrainingSession(
        model_id="model-1",
        session_id="session-1",
        model_seq_id=0,
        base_model=OPENPI_FAST_MODEL,
    )


def _make_observation() -> ModelInput:
    return ModelInput(
        chunks=[
            ImageChunk(data=b"img-0", format="png", expected_tokens=256),
            ImageChunk(data=b"img-1", format="png", expected_tokens=256),
            ImageChunk(data=b"img-2", format="png", expected_tokens=256),
            EncodedTextChunk(tokens=[11, 12, 13]),
        ]
    )


class _FakeTrainingRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"session": "created"}
        if op == "save_weights":
            save_path = Path(payload["save_path"])
            (save_path / "1" / "params").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "assets").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "train_state").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "params" / "_METADATA").write_text("params", encoding="utf-8")
            (save_path / "1" / "assets" / "asset.json").write_text("{}", encoding="utf-8")
            (save_path / "1" / "train_state" / "_METADATA").write_text("train", encoding="utf-8")
            return {"path": str(save_path)}
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected training op {op}")

    async def close(self) -> None:
        return None


class _FakeTrainingRuntimeFactory:
    def __init__(self) -> None:
        self.clients: list[_FakeTrainingRuntimeClient] = []

    async def __call__(self, *, session: TrainingSession, model_config, config_name: str):
        _ = session, model_config, config_name
        client = _FakeTrainingRuntimeClient()
        self.clients.append(client)
        return client


class _FakeActionRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"ready": True}
        if op == "act":
            return {
                "actions": {
                    "data": [0.0] * 28,
                    "shape": [4, 7],
                    "dtype": "float32",
                },
                "policy_timing": {"infer_ms": 4.0},
            }
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected action op {op}")

    async def close(self) -> None:
        return None


class _FakeActionRuntimeFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.clients: list[_FakeActionRuntimeClient] = []

    async def __call__(self, *, action_session_id: str, base_model: str, checkpoint_path: str, model_config, config_name: str):
        self.calls.append(
            {
                "action_session_id": action_session_id,
                "base_model": base_model,
                "checkpoint_path": checkpoint_path,
                "config_name": config_name,
                "camera_layout": model_config.camera_layout,
            }
        )
        client = _FakeActionRuntimeClient()
        self.clients.append(client)
        return client


def test_openpi_fast_default_runtime_factory_uses_action_ray_runtime(
    monkeypatch,
    configure_runtime_env,
) -> None:
    from tinker_server.backend.action_session_manager import _default_runtime_factory

    runtime_env = configure_runtime_env()
    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_action_ray_runtime(*, action_session_id: str, base_model: str, spec):
        calls.append(
            {
                "action_session_id": action_session_id,
                "base_model": base_model,
                "worker_module": spec.worker_module,
                "python_executable": spec.python_executable,
                "pythonpath": spec.pythonpath,
            }
        )
        return "action-ray-runtime-client"

    async def _unexpected_local_fast_start(spec=None):
        raise AssertionError(f"local fast action worker path must not run: {spec}")

    async def _unexpected_local_worker_start(spec):
        raise AssertionError(f"local worker path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.action_session_manager.start_openpi_action_ray_runtime",
        _fake_start_openpi_action_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_action_runtime.OpenPIFastActionWorkerClient.start",
        _unexpected_local_fast_start,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_worker_start,
    )

    runtime = asyncio.run(
        _default_runtime_factory(
            action_session_id="session-1:action:3",
            base_model=OPENPI_FAST_MODEL,
            checkpoint_path="/tmp/export-1",
            model_config=SimpleNamespace(action_dim=7, action_horizon=10, max_model_len=180),
            config_name="pi0_fast_libero_low_mem_finetune",
        )
    )

    assert runtime == "action-ray-runtime-client"
    assert calls == [
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "worker_module": "tinker_server.backend.openpi_fast_action_worker",
            "python_executable": str(runtime_env["layout"].host_python),
            "pythonpath": runtime_env["pythonpath"],
        }
    ]


def test_openpi_pi05_default_runtime_factory_uses_action_ray_runtime(
    monkeypatch,
    configure_runtime_env,
) -> None:
    from tinker_server.backend.action_session_manager import _default_pi05_runtime_factory

    runtime_env = configure_runtime_env()
    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_action_ray_runtime(*, action_session_id: str, base_model: str, spec):
        calls.append(
            {
                "action_session_id": action_session_id,
                "base_model": base_model,
                "worker_module": spec.worker_module,
                "python_executable": spec.python_executable,
                "pythonpath": spec.pythonpath,
            }
        )
        return "action-ray-runtime-client"

    async def _unexpected_local_fast_start(spec=None):
        raise AssertionError(f"local pi0.5 action worker path must not run: {spec}")

    async def _unexpected_local_worker_start(spec):
        raise AssertionError(f"local worker path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.action_session_manager.start_openpi_action_ray_runtime",
        _fake_start_openpi_action_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_action_runtime.OpenPIFastActionWorkerClient.start",
        _unexpected_local_fast_start,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_worker_start,
    )

    runtime = asyncio.run(
        _default_pi05_runtime_factory(
            action_session_id="session-1:action:9",
            base_model="openpi/pi05-libero-low-mem-finetune",
            checkpoint_path="/tmp/export-9",
            model_config=SimpleNamespace(action_dim=7, action_horizon=10, max_model_len=180),
            config_name="pi05_libero",
        )
    )

    assert runtime == "action-ray-runtime-client"
    assert calls == [
        {
            "action_session_id": "session-1:action:9",
            "base_model": "openpi/pi05-libero-low-mem-finetune",
            "worker_module": "tinker_server.backend.openpi_pi05_action_worker",
            "python_executable": str(runtime_env["layout"].host_python),
            "pythonpath": runtime_env["pythonpath"],
        }
    ]


def test_start_openpi_action_ray_runtime_registers_actor_metadata_in_resource_pool(monkeypatch) -> None:
    from tinker_server.backend import openpi_action_ray_runtime
    from tinker_server.backend.openpi_fast_action_runtime import OpenPIFastActionRuntimeSpec

    state: dict[str, object] = {}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, action_session_id, ready_timeout_s):
            state["client_init"] = {
                "actor": actor,
                "actor_name": actor_name,
                "action_session_id": action_session_id,
                "ready_timeout_s": ready_timeout_s,
                "worker_module": spec.worker_module,
            }

        async def ready(self):
            return {
                "actor_id": "actor-123",
                "node_id": "node-456",
                "node_ip": "192.168.0.8",
                "pid": 999,
                "cuda_visible_devices": "0",
                "action_session_id": "session-1:action:3",
            }

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"] = actor_name

    monkeypatch.setattr(openpi_action_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_action_ray_runtime, "get_resource_pool", lambda: _FakePool())

    client = asyncio.run(
        openpi_action_ray_runtime.start_openpi_action_ray_runtime(
            action_session_id="session-1:action:3",
            base_model=OPENPI_FAST_MODEL,
            spec=OpenPIFastActionRuntimeSpec(
                python_executable="/tmp/runtime/host-venv/bin/python",
                worker_module="tinker_server.backend.openpi_fast_action_worker",
                pythonpath=("/tmp/runtime/site-packages", "/tmp/runtime/src/openpi/src"),
            ),
        )
    )

    assert isinstance(client, _FakeClient)
    register = state["register"]
    assert register["actor_type"].value == "openpi"
    assert register["base_model"] == OPENPI_FAST_MODEL
    assert register["session_id"] == "session-1:action:3"
    assert register["node_id"] == "node-456"
    assert register["metadata"]["worker_module"] == "tinker_server.backend.openpi_fast_action_worker"
    assert register["metadata"]["actor_id"] == "actor-123"
    assert register["metadata"]["node_ip"] == "192.168.0.8"
    assert register["metadata"]["pid"] == 999
    assert register["metadata"]["cuda_visible_devices"] == "0"


def test_openpi_fast_save_weights_for_sampler_exports_policy_loadable_checkpoint(tmp_path: Path) -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeTrainingRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    export_path = asyncio.run(
        engine.save_weights_for_sampler(
            session=session,
            checkpoint_name="export-1",
            checkpoint_base_dir=str(tmp_path),
            use_per_expert_lora=False,
        )
    )

    export_dir = Path(export_path)
    assert export_dir == tmp_path / "model-1" / "export-1"
    assert (export_dir / "params" / "_METADATA").exists()
    assert (export_dir / "assets" / "asset.json").exists()
    assert not (export_dir / "train_state").exists()


def test_action_session_manager_create_session_starts_runtime_from_checkpoint(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIFastActionSessionManager(runtime_factory=factory)

    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=3,
            base_model=OPENPI_FAST_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    assert action_session_id == "session-1:action:3"
    assert factory.calls == [
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        }
    ]
    assert factory.clients[0].calls[0] == (
        "create_session",
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "action_dim": 7,
            "action_horizon": 10,
            "max_token_len": 180,
            "camera_layout": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        },
    )


def test_action_session_manager_act_returns_actions(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIFastActionSessionManager(runtime_factory=factory)
    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=None,
            base_model=OPENPI_FAST_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    result = asyncio.run(
        manager.act(
            action_session_id=action_session_id,
            observation=_make_observation(),
            extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
        )
    )

    assert result == {
        "actions": {
            "data": [0.0] * 28,
            "shape": [4, 7],
            "dtype": "float32",
        },
        "policy_timing": {"infer_ms": 4.0},
    }
    assert factory.clients[0].calls[-1][0] == "act"


def test_action_session_manager_rejects_unsupported_model_family(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    manager = OpenPIFastActionSessionManager(runtime_factory=_FakeActionRuntimeFactory())

    with pytest.raises(ValueError, match="OpenPI FAST"):
        asyncio.run(
            manager.create_session(
                session_id="session-1",
                action_session_seq_id=None,
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                model_path=f"file://{checkpoint_dir}",
                user_id="admin",
            )
        )
