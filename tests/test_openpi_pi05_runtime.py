from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.models.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    ForwardBackwardRequest,
    ImageChunk,
    ModelInput,
    OptimStepRequest,
    TensorData,
    TrainStepRequest,
)


OPENPI_PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"


def _make_session() -> TrainingSession:
    return TrainingSession(
        model_id="model-1",
        session_id="session-1",
        model_seq_id=0,
        base_model=OPENPI_PI05_MODEL,
    )


def _make_datum() -> Datum:
    return Datum(
        model_input=ModelInput(
            chunks=[
                ImageChunk(data=b"img-0", format="png", expected_tokens=256),
                ImageChunk(data=b"img-1", format="png", expected_tokens=256),
                ImageChunk(data=b"img-2", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[11, 12, 13]),
            ]
        ),
        loss_fn_inputs={
            "state": TensorData(data=[0.5] * 8, shape=[8], dtype="float32"),
            "actions": TensorData(
                data=[float(i) for i in range(10 * 7)],
                shape=[10, 7],
                dtype="float32",
            ),
        },
    )


class _FakeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self.closed = False

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"session": "created"}
        if op == "forward_backward":
            batch = payload["batch"]
            return {
                "loss_fn_output_type": "flow_matching_loss",
                "loss_fn_outputs": [
                    {"loss": {"data": [float(i + 1)], "shape": [1], "dtype": "float32"}}
                    for i, _ in enumerate(batch)
                ],
                "metrics": {"loss:mean": 1.25, "num_samples:sum": float(len(batch))},
            }
        if op == "optim_step":
            return {"metrics": {"learning_rate": payload["learning_rate"]}}
        if op == "save_weights":
            return {"path": payload["save_path"]}
        if op == "load_weights":
            return {"current_step": 5, "learning_rate": 0.001}
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected op {op}")

    async def close(self) -> None:
        self.closed = True


class _FakeRuntimeFactory:
    def __init__(self) -> None:
        self.clients: list[_FakeRuntimeClient] = []
        self.calls: list[dict] = []

    async def __call__(self, *, session: TrainingSession, model_config, config_name: str):
        self.calls.append(
            {
                "model_id": session.model_id,
                "base_model": session.base_model,
                "config_name": config_name,
                "camera_layout": model_config.camera_layout,
                "action_dim": model_config.action_dim,
                "action_horizon": model_config.action_horizon,
                "max_model_len": model_config.max_model_len,
            }
        )
        client = _FakeRuntimeClient()
        self.clients.append(client)
        return client


def _pi05_model_config():
    return SimpleNamespace(
        num_parameters=3.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=200,
        policy_family="flow_action",
        inference_modality="actions",
        training_backend="openpi_pi05",
        camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        action_dim=32,
        action_horizon=10,
    )


def test_openpi_pi05_engine_create_training_session_starts_runtime(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import (
        OPENPI_PI05_TRAINING_BACKEND,
        OpenPIPi05TrainingEngine,
    )

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.get_model_config",
        lambda base_model: _pi05_model_config(),
    )

    factory = _FakeRuntimeFactory()
    engine = OpenPIPi05TrainingEngine(runtime_factory=factory)
    session = _make_session()

    asyncio.run(engine.create_training_session(session))

    assert session.backend == OPENPI_PI05_TRAINING_BACKEND
    assert session.is_active is True
    assert factory.calls == [
        {
            "model_id": "model-1",
            "base_model": OPENPI_PI05_MODEL,
            "config_name": "pi05_libero",
            "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            "action_dim": 32,
            "action_horizon": 10,
            "max_model_len": 200,
        }
    ]
    assert factory.clients[0].calls[0][0] == "create_session"


def test_openpi_pi05_default_runtime_factory_uses_shared_ray_runtime(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import _default_runtime_factory

    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config):
        calls.append(
            {
                "model_id": session.model_id,
                "worker_module": spec.worker_module,
                "config_name": config_name,
                "max_model_len": model_config.max_model_len,
            }
        )
        return "shared-ray-runtime-client"

    async def _unexpected_local_start(spec):
        raise AssertionError(f"local subprocess path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.start_openpi_shared_ray_runtime",
        _fake_start_openpi_shared_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_start,
    )

    runtime = asyncio.run(
        _default_runtime_factory(
            session=_make_session(),
            model_config=_pi05_model_config(),
            config_name="pi05_libero",
        )
    )

    assert runtime == "shared-ray-runtime-client"
    assert calls == [
        {
            "model_id": "model-1",
            "worker_module": "tinker_server.backend.openpi_pi05_worker",
            "config_name": "pi05_libero",
            "max_model_len": 200,
        }
    ]


def test_openpi_pi05_engine_forward_backward_builds_payload_and_updates_grad_state(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import OpenPIPi05TrainingEngine

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.get_model_config",
        lambda base_model: _pi05_model_config(),
    )

    factory = _FakeRuntimeFactory()
    engine = OpenPIPi05TrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="flow_matching"),
    )

    result = asyncio.run(engine.forward_backward(session, request))

    assert session.accumulated_gradients == 1
    assert result["loss_fn_output_type"] == "flow_matching_loss"
    op, payload = factory.clients[0].calls[-1]
    assert op == "forward_backward"
    assert payload["loss_fn"] == "flow_matching"
    assert len(payload["batch"]) == 1
    assert payload["batch"][0]["tokenized_prompt"] == [11, 12, 13]
    assert len(payload["batch"][0]["state"]) == 32
    assert len(payload["batch"][0]["actions"]) == 10
    assert all(len(step) == 32 for step in payload["batch"][0]["actions"])


def test_openpi_pi05_engine_rejects_unknown_loss_functions(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import OpenPIPi05TrainingEngine

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.get_model_config",
        lambda base_model: _pi05_model_config(),
    )

    factory = _FakeRuntimeFactory()
    engine = OpenPIPi05TrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))
    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="cross_entropy"),
    )

    with pytest.raises(ValueError, match="flow_matching"):
        asyncio.run(engine.forward_backward(session, request))


def test_openpi_pi05_engine_train_step_composes_forward_backward_and_optim_step(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import OpenPIPi05TrainingEngine

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.get_model_config",
        lambda base_model: _pi05_model_config(),
    )

    factory = _FakeRuntimeFactory()
    engine = OpenPIPi05TrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    request = TrainStepRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="flow_matching"),
        adam_params=AdamParams(learning_rate=0.003),
    )

    result = asyncio.run(engine.train_step(session, request))

    assert session.current_step == 1
    assert session.accumulated_gradients == 0
    assert result["metrics"]["step"] == 1
    assert [name for name, _ in factory.clients[0].calls[-2:]] == ["forward_backward", "optim_step"]


def test_openpi_pi05_engine_save_load_and_shutdown_delegate_to_runtime(monkeypatch) -> None:
    from tinker_server.backend.openpi_pi05_training import OpenPIPi05TrainingEngine

    monkeypatch.setattr(
        "tinker_server.backend.openpi_pi05_training.get_model_config",
        lambda base_model: _pi05_model_config(),
    )

    factory = _FakeRuntimeFactory()
    engine = OpenPIPi05TrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    save_path = asyncio.run(engine.save_weights(session, "/tmp/openpi-pi05-ckpt"))
    asyncio.run(engine.load_weights(session, "/tmp/openpi-pi05-ckpt", load_optimizer=True))
    asyncio.run(engine.shutdown_session(session))

    assert save_path == "/tmp/openpi-pi05-ckpt"
    assert session.current_step == 5
    assert session.learning_rate == 0.001
    assert session.is_active is False
    assert factory.clients[0].closed is True
    assert [name for name, _ in factory.clients[0].calls[-3:]] == [
        "save_weights",
        "load_weights",
        "shutdown",
    ]
