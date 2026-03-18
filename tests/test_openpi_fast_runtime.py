from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tinker_server.backend.openpi_fast_training import OPENPI_FAST_TRAINING_BACKEND
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
    TrainStepRequest,
)


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _make_session() -> TrainingSession:
    return TrainingSession(
        model_id="model-1",
        session_id="session-1",
        model_seq_id=0,
        base_model=OPENPI_FAST_MODEL,
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
            "state": {"data": [0.1] * 7, "shape": [7], "dtype": "float32"},
            "target_tokens": {"data": [21, 22], "shape": [2], "dtype": "int64"},
            "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            "token_ar_mask": {"data": [1, 1], "shape": [2], "dtype": "int32"},
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
                "loss_fn_output_type": "cross_entropy_loss",
                "loss_fn_outputs": [
                    {"loss": {"data": [float(i + 1)], "shape": [1], "dtype": "float32"}}
                    for i, _ in enumerate(batch)
                ],
                "metrics": {"loss:mean": 1.0, "num_samples:sum": float(len(batch))},
            }
        if op == "optim_step":
            return {"metrics": {"learning_rate": payload["learning_rate"]}}
        if op == "save_weights":
            return {"path": payload["save_path"]}
        if op == "load_weights":
            return {"current_step": 7, "learning_rate": 0.002}
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
            }
        )
        client = _FakeRuntimeClient()
        self.clients.append(client)
        return client


def test_openpi_fast_engine_create_training_session_starts_runtime() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()

    asyncio.run(engine.create_training_session(session))

    assert session.backend == OPENPI_FAST_TRAINING_BACKEND
    assert session.is_active is True
    assert factory.calls == [
        {
            "model_id": "model-1",
            "base_model": OPENPI_FAST_MODEL,
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        }
    ]
    assert factory.clients[0].calls[0][0] == "create_session"


def test_openpi_fast_engine_forward_backward_builds_payload_and_updates_grad_state() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="cross_entropy"),
    )

    result = asyncio.run(engine.forward_backward(session, request))

    assert session.accumulated_gradients == 1
    assert result["loss_fn_output_type"] == "cross_entropy_loss"
    op, payload = factory.clients[0].calls[-1]
    assert op == "forward_backward"
    assert payload["loss_fn"] == "cross_entropy"
    assert len(payload["batch"]) == 1
    assert payload["batch"][0]["tokenized_prompt"] == [11, 12, 13, 21, 22]


def test_openpi_fast_engine_rejects_non_sft_loss_functions() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))
    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="ppo"),
    )

    with pytest.raises(ValueError, match="cross_entropy"):
        asyncio.run(engine.forward_backward(session, request))


def test_openpi_fast_engine_train_step_composes_forward_backward_and_optim_step() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    request = TrainStepRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="cross_entropy"),
        adam_params=AdamParams(learning_rate=0.003),
    )

    result = asyncio.run(engine.train_step(session, request))

    assert session.current_step == 1
    assert session.accumulated_gradients == 0
    assert result["metrics"]["step"] == 1
    assert factory.clients[0].calls[-2][0] == "forward_backward"
    assert factory.clients[0].calls[-1] == ("optim_step", {"learning_rate": 0.003, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12})


def test_openpi_fast_engine_save_load_and_shutdown_delegate_to_runtime() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    save_path = asyncio.run(engine.save_weights(session, "/tmp/openpi-fast-ckpt"))
    asyncio.run(engine.load_weights(session, "/tmp/openpi-fast-ckpt", load_optimizer=True))
    asyncio.run(engine.shutdown_session(session))

    assert save_path == "/tmp/openpi-fast-ckpt"
    assert session.current_step == 7
    assert session.learning_rate == 0.002
    assert session.is_active is False
    assert factory.clients[0].closed is True
    assert [name for name, _ in factory.clients[0].calls[-3:]] == [
        "save_weights",
        "load_weights",
        "shutdown",
    ]


def test_openpi_fast_runtime_spec_reads_operation_specific_timeouts(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    monkeypatch.setenv("MINT_OPENPI_FAST_CREATE_SESSION_TIMEOUT_S", "900")
    monkeypatch.setenv("MINT_OPENPI_FAST_SAVE_TIMEOUT_S", "1200")
    monkeypatch.setenv("MINT_OPENPI_FAST_LOAD_TIMEOUT_S", "1500")

    spec = OpenPIFastRuntimeSpec.from_env()

    assert spec.create_session_timeout_s == 900.0
    assert spec.save_weights_timeout_s == 1200.0
    assert spec.load_weights_timeout_s == 1500.0


def test_openpi_fast_runtime_init_overrides_accept_local_weight_path(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_worker import OpenPIFastRuntimeInitOverrides

    monkeypatch.setenv("MINT_OPENPI_FAST_WEIGHTS_PATH", "/tmp/local-params")
    monkeypatch.delenv("MINT_OPENPI_FAST_RANDOM_INIT", raising=False)

    overrides = OpenPIFastRuntimeInitOverrides.from_env()

    assert overrides.weights_path == "/tmp/local-params"
    assert overrides.random_init is False


def test_openpi_fast_runtime_init_overrides_accept_random_init(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_worker import OpenPIFastRuntimeInitOverrides

    monkeypatch.delenv("MINT_OPENPI_FAST_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("MINT_OPENPI_FAST_RANDOM_INIT", "1")

    overrides = OpenPIFastRuntimeInitOverrides.from_env()

    assert overrides.weights_path is None
    assert overrides.random_init is True


def test_openpi_fast_runtime_init_overrides_reject_conflicting_weight_sources(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_worker import OpenPIFastRuntimeInitOverrides

    monkeypatch.setenv("MINT_OPENPI_FAST_WEIGHTS_PATH", "/tmp/local-params")
    monkeypatch.setenv("MINT_OPENPI_FAST_RANDOM_INIT", "1")

    with pytest.raises(ValueError, match="exclusive"):
        OpenPIFastRuntimeInitOverrides.from_env()
