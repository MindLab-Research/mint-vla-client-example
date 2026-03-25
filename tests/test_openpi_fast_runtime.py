from __future__ import annotations

import asyncio
import os
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
    return _make_datum_with_rl()


def _make_datum_with_rl(
    *,
    logprobs: list[float] | None = None,
    advantages: list[float] | None = None,
) -> Datum:
    loss_fn_inputs: dict[str, object] = {
        "state": {"data": [0.1] * 7, "shape": [7], "dtype": "float32"},
        "target_tokens": {"data": [21, 22], "shape": [2], "dtype": "int64"},
        "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
        "token_ar_mask": {"data": [1, 1], "shape": [2], "dtype": "int32"},
    }
    if logprobs is not None:
        loss_fn_inputs["logprobs"] = {"data": logprobs, "shape": [2], "dtype": "float32"}
    if advantages is not None:
        loss_fn_inputs["advantages"] = {"data": advantages, "shape": [2], "dtype": "float32"}
    return Datum(
        model_input=ModelInput(
            chunks=[
                ImageChunk(data=b"img-0", format="png", expected_tokens=256),
                ImageChunk(data=b"img-1", format="png", expected_tokens=256),
                ImageChunk(data=b"img-2", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[11, 12, 13]),
            ]
        ),
        loss_fn_inputs=loss_fn_inputs,
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
            if payload["loss_fn"] == "importance_sampling":
                return {
                    "loss_fn_output_type": "importance_sampling_loss",
                    "loss_fn_outputs": [
                        {"loss": {"data": [float(i + 1)], "shape": [1], "dtype": "float32"}}
                        for i, _ in enumerate(batch)
                    ],
                    "metrics": {
                        "loss:mean": 0.5,
                        "num_samples:sum": float(len(batch)),
                        "ratio:mean": 1.25,
                    },
                }
            if payload["loss_fn"] == "ppo":
                return {
                    "loss_fn_output_type": "ppo_loss",
                    "loss_fn_outputs": [
                        {"loss": {"data": [float(i + 1)], "shape": [1], "dtype": "float32"}}
                        for i, _ in enumerate(batch)
                    ],
                    "metrics": {
                        "loss:mean": 0.75,
                        "num_samples:sum": float(len(batch)),
                        "ratio:mean": 1.15,
                        "clipfrac:mean": 0.5,
                    },
                }
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


def test_openpi_fast_default_runtime_factory_uses_shared_ray_runtime(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_training import _default_runtime_factory

    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config):
        calls.append(
            {
                "model_id": session.model_id,
                "worker_module": spec.worker_module,
                "python_executable": spec.python_executable,
                "config_name": config_name,
                "action_dim": model_config.action_dim,
                "action_horizon": model_config.action_horizon,
            }
        )
        return "shared-ray-runtime-client"

    async def _unexpected_local_start(spec):
        raise AssertionError(f"local subprocess path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_training.start_openpi_shared_ray_runtime",
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
            model_config=SimpleNamespace(action_dim=32, action_horizon=10),
            config_name="pi0_fast_libero_low_mem_finetune",
        )
    )

    assert runtime == "shared-ray-runtime-client"
    assert calls == [
        {
            "model_id": "model-1",
            "worker_module": "tinker_server.backend.openpi_fast_worker",
            "python_executable": os.environ.get("MINT_OPENPI_FAST_PYTHON") or os.sys.executable,
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "action_dim": 32,
            "action_horizon": 10,
        }
    ]


def test_openpi_fast_engine_create_training_session_surfaces_ray_start_failure_without_local_fallback(
    monkeypatch,
) -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    async def _failing_start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config):
        _ = session, spec, config_name, model_config
        raise RuntimeError("ray actor start failed")

    async def _unexpected_local_start(spec):
        raise AssertionError(f"local subprocess fallback must stay disabled: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_training.start_openpi_shared_ray_runtime",
        _failing_start_openpi_shared_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_start,
    )

    engine = OpenPIFastTrainingEngine()
    session = _make_session()

    with pytest.raises(RuntimeError, match="ray actor start failed"):
        asyncio.run(engine.create_training_session(session))


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


def test_openpi_fast_engine_rejects_unknown_loss_functions() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))
    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(data=[_make_datum()], loss_fn="mystery_loss"),
    )

    with pytest.raises(ValueError, match="cross_entropy|importance_sampling|ppo"):
        asyncio.run(engine.forward_backward(session, request))


def test_openpi_fast_engine_importance_sampling_builds_rl_payload_and_updates_grad_state() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))
    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(
            data=[
                _make_datum_with_rl(
                    logprobs=[-0.1, -0.2],
                    advantages=[1.5, -0.5],
                )
            ],
            loss_fn="importance_sampling",
        ),
    )

    result = asyncio.run(engine.forward_backward(session, request))

    assert session.accumulated_gradients == 1
    assert result["loss_fn_output_type"] == "importance_sampling_loss"
    op, payload = factory.clients[0].calls[-1]
    assert op == "forward_backward"
    assert payload["loss_fn"] == "importance_sampling"
    assert payload["batch"][0]["tokenized_prompt"] == [11, 12, 13, 21, 22]
    assert payload["batch"][0]["old_logprobs"] == [-0.1, -0.2]
    assert payload["batch"][0]["advantages"] == [1.5, -0.5]


def test_openpi_fast_engine_ppo_builds_rl_payload_and_updates_grad_state() -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))
    request = ForwardBackwardRequest(
        model_id=session.model_id,
        forward_backward_input=ForwardBackwardInput(
            data=[
                _make_datum_with_rl(
                    logprobs=[-0.1, -0.2],
                    advantages=[1.5, -0.5],
                )
            ],
            loss_fn="ppo",
            loss_fn_config={"epsilon": 0.15},
        ),
    )

    result = asyncio.run(engine.forward_backward(session, request))

    assert session.accumulated_gradients == 1
    assert result["loss_fn_output_type"] == "ppo_loss"
    assert result["metrics"]["clipfrac:mean"] == pytest.approx(0.5)
    op, payload = factory.clients[0].calls[-1]
    assert op == "forward_backward"
    assert payload["loss_fn"] == "ppo"
    assert payload["loss_fn_config"] == {"epsilon": 0.15}
    assert payload["batch"][0]["tokenized_prompt"] == [11, 12, 13, 21, 22]
    assert payload["batch"][0]["old_logprobs"] == [-0.1, -0.2]
    assert payload["batch"][0]["advantages"] == [1.5, -0.5]


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


def test_openpi_fast_runtime_spec_uses_canonical_conley_paths(monkeypatch, tmp_path) -> None:
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    conley_root = tmp_path / "conley"
    tinker_root = conley_root / "tinker-server"
    openpi_src = conley_root / "openpi" / "src"
    hf_home = conley_root / "_cache" / "huggingface"
    openpi_cache = conley_root / "_cache" / "openpi"
    worker_python = conley_root / "_envs" / "openpi-runtime" / "bin" / "python"
    worker_python_target = conley_root / "_envs" / "python-bin" / "python3.12"

    for path in (tinker_root, openpi_src, hf_home, openpi_cache, worker_python.parent, worker_python_target.parent):
        path.mkdir(parents=True, exist_ok=True)
    worker_python_target.write_text("#!/bin/sh\n")
    worker_python_target.chmod(0o755)
    worker_python.symlink_to(worker_python_target)

    monkeypatch.setenv("PFS_TINKER_PATH", str(tinker_root))
    monkeypatch.delenv("MINT_OPENPI_FAST_PYTHON", raising=False)

    spec = OpenPIFastRuntimeSpec.from_env()
    env = spec.build_env()

    assert spec.python_executable == str(worker_python)
    assert spec.pythonpath == (str(tinker_root.resolve()), str(openpi_src.resolve()))
    assert env["HF_HOME"] == str(hf_home.resolve())
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["OPENPI_DATA_HOME"] == str(openpi_cache.resolve())


def test_openpi_fast_runtime_build_env_does_not_inherit_parent_pythonpath(
    monkeypatch, tmp_path
) -> None:
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    conley_root = tmp_path / "conley"
    tinker_root = conley_root / "tinker-server"
    openpi_src = conley_root / "openpi" / "src"
    hf_home = conley_root / "_cache" / "huggingface"
    openpi_cache = conley_root / "_cache" / "openpi"
    worker_python = conley_root / "_envs" / "openpi-runtime" / "bin" / "python"

    for path in (tinker_root, openpi_src, hf_home, openpi_cache, worker_python.parent):
        path.mkdir(parents=True, exist_ok=True)
    worker_python.write_text("#!/bin/sh\n")
    worker_python.chmod(0o755)

    monkeypatch.setenv("PFS_TINKER_PATH", str(tinker_root))
    monkeypatch.setenv(
        "PYTHONPATH",
        "/vePFS-Mindverse/share/code/tinker-server-auth/.venv31213/lib/python3.12/site-packages",
    )
    monkeypatch.delenv("MINT_OPENPI_FAST_PYTHON", raising=False)

    spec = OpenPIFastRuntimeSpec.from_env()
    env = spec.build_env()

    assert env["PYTHONPATH"] == os.pathsep.join(
        (str(tinker_root.resolve()), str(openpi_src.resolve()))
    )


def test_openpi_fast_runtime_spec_rejects_missing_canonical_runtime_python(
    monkeypatch, tmp_path
) -> None:
    from tinker_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    conley_root = tmp_path / "conley"
    tinker_root = conley_root / "tinker-server"
    openpi_src = conley_root / "openpi" / "src"
    hf_home = conley_root / "_cache" / "huggingface"
    openpi_cache = conley_root / "_cache" / "openpi"

    for path in (tinker_root, openpi_src, hf_home, openpi_cache):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PFS_TINKER_PATH", str(tinker_root))
    monkeypatch.delenv("MINT_OPENPI_FAST_PYTHON", raising=False)

    with pytest.raises(FileNotFoundError, match="_envs/openpi-runtime/bin/python"):
        OpenPIFastRuntimeSpec.from_env()


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
