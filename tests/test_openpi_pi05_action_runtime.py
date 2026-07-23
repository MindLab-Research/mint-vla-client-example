from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mint_server.models.types import EncodedTextChunk, ImageChunk, ModelInput, TensorData


OPENPI_PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"


def _make_observation() -> ModelInput:
    return ModelInput(
        chunks=[
            ImageChunk(data=b"img-0", format="png", expected_tokens=256),
            ImageChunk(data=b"img-1", format="png", expected_tokens=256),
            ImageChunk(data=b"img-2", format="png", expected_tokens=256),
            EncodedTextChunk(tokens=[11, 12, 13]),
        ]
    )


class _FakeActionRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"ready": True}
        if op == "act":
            result = {
                "actions": {
                    "data": [0.0] * 70,
                    "shape": [10, 7],
                    "dtype": "float32",
                },
                "policy_timing": {"infer_ms": 5.0},
            }
            if payload and payload.get("return_rollout_trace"):
                result["rollout_trace"] = {
                    "chains": {"data": [0.0] * (2 * 10 * 32), "shape": [2, 10, 32], "dtype": "float32"},
                    "denoise_inds": {"data": [0], "shape": [1], "dtype": "int32"},
                    "logprobs": {"data": [0.0] * 70, "shape": [10, 7], "dtype": "float32"},
                }
            return result
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected action op {op}")

    async def close(self) -> None:
        return None


class _FakeActionRuntimeFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.clients: list[_FakeActionRuntimeClient] = []

    async def __call__(
        self,
        *,
        action_session_id: str,
        base_model: str,
        checkpoint_path: str,
        model_config,
        config_name: str,
    ):
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


def _fake_get_model_config(base_model: str):
    if base_model == OPENPI_PI05_MODEL:
        return type(
            "_Cfg",
            (),
            {
                "training_backend": "openpi_pi05",
                "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                "action_dim": 32,
                "action_horizon": 10,
                "max_model_len": 200,
            },
        )()
    raise KeyError(base_model)


def test_openpi_pi05_action_session_manager_create_session_starts_runtime_from_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mint_server.backend.openpi.action_session_manager import OpenPIPi05ActionSessionManager

    monkeypatch.setattr(
        "mint_server.backend.openpi.action_session_manager.get_model_config",
        _fake_get_model_config,
    )

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIPi05ActionSessionManager(runtime_factory=factory)

    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=3,
            base_model=OPENPI_PI05_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    assert action_session_id == "session-1:action:3"
    assert factory.calls == [
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_PI05_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "config_name": "pi05_libero",
            "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        }
    ]
    assert factory.clients[0].calls[0] == (
        "create_session",
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_PI05_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "config_name": "pi05_libero",
            "action_dim": 32,
            "action_horizon": 10,
            "max_token_len": 200,
            "camera_layout": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        },
    )


def test_openpi_pi05_action_session_manager_rejects_profiled_checkpoint_before_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    from mint_server.backend.openpi.action_session_manager import OpenPIPi05ActionSessionManager

    profiled_model = "openpi/pi05-action-lora-r16-finetune"
    monkeypatch.setattr(
        "mint_server.backend.openpi.action_session_manager.get_model_config",
        lambda base_model: type(
            "_Cfg",
            (),
            {
                "training_backend": "openpi_pi05",
                "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                "action_dim": 32,
                "action_horizon": 10,
                "max_model_len": 200,
                "profile": "pi05_action_lora_r16_v1",
            },
        )()
        if base_model == profiled_model
        else (_ for _ in ()).throw(KeyError(base_model)),
    )
    checkpoint_dir = tmp_path / "missing-profile-manifest"
    (checkpoint_dir / "params").mkdir(parents=True)
    factory = _FakeActionRuntimeFactory()

    with pytest.raises(FileNotFoundError, match="profile manifest"):
        asyncio.run(
            OpenPIPi05ActionSessionManager(runtime_factory=factory).create_session(
                session_id="session-1",
                action_session_seq_id=None,
                base_model=profiled_model,
                model_path=f"file://{checkpoint_dir}",
                user_id="admin",
            )
        )
    assert factory.calls == []


def test_openpi_pi05_action_session_manager_act_returns_actions(monkeypatch, tmp_path: Path) -> None:
    from mint_server.backend.openpi.action_session_manager import OpenPIPi05ActionSessionManager

    monkeypatch.setattr(
        "mint_server.backend.openpi.action_session_manager.get_model_config",
        _fake_get_model_config,
    )

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIPi05ActionSessionManager(runtime_factory=factory)
    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=None,
            base_model=OPENPI_PI05_MODEL,
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
            "data": [0.0] * 70,
            "shape": [10, 7],
            "dtype": "float32",
        },
        "policy_timing": {"infer_ms": 5.0},
    }
    assert factory.clients[0].calls[-1][0] == "act"


def test_openpi_pi05_action_session_manager_act_can_request_rollout_trace(monkeypatch, tmp_path: Path) -> None:
    from mint_server.backend.openpi.action_session_manager import OpenPIPi05ActionSessionManager

    monkeypatch.setattr(
        "mint_server.backend.openpi.action_session_manager.get_model_config",
        _fake_get_model_config,
    )

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIPi05ActionSessionManager(runtime_factory=factory)
    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=None,
            base_model=OPENPI_PI05_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    result = asyncio.run(
        manager.act(
            action_session_id=action_session_id,
            observation=_make_observation(),
            extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
            return_rollout_trace=True,
            rollout_trace_config={"noise_method": "flow_noise", "joint_logprob": True, "num_steps": 1},
        )
    )

    _, payload = factory.clients[0].calls[-1]
    assert payload["return_rollout_trace"] is True  # type: ignore[union-attr]
    assert payload["rollout_trace_config"] == {"noise_method": "flow_noise", "joint_logprob": True, "num_steps": 1}  # type: ignore[union-attr]
    assert result["rollout_trace"]["chains"]["shape"] == [2, 10, 32]
    assert result["rollout_trace"]["logprobs"]["shape"] == [10, 7]


def test_openpi_pi05_action_session_manager_rejects_unsupported_model_family(monkeypatch, tmp_path: Path) -> None:
    from mint_server.backend.openpi.action_session_manager import OpenPIPi05ActionSessionManager

    monkeypatch.setattr(
        "mint_server.backend.openpi.action_session_manager.get_model_config",
        _fake_get_model_config,
    )

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    manager = OpenPIPi05ActionSessionManager(runtime_factory=_FakeActionRuntimeFactory())

    with pytest.raises(ValueError, match="pi0.5"):
        asyncio.run(
            manager.create_session(
                session_id="session-1",
                action_session_seq_id=None,
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                model_path=f"file://{checkpoint_dir}",
                user_id="admin",
            )
        )
