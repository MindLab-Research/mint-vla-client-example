from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

pytest.importorskip("ray")

import mint_server.backend.inference.multinode_inference as mi


class _RegistryStub:
    def __init__(self, *, lora_id: int | None, adapter_path: str | None) -> None:
        self._lora_id = lora_id
        self._adapter_path = adapter_path

    async def get_lora_id(self, _sampling_session_id: str) -> int | None:
        return self._lora_id

    async def get_adapter_path(self, _lora_id: int) -> str | None:
        return self._adapter_path


def _fake_generate_engine():
    return SimpleNamespace(generate=SimpleNamespace(remote=lambda **_kwargs: object()))


def test_issue_358_generate_missing_lora_path_fails_without_killing_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        mi.ray_kill,
        "kill",
        lambda actor_handle, **kwargs: kill_calls.append(
            {"actor_handle": actor_handle, **kwargs}
        ),
    )

    fake_self = SimpleNamespace(
        _initialized=True,
        registry=_RegistryStub(lora_id=7, adapter_path="/tmp/definitely-missing-issue358-adapter"),
        actor_name="mint_vllm_qwen3_30b_a3b_instruct_2507",
        engine=_fake_generate_engine(),
    )

    with pytest.raises(ValueError, match="Missing LoRA adapter path"):
        anyio.run(
            mi.MultiNodeInferenceEngine.generate,
            fake_self,
            "sess-358",
            [101, 102, 103],
            "req-358-missing-path",
            4,
        )

    assert kill_calls == []


def test_issue_358_remote_adapter_file_missing_does_not_kill_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    remote_error = RuntimeError(
        f"vllm_add_request_failed request_id=req-358-asset\n"
        f"File not found: {adapter_dir / 'adapter_model.safetensors'}"
    )
    kill_calls: list[dict[str, object]] = []

    async def _raise_missing_adapter_asset(*_args, **_kwargs):
        raise remote_error

    monkeypatch.setattr(mi, "ray_get_with_model_actor_supervisor_keepalive", _raise_missing_adapter_asset)
    monkeypatch.setattr(mi, "get_current_traceparent", lambda: None)
    monkeypatch.setattr(
        mi.ray_kill,
        "kill",
        lambda actor_handle, **kwargs: kill_calls.append(
            {"actor_handle": actor_handle, **kwargs}
        ),
    )

    fake_self = SimpleNamespace(
        _initialized=True,
        registry=_RegistryStub(lora_id=8, adapter_path=str(adapter_dir)),
        actor_name="mint_vllm_qwen3_30b_a3b_instruct_2507",
        engine=_fake_generate_engine(),
    )

    with pytest.raises(RuntimeError, match="adapter_model\\.safetensors"):
        anyio.run(
            mi.MultiNodeInferenceEngine.generate,
            fake_self,
            "sess-358",
            [101, 102, 103],
            "req-358-asset",
            4,
        )

    assert kill_calls == []


def test_issue_358_unrelated_file_not_found_still_kills_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    unrelated_error = RuntimeError(
        "vllm_add_request_failed request_id=req-358-unrelated\n"
        "File not found: /tmp/unrelated-runtime-socket"
    )
    kill_calls: list[dict[str, object]] = []

    async def _raise_unrelated_missing_file(*_args, **_kwargs):
        raise unrelated_error

    monkeypatch.setattr(mi, "ray_get_with_model_actor_supervisor_keepalive", _raise_unrelated_missing_file)
    monkeypatch.setattr(mi, "get_current_traceparent", lambda: None)
    monkeypatch.setattr(
        mi.ray_kill,
        "kill",
        lambda actor_handle, **kwargs: kill_calls.append(
            {"actor_handle": actor_handle, **kwargs}
        ),
    )

    fake_self = SimpleNamespace(
        _initialized=True,
        registry=_RegistryStub(lora_id=9, adapter_path=str(adapter_dir)),
        actor_name="mint_vllm_qwen3_30b_a3b_instruct_2507",
        engine=_fake_generate_engine(),
    )

    with pytest.raises(RuntimeError, match="unrelated-runtime-socket"):
        anyio.run(
            mi.MultiNodeInferenceEngine.generate,
            fake_self,
            "sess-358",
            [101, 102, 103],
            "req-358-unrelated",
            4,
        )

    assert len(kill_calls) == 1
