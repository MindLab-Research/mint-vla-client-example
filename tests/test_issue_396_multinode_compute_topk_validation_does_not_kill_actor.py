from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

pytest.importorskip("ray")

import mint_server.backend.multinode_inference as mi


class _RegistryStub:
    async def get_lora_id(self, _sampling_session_id: str) -> None:
        return None

    async def get_adapter_path(self, _lora_id: int) -> None:
        return None


def test_issue_396_compute_topk_validation_error_does_not_kill_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_error = RuntimeError(
        "vllm_prompt_topk_add_request_failed request_id=req k=21\n"
        "vllm.exceptions.VLLMValidationError: Requested prompt logprobs of 21, "
        "which is greater than max allowed: 20 (parameter=prompt_logprobs, value=21)"
    )
    kill_calls: list[dict[str, object]] = []

    async def _raise_validation_error(*_args, **_kwargs):
        raise validation_error

    monkeypatch.setattr(mi, "ray_get_with_model_actor_supervisor_keepalive", _raise_validation_error)
    monkeypatch.setattr(
        mi.ray_kill,
        "kill",
        lambda actor_handle, **kwargs: kill_calls.append(
            {"actor_handle": actor_handle, **kwargs}
        ),
    )
    monkeypatch.setattr(mi, "get_current_traceparent", lambda: None)

    fake_engine = SimpleNamespace(compute_prompt_topk=SimpleNamespace(remote=lambda **_kwargs: object()))
    fake_self = SimpleNamespace(
        _initialized=True,
        max_model_len=None,
        registry=_RegistryStub(),
        actor_name="mint_vllm_qwen3-30b-a3b-instruct-2507",
        engine=fake_engine,
    )

    with pytest.raises(RuntimeError, match="Requested prompt logprobs of 21"):
        anyio.run(
            mi.MultiNodeInferenceEngine.compute_topk,
            fake_self,
            None,
            [101, 102, 103],
            "req",
            21,
        )

    assert kill_calls == []
