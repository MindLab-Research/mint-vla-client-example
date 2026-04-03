from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_registers_lazy_multilora_session(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    registration: dict[str, object] = {}
    eager_calls: list[tuple[str, str]] = []

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failures.append((request_id, error))

    class _EngineStub:
        async def add_lora_for_session_from_path(self, sampling_session_id: str, lora_path: str) -> int:
            eager_calls.append((sampling_session_id, lora_path))
            raise AssertionError("save_weights_for_sampler must not eager-load LoRA")

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, _base_model: str):
            return _EngineStub()

        def register_multi_lora_session(self, **kwargs) -> None:
            registration.update(kwargs)

    def _resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-408",
                session_id="sess-408",
                base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                current_step=11,
                backend="megatron",
                lora_config=SimpleNamespace(rank=32, train_mlp=True),
                inference_engine=None,
            ),
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(tr, "inference_manager", _InferenceManagerStub())
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(resolve=_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )

    assert failures == []
    assert eager_calls == []
    assert resolved["request_id"] == "req-408-save"
    response = resolved["response"]
    assert isinstance(response, dict)
    sampling_session_id = response["sampling_session_id"]
    assert isinstance(sampling_session_id, str) and sampling_session_id
    assert response["path"] is None
    assert registration == {
        "session_id": sampling_session_id,
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "lora_rank": 32,
        "adapter_path": str(ckpt_dir),
        "lora_loaded": False,
    }
