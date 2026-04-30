from types import SimpleNamespace

import pytest

ray = pytest.importorskip("ray")
if not hasattr(ray, "remote"):
    pytest.skip("ray runtime without actor decorators is not usable for Megatron tests", allow_module_level=True)

from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import VerlTrainingEngine
from tinker_server.backend.megatron_distributed import MegatronWorkerGroup
from tinker_server.models.types import LoRAConfig


def test_issue_476_reused_shared_actor_preserves_new_session_actual_rank():
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "session_rank64"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 3e-4
    group._actual_rank = 64
    group._step_count = 12
    group.lora_rank = 64
    group._last_session_switch_stats = None
    group._session_unknown_due_to_partial_swap = False
    group.workers = []

    saved_metadata: list[tuple[str, int, float, int | None]] = []
    reinit_calls: list[dict] = []
    reset_calls: list[tuple] = []

    def save_metadata(session_id, step, lr, actual_rank, **_kwargs):
        saved_metadata.append((session_id, step, lr, actual_rank))

    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_rank64",
        has_actor_only_state=lambda session_id: False,
        has_persisted_actor_only_state=lambda session_id: False,
        get_session_path=lambda session_id: f"/tmp/{session_id}",
        save_metadata=save_metadata,
    )
    group._bind_traceparent = lambda traceparent: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group.save_adapter_state = lambda *args, **kwargs: {"status": "ok"}
    group._swap_session_on_workers = lambda session_id, **kwargs: []

    def fake_reinit_lora_weights(*args, **kwargs):
        reinit_calls.append(kwargs)
        return {"status": "ok", "actual_rank": kwargs.get("actual_rank")}

    group.reinit_lora_weights = fake_reinit_lora_weights
    group.reset_expert_bias = lambda *args, **kwargs: reset_calls.append((args, kwargs))

    result = group._ensure_session_loaded(
        "session_rank16",
        actual_rank=16,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )

    assert result["switched"] is True
    assert saved_metadata == [("session_rank64", 12, 3e-4, 64)]
    assert reinit_calls == [
        {
            "actual_rank": 16,
            "new_session_id": "session_rank16",
            "traceparent": None,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": True,
        }
    ]
    assert len(reset_calls) == 1
    assert group._current_session == "session_rank16"
    assert group._actual_rank == 16
    assert group._step_count == 0


def test_issue_476_megatron_train_step_passes_actual_rank(monkeypatch):
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="model_issue_476_train_step",
        session_id="session_issue_476_train_step",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_config=LoRAConfig(rank=16, train_attn=True, train_mlp=True, train_unembed=True),
        backend="megatron",
    )

    class _FakeDatum:
        def model_dump(self):
            return {"datum": 1}

    class _FakeTrainStepRemote:
        def __init__(self):
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"loss_fn_outputs": [], "metrics": {}}

    fake_remote = _FakeTrainStepRemote()
    worker = SimpleNamespace(train_step=fake_remote)

    async def fake_get_live_worker(_session, *, op, allow_recover=False):
        assert _session is session
        assert op == "train_step"
        assert allow_recover is False
        return worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        return awaitable

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_touch_actor", lambda _session: None)
    monkeypatch.setattr(engine, "_record_megatron_result_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True),
    )

    request = SimpleNamespace(
        forward_backward_input=SimpleNamespace(
            data=[_FakeDatum()],
            loss_fn="cross_entropy",
            loss_fn_config={},
        ),
        adam_params=SimpleNamespace(learning_rate=2e-4),
    )

    import asyncio

    result = asyncio.run(engine.train_step(session, request))

    assert result["metrics"]["step"] == 1
    assert fake_remote.calls
    args, kwargs = fake_remote.calls[0]
    assert args[:7] == (
        [{"datum": 1}],
        "cross_entropy",
        {},
        None,
        2e-4,
        "model_issue_476_train_step",
        16,
    )
    assert kwargs["train_attn"] is True
    assert kwargs["train_mlp"] is True
    assert kwargs["train_unembed"] is True
