from types import SimpleNamespace

import pytest

ray = pytest.importorskip("ray")
if not hasattr(ray, "remote"):
    pytest.skip("ray runtime without actor decorators is not usable for Megatron tests", allow_module_level=True)

from tinker_server.backend.megatron_distributed import MegatronWorkerGroup


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

    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_rank64",
        has_actor_only_state=lambda session_id: False,
        has_persisted_actor_only_state=lambda session_id: False,
        get_session_path=lambda session_id: f"/tmp/{session_id}",
        save_metadata=lambda session_id, step, lr, actual_rank: saved_metadata.append(
            (session_id, step, lr, actual_rank)
        ),
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
