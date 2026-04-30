from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_issue_572_zero_rank_tail_masks_params_and_grads() -> None:
    from tinker_server.backend.lora_utils import zero_lora_rank_tail_named_parameters

    params = {
        "layers.0.self_attn.q_proj.lora_A.default.weight": torch.nn.Parameter(torch.ones(64, 3)),
        "layers.0.self_attn.q_proj.lora_B.default.weight": torch.nn.Parameter(torch.ones(5, 64)),
        "layers.0.mlp.adapter.linear_in.weight": torch.nn.Parameter(torch.ones(64, 7)),
        "layers.0.mlp.adapter.linear_out.weight": torch.nn.Parameter(torch.ones(11, 64)),
        "layers.0.mlp.base_layer.weight": torch.nn.Parameter(torch.ones(11, 7)),
    }
    for param in params.values():
        param.grad = torch.ones_like(param)

    stats = zero_lora_rank_tail_named_parameters(params.items(), actual_rank=16, trainer_rank=64)

    assert stats == {"params": 4, "grads": 4}
    assert torch.all(params["layers.0.self_attn.q_proj.lora_A.default.weight"].data[:16] == 1)
    assert torch.all(params["layers.0.self_attn.q_proj.lora_A.default.weight"].data[16:] == 0)
    assert torch.all(params["layers.0.self_attn.q_proj.lora_B.default.weight"].data[:, :16] == 1)
    assert torch.all(params["layers.0.self_attn.q_proj.lora_B.default.weight"].data[:, 16:] == 0)
    assert torch.all(params["layers.0.mlp.adapter.linear_in.weight"].grad[16:] == 0)
    assert torch.all(params["layers.0.mlp.adapter.linear_out.weight"].grad[:, 16:] == 0)
    assert torch.all(params["layers.0.mlp.base_layer.weight"].data == 1)


def test_issue_572_zero_rank_tail_supports_tp_sharded_rank_params() -> None:
    from tinker_server.backend.lora_utils import zero_lora_rank_tail_named_parameters

    rank0 = {
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": torch.nn.Parameter(torch.ones(16, 3)),
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": torch.nn.Parameter(torch.ones(5, 16)),
    }
    rank1 = {
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": torch.nn.Parameter(torch.ones(16, 3)),
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": torch.nn.Parameter(torch.ones(5, 16)),
    }
    rank2 = {
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": torch.nn.Parameter(torch.ones(16, 3)),
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": torch.nn.Parameter(torch.ones(5, 16)),
    }

    stats0 = zero_lora_rank_tail_named_parameters(
        rank0.items(),
        actual_rank=20,
        trainer_rank=64,
        rank_shard_index=0,
        rank_shard_count=4,
    )
    stats1 = zero_lora_rank_tail_named_parameters(
        rank1.items(),
        actual_rank=20,
        trainer_rank=64,
        rank_shard_index=1,
        rank_shard_count=4,
    )
    stats2 = zero_lora_rank_tail_named_parameters(
        rank2.items(),
        actual_rank=20,
        trainer_rank=64,
        rank_shard_index=2,
        rank_shard_count=4,
    )

    assert stats0 == {"params": 0, "grads": 0}
    assert stats1 == {"params": 2, "grads": 0}
    assert stats2 == {"params": 2, "grads": 0}
    assert torch.all(rank1["decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"].data[:4] == 1)
    assert torch.all(rank1["decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"].data[4:] == 0)
    assert torch.all(rank1["decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"].data[:, :4] == 1)
    assert torch.all(rank1["decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"].data[:, 4:] == 0)
    assert torch.all(rank2["decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"].data == 0)
    assert torch.all(rank2["decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"].data == 0)


def test_issue_572_pad_and_truncate_support_megatron_adapter_names() -> None:
    from tinker_server.backend.lora_utils import pad_lora_state_dict, truncate_lora_state_dict

    state = {
        "layers.0.mlp.adapter.linear_in.weight": torch.ones(16, 4),
        "layers.0.mlp.adapter.linear_out.weight": torch.ones(9, 16),
    }

    padded = pad_lora_state_dict(state, actual_rank=16, trainer_rank=64)
    assert padded["layers.0.mlp.adapter.linear_in.weight"].shape == (64, 4)
    assert padded["layers.0.mlp.adapter.linear_out.weight"].shape == (9, 64)
    assert torch.all(padded["layers.0.mlp.adapter.linear_in.weight"][16:] == 0)
    assert torch.all(padded["layers.0.mlp.adapter.linear_out.weight"][:, 16:] == 0)

    truncated = truncate_lora_state_dict(padded, trainer_rank=64, actual_rank=16)
    assert truncated["layers.0.mlp.adapter.linear_in.weight"].shape == (16, 4)
    assert truncated["layers.0.mlp.adapter.linear_out.weight"].shape == (9, 16)

    full_rank = {
        "layers.0.mlp.adapter.linear_in.weight": torch.ones(64, 4),
        "layers.0.mlp.adapter.linear_out.weight": torch.ones(9, 64),
    }
    accepted = pad_lora_state_dict(full_rank, actual_rank=16, trainer_rank=64)
    assert accepted["layers.0.mlp.adapter.linear_in.weight"].shape == (64, 4)
    assert accepted["layers.0.mlp.adapter.linear_out.weight"].shape == (9, 64)


def test_issue_572_fit_lora_state_dict_to_tp_local_reference() -> None:
    from tinker_server.backend.lora_utils import fit_lora_state_dict_to_reference

    state = {
        "layers.0.mlp.adapter.linear_in.weight": torch.arange(64 * 4).reshape(64, 4),
        "layers.0.mlp.adapter.linear_out.weight": torch.arange(9 * 64).reshape(9, 64),
    }
    reference = {
        "layers.0.mlp.adapter.linear_in.weight": torch.empty(16, 4),
        "layers.0.mlp.adapter.linear_out.weight": torch.empty(9, 16),
    }

    fitted = fit_lora_state_dict_to_reference(state, reference)

    assert fitted["layers.0.mlp.adapter.linear_in.weight"].shape == (16, 4)
    assert fitted["layers.0.mlp.adapter.linear_out.weight"].shape == (9, 16)
    assert torch.equal(fitted["layers.0.mlp.adapter.linear_in.weight"], state["layers.0.mlp.adapter.linear_in.weight"][:16])
    assert torch.equal(fitted["layers.0.mlp.adapter.linear_out.weight"], state["layers.0.mlp.adapter.linear_out.weight"][:, :16])

    shard1 = fit_lora_state_dict_to_reference(
        state,
        reference,
        rank_shard_index=1,
        rank_shard_count=4,
    )
    assert torch.equal(
        shard1["layers.0.mlp.adapter.linear_in.weight"],
        state["layers.0.mlp.adapter.linear_in.weight"][16:32],
    )
    assert torch.equal(
        shard1["layers.0.mlp.adapter.linear_out.weight"],
        state["layers.0.mlp.adapter.linear_out.weight"][:, 16:32],
    )

    rank20 = {
        "layers.0.mlp.adapter.linear_in.weight": torch.arange(20 * 4).reshape(20, 4),
        "layers.0.mlp.adapter.linear_out.weight": torch.arange(9 * 20).reshape(9, 20),
    }
    shard1_partial = fit_lora_state_dict_to_reference(
        rank20,
        reference,
        rank_shard_index=1,
        rank_shard_count=4,
    )
    assert torch.equal(
        shard1_partial["layers.0.mlp.adapter.linear_in.weight"][:4],
        rank20["layers.0.mlp.adapter.linear_in.weight"][16:20],
    )
    assert torch.all(shard1_partial["layers.0.mlp.adapter.linear_in.weight"][4:] == 0)
    assert torch.equal(
        shard1_partial["layers.0.mlp.adapter.linear_out.weight"][:, :4],
        rank20["layers.0.mlp.adapter.linear_out.weight"][:, 16:20],
    )
    assert torch.all(shard1_partial["layers.0.mlp.adapter.linear_out.weight"][:, 4:] == 0)


def test_issue_572_dense_export_config_uses_actual_rank() -> None:
    from tinker_server.backend.verl_training import TrainingWorker

    worker_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.max_lora_rank = 64
    worker._current_actual_rank = None
    worker._base_model = "Qwen/Qwen3-0.6B"
    worker.model = SimpleNamespace(
        peft_config={
            "default": SimpleNamespace(
                r=64,
                lora_alpha=64,
                lora_dropout=0.0,
                target_modules={"q_proj", "v_proj"},
                bias="none",
                task_type=SimpleNamespace(value="CAUSAL_LM"),
            )
        }
    )

    config = worker.get_lora_config(actual_rank=16)

    assert config["r"] == 16
    assert config["lora_alpha"] == 16


def test_issue_572_dense_load_checkpoint_pads_actual_rank_adapter(tmp_path, monkeypatch) -> None:
    from safetensors.torch import save_file
    from tinker_server.backend.verl_training import TrainingWorker

    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(16, 3),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.ones(5, 16),
        },
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "target_modules": ["linear_qkv"]}))
    (tmp_path / "training_meta.json").write_text(json.dumps({"actual_rank": 16, "current_step": 7}))

    worker_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.device = "cpu"
    worker.model = object()
    worker.max_lora_rank = 64
    worker._current_actual_rank = None
    worker._step_count = 0
    worker._bind_traceparent = lambda traceparent: None
    worker._touch = lambda: None
    worker.reset_optimizer = lambda lr: None

    ensure_calls = []
    zero_calls = []
    worker._ensure_session_loaded = lambda session_id, actual_rank=None: ensure_calls.append((session_id, actual_rank))
    worker._zero_lora_rank_tail = lambda actual_rank=None, zero_grads=True: zero_calls.append((actual_rank, zero_grads))

    captured = {}

    def _capture_state_dict(_model, state_dict):
        captured.update(state_dict)

    peft_mod = ModuleType("peft")
    peft_utils_mod = ModuleType("peft.utils")
    peft_save_mod = ModuleType("peft.utils.save_and_load")
    peft_save_mod.set_peft_model_state_dict = _capture_state_dict
    peft_utils_mod.save_and_load = peft_save_mod
    peft_mod.utils = peft_utils_mod
    monkeypatch.setitem(sys.modules, "peft", peft_mod)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils_mod)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save_mod)

    meta = worker.load_checkpoint(str(tmp_path), load_optimizer=False, session_id="session-r16")

    assert meta["actual_rank"] == 16
    assert ensure_calls == [("session-r16", 16)]
    assert zero_calls == [(16, True)]
    a = captured["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
    b = captured["base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight"]
    assert a.shape == (64, 3)
    assert b.shape == (5, 64)
    assert torch.all(a[:16] == 1)
    assert torch.all(a[16:] == 0)
    assert torch.all(b[:, :16] == 1)
    assert torch.all(b[:, 16:] == 0)


def test_issue_572_megatron_group_reinit_passes_actual_rank_to_rank_workers() -> None:
    ray = pytest.importorskip("ray")
    if not hasattr(ray, "remote"):
        pytest.skip("ray runtime without actor decorators is not usable for Megatron tests")

    from tinker_server.backend.megatron_distributed import MegatronWorkerGroup

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None
    group._actual_rank = 64
    group.lora_rank = 64
    group.learning_rate = 1e-4
    group._bind_traceparent = lambda traceparent: None
    group._invalidate_session_durability = lambda *_args, **_kwargs: None
    group._ray_get_group_results = lambda futures, **_kwargs: futures

    calls: list[tuple[tuple, dict]] = []

    class _Remote:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"reinit_count": 1, "opt_state_reset": 0, "lr_updated": False}

    group.workers = [SimpleNamespace(reinit_lora_weights=_Remote())]

    result = group.reinit_lora_weights(actual_rank=16, new_session_id="session-r16")

    assert result["actual_rank"] == 16
    assert calls == [
        (
            (None,),
            {
                "actual_rank": 16,
                "train_attn": None,
                "train_mlp": None,
                "train_unembed": None,
                "traceparent": None,
            },
        )
    ]


def test_issue_572_megatron_peft_export_truncates_to_actual_rank(tmp_path) -> None:
    ray = pytest.importorskip("ray")
    if not hasattr(ray, "remote"):
        pytest.skip("ray runtime without actor decorators is not usable for Megatron tests")

    from safetensors.torch import load_file
    from tinker_server.backend.megatron_distributed import MegatronRankWorker

    worker_cls = MegatronRankWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.rank = 0
    worker.lora_rank = 64
    worker.learning_rate = 1e-4
    worker.base_model = "unknown-test-model"
    worker._bind_traceparent = lambda traceparent: None
    class _TrainMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    worker.engine = SimpleNamespace(train_mode=lambda: _TrainMode())
    worker._zero_lora_rank_tail = lambda *args, **kwargs: {"params": 1, "grads": 0}
    worker.get_lora_state_dict = lambda **_kwargs: {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(64, 3),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(5, 64),
    }

    meta = worker.save_lora_weights(str(tmp_path), actual_rank=16)

    tensors = load_file(tmp_path / "adapter_model.safetensors")
    assert tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape == (16, 3)
    assert tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (5, 16)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == 16
    assert meta["actual_rank"] == 16


def test_issue_572_megatron_load_checkpoint_accepts_matching_actual_rank_metadata(tmp_path, monkeypatch) -> None:
    ray = pytest.importorskip("ray")
    if not hasattr(ray, "remote"):
        pytest.skip("ray runtime without actor decorators is not usable for Megatron tests")

    import tinker_server.backend.megatron_distributed as megatron_mod
    from tinker_server.backend.megatron_distributed import MegatronWorkerGroup

    (tmp_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "target_modules": ["linear_qkv"]}))
    (tmp_path / "training_meta.json").write_text(
        json.dumps({"actual_rank": 16, "current_step": 3, "learning_rate": 2e-4})
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.lora_rank = 64
    group.learning_rate = 1e-4
    group._current_session = None
    group._session_manager = None
    group._session_unknown_due_to_partial_swap = False
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id or "session-r16"
    group._prepare_session_for_explicit_load = lambda session_id, traceparent=None: None
    group._invalidate_session_durability = lambda *_args, **_kwargs: None
    group._clear_session_guards = lambda session_id: None
    group.load_adapter_state = lambda *args, **kwargs: {"status": "ok", "actual_rank": kwargs["actual_rank"]}
    group.reset_optimizer = lambda learning_rate, traceparent=None, zero_grad_buffers=True: {
        "status": "ok",
        "learning_rate": learning_rate,
    }

    mark_calls = []

    class _RemoteMethod:
        def remote(self, *args, **kwargs):
            mark_calls.append((args, kwargs))
            return {"status": "ok"}

    group.workers = [SimpleNamespace(mark_session_loaded=_RemoteMethod())]
    monkeypatch.setattr(megatron_mod.ray, "get", lambda value, timeout=None: value)

    result = group.load_checkpoint(str(tmp_path), load_optimizer=False, session_id="session-r16")

    assert result["actual_rank"] == 16
    assert result["current_step"] == 3
    assert result["learning_rate"] == 2e-4
    assert result["optimizer_reset"] is True
    assert group._actual_rank == 16
    assert mark_calls == [(("session-r16", 16), {})]


def test_issue_572_megatron_load_checkpoint_rejects_mismatched_actual_rank_metadata(tmp_path) -> None:
    ray = pytest.importorskip("ray")
    if not hasattr(ray, "remote"):
        pytest.skip("ray runtime without actor decorators is not usable for Megatron tests")

    from tinker_server.backend.megatron_distributed import MegatronWorkerGroup

    (tmp_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "target_modules": ["linear_qkv"]}))
    (tmp_path / "training_meta.json").write_text(
        json.dumps({"actual_rank": 64, "current_step": 3, "learning_rate": 2e-4})
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.lora_rank = 64
    group.learning_rate = 1e-4
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id or "session-r16"

    with pytest.raises(RuntimeError, match="metadata rank 64 does not match adapter_config rank 16"):
        group.load_checkpoint(str(tmp_path), load_optimizer=False, session_id="session-r16")
