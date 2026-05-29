import json
import sys
import types
from types import MethodType, SimpleNamespace

import pytest
import torch

from mint_server.checkpoints import checkpoint_has_optimizer_state, validate_checkpoint_dir
from mint_server.backend.verl_training import (
    VerlTrainingEngine,
    _select_moe_training_backend,
)
from mint_server.backend.bumblebee_distributed import (
    BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS,
    BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT,
    BUMBLEBEE_TRAIN_STATE_FILE,
    BUMBLEBEE_TRAIN_STATE_META_FILE,
    BumblebeeRankWorker,
    BumblebeeWorkerGroup,
    BumblebeeSessionMeta,
    _bumblebee_runtime_etp,
    _coerce_int,
    _make_bumblebee_pg_name,
    get_or_create_bumblebee_worker_group,
)


def test_qwen3_30b_defaults_to_bumblebee(monkeypatch):
    monkeypatch.delenv("MINT_QWEN3_30B_TRAINING_BACKEND", raising=False)
    monkeypatch.delenv("MINT_MOE_TRAINING_BACKEND", raising=False)

    assert _select_moe_training_backend("Qwen/Qwen3-30B-A3B-Instruct-2507") == "bumblebee"


def test_qwen3_30b_can_roll_back_to_megatron(monkeypatch):
    monkeypatch.setenv("MINT_QWEN3_30B_TRAINING_BACKEND", "megatron")

    assert _select_moe_training_backend("Qwen/Qwen3-30B-A3B-Instruct-2507") == "megatron"


def test_qwen3_235b_defaults_to_bumblebee(monkeypatch):
    monkeypatch.delenv("MINT_QWEN3_235B_TRAINING_BACKEND", raising=False)
    monkeypatch.delenv("MINT_MOE_TRAINING_BACKEND", raising=False)

    assert _select_moe_training_backend("Qwen/Qwen3-235B-A22B-Instruct-2507") == "bumblebee"


def test_bumblebee_30b_uses_folded_four_gpu_topology(monkeypatch):
    monkeypatch.delenv("MINT_QWEN3_30B_TRAINING_BACKEND", raising=False)
    engine = VerlTrainingEngine()

    cfg = engine._build_bumblebee_distributed_config(
        requested_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_model=None,
    )

    assert cfg.tensor_parallel_size == 4
    assert cfg.expert_parallel_size == 4
    assert cfg.expert_tensor_parallel_size == 1
    assert cfg.world_size == 4


def test_bumblebee_235b_keeps_existing_mint_topology_with_bumblebee_etp(monkeypatch):
    monkeypatch.delenv("MINT_QWEN3_235B_TRAINING_BACKEND", raising=False)
    engine = VerlTrainingEngine()

    cfg = engine._build_bumblebee_distributed_config(
        requested_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        base_model=None,
    )

    assert cfg.tensor_parallel_size == 4
    assert cfg.expert_parallel_size == 8
    assert cfg.expert_tensor_parallel_size is None
    assert cfg.world_size == 32
    assert _bumblebee_runtime_etp("Qwen/Qwen3-235B-A22B-Instruct-2507", cfg) == 1


def test_bumblebee_placement_group_name_is_namespace_scoped():
    assert (
        _make_bumblebee_pg_name("Qwen/Qwen3-30B-A3B-Instruct-2507", namespace="mint")
        == "mint_bumblebee_qwen3_30b_a3b_instruct_2507_mint_pg"
    )
    assert _make_bumblebee_pg_name(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        namespace="issue648_bumblebee_train9_20260528_102932",
    ).startswith("mint_bumblebee_qwen3_30b_a3b_instruct_2507_issue648_bumb")


def test_bumblebee_runtime_env_passthrough_includes_backend_knobs():
    assert "MINT_BUMBLEBEE_IMPL" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS
    assert "MINT_BUMBLEBEE_OPTIMIZER" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS
    assert "MINT_BUMBLEBEE_SKIP_HF_LOAD" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS
    assert "BUMBLEBEE_CKPT_TRACE" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS
    assert "NVTE_FLASH_ATTN" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS
    assert "BUMBLEBEE_TE_SDPA_FALLBACK" in BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS


def test_bumblebee_optimizer_metric_coerces_missing_num_zeros():
    assert _coerce_int(None) == 0


def _make_rank_worker_for_checkpoint_test():
    worker_cls = BumblebeeRankWorker.__ray_actor_class__
    worker = object.__new__(worker_cls)
    worker.rank = 0
    worker.world_size = 1
    worker.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    worker.lora_rank = 64
    worker.learning_rate = 1e-4
    worker._current_session = "session-a"
    worker._session_meta = {
        "session-a": BumblebeeSessionMeta(step_count=3, learning_rate=2e-5, actual_rank=16)
    }

    class FakeRuntime:
        def __init__(self):
            self.loaded = None

        def save_adapter_train_state(self, *args, **kwargs):
            return SimpleNamespace(
                adapter_id=kwargs.get("adapter_id", "session-a"),
                format="bumblebee_adapter_train_state_v1",
                tensors={"layer.lora_a": torch.tensor([1.0])},
                module_metadata={"layer": {"rank": 16}},
                optimizer_state={"state_dict": {"state": {0: {"exp_avg": torch.tensor([0.5])}}}},
                lr_scheduler_state={"last_epoch": 3},
                rng_state={"torch": torch.tensor([1, 2, 3], dtype=torch.uint8)},
                step=3,
                metadata=kwargs.get("metadata") or {},
                revision=7,
            )

        def load_adapter_train_state(self, handle, state, **kwargs):
            self.loaded = (state, kwargs)

    runtime = FakeRuntime()
    handle = SimpleNamespace(_extras={})
    worker._ensure_session_loaded = MethodType(lambda self, session_id, actual_rank: {}, worker)
    worker._require_runtime = MethodType(lambda self: (runtime, handle), worker)
    worker._reset_optimizer_state = MethodType(lambda self: None, worker)

    def fake_save_lora_adapter_artifacts(self, save_path, *, session_id, actual_rank, checkpoint_type):
        del session_id, actual_rank, checkpoint_type
        from safetensors.torch import save_file

        save_file({"layer.lora_a": torch.tensor([1.0])}, f"{save_path}/adapter_model.safetensors")
        return {"adapter_model": f"{save_path}/adapter_model.safetensors"}

    worker._save_lora_adapter_artifacts = MethodType(fake_save_lora_adapter_artifacts, worker)
    return worker, runtime


def _make_rank_worker_for_sft_loss_test():
    worker_cls = BumblebeeRankWorker.__ray_actor_class__
    worker = object.__new__(worker_cls)
    worker.rank = 0
    worker.world_size = 1
    worker.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    worker._current_session = None
    worker._session_meta = {}

    class FakeRuntime:
        def __init__(self):
            self.loss_fn = None
            self.forward_only = None

        def forward_backward(self, handle, batches, loss_fn, *, num_microbatches, forward_only=False):
            self.loss_fn = loss_fn
            self.forward_only = forward_only
            assert num_microbatches == 1
            assert batches
            return SimpleNamespace(
                metrics={
                    "loss": 3.0,
                    "loss:mean": 3.0,
                    "num_tokens:sum": 2.0,
                },
                model_output=SimpleNamespace(log_probs=torch.tensor([-10.0, -2.0, -4.0])),
            )

    runtime = FakeRuntime()
    worker._ensure_session_loaded = MethodType(
        lambda self, session_id, actual_rank: {"session_state": "loaded"},
        worker,
    )
    worker._require_runtime = MethodType(lambda self: (runtime, object()), worker)
    worker._mint_batch_to_runtime_dict = MethodType(lambda self, batch: {"loss_mask": batch.loss_mask}, worker)
    return worker, runtime


def test_bumblebee_unpads_thd_actor_logprobs_for_rl_adapter():
    worker_cls = BumblebeeRankWorker.__ray_actor_class__
    worker = object.__new__(worker_cls)

    class FakeBatch:
        actor_loss_mask = torch.ones(5)

        def sizes(self):
            return torch.tensor([2, 3], dtype=torch.int32)

    packed_seq_params = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    padded_logprobs = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0]])

    unpadded = worker._unpad_thd_actor_tensor_to_flat(
        padded_logprobs,
        batch=FakeBatch(),
        thd_loss_mask=torch.ones((1, 8)),
        packed_seq_params=packed_seq_params,
        name="actor_logprobs",
    )

    assert unpadded.shape == (5,)
    assert torch.equal(unpadded, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_bumblebee_forward_result_actor_logprobs_are_mutated_to_flat_layout():
    worker_cls = BumblebeeRankWorker.__ray_actor_class__
    worker = object.__new__(worker_cls)

    class FakeBatch:
        actor_loss_mask = torch.ones(3)

        def sizes(self):
            return torch.tensor([1, 2], dtype=torch.int32)

    result = SimpleNamespace(model_output=SimpleNamespace(log_probs=torch.tensor([[9.0, 0.0, 8.0, 7.0]])))
    runtime_batch = {
        "loss_mask": torch.ones((1, 4)),
        "packed_seq_params": SimpleNamespace(
            cu_seqlens_q_padded=torch.tensor([0, 2, 4], dtype=torch.int32),
        ),
    }

    worker._unpad_thd_forward_result_actor_outputs(result, batch=FakeBatch(), runtime_batch=runtime_batch)

    assert result.model_output.log_probs.shape == (3,)
    assert torch.equal(result.model_output.log_probs, torch.tensor([9.0, 8.0, 7.0]))


def test_bumblebee_worker_group_merges_only_numeric_tinker_metrics():
    group_cls = BumblebeeWorkerGroup.__ray_actor_class__
    group = object.__new__(group_cls)
    group.config = SimpleNamespace(world_size=4)

    payload = group._merge_rank_payloads(
        [
            {
                "loss_fn_outputs": [],
                "metrics": {
                    "loss": 1.25,
                    "loss:mean": 1.25,
                    "num_tokens:sum": 8,
                    "rank": 0,
                    "backend": "bumblebee",
                    "session_state": "restored",
                    "rank_metrics": [{"loss": 1.25}],
                },
            }
        ]
    )

    assert payload["metrics"] == {"loss:mean": 1.25, "num_tokens:sum": 8}


def test_issue_670_bumblebee_group_ready_checks_rank_workers():
    group_cls = BumblebeeWorkerGroup.__ray_actor_class__
    group = object.__new__(group_cls)
    group.config = SimpleNamespace(world_size=2)
    group.workers = [SimpleNamespace(__ray_ready__=SimpleNamespace(remote=lambda: "rank-ready-ref"))]

    with pytest.raises(RuntimeError, match="rank worker count mismatch"):
        group.__ray_ready__()


def test_issue_670_bumblebee_get_or_create_recreates_actor_when_rank_diagnostics_fail(monkeypatch):
    import mint_server.backend.bumblebee_distributed as bb

    stale_actor = SimpleNamespace()
    created_actor = object()
    kill_calls: list[dict] = []

    class _FakeRemoteOptions:
        def remote(self, **_kwargs):
            return created_actor

    class _FakeRemoteClass:
        def options(self, **_kwargs):
            return _FakeRemoteOptions()

    class _BrokenDiagnostics:
        def remote(self):
            return "broken-diagnostics-ref"

    monkeypatch.setattr(bb.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(bb.ray, "get_actor", lambda *_args, **_kwargs: stale_actor)

    def fake_ray_get(ref, timeout=None):
        assert ref == "broken-diagnostics-ref"
        assert timeout == 10
        raise bb.ray.exceptions.ActorDiedError()

    monkeypatch.setattr(bb.ray, "get", fake_ray_get)
    stale_actor.get_diagnostics = _BrokenDiagnostics()
    monkeypatch.setattr(bb.ray_kill, "kill", lambda *args, **kwargs: kill_calls.append(kwargs))
    monkeypatch.setattr(bb, "BumblebeeWorkerGroup", _FakeRemoteClass())
    monkeypatch.setattr(bb, "actor_runtime_env_vars", lambda **_kwargs: {})
    monkeypatch.setattr(
        "mint_server.backend.model_actor_publication.publish_backend_model_actor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(bb, "is_topology_desired_model", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bb, "_model_gpu_placement_for_model", lambda *_args, **_kwargs: None)

    actor = get_or_create_bumblebee_worker_group(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=64,
        learning_rate=1e-4,
        observability_base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    assert actor is created_actor
    assert kill_calls == [
        {
            "reason": "bumblebee_actor_rank_worker_unhealthy",
            "actor_name": "mint_bumblebee_qwen3_30b_a3b_instruct_2507",
            "namespace": bb.PERSISTENT_NAMESPACE,
            "no_restart": True,
            "verify_absent": True,
        }
    ]


def test_issue_670_bumblebee_get_or_create_keeps_actor_when_diagnostics_timeout(monkeypatch):
    import mint_server.backend.bumblebee_distributed as bb

    busy_actor = SimpleNamespace()
    kill_calls: list[dict] = []
    published: list[object] = []

    class _BusyDiagnostics:
        def remote(self):
            return "busy-diagnostics-ref"

    monkeypatch.setattr(bb.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(bb.ray, "get_actor", lambda *_args, **_kwargs: busy_actor)

    def fake_ray_get(ref, timeout=None):
        assert ref == "busy-diagnostics-ref"
        assert timeout == 10
        raise bb.ray.exceptions.GetTimeoutError("busy")

    monkeypatch.setattr(bb.ray, "get", fake_ray_get)
    busy_actor.get_diagnostics = _BusyDiagnostics()
    monkeypatch.setattr(bb.ray_kill, "kill", lambda *args, **kwargs: kill_calls.append(kwargs))
    monkeypatch.setattr(
        "mint_server.backend.model_actor_publication.publish_backend_model_actor",
        lambda launch, **_kwargs: published.append(launch),
    )
    monkeypatch.setattr(bb, "is_topology_desired_model", lambda *_args, **_kwargs: False)

    actor = get_or_create_bumblebee_worker_group(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=64,
        learning_rate=1e-4,
        observability_base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    assert actor is busy_actor
    assert kill_calls == []
    assert len(published) == 1


def test_bumblebee_checkpoint_save_writes_optimizer_backed_train_state(tmp_path):
    worker, _runtime = _make_rank_worker_for_checkpoint_test()

    meta = worker.save_training_state(
        str(tmp_path),
        session_id="session-a",
        actual_rank=16,
        include_optimizer=True,
    )

    state_path = tmp_path / "rank_00000" / BUMBLEBEE_TRAIN_STATE_FILE
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    assert payload["format"] == BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT
    assert payload["optimizer_state"] is not None
    assert payload["lr_scheduler_state"] is not None
    assert payload["rng_state"] is not None
    assert meta["optimizer_restored"] is True
    assert (tmp_path / BUMBLEBEE_TRAIN_STATE_META_FILE).exists()
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert checkpoint_has_optimizer_state(str(tmp_path)) is True
    validate_checkpoint_dir(str(tmp_path), checkpoint_type="training")


def test_bumblebee_checkpoint_load_restores_training_state(tmp_path):
    worker, runtime = _make_rank_worker_for_checkpoint_test()
    worker.save_training_state(
        str(tmp_path),
        session_id="session-a",
        actual_rank=16,
        include_optimizer=True,
    )

    meta = worker.load_training_state(
        str(tmp_path),
        load_optimizer=True,
        session_id="session-b",
        actual_rank=16,
    )

    loaded_state, flags = runtime.loaded
    assert loaded_state.adapter_id == "session-b"
    assert loaded_state.optimizer_state is not None
    assert flags["restore_optimizer"] is True
    assert flags["restore_lr_scheduler"] is True
    assert flags["restore_rng"] is True
    assert meta["optimizer_restored"] is True


def test_issue_662_bumblebee_loads_megatron_peft_adapter_as_weights_only_migration(
    tmp_path,
    monkeypatch,
):
    worker, runtime = _make_rank_worker_for_checkpoint_test()
    worker._session_meta = {}
    reset_calls = []
    worker._reset_optimizer_state = MethodType(lambda self: reset_calls.append(True), worker)
    handle = SimpleNamespace(
        _model=object(),
        _extras={
            "model_chunks": ["chunk"],
            "model_cfg": object(),
            "parallel_state": object(),
        },
    )
    worker._require_runtime = MethodType(lambda self: (runtime, handle), worker)

    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "backend": "megatron",
                "checkpoint_type": "training",
                "step": 75,
                "learning_rate": 1e-5,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_meta.json").write_text(
        json.dumps(
            {
                "current_step": 76,
                "learning_rate": 4e-5,
                "actual_rank": 32,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter-rank")
    (tmp_path / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    fake_lora_adapter = types.ModuleType("bumblebee.model.qwen3_moe.lite.lora_adapter")
    load_calls = []

    def fake_load_lora_adapter(chunks, adapter_dir, model_cfg, parallel_state, *, strict):
        load_calls.append(
            {
                "chunks": chunks,
                "adapter_dir": adapter_dir,
                "model_cfg": model_cfg,
                "parallel_state": parallel_state,
                "strict": strict,
            }
        )
        return {"loaded_tensors": 123}

    fake_lora_adapter.load_lora_adapter = fake_load_lora_adapter
    monkeypatch.setitem(sys.modules, "bumblebee.model.qwen3_moe.lite.lora_adapter", fake_lora_adapter)

    meta = worker.load_training_state(
        str(tmp_path),
        load_optimizer=True,
        session_id="session-b",
        actual_rank=32,
    )

    assert load_calls and load_calls[0]["strict"] is True
    assert reset_calls == [True]
    assert runtime.loaded is None
    assert meta["migration_source_backend"] == "megatron"
    assert meta["migration_target_backend"] == "bumblebee"
    assert meta["migration_mode"] == "weights_only"
    assert meta["optimizer_restored"] is False
    assert meta["optimizer_reset"] is True
    assert meta["requested_optimizer_restore"] is True
    assert meta["current_step"] == 76
    assert meta["learning_rate"] == 4e-5
    assert meta["loaded_tensors"] == 123


def test_issue_662_bumblebee_rejects_megatron_adapter_rank_mismatch(tmp_path):
    worker, _runtime = _make_rank_worker_for_checkpoint_test()
    (tmp_path / "metadata.json").write_text(
        json.dumps({"backend": "megatron", "checkpoint_type": "training"}),
        encoding="utf-8",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter-rank")

    with pytest.raises(RuntimeError, match="requires matching LoRA rank"):
        worker.load_training_state(
            str(tmp_path),
            load_optimizer=True,
            session_id="session-b",
            actual_rank=32,
        )


def test_bumblebee_sft_forward_backward_uses_model_masked_loss(monkeypatch):
    worker, runtime = _make_rank_worker_for_sft_loss_test()

    class FakeBatch:
        loss_mask = torch.tensor([0.0, 1.0, 1.0])

    mint_module = types.ModuleType("bumblebee.runtime.adapters.mint")
    mint_module.actor_update_output_to_mint_forward_backward = lambda *args, **kwargs: {}
    mint_module.make_mint_actor_loss_fn = lambda *args, **kwargs: None
    mint_module.mint_datums_to_packed_batch = lambda data_items, *, loss_fn, device: FakeBatch()
    monkeypatch.setitem(sys.modules, "bumblebee", types.ModuleType("bumblebee"))
    monkeypatch.setitem(sys.modules, "bumblebee.runtime", types.ModuleType("bumblebee.runtime"))
    monkeypatch.setitem(sys.modules, "bumblebee.runtime.adapters", types.ModuleType("bumblebee.runtime.adapters"))
    monkeypatch.setitem(sys.modules, "bumblebee.runtime.adapters.mint", mint_module)
    megatron_training_module = types.ModuleType("mint_server.backend.megatron_training")
    megatron_training_module.benchmark_debug_input_entries = lambda data_items: []
    monkeypatch.setitem(sys.modules, "mint_server.backend.megatron_training", megatron_training_module)

    payload = worker.forward_backward(
        [{"input_ids": [1, 2, 3]}],
        loss_fn="cross_entropy",
        loss_fn_config={},
        rollout_correction_config=None,
        session_id="session-a",
        actual_rank=16,
    )

    assert runtime.loss_fn is None
    assert runtime.forward_only is False
    assert payload["loss_fn_outputs"][0]["loss"]["data"] == [3.0]
    assert payload["metrics"]["loss"] == 3.0
    assert payload["metrics"]["num_tokens:sum"] == 2.0
