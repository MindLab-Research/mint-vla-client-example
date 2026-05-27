import sys
import types
from types import MethodType, SimpleNamespace

import torch

from mint_server.backend.verl_training import (
    VerlTrainingEngine,
    _select_moe_training_backend,
)
from mint_server.backend.bumblebee_distributed import (
    BUMBLEBEE_TRAIN_STATE_CHECKPOINT_FORMAT,
    BUMBLEBEE_TRAIN_STATE_FILE,
    BUMBLEBEE_TRAIN_STATE_META_FILE,
    BumblebeeRankWorker,
    BumblebeeWorkerGroup,
    BumblebeeSessionMeta,
    _bumblebee_runtime_etp,
    _coerce_int,
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
            loss, metrics = loss_fn(
                {"log_probs": torch.tensor([-10.0, -2.0, -4.0])},
                {"loss_mask": torch.tensor([0.0, 1.0, 1.0])},
            )
            assert num_microbatches == 1
            assert batches
            return SimpleNamespace(metrics={**metrics, "loss": float(loss.detach().item())})

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


def test_bumblebee_sft_forward_backward_uses_masked_external_loss(monkeypatch):
    worker, runtime = _make_rank_worker_for_sft_loss_test()

    class FakeBatch:
        loss_mask = torch.tensor([0.0, 1.0, 1.0])

    def fake_sft_loss_fn():
        def loss_fn(model_output, batch):
            loss_mask = batch["loss_mask"]
            log_probs = model_output["log_probs"]
            loss_sum = -(log_probs.float() * loss_mask.float()).sum()
            num_tokens = loss_mask.float().sum()
            loss = loss_sum / num_tokens
            return loss, {
                "loss": float(loss.detach().item()),
                "loss:mean": float(loss.detach().item()),
                "sft_loss_sum": float(loss_sum.detach().item()),
                "num_tokens:sum": float(num_tokens.detach().item()),
            }

        return loss_fn

    mint_module = types.ModuleType("bumblebee.runtime.adapters.mint")
    mint_module.actor_update_output_to_mint_forward_backward = lambda *args, **kwargs: {}
    mint_module.make_mint_actor_loss_fn = lambda *args, **kwargs: None
    mint_module.make_mint_sft_loss_fn = fake_sft_loss_fn
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

    assert runtime.loss_fn is not None
    assert runtime.forward_only is False
    assert payload["loss_fn_outputs"][0]["loss"]["data"] == [3.0]
    assert payload["metrics"]["loss"] == 3.0
    assert payload["metrics"]["sft_loss_sum"] == 6.0
    assert payload["metrics"]["num_tokens:sum"] == 2.0
