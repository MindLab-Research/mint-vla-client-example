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
