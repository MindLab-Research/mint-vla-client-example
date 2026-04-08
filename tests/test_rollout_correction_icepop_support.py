from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def test_rollout_correction_request_accepts_icepop_threshold_string() -> None:
    from tinker_server.models.types import CreateModelRequest

    request = CreateModelRequest.model_validate(
        {
            "session_id": "icepop-session",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "rollout_correction_config": {
                "rollout_is": "token",
                "rollout_is_threshold": "0.5_5.0",
                "bypass_mode": True,
                "loss_type": "reinforce",
            },
        }
    )

    assert request.rollout_correction_config is not None
    assert request.rollout_correction_config.rollout_is_threshold == "0.5_5.0"


def test_create_ppo_loss_fn_bypass_mode_uses_current_rollout_helper_signature(monkeypatch) -> None:
    helper_calls: dict[str, object] = {}

    ray = types.ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda obj: obj)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)

    tensordict_mod = types.ModuleType("tensordict")
    tensordict_mod.TensorDict = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensordict", tensordict_mod)

    tensorclass_mod = types.ModuleType("tensordict.tensorclass")

    class NonTensorData:
        def __init__(self, value):
            self.value = value

    tensorclass_mod.NonTensorData = NonTensorData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensordict.tensorclass", tensorclass_mod)

    verl_mod = types.ModuleType("verl")
    verl_utils_mod = types.ModuleType("verl.utils")
    verl_tu_mod = types.ModuleType("verl.utils.tensordict_utils")
    verl_utils_mod.tensordict_utils = verl_tu_mod  # type: ignore[attr-defined]
    verl_mod.utils = verl_utils_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl", verl_mod)
    monkeypatch.setitem(sys.modules, "verl.utils", verl_utils_mod)
    monkeypatch.setitem(sys.modules, "verl.utils.tensordict_utils", verl_tu_mod)

    trainer_mod = types.ModuleType("verl.trainer")
    trainer_config_mod = types.ModuleType("verl.trainer.config")
    trainer_ppo_mod = types.ModuleType("verl.trainer.ppo")
    trainer_ppo_core_mod = types.ModuleType("verl.trainer.ppo.core_algos")
    trainer_rollout_corr_mod = types.ModuleType("verl.trainer.ppo.rollout_corr_helper")
    workers_mod = types.ModuleType("verl.workers")
    workers_config_mod = types.ModuleType("verl.workers.config")

    class _ConfigDict(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self.__dict__.update(kwargs)

        def get(self, key, default=None):
            return getattr(self, key, default)

    class RolloutCorrectionConfig(_ConfigDict):
        pass

    class PolicyLossConfig(_ConfigDict):
        pass

    class ActorConfig:
        def __init__(self, **kwargs):
            self.loss_agg_mode = kwargs.pop("loss_agg_mode", "token-mean")
            self.loss_scale_factor = kwargs.pop("loss_scale_factor", 1.0)
            for key, value in kwargs.items():
                setattr(self, key, value)
            if not hasattr(self, "policy_loss"):
                self.policy_loss = PolicyLossConfig()

        def get(self, key, default=None):
            return getattr(self, key, default)

    def agg_loss(
        *,
        loss_mat,
        loss_mask,
        loss_agg_mode,
        dp_size,
        batch_num_tokens,
        global_batch_size,
        loss_scale_factor,
    ):
        _ = (loss_agg_mode, dp_size, batch_num_tokens, global_batch_size, loss_scale_factor)
        denom = loss_mask.sum().clamp(min=1.0)
        return (loss_mat * loss_mask).sum() / denom

    def compute_rollout_correction_and_rejection_mask(
        *,
        old_log_prob,
        rollout_log_prob,
        response_mask,
        rollout_is=None,
        rollout_is_threshold=2.0,
        rollout_rs=None,
        rollout_rs_threshold=None,
        rollout_is_batch_normalize=False,
    ):
        helper_calls["kwargs"] = {
            "old_log_prob": old_log_prob.clone(),
            "rollout_log_prob": rollout_log_prob.clone(),
            "response_mask": response_mask.clone(),
            "rollout_is": rollout_is,
            "rollout_is_threshold": rollout_is_threshold,
            "rollout_rs": rollout_rs,
            "rollout_rs_threshold": rollout_rs_threshold,
            "rollout_is_batch_normalize": rollout_is_batch_normalize,
        }
        proto = SimpleNamespace(batch={"rollout_is_weights": torch.tensor([[1.0, 0.0]], dtype=torch.float32)})
        return proto, response_mask.clone(), {}

    trainer_config_mod.RolloutCorrectionConfig = RolloutCorrectionConfig  # type: ignore[attr-defined]
    trainer_ppo_core_mod.agg_loss = agg_loss  # type: ignore[attr-defined]
    trainer_rollout_corr_mod.compute_rollout_correction_and_rejection_mask = (  # type: ignore[attr-defined]
        compute_rollout_correction_and_rejection_mask
    )
    workers_config_mod.ActorConfig = ActorConfig  # type: ignore[attr-defined]
    workers_config_mod.PolicyLossConfig = PolicyLossConfig  # type: ignore[attr-defined]

    trainer_ppo_mod.core_algos = trainer_ppo_core_mod  # type: ignore[attr-defined]
    trainer_ppo_mod.rollout_corr_helper = trainer_rollout_corr_mod  # type: ignore[attr-defined]
    trainer_mod.config = trainer_config_mod  # type: ignore[attr-defined]
    trainer_mod.ppo = trainer_ppo_mod  # type: ignore[attr-defined]
    workers_mod.config = workers_config_mod  # type: ignore[attr-defined]
    verl_mod.trainer = trainer_mod  # type: ignore[attr-defined]
    verl_mod.workers = workers_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "verl.trainer", trainer_mod)
    monkeypatch.setitem(sys.modules, "verl.trainer.config", trainer_config_mod)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", trainer_ppo_mod)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", trainer_ppo_core_mod)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.rollout_corr_helper", trainer_rollout_corr_mod)
    monkeypatch.setitem(sys.modules, "verl.workers", workers_mod)
    monkeypatch.setitem(sys.modules, "verl.workers.config", workers_config_mod)

    module_name = "test_megatron_training_icepop_support"
    module_path = Path(__file__).resolve().parents[1] / "tinker_server" / "backend" / "megatron_training.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    loss_fn = module.create_ppo_loss_fn(
        epsilon=0.2,
        rollout_correction_config={
            "rollout_is": "token",
            "rollout_is_threshold": "0.5_5.0",
            "rollout_rs": "seq_sum_k1",
            "rollout_rs_threshold": "0.5_2.0",
            "bypass_mode": True,
            "loss_type": "reinforce",
        },
    )

    loss, metrics = loss_fn(
        {"response_log_probs": torch.tensor([[0.3, -0.2]], dtype=torch.float32)},
        {
            "old_log_probs": torch.tensor([[0.1, -0.1]], dtype=torch.float32),
            "advantages": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "responses": torch.tensor([[11, 12]], dtype=torch.long),
            "dp_size": 1,
            "batch_num_tokens": 2,
            "global_batch_size": 1,
        },
    )

    assert torch.isfinite(loss)
    assert metrics["num_tokens"] == 2
    assert helper_calls["kwargs"]["rollout_is_threshold"] == "0.5_5.0"
    assert helper_calls["kwargs"]["rollout_rs_threshold"] == "0.5_2.0"
    assert helper_calls["kwargs"]["rollout_is_batch_normalize"] is False
