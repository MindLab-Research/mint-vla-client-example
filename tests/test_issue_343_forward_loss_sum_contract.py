from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("ray")

from tinker_server.backend.megatron_distributed import DistributedConfig, MegatronRankWorker


class _FakeEvalMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return None


class _FakeEngine:
    def eval_mode(self):
        return _FakeEvalMode()

    def forward_backward_batch(self, *, data, loss_function, forward_only):
        _ = (data, loss_function)
        assert forward_only is True
        return {
            "loss": [3.5],
            "metrics": {
                "num_tokens": [2],
                "loss_sum": [7.0],
                "log_probs": [],
            },
            "model_output": {
                "log_probs": torch.tensor([[-2.0, -3.0]], dtype=torch.float32),
            },
        }


def test_issue_343_megatron_forward_preserves_loss_sum_contract(monkeypatch) -> None:
    impl_cls = MegatronRankWorker.__ray_metadata__.modified_class
    worker = impl_cls(
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=12345,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        learning_rate=1e-4,
        distributed_config=DistributedConfig(),
    )
    worker.engine = _FakeEngine()
    worker._bind_traceparent = lambda traceparent: None
    worker._resolve_reset_bias = lambda val, default: False
    worker._is_output_rank = lambda: True
    worker.log_memory_breakdown = lambda tag: None

    fake_training = types.ModuleType("tinker_server.backend.megatron_training")
    fake_training.tinker_to_tensordict = lambda *args, **kwargs: "fake_tensordict"  # type: ignore[attr-defined]
    fake_training.create_logprob_extractor_fn = lambda: "fake_loss_fn"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinker_server.backend.megatron_training", fake_training)

    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.get_model_config",
        lambda model: types.SimpleNamespace(max_model_len=2048),
    )
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.flatten_encoded_text_chunks",
        lambda model_input: list(model_input["chunks"][0]["tokens"]),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    data_items = [
        {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [10, 11]}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": [11, 12], "shape": [2], "dtype": "int64"},
                "weights": {"data": [2.0, 1.0], "shape": [2], "dtype": "float32"},
            },
        }
    ]

    result = worker.forward(data_items)

    logprobs = result["loss_fn_outputs"][0]["logprobs"]["data"]
    weights = data_items[0]["loss_fn_inputs"]["weights"]["data"]
    expected_sum = -sum(lp * wt for lp, wt in zip(logprobs, weights))

    assert result["loss_sum_value"] == pytest.approx(expected_sum)
    assert result["loss_value"] == pytest.approx(result["loss_sum_value"] / result["num_tokens"])
    assert result["num_tokens"] == 2
    assert result["valid_count"] == 1
