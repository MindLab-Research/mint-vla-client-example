from __future__ import annotations

import pytest

from mint_server.backend.training.megatron.megatron_training import mint_datum_to_tensordict


def _tensor(data: list[float] | list[int], dtype: str = "float32") -> dict:
    return {"data": data, "shape": [len(data)], "dtype": dtype}


def test_megatron_rl_loss_mask_uses_causal_response_positions():
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [10, 11, 12, 13]}]},
        "loss_fn_inputs": {
            "target_tokens": _tensor([11, 12, 13, 14], "int64"),
            "weights": _tensor([0.0, 1.0, 1.0, 1.0]),
            "logprobs": _tensor([-1.0, -2.0, -3.0, -4.0]),
            "advantages": _tensor([0.0, 0.1, 0.2, 0.3]),
        },
    }

    td = mint_datum_to_tensordict([datum], device="cpu")

    assert td["loss_mask"].to_padded_tensor(0.0).tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert td["response_mask"].tolist() == [[1.0, 1.0]]
    assert td["old_log_probs"].tolist() == [[-2.0, -3.0]]
    assert td["advantages"][0].tolist() == pytest.approx([0.1, 0.2])
    assert float(td["batch_num_tokens"]) == 2.0
