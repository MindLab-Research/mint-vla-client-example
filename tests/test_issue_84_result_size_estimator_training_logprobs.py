from __future__ import annotations

from tinker_server.backend.result_size_estimator import estimate_forward_result_bytes
from tinker_server.models.types import Datum, ForwardBackwardInput, ForwardRequest, ModelInput


def _mk_forward_req(*, prompt_tokens: int, target_tokens: int | None) -> ForwardRequest:
    datum = Datum(
        model_input=ModelInput.from_ints(list(range(prompt_tokens))),
        loss_fn_inputs=(
            {}
            if target_tokens is None
            else {"target_tokens": {"data": list(range(target_tokens))}}
        ),
    )
    return ForwardRequest(
        model_id="m",
        forward_input=ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
    )


def test_estimate_forward_result_bytes_uses_target_tokens_when_present() -> None:
    req = _mk_forward_req(prompt_tokens=10, target_tokens=9)
    # raw = 8192 + total_targets*64 + num_items*4096 (see result_size_estimator.py)
    assert estimate_forward_result_bytes(req) == 8192 + 9 * 64 + 1 * 4096


def test_estimate_forward_result_bytes_falls_back_to_model_input_len() -> None:
    req = _mk_forward_req(prompt_tokens=10, target_tokens=None)
    assert estimate_forward_result_bytes(req) == 8192 + 10 * 64 + 1 * 4096

