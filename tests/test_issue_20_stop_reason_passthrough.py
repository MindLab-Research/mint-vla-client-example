from mint_server.utils.sampling_utils import resolve_stop_reason, sampled_sequence_from_result


def test_issue_20_prefers_explicit_stop_reason() -> None:
    assert resolve_stop_reason(stop_reason="stop", token_ids=[1, 2, 3]) == "stop"
    assert resolve_stop_reason(stop_reason="length", token_ids=[1, 2, 3]) == "length"
    assert resolve_stop_reason(stop_reason="eos", token_ids=[1, 2, 3]) == "eos"


def test_issue_20_falls_back_to_eos_heuristic() -> None:
    assert resolve_stop_reason(stop_reason=None, token_ids=[151645]) == "stop"
    assert resolve_stop_reason(stop_reason="unknown", token_ids=[151643]) == "stop"
    assert resolve_stop_reason(stop_reason=None, token_ids=[42]) == "length"


def test_issue_20_sampled_sequence_from_result_passthrough() -> None:
    class Dummy:
        def __init__(self, *, token_ids, stop_reason, logprobs=None, log_probs=None):
            self.token_ids = token_ids
            self.stop_reason = stop_reason
            if logprobs is not None:
                self.logprobs = logprobs
            if log_probs is not None:
                self.log_probs = log_probs

    seq = sampled_sequence_from_result(
        Dummy(token_ids=[1, 2, 3], stop_reason="stop", log_probs=[-1.0, -2.0, -3.0])
    )
    assert seq.stop_reason == "stop"
    assert seq.logprobs == [-1.0, -2.0, -3.0]
