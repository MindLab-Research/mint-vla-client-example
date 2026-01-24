from tinker_server.sampling_utils import resolve_stop_reason


def test_issue_20_prefers_explicit_stop_reason() -> None:
    assert resolve_stop_reason(stop_reason="stop", token_ids=[1, 2, 3]) == "stop"
    assert resolve_stop_reason(stop_reason="length", token_ids=[1, 2, 3]) == "length"
    assert resolve_stop_reason(stop_reason="eos", token_ids=[1, 2, 3]) == "eos"


def test_issue_20_falls_back_to_eos_heuristic() -> None:
    assert resolve_stop_reason(stop_reason=None, token_ids=[151645]) == "stop"
    assert resolve_stop_reason(stop_reason="unknown", token_ids=[151643]) == "stop"
    assert resolve_stop_reason(stop_reason=None, token_ids=[42]) == "length"

