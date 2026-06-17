import pytest

from mint_server.utils.sampling_utils import normalize_prompt_logprobs_for_tinker


def test_prompt_logprobs_len_equals_prompt_len_sets_first_none() -> None:
    out = normalize_prompt_logprobs_for_tinker([0.1, 0.2, 0.3], prompt_len=3)
    assert out == [None, 0.2, 0.3]


def test_prompt_logprobs_len_prompt_len_minus_one_prepends_none() -> None:
    out = normalize_prompt_logprobs_for_tinker([0.2, 0.3], prompt_len=3)
    assert out == [None, 0.2, 0.3]


def test_prompt_logprobs_len_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        normalize_prompt_logprobs_for_tinker([None], prompt_len=3)


def test_prompt_logprobs_empty_prompt() -> None:
    assert normalize_prompt_logprobs_for_tinker([], prompt_len=0) == []
    with pytest.raises(ValueError):
        normalize_prompt_logprobs_for_tinker([None], prompt_len=0)

