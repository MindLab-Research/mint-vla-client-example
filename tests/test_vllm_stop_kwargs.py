import pytest

from tinker_server.backend.vllm_stop import vllm_stop_kwargs


def test_vllm_stop_kwargs_defaults() -> None:
    assert vllm_stop_kwargs(None, default_stop_token_ids=[1, 2]) == {"stop_token_ids": [1, 2]}
    assert vllm_stop_kwargs(None, default_stop_token_ids=None) == {}


def test_vllm_stop_kwargs_strings() -> None:
    assert vllm_stop_kwargs("", default_stop_token_ids=[1]) == {}
    assert vllm_stop_kwargs("x", default_stop_token_ids=[1]) == {"stop": "x"}
    assert vllm_stop_kwargs(["x", ""], default_stop_token_ids=[1]) == {"stop": ["x"]}


def test_vllm_stop_kwargs_token_ids() -> None:
    assert vllm_stop_kwargs([], default_stop_token_ids=[1]) == {}
    assert vllm_stop_kwargs([3, 4], default_stop_token_ids=[1]) == {"stop_token_ids": [3, 4]}


def test_vllm_stop_kwargs_rejects_mixed_lists() -> None:
    with pytest.raises(ValueError):
        vllm_stop_kwargs([1, "x"], default_stop_token_ids=[1])  # type: ignore[list-item]


def test_vllm_stop_kwargs_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        vllm_stop_kwargs(123, default_stop_token_ids=[1])  # type: ignore[arg-type]

