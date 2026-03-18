from __future__ import annotations

from pathlib import Path

import pytest

from tinker_server.backend.verl_inference import _extract_sampled_token_logprobs


class _LogProb:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def test_issue_355_extract_sampled_token_logprobs_accepts_common_vllm_entry_shapes() -> None:
    result = _extract_sampled_token_logprobs(
        request_id="req-355",
        token_ids=[17, 0, 5],
        step_logprobs=[
            {17: _LogProb(-0.1)},
            {0: {"logprob": -0.2}},
            {5: -0.3},
        ],
    )

    assert result == [-0.1, -0.2, -0.3]


def test_issue_355_extract_sampled_token_logprobs_raises_explicit_error_for_missing_sampled_token() -> None:
    with pytest.raises(RuntimeError, match=r"missing sampled-token logprob: request_id=req-355 idx=1 token_id=0"):
        _extract_sampled_token_logprobs(
            request_id="req-355",
            token_ids=[17, 0],
            step_logprobs=[
                {17: _LogProb(-0.1)},
                {1: _LogProb(-0.2)},
            ],
        )


def test_issue_355_extract_sampled_token_logprobs_raises_when_payload_missing_entirely() -> None:
    with pytest.raises(RuntimeError, match=r"returned no sampled-token logprob payload: request_id=req-355 token_count=2"):
        _extract_sampled_token_logprobs(
            request_id="req-355",
            token_ids=[17, 0],
            step_logprobs=None,
        )


def test_issue_355_extract_sampled_token_logprobs_raises_when_payload_length_mismatches() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"mismatched sampled-token logprob payload length: request_id=req-355 token_count=2 logprob_count=1",
    ):
        _extract_sampled_token_logprobs(
            request_id="req-355",
            token_ids=[17, 0],
            step_logprobs=[{17: _LogProb(-0.1)}],
        )


def test_issue_355_verl_requests_positive_sampled_token_logprob_budget() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "tinker_server/backend/verl_inference.py").read_text(encoding="utf-8")

    assert "logprobs=0 if logprobs else None" not in text
    assert 'sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None' not in text
    assert text.count("logprobs=1 if logprobs else None") >= 2
    assert 'sampling_params["logprobs"] = 1 if sampling_params.pop("logprobs", False) else None' in text
