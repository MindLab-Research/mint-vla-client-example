from __future__ import annotations

import pytest

pytest.importorskip("ray")

import mint_server.backend.multinode_inference as mi


def test_issue_686_multinode_fully_sharded_lora_env_defaults_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_VLLM_FULLY_SHARDED_LORAS", raising=False)

    assert (
        mi._resolve_fully_sharded_loras_env_value(
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=16,
        )
        == "1"
    )


def test_issue_686_multinode_fully_sharded_lora_env_disables_when_not_divisible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_VLLM_FULLY_SHARDED_LORAS", raising=False)

    assert (
        mi._resolve_fully_sharded_loras_env_value(
            enable_lora=True,
            max_lora_rank=8,
            tensor_parallel_size=16,
        )
        == "0"
    )


def test_issue_686_multinode_fully_sharded_lora_env_respects_operator_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_VLLM_FULLY_SHARDED_LORAS", "0")

    assert (
        mi._resolve_fully_sharded_loras_env_value(
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=16,
        )
        == "0"
    )


def test_issue_686_multinode_fully_sharded_lora_env_rejects_invalid_force_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_VLLM_FULLY_SHARDED_LORAS", "1")

    assert (
        mi._resolve_fully_sharded_loras_env_value(
            enable_lora=True,
            max_lora_rank=8,
            tensor_parallel_size=16,
        )
        == "0"
    )


def test_issue_686_multinode_runtime_env_writes_derived_fully_sharded_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_VLLM_FULLY_SHARDED_LORAS", raising=False)
    env_vars = {"MINT_VLLM_FULLY_SHARDED_LORAS": "0"}

    mi._set_multinode_fully_sharded_loras_env(
        env_vars,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=16,
    )

    assert env_vars["MINT_VLLM_FULLY_SHARDED_LORAS"] == "1"
