from __future__ import annotations


def test_issue_439_queue_execution_runtime_forwards_actor_name_overrides(monkeypatch) -> None:
    from tinker_server.backend import queue_execution_runtime as qer

    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "queue-v20260309")
    monkeypatch.setenv("MINT_FUTURE_STORE_ACTOR_NAME", "future-v2")
    monkeypatch.setenv("TINKER_CAPACITY_MANAGER_ACTOR_NAME", "capacity-v3")
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-issue-439")

    out = qer._runtime_env_overrides()

    assert out["TINKER_API_WORK_QUEUE_ACTOR_NAME"] == "queue-v20260309"
    assert out["MINT_FUTURE_STORE_ACTOR_NAME"] == "future-v2"
    assert out["TINKER_CAPACITY_MANAGER_ACTOR_NAME"] == "capacity-v3"
    assert out["TINKER_RAY_NAMESPACE"] == "ns-issue-439"
