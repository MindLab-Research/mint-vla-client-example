from __future__ import annotations


def test_issue_439_queue_execution_runtime_forwards_actor_name_and_placement_overrides(monkeypatch) -> None:
    from tinker_server.backend import queue_execution_runtime as qer

    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "queue-v20260309")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY", "1024")
    monkeypatch.setenv("MINT_FUTURE_STORE_ACTOR_NAME", "future-v2")
    monkeypatch.setenv("TINKER_CAPACITY_MANAGER_ACTOR_NAME", "capacity-v3")
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-issue-439")
    monkeypatch.setenv("MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID", "rq-k2")
    monkeypatch.setenv("MINT_VLLM_VOLC_RESOURCE_QUEUE_ID", "rq-vllm")
    monkeypatch.setenv("MINT_MODEL_NODE_IPS_JSON", '{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.38.175"]}')
    monkeypatch.setenv(
        "MINT_MEGATRON_MODEL_NODE_IPS_JSON",
        '{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.38.175"]}',
    )
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GLOO_TIMEOUT_S", "123")
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GATHER_DEBUG", "1")
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GLOO_BARRIER_DEBUG", "1")

    out = qer._runtime_env_overrides()

    assert out["MINT_API_WORK_QUEUE_ACTOR_NAME"] == "queue-v20260309"
    assert out["MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY"] == "1024"
    assert out["MINT_FUTURE_STORE_ACTOR_NAME"] == "future-v2"
    assert out["MINT_CAPACITY_MANAGER_ACTOR_NAME"] == "capacity-v3"
    assert out["TINKER_RAY_NAMESPACE"] == "ns-issue-439"
    assert out["MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID"] == "rq-k2"
    assert out["MINT_VLLM_VOLC_RESOURCE_QUEUE_ID"] == "rq-vllm"
    assert out["MINT_MODEL_NODE_IPS_JSON"] == '{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.38.175"]}'
    assert out["MINT_MEGATRON_MODEL_NODE_IPS_JSON"] == '{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.38.175"]}'
    assert out["MINT_MBRIDGE_EXPORT_GLOO_TIMEOUT_S"] == "123"
    assert out["MINT_MBRIDGE_EXPORT_GATHER_DEBUG"] == "1"
    assert out["MINT_MBRIDGE_EXPORT_GLOO_BARRIER_DEBUG"] == "1"
