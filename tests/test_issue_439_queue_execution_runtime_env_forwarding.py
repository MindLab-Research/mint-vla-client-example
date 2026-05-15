from __future__ import annotations


def test_issue_439_queue_execution_runtime_records_actor_name_and_placement_overrides(monkeypatch) -> None:
    from tinker_server.backend import queue_execution_runtime as qer
    from tinker_server.runtime_config import actor_env_from_environ

    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "queue-v20260309")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY", "1024")
    monkeypatch.setenv("MINT_FUTURE_STORE_ACTOR_NAME", "future-v2")
    monkeypatch.setenv("TINKER_CAPACITY_MANAGER_ACTOR_NAME", "capacity-v3")
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-issue-439")
    monkeypatch.setenv("MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID", "rq-k2")
    monkeypatch.setenv("MINT_VLLM_VOLC_RESOURCE_QUEUE_ID", "rq-vllm")
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", '{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"worker_index":1,"gpu_count":4}}')
    monkeypatch.setenv(
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON",
        '{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"worker_index":1,"gpu_count":4}}',
    )
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GLOO_TIMEOUT_S", "123")
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GATHER_DEBUG", "1")
    monkeypatch.setenv("MINT_MBRIDGE_EXPORT_GLOO_BARRIER_DEBUG", "1")

    out = qer._runtime_env_overrides()
    actor_env = actor_env_from_environ(__import__("os").environ)

    assert out["MINT_API_WORK_QUEUE_ACTOR_NAME"] == "queue-v20260309"
    assert out["MINT_CAPACITY_MANAGER_ACTOR_NAME"] == "capacity-v3"
    assert "MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY" not in out
    assert "MINT_FUTURE_STORE_ACTOR_NAME" not in out
    assert "TINKER_RAY_NAMESPACE" not in out
    assert actor_env["MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY"] == "1024"
    assert actor_env["MINT_FUTURE_STORE_ACTOR_NAME"] == "future-v2"
    assert actor_env["MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID"] == "rq-k2"
    assert actor_env["MINT_VLLM_VOLC_RESOURCE_QUEUE_ID"] == "rq-vllm"
    assert actor_env["MINT_MODEL_PLACEMENT_JSON"] == '{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"worker_index":1,"gpu_count":4}}'
    assert actor_env["MINT_MEGATRON_MODEL_PLACEMENT_JSON"] == '{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"worker_index":1,"gpu_count":4}}'
    assert actor_env["MINT_MBRIDGE_EXPORT_GLOO_TIMEOUT_S"] == "123"
    assert actor_env["MINT_MBRIDGE_EXPORT_GATHER_DEBUG"] == "1"
    assert actor_env["MINT_MBRIDGE_EXPORT_GLOO_BARRIER_DEBUG"] == "1"


def test_issue_439_actor_runtime_env_vars_bootstraps_config_actor_hydration(monkeypatch) -> None:
    from tinker_server import config as server_config
    from tinker_server.runtime_config import actor_env_from_environ

    monkeypatch.setattr(server_config, "PFS_RUNTIME_ENV_ROOT", "/runtime")
    monkeypatch.setattr(server_config, "PFS_TINKER_PATH", "/repo")
    monkeypatch.setattr(server_config, "PFS_HF_MODULES_PATH", "/hf-modules")
    monkeypatch.setattr(server_config, "RAY_NAMESPACE", "tinker-test")

    monkeypatch.setenv("RAY_ADDRESS", "ray://127.0.0.1:10001")
    monkeypatch.setenv("TINKER_CHECKPOINT_INDEX_PG_DSN", "postgres://mint:pw@db:5432/mint")
    monkeypatch.setenv("TINKER_CHECKPOINT_INDEX_WRITE_TIMEOUT_MS", "4500")
    monkeypatch.setenv("TINKER_CHECKPOINT_INDEX_UPLOADING_STALE_S", "7200")
    monkeypatch.setenv("MINT_CHECKPOINT_INDEX_PUBLISH_RETRY_S", "30")

    env_vars = server_config.actor_runtime_env_vars(pythonpath="/runtime/pythonpath")

    assert env_vars["TINKER_RAY_NAMESPACE"] == "tinker-test"
    assert env_vars["RAY_ADDRESS"] == "ray://127.0.0.1:10001"
    assert env_vars["PYTHONPATH"] == "/runtime/pythonpath"
    assert env_vars["MINT_CONFIG_ACTOR_HYDRATE"] == "1"
    assert "TINKER_CHECKPOINT_INDEX_PG_DSN" not in env_vars
    assert actor_env_from_environ(__import__("os").environ)["TINKER_CHECKPOINT_INDEX_PG_DSN"] == "postgres://mint:pw@db:5432/mint"
