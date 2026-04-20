import json

from tinker_server.backend import api_work_queue as api_work_queue_module
from tinker_server.backend import queue_execution_runtime as runtime_module


def test_issue_440_queue_execution_runtime_propagates_mint_actor_and_routing_overrides(monkeypatch):
    expected = {
        "MINT_API_WORK_QUEUE_ACTOR_NAME": "api-q",
        "MINT_CAPACITY_MANAGER_ACTOR_NAME": "cap-q",
        "MINT_FUTURE_STORE_ACTOR_NAME": "future-q",
        "MINT_RESOURCE_POOL_ACTOR_NAME": "pool-q",
        "MINT_DENSE_MODEL_NODE_IPS_JSON": '{"Qwen/Qwen3-0.6B":["192.168.38.175"]}',
        "MINT_MODEL_NODE_IPS_JSON": '{"Qwen/Qwen3-0.6B":["192.168.38.175"]}',
        "MINT_VLLM_PINNED_NODE_IP_JSON": '{"Qwen/Qwen3-0.6B":"192.168.38.175"}',
        "MINT_SUPPORTED_MODELS": "Qwen/Qwen3-0.6B",
        "MINT_QUEUE_EXECUTION_RUNTIME_DEBUG_LOG_PATH": "/tmp/queue-runtime-debug.jsonl",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    out = runtime_module._runtime_env_overrides()

    for key, value in expected.items():
        assert out[key] == value
    assert "TINKER_RAY_NAMESPACE" not in out
    assert "MINT_RAY_NAMESPACE" not in out


def test_issue_440_queue_execution_runtime_uses_canonical_config_actor_names(monkeypatch):
    monkeypatch.setattr(runtime_module.server_config, "api_work_queue_actor_name", "api-q-config", raising=False)
    monkeypatch.setattr(runtime_module.server_config, "capacity_manager_actor_name", "cap-q-config", raising=False)

    out = runtime_module._runtime_env_overrides()

    assert out["MINT_API_WORK_QUEUE_ACTOR_NAME"] == "api-q-config"
    assert out["MINT_CAPACITY_MANAGER_ACTOR_NAME"] == "cap-q-config"
    assert "TINKER_API_WORK_QUEUE_ACTOR_NAME" not in out
    assert "TINKER_CAPACITY_MANAGER_ACTOR_NAME" not in out


def test_issue_440_queue_execution_runtime_debug_log_helpers(monkeypatch, tmp_path):
    log_path = tmp_path / "queue-runtime.jsonl"
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-issue-440")
    monkeypatch.setenv("MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME", "queue-runtime")
    monkeypatch.setenv("MINT_QUEUE_EXECUTION_RUNTIME_DEBUG_LOG_PATH", str(log_path))

    assert runtime_module._queue_runtime_debug_log_path() == str(log_path)

    runtime_module._append_queue_runtime_debug("stage", detail="x")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "stage"
    assert payload["detail"] == "x"
    assert payload["actor_name"] == "queue-runtime"
    assert payload["namespace"] == "ns-issue-440"


def test_issue_440_api_work_queue_debug_log_helpers(monkeypatch, tmp_path):
    log_path = tmp_path / "api-work-queue.jsonl"
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-issue-440")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_ACTOR_NAME", "api-work-queue")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_DEBUG_LOG_PATH", str(log_path))

    assert api_work_queue_module._api_work_queue_debug_log_path() == str(log_path)

    api_work_queue_module._append_api_work_queue_debug("stage", detail="y")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "stage"
    assert payload["detail"] == "y"
    assert payload["actor_name"] == "api-work-queue"
    assert payload["namespace"] == "ns-issue-440"
