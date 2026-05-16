import asyncio
import json

from tinker_server.backend import api_work_queue as api_work_queue_module
from tinker_server.backend import queue_execution_runtime as runtime_module
from tinker_server import ray_utils as ray_utils_module


def test_issue_440_queue_execution_runtime_propagates_mint_actor_and_routing_overrides(monkeypatch):
    actor_creation = {
        "MINT_API_WORK_QUEUE_ACTOR_NAME": "api-q",
        "MINT_CAPACITY_MANAGER_ACTOR_NAME": "cap-q",
    }
    snapshot_config = {
        "MINT_FUTURE_STORE_ACTOR_NAME": "future-q",
        "MINT_RESOURCE_POOL_ACTOR_NAME": "pool-q",
        "MINT_MODEL_PLACEMENT_JSON": '{"Qwen/Qwen3-0.6B":{"replica":0,"node_ip":"10.0.0.18","gpu_count":1}}',
        "MINT_DENSE_MODEL_PLACEMENT_JSON": '{"Qwen/Qwen3-0.6B":{"replica":0,"node_ip":"10.0.0.18","gpu_count":1}}',
        "MINT_VLLM_MODEL_PLACEMENT_JSON": '{"Qwen/Qwen3-0.6B":{"replica":0,"node_ip":"10.0.0.18","gpu_count":1}}',
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": '{"Qwen/Qwen3-0.6B":{"replica":0,"node_ip":"10.0.0.18","gpu_count":1}}',
        "MINT_SUPPORTED_MODELS": "Qwen/Qwen3-0.6B",
        "MINT_QUEUE_EXECUTION_RUNTIME_DEBUG_LOG_PATH": "/tmp/queue-runtime-debug.jsonl",
    }
    expected = {**actor_creation, **snapshot_config}
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    out = runtime_module._runtime_env_overrides()
    from tinker_server.runtime_config import actor_env_from_environ

    for key, value in actor_creation.items():
        assert out[key] == value
    for key in snapshot_config:
        assert key not in out
    actor_env = actor_env_from_environ(__import__("os").environ)
    for key, value in snapshot_config.items():
        assert actor_env[key] == value
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


def test_issue_440_debug_runtime_env_summary_redacts_env_values():
    runtime_env = {
        "env_vars": {
            "OTEL_EXPORTER_OTLP_HEADERS": "authorization=secret",
            "MINT_APMPLUS_APP_KEY": "secret-app-key",
            "PFS_TINKER_PATH": "/tmp/mint",
        },
        "working_dir": "/tmp/mint",
    }

    queue_summary = runtime_module._summarize_debug_runtime_env(runtime_env)
    api_summary = api_work_queue_module._summarize_debug_runtime_env(runtime_env)

    for summary in (queue_summary, api_summary):
        assert summary["working_dir"] == "/tmp/mint"
        assert summary["env_var_keys"] == [
            "MINT_APMPLUS_APP_KEY",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "PFS_TINKER_PATH",
        ]
        assert "env_vars" not in summary


def test_issue_440_api_work_queue_async_existing_ray_skips_init(monkeypatch):
    client = api_work_queue_module.ApiWorkQueueClient()
    actor = object()

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _unexpected_init_ray(*args, **kwargs):
        raise AssertionError("init_ray should not run when ray is already initialized")

    monkeypatch.setattr(api_work_queue_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(api_work_queue_module, "_append_api_work_queue_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(ray_utils_module, "init_ray", _unexpected_init_ray)

    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_actor", lambda *args, **kwargs: actor)

    out = asyncio.run(client._get_ray_actor_async(require_ready=False))

    assert out is actor
    assert client._ray_actor is actor
