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
