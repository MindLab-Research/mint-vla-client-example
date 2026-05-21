import json
import yaml


def _write_model_topology(tmp_path, models):
    path = tmp_path / "topology.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "deployment_env": "dev",
                "cluster_id": "volcano",
                "state_path": str(tmp_path / "topology_state.yaml"),
                "providers": {},
                "nodes": {"desired": []},
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_list_supported_models_env(monkeypatch):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv(
        "MINT_SUPPORTED_MODELS",
        "Qwen/Qwen3-0.6B, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-0.6B",
    )
    got = mr.list_supported_models()
    assert got == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"]


def test_list_supported_models_env_unknown_raises(monkeypatch):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv("MINT_SUPPORTED_MODELS", "does/not-exist")
    try:
        mr.list_supported_models()
    except ValueError as e:
        assert "Unsupported models" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_list_supported_models_env_accepts_gateway_routed_model(monkeypatch):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv("MINT_SUPPORTED_MODELS", "Qwen/Qwen3-0.6B, zai-org/GLM-5.1")
    monkeypatch.setenv(
        "MINT_GATEWAY_CONFIG_JSON",
        json.dumps(
            {
                "model_to_upstream": {"zai-org/GLM-5.1": "glm51"},
                "upstreams": {
                    "glm51": {
                        "base_url": "http://example.com:18000",
                        "auth_mode": "static_api_key",
                        "api_key": "secret",
                    }
                },
            }
        ),
    )

    got = mr.list_supported_models()

    assert got == ["Qwen/Qwen3-0.6B", "zai-org/GLM-5.1"]


def test_model_config_overrides_json(monkeypatch):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv(
        "MINT_MODEL_CONFIG_OVERRIDES_JSON",
        json.dumps({"Qwen/Qwen3-0.6B": {"inference_tp": 2, "gpu_memory_utilization": 0.91}}),
    )
    cfg = mr.get_model_config("Qwen/Qwen3-0.6B")
    assert cfg.inference_tp == 2
    assert cfg.gpu_memory_utilization == 0.91


def test_model_config_overrides_json_unknown_field_raises(monkeypatch):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv(
        "MINT_MODEL_CONFIG_OVERRIDES_JSON",
        json.dumps({"Qwen/Qwen3-0.6B": {"not_a_field": 1}}),
    )
    try:
        mr.get_model_config("Qwen/Qwen3-0.6B")
    except ValueError as e:
        assert "unknown fields" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_is_topology_desired_model_matches_hf_and_snapshot(monkeypatch, tmp_path):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_model_topology(tmp_path, {"Qwen/Qwen3-0.6B": {"vllm": {}}}),
    )
    assert mr.is_topology_desired_model("Qwen/Qwen3-0.6B")
    assert mr.is_topology_desired_model(
        "/vePFS-Mindverse/share/huggingface/models--Qwen--Qwen3-0.6B/snapshots/abc123"
    )


def test_is_topology_desired_model_accepts_snapshot_entries(monkeypatch, tmp_path):
    from mint_server.backend import model_registry as mr

    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_model_topology(
            tmp_path,
            {"/vePFS-Mindverse/share/huggingface/models--Qwen--Qwen3-0.6B/snapshots/abc123": {"vllm": {}}},
        ),
    )
    assert mr.is_topology_desired_model("Qwen/Qwen3-0.6B")
