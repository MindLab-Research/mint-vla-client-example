from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import tinker_server.config as server_config_module
from tinker_server.backend import verl_inference


def test_issue_593_verl_inference_builds_node_affinity_options(monkeypatch):
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        '{"Qwen/Qwen3-0.6B":{"gpu_count":1,"node_ip":"10.0.0.17","replica":0}}',
    )

    class _Placement:
        total_gpus = 1
        slices = [SimpleNamespace(node_ip="10.0.0.17")]

    seen_parse = []
    seen_capacity = []

    def _parse_model_gpu_placement(**kwargs):
        seen_parse.append(kwargs)
        return _Placement()

    monkeypatch.setattr(verl_inference, "parse_model_gpu_placement", _parse_model_gpu_placement)
    monkeypatch.setattr(
        verl_inference,
        "assert_node_ip_capacity",
        lambda **kwargs: seen_capacity.append(kwargs),
    )
    monkeypatch.setattr(
        verl_inference.ray,
        "nodes",
        lambda: [{"Alive": True, "NodeManagerAddress": "10.0.0.17", "NodeID": "a" * 56}],
    )

    out = verl_inference._vllm_actor_pin_options_for_model("Qwen/Qwen3-0.6B", required_gpus=1)

    assert out["resources"] == {"node:10.0.0.17": 0.001}
    assert out["scheduling_strategy"].node_id == "a" * 56
    assert seen_parse[0]["replica"] == 0
    assert seen_capacity[0]["required_gpus_by_node_ip"] == {"10.0.0.17": 1}


def test_issue_593_verl_inference_uses_replica_env_for_node_affinity(monkeypatch):
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        '{"Qwen/Qwen3-0.6B":[{"gpu_count":1,"node_ip":"10.0.0.17","replica":2}]}',
    )
    monkeypatch.setenv("MINT_MODEL_ACTOR_REPLICA_ID", "replica-2")

    class _Placement:
        total_gpus = 1
        slices = [SimpleNamespace(node_ip="10.0.0.17")]

    seen_parse = []

    def _parse_model_gpu_placement(**kwargs):
        seen_parse.append(kwargs)
        return _Placement()

    monkeypatch.setattr(verl_inference, "parse_model_gpu_placement", _parse_model_gpu_placement)
    monkeypatch.setattr(verl_inference, "assert_node_ip_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(
        verl_inference.ray,
        "nodes",
        lambda: [{"Alive": True, "NodeManagerAddress": "10.0.0.17", "NodeID": "a" * 56}],
    )

    out = verl_inference._vllm_actor_pin_options_for_model("Qwen/Qwen3-0.6B", required_gpus=1)

    assert out["resources"] == {"node:10.0.0.17": 0.001}
    assert seen_parse[0]["replica"] == 2


@pytest.mark.anyio
async def test_issue_593_verl_inference_disables_sleep_mode_by_default(monkeypatch):
    captured = {}

    class _FakeRolloutConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeCheckpointEngineConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeHFModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeRolloutMode:
        STANDALONE = "standalone"

    class _FakeLaunchServer:
        async def remote(self):
            return None

    class _FakeServerActor:
        launch_server = _FakeLaunchServer()

    class _FakeServer:
        @classmethod
        def options(cls, **_kwargs):
            return cls

        @classmethod
        def remote(cls, **_kwargs):
            return _FakeServerActor()

    class _FakeCfg:
        is_moe = False
        max_loras = 1
        max_cpu_loras = 1
        max_model_len = 128
        max_num_seqs = 8
        gpu_memory_utilization = 0.5
        max_num_batched_tokens = 128

    monkeypatch.delenv("MINT_VLLM_ENABLE_SLEEP_MODE", raising=False)
    monkeypatch.setenv("RAY_ADDRESS", "ray://test")
    monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", "/tmp/runtime")
    monkeypatch.setenv("MINT_CODE_ROOT", "/tmp/repo")
    monkeypatch.setenv("PFS_HF_MODULES_PATH", "/tmp/hf")
    config_module = sys.modules["tinker_server.config"]
    monkeypatch.setattr(config_module, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime")
    monkeypatch.setattr(config_module, "MINT_CODE_ROOT", "/tmp/repo")
    monkeypatch.setattr(config_module, "PFS_HF_MODULES_PATH", "/tmp/hf")
    config_module.actor_runtime_env_vars.__globals__["PFS_RUNTIME_ENV_ROOT"] = "/tmp/runtime"
    config_module.actor_runtime_env_vars.__globals__["MINT_CODE_ROOT"] = "/tmp/repo"
    config_module.actor_runtime_env_vars.__globals__["PFS_HF_MODULES_PATH"] = "/tmp/hf"
    monkeypatch.setattr(verl_inference, "get_model_config", lambda _model: _FakeCfg())
    monkeypatch.setattr(verl_inference, "_create_extended_server_class", lambda **_kwargs: _FakeServer)
    monkeypatch.setattr(verl_inference, "_vllm_actor_pin_options_for_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(verl_inference, "init_ray", lambda **_kwargs: None)
    monkeypatch.setattr(
            config_module,
            "actor_runtime_env_vars",
            lambda *, pythonpath, extra=None: {"PYTHONPATH": pythonpath, **(extra or {})},
        )
    monkeypatch.setattr(config_module, "actor_ld_library_path", lambda: "")
    monkeypatch.setattr(config_module, "preferred_vllm_python_executable", lambda: "")
    monkeypatch.setitem(
        __import__("sys").modules,
        "verl.workers.config",
        SimpleNamespace(
            CheckpointEngineConfig=_FakeCheckpointEngineConfig,
            HFModelConfig=_FakeHFModelConfig,
            RolloutConfig=_FakeRolloutConfig,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "verl.workers.rollout.replica",
        SimpleNamespace(RolloutMode=_FakeRolloutMode),
    )

    engine = verl_inference.VerlInferenceEngine(
        model_path="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    await engine.initialize()

    assert captured["enable_sleep_mode"] is False


def test_issue_593_verl_inference_sleep_mode_env_override(monkeypatch):
    monkeypatch.setenv("MINT_VLLM_ENABLE_SLEEP_MODE", "1")
    assert verl_inference._env_flag("MINT_VLLM_ENABLE_SLEEP_MODE", default=False) is True
