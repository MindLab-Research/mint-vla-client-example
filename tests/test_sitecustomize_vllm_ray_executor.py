from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_repo_sitecustomize():
    path = Path(__file__).resolve().parents[1] / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("mint_issue512_sitecustomize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_package(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def test_initialize_ray_cluster_uses_explicit_address_for_mint_init(monkeypatch):
    calls: dict[str, object] = {}

    def original(parallel_config, ray_address=None):
        calls["original"] = {
            "parallel_config": parallel_config,
            "ray_address": ray_address,
        }
        return "ok"

    fake_vllm = _fake_package("vllm")
    fake_vllm_v1 = _fake_package("vllm.v1")
    fake_vllm_executor = _fake_package("vllm.v1.executor")
    fake_ray_exec_mod = types.ModuleType("vllm.v1.executor.ray_executor")
    fake_ray_utils_mod = types.ModuleType("vllm.v1.executor.ray_utils")
    fake_ray_exec_mod.initialize_ray_cluster = original
    fake_ray_utils_mod.initialize_ray_cluster = original
    fake_vllm.executor = fake_vllm_executor  # type: ignore[attr-defined]
    fake_vllm_v1.executor = fake_vllm_executor  # type: ignore[attr-defined]
    fake_vllm_executor.ray_executor = fake_ray_exec_mod  # type: ignore[attr-defined]
    fake_vllm_executor.ray_utils = fake_ray_utils_mod  # type: ignore[attr-defined]

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: False  # type: ignore[attr-defined]

    fake_mint = _fake_package("mint_server")
    fake_mint_ray_utils = types.ModuleType("mint_server.ray_utils")

    def fake_init_ray(**kwargs):
        calls["mint_init_ray"] = kwargs
        return None

    fake_mint_ray_utils.init_ray = fake_init_ray  # type: ignore[attr-defined]
    fake_mint.ray_utils = fake_mint_ray_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor", fake_vllm_executor)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_executor", fake_ray_exec_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_utils", fake_ray_utils_mod)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "mint_server", fake_mint)
    monkeypatch.setitem(sys.modules, "mint_server.ray_utils", fake_mint_ray_utils)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    patched = fake_ray_utils_mod.initialize_ray_cluster
    parallel_config = types.SimpleNamespace(ray_runtime_env={"env_vars": {"A": "B"}})
    out = patched(parallel_config, ray_address="ray://192.168.39.87:10001")

    assert out == "ok"
    assert calls["mint_init_ray"] == {
        "address": "ray://192.168.39.87:10001",
        "runtime_env": {"env_vars": {"A": "B"}},
    }
    assert calls["original"] == {
        "parallel_config": parallel_config,
        "ray_address": "ray://192.168.39.87:10001",
    }


def test_initialize_ray_cluster_prefers_ray_address_env_for_original(monkeypatch):
    calls: dict[str, object] = {}

    def original(parallel_config, ray_address=None):
        calls["ray_address"] = ray_address
        return "ok"

    fake_vllm = _fake_package("vllm")
    fake_vllm_v1 = _fake_package("vllm.v1")
    fake_vllm_executor = _fake_package("vllm.v1.executor")
    fake_ray_exec_mod = types.ModuleType("vllm.v1.executor.ray_executor")
    fake_ray_utils_mod = types.ModuleType("vllm.v1.executor.ray_utils")
    fake_ray_exec_mod.initialize_ray_cluster = original
    fake_ray_utils_mod.initialize_ray_cluster = original
    fake_vllm_v1.executor = fake_vllm_executor  # type: ignore[attr-defined]
    fake_vllm_executor.ray_executor = fake_ray_exec_mod  # type: ignore[attr-defined]
    fake_vllm_executor.ray_utils = fake_ray_utils_mod  # type: ignore[attr-defined]

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor", fake_vllm_executor)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_executor", fake_ray_exec_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_utils", fake_ray_utils_mod)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("RAY_ADDRESS", "192.168.39.87:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    patched = fake_ray_utils_mod.initialize_ray_cluster
    out = patched(types.SimpleNamespace(ray_runtime_env=None), ray_address=None)

    assert out == "ok"
    assert calls["ray_address"] == "192.168.39.87:6379"


def test_qwen35_text_only_adapter_patch_runs_from_sitecustomize(monkeypatch):
    import torch

    class QwenNextMixtureOfExperts:
        def __init__(self, config=None):
            self.config = config

        def set_moe_parameters(self):
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

    config = types.SimpleNamespace(
        mint_qwen35_text_only_shim=True,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )

    class Qwen3NextForCausalLM:
        def __init__(self, config=config):
            self.config = config

        def load_weights(self, weights):
            return list(weights)

    class Qwen3NextModel:
        def __init__(self):
            self.config = config

        def load_weights(self, weights):
            return list(weights)

    fake_vllm = _fake_package("vllm")
    fake_executor = _fake_package("vllm.model_executor")
    fake_models = _fake_package("vllm.model_executor.models")
    fake_qwen3_next = types.ModuleType("vllm.model_executor.models.qwen3_next")
    fake_qwen3_next.QwenNextMixtureOfExperts = QwenNextMixtureOfExperts
    fake_qwen3_next.Qwen3NextForCausalLM = Qwen3NextForCausalLM
    fake_qwen3_next.Qwen3NextModel = Qwen3NextModel

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", fake_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", fake_models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_next", fake_qwen3_next)

    module = _load_repo_sitecustomize()
    module._patch_vllm_qwen35_text_only_adapter()

    unmarked_model = QwenNextMixtureOfExperts(types.SimpleNamespace())
    try:
        unmarked_model.set_moe_parameters()
    except RuntimeError as exc:
        assert "No Qwen3Next layer found" in str(exc)
    else:
        raise AssertionError("unmarked Qwen3Next MoE initialization must preserve upstream failure")

    model = QwenNextMixtureOfExperts(config)
    model.set_moe_parameters()

    assert model.moe_layers == []
    assert model.num_moe_layers == 0
    assert model.num_logical_experts == 0

    weights = [
        ("model.language_model.layers.0.mlp.gate_proj.weight", "text"),
        ("model.visual.blocks.0.attn.qkv.weight", "vision"),
        ("model.language_model.norm.weight", "native"),
    ]
    outer_expected = [
        ("model.layers.0.mlp.gate_proj.weight", "text"),
        ("model.norm.weight", "native"),
    ]
    inner_expected = [
        ("layers.0.mlp.gate_proj.weight", "text"),
        ("norm.weight", "native"),
    ]

    assert Qwen3NextForCausalLM().load_weights(weights) == outer_expected
    assert Qwen3NextModel().load_weights(weights) == inner_expected

    unmarked_weights = Qwen3NextModel()
    unmarked_weights.config = types.SimpleNamespace()
    assert unmarked_weights.load_weights(weights) == weights

    qkv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    z = 100 + torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    b = 200 + torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    a = 300 + torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    packed = Qwen3NextModel().load_weights(
        [
            ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", qkv),
            ("model.language_model.layers.0.linear_attn.in_proj_b.weight", b),
            ("model.language_model.layers.0.linear_attn.in_proj_z.weight", z),
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", a),
        ]
    )
    q, k, v = torch.split(qkv, [4, 4, 8], dim=0)
    q = q.reshape(2, 2, 3)
    k = k.reshape(2, 2, 3)
    v = v.reshape(2, 4, 3)
    z_grouped = z.reshape(2, 4, 3)
    expected_qkvz = torch.cat([q, k, v, z_grouped], dim=1).reshape(-1, 3)
    expected_ba = torch.cat([b.reshape(2, 2, 3), a.reshape(2, 2, 3)], dim=1).reshape(
        -1, 3
    )

    assert packed[0][0] == "layers.0.linear_attn.in_proj_qkvz.weight"
    assert torch.equal(packed[0][1], expected_qkvz)
    assert packed[1][0] == "layers.0.linear_attn.in_proj_ba.weight"
    assert torch.equal(packed[1][1], expected_ba)


def test_qwen35_linear_attention_packing_fails_fast_on_bad_shapes():
    import pytest
    import torch

    from mint_server.backend.qwen35_text_vllm_adapter import (
        _pack_qwen35_b_a,
        _pack_qwen35_qkv_z,
    )

    config = types.SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )

    with pytest.raises(ValueError, match="in_proj_qkv weight shape"):
        _pack_qwen35_qkv_z(
            config,
            torch.zeros((15, 3)),
            torch.zeros((8, 3)),
        )

    with pytest.raises(ValueError, match="in_proj_a weight shape"):
        _pack_qwen35_b_a(
            config,
            torch.zeros((4, 3)),
            torch.zeros((5, 3)),
        )
