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

    fake_tinker = _fake_package("tinker_server")
    fake_tinker_ray_utils = types.ModuleType("tinker_server.ray_utils")

    def fake_init_ray(**kwargs):
        calls["mint_init_ray"] = kwargs
        return None

    fake_tinker_ray_utils.init_ray = fake_init_ray  # type: ignore[attr-defined]
    fake_tinker.ray_utils = fake_tinker_ray_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor", fake_vllm_executor)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_executor", fake_ray_exec_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.ray_utils", fake_ray_utils_mod)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "tinker_server", fake_tinker)
    monkeypatch.setitem(sys.modules, "tinker_server.ray_utils", fake_tinker_ray_utils)
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
