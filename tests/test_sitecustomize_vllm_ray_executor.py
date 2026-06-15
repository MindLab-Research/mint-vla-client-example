from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import cast


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
        "runtime_env": {
            "env_vars": {
                "A": "B",
                "MINT_RAY_CLIENT_ADDRESS": "",
                "MINT_RAY_NODE_IP_ADDRESS": "",
                "MINT_RAY_TEMP_DIR": "",
                "RAY_ADDRESS": "",
                "RAY_CLIENT_ADDRESS": "",
                "RAY_TMPDIR": "",
                "TEMP": "",
                "TMP": "",
                "TMPDIR": "",
            }
        },
    }
    assert calls["original"] == {
        "parallel_config": parallel_config,
        "ray_address": "ray://192.168.39.87:10001",
    }


def test_initialize_ray_cluster_prefers_mint_gcs_address_env_for_original(monkeypatch):
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
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.39.87:6379")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    patched = fake_ray_utils_mod.initialize_ray_cluster
    out = patched(types.SimpleNamespace(ray_runtime_env=None), ray_address=None)

    assert out == "ok"
    assert calls["ray_address"] == "192.168.39.87:6379"


def test_initialize_ray_cluster_does_not_nested_init_in_ray_worker_bootstrap(monkeypatch):
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
    fake_vllm_v1.executor = fake_vllm_executor  # type: ignore[attr-defined]
    fake_vllm_executor.ray_executor = fake_ray_exec_mod  # type: ignore[attr-defined]
    fake_vllm_executor.ray_utils = fake_ray_utils_mod  # type: ignore[attr-defined]

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: False  # type: ignore[attr-defined]

    fake_mint = _fake_package("mint_server")
    fake_mint_ray_utils = types.ModuleType("mint_server.ray_utils")

    def fake_init_ray(**kwargs):
        calls["mint_init_ray"] = kwargs
        raise AssertionError("worker bootstrap must not nested ray.init via Mint")

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
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.39.87:6379")
    monkeypatch.setattr(sys, "argv", ["/repo/scripts/vllm_worker_python.py", "-m", "ray._private.workers.default_worker"])

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    parallel_config = types.SimpleNamespace(ray_runtime_env={"env_vars": {"A": "B"}})
    out = fake_ray_utils_mod.initialize_ray_cluster(parallel_config, ray_address=None)

    assert out == "ok"
    assert "mint_init_ray" not in calls
    assert calls["original"] == {
        "parallel_config": parallel_config,
        "ray_address": "192.168.39.87:6379",
    }


def test_initialize_ray_cluster_sanitizes_vllm_worker_env_before_upstream(monkeypatch):
    calls: dict[str, object] = {}

    def original(parallel_config, ray_address=None):
        calls["original"] = {
            "parallel_config": parallel_config,
            "ray_address": ray_address,
            "ray_address_env": __import__("os").environ.get("RAY_ADDRESS"),
            "ray_client_env": __import__("os").environ.get("RAY_CLIENT_ADDRESS"),
            "mint_ray_client_env": __import__("os").environ.get("MINT_RAY_CLIENT_ADDRESS"),
            "mint_ray_gcs_env": __import__("os").environ.get("MINT_RAY_GCS_ADDRESS"),
        }
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
    monkeypatch.delenv("MINT_ENABLE_VLLM_IMPORT_PATCHES", raising=False)
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    monkeypatch.setenv("MINT_ENABLE_VLLM_IMPORT_PATCHES", "1")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.39.87:6379")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.39.87:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.87:10001")

    parallel_config = types.SimpleNamespace(ray_runtime_env=None)
    out = fake_ray_utils_mod.initialize_ray_cluster(parallel_config, ray_address=None)

    assert out == "ok"
    assert calls["original"] == {
        "parallel_config": parallel_config,
        "ray_address": "192.168.39.87:6379",
        "ray_address_env": None,
        "ray_client_env": None,
        "mint_ray_client_env": None,
        "mint_ray_gcs_env": "192.168.39.87:6379",
    }


def test_initialize_ray_cluster_patches_v0_executor_layout(monkeypatch):
    calls: dict[str, object] = {}

    def original(parallel_config, ray_address=None):
        calls["original"] = {
            "parallel_config": parallel_config,
            "ray_address": ray_address,
            "ray_address_env": __import__("os").environ.get("RAY_ADDRESS"),
        }
        return "ok"

    fake_vllm = _fake_package("vllm")
    fake_vllm_executor = _fake_package("vllm.executor")
    fake_ray_utils_mod = types.ModuleType("vllm.executor.ray_utils")
    fake_ray_distributed_mod = types.ModuleType("vllm.executor.ray_distributed_executor")
    fake_ray_gpu_mod = types.ModuleType("vllm.executor.ray_gpu_executor")
    fake_ray_utils_mod.initialize_ray_cluster = original
    fake_ray_distributed_mod.initialize_ray_cluster = original
    fake_ray_gpu_mod.initialize_ray_cluster = original
    fake_vllm.executor = fake_vllm_executor  # type: ignore[attr-defined]
    fake_vllm_executor.ray_utils = fake_ray_utils_mod  # type: ignore[attr-defined]
    fake_vllm_executor.ray_distributed_executor = fake_ray_distributed_mod  # type: ignore[attr-defined]
    fake_vllm_executor.ray_gpu_executor = fake_ray_gpu_mod  # type: ignore[attr-defined]

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.executor", fake_vllm_executor)
    monkeypatch.setitem(sys.modules, "vllm.executor.ray_utils", fake_ray_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.executor.ray_distributed_executor",
        fake_ray_distributed_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm.executor.ray_gpu_executor", fake_ray_gpu_mod)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("MINT_ENABLE_VLLM_IMPORT_PATCHES", "1")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.40.99:6379")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.40.99:6379")

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_executor_use_explicit_cluster_address()

    assert getattr(
        fake_ray_utils_mod.initialize_ray_cluster,
        "_mint_patched_explicit_cluster_address",
    )
    assert (
        fake_ray_distributed_mod.initialize_ray_cluster
        is fake_ray_utils_mod.initialize_ray_cluster
    )
    assert fake_ray_gpu_mod.initialize_ray_cluster is fake_ray_utils_mod.initialize_ray_cluster

    parallel_config = types.SimpleNamespace(
        ray_runtime_env={
            "env_vars": {
                "PYTHONPATH": "/runtime:/repo",
                "RAY_ADDRESS": "192.168.40.99:6379",
                "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.40.99:10001",
            }
        }
    )
    out = fake_ray_distributed_mod.initialize_ray_cluster(
        parallel_config,
        ray_address=None,
    )

    assert out == "ok"
    assert calls["original"] == {
        "parallel_config": parallel_config,
        "ray_address": "192.168.40.99:6379",
        "ray_address_env": None,
    }
    assert parallel_config.ray_runtime_env["env_vars"] == {
        "PYTHONPATH": "/runtime:/repo",
        "RAY_ADDRESS": "",
        "MINT_RAY_CLIENT_ADDRESS": "",
        "MINT_RAY_GCS_ADDRESS": "192.168.40.99:6379",
        "MINT_RAY_TEMP_DIR": "",
        "MINT_RAY_NODE_IP_ADDRESS": "",
        "RAY_TMPDIR": "",
        "TMPDIR": "",
        "TMP": "",
        "TEMP": "",
        "RAY_CLIENT_ADDRESS": "",
    }


def test_runtime_env_to_dict_blanks_driver_attach_hints(monkeypatch):
    class FakeRuntimeEnv:
        def to_dict(self):
            return {
                "env_vars": {
                    "PYTHONPATH": "/runtime:/repo",
                    "MINT_RAY_NAMESPACE": "mint",
                    "MINT_RAY_TEMP_DIR": "/tmp/ray-driver",
                    "MINT_RAY_NODE_IP_ADDRESS": "192.168.39.234",
                    "RAY_TMPDIR": "/tmp/ray",
                    "TMPDIR": "/tmp/driver",
                    "TMP": "/tmp/driver",
                    "TEMP": "/tmp/driver",
                    "RAY_ADDRESS": "192.168.39.234:6379",
                    "RAY_CLIENT_ADDRESS": "ray://192.168.39.234:10001",
                    "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.39.234:10001",
                    "MINT_RAY_GCS_ADDRESS": "192.168.39.234:6379",
                },
                "py_executable": "/repo/scripts/vllm_worker_python.py",
            }

    fake_ray = _fake_package("ray")
    fake_runtime_env_mod = types.ModuleType("ray.runtime_env")
    fake_runtime_env_mod.RuntimeEnv = FakeRuntimeEnv  # type: ignore[attr-defined]
    fake_ray.runtime_env = fake_runtime_env_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.runtime_env", fake_runtime_env_mod)

    module = _load_repo_sitecustomize()
    module._patch_ray_runtime_env_to_dict_drop_driver_attach_hints()

    data = FakeRuntimeEnv().to_dict()
    env_vars = data["env_vars"]
    assert env_vars == {
        "PYTHONPATH": "/runtime:/repo",
        "MINT_RAY_NAMESPACE": "mint",
        "MINT_RAY_TEMP_DIR": "",
        "MINT_RAY_NODE_IP_ADDRESS": "",
        "RAY_TMPDIR": "",
        "TMPDIR": "",
        "TMP": "",
        "TEMP": "",
        "RAY_ADDRESS": "",
        "RAY_CLIENT_ADDRESS": "",
        "MINT_RAY_CLIENT_ADDRESS": "",
        "MINT_RAY_GCS_ADDRESS": "192.168.39.234:6379",
    }
    assert data["py_executable"] == "/repo/scripts/vllm_worker_python.py"


def test_parallel_config_ray_runtime_env_blanks_driver_attach_hints_and_sets_wrapper(monkeypatch):
    class ParallelConfig:
        ray_runtime_env = {
            "env_vars": {
                "PYTHONPATH": "/runtime:/repo",
                "RAY_ADDRESS": "192.168.39.234:6379",
                "RAY_CLIENT_ADDRESS": "ray://192.168.39.234:10001",
                "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.39.234:10001",
                "MINT_RAY_GCS_ADDRESS": "old-gcs:6379",
            }
        }

    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.40.99:6379")
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", "/repo/scripts/vllm_worker_python.py")

    module = _load_repo_sitecustomize()
    sanitized = module._sanitize_vllm_parallel_config_ray_runtime_env(ParallelConfig)

    assert sanitized == ParallelConfig.ray_runtime_env
    assert sanitized["env_vars"] == {
        "PYTHONPATH": "/runtime:/repo",
        "RAY_ADDRESS": "",
        "RAY_CLIENT_ADDRESS": "",
        "MINT_RAY_CLIENT_ADDRESS": "",
        "MINT_RAY_GCS_ADDRESS": "192.168.40.99:6379",
        "MINT_RAY_TEMP_DIR": "",
        "MINT_RAY_NODE_IP_ADDRESS": "",
        "RAY_TMPDIR": "",
        "TMPDIR": "",
        "TMP": "",
        "TEMP": "",
    }
    assert sanitized["py_executable"] == "/repo/scripts/vllm_worker_python.py"


def test_vllm_ray_env_carries_mint_gcs_address_without_ray_address(monkeypatch):
    calls: dict[str, object] = {}

    def original(*, exclude_vars=None, additional_vars=None, destination=None):
        calls["exclude_vars"] = exclude_vars
        calls["additional_vars"] = set(additional_vars or ())
        calls["destination"] = destination
        return {"ok": True}

    fake_vllm = _fake_package("vllm")
    fake_vllm_ray = _fake_package("vllm.ray")
    fake_ray_env = types.ModuleType("vllm.ray.ray_env")
    setattr(fake_ray_env, "get_env_vars_to_copy", original)
    setattr(fake_vllm, "ray", fake_vllm_ray)
    setattr(fake_vllm_ray, "ray_env", fake_ray_env)

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.ray", fake_vllm_ray)
    monkeypatch.setitem(sys.modules, "vllm.ray.ray_env", fake_ray_env)
    monkeypatch.setenv("PYTHONPATH", "/runtime:/repo")

    module = _load_repo_sitecustomize()
    module._patch_vllm_ray_env_carry_over_pythonpath()

    out = fake_ray_env.get_env_vars_to_copy(additional_vars={"VLLM_LOGGING_LEVEL"})

    assert out == {"ok": True}
    additional_vars = cast(set[str], calls["additional_vars"])
    assert "PYTHONPATH" in additional_vars
    assert "MINT_RAY_GCS_ADDRESS" in additional_vars
    assert "RAY_ADDRESS" not in additional_vars
    assert "RAY_CLIENT_ADDRESS" not in additional_vars
    assert "MINT_RAY_CLIENT_ADDRESS" not in additional_vars


def test_vllm_system_utils_spawn_hint_does_not_leave_ray_address(monkeypatch):
    calls: dict[str, object] = {}

    def original_maybe_force_spawn():
        import os

        calls["original"] = True
        os.environ["RAY_ADDRESS"] = "192.168.40.99:6379"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    fake_vllm = _fake_package("vllm")
    fake_utils = _fake_package("vllm.utils")
    fake_system_utils = types.ModuleType("vllm.utils.system_utils")
    fake_system_utils._maybe_force_spawn = original_maybe_force_spawn  # type: ignore[attr-defined]
    fake_vllm.utils = fake_utils  # type: ignore[attr-defined]
    fake_utils.system_utils = fake_system_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.system_utils", fake_system_utils)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    module = _load_repo_sitecustomize()
    module._patch_vllm_system_utils_spawn_without_ray_address()

    fake_system_utils._maybe_force_spawn()  # type: ignore[attr-defined]

    assert calls == {"original": True}
    assert "RAY_ADDRESS" not in __import__("os").environ
    assert __import__("os").environ["MINT_RAY_GCS_ADDRESS"] == "192.168.40.99:6379"
    assert __import__("os").environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    os.environ.pop("MINT_RAY_GCS_ADDRESS", None)


def test_ray_init_patch_cleans_runtime_ray_address_reintroduced_by_worker(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = types.ModuleType("ray")

    def original_init(*args, **kwargs):
        import os

        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "ray_address_env": os.environ.get("RAY_ADDRESS"),
                "ray_client_env": os.environ.get("RAY_CLIENT_ADDRESS"),
                "mint_ray_client_env": os.environ.get("MINT_RAY_CLIENT_ADDRESS"),
                "mint_ray_gcs_env": os.environ.get("MINT_RAY_GCS_ADDRESS"),
            }
        )
        return {"ok": True}

    fake_ray.init = original_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("MINT_ENABLE_VLLM_IMPORT_PATCHES", "1")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.40.99:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.40.99:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.40.99:10001")
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)

    module = _load_repo_sitecustomize()
    module._patch_ray_init_drop_worker_attach_hints()

    out = fake_ray.init(address="auto", namespace="mint")  # type: ignore[attr-defined]

    assert out == {"ok": True}
    assert calls == [
        {
            "args": (),
            "kwargs": {"address": "auto", "namespace": "mint"},
            "ray_address_env": None,
            "ray_client_env": None,
            "mint_ray_client_env": None,
            "mint_ray_gcs_env": "192.168.40.99:6379",
        }
    ]
    assert "RAY_ADDRESS" not in __import__("os").environ
    assert __import__("os").environ["MINT_RAY_GCS_ADDRESS"] == "192.168.40.99:6379"
    os.environ.pop("MINT_RAY_GCS_ADDRESS", None)


def test_ray_init_patch_leaves_driver_ray_address_outside_worker_env(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = types.ModuleType("ray")

    def original_init(*args, **kwargs):
        import os

        calls.append(
            {
                "ray_address_env": os.environ.get("RAY_ADDRESS"),
                "ray_client_env": os.environ.get("RAY_CLIENT_ADDRESS"),
            }
        )
        return {"ok": True}

    fake_ray.init = original_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.delenv("MINT_ENABLE_VLLM_IMPORT_PATCHES", raising=False)
    monkeypatch.delenv("RAY_ACTOR_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["python", "scripts/run_server.py"])
    monkeypatch.setenv("RAY_ADDRESS", "192.168.40.99:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.40.99:10001")

    module = _load_repo_sitecustomize()
    module._patch_ray_init_drop_worker_attach_hints()

    out = fake_ray.init(address="auto")  # type: ignore[attr-defined]

    assert out == {"ok": True}
    assert calls == [
        {
            "ray_address_env": "192.168.40.99:6379",
            "ray_client_env": "ray://192.168.40.99:10001",
        }
    ]
    assert __import__("os").environ["RAY_ADDRESS"] == "192.168.40.99:6379"


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
