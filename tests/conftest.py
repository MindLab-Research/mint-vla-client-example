import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "structlog" not in sys.modules:
    _fake_structlog = types.ModuleType("structlog")
    _fake_structlog.get_logger = lambda name=None: logging.getLogger(name or "mint_server")
    _fake_structlog.configure = lambda *a, **kw: None
    _fake_structlog.stdlib = types.ModuleType("structlog.stdlib")
    _fake_structlog.stdlib.ProcessorFormatter = type("ProcessorFormatter", (), {"__init__": lambda self, *a, **kw: None})
    _fake_structlog.stdlib.add_log_level = lambda *a, **kw: None
    _fake_structlog.stdlib.add_logger_name = lambda *a, **kw: None
    _fake_structlog.stdlib.LoggerFactory = type("LoggerFactory", (), {})
    _fake_structlog.processors = types.ModuleType("structlog.processors")
    _fake_structlog.processors.TimeStamper = lambda *a, **kw: None
    _fake_structlog.processors.StackInfoRenderer = lambda *a, **kw: None
    _fake_structlog.processors.format_exc_info = lambda *a, **kw: None
    _fake_structlog.processors.JSONRenderer = type("JSONRenderer", (), {"__init__": lambda self, *a, **kw: None})
    _fake_structlog.contextvars = types.ModuleType("structlog.contextvars")
    _fake_structlog.contextvars.merge_contextvars = lambda *a, **kw: None
    sys.modules["structlog"] = _fake_structlog


def _install_fake_ray() -> None:
    try:
        import ray  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    ray_module = types.ModuleType("ray")
    ray_exceptions = types.ModuleType("ray.exceptions")
    ray_util_module = types.ModuleType("ray.util")
    ray_sched_module = types.ModuleType("ray.util.scheduling_strategies")
    ray_private_module = types.ModuleType("ray._private")
    ray_private_state_module = types.ModuleType("ray._private.state")

    class _RayActorError(Exception):
        pass

    class _ActorDiedError(_RayActorError):
        pass

    class _GetTimeoutError(Exception):
        pass

    class _RayTaskError(Exception):
        def __init__(self, msg: str = "", *, cause=None):
            super().__init__(msg)
            self.cause = cause

    class _NodeAffinitySchedulingStrategy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    ray_exceptions.RayActorError = _RayActorError
    ray_exceptions.ActorDiedError = _ActorDiedError
    ray_exceptions.GetTimeoutError = _GetTimeoutError
    ray_exceptions.RayTaskError = _RayTaskError
    ray_sched_module.NodeAffinitySchedulingStrategy = _NodeAffinitySchedulingStrategy

    ray_private_state_module.available_resources_per_node = lambda: {}
    ray_private_state_module.actors = lambda *_args, **_kwargs: {}
    ray_private_module.state = ray_private_state_module

    ray_util_module.list_named_actors = lambda *args, **kwargs: []
    ray_util_module.get_placement_group = lambda *args, **kwargs: None
    ray_util_module.remove_placement_group = lambda *args, **kwargs: None
    ray_util_module.placement_group_table = lambda *args, **kwargs: {}

    ray_module.actor = types.SimpleNamespace(ActorHandle=object)
    ray_module.exceptions = ray_exceptions
    ray_module.util = ray_util_module
    ray_module._private = ray_private_module
    ray_module.init = lambda *args, **kwargs: None
    ray_module.is_initialized = lambda: False
    ray_module.get_actor = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("actor not found"))
    ray_module.get = lambda *args, **kwargs: None
    ray_module.nodes = lambda: []
    ray_module.kill = lambda *args, **kwargs: None
    ray_module.remote = lambda *args, **kwargs: (lambda cls: cls)
    ray_module.method = lambda *args, **kwargs: (lambda fn: fn)

    sys.modules["ray"] = ray_module
    sys.modules["ray.exceptions"] = ray_exceptions
    sys.modules["ray.util"] = ray_util_module
    sys.modules["ray.util.scheduling_strategies"] = ray_sched_module
    sys.modules["ray._private"] = ray_private_module
    sys.modules["ray._private.state"] = ray_private_state_module


_install_fake_ray()


def _runtime_manifest() -> dict:
    runtime = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "mint"
    ]["runtime_env"]
    return {
        "runtime_env": {
            "site_packages_dir": runtime.get("site_packages_dir", "site-packages"),
            "source_dir": runtime.get("source_dir", "src"),
            "base_python_dir": runtime.get("base_python_dir", "base-python"),
            "host_venv_dir": runtime.get("host_venv_dir", "host-venv"),
        },
        "sources": runtime["sources"],
        "tier": "gpu_rl",
    }


def _materialize_runtime_env(root: Path, *, with_host_python: bool) -> object:
    from mint_server.ray.runtime_env import checkout_runtime_env_layout

    layout = checkout_runtime_env_layout(str(root))
    tier_root = root / "gpu_rl"
    tier_root.mkdir(parents=True, exist_ok=True)
    (tier_root / "manifest.json").write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    Path(layout.site_packages).mkdir(parents=True, exist_ok=True)
    Path(layout.base_python_root).mkdir(parents=True, exist_ok=True)
    for entry in layout.pythonpath_entries[1:]:
        Path(entry).mkdir(parents=True, exist_ok=True)
    for entry in layout.host_pythonpath_entries:
        Path(entry).mkdir(parents=True, exist_ok=True)
    if with_host_python:
        Path(layout.host_python).parent.mkdir(parents=True, exist_ok=True)
        Path(layout.host_python).write_text("#!/bin/sh\n", encoding="utf-8")
        Path(layout.host_python).chmod(0o755)
    return layout


@pytest.fixture
def configure_runtime_env(monkeypatch, tmp_path):
    from mint_server.ray.runtime_env import bootstrap_runtime_pythonpath

    def _configure(*, with_host_python: bool = True) -> dict[str, object]:
        env_root = tmp_path / "runtime"
        layout = _materialize_runtime_env(env_root, with_host_python=with_host_python)
        mint_root = tmp_path / "repo"
        hf_modules = tmp_path / "hf-modules"
        hf_home = tmp_path / "hf-home"
        openpi_data_home = tmp_path / "openpi-cache"
        for path in (mint_root, hf_modules, hf_home, openpi_data_home):
            path.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", str(env_root))
        monkeypatch.setenv("MINT_CODE_ROOT", str(mint_root))
        monkeypatch.setenv("PFS_HF_MODULES_PATH", str(hf_modules))
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OPENPI_DATA_HOME", str(openpi_data_home))
        monkeypatch.delenv("MINT_OPENPI_FAST_PYTHON", raising=False)
        monkeypatch.delenv("MINT_OPENPI_FAST_PYTHONPATH", raising=False)
        monkeypatch.delenv("MINT_OPENPI_FAST_CWD", raising=False)

        pythonpath = tuple(
            entry
            for entry in bootstrap_runtime_pythonpath(
                os.environ,
                repo_root=str(REPO_ROOT),
            ).split(os.pathsep)
            if entry
        )
        return {
            "env_root": env_root,
            "layout": layout,
            "mint_root": mint_root,
            "hf_modules": hf_modules,
            "hf_home": hf_home,
            "openpi_data_home": openpi_data_home,
            "pythonpath": pythonpath,
        }

    return _configure

@pytest.fixture
def anyio_backend():
    return "asyncio"
