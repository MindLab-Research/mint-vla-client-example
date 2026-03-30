import json
import os
import sys
import types
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.ModuleType("structlog")


def _runtime_manifest() -> dict:
    runtime = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "tinker"
    ]["runtime_env"]
    return {
        "runtime_env": {
            "site_packages_dir": runtime.get("site_packages_dir", "site-packages"),
            "source_dir": runtime.get("source_dir", "src"),
            "base_python_dir": runtime.get("base_python_dir", "base-python"),
            "host_venv_dir": runtime.get("host_venv_dir", "host-venv"),
        },
        "sources": runtime["sources"],
    }


def _materialize_runtime_env(root: Path, *, with_host_python: bool) -> object:
    from tinker_server.runtime_env import checkout_runtime_env_layout

    layout = checkout_runtime_env_layout(str(root))
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
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
    from tinker_server.runtime_env import bootstrap_runtime_pythonpath

    def _configure(*, with_host_python: bool = True) -> dict[str, object]:
        env_root = tmp_path / "runtime"
        layout = _materialize_runtime_env(env_root, with_host_python=with_host_python)
        tinker_root = tmp_path / "repo"
        hf_modules = tmp_path / "hf-modules"
        hf_home = tmp_path / "hf-home"
        openpi_data_home = tmp_path / "openpi-cache"
        for path in (tinker_root, hf_modules, hf_home, openpi_data_home):
            path.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", str(env_root))
        monkeypatch.setenv("PFS_TINKER_PATH", str(tinker_root))
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
            "tinker_root": tinker_root,
            "hf_modules": hf_modules,
            "hf_home": hf_home,
            "openpi_data_home": openpi_data_home,
            "pythonpath": pythonpath,
        }

    return _configure
