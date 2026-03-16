from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from tinker_server.runtime_env import (
    DEFAULT_HF_MODULES_PATH,
    bootstrap_runtime_pythonpath,
    build_runtime_pythonpath,
    runtime_env_layout,
)


def _materialize_runtime_env(root: Path) -> None:
    layout = runtime_env_layout(str(root))
    Path(layout.site_packages).mkdir(parents=True, exist_ok=True)
    for entry in layout.pythonpath_entries[1:]:
        Path(entry).mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).parent.mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).write_text("#!/bin/sh\n", encoding="utf-8")


def test_build_runtime_pythonpath_uses_canonical_root(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)

    out = build_runtime_pythonpath(
        env_root=str(env_root),
        pfs_tinker_path="/vePFS/code/yiwen/tinker-server",
        pfs_hf_modules_path="/vePFS/hf/modules",
    )

    parts = out.split(":")
    assert parts[0] == str(env_root / "site-packages")
    assert str(env_root / "src" / "Megatron-Bridge" / "src") in parts
    assert str(env_root / "src" / "Megatron-Bridge") in parts
    assert str(env_root / "src" / "verl") in parts
    assert str(env_root / "src" / "Megatron-LM") in parts
    assert parts[-2] == "/vePFS/code/yiwen/tinker-server"
    assert parts[-1] == "/vePFS/hf/modules"


def test_build_runtime_pythonpath_fails_on_incomplete_root(tmp_path):
    env_root = tmp_path / "runtime"
    env_root.mkdir()
    with pytest.raises(RuntimeError):
        build_runtime_pythonpath(
            env_root=str(env_root),
            pfs_tinker_path="/repo",
            pfs_hf_modules_path="/hf",
        )


def test_build_runtime_pythonpath_does_not_require_host_python(tmp_path):
    env_root = tmp_path / "runtime"
    layout = runtime_env_layout(str(env_root))
    Path(layout.site_packages).mkdir(parents=True, exist_ok=True)
    for entry in layout.pythonpath_entries[1:]:
        Path(entry).mkdir(parents=True, exist_ok=True)

    out = build_runtime_pythonpath(
        env_root=str(env_root),
        pfs_tinker_path="/repo",
        pfs_hf_modules_path="/hf",
    )

    assert str(env_root / "site-packages") in out.split(":")


def test_bootstrap_runtime_pythonpath_prefers_runtime_root(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    environ = {
        "PFS_RUNTIME_ENV_ROOT": str(env_root),
        "PFS_TINKER_PATH": "/pfs/code/tinker-server",
    }

    out = bootstrap_runtime_pythonpath(environ, repo_root="/repo")

    assert str(env_root / "site-packages") in out.split(":")
    assert "/pfs/code/tinker-server" in out.split(":")
    assert DEFAULT_HF_MODULES_PATH in out.split(":")


def test_bootstrap_runtime_pythonpath_requires_runtime_root():
    with pytest.raises(RuntimeError, match="PFS_RUNTIME_ENV_ROOT is required"):
        bootstrap_runtime_pythonpath({}, repo_root="/repo")


def test_runtime_env_layout_tracks_pyproject_source_pythonpaths():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected = []
    for source in data["tool"]["tinker"]["runtime_env"]["sources"]:
        for rel in source.get("pythonpath", ["."]):
            rel_str = str(rel).strip()
            suffix = "" if rel_str in ("", ".") else f"/{rel_str}"
            expected.append(f"/tmp/runtime/src/{source['name']}{suffix}")
    layout = runtime_env_layout("/tmp/runtime")
    assert list(layout.pythonpath_entries[1:]) == expected
