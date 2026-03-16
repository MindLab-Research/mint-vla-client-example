from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
import tomllib
import subprocess
import sys

from tinker_server.runtime_env import (
    bootstrap_runtime_pythonpath,
    build_runtime_pythonpath,
    runtime_env_layout,
)


def _materialize_runtime_env(root: Path) -> None:
    layout = runtime_env_layout(str(root))
    Path(layout.site_packages).mkdir(parents=True, exist_ok=True)
    for entry in layout.pythonpath_entries[1:]:
        Path(entry).mkdir(parents=True, exist_ok=True)
    for entry in layout.host_pythonpath_entries:
        Path(entry).mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).parent.mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).write_text("#!/bin/sh\n", encoding="utf-8")
    Path(layout.host_python).chmod(0o755)


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
        "PFS_HF_MODULES_PATH": "/pfs/hf/modules",
    }

    out = bootstrap_runtime_pythonpath(environ, repo_root="/repo")

    assert str(env_root / "site-packages") in out.split(":")
    assert "/pfs/code/tinker-server" in out.split(":")
    assert "/pfs/hf/modules" in out.split(":")


def test_bootstrap_runtime_pythonpath_requires_runtime_root():
    with pytest.raises(RuntimeError, match="PFS_RUNTIME_ENV_ROOT is required"):
        bootstrap_runtime_pythonpath({}, repo_root="/repo")


def test_bootstrap_runtime_pythonpath_requires_tinker_path(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    with pytest.raises(RuntimeError, match="PFS_TINKER_PATH is required"):
        bootstrap_runtime_pythonpath(
            {
                "PFS_RUNTIME_ENV_ROOT": str(env_root),
                "PFS_HF_MODULES_PATH": "/hf",
            },
            repo_root="/repo",
        )


def test_bootstrap_runtime_pythonpath_requires_hf_modules_path(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    with pytest.raises(RuntimeError, match="PFS_HF_MODULES_PATH is required"):
        bootstrap_runtime_pythonpath(
            {
                "PFS_RUNTIME_ENV_ROOT": str(env_root),
                "PFS_TINKER_PATH": "/repo",
            },
            repo_root="/repo",
        )


def test_runtime_env_layout_tracks_pyproject_source_pythonpaths():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected = []
    for source in data["tool"]["tinker"]["runtime_env"]["sources"]:
        if source.get("host_only", False):
            continue
        for rel in source.get("pythonpath", ["."]):
            rel_str = str(rel).strip()
            suffix = "" if rel_str in ("", ".") else f"/{rel_str}"
            expected.append(f"/tmp/runtime/src/{source['name']}{suffix}")
    layout = runtime_env_layout("/tmp/runtime")
    assert list(layout.pythonpath_entries[1:]) == expected


def test_runtime_env_layout_tracks_host_only_pythonpaths():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected = []
    for source in data["tool"]["tinker"]["runtime_env"]["sources"]:
        if not source.get("host_only", False):
            continue
        for rel in source.get("pythonpath", ["."]):
            rel_str = str(rel).strip()
            suffix = "" if rel_str in ("", ".") else f"/{rel_str}"
            expected.append(f"/tmp/runtime/src/{source['name']}{suffix}")
    layout = runtime_env_layout("/tmp/runtime")
    assert list(layout.host_pythonpath_entries) == expected


def test_runtime_env_layout_prefers_manifest_sources(tmp_path):
    env_root = tmp_path / "runtime"
    manifest = {
        "runtime_env": {
            "site_packages_dir": "site-packages",
            "source_dir": "src",
            "host_venv_dir": "host-venv",
        },
        "sources": [
            {
                "name": "CustomSource",
                "pythonpath": ["src"],
            },
            {
                "name": "HostOnlySource",
                "pythonpath": ["."],
                "host_only": True,
            },
        ],
    }
    env_root.mkdir(parents=True, exist_ok=True)
    (env_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    layout = runtime_env_layout(str(env_root))
    assert list(layout.pythonpath_entries) == [
        str(env_root / "site-packages"),
        str(env_root / "src" / "CustomSource" / "src"),
    ]
    assert list(layout.host_pythonpath_entries) == [
        str(env_root / "src" / "HostOnlySource"),
    ]


def test_config_import_does_not_require_runtime_root():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tinker_server.config as c; print(c.PFS_RUNTIME_ENV_ROOT == '')",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    assert out.stdout.strip() == "True"


def test_run_server_parses_config_before_runtime_bootstrap(tmp_path):
    cfg = tmp_path / "tinker.toml"
    cfg.write_text(
        "\n".join(
            [
                "[paths]",
                f'pfs_runtime_env_root = "{tmp_path / "runtime"}"',
                f'pfs_tinker_path = "{tmp_path / "repo"}"',
                f'pfs_hf_modules_path = "{tmp_path / "hf"}"',
                "",
                "[server]",
                "port = 9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    out = subprocess.run(
        [
            sys.executable,
            "scripts/run_server.py",
            "--config",
            str(cfg),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={**dict(), "TINKER_HOST": "127.0.0.1"},
    )
    assert "PFS_RUNTIME_ENV_ROOT is required" not in (out.stdout + out.stderr)


def test_seed_runtime_env_from_config_overrides_stale_env(tmp_path, monkeypatch):
    from scripts.run_server import _seed_runtime_env_from_config

    cfg = tmp_path / "tinker.toml"
    cfg.write_text(
        "\n".join(
            [
                "[paths]",
                f'pfs_runtime_env_root = "{tmp_path / "runtime"}"',
                f'pfs_tinker_path = "{tmp_path / "repo"}"',
                f'pfs_hf_modules_path = "{tmp_path / "hf"}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", "/stale/runtime")
    monkeypatch.setenv("PFS_TINKER_PATH", "/stale/repo")
    monkeypatch.setenv("PFS_HF_MODULES_PATH", "/stale/hf")

    _seed_runtime_env_from_config(str(cfg))

    assert os.environ["PFS_RUNTIME_ENV_ROOT"] == str(tmp_path / "runtime")
    assert os.environ["PFS_TINKER_PATH"] == str(tmp_path / "repo")
    assert os.environ["PFS_HF_MODULES_PATH"] == str(tmp_path / "hf")


def test_seed_runtime_env_from_config_requires_all_paths(tmp_path, monkeypatch):
    from scripts.run_server import _seed_runtime_env_from_config

    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        "\n".join(
            [
                "[paths]",
                f'pfs_runtime_env_root = "{tmp_path / "runtime"}"',
                f'pfs_tinker_path = "{tmp_path / "repo"}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", "/stale/runtime")
    monkeypatch.setenv("PFS_TINKER_PATH", "/stale/repo")
    monkeypatch.setenv("PFS_HF_MODULES_PATH", "/stale/hf")

    with pytest.raises(RuntimeError, match="missing=.*pfs_hf_modules_path"):
        _seed_runtime_env_from_config(str(cfg))


def test_set_exact_pythonpath_removes_local_checkout_masking(monkeypatch):
    import scripts.run_server as run_server

    monkeypatch.setattr(
        run_server,
        "sys",
        type(
            "FakeSys",
            (),
            {
                "path": [
                    str(Path.cwd()),
                    str(Path.cwd() / "scripts"),
                    "/home/yiwen/.local/lib/python3.14/site-packages",
                    "/home/yiwen/tinker_project/mindlab-toolkit/src",
                    "/opt/host-venv/lib/python3.12",
                    "/usr/lib/python3.12",
                ],
                "prefix": "/opt/host-venv",
                "exec_prefix": "/opt/host-venv",
                "base_prefix": "/usr",
                "base_exec_prefix": "/usr",
            },
        )(),
    )
    out = run_server._set_exact_pythonpath("/canonical/repo:/canonical/hf")
    assert out == "/canonical/repo:/canonical/hf"
    assert run_server.sys.path[:2] == ["/canonical/repo", "/canonical/hf"]
    assert str(Path.cwd()) not in run_server.sys.path
    assert str(Path.cwd() / "scripts") not in run_server.sys.path
    assert "/home/yiwen/.local/lib/python3.14/site-packages" not in run_server.sys.path
    assert "/home/yiwen/tinker_project/mindlab-toolkit/src" not in run_server.sys.path
    assert "/opt/host-venv/lib/python3.12" in run_server.sys.path
    assert "/usr/lib/python3.12" in run_server.sys.path
