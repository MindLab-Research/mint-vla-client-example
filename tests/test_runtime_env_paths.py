from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

import pytest
import tomllib
import subprocess
import sys

from tinker_server.runtime_env import (
    bootstrap_runtime_pythonpath,
    build_runtime_pythonpath,
    checkout_runtime_env_layout,
    runtime_env_layout,
    validate_runtime_env_layout,
)


def _materialize_runtime_env(root: Path) -> None:
    layout = checkout_runtime_env_layout(str(root))
    manifest = {
        "runtime_env": {
            "site_packages_dir": "site-packages",
            "source_dir": "src",
            "base_python_dir": "base-python",
            "host_venv_dir": "host-venv",
        },
        "sources": tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["tinker"]["runtime_env"]["sources"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    Path(layout.site_packages).mkdir(parents=True, exist_ok=True)
    Path(layout.base_python_root).mkdir(parents=True, exist_ok=True)
    for entry in layout.pythonpath_entries[1:]:
        Path(entry).mkdir(parents=True, exist_ok=True)
    for entry in layout.host_pythonpath_entries:
        Path(entry).mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).parent.mkdir(parents=True, exist_ok=True)
    Path(layout.host_python).write_text("#!/bin/sh\n", encoding="utf-8")
    Path(layout.host_python).chmod(0o755)


def _materialize_runtime_env_with_real_host_python(root: Path) -> None:
    _materialize_runtime_env(root)
    layout = checkout_runtime_env_layout(str(root))
    host_python = Path(layout.host_python)
    host_python.unlink()
    host_python.symlink_to(Path(sys.executable))


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
    assert str(env_root / "src" / "openpi" / "src") in parts
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
    layout = checkout_runtime_env_layout(str(env_root))
    manifest = {
        "runtime_env": {
            "site_packages_dir": "site-packages",
            "source_dir": "src",
            "base_python_dir": "base-python",
            "host_venv_dir": "host-venv",
        },
        "sources": tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["tinker"]["runtime_env"]["sources"],
    }
    env_root.mkdir(parents=True, exist_ok=True)
    (env_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
    assert str(env_root / "src" / "vllm") in out.split(":")
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
    layout = checkout_runtime_env_layout("/tmp/runtime")
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
    layout = checkout_runtime_env_layout("/tmp/runtime")
    assert list(layout.host_pythonpath_entries) == expected


def test_runtime_env_layout_includes_openpi_source_checkout():
    layout = checkout_runtime_env_layout("/tmp/runtime")

    assert str(Path("/tmp/runtime/src/openpi/src")) in layout.pythonpath_entries


def test_runtime_env_layout_includes_openpi_client_source_checkout():
    layout = checkout_runtime_env_layout("/tmp/runtime")

    assert str(Path("/tmp/runtime/src/openpi/packages/openpi-client/src")) in layout.pythonpath_entries


def test_runtime_env_host_dependencies_include_openpi_worker_stack():
    from scripts import build_runtime_env as build_runtime_env

    deps = build_runtime_env._host_deps(build_runtime_env._load_pyproject())

    for requirement in (
        "augmax>=0.3.4",
        "beartype==0.19.0",
        "flax==0.10.2",
        "filelock>=3.16.1",
        "fsspec[gcs]>=2024.6.0",
        "jax[cuda12]==0.5.3",
        "jaxtyping==0.2.36",
        "ml_collections==1.0.0",
        "numpydantic>=1.6.6",
        "optax==0.2.4",
        "orbax-checkpoint==0.11.13",
        "tqdm-loggable>=0.2",
        "tyro>=0.9.5",
    ):
        assert requirement in deps


def test_build_runtime_env_normalizes_host_only_vllm_source_metadata(tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    env_root = tmp_path / "runtime"
    version_file = env_root / "src" / "vllm" / "vllm" / "_version.py"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(
        '__version__ = "0.1.dev1+g89a77b108"\n__version_tuple__ = (0, 1, "dev1", "g89a77b108")\n',
        encoding="utf-8",
    )
    pkg_info = env_root / "src" / "vllm" / "vllm.egg-info" / "PKG-INFO"
    pkg_info.parent.mkdir(parents=True, exist_ok=True)
    pkg_info.write_text(
        "Metadata-Version: 2.4\nName: vllm\nVersion: 0.1.dev1+g89a77b108\n",
        encoding="utf-8",
    )
    pyproject = {
        "tool": {
            "tinker": {
                "runtime_env": {
                    "source_dir": "src",
                    "sources": [
                        {
                            "name": "vllm",
                            "host_only": True,
                            "package_name": "vllm",
                            "version": "0.16.0",
                        }
                    ],
                }
            }
        }
    }

    build_runtime_env._write_host_source_version_files(pyproject, env_root)

    assert '__version__ = "0.16.0"' in version_file.read_text(encoding="utf-8")
    pkg_info_text = pkg_info.read_text(encoding="utf-8")
    assert "Name: vllm" in pkg_info_text
    assert "Version: 0.16.0" in pkg_info_text


def test_inspect_runtime_env_reports_probe_results(tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    env_root = tmp_path / "runtime"
    _materialize_runtime_env_with_real_host_python(env_root)

    snapshot = build_runtime_env.inspect_runtime_env(
        env_root,
        probe_modules=["json", "pathlib"],
    )

    assert snapshot["env_root"] == str(env_root)
    assert snapshot["manifest_path"] == str(env_root / "manifest.json")
    assert snapshot["valid_layout"] is True
    assert snapshot["missing_paths"] == []
    assert snapshot["host_python"] == str(env_root / "host-venv" / "bin" / "python")
    assert snapshot["probe_results"]["json"]["ok"] is True
    assert snapshot["probe_results"]["pathlib"]["ok"] is True


def test_build_runtime_env_inspect_cli_returns_nonzero_on_probe_failure(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env_with_real_host_python(env_root)

    out = subprocess.run(
        [
            sys.executable,
            "scripts/build_runtime_env.py",
            "--inspect",
            "--env-root",
            str(env_root),
            "--probe-module",
            "json",
            "--probe-module",
            "does_not_exist_for_runtime_env_probe",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert out.returncode == 1
    snapshot = json.loads(out.stdout)
    assert snapshot["valid_layout"] is True
    assert snapshot["probe_results"]["json"]["ok"] is True
    assert snapshot["probe_results"]["does_not_exist_for_runtime_env_probe"]["ok"] is False


def test_runtime_env_layout_prefers_manifest_sources(tmp_path):
    env_root = tmp_path / "runtime"
    manifest = {
        "runtime_env": {
            "site_packages_dir": "site-packages",
            "source_dir": "src",
            "base_python_dir": "base-python",
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
    assert layout.base_python_root == str(env_root / "base-python")
    assert list(layout.host_pythonpath_entries) == [
        str(env_root / "src" / "HostOnlySource"),
    ]


def test_validate_runtime_env_layout_requires_base_python_when_host_python_required(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    shutil.rmtree(env_root / "base-python")
    with pytest.raises(RuntimeError, match="missing="):
        validate_runtime_env_layout(str(env_root), require_host_python=True)


def test_runtime_env_layout_requires_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="missing manifest.json"):
        runtime_env_layout(str(tmp_path / "runtime"))


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


def test_gateway_session_store_namespace_respects_config_file(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[ray]\nnamespace = 'cfg_ns'\n", encoding="utf-8")

    env = os.environ.copy()
    env["TINKER_CONFIG_PATH"] = str(cfg)
    env.pop("TINKER_RAY_NAMESPACE", None)
    env.pop("MINT_RAY_NAMESPACE", None)

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tinker_server.config as c; "
                "import tinker_server.backend.gateway_session_store as g; "
                "print(c.RAY_NAMESPACE); "
                "print(g._ray_namespace())"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.stdout.strip().splitlines() == ["cfg_ns", "cfg_ns"]


def test_detached_store_namespaces_respect_config_file(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[ray]\nnamespace = 'cfg_ns'\n", encoding="utf-8")

    env = os.environ.copy()
    env["TINKER_CONFIG_PATH"] = str(cfg)
    env.pop("TINKER_RAY_NAMESPACE", None)
    env.pop("MINT_RAY_NAMESPACE", None)

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tinker_server.config as c; "
                "import tinker_server.backend.gateway_session_store as g; "
                "import tinker_server.backend.session_index_store as s; "
                "import tinker_server.backend.training_session_store as t; "
                "print(c.RAY_NAMESPACE); "
                "print(g._ray_namespace()); "
                "print(s._ray_namespace()); "
                "print(t._ray_namespace())"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.stdout.strip().splitlines() == ["cfg_ns", "cfg_ns", "cfg_ns", "cfg_ns"]


def test_config_import_fails_on_namespace_mismatch(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[ray]\nnamespace = 'cfg_ns'\n", encoding="utf-8")

    env = os.environ.copy()
    env["TINKER_CONFIG_PATH"] = str(cfg)
    env["TINKER_RAY_NAMESPACE"] = "env_ns"

    out = subprocess.run(
        [sys.executable, "-c", "import tinker_server.config"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode != 0
    assert "Ray namespace mismatch" in (out.stdout + out.stderr)


def test_config_import_fails_on_runtime_path_mismatch(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "\n".join(
            [
                "[paths]",
                'pfs_runtime_env_root = "/cfg/runtime"',
                'pfs_tinker_path = "/cfg/repo"',
                'pfs_hf_modules_path = "/cfg/hf"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TINKER_CONFIG_PATH"] = str(cfg)
    env["PFS_RUNTIME_ENV_ROOT"] = "/env/runtime"
    env["PFS_TINKER_PATH"] = "/env/repo"
    env["PFS_HF_MODULES_PATH"] = "/env/hf"

    out = subprocess.run(
        [sys.executable, "-c", "import tinker_server.config"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode != 0
    assert "mismatch between environment and config file" in (out.stdout + out.stderr)


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


def test_run_server_honors_env_config_before_runtime_bootstrap(tmp_path):
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
        [sys.executable, "scripts/run_server.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={
            "TINKER_HOST": "127.0.0.1",
            "TINKER_CONFIG_PATH": str(cfg),
        },
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


def test_actor_runtime_env_vars_forwards_config_path(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    cfg = tmp_path / "tinker.toml"
    cfg.write_text(
        "\n".join(
            [
                "[paths]",
                f'pfs_runtime_env_root = "{env_root}"',
                f'pfs_tinker_path = "{tmp_path / "repo"}"',
                f'pfs_hf_modules_path = "{tmp_path / "hf"}"',
                "",
                "[ray]",
                "namespace = 'cfg_ns'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from tinker_server.config import actor_runtime_env_vars; "
                "print(json.dumps(actor_runtime_env_vars(pythonpath='X')))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PFS_RUNTIME_ENV_ROOT": str(env_root),
            "PFS_TINKER_PATH": str(tmp_path / 'repo'),
            "PFS_HF_MODULES_PATH": str(tmp_path / 'hf'),
            "RAY_ADDRESS": "ray://cfg-test",
            "TINKER_CONFIG_PATH": str(cfg),
        },
    )
    data = json.loads(out.stdout)
    assert data["RAY_ADDRESS"] == "ray://cfg-test"
    assert data["TINKER_CONFIG_PATH"] == str(cfg)
    assert data["TINKER_RAY_NAMESPACE"] == "cfg_ns"
