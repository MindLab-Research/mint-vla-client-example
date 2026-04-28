from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

import pytest
import tomllib
import subprocess
import sys

import tinker_server.config as server_config
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
    lerobot_requirement = (
        "lerobot @ git+https://github.com/huggingface/lerobot.git"
        "@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
    )

    for requirement in (
        "augmax>=0.3.4",
        "beartype==0.19.0",
        "flax==0.10.2",
        "filelock>=3.16.1",
        "fsspec[gcs]>=2024.6.0",
        "jax[cuda12]==0.5.3",
        "jaxtyping==0.2.36",
        lerobot_requirement,
        "ml_collections==1.0.0",
        "numpydantic>=1.6.6",
        "optax==0.2.4",
        "orbax-checkpoint==0.11.13",
        "pytest>=7.0.0",
        "tqdm-loggable>=0.2",
        "tyro>=0.9.5",
    ):
        assert requirement in deps


def test_subprocess_env_sets_default_uv_http_timeout(monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    monkeypatch.delenv("UV_HTTP_TIMEOUT", raising=False)

    env = build_runtime_env._subprocess_env()

    assert env["UV_HTTP_TIMEOUT"] == "300"


def test_subprocess_env_respects_existing_uv_http_timeout(monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    monkeypatch.setenv("UV_HTTP_TIMEOUT", "900")

    env = build_runtime_env._subprocess_env()

    assert env["UV_HTTP_TIMEOUT"] == "900"


def test_resolve_uv_prefers_explicit_uv_bin_override(tmp_path, monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    fake_uv = tmp_path / "bin" / "uv"
    fake_uv.parent.mkdir(parents=True, exist_ok=True)
    fake_uv.write_text("", encoding="utf-8")
    fake_uv.chmod(0o755)

    monkeypatch.setenv("UV_BIN", str(fake_uv))
    monkeypatch.setattr(build_runtime_env.shutil, "which", lambda _: "/usr/bin/uv")

    assert build_runtime_env._resolve_uv() == str(fake_uv)


def test_subprocess_env_sets_tmpdir_under_xdg_cache_home(monkeypatch, tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    xdg_cache_home = tmp_path / "cache"
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_home))

    env = build_runtime_env._subprocess_env()

    assert env["TMPDIR"] == str(xdg_cache_home / "tmp")
    assert (xdg_cache_home / "tmp").is_dir()


def test_subprocess_env_sets_uv_cache_dir_under_xdg_cache_home(monkeypatch, tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    xdg_cache_home = tmp_path / "cache"
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_home))

    env = build_runtime_env._subprocess_env()

    assert env["UV_CACHE_DIR"] == str(xdg_cache_home / "uv")


def test_subprocess_env_respects_existing_tmpdir(monkeypatch, tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    custom_tmpdir = tmp_path / "custom-tmp"
    monkeypatch.setenv("TMPDIR", str(custom_tmpdir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    env = build_runtime_env._subprocess_env()

    assert env["TMPDIR"] == str(custom_tmpdir)


def test_export_host_requirements_writes_runtime_worker_stack(tmp_path):
    from scripts import build_runtime_env as build_runtime_env

    out = tmp_path / "host-requirements.txt"
    build_runtime_env._export_host_requirements(build_runtime_env._load_pyproject(), out)
    text = out.read_text(encoding="utf-8")
    lerobot_requirement = (
        "lerobot @ git+https://github.com/huggingface/lerobot.git"
        "@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
    )

    for requirement in (
        "torch==2.9.1+cpu",
        "jax[cuda12]==0.5.3",
        "flax==0.10.2",
        "optax==0.2.4",
        "orbax-checkpoint==0.11.13",
        "ml_collections==1.0.0",
        "jaxtyping==0.2.36",
        "augmax>=0.3.4",
        lerobot_requirement,
        "pytest>=7.0.0",
        "tqdm-loggable>=0.2",
        "tyro>=0.9.5",
    ):
        assert requirement in text


def test_default_inspect_probe_modules_cover_openpi_training_loader():
    from scripts import build_runtime_env as build_runtime_env

    assert "openpi.training.data_loader" in build_runtime_env.DEFAULT_INSPECT_PROBE_MODULES


def test_partition_host_requirements_keeps_only_torch_on_torch_backend():
    from scripts import build_runtime_env as build_runtime_env

    torch_backend_reqs, generic_reqs = build_runtime_env._partition_host_requirements(
        [
            "torch==2.9.1+cpu",
            "flax==0.10.2",
            "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        ]
    )

    assert torch_backend_reqs == ["torch==2.9.1+cpu"]
    assert generic_reqs == [
        "flax==0.10.2",
        "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
    ]


def test_create_host_venv_installs_torch_backend_and_generic_requirements_separately(tmp_path, monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    base_python = tmp_path / "base-python" / "bin" / "python3.12"
    base_python.parent.mkdir(parents=True, exist_ok=True)
    base_python.write_text("", encoding="utf-8")
    host_venv = tmp_path / "host-venv"
    host_requirements = tmp_path / "host-requirements.txt"
    host_requirements.write_text(
        "\n".join(
            [
                "torch==2.9.1+cpu",
                "flax==0.10.2",
                "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None):
        calls.append(cmd)
        if cmd[:4] == [str(base_python), "-m", "venv", "--copies"]:
            python = host_venv / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")

    monkeypatch.setattr(build_runtime_env, "_run", fake_run)
    monkeypatch.setattr(build_runtime_env, "_resolve_uv", lambda: "/fake/uv")

    python = build_runtime_env._create_host_venv(base_python, host_venv, host_requirements)

    generic_requirements = tmp_path / "host-requirements-generic.txt"
    assert python == host_venv / "bin" / "python"
    assert calls[0] == [str(base_python), "-m", "venv", "--copies", str(host_venv)]
    assert calls[1] == [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    assert calls[2] == [
        "/fake/uv",
        "pip",
        "install",
        "--python",
        str(python),
        "torch==2.9.1+cpu",
        "--torch-backend",
        "cpu",
    ]
    assert calls[3] == [
        "/fake/uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--torch-backend",
        "cpu",
        "--requirements",
        str(generic_requirements),
    ]
    assert generic_requirements.read_text(encoding="utf-8").splitlines() == [
        "flax==0.10.2",
        "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
    ]


def test_write_host_pth_prepends_runtime_paths(tmp_path, monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    purelib = tmp_path / "purelib"
    purelib.mkdir()

    monkeypatch.setattr(
        build_runtime_env.subprocess,
        "check_output",
        lambda *args, **kwargs: str(purelib),
    )

    build_runtime_env._write_host_pth(env_root, env_root / "host-venv" / "bin" / "python")

    line = (purelib / "tinker_runtime_env.pth").read_text(encoding="utf-8").strip()
    original_sys_path = sys.path[:]
    sys.path[:] = [
        "/host/site-packages",
        str(env_root / "src" / "openpi" / "src"),
        "/other",
    ]
    try:
        exec(line, {})
        actual = sys.path[:]
    finally:
        sys.path[:] = original_sys_path

    expected = [
        str(env_root / "site-packages"),
        str(env_root / "src" / "Megatron-LM"),
        str(env_root / "src" / "Megatron-Bridge" / "src"),
        str(env_root / "src" / "Megatron-Bridge"),
        str(env_root / "src" / "verl"),
        str(env_root / "src" / "openpi" / "src"),
        str(env_root / "src" / "openpi" / "packages" / "openpi-client" / "src"),
        str(env_root / "src" / "vllm"),
    ]
    assert actual[: len(expected)] == expected
    assert actual.count(str(env_root / "src" / "openpi" / "src")) == 1


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


def test_materialize_base_python_uses_current_uv_find_args(tmp_path, monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    bootstrap_root = tmp_path / "bootstrap-root"
    bootstrap_python = bootstrap_root / "bin" / "python3.12"
    bootstrap_python.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_python.write_text("", encoding="utf-8")
    seen: list[list[str]] = []

    def fake_capture(cmd, *, cwd=None, env=None):
        seen.append(cmd)
        return str(bootstrap_python)

    monkeypatch.setattr(build_runtime_env, "sys", type("FakeSys", (), {"version_info": (3, 12, 13), "_base_executable": "/missing/python", "executable": "/missing/python"})())
    monkeypatch.setattr(build_runtime_env, "_resolve_uv", lambda: "/tmp/fake-uv")
    monkeypatch.setattr(build_runtime_env, "_capture", fake_capture)

    materialized = build_runtime_env._materialize_base_python("3.12.14", tmp_path / "runtime" / "base-python")

    assert seen == [["/tmp/fake-uv", "python", "find", "--managed-python", "3.12.14"]]
    assert materialized == tmp_path / "runtime" / "base-python" / "bin" / "python3.12"
    assert materialized.exists()


def test_materialize_base_python_reuses_current_base_executable(tmp_path, monkeypatch):
    from scripts import build_runtime_env as build_runtime_env

    bootstrap_root = tmp_path / "bootstrap-root"
    bootstrap_python = bootstrap_root / "bin" / "python3.12"
    bootstrap_python.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        build_runtime_env,
        "sys",
        type(
            "FakeSys",
            (),
            {
                "version_info": (3, 12, 13),
                "_base_executable": str(bootstrap_python),
                "executable": str(tmp_path / "venv" / "bin" / "python3.12"),
            },
        )(),
    )

    def fail_resolve_uv():
        raise AssertionError("uv lookup should not run when current base executable matches")

    monkeypatch.setattr(build_runtime_env, "_resolve_uv", fail_resolve_uv)

    materialized = build_runtime_env._materialize_base_python("3.12.13", tmp_path / "runtime" / "base-python")

    assert materialized == tmp_path / "runtime" / "base-python" / "bin" / "python3.12"
    assert materialized.exists()


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


def test_preferred_vllm_python_executable_prefers_explicit_env(monkeypatch, tmp_path):
    repo_root = tmp_path / "local-tinker"
    worker_wrapper = repo_root / "scripts" / "vllm_worker_python.py"
    worker_wrapper.parent.mkdir(parents=True, exist_ok=True)
    worker_wrapper.write_text("#!/bin/sh\n", encoding="utf-8")

    explicit = tmp_path / "shared" / "scripts" / "vllm_worker_python.py"
    explicit.parent.mkdir(parents=True, exist_ok=True)
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(server_config, "PFS_TINKER_PATH", str(repo_root))
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(explicit))

    assert server_config.preferred_vllm_python_executable() == str(explicit)


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
                "import tinker_server.backend.sampling_session_store as p; "
                "import tinker_server.backend.session_index_store as s; "
                "import tinker_server.backend.training_session_store as t; "
                "print(c.RAY_NAMESPACE); "
                "print(g._ray_namespace()); "
                "print(p._ray_namespace()); "
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
    assert out.stdout.strip().splitlines() == ["cfg_ns", "cfg_ns", "cfg_ns", "cfg_ns", "cfg_ns"]


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


def test_actor_runtime_env_vars_forwards_control_plane_pin_envs(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
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
            "MINT_CONTROL_PLANE_PINNED_NODE_IP": "192.168.38.176",
            "MINT_API_WORK_QUEUE_PINNED_NODE_IP": "192.168.38.176",
            "MINT_STARTUP_LEASE_PINNED_NODE_IP": "192.168.38.176",
        },
    )
    data = json.loads(out.stdout)
    assert data["MINT_CONTROL_PLANE_PINNED_NODE_IP"] == "192.168.38.176"
    assert data["MINT_API_WORK_QUEUE_PINNED_NODE_IP"] == "192.168.38.176"
    assert data["MINT_STARTUP_LEASE_PINNED_NODE_IP"] == "192.168.38.176"


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
            "MINT_DETACHED_ACTOR_NODE_IP": "192.168.38.175",
            "MINT_API_WORK_QUEUE_ACTOR_NAME": "issue440-api-work-queue",
            "MINT_CAPACITY_MANAGER_ACTOR_NAME": "issue440-capacity-manager",
        },
    )
    data = json.loads(out.stdout)
    assert data["RAY_ADDRESS"] == "ray://cfg-test"
    assert data["TINKER_CONFIG_PATH"] == str(cfg)
    assert data["TINKER_RAY_NAMESPACE"] == "cfg_ns"
    assert data["MINT_DETACHED_ACTOR_NODE_IP"] == "192.168.38.175"
    assert data["MINT_API_WORK_QUEUE_ACTOR_NAME"] == "issue440-api-work-queue"
    assert data["MINT_CAPACITY_MANAGER_ACTOR_NAME"] == "issue440-capacity-manager"


def test_server_config_prefers_mint_actor_names_and_accepts_legacy_tinker_aliases():
    cfg = server_config.ServerConfig.from_sources(
        environ={
            "TINKER_API_KEY": "dev-key",
            "MINT_API_WORK_QUEUE_ACTOR_NAME": "mint-api",
            "TINKER_API_WORK_QUEUE_ACTOR_NAME": "legacy-api",
            "MINT_CAPACITY_MANAGER_ACTOR_NAME": "mint-cap",
            "TINKER_CAPACITY_MANAGER_ACTOR_NAME": "legacy-cap",
        },
        config_path=None,
        config_file=None,
    )

    assert cfg.api_work_queue_actor_name == "mint-api"
    assert cfg.capacity_manager_actor_name == "mint-cap"

    legacy_only = server_config.ServerConfig.from_sources(
        environ={
            "TINKER_API_KEY": "dev-key",
            "TINKER_API_WORK_QUEUE_ACTOR_NAME": "legacy-api",
            "TINKER_CAPACITY_MANAGER_ACTOR_NAME": "legacy-cap",
        },
        config_path=None,
        config_file=None,
    )

    assert legacy_only.api_work_queue_actor_name == "legacy-api"
    assert legacy_only.capacity_manager_actor_name == "legacy-cap"


def test_actor_runtime_env_vars_requires_ray_address(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tinker_server.config import actor_runtime_env_vars; "
                "actor_runtime_env_vars(pythonpath='X')"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={
            "PFS_RUNTIME_ENV_ROOT": str(env_root),
            "PFS_TINKER_PATH": str(tmp_path / 'repo'),
            "PFS_HF_MODULES_PATH": str(tmp_path / 'hf'),
        },
    )
    assert out.returncode != 0
    assert "RAY_ADDRESS is required" in (out.stdout + out.stderr)


def test_actor_runtime_env_vars_canonicalize_legacy_tinker_actor_aliases(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
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
            "TINKER_API_WORK_QUEUE_ACTOR_NAME": "legacy-api-work-queue",
            "TINKER_CAPACITY_MANAGER_ACTOR_NAME": "legacy-capacity-manager",
        },
    )
    data = json.loads(out.stdout)
    assert data["MINT_API_WORK_QUEUE_ACTOR_NAME"] == "legacy-api-work-queue"
    assert data["MINT_CAPACITY_MANAGER_ACTOR_NAME"] == "legacy-capacity-manager"
    assert "TINKER_API_WORK_QUEUE_ACTOR_NAME" not in data
    assert "TINKER_CAPACITY_MANAGER_ACTOR_NAME" not in data


def test_actor_runtime_env_vars_forwards_usage_envs(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
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
            "TINKER_USAGE_LOG_DIR": "/vePFS/shared/usage",
            "TINKER_USAGE_BACKEND": "postgres",
            "TINKER_USAGE_PG_DSN": "postgresql://mint:test@db/usage",
        },
    )
    data = json.loads(out.stdout)
    assert data["TINKER_USAGE_LOG_DIR"] == "/vePFS/shared/usage"
    assert data["TINKER_USAGE_BACKEND"] == "postgres"
    assert data["TINKER_USAGE_PG_DSN"] == "postgresql://mint:test@db/usage"


def test_actor_runtime_env_vars_forwards_ray_attach_hints(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
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
            "RAY_ADDRESS": "192.168.39.87:6379",
            "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.39.87:10001",
            "RAY_CLIENT_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_RAY_NODE_IP_ADDRESS": "192.168.33.190",
            "MINT_RAY_TEMP_DIR": "/tmp/mdw/t",
        },
    )
    data = json.loads(out.stdout)
    assert data["RAY_ADDRESS"] == "192.168.39.87:6379"
    assert data["MINT_RAY_CLIENT_ADDRESS"] == "ray://192.168.39.87:10001"
    assert data["RAY_CLIENT_ADDRESS"] == "ray://192.168.39.87:10001"
    assert data["MINT_RAY_NODE_IP_ADDRESS"] == "192.168.33.190"
    assert data["MINT_RAY_TEMP_DIR"] == "/tmp/mdw/t"


def test_actor_runtime_env_skips_local_working_dir_in_ray_client_mode(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    local_repo = tmp_path / "repo"
    local_repo.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from tinker_server.config import actor_runtime_env; "
                "print(json.dumps(actor_runtime_env(pythonpath='X')))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PFS_RUNTIME_ENV_ROOT": str(env_root),
            "PFS_TINKER_PATH": str(local_repo),
            "PFS_HF_MODULES_PATH": str(tmp_path / "hf"),
            "RAY_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_RAY_WORKING_DIR": str(local_repo),
        },
    )
    data = json.loads(out.stdout)
    assert "working_dir" not in data


def test_actor_runtime_env_skips_local_py_modules_in_ray_client_mode(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    repo_pkg = tmp_path / "repo" / "tinker_server"
    repo_pkg.mkdir(parents=True)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from tinker_server.config import actor_runtime_env; "
                "print(json.dumps(actor_runtime_env(pythonpath='X')))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PFS_RUNTIME_ENV_ROOT": str(env_root),
            "PFS_TINKER_PATH": str(tmp_path / "repo"),
            "PFS_HF_MODULES_PATH": str(tmp_path / "hf"),
            "RAY_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_RAY_PY_MODULES_CSV": str(repo_pkg),
        },
    )
    data = json.loads(out.stdout)
    assert "py_modules" not in data


def test_actor_runtime_env_keeps_local_working_dir_for_direct_ray(tmp_path):
    env_root = tmp_path / "runtime"
    _materialize_runtime_env(env_root)
    local_repo = tmp_path / "repo"
    local_repo.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from tinker_server.config import actor_runtime_env; "
                "print(json.dumps(actor_runtime_env(pythonpath='X')))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PFS_RUNTIME_ENV_ROOT": str(env_root),
            "PFS_TINKER_PATH": str(local_repo),
            "PFS_HF_MODULES_PATH": str(tmp_path / "hf"),
            "RAY_ADDRESS": "192.168.39.87:6379",
            "MINT_RAY_WORKING_DIR": str(local_repo),
        },
    )
    data = json.loads(out.stdout)
    assert data["working_dir"] == str(local_repo)
