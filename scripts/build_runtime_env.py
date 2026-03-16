#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_server.runtime_env import (
    DEFAULT_HOST_VENV_DIRNAME,
    DEFAULT_SITE_PACKAGES_DIRNAME,
    DEFAULT_SOURCE_DIRNAME,
    runtime_env_layout,
)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _resolve_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("uv executable not found; install uv or add it to PATH")


def _load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _runtime_table(pyproject: dict[str, Any]) -> dict[str, Any]:
    return pyproject["tool"]["tinker"]["runtime_env"]


def _shared_deps(pyproject: dict[str, Any]) -> list[str]:
    return list(pyproject["project"]["dependencies"])


def _host_deps(pyproject: dict[str, Any]) -> list[str]:
    groups = pyproject.get("dependency-groups", {})
    return list(groups.get("host-runtime", []))


def _host_pip_index_args(pyproject: dict[str, Any]) -> list[str]:
    uv_cfg = pyproject.get("tool", {}).get("uv", {})
    sources = uv_cfg.get("sources", {})
    index_rows = uv_cfg.get("index", [])
    index_urls = {
        row["name"]: row["url"]
        for row in index_rows
        if isinstance(row, dict) and row.get("name") and row.get("url")
    }
    args: list[str] = []
    seen: set[str] = set()
    for dep in _host_deps(pyproject):
        name = dep.split(";", 1)[0].split("[", 1)[0].split("==", 1)[0].strip()
        source = sources.get(name)
        if not isinstance(source, dict):
            continue
        index_name = source.get("index")
        if not index_name:
            continue
        index_url = index_urls.get(index_name)
        if not index_url or index_url in seen:
            continue
        seen.add(index_url)
        args.extend(["--extra-index-url", index_url])
    return args


def _clone_checkout(target: Path, *, repo: str, commit: str) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", str(target)])
    _run(["git", "-C", str(target), "remote", "add", "origin", repo])
    _run(["git", "-C", str(target), "fetch", "--depth=1", "origin", commit])
    _run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"])


def _export_shared_requirements(pyproject: dict[str, Any], output: Path) -> None:
    runtime = _runtime_table(pyproject)
    prune_args: list[str] = []
    for name in runtime.get("image_managed", {}).keys():
        prune_args.extend(["--prune", name])
    _run(
        [
            _resolve_uv(),
            "export",
            "--frozen",
            "--no-hashes",
            "--no-emit-project",
            "--no-dev",
            "--no-default-groups",
            "--output-file",
            str(output),
            *prune_args,
        ],
        cwd=REPO_ROOT,
    )


def _install_target(python: Path, target: Path, requirements_file: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            "-r",
            str(requirements_file),
        ]
    )


def _create_host_venv(
    uv_python: str,
    host_venv: Path,
    host_deps: list[str],
    pip_index_args: list[str],
) -> Path:
    if host_venv.exists():
        shutil.rmtree(host_venv)
    _run([_resolve_uv(), "venv", "--seed", "--python", uv_python, str(host_venv)])
    python = host_venv / "bin" / "python"
    if host_deps:
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([str(python), "-m", "pip", "install", *pip_index_args, *host_deps])
    return python


def _write_manifest(env_root: Path, pyproject: dict[str, Any], host_python: Path, shared_deps: list[str]) -> None:
    runtime = _runtime_table(pyproject)
    manifest = {
        "python_version": runtime["python_version"],
        "env_root": str(env_root),
        "host_python": str(host_python),
        "shared_dependencies": shared_deps,
        "host_dependencies": _host_deps(pyproject),
        "sources": runtime["sources"],
        "image_managed": runtime["image_managed"],
    }
    (env_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_activation(env_root: Path) -> None:
    layout = runtime_env_layout(str(env_root))
    text = "\n".join(
        [
            "# Generated by scripts/build_runtime_env.py",
            f"export PFS_RUNTIME_ENV_ROOT={layout.root}",
            f"export TINKER_HOST_PYTHON={layout.host_python}",
            "",
        ]
    )
    (env_root / "activate_runtime_env.sh").write_text(text, encoding="utf-8")


def _write_host_pth(env_root: Path, host_python: Path) -> None:
    layout = runtime_env_layout(str(env_root))
    purelib = subprocess.check_output(
        [str(host_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    pth = Path(purelib) / "tinker_runtime_env.pth"
    lines = [layout.site_packages, *layout.pythonpath_entries[1:]]
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_host_wrappers(env_root: Path, host_python: Path) -> None:
    bindir = host_python.parent
    wrappers = {
        "ray": "ray.scripts.scripts",
    }
    for name, module in wrappers.items():
        script = bindir / name
        script.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    f'exec "{host_python}" -m {module} "$@"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        script.chmod(0o755)


def build_runtime_env(env_root: Path) -> None:
    pyproject = _load_pyproject()
    runtime = _runtime_table(pyproject)
    shared_deps = _shared_deps(pyproject)
    host_deps = _host_deps(pyproject)
    host_pip_index_args = _host_pip_index_args(pyproject)
    env_root.mkdir(parents=True, exist_ok=True)
    shared_site_packages = env_root / runtime.get("site_packages_dir", DEFAULT_SITE_PACKAGES_DIRNAME)
    host_venv = env_root / runtime.get("host_venv_dir", DEFAULT_HOST_VENV_DIRNAME)
    source_root = env_root / runtime.get("source_dir", DEFAULT_SOURCE_DIRNAME)
    shared_requirements = env_root / "shared-requirements.txt"

    host_python = _create_host_venv(
        runtime["python_version"],
        host_venv,
        host_deps,
        host_pip_index_args,
    )
    _export_shared_requirements(pyproject, shared_requirements)
    _install_target(host_python, shared_site_packages, shared_requirements)

    for source in runtime["sources"]:
        _clone_checkout(
            source_root / source["name"],
            repo=source["repo"],
            commit=source["commit"],
        )

    _write_host_pth(env_root, host_python)
    _write_host_wrappers(env_root, host_python)
    _write_manifest(env_root, pyproject, host_python, shared_deps)
    _write_activation(env_root)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-root", required=True, help="Destination PFS runtime env root")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    build_runtime_env(Path(args.env_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
