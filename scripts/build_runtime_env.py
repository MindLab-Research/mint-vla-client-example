#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
DEFAULT_INSPECT_PROBE_MODULES = (
    "openpi.training.config",
    "openpi.training.data_loader",
    "openpi_client.image_tools",
    "jax",
    "flax",
    "optax",
    "orbax.checkpoint",
)
DEFAULT_UV_HTTP_TIMEOUT = "300"
DEFAULT_MINT_ROOT = Path("/vePFS-Mindverse/share/mint")
MINT_ENVS = ("prod", "dev")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged.setdefault("UV_HTTP_TIMEOUT", DEFAULT_UV_HTTP_TIMEOUT)
    xdg_cache_home = merged.get("XDG_CACHE_HOME")
    if "UV_CACHE_DIR" not in merged and xdg_cache_home:
        uv_cache_dir = Path(xdg_cache_home) / "uv"
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        merged["UV_CACHE_DIR"] = str(uv_cache_dir)
    if "TMPDIR" not in merged:
        if xdg_cache_home:
            tmpdir = Path(xdg_cache_home) / "tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            merged["TMPDIR"] = str(tmpdir)
    return merged


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=_subprocess_env(env), check=True)


def _capture(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, env=_subprocess_env(env), text=True)


def _resolve_uv() -> str:
    explicit = (os.environ.get("UV_BIN") or "").strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return str(candidate)
        raise RuntimeError(f"UV_BIN points to a missing path: {explicit}")
    uv = shutil.which("uv")
    if uv:
        return uv
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("uv executable not found; set UV_BIN, install uv, or add it to PATH")


def _load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _runtime_table(pyproject: dict[str, Any]) -> dict[str, Any]:
    return pyproject["tool"]["tinker"]["runtime_env"]


def _shared_deps(pyproject: dict[str, Any]) -> list[str]:
    return list(pyproject["project"]["dependencies"])


def _host_deps(pyproject: dict[str, Any]) -> list[str]:
    groups = pyproject.get("dependency-groups", {})
    deps = list(groups.get("host-runtime", []))
    deps.extend(_runtime_table(pyproject).get("host_requirements", []))
    seen: set[str] = set()
    out: list[str] = []
    for dep in deps:
        if dep in seen:
            continue
        seen.add(dep)
        out.append(dep)
    return out


def _requirement_name(requirement: str) -> str:
    name = requirement.split(";", 1)[0].strip()
    name = name.split("@", 1)[0].strip()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        name = name.split(marker, 1)[0].strip()
    name = name.split("[", 1)[0].strip()
    return name.lower().replace("_", "-")


def _partition_host_requirements(requirements: list[str]) -> tuple[list[str], list[str]]:
    torch_backend_requirements: list[str] = []
    generic_requirements: list[str] = []
    for requirement in requirements:
        if _requirement_name(requirement) == "torch":
            torch_backend_requirements.append(requirement)
            continue
        generic_requirements.append(requirement)
    return torch_backend_requirements, generic_requirements


def _runtime_env_symbols():
    from tinker_server.runtime_env import (
        DEFAULT_BASE_PYTHON_DIRNAME,
        DEFAULT_HOST_VENV_DIRNAME,
        DEFAULT_SITE_PACKAGES_DIRNAME,
        DEFAULT_SOURCE_DIRNAME,
        checkout_runtime_env_layout,
        runtime_env_layout,
    )

    return {
        "DEFAULT_BASE_PYTHON_DIRNAME": DEFAULT_BASE_PYTHON_DIRNAME,
        "DEFAULT_HOST_VENV_DIRNAME": DEFAULT_HOST_VENV_DIRNAME,
        "DEFAULT_SITE_PACKAGES_DIRNAME": DEFAULT_SITE_PACKAGES_DIRNAME,
        "DEFAULT_SOURCE_DIRNAME": DEFAULT_SOURCE_DIRNAME,
        "checkout_runtime_env_layout": checkout_runtime_env_layout,
        "runtime_env_layout": runtime_env_layout,
    }


def _load_manifest(env_root: Path) -> dict[str, Any]:
    manifest_path = env_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest.json under {env_root}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _git_short_sha() -> str:
    try:
        return _capture(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT).strip()
    except Exception:
        return "nogit"


def _default_build_id() -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"py31213-{timestamp}-{_git_short_sha()}"


def _mint_env_root(mint_root: Path, mint_env: str) -> Path:
    if mint_env not in MINT_ENVS:
        raise ValueError(f"unsupported Mint runtime env {mint_env!r}; expected one of {MINT_ENVS!r}")
    return mint_root / mint_env


def _mint_runtime_build_root(mint_root: Path, mint_env: str, build_id: str) -> Path:
    return _mint_env_root(mint_root, mint_env) / "runtime-builds" / build_id


def _mint_runtime_link(mint_root: Path, mint_env: str) -> Path:
    return _mint_env_root(mint_root, mint_env) / "runtime"


def _promote_runtime_symlink(runtime_root: Path, link_path: Path) -> None:
    runtime_root = runtime_root.resolve()
    if not runtime_root.is_dir():
        raise RuntimeError(f"runtime root does not exist: {runtime_root}")
    if not (runtime_root / "manifest.json").is_file():
        raise RuntimeError(f"runtime root is missing manifest.json: {runtime_root}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    next_link = link_path.with_name(f".{link_path.name}.next-{os.getpid()}")
    if next_link.exists() or next_link.is_symlink():
        next_link.unlink()
    next_link.symlink_to(runtime_root)
    os.replace(next_link, link_path)


def copy_runtime_env(src_root: Path, dst_root: Path) -> None:
    src_root = src_root.resolve()
    if not src_root.is_dir():
        raise RuntimeError(f"source runtime root does not exist: {src_root}")
    if not (src_root / "manifest.json").is_file():
        raise RuntimeError(f"source runtime root is missing manifest.json: {src_root}")
    if dst_root.exists() or dst_root.is_symlink():
        raise RuntimeError(f"destination runtime root already exists: {dst_root}")
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root, symlinks=True)
    _load_manifest(dst_root)


def _required_runtime_paths(layout) -> list[str]:
    return [
        layout.site_packages,
        *layout.pythonpath_entries[1:],
        layout.base_python_root,
        layout.host_venv_root,
        layout.host_python,
        *layout.host_pythonpath_entries,
    ]


def _probe_module(host_python: str, module: str) -> dict[str, Any]:
    out = subprocess.run(
        [
            host_python,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module(sys.argv[1])"
            ),
            module,
        ],
        capture_output=True,
        text=True,
    )
    detail = (out.stderr or out.stdout).strip()
    result = {
        "ok": out.returncode == 0,
    }
    if detail:
        result["detail"] = detail
    return result

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


def _export_host_requirements(pyproject: dict[str, Any], output: Path) -> None:
    lines = [
        "# Direct host requirements for the Mint runtime host venv.",
        *(_host_deps(pyproject)),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


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


def _materialize_base_python(
    python_request: str,
    base_python_root: Path,
) -> Path:
    requested = tuple(int(part) for part in python_request.split("."))
    current = tuple(sys.version_info[: len(requested)])
    if current == requested:
        base_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        if base_executable.exists():
            bootstrap_root = base_executable.parent.parent
        else:
            bootstrap_root = None
    else:
        bootstrap_root = None
    if bootstrap_root is None:
        uv = _resolve_uv()
        find_cmd = [uv, "python", "find", "--managed-python", python_request]
        try:
            bootstrap_python = Path(_capture(find_cmd, cwd=REPO_ROOT).strip())
        except subprocess.CalledProcessError:
            _run([uv, "python", "install", python_request], cwd=REPO_ROOT)
            bootstrap_python = Path(_capture(find_cmd, cwd=REPO_ROOT).strip())
        if not bootstrap_python.exists():
            raise RuntimeError(f"uv python find returned missing interpreter: {bootstrap_python}")
        bootstrap_root = bootstrap_python.resolve().parent.parent
    if base_python_root.exists():
        shutil.rmtree(base_python_root)
    shutil.copytree(bootstrap_root, base_python_root)
    materialized_python = base_python_root / "bin" / "python3.12"
    if not materialized_python.exists():
        executables = sorted((base_python_root / "bin").glob("python*"))
        if executables:
            materialized_python = executables[0]
    if not materialized_python.exists():
        raise RuntimeError(
            f"materialized base python missing after copy: {materialized_python}"
        )
    return materialized_python


def _create_host_venv(
    base_python: Path,
    host_venv: Path,
    host_requirements: Path,
) -> Path:
    if host_venv.exists():
        shutil.rmtree(host_venv)
    _run([str(base_python), "-m", "venv", "--copies", str(host_venv)])
    python = host_venv / "bin" / "python"
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    requirements = [
        line.strip()
        for line in host_requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    torch_backend_requirements, generic_requirements = _partition_host_requirements(requirements)
    if torch_backend_requirements:
        _run(
            [
                _resolve_uv(),
                "pip",
                "install",
                "--python",
                str(python),
                *torch_backend_requirements,
                "--torch-backend",
                "cpu",
            ],
            cwd=REPO_ROOT,
        )
    if generic_requirements:
        generic_requirements_path = host_requirements.with_name(
            f"{host_requirements.stem}-generic{host_requirements.suffix}"
        )
        generic_requirements_path.write_text("\n".join([*generic_requirements, ""]), encoding="utf-8")
        _run(
            [
                _resolve_uv(),
                "pip",
                "install",
                "--python",
                str(python),
                "--torch-backend",
                "cpu",
                "--requirements",
                str(generic_requirements_path),
            ],
            cwd=REPO_ROOT,
        )
    return python


def _write_manifest(env_root: Path, pyproject: dict[str, Any], host_python: Path, shared_deps: list[str]) -> None:
    runtime = _runtime_table(pyproject)
    runtime_env = _runtime_env_symbols()
    manifest = {
        "python_version": runtime["python_version"],
        "env_root": str(env_root),
        "host_python": str(host_python),
        "shared_dependencies": shared_deps,
        "host_dependencies": _host_deps(pyproject),
        "runtime_env": {
            "site_packages_dir": runtime.get(
                "site_packages_dir", runtime_env["DEFAULT_SITE_PACKAGES_DIRNAME"]
            ),
            "source_dir": runtime.get("source_dir", runtime_env["DEFAULT_SOURCE_DIRNAME"]),
            "base_python_dir": runtime.get("base_python_dir", runtime_env["DEFAULT_BASE_PYTHON_DIRNAME"]),
            "host_venv_dir": runtime.get("host_venv_dir", runtime_env["DEFAULT_HOST_VENV_DIRNAME"]),
        },
        "sources": runtime["sources"],
        "image_managed": runtime["image_managed"],
    }
    (env_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_activation(env_root: Path) -> None:
    layout = _runtime_env_symbols()["checkout_runtime_env_layout"](str(env_root))
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
    layout = _runtime_env_symbols()["checkout_runtime_env_layout"](str(env_root))
    purelib = subprocess.check_output(
        [str(host_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    pth = Path(purelib) / "tinker_runtime_env.pth"
    lines = [layout.site_packages, *layout.pythonpath_entries[1:], *layout.host_pythonpath_entries]
    code = (
        "import sys; "
        f"paths = {lines!r}; "
        "sys.path[:] = [p for p in sys.path if p not in paths]; "
        "sys.path[:0] = paths"
    )
    pth.write_text(code + "\n", encoding="utf-8")


def _write_host_source_dist_info(pyproject: dict[str, Any], host_python: Path) -> None:
    runtime = _runtime_table(pyproject)
    purelib = Path(
        subprocess.check_output(
            [str(host_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    for source in runtime.get("sources", []):
        if not source.get("host_only", False):
            continue
        package_name = str(source.get("package_name") or source["name"]).strip()
        package_version = str(source.get("version") or "").strip()
        if not package_name or not package_version:
            raise RuntimeError(
                f"host_only runtime source must declare package_name and version: {source!r}"
            )
        dist = purelib / f"{package_name.replace('-', '_')}-{package_version}.dist-info"
        if dist.exists():
            shutil.rmtree(dist)
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text(
            "\n".join(
                [
                    "Metadata-Version: 2.1",
                    f"Name: {package_name}",
                    f"Version: {package_version}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (dist / "WHEEL").write_text(
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: scripts/build_runtime_env.py",
                    "Root-Is-Purelib: true",
                    "Tag: py3-none-any",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (dist / "top_level.txt").write_text(f"{package_name.split('-', 1)[0]}\n", encoding="utf-8")
        (dist / "INSTALLER").write_text("scripts/build_runtime_env.py\n", encoding="utf-8")


def _write_host_source_version_files(pyproject: dict[str, Any], env_root: Path) -> None:
    runtime = _runtime_table(pyproject)
    runtime_env = _runtime_env_symbols()
    source_root = env_root / runtime.get("source_dir", runtime_env["DEFAULT_SOURCE_DIRNAME"])
    for source in runtime.get("sources", []):
        if not source.get("host_only", False):
            continue
        package_name = str(source.get("package_name") or source["name"]).strip()
        package_version = str(source.get("version") or "").strip()
        if not package_name or not package_version:
            raise RuntimeError(
                f"host_only runtime source must declare package_name and version: {source!r}"
            )
        package_root = source_root / source["name"] / package_name.replace("-", "_")
        if package_name == "vllm":
            version_file = package_root / "_version.py"
            version_tuple = package_version.split(".")
            if len(version_tuple) < 3:
                raise RuntimeError(f"vllm host version must be x.y.z: {package_version!r}")
            major, minor, patch = (int(part) for part in version_tuple[:3])
            version_file.write_text(
                "\n".join(
                    [
                        "# Generated by scripts/build_runtime_env.py",
                        f'__version__ = "{package_version}"',
                        f"__version_tuple__ = ({major}, {minor}, {patch})",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        egg_info = source_root / source["name"] / f"{package_name.replace('-', '_')}.egg-info"
        if egg_info.exists():
            (egg_info / "PKG-INFO").write_text(
                "\n".join(
                    [
                        "Metadata-Version: 2.1",
                        f"Name: {package_name}",
                        f"Version: {package_version}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )


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


def inspect_runtime_env(
    env_root: Path,
    *,
    probe_modules: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    env_root = env_root.resolve()
    runtime_env = _runtime_env_symbols()
    snapshot: dict[str, Any] = {
        "env_root": str(env_root),
        "manifest_path": str(env_root / "manifest.json"),
        "manifest_present": False,
        "valid_layout": False,
        "missing_paths": [],
        "probe_modules": list(probe_modules or DEFAULT_INSPECT_PROBE_MODULES),
        "probe_results": {},
    }
    try:
        manifest = _load_manifest(env_root)
    except Exception as exc:
        snapshot["layout_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot

    snapshot["manifest_present"] = True
    snapshot["runtime_env"] = manifest.get("runtime_env", {})
    snapshot["sources"] = [source.get("name", "") for source in manifest.get("sources", [])]

    layout = runtime_env["runtime_env_layout"](str(env_root))
    snapshot["host_python"] = layout.host_python
    snapshot["site_packages"] = layout.site_packages
    snapshot["pythonpath_entries"] = list(layout.pythonpath_entries)
    snapshot["host_pythonpath_entries"] = list(layout.host_pythonpath_entries)

    missing_paths = [path for path in _required_runtime_paths(layout) if not Path(path).exists()]
    snapshot["missing_paths"] = missing_paths
    snapshot["valid_layout"] = not missing_paths
    if missing_paths:
        snapshot["layout_error"] = (
            "PFS runtime env root is incomplete. "
            f"root={env_root!s} missing={missing_paths!r}"
        )
        return snapshot

    for module in snapshot["probe_modules"]:
        snapshot["probe_results"][module] = _probe_module(layout.host_python, module)
    return snapshot


def build_runtime_env(env_root: Path) -> None:
    pyproject = _load_pyproject()
    runtime = _runtime_table(pyproject)
    runtime_env = _runtime_env_symbols()
    shared_deps = _shared_deps(pyproject)
    env_root.mkdir(parents=True, exist_ok=True)
    shared_site_packages = env_root / runtime.get(
        "site_packages_dir", runtime_env["DEFAULT_SITE_PACKAGES_DIRNAME"]
    )
    base_python_root = env_root / runtime.get("base_python_dir", runtime_env["DEFAULT_BASE_PYTHON_DIRNAME"])
    host_venv = env_root / runtime.get("host_venv_dir", runtime_env["DEFAULT_HOST_VENV_DIRNAME"])
    source_root = env_root / runtime.get("source_dir", runtime_env["DEFAULT_SOURCE_DIRNAME"])
    shared_requirements = env_root / "shared-requirements.txt"
    host_requirements = env_root / "host-requirements.txt"

    _export_host_requirements(pyproject, host_requirements)
    base_python = _materialize_base_python(runtime["python_version"], base_python_root)
    host_python = _create_host_venv(base_python, host_venv, host_requirements)
    _export_shared_requirements(pyproject, shared_requirements)
    _install_target(host_python, shared_site_packages, shared_requirements)

    for source in runtime["sources"]:
        _clone_checkout(
            source_root / source["name"],
            repo=source["repo"],
            commit=source["commit"],
        )

    _write_host_source_version_files(pyproject, env_root)
    _write_host_pth(env_root, host_python)
    _write_host_source_dist_info(pyproject, host_python)
    _write_host_wrappers(env_root, host_python)
    _write_manifest(env_root, pyproject, host_python, shared_deps)
    _write_activation(env_root)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-root", help="Destination PFS runtime env root")
    p.add_argument(
        "--mint-env",
        choices=MINT_ENVS,
        help="Build or copy into <mint-root>/<env>/runtime-builds/<build-id>",
    )
    p.add_argument(
        "--mint-root",
        default=str(DEFAULT_MINT_ROOT),
        help=f"Mint runtime root that contains prod/ and dev/ (default: {DEFAULT_MINT_ROOT})",
    )
    p.add_argument(
        "--build-id",
        help="Build directory name under runtime-builds. Defaults to py31213-<UTC timestamp>-<git sha>.",
    )
    p.add_argument(
        "--copy-from",
        help="Copy an existing runtime root instead of building from pyproject.",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="Atomically update <mint-root>/<env>/runtime to the built or copied runtime root.",
    )
    p.add_argument("--inspect", action="store_true", help="Inspect an existing runtime env root")
    p.add_argument(
        "--probe-module",
        action="append",
        default=None,
        help="Module name to import with the runtime env host python during --inspect",
    )
    args = p.parse_args(argv)
    if args.inspect:
        if not args.env_root:
            p.error("--inspect requires --env-root")
        if args.mint_env or args.copy_from or args.promote:
            p.error("--inspect only supports --env-root and --probe-module")
        return args
    if args.mint_env:
        if args.env_root:
            p.error("--mint-env and --env-root are mutually exclusive for build/copy")
        return args
    if args.copy_from:
        p.error("--copy-from requires --mint-env")
    if args.promote:
        p.error("--promote requires --mint-env")
    if not args.env_root:
        p.error("one of --env-root or --mint-env is required")
    return args


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.inspect:
        env_root = Path(args.env_root).resolve()
        snapshot = inspect_runtime_env(env_root, probe_modules=args.probe_module)
        print(json.dumps(snapshot, indent=2))
        probe_failed = any(not result["ok"] for result in snapshot["probe_results"].values())
        return 0 if snapshot["valid_layout"] and not probe_failed else 1

    if args.mint_env:
        mint_root = Path(args.mint_root).resolve()
        build_id = args.build_id or _default_build_id()
        env_root = _mint_runtime_build_root(mint_root, args.mint_env, build_id).resolve()
        if args.copy_from:
            copy_runtime_env(Path(args.copy_from), env_root)
        else:
            build_runtime_env(env_root)
        if args.promote:
            _promote_runtime_symlink(env_root, _mint_runtime_link(mint_root, args.mint_env))
        print(json.dumps({"env_root": str(env_root), "promoted": bool(args.promote)}, indent=2))
        return 0

    env_root = Path(args.env_root).resolve()
    build_runtime_env(env_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
