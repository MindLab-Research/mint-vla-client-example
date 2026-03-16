from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_HF_MODULES_PATH = "/vePFS-Mindverse/share/huggingface/modules"
DEFAULT_SITE_PACKAGES_DIRNAME = "site-packages"
DEFAULT_SOURCE_DIRNAME = "src"
DEFAULT_HOST_VENV_DIRNAME = "host-venv"


def env_nonempty(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def split_pythonpath(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part for part in str(raw).split(":") if part]


def join_pythonpath(*parts: str | Sequence[str]) -> str:
    flat: list[str] = []
    for part in parts:
        if isinstance(part, str):
            flat.extend(split_pythonpath(part))
        else:
            flat.extend([item for item in part if item])
    return ":".join(flat)


@dataclass(frozen=True)
class RuntimeEnvLayout:
    root: str
    site_packages: str
    source_root: str
    pythonpath_entries: tuple[str, ...]
    host_pythonpath_entries: tuple[str, ...]
    host_python: str


@dataclass(frozen=True)
class RuntimeEnvSettings:
    site_packages_dir: str
    source_dir: str
    host_venv_dir: str
    sources: tuple[tuple[str, tuple[str, ...]], ...]
    host_sources: tuple[tuple[str, tuple[str, ...]], ...]


@lru_cache(maxsize=1)
def _runtime_env_settings() -> RuntimeEnvSettings:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    runtime = data["tool"]["tinker"]["runtime_env"]
    shared_entries: list[tuple[str, tuple[str, ...]]] = []
    host_entries: list[tuple[str, tuple[str, ...]]] = []
    for source in runtime["sources"]:
        name = str(source["name"])
        bucket = host_entries if bool(source.get("host_only", False)) else shared_entries
        for rel in source.get("pythonpath", ["."]):
            rel_str = str(rel).strip()
            parts = () if rel_str in ("", ".") else tuple(part for part in rel_str.split("/") if part)
            bucket.append((name, parts))

    def _dedupe(entries: list[tuple[str, tuple[str, ...]]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        deduped: list[tuple[str, tuple[str, ...]]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for entry in entries:
            if entry in seen:
                continue
            seen.add(entry)
            deduped.append(entry)
        return tuple(deduped)

    return RuntimeEnvSettings(
        site_packages_dir=str(runtime.get("site_packages_dir", DEFAULT_SITE_PACKAGES_DIRNAME)),
        source_dir=str(runtime.get("source_dir", DEFAULT_SOURCE_DIRNAME)),
        host_venv_dir=str(runtime.get("host_venv_dir", DEFAULT_HOST_VENV_DIRNAME)),
        sources=_dedupe(shared_entries),
        host_sources=_dedupe(host_entries),
    )


def runtime_env_layout(env_root: str) -> RuntimeEnvLayout:
    settings = _runtime_env_settings()
    root = os.path.abspath(env_root)
    source_root = os.path.join(root, settings.source_dir)
    site_packages = os.path.join(root, settings.site_packages_dir)
    entries = [site_packages]
    for repo_name, rel_parts in settings.sources:
        entries.append(os.path.join(source_root, repo_name, *rel_parts))
    host_entries = [
        os.path.join(source_root, repo_name, *rel_parts)
        for repo_name, rel_parts in settings.host_sources
    ]
    host_python = os.path.join(root, settings.host_venv_dir, "bin", "python")
    return RuntimeEnvLayout(
        root=root,
        site_packages=site_packages,
        source_root=source_root,
        pythonpath_entries=tuple(entries),
        host_pythonpath_entries=tuple(host_entries),
        host_python=host_python,
    )


def validate_runtime_env_layout(env_root: str, *, require_host_python: bool = True) -> RuntimeEnvLayout:
    layout = runtime_env_layout(env_root)
    required = [layout.site_packages, *layout.pythonpath_entries[1:]]
    if require_host_python:
        required.extend([layout.host_python, *layout.host_pythonpath_entries])
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(
            "PFS runtime env root is incomplete. "
            f"root={layout.root!r} missing={missing!r}"
        )
    return layout


def build_runtime_pythonpath(
    *,
    env_root: str,
    pfs_tinker_path: str,
    pfs_hf_modules_path: str,
) -> str:
    layout = validate_runtime_env_layout(env_root, require_host_python=False)
    return join_pythonpath(
        layout.pythonpath_entries,
        pfs_tinker_path,
        pfs_hf_modules_path,
    )


def bootstrap_runtime_pythonpath(
    environ: Mapping[str, str],
    *,
    repo_root: str,
    default_hf_modules_path: str = DEFAULT_HF_MODULES_PATH,
) -> str:
    env_root = env_nonempty(environ, "PFS_RUNTIME_ENV_ROOT")
    if not env_root:
        raise RuntimeError("PFS_RUNTIME_ENV_ROOT is required")
    pfs_tinker_path = env_nonempty(environ, "PFS_TINKER_PATH")
    if not pfs_tinker_path:
        raise RuntimeError("PFS_TINKER_PATH is required")
    pfs_hf_modules_path = env_nonempty(environ, "PFS_HF_MODULES_PATH")
    if not pfs_hf_modules_path:
        raise RuntimeError("PFS_HF_MODULES_PATH is required")
    layout = runtime_env_layout(env_root)
    return join_pythonpath(
        layout.pythonpath_entries,
        pfs_tinker_path,
        pfs_hf_modules_path,
    )
