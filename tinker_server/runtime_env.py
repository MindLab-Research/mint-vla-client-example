from __future__ import annotations

import json
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
DEFAULT_BASE_PYTHON_DIRNAME = "base-python"
DEFAULT_HOST_VENV_DIRNAME = "host-venv"


def canonical_mint_env_name(name: str) -> str:
    if name.startswith("TINKER_"):
        return f"MINT_{name[len('TINKER_'):]}"
    return name


def env_alias_names(name: str) -> tuple[str, ...]:
    if name.startswith("TINKER_"):
        return (canonical_mint_env_name(name), name)
    if name.startswith("MINT_"):
        return (name, f"TINKER_{name[len('MINT_'):]}")
    return (name,)


def env_get(environ: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    for alias in env_alias_names(name):
        value = environ.get(alias)
        if value is not None:
            return str(value)
    return default


def env_nonempty(environ: Mapping[str, str], name: str) -> str | None:
    value = env_get(environ, name)
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


def _norm_pythonpath_entry(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass(frozen=True)
class RuntimeEnvLayout:
    root: str
    site_packages: str
    source_root: str
    base_python_root: str
    host_venv_root: str
    pythonpath_entries: tuple[str, ...]
    host_pythonpath_entries: tuple[str, ...]
    host_python: str


@dataclass(frozen=True)
class RuntimeEnvSettings:
    site_packages_dir: str
    source_dir: str
    base_python_dir: str
    host_venv_dir: str
    sources: tuple[tuple[str, tuple[str, ...]], ...]
    host_sources: tuple[tuple[str, tuple[str, ...]], ...]


def _dedupe(entries: list[tuple[str, tuple[str, ...]]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    deduped: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return tuple(deduped)


def _settings_from_runtime_metadata(runtime: Mapping[str, object], sources: Sequence[Mapping[str, object]]) -> RuntimeEnvSettings:
    shared_entries: list[tuple[str, tuple[str, ...]]] = []
    host_entries: list[tuple[str, tuple[str, ...]]] = []
    for source in sources:
        if "name" not in source:
            raise RuntimeError(f"runtime source missing name: {source!r}")
        name = str(source["name"])
        bucket = host_entries if bool(source.get("host_only", False)) else shared_entries
        for rel in source.get("pythonpath", ["."]):
            rel_str = str(rel).strip()
            parts = () if rel_str in ("", ".") else tuple(part for part in rel_str.split("/") if part)
            bucket.append((name, parts))
    return RuntimeEnvSettings(
        site_packages_dir=str(runtime.get("site_packages_dir", DEFAULT_SITE_PACKAGES_DIRNAME)),
        source_dir=str(runtime.get("source_dir", DEFAULT_SOURCE_DIRNAME)),
        base_python_dir=str(runtime.get("base_python_dir", DEFAULT_BASE_PYTHON_DIRNAME)),
        host_venv_dir=str(runtime.get("host_venv_dir", DEFAULT_HOST_VENV_DIRNAME)),
        sources=_dedupe(shared_entries),
        host_sources=_dedupe(host_entries),
    )


@lru_cache(maxsize=1)
def _checkout_runtime_env_settings() -> RuntimeEnvSettings:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    runtime = data["tool"]["tinker"]["runtime_env"]
    return _settings_from_runtime_metadata(runtime, runtime["sources"])


def _runtime_env_settings_from_manifest(env_root: str) -> RuntimeEnvSettings:
    manifest_path = Path(os.path.abspath(env_root)) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"PFS runtime env root is missing manifest.json: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = data.get("runtime_env")
    sources = data.get("sources")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"runtime manifest missing runtime_env table: {manifest_path}")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"runtime manifest missing sources list: {manifest_path}")
    return _settings_from_runtime_metadata(runtime, sources)


def _layout_from_settings(env_root: str, settings: RuntimeEnvSettings) -> RuntimeEnvLayout:
    root = os.path.abspath(env_root)
    source_root = os.path.join(root, settings.source_dir)
    site_packages = os.path.join(root, settings.site_packages_dir)
    base_python_root = os.path.join(root, settings.base_python_dir)
    host_venv_root = os.path.join(root, settings.host_venv_dir)
    entries = [site_packages]
    for repo_name, rel_parts in settings.sources:
        entries.append(os.path.join(source_root, repo_name, *rel_parts))
    host_entries = [
        os.path.join(source_root, repo_name, *rel_parts)
        for repo_name, rel_parts in settings.host_sources
    ]
    host_python = os.path.join(host_venv_root, "bin", "python")
    return RuntimeEnvLayout(
        root=root,
        site_packages=site_packages,
        source_root=source_root,
        base_python_root=base_python_root,
        host_venv_root=host_venv_root,
        pythonpath_entries=tuple(entries),
        host_pythonpath_entries=tuple(host_entries),
        host_python=host_python,
    )


def runtime_env_layout(env_root: str) -> RuntimeEnvLayout:
    return _layout_from_settings(env_root, _runtime_env_settings_from_manifest(env_root))


def checkout_runtime_env_layout(env_root: str) -> RuntimeEnvLayout:
    return _layout_from_settings(env_root, _checkout_runtime_env_settings())


def validate_runtime_env_layout(env_root: str, *, require_host_python: bool = True) -> RuntimeEnvLayout:
    layout = runtime_env_layout(env_root)
    required = [layout.site_packages, *layout.pythonpath_entries[1:]]
    if require_host_python:
        required.extend(
            [
                layout.base_python_root,
                layout.host_venv_root,
                layout.host_python,
                *layout.host_pythonpath_entries,
            ]
        )
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(
            "PFS runtime env root is incomplete. "
            f"root={layout.root!r} missing={missing!r}"
        )
    return layout


def host_only_pythonpath_entries(env_root: str) -> tuple[str, ...]:
    layout = runtime_env_layout(env_root)
    return tuple(layout.host_pythonpath_entries)


def sanitize_worker_pythonpath(raw: str | None, *, env_root: str | None) -> str:
    entries = split_pythonpath(raw)
    excluded: set[str] = set()
    if env_root:
        try:
            excluded.update(_norm_pythonpath_entry(path) for path in host_only_pythonpath_entries(env_root))
        except Exception:
            pass

    sanitized: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        norm = _norm_pythonpath_entry(entry)
        if norm in excluded or norm in seen:
            continue
        seen.add(norm)
        sanitized.append(entry)
    return join_pythonpath(sanitized)


def build_runtime_pythonpath(
    *,
    env_root: str,
    mint_code_root: str,
    pfs_hf_modules_path: str,
) -> str:
    layout = validate_runtime_env_layout(env_root, require_host_python=False)
    return join_pythonpath(
        layout.pythonpath_entries,
        mint_code_root,
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
    mint_code_root = env_nonempty(environ, "MINT_CODE_ROOT") or env_nonempty(environ, "PFS_TINKER_PATH")
    if not mint_code_root:
        raise RuntimeError("MINT_CODE_ROOT is required")
    pfs_hf_modules_path = env_nonempty(environ, "PFS_HF_MODULES_PATH")
    if not pfs_hf_modules_path:
        raise RuntimeError("PFS_HF_MODULES_PATH is required")
    layout = runtime_env_layout(env_root)
    return join_pythonpath(
        layout.pythonpath_entries,
        layout.host_pythonpath_entries,
        mint_code_root,
        pfs_hf_modules_path,
    )
