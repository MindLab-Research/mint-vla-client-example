"""Resolve the canonical MANO dataset release from one machine-readable manifest.

This module deliberately uses only the Python standard library so shell launchers
can resolve canonical paths before the deployed Lance/OpenPI runtime is loaded.
The manifest owns path/version/hash authority; documentation is explanatory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_MANIFEST = REPO_ROOT / "config/datasets/mano_dataset_release.json"
SUPPORTED_SCHEMA_VERSION = 1


class ReleaseContractError(ValueError):
    """The release manifest is absent, malformed, or internally inconsistent."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseContractError(f"{name} must be a JSON object")
    return value


def load_release(path: str | Path = DEFAULT_RELEASE_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseContractError(f"MANO release manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseContractError(f"invalid MANO release JSON at {manifest_path}: {exc}") from exc
    release = dict(_require_mapping(payload, "release manifest"))
    if release.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ReleaseContractError(
            f"unsupported MANO release schema {release.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    release_id = release.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ReleaseContractError("release_id must be a non-empty string")
    artifacts = _require_mapping(release.get("artifacts"), "artifacts")
    roles = _require_mapping(release.get("roles"), "roles")
    for role, artifact_id in roles.items():
        if not isinstance(role, str) or not role:
            raise ReleaseContractError(f"invalid release role {role!r}")
        if not isinstance(artifact_id, str) or artifact_id not in artifacts:
            raise ReleaseContractError(
                f"release role {role!r} references unknown artifact {artifact_id!r}"
            )
    release["_manifest_path"] = str(manifest_path)
    return release


def client_root(manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST) -> Path:
    """Return the checkout containing config/datasets/<manifest>."""
    resolved = Path(manifest_path).expanduser().resolve()
    if resolved.parent.name != "datasets" or resolved.parent.parent.name != "config":
        raise ReleaseContractError(
            f"client:// resolution requires a config/datasets manifest, got {resolved}"
        )
    return resolved.parents[2]


def resolve_path(value: str, *, manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST) -> Path:
    if not isinstance(value, str) or not value:
        raise ReleaseContractError(f"artifact path must be a non-empty string, got {value!r}")
    if value.startswith("client://"):
        relative = value.removeprefix("client://")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ReleaseContractError(f"invalid client:// path {value!r}")
        return Path(os.path.abspath(client_root(manifest_path) / relative))
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReleaseContractError(
            f"external artifact paths must be absolute or client:// URIs, got {value!r}"
        )
    return Path(os.path.abspath(path))


def artifact_for(
    artifact_id: str,
    *,
    release: Mapping[str, Any] | None = None,
    manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> Mapping[str, Any]:
    loaded = load_release(manifest_path) if release is None else release
    artifacts = _require_mapping(loaded.get("artifacts"), "artifacts")
    try:
        artifact = artifacts[artifact_id]
    except KeyError as exc:
        raise ReleaseContractError(f"unknown MANO release artifact {artifact_id!r}") from exc
    return _require_mapping(artifact, f"artifact {artifact_id!r}")


def resolve_artifact(
    artifact_id: str,
    *,
    release: Mapping[str, Any] | None = None,
    manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> Path:
    artifact = artifact_for(artifact_id, release=release, manifest_path=manifest_path)
    if "path" not in artifact:
        raise ReleaseContractError(f"artifact {artifact_id!r} has no resolvable path")
    return resolve_path(str(artifact["path"]), manifest_path=manifest_path)


def resolve_role(
    role: str,
    *,
    release: Mapping[str, Any] | None = None,
    manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> Path:
    loaded = load_release(manifest_path) if release is None else release
    roles = _require_mapping(loaded.get("roles"), "roles")
    try:
        artifact_id = roles[role]
    except KeyError as exc:
        raise ReleaseContractError(f"unknown MANO release role {role!r}") from exc
    return resolve_artifact(
        str(artifact_id), release=loaded, manifest_path=manifest_path
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve", help="print the path for one canonical role")
    resolve.add_argument("role")
    artifact = subparsers.add_parser("artifact", help="print the path for one artifact ID")
    artifact.add_argument("artifact_id")
    subparsers.add_parser("release-id", help="print the canonical release ID")
    args = parser.parse_args()
    release = load_release(args.manifest)
    if args.command == "resolve":
        print(resolve_role(args.role, release=release, manifest_path=args.manifest))
    elif args.command == "artifact":
        print(resolve_artifact(args.artifact_id, release=release, manifest_path=args.manifest))
    else:
        print(release["release_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
