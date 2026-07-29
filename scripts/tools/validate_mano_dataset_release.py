#!/usr/bin/env python3
"""Fail-closed validator for config/datasets/mano_dataset_release.json.

Fast mode verifies every pinned file, producer commit, asset closure, JSON
contract, and Lance metadata without loading image payloads. Deep mode additionally
checks full row/frame alignment, contact-window population, and all physics-replay
records. There is no degraded success: any requested check that cannot run fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

from scripts import mano_dataset_release as release_lib


@dataclass
class Validation:
    release_id: str
    mode: str
    manifest_path: str
    manifest_sha256: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def pass_check(self, artifact_id: str, **details: Any) -> None:
        self.checks.append({"artifact": artifact_id, "status": "ok", **details})

    def fail_check(self, artifact_id: str, error: Exception | str) -> None:
        message = f"{artifact_id}: {error}"
        self.errors.append(message)
        self.checks.append({"artifact": artifact_id, "status": "error", "error": str(error)})

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "release_id": self.release_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "mode": self.mode,
            "status": "ok" if not self.errors else "error",
            "check_count": len(self.checks),
            "checks": self.checks,
            "errors": self.errors,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def lance_metadata_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    for item in sorted(path.rglob("*")):
        if item.is_file() and ("manifest" in item.name or item.name.startswith("_versions")):
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
            count += 1
    if count == 0:
        raise ValueError(f"no Lance metadata files found under {path}")
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _mesh_references(urdf: Path) -> list[Path]:
    files = [urdf]
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            files.append((urdf.parent / filename).resolve())
    return files


def asset_bundle_fingerprint(spec: Mapping[str, Any], manifest_path: Path) -> tuple[str, int]:
    hand_urdf = release_lib.resolve_path(str(spec["hand_urdf"]), manifest_path=manifest_path)
    object_root = release_lib.resolve_path(str(spec["object_urdf_root"]), manifest_path=manifest_path)
    object_names = spec.get("object_names")
    if not isinstance(object_names, list) or not object_names or any(not isinstance(x, str) for x in object_names):
        raise ValueError("asset bundle requires non-empty string object_names")
    keyed: list[tuple[str, Path]] = []
    for item in _mesh_references(hand_urdf):
        if not item.is_file():
            raise FileNotFoundError(item)
        keyed.append(("hand/" + str(item.relative_to(hand_urdf.parent)), item))
    for object_name in sorted(object_names):
        urdf = object_root / f"{object_name}.urdf"
        for item in _mesh_references(urdf):
            if not item.is_file():
                raise FileNotFoundError(item)
            keyed.append(
                ("object/" + object_name + "/" + str(item.relative_to(object_root.parent)), item)
            )
    digest = hashlib.sha256()
    for key, item in sorted(keyed):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        digest.update(b"\n")
    return digest.hexdigest(), len(keyed)


def validate_file(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    details: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    expected = spec.get("sha256")
    if expected is not None:
        details["sha256"] = require_sha(path, str(expected))
    return details


def validate_directory(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {"path": str(path)}


def validate_git_repo(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    actual = git_head(path)
    expected = str(spec["commit"])
    if actual != expected:
        raise ValueError(f"git HEAD mismatch: expected {expected}, got {actual}")
    file_hashes = spec.get("file_sha256") or {}
    for relative, digest in file_hashes.items():
        require_sha(path / relative, str(digest))
    return {"path": str(path), "commit": actual, "file_count": len(file_hashes)}


def validate_gesture_index(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    details = validate_file(artifact_id, spec, path)
    payload = load_json(path)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("gesture index entries is not a list")
    expected = spec["contract"]
    stats = {
        "row_count": len(entries),
        "object_count": len({entry["object_type"] for entry in entries}),
        "gesture_count": len({entry["gesture"] for entry in entries}),
        "stratum_count": len({(entry["object_type"], entry["gesture"]) for entry in entries}),
    }
    for key, value in expected.items():
        if key == "version":
            actual = payload.get("version")
        elif key == "dataset_name":
            actual = payload.get("dataset_name")
        else:
            actual = stats.get(key)
        if actual != value:
            raise ValueError(f"gesture index {key} mismatch: expected {value!r}, got {actual!r}")
    if [entry.get("row_index") for entry in entries] != list(range(len(entries))):
        raise ValueError("gesture index row order is not contiguous")
    if len({entry.get("uuid") for entry in entries}) != len(entries):
        raise ValueError("gesture index UUIDs are not unique")
    details.update(stats)
    return details


def validate_contact_manifest(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    details = validate_file(artifact_id, spec, path)
    payload = load_json(path)
    contract = spec["contract"]
    for key, expected in contract.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(f"contact manifest {key} mismatch: expected {expected!r}, got {actual!r}")
    windows = payload.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("contact manifest windows is not an object")
    if len(windows) != int(contract["row_count"]):
        raise ValueError(f"contact manifest window count {len(windows)} != {contract['row_count']}")
    details["window_count"] = len(windows)
    return details


def validate_json_contract(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    details = validate_file(artifact_id, spec, path)
    payload = load_json(path)
    expected = spec.get("top_level_equals") or {}
    for key, value in expected.items():
        actual = payload.get(key) if isinstance(payload, dict) else None
        if actual != value:
            raise ValueError(f"JSON field {key} mismatch: expected {value!r}, got {actual!r}")
    return details


def _schema_names(data_type: Any) -> list[str]:
    return [str(field.name) for field in data_type]


def validate_lance(
    artifact_id: str, spec: Mapping[str, Any], path: Path, *, deep: bool
) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError(
            "pylance is unavailable; run through scripts/remote/run_client.sh"
        ) from exc
    dataset = lance.dataset(str(path))
    expected = spec["contract"]
    version = int(dataset.version)
    row_count = int(dataset.count_rows())
    if version != int(expected["version"]):
        raise ValueError(f"Lance version mismatch: expected {expected['version']}, got {version}")
    if row_count != int(expected["row_count"]):
        raise ValueError(f"Lance row count mismatch: expected {expected['row_count']}, got {row_count}")
    fields = list(dataset.schema.names)
    if fields != list(expected["top_level_fields"]):
        raise ValueError(f"Lance top-level schema mismatch: expected {expected['top_level_fields']}, got {fields}")
    hands_fields = _schema_names(dataset.schema.field("hands").type.value_type)
    if hands_fields != list(expected["hands_fields"]):
        raise ValueError(f"Lance hands schema mismatch: expected {expected['hands_fields']}, got {hands_fields}")
    fingerprint = lance_metadata_fingerprint(path)
    if fingerprint != spec["metadata_fingerprint_sha256"]:
        raise ValueError(
            f"Lance metadata fingerprint mismatch: expected {spec['metadata_fingerprint_sha256']}, got {fingerprint}"
        )
    tags = dataset.tags.list()
    for tag, tag_version in (expected.get("tags") or {}).items():
        actual = int(tags.get(tag, {}).get("version", -1))
        if actual != int(tag_version):
            raise ValueError(f"Lance tag {tag!r} mismatch: expected {tag_version}, got {actual}")
    details: dict[str, Any] = {
        "path": str(path),
        "version": version,
        "row_count": row_count,
        "metadata_fingerprint_sha256": fingerprint,
    }
    if deep:
        frame_field = expected.get("frame_count_field")
        expected_frames = expected.get("frame_count")
        if frame_field and expected_frames is not None:
            column, child = frame_field.split(".", 1)
            frame_count = 0
            for batch in dataset.to_batches(columns=[column], batch_size=512):
                for value in batch.column(0).to_pylist():
                    frame_count += int((value or {}).get(child) or 0)
            if frame_count != int(expected_frames):
                raise ValueError(f"Lance frame count mismatch: expected {expected_frames}, got {frame_count}")
            details["frame_count"] = frame_count
    return details


def validate_asset_bundle(
    artifact_id: str, spec: Mapping[str, Any], path: Path, manifest_path: Path
) -> dict[str, Any]:
    # The artifact path is the hand URDF for simple resolution; the contract
    # closes over all referenced hand/object URDF+mesh files.
    if not path.is_file():
        raise FileNotFoundError(path)
    actual, file_count = asset_bundle_fingerprint(spec["contract"], manifest_path)
    expected = spec["sha256"]
    if actual != expected:
        raise ValueError(f"asset bundle mismatch: expected {expected}, got {actual}")
    expected_count = int(spec["contract"].get("file_count", file_count))
    if file_count != expected_count:
        raise ValueError(f"asset bundle file count mismatch: expected {expected_count}, got {file_count}")
    return {"path": str(path), "sha256": actual, "file_count": file_count}


def validate_physics_evidence(
    artifact_id: str, spec: Mapping[str, Any], path: Path, *, deep: bool
) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    marker = path / str(spec["completion_marker"])
    require_sha(marker, str(spec["completion_marker_sha256"]))
    objects_root = path / "objects"
    object_dirs = sorted(item for item in objects_root.iterdir() if item.is_dir())
    expected = spec["contract"]
    if len(object_dirs) != int(expected["object_count"]):
        raise ValueError(f"physics object count {len(object_dirs)} != {expected['object_count']}")
    details: dict[str, Any] = {
        "path": str(path),
        "object_count": len(object_dirs),
        "completion_marker_sha256": spec["completion_marker_sha256"],
    }
    if deep:
        records: list[dict[str, Any]] = []
        for object_dir in object_dirs:
            aggregate = load_json(object_dir / "aggregate_verification.json")
            current = [load_json(item) for item in sorted((object_dir / "records").glob("*.json"))]
            if len(current) != int(aggregate["row_count"]):
                raise ValueError(f"{object_dir.name}: record count disagrees with aggregate")
            counts = Counter(record["grade"] for record in current)
            normalized = {grade: int(counts.get(grade, 0)) for grade in ("A", "B", "C")}
            expected_counts = {grade: int(aggregate["grade_counts"].get(grade, 0)) for grade in ("A", "B", "C")}
            if normalized != expected_counts:
                raise ValueError(f"{object_dir.name}: grade counts disagree with aggregate")
            records.extend(current)
        if len(records) != int(expected["row_count"]):
            raise ValueError(f"physics row count {len(records)} != {expected['row_count']}")
        if any(record.get("status") != "ok" for record in records):
            raise ValueError("physics evidence contains non-ok records")
        grade_counts = Counter(record["grade"] for record in records)
        expected_grades = expected["grade_counts"]
        for grade in ("A", "B", "C"):
            if int(grade_counts.get(grade, 0)) != int(expected_grades[grade]):
                raise ValueError(f"physics grade {grade} mismatch")
        details.update(
            row_count=len(records),
            grade_counts={grade: int(grade_counts.get(grade, 0)) for grade in ("A", "B", "C")},
        )
    return details


def validate_symlink(artifact_id: str, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_symlink():
        raise ValueError(f"expected compatibility symlink at {path}")
    target = path.resolve()
    expected_target = release_lib.resolve_path(str(spec["target"]), manifest_path=Path(spec["_manifest_path"]))
    if target != expected_target:
        raise ValueError(f"symlink target mismatch: expected {expected_target}, got {target}")
    return {"path": str(path), "target": str(target)}


def deep_validate_index_alignment(
    release: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is unavailable for deep index alignment") from exc
    index_path = release_lib.resolve_role("language_index", release=release, manifest_path=manifest_path)
    dataset_path = release_lib.resolve_role("training_dataset", release=release, manifest_path=manifest_path)
    entries = load_json(index_path)["entries"]
    dataset = lance.dataset(str(dataset_path))
    matched = 0
    for offset in range(0, len(entries), 256):
        expected_batch = entries[offset : offset + 256]
        indices = list(range(offset, offset + len(expected_batch)))
        rows = dataset.take(indices, columns=["index", "trajectory_metadata", "episode_metadata"]).to_pylist()
        for row_index, expected, row in zip(indices, expected_batch, rows, strict=True):
            names = row["trajectory_metadata"].get("object_names") or []
            actual = {
                "row_index": row_index,
                "uuid": row["index"]["uuid"],
                "seed_uuid": row["index"]["seed_uuid"],
                "object_type": names[0] if len(names) == 1 else None,
                "total_frames": int(row["episode_metadata"]["total_frames"]),
            }
            wanted = {key: expected[key] for key in actual}
            if actual != wanted:
                raise ValueError(f"row {row_index} index/Lance mismatch: {actual} != {wanted}")
            matched += 1
    return {"matched_rows": matched}


def validate_release(manifest_path: Path, mode: str) -> Validation:
    release = release_lib.load_release(manifest_path)
    validation = Validation(
        str(release["release_id"]),
        mode,
        str(manifest_path),
        sha256_file(manifest_path),
    )
    artifacts = release["artifacts"]
    for artifact_id, raw_spec in artifacts.items():
        spec = dict(raw_spec)
        try:
            path = release_lib.resolve_artifact(
                artifact_id, release=release, manifest_path=manifest_path
            )
            kind = spec.get("kind")
            if kind in {"file", "code_file"}:
                details = validate_file(artifact_id, spec, path)
            elif kind == "directory":
                details = validate_directory(artifact_id, spec, path)
            elif kind == "git_repo":
                details = validate_git_repo(artifact_id, spec, path)
            elif kind == "gesture_index":
                details = validate_gesture_index(artifact_id, spec, path)
            elif kind == "contact_manifest":
                details = validate_contact_manifest(artifact_id, spec, path)
            elif kind == "json_contract":
                details = validate_json_contract(artifact_id, spec, path)
            elif kind == "lance_dataset":
                if mode == "paths":
                    details = validate_directory(artifact_id, spec, path)
                else:
                    details = validate_lance(artifact_id, spec, path, deep=mode == "deep")
            elif kind == "asset_bundle":
                details = validate_asset_bundle(artifact_id, spec, path, manifest_path)
            elif kind == "physics_evidence":
                details = validate_physics_evidence(
                    artifact_id, spec, path, deep=mode == "deep"
                )
            elif kind == "symlink":
                spec["_manifest_path"] = str(manifest_path)
                details = validate_symlink(artifact_id, spec, path)
            else:
                raise ValueError(f"unsupported artifact kind {kind!r}")
            validation.pass_check(artifact_id, **details)
        except Exception as exc:
            validation.fail_check(artifact_id, exc)
    if mode == "deep" and not validation.errors:
        try:
            details = deep_validate_index_alignment(release, manifest_path)
            validation.pass_check("relation:index_lance_alignment", **details)
        except Exception as exc:
            validation.fail_check("relation:index_lance_alignment", exc)
    return validation


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=release_lib.DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument("--mode", choices=("paths", "fast", "deep"), default="fast")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    validation = validate_release(args.manifest.expanduser().resolve(), args.mode)
    payload = validation.as_dict()
    if args.output is not None:
        atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
