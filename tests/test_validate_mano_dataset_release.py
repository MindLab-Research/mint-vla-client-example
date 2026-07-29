from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.tools import validate_mano_dataset_release as validator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gesture_index_contract_is_semantic_not_just_hash(tmp_path: Path):
    path = tmp_path / "index.json"
    payload = {
        "version": "1.0",
        "dataset_name": "new_all_generated_mano.lance",
        "entries": [
            {"row_index": 0, "uuid": "u0", "object_type": "cube", "gesture": "01"},
            {"row_index": 1, "uuid": "u1", "object_type": "cube", "gesture": "02"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = {
        "sha256": digest(path),
        "contract": {
            "version": "1.0",
            "dataset_name": "new_all_generated_mano.lance",
            "row_count": 2,
            "object_count": 1,
            "gesture_count": 2,
            "stratum_count": 2,
        },
    }
    details = validator.validate_gesture_index("gesture", spec, path)
    assert details["row_count"] == 2
    assert details["stratum_count"] == 2


def test_contact_manifest_requires_full_population(tmp_path: Path):
    path = tmp_path / "contact.json"
    payload = {
        "manifest_version": 1,
        "dataset": "/data/canonical.lance",
        "row_count": 2,
        "context_frames": 100,
        "missing_policy": "error",
        "windows": {"0": {}, "1": {}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = {
        "sha256": digest(path),
        "contract": {key: payload[key] for key in (
            "manifest_version", "dataset", "row_count", "context_frames", "missing_policy"
        )},
    }
    assert validator.validate_contact_manifest("contact", spec, path)["window_count"] == 2


def test_asset_bundle_hash_closes_over_referenced_meshes(tmp_path: Path):
    hand_dir = tmp_path / "hand"
    object_root = tmp_path / "assets" / "mano_objects_urdf"
    mesh_root = tmp_path / "assets" / "mano_assets" / "objects" / "cube"
    hand_dir.mkdir(parents=True)
    object_root.mkdir(parents=True)
    mesh_root.mkdir(parents=True)
    (hand_dir / "hand.stl").write_bytes(b"hand-mesh")
    hand = hand_dir / "mano_hand.urdf"
    hand.write_text('<robot><link><visual><geometry><mesh filename="hand.stl"/></geometry></visual></link></robot>')
    (mesh_root / "cube.obj").write_bytes(b"cube-mesh")
    (object_root / "cube.urdf").write_text(
        '<robot><link><visual><geometry><mesh filename="../mano_assets/objects/cube/cube.obj"/></geometry></visual></link></robot>'
    )
    manifest = tmp_path / "config" / "datasets" / "release.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    spec = {
        "hand_urdf": str(hand),
        "object_urdf_root": str(object_root),
        "object_names": ["cube"],
    }
    first, count = validator.asset_bundle_fingerprint(spec, manifest)
    assert count == 4
    (mesh_root / "cube.obj").write_bytes(b"changed")
    second, _ = validator.asset_bundle_fingerprint(spec, manifest)
    assert first != second


def test_lance_metadata_fingerprint_ignores_payload_files(tmp_path: Path):
    root = tmp_path / "dataset.lance"
    (root / "_versions").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "_versions" / "1.manifest").write_bytes(b"metadata")
    (root / "data" / "payload.lance").write_bytes(b"payload-one")
    first = validator.lance_metadata_fingerprint(root)
    (root / "data" / "payload.lance").write_bytes(b"payload-two")
    assert validator.lance_metadata_fingerprint(root) == first
    (root / "_versions" / "1.manifest").write_bytes(b"changed")
    assert validator.lance_metadata_fingerprint(root) != first
