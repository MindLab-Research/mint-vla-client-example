from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import contact_windows
from scripts import mano_dataset_release as release


def write_manifest(root: Path, payload: dict) -> Path:
    path = root / "config" / "datasets" / "mano_dataset_release.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "release_id": "test-release",
        "roles": {"training_dataset": "canonical"},
        "artifacts": {
            "canonical": {"kind": "file", "path": "client://data/canonical.lance"},
            "external": {"kind": "file", "path": "/tmp/external.json"},
        },
    }


def test_release_resolves_client_and_external_paths(tmp_path: Path):
    manifest = write_manifest(tmp_path, valid_payload())
    loaded = release.load_release(manifest)
    assert release.resolve_role(
        "training_dataset", release=loaded, manifest_path=manifest
    ) == tmp_path / "data" / "canonical.lance"
    assert release.resolve_artifact(
        "external", release=loaded, manifest_path=manifest
    ) == Path("/tmp/external.json")


def test_release_rejects_unknown_role_artifact(tmp_path: Path):
    payload = valid_payload()
    payload["roles"]["training_dataset"] = "missing"
    manifest = write_manifest(tmp_path, payload)
    with pytest.raises(release.ReleaseContractError, match="unknown artifact"):
        release.load_release(manifest)


def test_release_rejects_escape_from_client_root(tmp_path: Path):
    payload = valid_payload()
    payload["artifacts"]["canonical"]["path"] = "client://../secret"
    manifest = write_manifest(tmp_path, payload)
    loaded = release.load_release(manifest)
    with pytest.raises(release.ReleaseContractError, match="invalid client"):
        release.resolve_role(
            "training_dataset", release=loaded, manifest_path=manifest
        )


def test_release_requires_absolute_external_paths(tmp_path: Path):
    payload = valid_payload()
    payload["artifacts"]["external"]["path"] = "relative/path"
    manifest = write_manifest(tmp_path, payload)
    loaded = release.load_release(manifest)
    with pytest.raises(release.ReleaseContractError, match="must be absolute"):
        release.resolve_artifact("external", release=loaded, manifest_path=manifest)


def test_release_rejects_schema_drift(tmp_path: Path):
    payload = valid_payload()
    payload["schema_version"] = 2
    manifest = write_manifest(tmp_path, payload)
    with pytest.raises(release.ReleaseContractError, match="unsupported"):
        release.load_release(manifest)


def test_contact_default_uses_release_role_for_canonical_dataset():
    canonical = release.resolve_role("training_dataset")
    expected = release.resolve_role("contact_windows")
    assert contact_windows.default_manifest_path(canonical) == expected


def test_contact_default_derives_path_only_for_nonrelease_dataset(tmp_path: Path):
    dataset = tmp_path / "experiment.lance"
    assert contact_windows.default_manifest_path(dataset) == Path(
        f"{dataset.resolve()}.contact_windows.json"
    )
