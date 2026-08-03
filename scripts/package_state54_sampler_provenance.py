#!/usr/bin/env python3
"""Create a fail-closed provenance companion for a replay-State54 sampler."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def authenticated_json(path: Path, expected_sha: str | None, label: str) -> tuple[dict[str, Any], str]:
    actual = sha256(path)
    if expected_sha is not None and actual != expected_sha.lower():
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected_sha}")
    return load_json(path), actual


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", type=Path, required=True)
    p.add_argument("--data-contract", type=Path, required=True)
    p.add_argument("--data-contract-sha256", required=True)
    p.add_argument("--data-release", type=Path, required=True)
    p.add_argument("--data-release-sha256", required=True)
    p.add_argument("--feature-release", type=Path, required=True)
    p.add_argument("--feature-release-sha256", required=True)
    p.add_argument("--norm-dir", type=Path, required=True)
    p.add_argument("--mode4-validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sampler = args.sampler.resolve()
    metadata_path = sampler / "metadata.json"
    profile_path = sampler / "mint_pi05_profile.json"
    params_metadata_path = sampler / "params" / "_METADATA"
    for path in (metadata_path, profile_path, params_metadata_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"sampler artifact missing or empty: {path}")
    metadata, metadata_sha = authenticated_json(metadata_path, None, "sampler metadata")
    profile, profile_sha = authenticated_json(profile_path, None, "sampler profile")
    params_metadata_sha = sha256(params_metadata_path)
    if metadata.get("checkpoint_type") != "sampler" or metadata.get("type") != "sampler":
        raise ValueError("sampler metadata does not identify a sampler")
    if metadata.get("optimizer_present") is not False:
        raise ValueError("provenance packager requires inference-only sampler optimizer_present=false")
    if profile.get("state_dim") != 54 or profile.get("action_dim") != 32:
        raise ValueError("sampler profile is not State54/Action32")
    if profile.get("profile_id") != "pi05_action_lora_r16_state54_v1":
        raise ValueError("unexpected sampler State54 profile")
    if profile.get("max_token_len") != 256 or profile.get("fail_on_token_truncation") is not True:
        raise ValueError("sampler token contract is not fail-closed max256")
    if profile.get("expected_trainable_count") != 13224992:
        raise ValueError("sampler trainable count mismatch")

    contract, contract_sha = authenticated_json(
        args.data_contract.resolve(), args.data_contract_sha256, "data contract"
    )
    if contract.get("status") != "accepted":
        raise ValueError("data contract is not accepted")
    if contract.get("state_contract") != "mano_object_dynamics_state54_v1":
        raise ValueError("data contract is not State54")
    if contract.get("state_dim") != 54 or contract.get("action_dim") != 32:
        raise ValueError("data contract dimensions mismatch")
    if contract.get("norm_stats_sha256") is None or contract.get("token_audit_sha256") is None:
        raise ValueError("data contract lacks norm/token authentication")

    data_release, data_release_sha = authenticated_json(
        args.data_release.resolve(), args.data_release_sha256, "data release"
    )
    if data_release.get("status") != "accepted":
        raise ValueError("data release is not accepted")
    feature_release_path = args.feature_release.resolve() / "release.json"
    feature_release, feature_release_sha = authenticated_json(
        feature_release_path, args.feature_release_sha256, "feature release"
    )
    if feature_release.get("status") != "accepted":
        raise ValueError("feature release is not accepted")
    if contract.get("data_release_sha256") != data_release_sha:
        raise ValueError("contract/data release SHA mismatch")
    if contract.get("feature_release_sha256") != feature_release_sha:
        raise ValueError("contract/feature release SHA mismatch")
    if data_release.get("feature_release_sha256") != feature_release_sha:
        raise ValueError("data release/feature release SHA mismatch")

    norm_dir = args.norm_dir.resolve()
    norm_path = norm_dir / "norm_stats.json"
    token_path = norm_dir / "token_audit.json"
    norm_sha = sha256(norm_path)
    token_sha = sha256(token_path)
    if contract.get("norm_stats_sha256") != norm_sha:
        raise ValueError("contract/norm SHA mismatch")
    if contract.get("token_audit_sha256") != token_sha:
        raise ValueError("contract/token audit SHA mismatch")

    mode4, mode4_sha = authenticated_json(args.mode4_validation.resolve(), None, "Mode4 validation")
    if mode4.get("status") != "accepted":
        raise ValueError("Mode4 validation is not accepted")
    if mode4.get("first_observation_matches_training_state54") is not True:
        raise ValueError("Mode4 validation lacks training-state parity")
    if mode4.get("all_arrays_finite") is not True:
        raise ValueError("Mode4 validation lacks finite-array gate")
    if mode4.get("all_queries_fixed_batch4_sharded") is not True:
        raise ValueError("Mode4 validation lacks batch4/data-sharding gate")
    if mode4.get("state54_data_contract_sha256") != contract_sha:
        raise ValueError("Mode4 validation/data contract SHA mismatch")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance_id": "replay_state54_sampler_companion_v1",
        "sampler": {
            "path": str(sampler),
            "metadata_sha256": metadata_sha,
            "profile_sha256": profile_sha,
            "params_metadata_sha256": params_metadata_sha,
            "checkpoint_id": metadata.get("checkpoint_id"),
            "model_id": metadata.get("model_id"),
            "step": metadata.get("step"),
            "optimizer_present": metadata.get("optimizer_present"),
            "profile_id": profile.get("profile_id"),
            "state_dim": profile.get("state_dim"),
            "action_dim": profile.get("action_dim"),
            "action_horizon": profile.get("action_horizon"),
            "max_token_len": profile.get("max_token_len"),
            "expected_trainable_count": profile.get("expected_trainable_count"),
        },
        "data_contract": {
            "path": str(args.data_contract.resolve()),
            "sha256": contract_sha,
            "contract_id": contract.get("contract_id"),
            "state_contract": contract.get("state_contract"),
            "row_indices_sha256": contract.get("row_indices_sha256"),
            "trajectory_count": contract.get("trajectory_count"),
            "active_frame_count": contract.get("active_frame_count"),
            "norm_stats_sha256": norm_sha,
            "token_audit_sha256": token_sha,
        },
        "data_release": {
            "path": str(args.data_release.resolve()),
            "sha256": data_release_sha,
            "source_release_sha256": data_release.get("source_release_sha256"),
            "feature_release_sha256": feature_release_sha,
        },
        "feature_release": {
            "path": str(args.feature_release.resolve()),
            "sha256": feature_release_sha,
            "feature_schema_id": feature_release.get("feature_schema_id"),
            "feature_manifest_sha256": feature_release.get("feature_manifest_sha256"),
        },
        "norm": {"path": str(norm_path), "sha256": norm_sha},
        "token_audit": {"path": str(token_path), "sha256": token_sha},
        "mode4_validation": {
            "path": str(args.mode4_validation.resolve()),
            "sha256": mode4_sha,
            "row_count": mode4.get("row_count"),
            "first_observation_tolerance": mode4.get("first_observation_tolerance"),
        },
        "mode4_initialization": contract.get("mode4_initialization"),
        "force_derivation": contract.get("force_derivation"),
        "client_commit": contract.get("client_commit"),
        "mint_commit": contract.get("mint_commit"),
        "openpi_commit": contract.get("openpi_commit"),
    }
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "accepted", "output": str(output), "sha256": sha256(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
