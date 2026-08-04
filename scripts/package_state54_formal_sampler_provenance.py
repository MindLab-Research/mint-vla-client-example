#!/usr/bin/env python3
"""Package a formal train-only replay-State54 sampler with fail-closed provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_json(path: Path, expected: str | None, label: str) -> tuple[dict[str, Any], str]:
    actual = sha256(path)
    if expected is not None and actual != expected.lower():
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--data-contract-sha256", required=True)
    parser.add_argument("--data-release", type=Path, required=True)
    parser.add_argument("--data-release-sha256", required=True)
    parser.add_argument("--feature-release", type=Path, required=True)
    parser.add_argument("--feature-release-sha256", required=True)
    parser.add_argument("--formal-protocol", type=Path, required=True)
    parser.add_argument("--formal-protocol-sha256", required=True)
    parser.add_argument("--coverage-schedule", type=Path, required=True)
    parser.add_argument("--coverage-schedule-sha256", required=True)
    parser.add_argument("--mode4-validation", type=Path, required=True)
    parser.add_argument("--mode4-validation-sha256", required=True)
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sampler = args.sampler.resolve()
    files = {
        "metadata": sampler / "metadata.json",
        "profile": sampler / "mint_pi05_profile.json",
        "params_metadata": sampler / "params" / "_METADATA",
        "norm_provenance": sampler / "mint_pi05_norm_provenance.json",
        "embedded_norm": sampler / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
    }
    for label, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"sampler {label} missing or empty: {path}")
    metadata, metadata_sha = authenticated_json(files["metadata"], None, "sampler metadata")
    profile, profile_sha = authenticated_json(files["profile"], None, "sampler profile")
    norm_provenance, norm_provenance_sha = authenticated_json(files["norm_provenance"], None, "sampler norm provenance")
    if metadata.get("checkpoint_type") != "sampler" or metadata.get("type") != "sampler" or metadata.get("optimizer_present") is not False:
        raise ValueError("sampler metadata must identify an inference-only sampler")
    if int(metadata.get("step", -1)) != args.expected_step:
        raise ValueError("sampler step mismatch")
    expected_profile = {
        "profile_id": "pi05_action_lora_r16_state54_v1",
        "state_dim": 54,
        "action_dim": 32,
        "action_horizon": 10,
        "max_token_len": 256,
        "fail_on_token_truncation": True,
        "expected_trainable_count": 13224992,
    }
    for key, value in expected_profile.items():
        if profile.get(key) != value:
            raise ValueError(f"sampler profile {key} mismatch: {profile.get(key)!r} != {value!r}")
    contract, contract_sha = authenticated_json(args.data_contract.resolve(), args.data_contract_sha256, "data contract")
    if contract.get("status") != "accepted" or contract.get("contract_id") != "state54_replay_train_only_noaug_v1" or contract.get("state_contract") != "mano_object_dynamics_state54_v1":
        raise ValueError("data contract is not formal train-only State54")
    if contract.get("augmentation") != {"state_noise_std": 0.0, "target_noise_std": 0.0}:
        raise ValueError("formal State54 contract is not no-augmentation")
    data_release, data_release_sha = authenticated_json(args.data_release.resolve(), args.data_release_sha256, "data release")
    feature_release, feature_release_sha = authenticated_json(args.feature_release.resolve(), args.feature_release_sha256, "feature release")
    if data_release.get("status") != "accepted" or feature_release.get("status") != "accepted":
        raise ValueError("data/feature release is not accepted")
    if contract.get("data_release_sha256") != data_release_sha or contract.get("feature_release_sha256") != feature_release_sha or data_release.get("feature_release_sha256") != feature_release_sha:
        raise ValueError("contract/data/feature release binding mismatch")
    protocol, protocol_sha = authenticated_json(args.formal_protocol.resolve(), args.formal_protocol_sha256, "formal protocol")
    if protocol.get("protocol_id") != "state54_replay_train_only_v1" or protocol.get("status") != "frozen_not_launched":
        raise ValueError("wrong State54-only formal protocol")
    schedule, schedule_sha = authenticated_json(args.coverage_schedule.resolve(), args.coverage_schedule_sha256, "coverage schedule")
    schedules = {int(item["seed"]): item for item in schedule.get("schedules", [])}
    seed_schedule = schedules.get(args.training_seed)
    if seed_schedule is None or seed_schedule.get("samples") != 1_200_000:
        raise ValueError("coverage schedule lacks selected 1.2M-sample seed")
    if contract.get("formal_protocol_sha256") != protocol_sha or contract.get("coverage_schedule_sha256") != schedule_sha:
        raise ValueError("contract protocol/schedule binding mismatch")
    norm_path = Path(contract["norm_stats"])
    token_path = Path(contract["token_audit"])
    norm_sha, token_sha = sha256(norm_path), sha256(token_path)
    if norm_sha != contract.get("norm_stats_sha256") or token_sha != contract.get("token_audit_sha256"):
        raise ValueError("formal norm/token artifacts changed")
    if norm_provenance.get("sha256") != norm_sha or norm_provenance.get("state_dim") != 54 or norm_provenance.get("action_dim") != 32:
        raise ValueError("sampler norm provenance mismatch")
    if sha256(files["embedded_norm"]) != norm_sha:
        raise ValueError("sampler embedded norm bytes mismatch")
    validation, validation_sha = authenticated_json(args.mode4_validation.resolve(), args.mode4_validation_sha256, "Mode4 validation")
    if validation.get("status") != "accepted" or validation.get("protocol") != "state54_replay_train_only_mode4_gate_v1":
        raise ValueError("Mode4 validation is not the formal State54 gate")
    if validation.get("state54_data_contract_sha256") != contract_sha or validation.get("row_count") != 4 or validation.get("first_observation_matches_training_state54") is not True or validation.get("all_arrays_finite") is not True or validation.get("all_queries_fixed_batch4_sharded") is not True or validation.get("action_session_released") is not True:
        raise ValueError("Mode4 validation gates or data-contract binding failed")
    provenance = {
        "schema_version": 1,
        "provenance_id": "state54_replay_train_only_sampler_companion_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted",
        "training_seed": args.training_seed,
        "sample_schedule_sha256": seed_schedule["schedule_sha256"],
        "sampler": {
            "path": str(sampler), "step": args.expected_step,
            "metadata_sha256": metadata_sha, "profile_sha256": profile_sha,
            "params_metadata_sha256": sha256(files["params_metadata"]),
            "norm_provenance_sha256": norm_provenance_sha,
            "optimizer_present": False,
        },
        "data_contract": {"path": str(args.data_contract.resolve()), "sha256": contract_sha},
        "data_release": {"path": str(args.data_release.resolve()), "sha256": data_release_sha},
        "feature_release": {"path": str(args.feature_release.resolve()), "sha256": feature_release_sha},
        "formal_protocol": {"path": str(args.formal_protocol.resolve()), "sha256": protocol_sha},
        "coverage_schedule": {"path": str(args.coverage_schedule.resolve()), "sha256": schedule_sha},
        "norm": {"path": str(norm_path), "sha256": norm_sha},
        "token_audit": {"path": str(token_path), "sha256": token_sha},
        "mode4_validation": {"path": str(args.mode4_validation.resolve()), "sha256": validation_sha},
        "runtime_commits": contract.get("runtime_commits"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "accepted", "output": str(args.output.resolve()), "sha256": sha256(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
