import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "releases" / "state41_28dof_v1.json"


def load_release() -> dict:
    return json.loads(MANIFEST.read_text())


def test_state41_release_candidate_binds_validated_source_tuple() -> None:
    release = load_release()

    assert release["schema_version"] == 1
    assert release["release_id"] == "state41_28dof_v1_rc1"
    assert release["status"] == "release_candidate"
    assert release["audience"] == "internal_vepfs"
    assert release["repositories"]["client"] == {
        "repository": "MindLab-Research/mint-vla-client-example",
        "branch": "feature/mano-state41-28dof-v1",
        "validated_code_commit": "f0a4b69d784586d6695c1a8f1a53b835f067f6d1",
    }
    assert release["repositories"]["mint"]["commit"] == (
        "9e1d5491fade1ace61ca464754d5928c511c20cf"
    )
    assert release["repositories"]["openpi"]["commit"] == (
        "33ccdae4dc08fa2ac1c4b0d7788634b1fb6d755f"
    )
    assert release["repositories"]["manorl"]["commit"] == (
        "e17f0122decddffc348ec10d0ed42552a0540e1b"
    )


def test_state41_release_candidate_separates_observation_action_and_control() -> None:
    contract = load_release()["runtime_contract"]

    assert contract["state_contract"] == "mano_state41_native_sim_28d_v1"
    assert contract["state_dim"] == 41
    assert contract["action_dim"] == 32
    assert contract["physical_hand_target_dim"] == 28
    assert contract["action_horizon"] == 10
    assert contract["delta_mask_segments"] == [3, -3, 22, -4]
    assert contract["language_source"] == "formal_release_metadata"
    assert contract["contact_window"] == {
        "frame_window": "contact",
        "context_frames": 100,
        "missing_policy": "error",
    }
    assert "frame_count-1" in contract["frame_count_semantics"]


def test_state41_release_candidate_binds_train_only_profile() -> None:
    release = load_release()
    dataset = release["dataset"]
    profile = release["profile"]

    assert dataset["grade_a_population_rows"] == 4856
    assert dataset["train_rows"] == 4613
    assert dataset["validation_rows"] == 243
    assert dataset["split_seed"] == 42
    assert dataset["split_strata"] == "object_x_gesture"
    assert profile["norm_population"] == "train_only_contact_window"
    assert profile["norm_sha256"] == (
        "c276e12682dca4cd6559bd1d8c201f4cc7e488da6ebdcc2a67c8f137458f28ec"
    )


def test_state41_release_candidate_records_acceptance_evidence() -> None:
    evidence = load_release()["validation_evidence"]
    smoke = evidence["cube1_gesture03_smoke"]
    full = evidence["grade_a_step20000_matched_mode4"]
    matched = evidence["matched_population"]

    assert smoke["status"] == "passed"
    assert (smoke["strict_success_count"], smoke["row_count"], smoke["video_count"]) == (
        15,
        15,
        45,
    )
    assert full["status"] == "passed"
    assert full["validation"]["strict_success_count"] == 107
    assert full["validation"]["row_count"] == 243
    assert full["matched_train"]["strict_success_count"] == 113
    assert full["matched_train"]["row_count"] == 243
    assert sum(full["paired_outcomes"].values()) == 243
    assert matched["pair_count"] == 243
    assert matched["validation_train_row_overlap"] == 0
    assert matched["exact_frame_count_pairs"] == 239


def test_state41_release_candidate_is_documented_without_replacing_state32_release() -> None:
    release = load_release()
    historical = json.loads(
        (ROOT / "config" / "datasets" / "mano_dataset_release.json").read_text()
    )
    readme = (ROOT / "README.md").read_text()

    assert historical["runtime_contracts"]["state"]["id"] == (
        "mano_five_finger_contact_lift_v1"
    )
    assert release["runtime_contract"]["state_dim"] == 41
    assert "config/releases/state41_28dof_v1.json" in readme
    assert "Frozen State41/Action32 28DoF release provenance" in readme
    assert "Maintained State41-derived full-task contract" in readme
    retired_width_marker = "state" + str(46) + "-28dof"
    assert retired_width_marker not in readme.lower()
    assert release["profile"]["norm_sha256"] in readme
