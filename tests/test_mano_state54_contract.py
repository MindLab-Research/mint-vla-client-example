import numpy as np
import pytest

from scripts.mano_state54_contract import (
    FINGER_CONTACT_SLICE,
    FINGER_FORCE_SLICE,
    FINGERTIP_OBJECT_SLICE,
    MULTIFINGER_CONTACT_AGE_INDEX,
    OBJECT_COLLISION_BOXES,
    RELATIVE_VERTICAL_VELOCITY_INDEX,
    STATE_DIM,
    aggregate_finger_contact_and_force,
    axis_angle_to_matrix,
    build_state54,
    build_state54_window_from_features,
    contact_age_seconds,
    fingertips_in_collision_box_frame,
)


def test_collision_box_coordinates_are_scale_and_offset_aware():
    rotation = axis_angle_to_matrix(np.array([0.2, -0.1, 0.3]))
    position = np.array([0.4, -0.2, 0.1])
    for object_name, box in OBJECT_COLLISION_BOXES.items():
        center = position + rotation @ box.local_center
        expected = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        local = expected * box.half_extents
        world = center + local @ rotation.T
        actual = fingertips_in_collision_box_frame(world, position, rotation, object_name)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def test_force_aggregation_sums_record_magnitudes_then_log1p():
    records = [
        {"joint_name": "index2", "object_name": "cube1", "contact_pairs": [{"force_normal": [3, 4, 0]}]},
        {"joint_name": "index3", "object_name": "cube1", "contact_pairs": [{"force_normal": [0, 0, 12]}]},
        {"joint_name": "thumb3", "object_name": "cube1", "contact_pairs": [{"force_normal": [0, 8, 15]}]},
        {"joint_name": "ring3", "object_name": "cube2", "contact_pairs": [{"force_normal": [99, 0, 0]}]},
    ]
    contacts, forces = aggregate_finger_contact_and_force(records, "cube1")
    np.testing.assert_array_equal(contacts, [1, 1, 0, 0, 0])
    np.testing.assert_allclose(forces[:2], np.log1p([17, 17]), rtol=1e-6)
    np.testing.assert_array_equal(forces[2:], 0)


def test_contact_age_resets_and_clips_at_one_second():
    contacts = np.zeros((207, 5), dtype=np.float32)
    contacts[1:204, :2] = 1
    contacts[204, 0] = 1
    contacts[205:, 1:3] = 1
    age = contact_age_seconds(contacts)
    assert age[0] == 0 and age[1] == 0
    assert age[2] == pytest.approx(0.005)
    assert age[203] == pytest.approx(1.0)
    assert age[204] == 0 and age[205] == 0 and age[206] == pytest.approx(0.005)


def test_build_state54_requires_every_feature_and_preserves_layout():
    state = build_state54(
        hand_qpos=np.arange(26, dtype=np.float32),
        finger_contacts=np.array([1, 0, 1, 0, 0], dtype=np.float32),
        lift_height=0.25,
        fingertip_collision_box_xyz=np.arange(15, dtype=np.float32).reshape(5, 3),
        finger_log1p_force=np.arange(5, dtype=np.float32),
        relative_vertical_velocity=-0.2,
        multifinger_contact_age=0.4,
    )
    assert state.shape == (STATE_DIM,)
    np.testing.assert_array_equal(state[FINGER_CONTACT_SLICE], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(state[FINGERTIP_OBJECT_SLICE], np.arange(15))
    np.testing.assert_array_equal(state[FINGER_FORCE_SLICE], np.arange(5))
    assert state[RELATIVE_VERTICAL_VELOCITY_INDEX] == pytest.approx(-0.2)
    assert state[MULTIFINGER_CONTACT_AGE_INDEX] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="finger_log1p_force"):
        build_state54(
            hand_qpos=np.zeros(26), finger_contacts=np.zeros(5), lift_height=0,
            fingertip_collision_box_xyz=np.zeros((5, 3)), finger_log1p_force=np.zeros(4),
            relative_vertical_velocity=0, multifinger_contact_age=0,
        )


def test_replay_feature_window_preserves_force_and_resets_temporal_state():
    frames = 4
    qpos = np.zeros((frames, 26), dtype=np.float32)
    qpos[:, 2] = [0.0, 0.01, 0.02, 0.03]
    positions = np.zeros((frames, 3), dtype=np.float32)
    positions[:, 2] = [0.1, 0.12, 0.14, 0.16]
    contacts = np.zeros((frames, 5), dtype=np.float32)
    contacts[1:, :2] = 1
    forces = np.arange(frames * 5, dtype=np.float32).reshape(frames, 5) / 10
    tips = np.arange(frames * 15, dtype=np.float32).reshape(frames, 5, 3) / 10
    state = build_state54_window_from_features(
        hand_qpos=qpos,
        finger_contacts=contacts,
        finger_log1p_force=forces,
        fingertip_collision_box_xyz=tips,
        object_position_world=positions,
        window_start=1,
        window_end=3,
    )
    assert state.shape == (3, 54)
    np.testing.assert_array_equal(state[0, FINGER_FORCE_SLICE], forces[1])
    np.testing.assert_array_equal(state[0, FINGERTIP_OBJECT_SLICE], tips[1].reshape(-1))
    assert state[0, RELATIVE_VERTICAL_VELOCITY_INDEX] == 0
    assert state[0, MULTIFINGER_CONTACT_AGE_INDEX] == 0
    assert state[1, RELATIVE_VERTICAL_VELOCITY_INDEX] == pytest.approx(2.0)
    assert state[1, MULTIFINGER_CONTACT_AGE_INDEX] == pytest.approx(0.005)
    assert state[0, 31] == pytest.approx(0.02)


def test_replay_feature_window_rejects_missing_or_invalid_features():
    with pytest.raises(ValueError, match="finger_log1p_force"):
        build_state54_window_from_features(
            hand_qpos=np.zeros((2, 26)),
            finger_contacts=np.zeros((2, 5)),
            finger_log1p_force=np.zeros((2, 4)),
            fingertip_collision_box_xyz=np.zeros((2, 5, 3)),
            object_position_world=np.zeros((2, 3)),
            window_start=0,
            window_end=1,
        )
    bad_contacts = np.zeros((2, 5)); bad_contacts[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        build_state54_window_from_features(
            hand_qpos=np.zeros((2, 26)),
            finger_contacts=bad_contacts,
            finger_log1p_force=np.zeros((2, 5)),
            fingertip_collision_box_xyz=np.zeros((2, 5, 3)),
            object_position_world=np.zeros((2, 3)),
            window_start=0,
            window_end=1,
        )


def test_state54_openpi_config_keeps_action32_and_strict_state54():
    import openpi_vla_smoke_lance_base as lance_base
    from scripts.openpi_profiles import ACTION_LORA_R16_STATE54_MODEL

    config = lance_base._build_model_config(
        10, action_dim=32, base_model=ACTION_LORA_R16_STATE54_MODEL
    )
    assert config.state_dim == 54
    assert config.action_dim == 32
    assert config.action_horizon == 10
    assert config.max_token_len == 256
    assert config.fail_on_token_truncation is True


def test_augmentation_diagnostics_support_state54_width():
    from scripts.train.train_cube1_01_compare import AugmentationDiagnostics

    diagnostics = AugmentationDiagnostics(54)
    clean = np.zeros(54, dtype=np.float32)
    augmented = clean.copy(); augmented[:26] = 0.1; augmented[32:47] = 0.05
    valid = np.zeros(54, dtype=bool); valid[:26] = True
    causal_derived = np.zeros(54, dtype=bool); causal_derived[32:47] = True
    diagnostics.record(clean, augmented, valid, [1], [2], causal_derived)
    summary = diagnostics.summary(0.05, token_budget=256)
    assert summary["valid_coordinates"] == 26
    assert len(summary["bin_changed_fraction_by_dimension"]) == 54
    assert summary["causal_derived_valid_coordinates"] == 15
    assert summary["causal_derived_bin_changed_fraction"] == 1.0
    assert summary["causal_derived_rms_delta_normalized"] == pytest.approx(0.05)


def test_state54_norm_verifier_authenticates_token_audit(tmp_path, monkeypatch):
    import hashlib, json
    import scripts.mano_state54_contract as contract

    norm = tmp_path / "norm_stats.json"; norm.write_text("exact-norm-bytes")
    norm_sha = hashlib.sha256(norm.read_bytes()).hexdigest()
    monkeypatch.setattr(contract, "STATE54_NORM_SHA256", norm_sha)
    audit_payload = {
        "zero_truncation": True, "overflow_count": 0,
        "audited_active_frames": contract.POPULATION_ACTIVE_FRAMES,
        "profile_max_token_len": contract.PROFILE_MAX_TOKEN_LEN,
        "maximum_token_length": 229, "norm_stats_sha256": norm_sha,
        "population_row_indices_sha256": contract.POPULATION_ROW_INDICES_SHA256,
    }
    audit = tmp_path / "token_audit.json"
    audit.write_text(json.dumps(audit_payload, sort_keys=True))
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    data_contract = {
        "norm_stats_sha256": norm_sha, "state_contract": contract.STATE_CONTRACT_ID,
        "state_dim": 54, "action_dim": 32, "action_horizon": 10,
        "action_source": "urdf_target_absolute",
        "row_indices_sha256": contract.POPULATION_ROW_INDICES_SHA256,
        "trajectory_count": contract.POPULATION_TRAJECTORIES,
        "active_frame_count": contract.POPULATION_ACTIVE_FRAMES,
        "action_vector_count": contract.POPULATION_ACTIVE_FRAMES * 10,
        "force_reference_newtons": contract.FORCE_REFERENCE_NEWTONS,
        "source_interval_seconds": contract.SOURCE_INTERVAL_SECONDS,
        "contact_age_clip_seconds": contract.CONTACT_AGE_CLIP_SECONDS,
        "max_token_len": contract.PROFILE_MAX_TOKEN_LEN,
        "token_audit_sha256": audit_sha,
    }
    (tmp_path / "data_contract.json").write_text(json.dumps(data_contract))
    assert contract.verify_locked_state54_norm_stats(tmp_path) == (norm, norm_sha)

    audit_payload["maximum_token_length"] = 257
    audit.write_text(json.dumps(audit_payload, sort_keys=True))
    data_contract["token_audit_sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
    (tmp_path / "data_contract.json").write_text(json.dumps(data_contract))
    with pytest.raises(ValueError, match="token audit contract is invalid"):
        contract.verify_locked_state54_norm_stats(tmp_path)
