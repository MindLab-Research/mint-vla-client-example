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
