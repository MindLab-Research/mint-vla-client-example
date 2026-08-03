import numpy as np
import pytest

from scripts import mano_state46_contract as state46


def test_state46_layout_is_28d_state44_extension():
    frames = 8
    qpos = np.arange(frames * 28, dtype=np.float32).reshape(frames, 28) / 100.0
    contacts = np.zeros((frames, 5), dtype=np.float32)
    contacts[1:4, :2] = 1.0
    lift = np.linspace(0.0, 0.07, frames, dtype=np.float32)
    surface = np.full((frames, 5), 0.02, dtype=np.float32)
    radial = np.tile(np.arange(frames, dtype=np.float32)[:, None], (1, 5)) * -0.001
    floor = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    result = state46.assemble_state46_sequence(
        hand_qpos=qpos,
        contacts=contacts,
        object_lift=lift,
        signed_surface_distances=surface,
        radial_distances=radial,
        floor_support=floor,
    )

    assert result.shape == (frames, 46)
    np.testing.assert_array_equal(result[:, :28], qpos)
    np.testing.assert_array_equal(result[:, 28:33], contacts)
    np.testing.assert_array_equal(result[:, 33], lift)
    np.testing.assert_array_equal(result[:, 34:39], surface)
    np.testing.assert_array_equal(result[:, 44], floor)
    np.testing.assert_allclose(result[:5, 39:44], 0.0)
    np.testing.assert_allclose(result[5:, 39:44], 0.2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result[1:4, 45], [0.0, 0.005, 0.010], atol=1e-7)


def test_action32_is_absolute_target28_plus_physical_zero_padding():
    targets = np.arange(4 * 28, dtype=np.float32).reshape(4, 28)
    actions = state46.absolute_target_actions32(targets)
    assert actions.shape == (4, 32)
    np.testing.assert_array_equal(actions[:, :28], targets)
    np.testing.assert_array_equal(actions[:, 28:], 0.0)
    assert state46.MANO_28D_DELTA_MASK_SEGMENTS == (3, -3, 22, -4)


def test_state46_timestamp_contract_is_exact_200hz():
    state46.validate_timestamps(np.arange(5) * 0.005, 5)
    with pytest.raises(ValueError, match="5 ms"):
        state46.validate_timestamps(np.asarray([0.0, 0.006]), 2)


def test_client_target_projection_accepts_target28_and_padding4():
    from scripts.target_actions import (
        MANO_28D_DELTA_MASK_SEGMENTS,
        urdf_target_absolute_actions,
    )

    state = np.zeros((3, 46), dtype=np.float32)
    target = np.arange(3 * 28, dtype=np.float32).reshape(3, 28)
    row = {
        "state": state,
        "actions": np.zeros((3, 32), dtype=np.float32),
        "hands": [{"hand_name": "right", "urdf_dof_target": target}],
    }
    actions = urdf_target_absolute_actions(row, action_dim=32)
    np.testing.assert_array_equal(actions[:, :28], target)
    np.testing.assert_array_equal(actions[:, 28:], 0.0)
    assert MANO_28D_DELTA_MASK_SEGMENTS == (3, -3, 22, -4)
