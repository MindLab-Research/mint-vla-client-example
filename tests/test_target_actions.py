from __future__ import annotations

import unittest

import numpy as np

from scripts.target_actions import (
    MEASURED_DELTA,
    PD_TARGET_DELTA,
    fit_pd_response_gains,
    pd_target_delta_actions,
    project_row_actions,
)


class TargetActionsTests(unittest.TestCase):
    def test_pd_target_is_projected_to_delta_with_zero_padding(self) -> None:
        state = np.zeros((3, 32), dtype=np.float32)
        state[:, :26] = np.arange(26, dtype=np.float32)
        target = state[:, :26] + np.asarray([0.1, 0.2, 0.3], dtype=np.float32)[:, None]
        row = {
            "state": state,
            "actions": np.full((3, 32), 9.0, dtype=np.float32),
            "hands": [{"urdf_dof_target": target}],
        }
        actions = pd_target_delta_actions(row)
        np.testing.assert_allclose(actions[:, :26], [[0.1] * 26, [0.2] * 26, [0.3] * 26], atol=1e-6)
        np.testing.assert_array_equal(actions[:, 26:], 0)
        projected = project_row_actions(row, PD_TARGET_DELTA)
        self.assertIsNot(projected, row)
        self.assertIs(projected["state"], row["state"])
        np.testing.assert_array_equal(projected["actions"], actions)
        np.testing.assert_array_equal(projected["measured_actions"], row["actions"])

    def test_measured_delta_path_is_literal(self) -> None:
        row = {"actions": np.zeros((1, 32), dtype=np.float32)}
        self.assertIs(project_row_actions(row, MEASURED_DELTA), row)

    def test_diagonal_pd_response_gain_fit(self) -> None:
        state = np.zeros((4, 32), dtype=np.float32)
        target = np.ones((4, 26), dtype=np.float32)
        measured = np.zeros((4, 32), dtype=np.float32)
        measured[:, :26] = 0.25
        gains = fit_pd_response_gains(state, measured, target)
        np.testing.assert_allclose(gains, 0.25)

    def test_target_shape_mismatch_fails(self) -> None:
        row = {
            "state": np.zeros((3, 32), dtype=np.float32),
            "hands": [{"urdf_dof_target": np.zeros((2, 26), dtype=np.float32)}],
        }
        with self.assertRaisesRegex(ValueError, "urdf_dof_target"):
            pd_target_delta_actions(row)


if __name__ == "__main__":
    unittest.main()
