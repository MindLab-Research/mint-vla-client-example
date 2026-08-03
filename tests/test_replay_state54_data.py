from __future__ import annotations

import numpy as np
import pytest

from scripts import replay_state54_data as replay


def _write_trace(path, *, frames: int = 3):
    np.savez_compressed(
        path,
        timestamp=np.arange(frames, dtype=np.float64) * 0.005,
        hand_qpos=np.zeros((frames, 26), dtype=np.float32),
        object_position=np.zeros((frames, 3), dtype=np.float32),
        object_quaternion_wxyz=np.tile(
            np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1)
        ),
        contacts=np.zeros((frames, 5), dtype=np.float32),
        absolute_target_qpos=np.zeros((frames, 26), dtype=np.float32),
    )


def test_load_accepted_trace_requires_exact_shapes_and_clock(tmp_path):
    path = tmp_path / "trace.npz"
    _write_trace(path)
    trace = replay.load_accepted_trace(path)
    assert trace["hand_qpos"].shape == (3, 26)
    assert trace["contacts"].shape == (3, 5)

    with np.load(path) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["timestamp"] = np.array([0.0, 0.005, 0.011], dtype=np.float64)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="exact 5 ms"):
        replay.load_accepted_trace(path)


def test_load_accepted_trace_rejects_missing_semantic_array(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        timestamp=np.array([0.0, 0.005]),
        hand_qpos=np.zeros((2, 26)),
        object_position=np.zeros((2, 3)),
        object_quaternion_wxyz=np.zeros((2, 4)),
        absolute_target_qpos=np.zeros((2, 26)),
    )
    with pytest.raises(ValueError, match="contacts"):
        replay.load_accepted_trace(path)


def test_quaternion_error_is_sign_invariant():
    expected = np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    actual = np.array([[-1, 0, 0, 0], [-0.5, -0.5, -0.5, -0.5]], dtype=np.float32)
    np.testing.assert_array_equal(replay._quaternion_error(actual, expected), 0)


def test_atomic_npz_round_trip(tmp_path):
    path = tmp_path / "features.npz"
    replay.atomic_npz(path, force=np.arange(10, dtype=np.float32).reshape(2, 5))
    with np.load(path) as archive:
        np.testing.assert_array_equal(
            archive["force"], np.arange(10, dtype=np.float32).reshape(2, 5)
        )
