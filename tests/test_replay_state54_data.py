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


def test_feature_store_authenticates_release_identity_and_arrays(tmp_path):
    root = tmp_path / "release"
    features = root / "features"
    features.mkdir(parents=True)
    feature_path = features / "row00007.npz"
    replay.atomic_npz(
        feature_path,
        timestamp=np.arange(3, dtype=np.float64) * 0.005,
        finger_contacts=np.zeros((3, 5), dtype=np.float32),
        finger_log1p_force=np.ones((3, 5), dtype=np.float32),
        fingertip_collision_box_xyz=np.zeros((3, 5, 3), dtype=np.float32),
    )
    source = tmp_path / "source.lance"
    release = {
        "status": "accepted",
        "feature_schema_id": replay.FEATURE_SCHEMA_ID,
        "source_dataset": str(source),
        "row_count": 1,
        "entry_count": 1,
        "entries": [
            {
                "row_index": 7,
                "row_uuid": "uuid-7",
                "object_name": "cube1",
                "frame_count": 3,
                "output_npz": str(feature_path),
                "output_npz_sha256": replay.sha256_file(feature_path),
            }
        ],
    }
    replay.atomic_json(root / "release.json", release)
    store = replay.ReplayState54FeatureStore(
        root,
        source_dataset=source,
        expected_release_sha256=replay.sha256_file(root / "release.json"),
    )
    loaded = store.load(7, row_uuid="uuid-7", object_name="cube1", frame_count=3)
    assert loaded["timestamp"].dtype == np.float64
    np.testing.assert_array_equal(loaded["finger_log1p_force"], 1)
    with pytest.raises(ValueError, match="row_uuid mismatch"):
        store.load(7, row_uuid="wrong", object_name="cube1", frame_count=3)


def test_lance_adapter_uses_replay_features_without_mano_joints():
    import types
    from scripts.mano_state54_contract import STATE_CONTRACT_ID
    from scripts.train.openpi_vla_smoke_lance_base import LanceViewpi05Dataset

    class Store:
        def load(self, row_index, *, row_uuid, object_name, frame_count):
            assert (row_index, row_uuid, object_name, frame_count) == (7, "uuid-7", "cube1", 3)
            return {
                "timestamp": np.arange(3, dtype=np.float64) * 0.005,
                "finger_contacts": np.zeros((3, 5), dtype=np.float32),
                "finger_log1p_force": np.ones((3, 5), dtype=np.float32),
                "fingertip_collision_box_xyz": np.zeros((3, 5, 3), dtype=np.float32),
            }

    dataset = LanceViewpi05Dataset.__new__(LanceViewpi05Dataset)
    dataset._state_contract = STATE_CONTRACT_ID
    dataset._source_row_indices = [7]
    dataset._rows = [
        {
            "index": {"uuid": "uuid-7"},
            "episode_metadata": {"total_frames": 3},
            "trajectory_metadata": {"object_names": ["cube1"]},
        }
    ]
    dataset._row_windows = {0: types.SimpleNamespace(start_frame=1, end_frame=2)}
    dataset._state54_replay_feature_store = Store()
    row = {
        "hands": [{"urdf_dof": np.zeros((3, 26), dtype=np.float32)}],
        "objects": [
            {
                "pos": np.zeros((3, 3), dtype=np.float32),
                "rot_aa": np.zeros((3, 3), dtype=np.float32),
            }
        ],
        "contact": [[], [], []],
    }
    attached = dataset._attach_state54_window(row, 0)
    assert attached["_state54_window"].shape == (2, 54)
    np.testing.assert_array_equal(attached["_state54_window"][:, 47:52], 1)


def test_atomic_npz_round_trip(tmp_path):
    path = tmp_path / "features.npz"
    replay.atomic_npz(path, force=np.arange(10, dtype=np.float32).reshape(2, 5))
    with np.load(path) as archive:
        np.testing.assert_array_equal(
            archive["force"], np.arange(10, dtype=np.float32).reshape(2, 5)
        )
