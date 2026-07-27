from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from scripts.eval import infer_mano_mode3 as mode3
from scripts.eval.mode4_support import parse_ordered_unique_csv


def test_b_reconstruction_is_query_anchored_with_absolute_euler():
    query = np.arange(26, dtype=np.float32)
    output = np.full((10, 32), 0.5, dtype=np.float32)
    target = mode3.reconstruct_absolute_target_chunk(query, output)
    np.testing.assert_allclose(target[:, :3], query[:3] + 0.5)
    np.testing.assert_allclose(target[:, 3:6], 0.5)
    np.testing.assert_allclose(target[:, 6:26], query[6:26] + 0.5)


def test_mode3_queries_non_overlapping_ten_frame_chunks_without_smoothing():
    assert mode3.mode3_query_frames(7, 31) == [7, 17, 27]


def test_unnormalization_tail_is_canonicalized_to_exact_physical_zero():
    epsilon_tail = np.zeros((10, 32), dtype=np.float32)
    epsilon_tail[:, 26:] = 1e-6
    canonical = mode3.canonicalize_physical_padding(epsilon_tail)
    mode3.assert_physical_padding_zero(canonical)
    assert np.array_equal(canonical[:, 26:], np.zeros((10, 6), dtype=np.float32))


def test_mode3_video_frame_is_side_by_side_sim_and_dataset():
    sim = np.full((40, 20, 3), 10, dtype=np.uint8)
    dataset = np.full((40, 20, 3), 200, dtype=np.uint8)
    comparison = mode3.side_by_side_video_frame(
        sim, dataset, sim_label="SIM", dataset_label="DATASET",
    )
    assert comparison.shape == (40, 40, 3)
    np.testing.assert_array_equal(comparison[39, :20], sim[39])
    np.testing.assert_array_equal(comparison[39, 20:], dataset[39])


def test_reference_render_only_forwards_never_steps():
    data = SimpleNamespace(qpos=np.ones(40), qvel=np.ones(40))
    renderer = object()
    with mock.patch.object(mode3.mujoco, "mj_forward") as forward, mock.patch.object(
        mode3.physics, "render_current_state", return_value=("head", "wrist")
    ) as render, mock.patch.object(mode3.mujoco, "mj_step", side_effect=AssertionError("must not step")):
        assert mode3.render_reference_state(
            model=object(), data=data, renderer=renderer, object_addr=0,
            hand_addrs=list(range(7, 33)), hand_state=np.arange(26),
            object_pos=np.array([1.0, 2.0, 3.0]), object_rot=np.zeros(3),
        ) == ("head", "wrist")
    forward.assert_called_once()
    render.assert_called_once()
    np.testing.assert_array_equal(data.qvel, 0)


def test_extended_state_uses_pair_presence_and_reference_lift():
    expected_contacts = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    with mock.patch.object(mode3.physics, "finger_contacts_from_mujoco", return_value=expected_contacts) as contacts:
        state = mode3.build_extended_sim_state(
            hand_qpos=np.arange(26), model="m", data="d", object_name="cube1",
            keypoint_geom_ids={1}, object_geom_ids={2}, geom_id_to_finger={1: "index"},
            reference_object_z=0.8, source_object_z0=0.25,
        )
    np.testing.assert_array_equal(state[:26], np.arange(26, dtype=np.float32))
    np.testing.assert_array_equal(state[26:31], expected_contacts)
    assert state[31] == pytest.approx(0.55)
    contacts.assert_called_once_with("m", "d", "cube1", keypoint_geom_ids={1}, object_geom_ids={2}, geom_id_to_finger={1: "index"})


def test_norm_failure_fails_before_norm_stats_load(monkeypatch, tmp_path):
    args = SimpleNamespace(
        base_url="x", act_batch_size=4, contact_context_frames=0, norm_stats_dir=tmp_path,
    )
    monkeypatch.setattr(mode3, "parse_args", lambda: args)
    with mock.patch.object(mode3, "verify_locked_norm_stats", side_effect=ValueError("bad sha")), \
         mock.patch.object(mode3.L.normalize, "load") as load:
        with pytest.raises(ValueError, match="bad sha"):
            mode3.main()
    load.assert_not_called()


def test_ordered_rows_are_unique_and_preserve_first_occurrence():
    assert parse_ordered_unique_csv("9,2,9,1", option="--row-indices") == [9, 2, 1]


def test_metadata_identifies_kinematic_contract(tmp_path):
    args = SimpleNamespace(
        act_batch_size=4, norm_sha_expected="expected", norm_sha_actual="actual",
        client_commit="client", backend_commit="backend", model_commit="model",
        language_conditioning="gesture",
    )
    window = SimpleNamespace(frame_count=3, as_dict=lambda: {"start_frame": 4, "end_frame": 6, "frame_count": 3})
    metadata = mode3._result_metadata(
        args, row_index=12, object_name="cube1", window=window, source_frames=20,
        query_timings=[], out=tmp_path, head_path=tmp_path / "head.mp4", wrist_path=tmp_path / "wrist.mp4",
    )
    assert metadata["mode"] == "historical_kinematic_mode3_sim_no_smooth"
    assert metadata["physics_dynamics"] is False
    assert metadata["object_pose_source"] == "reference_trajectory"
    assert metadata["state_observation_source"] == metadata["image_observation_source"] == "sim"
    assert metadata["temporal_ensemble"] is False
    assert metadata["query_stride"] == 10
    assert metadata["contact_rule"] == mode3.CONTACT_RULE
    assert metadata["norm_sha_expected"] == "expected"
    assert metadata["client_commit"] == "client"
