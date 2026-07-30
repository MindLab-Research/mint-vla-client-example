from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import unittest
from unittest import mock
from io import BytesIO
from pathlib import Path
import tempfile
import types
from PIL import Image
import mujoco
import numpy as np
from scripts.eval import mano_physics_core as physics
from scripts.eval import infer_mano_mode4 as mode4
from scripts.eval.infer_mano_mode4 import (
    MANORL_PHYSICS_SUBSTEPS,
    MANORL_PHYSICS_TIMESTEP,
    expected_mode4_mj_steps,
    reconstruct_absolute_target_chunk,
)


class Mode4ContractTests(unittest.TestCase):
    def test_b_reconstruction_uses_delta_xyz_fingers_absolute_euler(self):
        q = np.arange(26, dtype=np.float32)
        out = np.ones((2, 32), dtype=np.float32) * 0.5
        target = reconstruct_absolute_target_chunk(q, out)
        np.testing.assert_allclose(target[:, :3], np.broadcast_to(q[:3] + 0.5, (2, 3)))
        np.testing.assert_allclose(target[:, 3:6], np.full((2, 3), 0.5, dtype=np.float32))
        np.testing.assert_allclose(target[:, 6:26], np.broadcast_to(q[6:26] + 0.5, (2, 20)))

    def test_exact_replay_timeline(self):
        self.assertEqual(MANORL_PHYSICS_TIMESTEP, 0.0025)
        self.assertEqual(MANORL_PHYSICS_SUBSTEPS, 2)
        self.assertEqual(expected_mode4_mj_steps(715), 1428)

    def test_padded_action_chunk_does_not_alias_source_labels(self):
        actions = np.arange(12 * 32, dtype=np.float32).reshape(12, 32)
        original = actions.copy()
        chunk = mode4.pad_actions(actions, frame=1, window_end=11)
        chunk[:] = -1
        np.testing.assert_array_equal(actions, original)

    def test_row_summary_records_all_source_commits(self):
        args = types.SimpleNamespace(
            model_path="mint://checkpoint",
            model="openpi/pi05-action-lora-r16-finetune",
            client_commit="client-sha",
            backend_commit="mint-sha",
            model_commit="openpi-sha",
            action_source="urdf_target_absolute",
            language_conditioning="gesture",
            _gesture_index=None,
            lance_dataset=Path("/dataset.lance"),
            norm_stats_dir=Path("/norm"),
            extended_state=False,
            norm_sha_expected=None,
            norm_sha_actual=None,
            act_mode="batch",
            act_batch_size=4,
            max_warm_request_seconds=2.0,
        )
        summary = mode4.build_row_summary(
            args=args, row_index=3, normalization_rows=[1, 2], results=[]
        )
        self.assertEqual(summary["client_commit"], "client-sha")
        self.assertEqual(summary["backend_commit"], "mint-sha")
        self.assertEqual(summary["model_commit"], "openpi-sha")

    def test_retained_session_marker_records_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            args = types.SimpleNamespace(
                output_dir=Path(temp),
                base_url="http://127.0.0.1:30536",
                model="openpi/pi05-action-lora-r16-finetune",
                model_path="mint://owner/checkpoint",
                owner_id="owner-1",
            )
            marker = mode4.write_retained_session_marker(args, "session-123")
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["status"], "retained")
            self.assertEqual(payload["action_session_id"], "session-123")
            self.assertEqual(payload["model_path"], "mint://owner/checkpoint")
            self.assertEqual(payload["owner_id"], "owner-1")

    def test_native_servo_and_collision_scene(self):
        kp, dampratio, effort = physics.servo_parameters()
        np.testing.assert_allclose(kp[:6], 100)
        np.testing.assert_allclose(dampratio[:6], 1.4)
        np.testing.assert_allclose(effort[:6], 5000)
        scene = physics.make_scene(
            "cube1", 64, 64, physics=True, physics_timestep=physics.DT, create_renderer=False
        )
        tmp, model, data, renderer, objaddr, _, handaddr, _, limits = scene
        try:
            self.assertIsNone(renderer)
            self.assertEqual(model.nu, 26)
            self.assertAlmostEqual(model.opt.timestep, 0.0025)
            data.qpos[objaddr + 3] = 1.0
            data.qpos[handaddr] = 0
            mujoco.mj_forward(model, data)
            physics.step_servo(
                model=model, data=data, target=np.zeros(26), substeps=2, object_name="cube1"
            )
            self.assertAlmostEqual(data.time, 0.005)
        finally:
            tmp.cleanup()


class Mode4SessionLifecycleTests(unittest.TestCase):
    def make_args(self, output_dir: Path, *, action_session_id=None, keep_action_session=False):
        return types.SimpleNamespace(
            base_url="http://127.0.0.1:30536",
            api_key="tml-dummy",
            model_path="mint://owner/checkpoint",
            model="openpi/pi05-action-lora-r16-finetune",
            owner_id="owner-1",
            lance_dataset=Path("/dataset.lance"),
            row_index=None,
            row_indices="0",
            action_source="urdf_target_absolute",
            extended_state=False,
            language_conditioning="object_only",
            gesture_index=Path("/gesture.index.json"),
            normalization_row_indices="0",
            norm_stats_dir=Path("/norm"),
            output_dir=output_dir,
            action_session_id=action_session_id,
            keep_action_session=keep_action_session,
            chunk_stride=5,
            max_frames=2,
            act_mode="batch",
            act_batch_size=4,
            max_warm_request_seconds=2.0,
            temporal_decay=0.4,
            fps=10.0,
            width=32,
            height=24,
            video_mode="none",
            client_commit="client-sha",
            backend_commit="mint-sha",
            model_commit="openpi-sha",
        )

    def run_main_with_session(self, args):
        row = {"index": {"uuid": "u"}, "state": np.zeros((2, 32), np.float32)}
        target = {
            "index": {"uuid": "u"},
            "hands": [{"urdf_dof": np.zeros((2, 26), np.float32)}],
        }

        def take(_indices, columns):
            return types.SimpleNamespace(
                to_pylist=lambda: [target if columns == ["hands", "index"] else row]
            )

        source = types.SimpleNamespace(count_rows=lambda: 1, take=take)
        patches = (
            mock.patch.object(mode4, "parse_args", return_value=args),
            mock.patch.object(mode4.lance, "dataset", return_value=source),
            mock.patch.object(mode4.L.normalize, "load", return_value={}),
            mock.patch.object(mode4.full, "build_model_config", return_value=object()),
            mock.patch.object(mode4.L, "_make_data_config", return_value=object()),
            mock.patch.object(mode4, "project_row_actions", side_effect=lambda value, _: value),
            mock.patch.object(mode4, "condition_row_language", side_effect=lambda value, *_a, **_k: value),
            mock.patch.object(mode4.full, "safe_object_name", return_value="cube1"),
            mock.patch.object(mode4, "run_variant", return_value={"ok": True}),
            mock.patch.object(mode4, "build_row_summary", return_value={"mode": "test"}),
            mock.patch.object(mode4, "create_session", return_value="created-session"),
            mock.patch.object(mode4, "delete_session"),
        )
        with ExitStack() as stack:
            entered = [stack.enter_context(patch) for patch in patches]
            self.assertEqual(mode4.main(), 0)
            run_mock = entered[8]
            create_mock = entered[10]
            delete_mock = entered[11]
        return run_mock, create_mock, delete_mock

    def test_keep_action_session_retains_created_session_and_writes_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.make_args(Path(temp), keep_action_session=True)
            run_mock, create_mock, delete_mock = self.run_main_with_session(args)
            create_mock.assert_called_once()
            delete_mock.assert_not_called()
            self.assertEqual(run_mock.call_args.kwargs["session_id"], "created-session")
            marker = json.loads((Path(temp) / "action_session.retained.json").read_text())
            self.assertEqual(marker["action_session_id"], "created-session")
            summary = json.loads((Path(temp) / "summary.json").read_text())
            self.assertTrue(summary["action_session"]["retained"])
            self.assertEqual(summary["action_session"]["cleanup_owner"], "caller")

    def test_external_action_session_is_neither_created_nor_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.make_args(Path(temp), action_session_id="external-session")
            run_mock, create_mock, delete_mock = self.run_main_with_session(args)
            create_mock.assert_not_called()
            delete_mock.assert_not_called()
            self.assertEqual(run_mock.call_args.kwargs["session_id"], "external-session")
            summary = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(summary["action_session"]["source"], "external")
            self.assertEqual(summary["action_session"]["cleanup_owner"], "external")


class LockstepBatcherTests(unittest.TestCase):
    def test_batches_real_rows_and_retires_shorter_trajectory(self):
        dispatched = []

        def dispatch(datums):
            dispatched.append(list(datums))
            real = len(datums)
            return [
                (datum, datum, datum, {"padding_count": 4 - real})
                for datum in datums
            ]

        batcher = mode4.LockstepActionBatcher([0, 1], dispatch)

        def row_zero():
            try:
                return [batcher.query(0, "row0-step0"), batcher.query(0, "row0-step1")]
            finally:
                batcher.retire(0)

        def row_one():
            try:
                return [batcher.query(1, "row1-step0")]
            finally:
                batcher.retire(1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            zero_future = pool.submit(row_zero)
            one_future = pool.submit(row_one)
            zero = zero_future.result(timeout=5)
            one = one_future.result(timeout=5)
        stats = batcher.close()

        self.assertEqual(dispatched[0], ["row0-step0", "row1-step0"])
        self.assertEqual(dispatched[1], ["row0-step1"])
        self.assertEqual(zero[0][0], "row0-step0")
        self.assertEqual(one[0][0], "row1-step0")
        self.assertEqual(
            stats,
            {"batch_requests": 2, "real_observations": 3, "padding_observations": 5},
        )


class Mode4LoopTests(unittest.TestCase):
    def test_contact_window_initializes_at_absolute_start_and_keeps_full_reference(self):
        frame = np.full((24, 32, 3), 127, dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG")
        blob = buf.getvalue()
        source_state = np.zeros((3, 32), np.float32)
        source_state[0, 0] = 0.123
        source_state[1, 0] = 0.456
        row = {
            "state": source_state,
            "actions": np.zeros((3, 32), np.float32),
            "image": [blob, blob, blob],
            "wrist_image": [blob, blob, blob],
            "objects": [
                {
                    "pos": [[0.3, 0, 0.1], [0.4, 0, 0.1], [0.4, 0, 0.1]],
                    "rot_aa": [[0, 0, 0]] * 3,
                }
            ],
            "timestamp": np.asarray([0.0, 0.005, 0.01], dtype=np.float64),
            "prompt": "pick up cube1",
            "trajectory_metadata": {"data_fps": 100},
            "episode_metadata": {"fps": 100},
        }
        window = types.SimpleNamespace(
            start_frame=1,
            end_frame=2,
            frame_count=2,
            status="contact_window",
            first_contact_frame=2,
            last_contact_frame=2,
            context_frames=100,
        )
        data = types.SimpleNamespace(qpos=np.zeros(33), qvel=np.zeros(32), time=0.0)
        data.qpos[3] = 1
        renderer = types.SimpleNamespace(close=lambda: None)
        scene_tmp = tempfile.TemporaryDirectory()
        args = types.SimpleNamespace(
            action_source="urdf_target_absolute",
            row_index=0,
            max_frames=0,
            width=32,
            height=24,
            chunk_stride=1,
            output_dir=None,
            fps=10,
            model="m",
            act_mode="single",
            extended_state=True,
            language_conditioning="gesture",
            max_warm_request_seconds=0,
            temporal_decay=0.4,
            frame_window="contact",
            contact_context_frames=100,
            missing_contact_policy="error",
            contact_window_manifest=Path("/contact.json"),
            base_url="x",
            model_path="p",
            client_commit="client-sha",
            backend_commit="mint-sha",
            model_commit="openpi-sha",
        )
        pred = np.zeros((10, 32), np.float32)
        pred[:, :26] = 0.01
        set_calls = []

        def set_scene(_m, d, **kw):
            set_calls.append(kw)
            d.qpos[0:3] = kw["object_pos"]
            d.qpos[3] = 1
            d.qpos[7:33] = kw["state"]

        def step(**kw):
            d = kw["data"]
            d.qpos[7:33] = kw["target"]
            d.qpos[0] += 0.02
            d.time += 0.005
            return {
                "hand_object_contact": True,
                "object_floor_contact": False,
                "hand_floor_contact": False,
                "max_ncon": 1,
                "max_contact_force": 2,
                "max_abs_actuator_force": 3,
            }

        observation_contacts = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        with tempfile.TemporaryDirectory() as out, mock.patch.object(
            mode4.full, "row_frame_count", return_value=3
        ), mock.patch.object(
            mode4.full, "resolve_row_window", return_value=window
        ), mock.patch.object(
            mode4.full, "set_scene_state", side_effect=set_scene
        ), mock.patch.object(
            mode4.physics,
            "make_scene",
            return_value=(
                scene_tmp,
                object(),
                data,
                renderer,
                0,
                0,
                list(range(7, 33)),
                list(range(6, 32)),
                object(),
            ),
        ), mock.patch.object(
            mode4.physics,
            "resolve_keypoint_geom_ids",
            return_value=(object(), object(), object()),
        ), mock.patch.object(
            mode4.physics,
            "finger_contacts_from_mujoco",
            return_value=observation_contacts,
        ), mock.patch.object(
            mode4.physics,
            "physics_substeps_for_row",
            side_effect=AssertionError("metadata timing must not drive Mode4"),
        ), mock.patch.object(
            mode4.physics, "render_current_state", return_value=(frame, frame)
        ), mock.patch.object(
            mode4.physics,
            "nearest_wrapped_position_target",
            side_effect=lambda current, delta, limits: (current + delta + 0.001, None),
        ), mock.patch.object(
            mode4.physics, "step_servo", side_effect=step
        ), mock.patch.object(
            mode4, "new_clipping_diagnostics", return_value={}
        ), mock.patch.object(
            mode4, "record_clipping"
        ), mock.patch.object(
            mode4, "build_datum", return_value={"observation": {}, "data_config": object()}
        ), mock.patch.object(
            mode4,
            "query_action",
            return_value=(pred, pred, pred, {"wall_seconds": 0.01, "used_data_sharding": False}),
        ), mock.patch.object(
            mode4, "acquire_action_session", return_value=("s", False)
        ), mock.patch.object(
            mode4.mujoco, "mj_forward"
        ):
            result = mode4.run_variant(
                args=args,
                row=row,
                data_config=object(),
                mode="mode4",
                headers={},
                object_name="cube1",
                session_id="s",
                output_dir=Path(out),
            )
            self.assertEqual(len(set_calls), 1)
            self.assertAlmostEqual(float(set_calls[0]["state"][0]), 0.456, places=6)
            self.assertAlmostEqual(float(set_calls[0]["object_pos"][0]), 0.4, places=6)
            self.assertEqual(result["physics"]["mj_step_calls"], 2)
            self.assertEqual(result["physics"]["intervals"], 1)
            self.assertEqual(result["object_pose_source"], "sim_owned_after_source_frame_1")
            self.assertEqual(result["frame_window"]["type"], "contact")
            self.assertEqual(result["frame_window"]["start_frame"], 1)
            self.assertEqual(result["frame_window"]["end_frame"], 2)
            self.assertEqual(result["video_windows"]["dataset_reference"]["frame_count"], 3)
            self.assertEqual(result["video_windows"]["physics_comparison"]["frame_count"], 2)
            self.assertEqual(result["query_timings"][0]["source_frame"], 1)
            self.assertEqual(result["client_commit"], "client-sha")
            self.assertEqual(result["backend_commit"], "mint-sha")
            self.assertEqual(result["model_commit"], "openpi-sha")
            self.assertEqual(result["action_session_id"], "s")
            self.assertFalse(result["action_session_owned_by_variant"])
            output = Path(out) / "mode4"
            self.assertAlmostEqual(
                np.load(output / "object_position_sim.npy")[-1, 0], 0.42, places=6
            )
            expected_arrays = {
                "actions_raw_pred_normalized",
                "actions_raw_pred_physical",
                "actions_commanded_physical",
                "preclip_absolute_targets",
                "servo_position_targets",
                "servo_target_clipping_correction",
                "actions_applied_physical",
                "rollout_observation_state",
                "rollout_observation_contacts",
                "rollout_observation_lift",
                "physics_contact_flags",
                "step_max_contact_force",
                "hand_state_sim",
                "object_position_sim",
                "object_quaternion_sim",
            }
            self.assertEqual(set(result["arrays"]), expected_arrays)
            for name, path in result["arrays"].items():
                self.assertEqual(Path(path), output / f"{name}.npy")
                self.assertTrue(Path(path).is_file())

            hand_before = np.load(output / "hand_state_sim.npy")[0]
            commanded = np.load(output / "actions_commanded_physical.npy")[0]
            preclip_target = np.load(output / "preclip_absolute_targets.npy")[0]
            servo_target = np.load(output / "servo_position_targets.npy")[0]
            correction = np.load(output / "servo_target_clipping_correction.npy")[0]
            np.testing.assert_allclose(preclip_target, hand_before + commanded[:26])
            np.testing.assert_array_equal(correction, servo_target - preclip_target)

            observation_state = np.load(output / "rollout_observation_state.npy")
            np.testing.assert_array_equal(observation_state[0, 26:31], observation_contacts)
            np.testing.assert_array_equal(
                np.load(output / "rollout_observation_contacts.npy")[0], observation_contacts
            )
            np.testing.assert_array_equal(
                np.load(output / "rollout_observation_lift.npy"), np.asarray([0.0], np.float32)
            )
            np.testing.assert_array_equal(
                np.load(output / "physics_contact_flags.npy"), np.asarray([[True, False, False]])
            )
            np.testing.assert_array_equal(
                np.load(output / "step_max_contact_force.npy"), np.asarray([2.0], np.float32)
            )


class Mode4State44LoopTests(unittest.TestCase):
    def test_state44_live_observation_and_component_artifacts(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        row = {
            "state": np.zeros((2, 32), np.float32),
            "actions": np.zeros((2, 32), np.float32),
            "image": [b"", b""],
            "wrist_image": [b"", b""],
            "objects": [{"pos": [[0.3, 0, 0.1]] * 2, "rot_aa": [[0, 0, 0]] * 2}],
            "timestamp": np.asarray([0.0, 0.005]),
            "prompt": "pick up cube1",
            "trajectory_metadata": {"data_fps": 100},
            "episode_metadata": {"fps": 100},
        }
        window = types.SimpleNamespace(
            start_frame=0, end_frame=1, frame_count=2, status="full",
            first_contact_frame=None, last_contact_frame=None, context_frames=0,
        )
        data = types.SimpleNamespace(qpos=np.zeros(33), qvel=np.zeros(32), time=0.0)
        data.qpos[3] = 1
        renderer = types.SimpleNamespace(close=lambda: None)
        scene_tmp = tempfile.TemporaryDirectory()
        args = types.SimpleNamespace(
            action_source="urdf_target_absolute", row_index=0, max_frames=0,
            width=8, height=8, chunk_stride=1, output_dir=None, fps=10,
            model="openpi/pi05-action-lora-r16-state44-finetune", act_mode="single",
            extended_state=True, state_contract="state44", language_conditioning="gesture",
            max_warm_request_seconds=0, temporal_decay=0.4, frame_window="full",
            contact_context_frames=0, missing_contact_policy="error",
            contact_window_manifest=None, base_url="x", model_path="p",
            client_commit="client", backend_commit="backend", model_commit="model",
            video_mode="none",
        )
        pred = np.zeros((10, 32), np.float32)
        contacts = np.asarray([1, 1, 0, 0, 0], dtype=np.float32)
        surface = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
        radial = np.asarray([0.1, 0.11, 0.12, 0.13, 0.14], dtype=np.float32)

        def set_scene(_model, current, **kwargs):
            current.qpos[0:3] = kwargs["object_pos"]
            current.qpos[3] = 1
            current.qpos[7:33] = kwargs["state"]

        def step(**kwargs):
            current = kwargs["data"]
            current.qpos[7:33] = kwargs["target"]
            current.time += 0.005
            return {
                "hand_object_contact": True, "object_floor_contact": True,
                "hand_floor_contact": False, "max_ncon": 2,
                "max_contact_force": 1.0, "max_abs_actuator_force": 2.0,
            }

        output_tmp = tempfile.TemporaryDirectory()
        out = output_tmp.name
        with mock.patch.object(
            mode4.full, "row_frame_count", return_value=2
        ), mock.patch.object(
            mode4.full, "resolve_row_window", return_value=window
        ), mock.patch.object(
            mode4.full, "set_scene_state", side_effect=set_scene
        ), mock.patch.object(
            mode4.physics, "make_scene",
            return_value=(scene_tmp, object(), data, renderer, 0, 0, list(range(7, 33)), list(range(6, 32)), object()),
        ), mock.patch.object(
            mode4.physics, "resolve_keypoint_geom_ids", return_value=(object(), object(), object())
        ), mock.patch.object(
            mode4.physics, "resolve_state44_feature_ids", return_value=((1, 2, 3, 4, 5), (6,), 7)
        ), mock.patch.object(
            mode4.physics, "finger_contacts_from_mujoco", return_value=contacts
        ), mock.patch.object(
            mode4.physics, "state44_geometry_from_mujoco", return_value=(surface, radial, np.float32(1.0))
        ), mock.patch.object(
            mode4.physics, "render_current_state", return_value=(frame, frame)
        ), mock.patch.object(
            mode4.physics, "nearest_wrapped_position_target",
            side_effect=lambda current, delta, limits: (current + delta, None),
        ), mock.patch.object(
            mode4.physics, "step_servo", side_effect=step
        ), mock.patch.object(
            mode4, "new_clipping_diagnostics", return_value={}
        ), mock.patch.object(
            mode4, "record_clipping"
        ), mock.patch.object(
            mode4, "build_datum", return_value={"observation": {}, "data_config": object()}
        ), mock.patch.object(
            mode4, "query_action", return_value=(pred, pred, pred, {"wall_seconds": 0.01, "used_data_sharding": False})
        ), mock.patch.object(
            mode4, "acquire_action_session", return_value=("s", False)
        ), mock.patch.object(mode4.mujoco, "mj_forward"):
            result = mode4.run_variant(
                args=args, row=row, data_config=object(), mode="mode4", headers={},
                object_name="cube1", session_id="s", output_dir=Path(out),
            )
        output = Path(out) / "mode4"
        observed = np.load(output / "rollout_observation_state.npy")
        self.assertEqual(observed.shape, (1, 44))
        np.testing.assert_array_equal(observed[0, 26:31], contacts)
        np.testing.assert_array_equal(observed[0, 32:37], surface)
        np.testing.assert_array_equal(observed[0, 37:42], 0.0)
        self.assertEqual(observed[0, 42], 1.0)
        self.assertEqual(observed[0, 43], 0.0)
        self.assertEqual(result["state_dim"], 44)
        self.assertEqual(result["action_dim"], 32)
        self.assertEqual(result["state_contract"], "mano_five_finger_contact_geom_rate_v2")
        self.assertIn("rollout_observation_surface_distance", result["arrays"])
        self.assertIn("rollout_observation_radial_rate", result["arrays"])
        self.assertIn("rollout_observation_floor_support", result["arrays"])
        self.assertIn("rollout_observation_multicontact_persistence", result["arrays"])
        output_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
