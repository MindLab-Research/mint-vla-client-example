from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts.eval import replay_mano_target_physics as replay


def test_grade_boundaries_are_contractual():
    assert replay.grade_from_max_error(0.0) == "A"
    assert replay.grade_from_max_error(0.029999) == "A"
    assert replay.grade_from_max_error(0.03) == "B"
    assert replay.grade_from_max_error(0.079999) == "B"
    assert replay.grade_from_max_error(0.08) == "C"


def test_resume_valid_uses_current_trace_keys(tmp_path):
    frames = 3
    run_id = {"contract": "resume-test"}
    trace = tmp_path / "row.npz"
    report = tmp_path / "row.json"
    np.savez_compressed(
        trace,
        object_position_error=np.zeros(frames, dtype=np.float32),
        simulated_full_qpos=np.zeros((frames, 35), dtype=np.float32),
        simulated_hand_qpos=np.zeros((frames, 28), dtype=np.float32),
        source_target_qpos=np.zeros((frames, 28), dtype=np.float32),
    )
    replay.atomic_json(report, {
        "status": "ok", "row_index": 4, "frames": frames,
        "provenance": run_id, "trace_sha256": replay.sha256(trace),
    })

    assert replay.resume_valid(report, trace, 4, run_id) is True

    with np.load(trace) as values:
        old = {name: values[name] for name in values.files if name != "object_position_error"}
    np.savez_compressed(trace, position_error_m=np.zeros(frames), **old)
    replay.atomic_json(report, {
        "status": "ok", "row_index": 4, "frames": frames,
        "provenance": run_id, "trace_sha256": replay.sha256(trace),
    })
    assert replay.resume_valid(report, trace, 4, run_id) is False


def test_parse_rows_uses_end_exclusive_ranges_and_deduplicates():
    assert replay.parse_rows("2,4:7,5", 10) == [2, 4, 5, 6]
    with pytest.raises(ValueError, match="outside"):
        replay.parse_rows("10", 10)
    with pytest.raises(ValueError, match="invalid"):
        replay.parse_rows("7:7", 10)


def test_population_validation_rejects_missing_rows_and_uuid_aliases():
    entries = [
        {"uuid": "u0", "original_merged_row_index": 10},
        {"uuid": "u1", "original_merged_row_index": 20},
    ]
    valid = [
        {"row_index": 0, "original_merged_row_index": 10, "status": "ok", "row_uuid": "u0", "object": "cube"},
        {"row_index": 1, "original_merged_row_index": 20, "status": "ok", "row_uuid": "u1", "object": "cube"},
    ]
    replay.validate_record_population(valid, [0, 1], entries, "cube")
    with pytest.raises(ValueError, match="population mismatch"):
        replay.validate_record_population(valid[:1], [0, 1], entries, "cube")
    aliased = [dict(valid[0]), {**valid[1], "row_uuid": "u0"}]
    with pytest.raises(ValueError, match="UUID"):
        replay.validate_record_population(aliased, [0, 1], entries, "cube")


def _row(frames=4):
    recorded = np.zeros((frames, 28), dtype=np.float64)
    targets = np.arange(frames * 28, dtype=np.float64).reshape(frames, 28) / 1000.0
    selected_position = np.tile([0.1, 0.2, 0.3], (frames, 1))
    return {
        "index": {
            "uuid": "generated",
            "seed_uuid": "seed",
            "scene": "cube1",
            "is_generated": True,
        },
        "trajectory_metadata": {
            "total_frames": frames,
            "data_fps": 200,
            "gesture": "01",
            "hand_names": ["right"],
            "hand_slots": ["left", "right"],
            "object_names": ["decoy", "cube1"],
        },
        "timestamp": (np.arange(frames) * 0.005).tolist(),
        "hands": [
            {"hand_name": None, "urdf_dof": [], "urdf_dof_target": []},
            {
                "hand_name": "right",
                "urdf_dof": recorded.tolist(),
                "urdf_dof_target": targets.tolist(),
            },
        ],
        "objects": [
            {
                "pos": np.tile([9.0, 9.0, 9.0], (frames, 1)).tolist(),
                "rot_aa": np.zeros((frames, 3)).tolist(),
            },
            {
                "pos": selected_position.tolist(),
                "rot_aa": np.zeros((frames, 3)).tolist(),
            },
        ],
        "provenance": {
            "contract": replay.SOURCE_CONTRACT,
            "source_identity": "cube1_01_1",
        },
    }


def _identity():
    return {
        "row_index": 7,
        "uuid": "generated",
        "seed_uuid": "seed",
        "object_type": "cube1",
        "gesture": "01",
        "source_identity": "cube1_01_1",
        "original_merged_row_index": 2239,
    }


def test_validate_row_resolves_right_hand_and_selected_object_from_metadata():
    frames, qpos, targets, position, _, timestamps, object_name = replay.validate_row(
        _row(), 7, _identity()
    )

    assert frames == 4 and object_name == "cube1"
    assert qpos.shape == targets.shape == (4, 28)
    np.testing.assert_allclose(position, np.tile([0.1, 0.2, 0.3], (4, 1)))
    np.testing.assert_allclose(timestamps, np.arange(4) * 0.005)


def test_replay_preserves_old_loop_but_executes_only_raw_targets_with_successors(
    monkeypatch,
):
    row = _row()
    model = SimpleNamespace(
        nq=35,
        nv=34,
        nu=28,
        opt=SimpleNamespace(timestep=0.0025),
        actuator_ctrlrange=np.tile([-10.0, 10.0], (28, 1)),
    )

    class Data:
        def __init__(self, _model):
            self.qpos = np.zeros(35, dtype=np.float64)
            self.qvel = np.zeros(34, dtype=np.float64)
            self.xpos = np.zeros((2, 3), dtype=np.float64)
            self.xquat = np.zeros((2, 4), dtype=np.float64)
            self.xquat[:, 0] = 1.0
            self.time = 0.0

    fake_mujoco = SimpleNamespace(
        MjData=Data,
        mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mj_name2id=lambda *_args: 1,
        mj_forward=lambda _model, data: (
            data.xpos.__setitem__(1, data.qpos[28:31]),
            data.xquat.__setitem__(1, data.qpos[31:35]),
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(
        replay,
        "scene",
        lambda _object: (
            None,
            model,
            None,
            None,
            28,
            28,
            np.arange(28),
            np.arange(28),
            model.actuator_ctrlrange,
        ),
    )
    executed = []

    def step_servo(*, model, data, target, substeps, object_name):
        executed.append(np.asarray(target).copy())
        data.time += substeps * model.opt.timestep
        data.qpos[:28] = target
        data.xpos[1] = data.qpos[28:31]
        data.xquat[1] = data.qpos[31:35]
        return {
            "warnings": [],
            "hand_object_contact": False,
            "object_floor_contact": False,
            "hand_floor_contact": False,
            "max_ncon": 0,
            "max_contact_force": 0.0,
            "max_abs_actuator_force": 0.0,
        }

    monkeypatch.setattr(replay.physics, "object_body_id", lambda _model, _object: 1)
    monkeypatch.setattr(replay.physics, "step_servo", step_servo)
    result, trace = replay.replay(7, row, _identity(), {"contract": "test"})
    source_targets = np.asarray(row["hands"][1]["urdf_dof_target"])

    assert result["original_merged_row_index"] == 2239
    assert result["metrics"]["timing"]["mj_steps"] == 6
    assert result["metrics"]["timing"]["final_source_target_executed"] is False
    np.testing.assert_allclose(executed, source_targets[:-1])
    np.testing.assert_allclose(trace["target_qpos"], source_targets[:-1])
    np.testing.assert_allclose(trace["source_target_qpos"], source_targets)
    np.testing.assert_array_equal(trace["target_index"], [0, 1, 2])


def test_compact_index_binds_filtered_rows_to_accepted_lineage(monkeypatch):
    class Batch:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return self._rows

    class Scanner:
        def __init__(self, rows):
            self._rows = rows

        def to_batches(self):
            return [Batch(self._rows)]

    class Dataset:
        def scanner(self, *, columns, batch_size):
            assert columns == ["index", "provenance"]
            assert batch_size == 16
            return Scanner([
                {
                    "index": {"uuid": "u0", "seed_uuid": "s0", "scene": "cube1", "is_generated": True},
                    "provenance": {"contract": replay.SOURCE_CONTRACT, "source_identity": "cube1_01_7"},
                },
                {
                    "index": {"uuid": "u1", "seed_uuid": "s1", "scene": "banana", "is_generated": True},
                    "provenance": {"contract": replay.SOURCE_CONTRACT, "source_identity": "banana_02_9"},
                },
            ])

    accepted = [
        {"row_index": 7, "uuid": "u0", "object": "cube1", "pair": "cube1_01", "source_identity": "cube1_01_7", "frames": 4},
        {"row_index": 99, "uuid": "u1", "object": "banana", "pair": "banana_02", "source_identity": "banana_02_9", "frames": 5},
    ]
    monkeypatch.setattr(replay, "EXPECTED_ROWS", 2)

    entries = replay.build_source_index(Dataset(), accepted, batch_size=16)

    assert [entry["row_index"] for entry in entries] == [0, 1]
    assert [entry["original_merged_row_index"] for entry in entries] == [7, 99]
    assert [entry["gesture"] for entry in entries] == ["01", "02"]
    changed = [dict(accepted[0]), {**accepted[1], "uuid": "wrong"}]
    with pytest.raises(ValueError, match="filtered/accepted identity mismatch"):
        replay.build_source_index(Dataset(), changed, batch_size=16)
