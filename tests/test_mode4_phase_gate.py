from __future__ import annotations

import numpy as np
import pytest

from scripts.eval.mode4_phase_gate import (
    GraspGatePhase,
    GraspProbeGateConfig,
    GraspProbePhaseGate,
)


def inputs(*, root_z: float = 0.1, finger_target: float = 0.7):
    current = np.zeros(26, dtype=np.float64)
    current[2] = root_z
    proposed = current.copy()
    proposed[:6] += np.asarray([0.01, -0.02, 0.03, 0.1, -0.1, 0.2])
    proposed[6:] = finger_target
    contacts = np.asarray([1, 1, 1, 1, 0], dtype=np.float64)
    obj = np.asarray([0.2, 0.0, 0.04], dtype=np.float64)
    return current, proposed, contacts, obj


def small_config() -> GraspProbeGateConfig:
    return GraspProbeGateConfig(
        min_contact_count=4,
        contact_persistence_frames=2,
        probe_lift_m=0.005,
        probe_frames=2,
        probe_follow_min_m=0.003,
        probe_min_contact_count=3,
        retention_lift_m=0.050,
        retention_frames=2,
        retention_follow_min_m=0.020,
        retention_min_contact_count=3,
    )


def run_step(gate, frame, current, proposed, contacts, obj, *, floor=True):
    return gate.step(
        frame_index=frame,
        current_q=current,
        proposed_servo_target=proposed,
        finger_contacts=contacts,
        object_floor_contact=floor,
        object_position=obj,
    )


def test_waiting_is_exact_passthrough() -> None:
    gate = GraspProbePhaseGate(small_config())
    current, proposed, contacts, obj = inputs()
    result = run_step(gate, 0, current, proposed, contacts, obj)
    assert result.phase is GraspGatePhase.WAITING
    assert result.override is False
    np.testing.assert_array_equal(result.target, proposed)


def test_probe_latches_achieved_root_and_policy_finger_target() -> None:
    gate = GraspProbePhaseGate(small_config())
    current, proposed, contacts, obj = inputs()
    run_step(gate, 0, current, proposed, contacts, obj, floor=False)
    result = run_step(gate, 1, current, proposed, contacts, obj, floor=False)
    assert result.phase is GraspGatePhase.PROBE
    assert result.override is True
    np.testing.assert_allclose(result.target[:2], current[:2])
    np.testing.assert_allclose(result.target[3:6], current[3:6])
    assert result.target[2] == pytest.approx(current[2] + 0.0025)
    np.testing.assert_allclose(result.target[6:], proposed[6:])
    assert gate.summary()["events"][0]["event"] == "contact_candidate"


def test_physical_probe_and_retention_pass() -> None:
    gate = GraspProbePhaseGate(small_config())
    current, proposed, contacts, obj = inputs()
    run_step(gate, 0, current, proposed, contacts, obj, floor=False)
    run_step(gate, 1, current, proposed, contacts, obj, floor=False)
    run_step(gate, 2, current, proposed, contacts, obj, floor=False)

    probe_observation = obj.copy(); probe_observation[2] += 0.004
    result = run_step(
        gate, 3, current, proposed, contacts, probe_observation, floor=False
    )
    assert result.phase is GraspGatePhase.RETENTION
    assert result.override is True
    assert result.target[2] == pytest.approx(current[2] + 0.030)

    run_step(gate, 4, current, proposed, contacts, probe_observation, floor=False)
    retained = obj.copy(); retained[2] += 0.030
    result = run_step(gate, 5, current, proposed, contacts, retained, floor=False)
    assert result.phase is GraspGatePhase.PASSED
    assert result.override is True
    assert result.target[2] == pytest.approx(current[2] + 0.055)
    assert [event["event"] for event in gate.summary()["events"]] == [
        "contact_candidate", "probe_passed", "retention_passed"
    ]


def test_probe_failure_returns_policy_control() -> None:
    gate = GraspProbePhaseGate(small_config())
    current, proposed, contacts, obj = inputs()
    run_step(gate, 0, current, proposed, contacts, obj, floor=False)
    run_step(gate, 1, current, proposed, contacts, obj, floor=False)
    run_step(gate, 2, current, proposed, contacts, obj, floor=False)
    result = run_step(gate, 3, current, proposed, contacts, obj, floor=True)
    assert result.phase is GraspGatePhase.FAILED
    assert result.override is False
    np.testing.assert_array_equal(result.target, proposed)
    assert gate.summary()["events"][-1]["event"] == "probe_failed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_contact_count": 0},
        {"contact_persistence_frames": 0},
        {"probe_lift_m": 0},
        {"retention_follow_min_m": float("nan")},
    ],
)
def test_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        GraspProbeGateConfig(**kwargs)
