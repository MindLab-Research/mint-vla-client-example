from __future__ import annotations

import numpy as np
import pytest

from scripts.mano_forced_grasp_retry import (
    FINGER_QPOS_SLICE,
    ForcedGraspRetryController,
    ForcedRetryConfig,
    ForcedRetryStage,
)
from scripts.mano_task_phase import TaskPhase


def _observe(
    controller: ForcedGraspRetryController,
    frame: int,
    *,
    lift: float = 0.0,
    hand: bool = False,
    floor: bool = True,
    stable: bool = False,
    phase: TaskPhase = TaskPhase.ACQUIRE,
) -> None:
    controller.observe(
        control_frame=frame,
        object_lift_m=lift,
        hand_object_contact=hand,
        object_floor_contact=floor,
        stable_lift_achieved=stable,
        task_phase=phase,
    )


def test_config_requires_pre_latch_trigger_and_positive_counters() -> None:
    with pytest.raises(ValueError, match="below"):
        ForcedRetryConfig(trigger_lift_m=0.05)
    with pytest.raises(ValueError, match="positive integer"):
        ForcedRetryConfig(no_hand_contact_frames=0)


def test_trigger_overrides_only_fingers_and_preserves_policy_root() -> None:
    controller = ForcedGraspRetryController()
    _observe(controller, 0)
    _observe(controller, 1, hand=True, floor=True)
    _observe(controller, 2, lift=0.03, hand=True, floor=False)
    assert controller.stage == ForcedRetryStage.FORCED_RELEASE
    policy = np.linspace(-0.2, 0.2, 28, dtype=np.float32)
    open_pose = np.linspace(0.4, -0.4, 28, dtype=np.float32)
    actual, correction = controller.apply_action_override(policy, open_pose)
    np.testing.assert_array_equal(actual[:6], policy[:6])
    np.testing.assert_array_equal(actual[FINGER_QPOS_SLICE], open_pose[FINGER_QPOS_SLICE])
    np.testing.assert_array_equal(correction, actual - policy)
    controller.record_override_action()
    assert controller.forced_release_action_frames == 1


def test_clean_forced_failure_then_second_grasp_reaches_retry_success() -> None:
    controller = ForcedGraspRetryController(
        ForcedRetryConfig(
            trigger_lift_m=0.03,
            floor_contact_frames=2,
            no_hand_contact_frames=2,
            max_forced_release_frames=5,
        )
    )
    _observe(controller, 0)
    _observe(controller, 1, hand=True)
    _observe(controller, 2, lift=0.031, hand=True, floor=False)
    controller.record_override_action()
    _observe(controller, 3, lift=0.01, hand=False, floor=True)
    controller.record_override_action()
    _observe(controller, 4, lift=0.0, hand=False, floor=True)
    assert controller.stage == ForcedRetryStage.RETRY_FREE_POLICY
    assert controller.clean_failure_frame == 4
    assert not controller.override_active

    _observe(controller, 5, hand=True)
    _observe(
        controller,
        6,
        lift=0.06,
        hand=True,
        floor=False,
        stable=True,
        phase=TaskPhase.PLACE,
    )
    _observe(
        controller,
        7,
        lift=0.0,
        hand=False,
        floor=True,
        stable=True,
        phase=TaskPhase.DONE,
    )
    ledger = controller.finalize(termination_reason="done", control_frame=7)
    assert ledger["intervention_valid"] is True
    assert ledger["first_contact_frame"] == 1
    assert ledger["trigger_frame"] == 2
    assert ledger["clean_failure_frame"] == 4
    assert ledger["second_contact_frame"] == 5
    assert ledger["retry_stable_lift_frame"] == 6
    assert ledger["retry_place_frame"] == 6
    assert ledger["retry_done_frame"] == 7
    assert ledger["retry_success"] is True


def test_stable_lift_during_forced_release_invalidates_intervention() -> None:
    controller = ForcedGraspRetryController()
    _observe(controller, 0, hand=True)
    _observe(controller, 1, lift=0.03, hand=True, floor=False)
    _observe(
        controller,
        2,
        lift=0.06,
        hand=True,
        floor=False,
        stable=True,
        phase=TaskPhase.PLACE,
    )
    assert controller.intervention_invalid
    assert controller.invalid_reason == "stable_lift_activated_during_forced_release"


def test_override_limit_surfaces_invalid_clean_failure() -> None:
    controller = ForcedGraspRetryController(
        ForcedRetryConfig(max_forced_release_frames=2)
    )
    _observe(controller, 0, hand=True)
    _observe(controller, 1, lift=0.03, hand=True, floor=False)
    controller.record_override_action()
    _observe(controller, 2, lift=0.02, hand=True, floor=False)
    controller.record_override_action()
    _observe(controller, 3, lift=0.01, hand=True, floor=False)
    assert controller.intervention_invalid
    assert controller.invalid_reason == "clean_first_failure_not_achieved_within_override_limit"


def test_finalize_without_trigger_is_explicitly_invalid() -> None:
    controller = ForcedGraspRetryController()
    _observe(controller, 0)
    ledger = controller.finalize(termination_reason="timeout", control_frame=3000)
    assert ledger["intervention_valid"] is False
    assert ledger["invalid_reason"] == "forced_release_trigger_not_reached"
    assert ledger["retry_success"] is False


def test_observation_frames_must_be_strictly_increasing() -> None:
    controller = ForcedGraspRetryController()
    _observe(controller, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        _observe(controller, 0)
