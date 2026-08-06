"""Causal evaluator-only intervention for a forced first-grasp failure.

This module does not modify State45 or model inputs.  It drives a temporary
finger-target override in the evaluator, verifies that the first attempt ended
cleanly while the task remained in ACQUIRE, and then observes whether the
unmodified policy enters a second contact episode and finishes the task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from scripts.mano_state41_contract import HAND_QPOS_DIM
from scripts.mano_task_phase import TaskPhase


FORCED_GRASP_RETRY_CONTRACT_ID = "mano_forced_first_grasp_retry_v1"
FINGER_QPOS_SLICE = slice(6, HAND_QPOS_DIM)


class ForcedRetryStage(str, Enum):
    AWAIT_TRIGGER = "await_trigger"
    FORCED_RELEASE = "forced_release"
    RETRY_FREE_POLICY = "retry_free_policy"
    INTERVENTION_INVALID = "intervention_invalid"


@dataclass(frozen=True)
class ForcedRetryConfig:
    trigger_lift_m: float = 0.03
    floor_contact_frames: int = 10
    no_hand_contact_frames: int = 20
    max_forced_release_frames: int = 150
    active_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.trigger_lift_m) or self.trigger_lift_m <= 0:
            raise ValueError("trigger_lift_m must be finite and positive")
        if self.trigger_lift_m >= 0.05:
            raise ValueError("trigger_lift_m must remain below the 0.05m stable-lift threshold")
        for name, value in {
            "floor_contact_frames": self.floor_contact_frames,
            "no_hand_contact_frames": self.no_hand_contact_frames,
            "max_forced_release_frames": self.max_forced_release_frames,
        }.items():
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.active_threshold):
            raise ValueError("active_threshold must be finite")


class ForcedGraspRetryController:
    """Track and apply one forced pre-latch release, then observe free retry."""

    def __init__(self, config: ForcedRetryConfig | None = None) -> None:
        self.config = config or ForcedRetryConfig()
        self.stage = ForcedRetryStage.AWAIT_TRIGGER
        self.first_contact_frame: int | None = None
        self.trigger_frame: int | None = None
        self.clean_failure_frame: int | None = None
        self.second_contact_frame: int | None = None
        self.retry_stable_lift_frame: int | None = None
        self.retry_place_frame: int | None = None
        self.retry_done_frame: int | None = None
        self.invalid_frame: int | None = None
        self.invalid_reason: str | None = None
        self.first_attempt_max_lift_m = 0.0
        self.forced_release_action_frames = 0
        self._floor_run = 0
        self._no_contact_run = 0
        self._last_observation_frame: int | None = None
        self._intervention_valid = False
        self._finalized = False
        self._termination_reason: str | None = None

    @staticmethod
    def _active(value: bool | float, threshold: float) -> bool:
        return float(value) > threshold

    def _invalidate(self, frame: int, reason: str) -> None:
        self.stage = ForcedRetryStage.INTERVENTION_INVALID
        self.invalid_frame = int(frame)
        self.invalid_reason = str(reason)
        self._intervention_valid = False

    def observe(
        self,
        *,
        control_frame: int,
        object_lift_m: float,
        hand_object_contact: bool | float,
        object_floor_contact: bool | float,
        stable_lift_achieved: bool | float,
        task_phase: TaskPhase | int | float,
    ) -> None:
        """Consume one live causal observation before the next control action."""
        frame = int(control_frame)
        if frame < 0 or frame != control_frame:
            raise ValueError("control_frame must be a non-negative integer")
        if self._last_observation_frame is not None and frame <= self._last_observation_frame:
            raise ValueError("forced-retry observations must have strictly increasing frames")
        self._last_observation_frame = frame
        lift = float(object_lift_m)
        if not np.isfinite(lift):
            raise ValueError("object_lift_m must be finite")
        hand_contact = self._active(hand_object_contact, self.config.active_threshold)
        floor_contact = self._active(object_floor_contact, self.config.active_threshold)
        stable = self._active(stable_lift_achieved, self.config.active_threshold)
        phase = TaskPhase(int(task_phase))

        if self.first_contact_frame is None and hand_contact:
            self.first_contact_frame = frame

        if self.stage == ForcedRetryStage.AWAIT_TRIGGER:
            self.first_attempt_max_lift_m = max(self.first_attempt_max_lift_m, lift)
            if phase != TaskPhase.ACQUIRE or stable:
                self._invalidate(frame, "stable_lift_or_place_before_forced_release_trigger")
                return
            if lift >= self.config.trigger_lift_m:
                if self.first_contact_frame is None:
                    self._invalidate(frame, "lift_trigger_without_prior_hand_object_contact")
                    return
                self.stage = ForcedRetryStage.FORCED_RELEASE
                self.trigger_frame = frame
                self._floor_run = int(floor_contact)
                self._no_contact_run = int(not hand_contact)
            return

        if self.stage == ForcedRetryStage.FORCED_RELEASE:
            self.first_attempt_max_lift_m = max(self.first_attempt_max_lift_m, lift)
            if phase != TaskPhase.ACQUIRE or stable:
                self._invalidate(frame, "stable_lift_activated_during_forced_release")
                return
            self._floor_run = self._floor_run + 1 if floor_contact else 0
            self._no_contact_run = self._no_contact_run + 1 if not hand_contact else 0
            if (
                self._floor_run >= self.config.floor_contact_frames
                and self._no_contact_run >= self.config.no_hand_contact_frames
            ):
                self.stage = ForcedRetryStage.RETRY_FREE_POLICY
                self.clean_failure_frame = frame
                self._intervention_valid = True
                return
            if self.forced_release_action_frames >= self.config.max_forced_release_frames:
                self._invalidate(frame, "clean_first_failure_not_achieved_within_override_limit")
            return

        if self.stage == ForcedRetryStage.RETRY_FREE_POLICY:
            if self.second_contact_frame is None and hand_contact:
                self.second_contact_frame = frame
            if self.retry_stable_lift_frame is None and stable:
                self.retry_stable_lift_frame = frame
            if self.retry_place_frame is None and phase >= TaskPhase.PLACE:
                self.retry_place_frame = frame
            if self.retry_done_frame is None and phase == TaskPhase.DONE:
                self.retry_done_frame = frame
            return

    @property
    def override_active(self) -> bool:
        return self.stage == ForcedRetryStage.FORCED_RELEASE

    @property
    def intervention_invalid(self) -> bool:
        return self.stage == ForcedRetryStage.INTERVENTION_INVALID

    @property
    def intervention_valid(self) -> bool:
        return self._intervention_valid

    def apply_action_override(
        self, policy_absolute_target: np.ndarray, open_hand_qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return actual absolute target and actual-minus-policy correction."""
        policy = np.asarray(policy_absolute_target, dtype=np.float32)
        open_pose = np.asarray(open_hand_qpos, dtype=np.float32)
        if policy.shape != (HAND_QPOS_DIM,) or open_pose.shape != (HAND_QPOS_DIM,):
            raise ValueError(
                f"forced-retry target shapes must be ({HAND_QPOS_DIM},), got "
                f"{policy.shape}/{open_pose.shape}"
            )
        if not np.isfinite(policy).all() or not np.isfinite(open_pose).all():
            raise FloatingPointError("forced-retry action inputs must be finite")
        actual = policy.copy()
        if self.override_active:
            actual[FINGER_QPOS_SLICE] = open_pose[FINGER_QPOS_SLICE]
        correction = actual - policy
        if not np.array_equal(actual[:6], policy[:6]):
            raise RuntimeError("forced release modified policy root/wrist target")
        return actual, correction

    def record_override_action(self) -> None:
        if not self.override_active:
            raise RuntimeError("cannot record forced-release action outside override stage")
        self.forced_release_action_frames += 1

    def finalize(self, *, termination_reason: str, control_frame: int) -> dict[str, Any]:
        """Freeze and return a machine-readable intervention/retry ledger."""
        if self._finalized:
            raise RuntimeError("forced-retry controller was already finalized")
        self._finalized = True
        self._termination_reason = str(termination_reason)
        frame = int(control_frame)
        if self.stage == ForcedRetryStage.AWAIT_TRIGGER:
            self._invalidate(frame, "forced_release_trigger_not_reached")
        elif self.stage == ForcedRetryStage.FORCED_RELEASE:
            self._invalidate(frame, "rollout_ended_before_clean_first_failure")
        retry_success = bool(
            self._intervention_valid
            and self.second_contact_frame is not None
            and self.retry_stable_lift_frame is not None
            and self.retry_place_frame is not None
            and self.retry_done_frame is not None
            and termination_reason == "done"
        )
        return {
            "contract": FORCED_GRASP_RETRY_CONTRACT_ID,
            "config": asdict(self.config),
            "stage": self.stage.value,
            "intervention_valid": self._intervention_valid,
            "invalid_frame": self.invalid_frame,
            "invalid_reason": self.invalid_reason,
            "first_contact_frame": self.first_contact_frame,
            "trigger_frame": self.trigger_frame,
            "first_attempt_max_lift_m": float(self.first_attempt_max_lift_m),
            "forced_release_action_frames": self.forced_release_action_frames,
            "clean_failure_frame": self.clean_failure_frame,
            "second_contact_frame": self.second_contact_frame,
            "retry_stable_lift_frame": self.retry_stable_lift_frame,
            "retry_place_frame": self.retry_place_frame,
            "retry_done_frame": self.retry_done_frame,
            "termination_reason": self._termination_reason,
            "retry_success": retry_success,
        }
