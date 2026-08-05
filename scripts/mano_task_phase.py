"""Causal acquire/lift/place task-phase state shared by training and inference.

The tracker consumes only the current observation and its own past state.  It
never inspects future frames, so scanning a recorded trajectory offline is
identical to updating the tracker online during a rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


PHASE_TRACKER_CONTRACT_ID = "mano_pick_place_phase_tracker_v1"


class TaskPhase(IntEnum):
    ACQUIRE = 0
    PLACE = 1
    DONE = 2


@dataclass(frozen=True)
class PhaseTrackerConfig:
    sample_dt_seconds: float = 0.005
    stable_lift_threshold_m: float = 0.05
    stable_lift_frames: int = 100
    vertical_velocity_alpha: float = 0.2
    settle_velocity_threshold_mps: float = 0.02
    settle_frames: int = 50
    release_frames: int = 20
    active_threshold: float = 0.5

    def __post_init__(self) -> None:
        finite_positive = {
            "sample_dt_seconds": self.sample_dt_seconds,
            "stable_lift_threshold_m": self.stable_lift_threshold_m,
            "settle_velocity_threshold_mps": self.settle_velocity_threshold_mps,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        for name, value in {
            "stable_lift_frames": self.stable_lift_frames,
            "settle_frames": self.settle_frames,
            "release_frames": self.release_frames,
        }.items():
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not np.isfinite(self.vertical_velocity_alpha) or not 0 < self.vertical_velocity_alpha <= 1:
            raise ValueError(
                "vertical_velocity_alpha must be finite and in (0,1], got "
                f"{self.vertical_velocity_alpha!r}"
            )
        if not np.isfinite(self.active_threshold):
            raise ValueError("active_threshold must be finite")


@dataclass(frozen=True)
class PhaseObservation:
    peak_lift_so_far: np.float32
    stable_lift_achieved: np.float32
    object_vertical_velocity: np.float32
    task_phase: np.float32

    def as_array(self) -> np.ndarray:
        result = np.asarray(
            [
                self.peak_lift_so_far,
                self.stable_lift_achieved,
                self.object_vertical_velocity,
                self.task_phase,
            ],
            dtype=np.float32,
        )
        if result.shape != (4,) or not np.isfinite(result).all():
            raise FloatingPointError(f"invalid phase observation {result!r}")
        return result


class ManoTaskPhaseTracker:
    """Deterministic causal state machine for one pick-then-place episode."""

    def __init__(self, config: PhaseTrackerConfig | None = None) -> None:
        self.config = config or PhaseTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self._frame_count = 0
        self._previous_lift = np.float32(0.0)
        self._peak_lift = np.float32(0.0)
        self._filtered_vz = np.float32(0.0)
        self._stable_lift_run_frames = 0
        self._stable_lift_achieved = False
        self._left_floor_after_lift = False
        self._settle_run_frames = 0
        self._release_run_frames = 0
        self._phase = TaskPhase.ACQUIRE

    @property
    def phase(self) -> TaskPhase:
        return self._phase

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update(
        self,
        *,
        object_lift_m: float,
        hand_object_contact: bool | float,
        object_floor_contact: bool | float,
    ) -> PhaseObservation:
        lift = float(object_lift_m)
        if not np.isfinite(lift):
            raise ValueError(f"object_lift_m must be finite, got {object_lift_m!r}")
        hand_contact = float(hand_object_contact) > self.config.active_threshold
        floor_contact = float(object_floor_contact) > self.config.active_threshold

        self._peak_lift = np.float32(max(float(self._peak_lift), lift))
        if self._frame_count == 0:
            filtered_vz = 0.0
        else:
            raw_vz = (lift - float(self._previous_lift)) / self.config.sample_dt_seconds
            alpha = self.config.vertical_velocity_alpha
            filtered_vz = (
                (1.0 - alpha) * float(self._filtered_vz) + alpha * raw_vz
            )
        if not np.isfinite(filtered_vz):
            raise FloatingPointError("filtered object vertical velocity is non-finite")
        self._filtered_vz = np.float32(filtered_vz)
        self._previous_lift = np.float32(lift)

        if not self._stable_lift_achieved:
            if lift > self.config.stable_lift_threshold_m:
                self._stable_lift_run_frames += 1
            else:
                self._stable_lift_run_frames = 0
            if self._stable_lift_run_frames >= self.config.stable_lift_frames:
                self._stable_lift_achieved = True
                self._phase = TaskPhase.PLACE

        if self._stable_lift_achieved and self._phase != TaskPhase.DONE:
            # Initial floor contact cannot complete the task: the object must
            # first leave the floor after the stable-lift latch has fired.
            if not floor_contact:
                self._left_floor_after_lift = True
                self._settle_run_frames = 0
                self._release_run_frames = 0
            elif self._left_floor_after_lift:
                if abs(float(self._filtered_vz)) < self.config.settle_velocity_threshold_mps:
                    self._settle_run_frames += 1
                else:
                    self._settle_run_frames = 0
                if hand_contact:
                    self._release_run_frames = 0
                else:
                    self._release_run_frames += 1
                if (
                    self._settle_run_frames >= self.config.settle_frames
                    and self._release_run_frames >= self.config.release_frames
                ):
                    self._phase = TaskPhase.DONE

        if self._phase == TaskPhase.ACQUIRE and self._stable_lift_achieved:
            raise RuntimeError("phase invariant violated: stable lift in ACQUIRE")
        if self._phase != TaskPhase.ACQUIRE and not self._stable_lift_achieved:
            raise RuntimeError("phase invariant violated: PLACE/DONE without stable lift")

        self._frame_count += 1
        return PhaseObservation(
            peak_lift_so_far=np.float32(self._peak_lift),
            stable_lift_achieved=np.float32(self._stable_lift_achieved),
            object_vertical_velocity=np.float32(self._filtered_vz),
            task_phase=np.float32(int(self._phase)),
        )


def phase_feature_sequence(
    *,
    object_lift_m: Iterable[float] | np.ndarray,
    hand_object_contact: Iterable[bool] | np.ndarray,
    object_floor_contact: Iterable[bool] | np.ndarray,
    config: PhaseTrackerConfig | None = None,
) -> np.ndarray:
    """Scan a recorded trajectory with exactly the online tracker semantics."""
    lift = np.asarray(object_lift_m, dtype=np.float64)
    hand = np.asarray(hand_object_contact)
    floor = np.asarray(object_floor_contact)
    if lift.ndim != 1:
        raise ValueError(f"object_lift_m must have shape [T], got {lift.shape}")
    if hand.shape != lift.shape or floor.shape != lift.shape:
        raise ValueError(
            "phase feature inputs must have equal [T] shapes, got "
            f"lift={lift.shape}, hand={hand.shape}, floor={floor.shape}"
        )
    if not np.isfinite(lift).all():
        raise ValueError("object_lift_m contains non-finite values")
    tracker = ManoTaskPhaseTracker(config)
    result = np.empty((len(lift), 4), dtype=np.float32)
    for frame in range(len(lift)):
        result[frame] = tracker.update(
            object_lift_m=float(lift[frame]),
            hand_object_contact=bool(hand[frame]),
            object_floor_contact=bool(floor[frame]),
        ).as_array()
    if not np.isfinite(result).all():
        raise FloatingPointError("phase feature sequence contains non-finite values")
    return result
