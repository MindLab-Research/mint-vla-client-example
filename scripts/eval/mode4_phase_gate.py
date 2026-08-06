"""Event-driven grasp probe gate for diagnostic Mode4 rollouts.

The gate is deliberately external to the policy.  It leaves approach/closure
commands untouched until persistent multi-finger contact is observed while
object-floor support is clear, then holds the policy's contemporaneous finger target while applying
a small root-Z probe followed by a larger retention lift.  Physical object
motion, floor support, and persistent contacts decide whether each phase passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


HAND_DIM = 26
ROOT_DIM = 6


class GraspGatePhase(IntEnum):
    WAITING = 0
    PROBE = 1
    RETENTION = 2
    PASSED = 3
    FAILED = 4


@dataclass(frozen=True)
class GraspProbeGateConfig:
    min_contact_count: int = 4
    contact_persistence_frames: int = 20
    probe_lift_m: float = 0.005
    probe_frames: int = 20
    probe_follow_min_m: float = 0.003
    probe_min_contact_count: int = 3
    retention_lift_m: float = 0.050
    retention_frames: int = 100
    retention_follow_min_m: float = 0.020
    retention_min_contact_count: int = 3
    require_floor_clear: bool = True

    def __post_init__(self) -> None:
        for name in ("min_contact_count", "probe_min_contact_count", "retention_min_contact_count"):
            value = int(getattr(self, name))
            if not 1 <= value <= 5:
                raise ValueError(f"{name} must be in [1,5], got {value}")
        for name in ("contact_persistence_frames", "probe_frames", "retention_frames"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in (
            "probe_lift_m",
            "probe_follow_min_m",
            "retention_lift_m",
            "retention_follow_min_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value}")


@dataclass(frozen=True)
class GraspGateStep:
    target: np.ndarray
    phase: GraspGatePhase
    override: bool
    contact_count: int


class GraspProbePhaseGate:
    """Per-row finite-state diagnostic gate.

    At trigger time root X/Y/orientation are latched at achieved qpos while all
    finger targets are latched at the policy's already-proposed servo target.
    Thus the intervention changes duration and root-Z motion, not the achieved
    closure target that produced the contact candidate.
    """

    def __init__(self, config: GraspProbeGateConfig) -> None:
        self.config = config
        self.phase = GraspGatePhase.WAITING
        self.contact_run = 0
        self.phase_step = 0
        self.trigger_frame: int | None = None
        self.trigger_object_position: np.ndarray | None = None
        self.latched_target: np.ndarray | None = None
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite shape ({size},), got {result.shape}")
        return result

    def _event(self, frame_index: int, event: str, **details: Any) -> None:
        self.events.append({"frame_index": int(frame_index), "event": event, **details})

    def _object_delta_z(self, object_position: np.ndarray) -> float:
        if self.trigger_object_position is None:
            raise RuntimeError("phase gate has no trigger object position")
        return float(object_position[2] - self.trigger_object_position[2])

    def _physical_pass(
        self,
        *,
        object_position: np.ndarray,
        object_floor_contact: bool,
        contact_count: int,
        follow_min_m: float,
        min_contact_count: int,
    ) -> tuple[bool, dict[str, Any]]:
        delta_z = self._object_delta_z(object_position)
        floor_clear = not bool(object_floor_contact)
        passed = (
            delta_z >= float(follow_min_m)
            and contact_count >= int(min_contact_count)
            and (floor_clear or not self.config.require_floor_clear)
        )
        return passed, {
            "object_delta_z_m": delta_z,
            "contact_count": int(contact_count),
            "object_floor_contact": bool(object_floor_contact),
            "follow_min_m": float(follow_min_m),
            "min_contact_count": int(min_contact_count),
        }

    def _latched_lift_target(self, lift_m: float) -> np.ndarray:
        if self.latched_target is None:
            raise RuntimeError("phase gate has no latched target")
        target = self.latched_target.copy()
        target[2] += float(lift_m)
        return target

    def step(
        self,
        *,
        frame_index: int,
        current_q: np.ndarray,
        proposed_servo_target: np.ndarray,
        finger_contacts: np.ndarray,
        object_floor_contact: bool,
        object_position: np.ndarray,
    ) -> GraspGateStep:
        current = self._vector("current_q", current_q, HAND_DIM)
        proposed = self._vector("proposed_servo_target", proposed_servo_target, HAND_DIM)
        contacts = self._vector("finger_contacts", finger_contacts, 5)
        obj = self._vector("object_position", object_position, 3)
        contact_count = int(np.count_nonzero(contacts > 0.5))

        if self.phase == GraspGatePhase.WAITING:
            candidate_now = (
                contact_count >= self.config.min_contact_count
                and (
                    not self.config.require_floor_clear
                    or not bool(object_floor_contact)
                )
            )
            self.contact_run = self.contact_run + 1 if candidate_now else 0
            if self.contact_run < self.config.contact_persistence_frames:
                return GraspGateStep(proposed.copy(), self.phase, False, contact_count)
            self.phase = GraspGatePhase.PROBE
            self.phase_step = 0
            self.trigger_frame = int(frame_index)
            self.trigger_object_position = obj.copy()
            self.latched_target = proposed.copy()
            # Preserve achieved root pose; preserve the policy's contemporaneous
            # finger servo target that produced the persistent contact event.
            self.latched_target[:ROOT_DIM] = current[:ROOT_DIM]
            self._event(
                frame_index,
                "contact_candidate",
                contact_count=contact_count,
                persistence_frames=int(self.contact_run),
                object_floor_contact=bool(object_floor_contact),
                object_position_m=obj.tolist(),
            )

        if self.phase == GraspGatePhase.PROBE:
            if self.phase_step >= self.config.probe_frames:
                passed, evidence = self._physical_pass(
                    object_position=obj,
                    object_floor_contact=object_floor_contact,
                    contact_count=contact_count,
                    follow_min_m=self.config.probe_follow_min_m,
                    min_contact_count=self.config.probe_min_contact_count,
                )
                self._event(frame_index, "probe_passed" if passed else "probe_failed", **evidence)
                if not passed:
                    self.phase = GraspGatePhase.FAILED
                    return GraspGateStep(proposed.copy(), self.phase, False, contact_count)
                self.phase = GraspGatePhase.RETENTION
                self.phase_step = 0
            if self.phase == GraspGatePhase.PROBE:
                self.phase_step += 1
                fraction = min(1.0, self.phase_step / self.config.probe_frames)
                target = self._latched_lift_target(self.config.probe_lift_m * fraction)
                return GraspGateStep(target, self.phase, True, contact_count)

        if self.phase == GraspGatePhase.RETENTION:
            if self.phase_step >= self.config.retention_frames:
                passed, evidence = self._physical_pass(
                    object_position=obj,
                    object_floor_contact=object_floor_contact,
                    contact_count=contact_count,
                    follow_min_m=self.config.retention_follow_min_m,
                    min_contact_count=self.config.retention_min_contact_count,
                )
                self._event(
                    frame_index,
                    "retention_passed" if passed else "retention_failed",
                    **evidence,
                )
                self.phase = GraspGatePhase.PASSED if passed else GraspGatePhase.FAILED
                if not passed:
                    return GraspGateStep(proposed.copy(), self.phase, False, contact_count)
            if self.phase == GraspGatePhase.RETENTION:
                self.phase_step += 1
                fraction = min(1.0, self.phase_step / self.config.retention_frames)
                lift = self.config.probe_lift_m + self.config.retention_lift_m * fraction
                return GraspGateStep(
                    self._latched_lift_target(lift), self.phase, True, contact_count
                )

        if self.phase == GraspGatePhase.PASSED:
            return GraspGateStep(
                self._latched_lift_target(
                    self.config.probe_lift_m + self.config.retention_lift_m
                ),
                self.phase,
                True,
                contact_count,
            )
        return GraspGateStep(proposed.copy(), self.phase, False, contact_count)

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name.lower(),
            "phase_id": int(self.phase),
            "trigger_frame": self.trigger_frame,
            "events": list(self.events),
            "config": {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__
            },
        }
