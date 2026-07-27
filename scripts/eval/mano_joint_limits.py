"""Name-aligned scalar MuJoCo joint-limit handling for MANO visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# MuJoCo's mjtJoint values. Keeping the helper independent from mujoco makes its
# contract unit-testable without a renderer or compiled assets.
_SCALAR_JOINT_TYPES = frozenset((2, 3))  # slide, hinge


@dataclass(frozen=True)
class HandJointLimits:
    """Compiled scalar limit metadata in the caller's hand-state ordering."""

    names: tuple[str, ...]
    limited: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def compiled_hand_joint_limits(
    model: Any, joint_names: Sequence[str], joint_ids: Sequence[int]
) -> HandJointLimits:
    """Read scalar limits from a compiled ``MjModel`` in ``joint_names`` order.

    MANO visualization state maps one scalar to each hand joint. Free and ball
    joints therefore have no valid mapping and fail explicitly rather than
    accidentally treating their first qpos coordinate as a scalar joint.
    """
    names = tuple(str(name) for name in joint_names)
    ids = np.asarray(joint_ids)
    if not names or ids.ndim != 1 or ids.shape[0] != len(names):
        raise ValueError("joint names and joint ids must be non-empty, aligned one-dimensional sequences")
    if len(set(names)) != len(names):
        raise ValueError("hand joint names must be unique")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("joint ids must be integers")

    joint_type = np.asarray(model.jnt_type)
    limited_all = np.asarray(model.jnt_limited, dtype=bool)
    ranges = np.asarray(model.jnt_range, dtype=np.float64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"model.jnt_range must have shape (n, 2), got {ranges.shape}")
    if any(index < 0 or index >= len(joint_type) for index in ids):
        raise ValueError("hand joint id is outside compiled model joint arrays")
    if len(limited_all) != len(joint_type) or len(ranges) != len(joint_type):
        raise ValueError("compiled joint arrays have inconsistent lengths")

    types = joint_type[ids]
    invalid = [names[i] for i, value in enumerate(types) if int(value) not in _SCALAR_JOINT_TYPES]
    if invalid:
        raise ValueError(
            "MANO scalar hand mapping cannot represent free/ball or unknown joint types: "
            + ", ".join(invalid)
        )
    limited = limited_all[ids].copy()
    selected_ranges = ranges[ids].copy()
    if limited.any():
        bad = limited & (
            ~np.isfinite(selected_ranges).all(axis=1)
            | (selected_ranges[:, 0] > selected_ranges[:, 1])
        )
        if bad.any():
            bad_names = ", ".join(name for name, is_bad in zip(names, bad, strict=True) if is_bad)
            raise ValueError(f"limited hand joints have invalid compiled ranges: {bad_names}")
    return HandJointLimits(
        names=names,
        limited=limited,
        lower=selected_ranges[:, 0].copy(),
        upper=selected_ranges[:, 1].copy(),
    )


def clip_hand_state(state: np.ndarray, limits: HandJointLimits) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a bounded state and diagnostics without modifying ``state``.

    Unlimited joints are copied exactly, even if their model range has values.
    """
    values = np.asarray(state)
    if values.ndim != 1 or values.shape[0] != len(limits.names):
        raise ValueError(
            f"hand state must have shape ({len(limits.names)},), got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("hand state must contain only finite numeric values")
    bounded = values.astype(np.float64, copy=True)
    bounded[limits.limited] = np.clip(
        bounded[limits.limited], limits.lower[limits.limited], limits.upper[limits.limited]
    )
    correction = bounded - values
    clipped = correction != 0
    per_joint = {
        name: {
            "clipped_values": int(clipped[index]),
            "range": [float(limits.lower[index]), float(limits.upper[index])],
        }
        for index, name in enumerate(limits.names)
        if limits.limited[index]
    }
    return bounded, {
        "limited_joint_count": int(limits.limited.sum()),
        "clipped_values": int(clipped.sum()),
        "max_correction": float(np.abs(correction).max(initial=0.0)),
        "per_joint": per_joint,
    }


def new_clipping_diagnostics(limits: HandJointLimits) -> dict[str, Any]:
    return {
        "limited_joint_count": int(limits.limited.sum()),
        "clipped_steps": 0,
        "clipped_values": 0,
        "max_correction": 0.0,
        "per_joint": {
            name: {"clipped_values": 0, "range": [float(limits.lower[i]), float(limits.upper[i])]}
            for i, name in enumerate(limits.names)
            if limits.limited[i]
        },
        "initial_state": None,
    }


def record_clipping(diagnostics: dict[str, Any], event: dict[str, Any], *, initial: bool = False) -> None:
    """Accumulate one clipping event produced by :func:`clip_hand_state`."""
    if initial:
        diagnostics["initial_state"] = event
    if event["clipped_values"]:
        diagnostics["clipped_steps"] += 1
    diagnostics["clipped_values"] += event["clipped_values"]
    diagnostics["max_correction"] = max(diagnostics["max_correction"], event["max_correction"])
    for name, joint_event in event["per_joint"].items():
        diagnostics["per_joint"][name]["clipped_values"] += joint_event["clipped_values"]
