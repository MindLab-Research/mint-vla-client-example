"""Action-label projections for MANO trajectories with recorded PD setpoints."""

from __future__ import annotations

from typing import Any

import numpy as np


MEASURED_DELTA = "measured_delta"
PD_TARGET_DELTA = "pd_target_delta"
# B scheme: store the ABSOLUTE urdf_dof_target as the action label and let the
# OpenPI DeltaActions/AbsoluteActions transforms perform delta/undelta w.r.t. the
# current query state. This matches the official Physical-Intelligence contract
# for datasets with absolute target joint angles.
URDF_TARGET_ABSOLUTE = "urdf_target_absolute"
ACTION_SOURCES = (MEASURED_DELTA, PD_TARGET_DELTA, URDF_TARGET_ABSOLUTE)

# B-schema masks. Euler XYZ is absolute; translation and finger joints are
# delta-eligible; tail dimensions are physical zero padding. State32/state44
# retain the 26D contract while state41 uses the authoritative 28D ManoRL order.
MANO_26D_DELTA_MASK_SEGMENTS: tuple[int, ...] = (3, -3, 20, -6)
MANO_28D_DELTA_MASK_SEGMENTS: tuple[int, ...] = (3, -3, 22, -4)
# Compatibility name retained for existing 26D callers.
MANO_DELTA_MASK_SEGMENTS = MANO_26D_DELTA_MASK_SEGMENTS


def _absolute_target(
    row: dict[str, Any], action_dim: int
) -> tuple[np.ndarray, np.ndarray, int]:
    state = np.asarray(row["state"], dtype=np.float32)
    hands = row.get("hands")
    if not isinstance(hands, list) or len(hands) != 1:
        raise ValueError(
            "urdf target actions require exactly one hand, got "
            f"{type(hands).__name__}:{len(hands) if isinstance(hands, list) else 'n/a'}"
        )
    target = np.asarray(hands[0].get("urdf_dof_target"), dtype=np.float32)
    if target.ndim != 2 or target.shape[1] not in (26, 28):
        raise ValueError(f"urdf_dof_target must have shape [T,26|28], got {target.shape}")
    active_dim = int(target.shape[1])
    if state.ndim != 2 or target.shape[0] != state.shape[0]:
        raise ValueError(
            "urdf_dof_target frame count must align with state: "
            f"target={target.shape}, state={state.shape}"
        )
    if state.shape[1] < active_dim:
        raise ValueError(
            f"state must have shape [T,>={active_dim}] aligned with target, got {state.shape}"
        )
    if action_dim < active_dim:
        raise ValueError(f"action_dim must be at least {active_dim}, got {action_dim}")
    if not np.isfinite(target).all() or not np.isfinite(state[:, :active_dim]).all():
        raise ValueError("state/urdf target contains non-finite values")
    return state, target, active_dim


def urdf_target_absolute_actions(row: dict[str, Any], *, action_dim: int = 32) -> np.ndarray:
    """Return the ABSOLUTE per-frame ``urdf_dof_target`` in the model action space.

    The framework's ``DeltaActions(mask)`` converts the delta-eligible dims
    (xyz + fingers) to ``target - q[t]`` before normalization, and
    ``AbsoluteActions(mask)`` inverts it at inference. Euler dims are left
    absolute by the mask. The final padding dims remain zeros.
    """
    state, target, active_dim = _absolute_target(row, action_dim)
    actions = np.zeros((state.shape[0], action_dim), dtype=np.float32)
    actions[:, :active_dim] = target
    if not np.isfinite(actions).all():
        raise ValueError("urdf_target_absolute produced non-finite actions")
    return actions


def pd_target_delta_actions(row: dict[str, Any], *, action_dim: int = 32) -> np.ndarray:
    """Return per-frame ``PD setpoint - observed q`` in the model action space.

    The recorded setpoint is an absolute joint configuration.  Projecting it to
    a residual preserves the existing rollout contract ``q_next = q + action``:
    applying the complete residual reaches the commanded PD target.  The final
    six model dimensions remain padding zeros.
    """
    state, target, active_dim = _absolute_target(row, action_dim)
    actions = np.zeros((state.shape[0], action_dim), dtype=np.float32)
    actions[:, :active_dim] = target - state[:, :active_dim]
    if not np.isfinite(actions).all():
        raise ValueError("pd_target_delta produced non-finite actions")
    return actions


def project_row_actions(
    row: dict[str, Any], action_source: str, *, action_dim: int = 32
) -> dict[str, Any]:
    """Return ``row`` with the selected action label, preserving other leaves."""
    if action_source == MEASURED_DELTA:
        return row
    if action_source not in (PD_TARGET_DELTA, URDF_TARGET_ABSOLUTE):
        raise ValueError(f"unsupported action_source {action_source!r}; expected one of {ACTION_SOURCES}")
    projected = dict(row)
    projected["measured_actions"] = np.asarray(row["actions"], dtype=np.float32)
    if action_source == URDF_TARGET_ABSOLUTE:
        projected["actions"] = urdf_target_absolute_actions(row, action_dim=action_dim)
    else:
        projected["actions"] = pd_target_delta_actions(row, action_dim=action_dim)
    return projected


def fit_pd_response_gains(
    state: np.ndarray,
    measured_actions: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Fit a diagonal one-step response ``dq ~= gain * (target - q)``.

    Gains are clipped to [0, 1]. They approximate the recorded PD dynamics for
    kinematic visualization; they are not physical simulator parameters.
    """
    state = np.asarray(state, dtype=np.float64)
    measured_actions = np.asarray(measured_actions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 26:
        raise ValueError(f"state must have shape [N,>=26], got {state.shape}")
    if measured_actions.shape[0] != state.shape[0] or measured_actions.shape[1] < 26:
        raise ValueError(f"measured_actions must have shape [N,>=26], got {measured_actions.shape}")
    if target.shape != (state.shape[0], 26):
        raise ValueError(f"target must have shape {(state.shape[0], 26)}, got {target.shape}")
    error = target - state[:, :26]
    numerator = np.sum(error * measured_actions[:, :26], axis=0)
    denominator = np.sum(error * error, axis=0)
    gains = np.divide(numerator, denominator, out=np.zeros(26), where=denominator > 1e-12)
    gains = np.clip(gains, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(gains).all():
        raise ValueError("PD response gain fit produced non-finite values")
    return gains
