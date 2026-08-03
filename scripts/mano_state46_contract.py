"""Native simulated MANO state46/action32 contract.

state[0:28]   simulated right-hand qpos28
state[28:33]  simulated finger/object contact presence5
state[33]     simulated object lift from frame0
state[34:39]  simulated fingertip/object signed surface distance5
state[39:44]  causal 25 ms fingertip-to-palm radial closing rate5
state[44]     simulated object/floor contact presence
state[45]     elapsed duration of the current >=2-finger contact run

action[0:28] source absolute right-hand urdf_dof_target
action[28:32] physical zero padding

Finger order remains index/thumb/ring/middle/pinky. All features use the exact
5 ms native trace clock. Rates and contact duration are strictly causal.
"""
from __future__ import annotations

import numpy as np

STATE46_CONTRACT_ID = "mano_state46_native_sim_28d_v1"
ACTION32_CONTRACT_ID = "mano_action32_b_target28_v1"
HAND_QPOS_DIM = 28
STATE_DIM = 46
ACTION_DIM = 32
ACTION_PADDING = 4
SAMPLE_DT_SECONDS = 0.005
RATE_WINDOW_STEPS = 5
RATE_WINDOW_SECONDS = SAMPLE_DT_SECONDS * RATE_WINDOW_STEPS
MIN_MULTICONTACT_FINGERS = 2
FINGER_NAMES = ("index", "thumb", "ring", "middle", "pinky")

HAND_QPOS_SLICE = slice(0, 28)
CONTACT_SLICE = slice(28, 33)
LIFT_HEIGHT_INDEX = 33
SURFACE_DISTANCE_SLICE = slice(34, 39)
RADIAL_RATE_SLICE = slice(39, 44)
FLOOR_SUPPORT_INDEX = 44
MULTICONTACT_PERSISTENCE_INDEX = 45
# xyz delta, intrinsic XYZ Euler absolute, remaining 22 joints delta, pad4.
MANO_28D_DELTA_MASK_SEGMENTS: tuple[int, ...] = (3, -3, 22, -4)


def validate_timestamps(timestamps: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.shape != (frame_count,) or not np.isfinite(values).all():
        raise ValueError(
            f"state46 requires finite timestamps shape {(frame_count,)}, got {values.shape}"
        )
    if not np.isclose(values[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("state46 timestamps must start at zero")
    if frame_count > 1 and not np.allclose(
        np.diff(values), SAMPLE_DT_SECONDS, rtol=0.0, atol=1e-10
    ):
        raise ValueError("state46 requires exact monotonic 5 ms timestamps")
    return values


def causal_fingertip_radial_rates(radial_distances: np.ndarray) -> np.ndarray:
    """Return [T,5] m/s; positive means fingertip moved toward the palm."""
    distances = np.asarray(radial_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != len(FINGER_NAMES):
        raise ValueError(f"radial distances must have shape [T,5], got {distances.shape}")
    if not np.isfinite(distances).all():
        raise ValueError("radial distances must be finite")
    result = np.zeros_like(distances, dtype=np.float64)
    if len(distances) > RATE_WINDOW_STEPS:
        result[RATE_WINDOW_STEPS:] = -(
            distances[RATE_WINDOW_STEPS:] - distances[:-RATE_WINDOW_STEPS]
        ) / RATE_WINDOW_SECONDS
    return result.astype(np.float32)


def multicontact_persistence(contacts: np.ndarray) -> np.ndarray:
    """Seconds since current >=2-finger run began; zero at onset and when off."""
    values = np.asarray(contacts, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FINGER_NAMES):
        raise ValueError(f"contacts must have shape [T,5], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("contacts must be finite")
    result = np.zeros(len(values), dtype=np.float32)
    run_frames = 0
    for frame, active in enumerate(
        np.sum(values > 0.5, axis=1) >= MIN_MULTICONTACT_FINGERS
    ):
        if active:
            run_frames += 1
            result[frame] = np.float32((run_frames - 1) * SAMPLE_DT_SECONDS)
        else:
            run_frames = 0
    return result


def assemble_state46_sequence(
    *,
    hand_qpos: np.ndarray,
    contacts: np.ndarray,
    object_lift: np.ndarray,
    signed_surface_distances: np.ndarray,
    radial_distances: np.ndarray,
    floor_support: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(hand_qpos, dtype=np.float32)
    contact = np.asarray(contacts, dtype=np.float32)
    lift = np.asarray(object_lift, dtype=np.float32)
    surface = np.asarray(signed_surface_distances, dtype=np.float32)
    radial = np.asarray(radial_distances, dtype=np.float32)
    floor = np.asarray(floor_support, dtype=np.float32)
    frame_count = len(qpos)
    if qpos.shape != (frame_count, HAND_QPOS_DIM):
        raise ValueError(
            f"hand qpos must have shape {(frame_count, HAND_QPOS_DIM)}, got {qpos.shape}"
        )
    expected5 = (frame_count, len(FINGER_NAMES))
    for name, value in (("contacts", contact), ("surface", surface), ("radial", radial)):
        if value.shape != expected5:
            raise ValueError(f"{name} must have shape {expected5}, got {value.shape}")
    for name, value in (("lift", lift), ("floor", floor)):
        if value.shape != (frame_count,):
            raise ValueError(f"{name} must have shape {(frame_count,)}, got {value.shape}")
    rates = causal_fingertip_radial_rates(radial)
    persistence = multicontact_persistence(contact)
    state = np.empty((frame_count, STATE_DIM), dtype=np.float32)
    state[:, HAND_QPOS_SLICE] = qpos
    state[:, CONTACT_SLICE] = contact
    state[:, LIFT_HEIGHT_INDEX] = lift
    state[:, SURFACE_DISTANCE_SLICE] = surface
    state[:, RADIAL_RATE_SLICE] = rates
    state[:, FLOOR_SUPPORT_INDEX] = floor
    state[:, MULTICONTACT_PERSISTENCE_INDEX] = persistence
    if not np.isfinite(state).all():
        raise FloatingPointError("state46 contains non-finite values")
    return state


def absolute_target_actions32(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != HAND_QPOS_DIM:
        raise ValueError(f"absolute targets must have shape [T,28], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("absolute targets must be finite")
    actions = np.zeros((len(values), ACTION_DIM), dtype=np.float32)
    actions[:, :HAND_QPOS_DIM] = values
    return actions
