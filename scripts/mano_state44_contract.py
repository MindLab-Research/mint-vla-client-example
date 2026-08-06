"""MANO state44 observation contract shared by training and Mode4.

state[0:26]  MANO qpos26
state[26:31] five-finger target-object contact presence
state[31]    object lift from source/rollout initialization
state[32:37] signed URDF fingertip-sphere to object-collision-surface distance5
state[37:42] causal 25 ms fingertip-to-palm radial rate5 (+closing, -opening)
state[42]    object-floor contact presence
state[43]    elapsed duration of the current >=2-finger contact run

The feature clock is the dataset's exact 5 ms timestamp interval. All rates use
only current and past observations; no centered difference or future frame is
permitted. Finger order is index/thumb/ring/middle/pinky.
"""
from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from scripts.mano_state_contract import FINGER_NAMES, HAND_QPOS_DIM, aggregate_finger_contacts

STATE44_CONTRACT_ID = "mano_five_finger_contact_geom_rate_v2"
STATE44_DIM = 44
STATE44_ACTION_DIM = 32
STATE44_SAMPLE_DT_SECONDS = 0.005
STATE44_RATE_WINDOW_STEPS = 5
STATE44_RATE_WINDOW_SECONDS = STATE44_SAMPLE_DT_SECONDS * STATE44_RATE_WINDOW_STEPS
STATE44_MIN_MULTICONTACT_FINGERS = 2
# Persistence remains a physical duration without a hand-chosen hard clip;
# the authenticated population q01/q99 provides its model normalization.
STATE44_PERSISTENCE_CLIP_SECONDS = None

CONTACT_SLICE = slice(26, 31)
LIFT_HEIGHT_INDEX = 31
SURFACE_DISTANCE_SLICE = slice(32, 37)
RADIAL_RATE_SLICE = slice(37, 42)
FLOOR_SUPPORT_INDEX = 42
MULTICONTACT_PERSISTENCE_INDEX = 43


@dataclass(frozen=True)
class _CachedScene:
    temporary_directory: Any
    model: Any
    object_addr: int
    hand_addrs: tuple[int, ...]
    tip_geom_ids: tuple[int, ...]
    object_geom_ids: tuple[int, ...]
    palm_body_id: int


_SCENE_CACHE: dict[str, _CachedScene] = {}
_SCENE_CACHE_LOCK = threading.RLock()


def _close_scene_cache() -> None:
    with _SCENE_CACHE_LOCK:
        entries = list(_SCENE_CACHE.values())
        _SCENE_CACHE.clear()
    for entry in entries:
        entry.temporary_directory.cleanup()


atexit.register(_close_scene_cache)


def _cached_scene(object_name: str) -> _CachedScene:
    with _SCENE_CACHE_LOCK:
        cached = _SCENE_CACHE.get(object_name)
        if cached is not None:
            return cached
        from scripts.eval import mano_physics_core as physics

        temporary, model, _data, renderer, object_addr, _object_dof_addr, hand_addrs, _hand_dof_addrs, _limits = (
            physics.make_scene(
                object_name,
                32,
                32,
                physics=True,
                create_renderer=False,
                state44_features=True,
            )
        )
        if renderer is not None:
            raise RuntimeError("state44 cached geometry scene unexpectedly created a renderer")
        tip_geom_ids, object_geom_ids, palm_body_id = physics.resolve_state44_feature_ids(model, object_name)
        cached = _CachedScene(
            temporary_directory=temporary,
            model=model,
            object_addr=int(object_addr),
            hand_addrs=tuple(int(value) for value in hand_addrs),
            tip_geom_ids=tuple(int(value) for value in tip_geom_ids),
            object_geom_ids=tuple(int(value) for value in object_geom_ids),
            palm_body_id=int(palm_body_id),
        )
        _SCENE_CACHE[object_name] = cached
        return cached


def validate_state44_timestamps(timestamps: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.shape != (frame_count,) or not np.isfinite(values).all():
        raise ValueError(f"state44 requires finite timestamps shape {(frame_count,)}, got {values.shape}")
    if frame_count > 1 and not np.allclose(
        np.diff(values), STATE44_SAMPLE_DT_SECONDS, rtol=0, atol=1e-10
    ):
        raise ValueError("state44 requires exact monotonic 5 ms source timestamps")
    return values


def causal_fingertip_radial_rates(radial_distances: np.ndarray) -> np.ndarray:
    """Return [T,5] rates in m/s; positive means fingertip moved toward palm."""
    distances = np.asarray(radial_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != len(FINGER_NAMES):
        raise ValueError(f"radial distances must have shape [T,5], got {distances.shape}")
    if not np.isfinite(distances).all():
        raise ValueError("radial distances must be finite")
    result = np.zeros_like(distances, dtype=np.float64)
    steps = STATE44_RATE_WINDOW_STEPS
    if len(distances) > steps:
        result[steps:] = -(
            distances[steps:] - distances[:-steps]
        ) / STATE44_RATE_WINDOW_SECONDS
    return result.astype(np.float32)


def multicontact_persistence(contacts: np.ndarray) -> np.ndarray:
    """Elapsed seconds since the current >=2-finger run began; 0 at onset/off."""
    values = np.asarray(contacts, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FINGER_NAMES):
        raise ValueError(f"contacts must have shape [T,5], got {values.shape}")
    result = np.zeros(len(values), dtype=np.float32)
    run_frames = 0
    for index, active in enumerate(np.sum(values > 0.5, axis=1) >= STATE44_MIN_MULTICONTACT_FINGERS):
        if active:
            run_frames += 1
            result[index] = np.float32((run_frames - 1) * STATE44_SAMPLE_DT_SECONDS)
        else:
            run_frames = 0
    return result


def assemble_state44_sequence(
    *,
    hand_qpos: np.ndarray,
    contacts: np.ndarray,
    object_lift: np.ndarray,
    signed_surface_distances: np.ndarray,
    radial_rates: np.ndarray,
    floor_support: np.ndarray,
    persistence: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(hand_qpos, dtype=np.float32)
    contacts = np.asarray(contacts, dtype=np.float32)
    lift = np.asarray(object_lift, dtype=np.float32)
    surface = np.asarray(signed_surface_distances, dtype=np.float32)
    rates = np.asarray(radial_rates, dtype=np.float32)
    floor = np.asarray(floor_support, dtype=np.float32)
    persistence = np.asarray(persistence, dtype=np.float32)
    frame_count = len(qpos)
    expected5 = (frame_count, len(FINGER_NAMES))
    if qpos.shape != (frame_count, HAND_QPOS_DIM):
        raise ValueError(f"hand qpos must have shape {(frame_count, HAND_QPOS_DIM)}, got {qpos.shape}")
    for name, value in (("contacts", contacts), ("surface", surface), ("rates", rates)):
        if value.shape != expected5:
            raise ValueError(f"{name} must have shape {expected5}, got {value.shape}")
    for name, value in (("lift", lift), ("floor", floor), ("persistence", persistence)):
        if value.shape != (frame_count,):
            raise ValueError(f"{name} must have shape {(frame_count,)}, got {value.shape}")
    state = np.zeros((frame_count, STATE44_DIM), dtype=np.float32)
    state[:, :HAND_QPOS_DIM] = qpos
    state[:, CONTACT_SLICE] = contacts
    state[:, LIFT_HEIGHT_INDEX] = lift
    state[:, SURFACE_DISTANCE_SLICE] = surface
    state[:, RADIAL_RATE_SLICE] = rates
    state[:, FLOOR_SUPPORT_INDEX] = floor
    state[:, MULTICONTACT_PERSISTENCE_INDEX] = persistence
    if not np.isfinite(state).all():
        raise FloatingPointError("state44 contains non-finite values")
    return state


def compute_state44_geometry_frame(
    *,
    object_name: str,
    hand_qpos: np.ndarray,
    object_position: np.ndarray,
    object_rotation_axis_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Evaluate one physical state with the same geometry used by source replay."""
    import mujoco

    from scripts.eval import mano_action_support as action_support
    from scripts.eval import mano_physics_core as physics

    scene = _cached_scene(object_name)
    data = mujoco.MjData(scene.model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[scene.object_addr : scene.object_addr + 3] = np.asarray(
        object_position, dtype=np.float64
    )
    data.qpos[scene.object_addr + 3 : scene.object_addr + 7] = action_support.axis_angle_to_wxyz(
        np.asarray(object_rotation_axis_angle, dtype=np.float64)
    )
    qpos = np.asarray(hand_qpos, dtype=np.float64)
    if qpos.shape != (HAND_QPOS_DIM,):
        raise ValueError(f"state44 frame hand_qpos must have shape {(HAND_QPOS_DIM,)}, got {qpos.shape}")
    data.qpos[list(scene.hand_addrs)] = qpos
    mujoco.mj_forward(scene.model, data)
    return physics.state44_geometry_from_mujoco(
        scene.model,
        data,
        object_name,
        tip_geom_ids=scene.tip_geom_ids,
        object_geom_ids=scene.object_geom_ids,
        palm_body_id=scene.palm_body_id,
    )


def compute_source_state44_sequence(row: dict[str, Any], object_name: str) -> np.ndarray:
    """Replay source qpos/object poses through the canonical MuJoCo geometry extractor."""
    import mujoco

    from scripts.eval import mano_action_support as action_support
    from scripts.eval import mano_physics_core as physics

    hand_qpos = np.asarray(row["state"], dtype=np.float64)[:, :HAND_QPOS_DIM]
    frame_count = len(hand_qpos)
    validate_state44_timestamps(np.asarray(row["timestamp"]), frame_count)
    objects = row.get("objects") or []
    if len(objects) != 1:
        raise ValueError(f"state44 requires exactly one object trajectory, got {len(objects)}")
    object_positions = np.asarray(objects[0]["pos"], dtype=np.float64)
    object_rotations = np.asarray(objects[0]["rot_aa"], dtype=np.float64)
    if object_positions.shape != (frame_count, 3) or object_rotations.shape != (frame_count, 3):
        raise ValueError(
            f"state44 object pose shapes must be {(frame_count, 3)}, got "
            f"{object_positions.shape} and {object_rotations.shape}"
        )
    frame_contacts = row.get("contact")
    if frame_contacts is None or len(frame_contacts) != frame_count:
        raise ValueError("state44 requires one Lance contact list per source frame")

    scene = _cached_scene(object_name)
    data = mujoco.MjData(scene.model)
    signed_surface = np.empty((frame_count, len(FINGER_NAMES)), dtype=np.float32)
    radial_distance = np.empty_like(signed_surface)
    floor_support = np.empty(frame_count, dtype=np.float32)
    for frame in range(frame_count):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[scene.object_addr : scene.object_addr + 3] = object_positions[frame]
        data.qpos[scene.object_addr + 3 : scene.object_addr + 7] = action_support.axis_angle_to_wxyz(
            object_rotations[frame]
        )
        data.qpos[list(scene.hand_addrs)] = hand_qpos[frame]
        mujoco.mj_forward(scene.model, data)
        signed_surface[frame], radial_distance[frame], floor_support[frame] = (
            physics.state44_geometry_from_mujoco(
                scene.model,
                data,
                object_name,
                tip_geom_ids=scene.tip_geom_ids,
                object_geom_ids=scene.object_geom_ids,
                palm_body_id=scene.palm_body_id,
            )
        )

    contacts = np.stack(
        [aggregate_finger_contacts(list(records or []), object_name) for records in frame_contacts]
    ).astype(np.float32)
    lift = (object_positions[:, 2] - object_positions[0, 2]).astype(np.float32)
    rates = causal_fingertip_radial_rates(radial_distance)
    persistence = multicontact_persistence(contacts)
    return assemble_state44_sequence(
        hand_qpos=hand_qpos.astype(np.float32),
        contacts=contacts,
        object_lift=lift,
        signed_surface_distances=signed_surface,
        radial_rates=rates,
        floor_support=floor_support,
        persistence=persistence,
    )


class State44History:
    """Causal Mode4 history for radial-rate and multi-contact persistence."""

    def __init__(
        self,
        *,
        prior_radial_distances: np.ndarray | None = None,
        prior_contacts: np.ndarray | None = None,
    ) -> None:
        self._radial = deque(maxlen=STATE44_RATE_WINDOW_STEPS + 1)
        if prior_radial_distances is not None:
            values = np.asarray(prior_radial_distances, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != len(FINGER_NAMES):
                raise ValueError(f"prior radial distances must have shape [T,5], got {values.shape}")
            for value in values[-self._radial.maxlen :]:
                self._radial.append(value.copy())
        self._multicontact_run_frames = 0
        if prior_contacts is not None:
            values = np.asarray(prior_contacts, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != len(FINGER_NAMES):
                raise ValueError(f"prior contacts must have shape [T,5], got {values.shape}")
            for active in np.sum(values > 0.5, axis=1)[::-1] >= STATE44_MIN_MULTICONTACT_FINGERS:
                if not active:
                    break
                self._multicontact_run_frames += 1

    def observe(self, radial_distances: np.ndarray, contacts: np.ndarray) -> tuple[np.ndarray, np.float32]:
        radial = np.asarray(radial_distances, dtype=np.float32)
        contacts = np.asarray(contacts, dtype=np.float32)
        if radial.shape != (len(FINGER_NAMES),) or contacts.shape != (len(FINGER_NAMES),):
            raise ValueError("state44 live radial distances and contacts must both have shape (5,)")
        self._radial.append(radial.copy())
        if len(self._radial) == self._radial.maxlen:
            rate = -(
                np.asarray(self._radial[-1], dtype=np.float64)
                - np.asarray(self._radial[0], dtype=np.float64)
            ) / STATE44_RATE_WINDOW_SECONDS
        else:
            rate = np.zeros(len(FINGER_NAMES), dtype=np.float64)
        if np.count_nonzero(contacts > 0.5) >= STATE44_MIN_MULTICONTACT_FINGERS:
            self._multicontact_run_frames += 1
            persistence = np.float32(
                max(0, self._multicontact_run_frames - 1) * STATE44_SAMPLE_DT_SECONDS
            )
        else:
            self._multicontact_run_frames = 0
            persistence = np.float32(0.0)
        return rate.astype(np.float32), persistence


def assemble_live_state44(
    *,
    hand_qpos: np.ndarray,
    contacts: np.ndarray,
    object_lift: float,
    signed_surface_distances: np.ndarray,
    radial_rates: np.ndarray,
    floor_support: float,
    persistence: float,
) -> np.ndarray:
    return assemble_state44_sequence(
        hand_qpos=np.asarray(hand_qpos, dtype=np.float32)[None, :],
        contacts=np.asarray(contacts, dtype=np.float32)[None, :],
        object_lift=np.asarray([object_lift], dtype=np.float32),
        signed_surface_distances=np.asarray(signed_surface_distances, dtype=np.float32)[None, :],
        radial_rates=np.asarray(radial_rates, dtype=np.float32)[None, :],
        floor_support=np.asarray([floor_support], dtype=np.float32),
        persistence=np.asarray([persistence], dtype=np.float32),
    )[0]
