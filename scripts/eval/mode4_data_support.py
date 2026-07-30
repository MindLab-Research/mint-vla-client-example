"""Minimal dataset/scene helpers for the sole MANO Mode4 evaluator."""

from __future__ import annotations

import mujoco
import numpy as np

import mano_action_support as action_support
from scripts import contact_windows


L = action_support.L
HORIZON = 10
ACTION_DIM = 32


def safe_object_name(row: dict) -> str:
    names = (row.get("trajectory_metadata") or {}).get("object_names") or []
    if len(names) != 1 or not isinstance(names[0], str):
        raise ValueError(f"Mode4 requires exactly one named object, got {names!r}")
    return names[0]


def row_frame_count(row: dict) -> int:
    lengths: list[int] = []
    for key in ("state", "actions", "image", "wrist_image", "timestamp"):
        value = row.get(key)
        if value is not None:
            lengths.append(len(value))
    objects = row.get("objects") or []
    if len(objects) != 1:
        raise ValueError(f"Mode4 requires exactly one object trajectory, got {len(objects)}")
    lengths.extend((len(objects[0]["pos"]), len(objects[0]["rot_aa"])))
    metadata_frames = int((row.get("episode_metadata") or {}).get("total_frames") or 0)
    if metadata_frames > 0:
        lengths.append(metadata_frames)
    if not lengths or min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError(f"inconsistent Mode4 frame counts: {lengths}")
    return lengths[0]


def resolve_row_window(
    row: dict,
    *,
    row_index: int,
    frame_window: str,
    contact_context_frames: int,
    missing_contact_policy: str,
    manifest_entry: dict | None = None,
) -> contact_windows.ContactWindow | None:
    """Resolve the absolute source-frame interval used to initialize and run Mode4."""
    count = row_frame_count(row)
    return contact_windows.select_window(
        row,
        row_index=row_index,
        total_frames=count,
        mode=frame_window,
        manifest_entry=manifest_entry,
        context_frames=contact_context_frames,
        missing_policy=missing_contact_policy,
    )


def build_model_config(base_model: str):
    profile = L.resolve_profile(base_model)
    return L._build_model_config(
        HORIZON,
        state_dim=profile.state_dim,
        action_dim=ACTION_DIM,
        base_model=base_model,
    )


def set_scene_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    state: np.ndarray,
    object_pos: np.ndarray,
    object_rot_aa: np.ndarray,
    object_addr: int,
    hand_addrs: list[int],
) -> None:
    """Initialize frame 0 exactly once, then forward the fresh MuJoCo state."""
    hand = np.asarray(state, dtype=np.float64)
    if hand.shape != (26,):
        raise ValueError(f"expected initial hand shape (26,), got {hand.shape}")
    data.qpos[:] = 0
    data.qpos[object_addr : object_addr + 3] = np.asarray(object_pos, dtype=np.float64)
    data.qpos[object_addr + 3 : object_addr + 7] = action_support.axis_angle_to_wxyz(object_rot_aa)
    data.qpos[hand_addrs] = hand
    mujoco.mj_forward(model, data)
