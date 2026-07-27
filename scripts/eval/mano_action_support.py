"""Minimal policy/scene helpers for the sole MANO Mode4 evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openpi_vla_smoke_lance_base as L
from export_mano_sim_video import (
    MjcfBuilder,
    WRIST_FRAME_BODY,
    add_object,
    add_urdf_hand,
)


DEFAULT_HAND_URDF = Path(
    "/vePFS-Mindverse/user/intern/wenxi/pi-finetune/3rd-party/all_assets/"
    "Assets/HAND/s02/mano/Z_upNew/mano_hand.urdf"
)
DEFAULT_OBJECTS_URDF_DIR = Path(
    "/vePFS-Mindverse/share/ylang/all_assets/Assets/sim/mano_objects_urdf"
)
HEAD_CAMERA_NAME = "dataset_b_head_camera"
WRIST_CAMERA_NAME = "dataset_b_wrist_camera"
HEAD_CAMERA = {
    "position": (0.8, 0.0, 0.5),
    "target": (0.3, 0.0, 0.1),
    "horizontal_fov": 75.0,
}
WRIST_CAMERA = {
    "position": (-0.08, 0.0, -0.08),
    "target": (0.06, 0.0, -0.05),
    "horizontal_fov": 95.0,
    "parent_body": WRIST_FRAME_BODY,
}


class _ObservationHelpers:
    @staticmethod
    def _unnormalize_actions(norm_actions: np.ndarray, data_config) -> np.ndarray:
        """Convert server-normalized action chunks back to dataset units."""
        values = np.asarray(norm_actions, dtype=np.float32)
        dimensions = values.shape[-1]
        stats = data_config.norm_stats["actions"]
        if data_config.use_quantile_norm:
            q01 = np.asarray(stats.q01, dtype=np.float32)[:dimensions]
            q99 = np.asarray(stats.q99, dtype=np.float32)[:dimensions]
            return (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean = np.asarray(stats.mean, dtype=np.float32)[:dimensions]
        std = np.asarray(stats.std, dtype=np.float32)[:dimensions]
        return values * (std + 1e-6) + mean


OBS = _ObservationHelpers()


def axis_angle_to_wxyz(rot_aa: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rot_aa, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    xyz = (rotvec / angle) * np.sin(angle / 2.0)
    return np.asarray([np.cos(angle / 2.0), *xyz], dtype=np.float64)


def build_scene(*, hand_urdf: Path, object_urdf: Path, object_code: str, width: int, height: int):
    builder = MjcfBuilder(offwidth=width, offheight=height, show_debug_frames=False)
    object_freejoint = add_object(builder, object_urdf, object_code)
    hand_joints = add_urdf_hand(builder, hand_urdf)
    builder.add_camera(
        name=HEAD_CAMERA_NAME,
        position=HEAD_CAMERA["position"],
        target=HEAD_CAMERA["target"],
        horizontal_fov=HEAD_CAMERA["horizontal_fov"],
    )
    builder.add_camera(
        name=WRIST_CAMERA_NAME,
        position=WRIST_CAMERA["position"],
        target=WRIST_CAMERA["target"],
        horizontal_fov=WRIST_CAMERA["horizontal_fov"],
        body_name=WRIST_CAMERA["parent_body"],
    )
    return builder, object_freejoint, hand_joints
