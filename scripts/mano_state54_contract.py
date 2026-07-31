"""Authenticated MANO 54D observation-state contract (dev_v3 candidate v1).

The action contract remains 32D.  Finger order is always
``index, thumb, ring, middle, pinky``.

Layout::

    [0:26]   MANO URDF qpos
    [26:31]  binary target-object finger contacts
    [31]     target-object body-origin lift from source frame zero (metres)
    [32:47]  five true fingertip endpoints in collision-box coordinates (XYZ)
    [47:52]  log1p summed normal-load magnitude per finger (newtons)
    [52]     object-minus-palm world vertical velocity (metres/second)
    [53]     consecutive >=2-finger contact duration (seconds, clipped to 1)

Collision-box coordinates subtract the box collision centre, rotate world to
object-body coordinates, then divide XYZ by the object's audited half extents.
Thus the six collision faces are approximately +/-1 even though cube1 and
cube2 have different sizes and collision-origin offsets.

Temporal features reset at the selected contact-window start.  Velocity is zero
there.  Contact duration is zero on the first qualifying frame, then increases
by the exact 0.005-second source interval while >=2 distinct fingers remain in
contact.  This gives offline random access and online Mode4 the same history
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np

STATE_CONTRACT_ID = "mano_object_dynamics_state54_v1"
CONTACT_SEMANTICS = "lance_record_or_mujoco_keypoint_pair_presence_v1"
CONTACT_RULE = (
    "Binary contact is target-object pair presence. Force is log1p of summed "
    "per-contact-pair normal loads by finger; no threshold. Finger order: "
    "index/thumb/ring/middle/pinky."
)
STATE_DIM = 54
ACTION_DIM = 32
HAND_QPOS_DIM = 26
SOURCE_INTERVAL_SECONDS = 0.005
CONTACT_AGE_CLIP_SECONDS = 1.0
FORCE_REFERENCE_NEWTONS = 50.0
FORCE_LOG1P_MAX = float(np.log1p(FORCE_REFERENCE_NEWTONS))
STATE54_NORM_SHA256 = "f91a0f1b326b33df0aa90bd1e5433bbae1128276bdd690415b5cebd8b50e1cc9"
POPULATION_ROW_INDICES_SHA256 = "5fd0ed493c18b563d1deb280624f0f8b3c47fd7d4d11fcb8ee17650d3d74a654"
POPULATION_TRAJECTORIES = 1_997
POPULATION_ACTIVE_FRAMES = 1_160_274
PROFILE_MAX_TOKEN_LEN = 256

FINGER_NAMES = ("index", "thumb", "ring", "middle", "pinky")
FINGER_CONTACT_SLICE = slice(26, 31)
LIFT_HEIGHT_INDEX = 31
FINGERTIP_OBJECT_SLICE = slice(32, 47)
FINGER_FORCE_SLICE = slice(47, 52)
RELATIVE_VERTICAL_VELOCITY_INDEX = 52
MULTIFINGER_CONTACT_AGE_INDEX = 53

# Lance mano_joint_pos endpoints in contract finger order.
FINGERTIP_JOINT_INDICES = np.asarray([17, 16, 19, 18, 20], dtype=np.int64)
# Corresponding MuJoCo distal bodies and measured endpoint offsets in each body
# frame.  Across 264 audited poses the maximum deviation was < 0.018 mm.
FINGERTIP_DISTAL_BODIES = ("index_dip", "thumb_ip", "ring_dip", "middle_dip", "pinky_dip")
FINGERTIP_LOCAL_OFFSETS = np.asarray(
    [
        [-0.027383491, -0.000215131, -0.000966290],
        [-0.028633220, -0.004190809, 0.023667239],
        [-0.026163518, -0.000075790, -0.007781369],
        [-0.027548900, 0.000499898, -0.004523209],
        [-0.018526394, -0.001581128, -0.011018308],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ObjectCollisionBox:
    half_extents: np.ndarray
    local_center: np.ndarray


OBJECT_COLLISION_BOXES: Mapping[str, ObjectCollisionBox] = {
    "cube1": ObjectCollisionBox(
        half_extents=np.asarray([0.043363894, 0.043665087, 0.040606263], dtype=np.float64),
        local_center=np.asarray([0.005175592, -0.005187185, 0.001516277], dtype=np.float64),
    ),
    "cube2": ObjectCollisionBox(
        half_extents=np.asarray([0.026036281, 0.025849256, 0.025415827], dtype=np.float64),
        local_center=np.asarray([-0.000047717, -0.003572410, -0.000630386], dtype=np.float64),
    ),
}


def verify_locked_state54_norm_stats(
    norm_stats_dir: Path, *, expected_sha256: str | None = None
) -> tuple[Path, str]:
    """Authenticate norm bytes, population/data contract, and zero-truncation audit."""
    directory = Path(norm_stats_dir)
    norm_path = directory / "norm_stats.json"
    if not norm_path.is_file():
        raise ValueError(f"state54 requires norm_stats.json at {norm_path}")
    actual_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    expected = STATE54_NORM_SHA256 if expected_sha256 is None else str(expected_sha256).lower()
    if expected != STATE54_NORM_SHA256:
        raise ValueError(f"state54 expected norm SHA is not allowlisted: {expected}")
    if actual_sha != expected:
        raise ValueError(f"state54 norm SHA mismatch: expected {expected}, got {actual_sha}")

    contract_path = directory / "data_contract.json"
    if not contract_path.is_file():
        raise ValueError(f"state54 norm requires data_contract.json at {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "norm_stats_sha256": actual_sha,
        "state_contract": STATE_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": 10,
        "action_source": "urdf_target_absolute",
        "row_indices_sha256": POPULATION_ROW_INDICES_SHA256,
        "trajectory_count": POPULATION_TRAJECTORIES,
        "active_frame_count": POPULATION_ACTIVE_FRAMES,
        "action_vector_count": POPULATION_ACTIVE_FRAMES * 10,
        "force_reference_newtons": FORCE_REFERENCE_NEWTONS,
        "source_interval_seconds": SOURCE_INTERVAL_SECONDS,
        "contact_age_clip_seconds": CONTACT_AGE_CLIP_SECONDS,
        "max_token_len": PROFILE_MAX_TOKEN_LEN,
    }
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError(
                f"state54 data contract {key!r} mismatch: expected {value!r}, "
                f"got {contract.get(key)!r}: {contract_path}"
            )

    audit_path = directory / "token_audit.json"
    if not audit_path.is_file():
        raise ValueError(f"state54 norm requires token_audit.json at {audit_path}")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if contract.get("token_audit_sha256") != audit_sha:
        raise ValueError("state54 token audit SHA does not match data_contract.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not (
        audit.get("zero_truncation") is True
        and audit.get("overflow_count") == 0
        and audit.get("audited_active_frames") == POPULATION_ACTIVE_FRAMES
        and audit.get("profile_max_token_len") == PROFILE_MAX_TOKEN_LEN
        and int(audit.get("maximum_token_length", PROFILE_MAX_TOKEN_LEN + 1))
        <= PROFILE_MAX_TOKEN_LEN
        and audit.get("norm_stats_sha256") == actual_sha
        and audit.get("population_row_indices_sha256") == POPULATION_ROW_INDICES_SHA256
    ):
        raise ValueError(f"state54 token audit contract is invalid: {audit_path}")
    return norm_path, actual_sha


def _require_shape(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def axis_angle_to_matrix(rot_aa: np.ndarray) -> np.ndarray:
    """Convert one or a batch of rotation vectors to local-to-world matrices."""
    value = np.asarray(rot_aa, dtype=np.float64)
    if value.shape[-1:] != (3,):
        raise ValueError(f"axis-angle rotations must end in width 3, got {value.shape}")
    flat = value.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1)
    result = np.repeat(np.eye(3, dtype=np.float64)[None], flat.shape[0], axis=0)
    active = theta > 1e-12
    if np.any(active):
        axis = flat[active] / theta[active, None]
        x, y, z = axis.T
        zeros = np.zeros_like(x)
        skew = np.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=1
        ).reshape(-1, 3, 3)
        angles = theta[active]
        sin = np.sin(angles)[:, None, None]
        cos = np.cos(angles)[:, None, None]
        result[active] = np.eye(3)[None] + sin * skew + (1.0 - cos) * (skew @ skew)
    return result.reshape(*value.shape[:-1], 3, 3)


def fingertips_in_collision_box_frame(
    fingertip_world: np.ndarray,
    object_position_world: np.ndarray,
    object_rotation_local_to_world: np.ndarray,
    object_name: str,
) -> np.ndarray:
    """Return scale-normalized collision-box coordinates for five endpoints."""
    tips = _require_shape("fingertip_world", fingertip_world, (5, 3)).astype(np.float64)
    position = _require_shape("object_position_world", object_position_world, (3,)).astype(np.float64)
    rotation = _require_shape(
        "object_rotation_local_to_world", object_rotation_local_to_world, (3, 3)
    ).astype(np.float64)
    try:
        box = OBJECT_COLLISION_BOXES[object_name]
    except KeyError as exc:
        raise ValueError(f"state54 has no collision-box contract for {object_name!r}") from exc
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-6):
        raise ValueError("object rotation is not orthonormal")
    collision_center_world = position + rotation @ box.local_center
    local = (tips - collision_center_world) @ rotation
    return (local / box.half_extents).astype(np.float32)


def aggregate_finger_contact_and_force(
    frame_contacts: Sequence[Mapping[str, Any]], object_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate binary contacts and summed normal-load magnitudes by finger.

    Magnitudes of every Lance ``contact_pairs[].force_normal`` vector are
    summed across records belonging to a finger.  This matches Mode4's sum of
    non-negative ``mj_contactForce(...)[0]`` normal loads across the same
    finger's object-contact pairs.  The state stores ``log1p(newtons)``.
    """
    contacts = np.zeros(5, dtype=np.float32)
    loads = np.zeros(5, dtype=np.float64)
    for record in frame_contacts:
        if record.get("object_name") != object_name:
            continue
        joint_name = str(record.get("joint_name") or "")
        finger_index = next(
            (index for index, finger in enumerate(FINGER_NAMES) if joint_name.startswith(finger)),
            None,
        )
        if finger_index is None:
            continue
        pairs = record.get("contact_pairs")
        if not isinstance(pairs, Sequence) or not pairs:
            raise ValueError(f"contact[{joint_name}] has no contact_pairs")
        contacts[finger_index] = 1.0
        for pair_index, pair in enumerate(pairs):
            force_normal = _require_shape(
                f"contact[{joint_name}].contact_pairs[{pair_index}].force_normal",
                pair.get("force_normal"),
                (3,),
            ).astype(np.float64)
            loads[finger_index] += float(np.linalg.norm(force_normal))
    return contacts, np.log1p(loads).astype(np.float32)


@dataclass
class State54TemporalTracker:
    """Online window-local velocity/contact-duration state shared with Mode4."""

    previous_relative_height: float | None = None
    previous_multifinger_contact: bool = False
    multifinger_contact_age: float = 0.0

    def update(
        self, *, object_z: float, palm_z: float, finger_contacts: np.ndarray
    ) -> tuple[float, float]:
        contacts = _require_shape("finger_contacts", finger_contacts, (5,))
        relative_height = float(object_z) - float(palm_z)
        velocity = (
            0.0
            if self.previous_relative_height is None
            else (relative_height - self.previous_relative_height) / SOURCE_INTERVAL_SECONDS
        )
        multifinger = int(np.count_nonzero(contacts > 0.5)) >= 2
        if self.previous_relative_height is None or not (
            multifinger and self.previous_multifinger_contact
        ):
            self.multifinger_contact_age = 0.0
        else:
            self.multifinger_contact_age = min(
                self.multifinger_contact_age + SOURCE_INTERVAL_SECONDS,
                CONTACT_AGE_CLIP_SECONDS,
            )
        self.previous_relative_height = relative_height
        self.previous_multifinger_contact = multifinger
        return float(velocity), float(self.multifinger_contact_age if multifinger else 0.0)


def contact_age_seconds(contacts: np.ndarray) -> np.ndarray:
    """Compute selected-window-local >=2-finger stable-contact duration."""
    values = np.asarray(contacts, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"contacts must have shape (T,5), got {values.shape}")
    result = np.zeros(values.shape[0], dtype=np.float32)
    tracker = State54TemporalTracker()
    for index, row in enumerate(values):
        _, result[index] = tracker.update(object_z=0.0, palm_z=0.0, finger_contacts=row)
    return result


def build_state54(
    *,
    hand_qpos: np.ndarray,
    finger_contacts: np.ndarray,
    lift_height: float,
    fingertip_collision_box_xyz: np.ndarray,
    finger_log1p_force: np.ndarray,
    relative_vertical_velocity: float,
    multifinger_contact_age: float,
) -> np.ndarray:
    """Assemble one raw state vector, rejecting absent or non-finite features."""
    state = np.empty(STATE_DIM, dtype=np.float32)
    state[:26] = _require_shape("hand_qpos", hand_qpos, (26,))
    state[FINGER_CONTACT_SLICE] = _require_shape("finger_contacts", finger_contacts, (5,))
    state[LIFT_HEIGHT_INDEX] = float(lift_height)
    state[FINGERTIP_OBJECT_SLICE] = _require_shape(
        "fingertip_collision_box_xyz", fingertip_collision_box_xyz, (5, 3)
    ).reshape(-1)
    state[FINGER_FORCE_SLICE] = _require_shape("finger_log1p_force", finger_log1p_force, (5,))
    state[RELATIVE_VERTICAL_VELOCITY_INDEX] = float(relative_vertical_velocity)
    state[MULTIFINGER_CONTACT_AGE_INDEX] = float(multifinger_contact_age)
    if not np.all(np.isfinite(state)):
        raise ValueError("assembled state54 contains non-finite values")
    return state


def build_state54_window(
    *,
    hand_qpos: np.ndarray,
    mano_joint_pos: np.ndarray,
    frame_contacts: Sequence[Sequence[Mapping[str, Any]]],
    object_name: str,
    object_position_world: np.ndarray,
    object_rotation_aa: np.ndarray,
    window_start: int,
    window_end: int,
) -> np.ndarray:
    """Build every raw 54D state in one inclusive selected source window."""
    qpos = np.asarray(hand_qpos, dtype=np.float32)
    joints = np.asarray(mano_joint_pos, dtype=np.float64)
    positions = np.asarray(object_position_world, dtype=np.float64)
    rotations_aa = np.asarray(object_rotation_aa, dtype=np.float64)
    frame_count = qpos.shape[0]
    if qpos.shape != (frame_count, 26):
        raise ValueError(f"hand_qpos must have shape (T,26), got {qpos.shape}")
    if joints.shape != (frame_count, 21, 3):
        raise ValueError(f"mano_joint_pos must have shape (T,21,3), got {joints.shape}")
    if positions.shape != (frame_count, 3) or rotations_aa.shape != (frame_count, 3):
        raise ValueError("object pose arrays must both have shape (T,3)")
    if len(frame_contacts) != frame_count:
        raise ValueError("contact frame count does not match qpos")
    start, end = int(window_start), int(window_end)
    if start < 0 or end < start or end >= frame_count:
        raise ValueError(f"invalid inclusive window [{start},{end}] for {frame_count} frames")
    sl = slice(start, end + 1)
    count = end - start + 1
    object_rotations = axis_angle_to_matrix(rotations_aa[sl])
    tip_world = joints[sl][:, FINGERTIP_JOINT_INDICES, :]
    tip_features = np.empty((count, 5, 3), dtype=np.float32)
    contact_features = np.empty((count, 5), dtype=np.float32)
    force_features = np.empty((count, 5), dtype=np.float32)
    for local_index, source_index in enumerate(range(start, end + 1)):
        tip_features[local_index] = fingertips_in_collision_box_frame(
            tip_world[local_index], positions[source_index], object_rotations[local_index], object_name
        )
        contact_features[local_index], force_features[local_index] = aggregate_finger_contact_and_force(
            frame_contacts[source_index], object_name
        )
    lift = positions[sl, 2] - positions[0, 2]
    # Palm is the qpos root translation in both training and Mode4.  The shared
    # tracker resets at selected-window start.
    tracker = State54TemporalTracker()
    states = np.empty((count, STATE_DIM), dtype=np.float32)
    for index in range(count):
        velocity, age = tracker.update(
            object_z=positions[start + index, 2],
            palm_z=qpos[start + index, 2],
            finger_contacts=contact_features[index],
        )
        states[index] = build_state54(
            hand_qpos=qpos[start + index],
            finger_contacts=contact_features[index],
            lift_height=lift[index],
            fingertip_collision_box_xyz=tip_features[index],
            finger_log1p_force=force_features[index],
            relative_vertical_velocity=velocity,
            multifinger_contact_age=age,
        )
    return states


def fingertip_world_from_mujoco(model: Any, data: Any) -> np.ndarray:
    """Compute the five true endpoints from audited distal-body offsets."""
    import mujoco

    tips = np.empty((5, 3), dtype=np.float64)
    for index, (body_name, offset) in enumerate(
        zip(FINGERTIP_DISTAL_BODIES, FINGERTIP_LOCAL_OFFSETS, strict=True)
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing distal body {body_name!r}")
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        tips[index] = np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ offset
    return tips.astype(np.float32)


class ManoFingertipFK:
    """Thread-local MuJoCo FK used only to couple StateAug qpos and fingertips."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _scene(self) -> tuple[Any, Any, list[int], Any]:
        cached = getattr(self._local, "scene", None)
        if cached is None:
            from scripts.eval import mano_physics_core as physics

            tmp, model, data, _, _, _, hand_addrs, _, _ = physics.make_scene(
                "cube1", 16, 16, physics=False, create_renderer=False
            )
            cached = (model, data, hand_addrs, tmp)
            self._local.scene = cached
        return cached

    def __call__(self, hand_qpos: np.ndarray) -> np.ndarray:
        import mujoco

        qpos = _require_shape("augmented hand_qpos", hand_qpos, (26,)).astype(np.float64)
        model, data, hand_addrs, _tmp = self._scene()
        data.qpos[hand_addrs] = qpos
        mujoco.mj_forward(model, data)
        return fingertip_world_from_mujoco(model, data)
