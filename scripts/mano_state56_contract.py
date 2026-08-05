"""Authenticated native28 MANO State56 observation/action contract.

Layout::

    [0:28]   native ManoRL qpos28
    [28:33]  target-object binary finger contact5
    [33]     object body-origin lift relative to source frame0
    [34:49]  five fingertip centres in normalized object collision-AABB XYZ
    [49:54]  log1p summed normal-force magnitude per finger
    [54]     object-minus-palm world vertical velocity
    [55]     window-local consecutive >=2-finger contact age, clipped to1s

Action32 is absolute native target28 followed by four exact zeros. The model
residual mask is ``(3,-3,22,-4)`` and the action horizon is10.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STATE_CONTRACT_ID = "mano_object_dynamics_state56_native28_v1"
ACTION_CONTRACT_ID = "mano_native28_target32_pad4_v1"
PROFILE_ID = "pi05_action_lora_r16_state56_28dof_v1"
MODEL_ID = "openpi/pi05-action-lora-r16-state56-28dof-finetune"
STATE_DIM = 56
ACTION_DIM = 32
ACTION_PHYSICAL_DIM = 28
ACTION_PADDING_DIM = 4
ACTION_HORIZON = 10
ACTION_DELTA_MASK = (3, -3, 22, -4)
HAND_QPOS_DIM = 28
SOURCE_INTERVAL_SECONDS = 0.005
CONTACT_AGE_CLIP_SECONDS = 1.0
FORCE_REFERENCE_NEWTONS = 50.0
PROFILE_MAX_TOKEN_LEN = 256
FINGER_NAMES = ("index", "thumb", "ring", "middle", "pinky")
FINGER_CONTACT_SLICE = slice(28, 33)
LIFT_HEIGHT_INDEX = 33
FINGERTIP_OBJECT_SLICE = slice(34, 49)
FINGER_FORCE_SLICE = slice(49, 54)
RELATIVE_VERTICAL_VELOCITY_INDEX = 54
MULTIFINGER_CONTACT_AGE_INDEX = 55
GEOMETRY_CONTRACT_FILENAME = "state56_native28_geometry_contract.json"
GEOMETRY_CONTRACT_SHA256 = "e2b029d9adc24925b387c3773f513e4c8fafbf08b2d7ffa997aa76875de2f07c"
EXPECTED_MANORL_COMMIT = "e17f0122decddffc348ec10d0ed42552a0540e1b"
EXPECTED_ASSET_SOURCE_COMMIT = "e7910212e54367008ecb7484e5e9354e822de03e"
CONTACT_SEMANTICS = "state41_native_replay_pair_presence_and_normal_force_norm_v1"


@dataclass(frozen=True)
class ObjectCollisionBox:
    half_extents: np.ndarray
    local_center: np.ndarray


def _load_geometry_contract() -> tuple[Mapping[str, ObjectCollisionBox], tuple[str, ...], np.ndarray]:
    path = Path(__file__).with_name(GEOMETRY_CONTRACT_FILENAME)
    if not path.is_file():
        raise RuntimeError(f"State56 geometry contract is missing: {path}")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != GEOMETRY_CONTRACT_SHA256:
        raise RuntimeError(
            f"State56 geometry contract SHA mismatch: expected {GEOMETRY_CONTRACT_SHA256}, got {actual_sha}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "contract": "mano_native28_collision_aabb_v1",
        "manorl_commit": EXPECTED_MANORL_COMMIT,
        "asset_source_commit": EXPECTED_ASSET_SOURCE_COMMIT,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"State56 geometry contract {key} mismatch: expected {expected!r}, got {payload.get(key)!r}"
            )
    boxes: dict[str, ObjectCollisionBox] = {}
    for object_name, values in payload.get("objects", {}).items():
        half = np.asarray(values.get("half_extents"), dtype=np.float64)
        center = np.asarray(values.get("local_center"), dtype=np.float64)
        if half.shape != (3,) or center.shape != (3,) or not np.all(np.isfinite(half)) or not np.all(np.isfinite(center)):
            raise RuntimeError(f"invalid State56 collision box for {object_name!r}")
        if np.any(half <= 0):
            raise RuntimeError(f"non-positive State56 collision-box half extent for {object_name!r}")
        boxes[str(object_name)] = ObjectCollisionBox(half_extents=half, local_center=center)
    if len(boxes) != 11:
        raise RuntimeError(f"State56 geometry contract must contain11 objects, got {len(boxes)}")
    fingertips = payload.get("fingertips")
    if not isinstance(fingertips, list) or len(fingertips) != 5:
        raise RuntimeError("State56 geometry contract must contain five fingertip records")
    if tuple(item.get("finger") for item in fingertips) != FINGER_NAMES:
        raise RuntimeError("State56 fingertip order mismatch")
    body_names = tuple(str(item["body"]) for item in fingertips)
    offsets = np.asarray([item["local_offset"] for item in fingertips], dtype=np.float64)
    if offsets.shape != (5, 3) or not np.all(np.isfinite(offsets)):
        raise RuntimeError("invalid State56 fingertip local offsets")
    return boxes, body_names, offsets


OBJECT_COLLISION_BOXES, FINGERTIP_DISTAL_BODIES, FINGERTIP_LOCAL_OFFSETS = _load_geometry_contract()


def _require_shape(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = _require_shape("object quaternion", quaternion, (4,)).astype(np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError(f"object quaternion is not normalized: norm={norm}")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def fingertips_in_collision_box_frame(
    fingertip_world: np.ndarray,
    object_position_world: np.ndarray,
    object_rotation_local_to_world: np.ndarray,
    object_name: str,
) -> np.ndarray:
    tips = _require_shape("fingertip_world", fingertip_world, (5, 3)).astype(np.float64)
    position = _require_shape("object_position_world", object_position_world, (3,)).astype(np.float64)
    rotation = _require_shape(
        "object_rotation_local_to_world", object_rotation_local_to_world, (3, 3)
    ).astype(np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2e-6):
        raise ValueError("object rotation is not orthonormal")
    try:
        box = OBJECT_COLLISION_BOXES[object_name]
    except KeyError as exc:
        raise ValueError(f"State56 has no collision-box contract for {object_name!r}") from exc
    collision_center_world = position + rotation @ box.local_center
    object_local = (tips - collision_center_world) @ rotation
    return (object_local / box.half_extents).astype(np.float32)


def aggregate_state41_contact_frame(
    frame_contact: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return contact5 and log1p summed normal load from one State41 frame."""
    contacts = _require_shape(
        "contact.finger_contacts", frame_contact.get("finger_contacts"), (5,)
    ).astype(np.float32)
    if np.any((contacts != 0.0) & (contacts != 1.0)):
        raise ValueError("State56 source finger contacts must be binary")
    pairs = frame_contact.get("pairs")
    if not isinstance(pairs, Sequence):
        raise ValueError("State56 source contact.pairs must be a sequence")
    if int(frame_contact.get("pair_count", -1)) != len(pairs):
        raise ValueError("State56 source contact pair_count mismatch")
    loads = np.zeros(5, dtype=np.float64)
    pair_fingers: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        finger = str(pair.get("finger") or "")
        if finger == "palm":
            continue
        if finger not in FINGER_NAMES:
            raise ValueError(f"unknown State56 contact finger {finger!r} at pair {pair_index}")
        magnitude = float(pair.get("normal_force_norm"))
        if not np.isfinite(magnitude) or magnitude < 0:
            raise ValueError(f"invalid normal force magnitude at pair {pair_index}: {magnitude}")
        loads[FINGER_NAMES.index(finger)] += magnitude
        pair_fingers.add(finger)
    expected_pair_fingers = {
        FINGER_NAMES[index] for index in range(5) if contacts[index] > 0.5
    }
    if pair_fingers != expected_pair_fingers:
        raise ValueError(
            "State56 contact flags/pairs disagree: "
            f"flags={sorted(expected_pair_fingers)} pairs={sorted(pair_fingers)}"
        )
    return contacts, np.log1p(loads).astype(np.float32)


@dataclass
class State56TemporalTracker:
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


def build_action32(target28: np.ndarray) -> np.ndarray:
    target = _require_shape("native28 target", target28, (ACTION_PHYSICAL_DIM,)).astype(np.float32)
    result = np.zeros(ACTION_DIM, dtype=np.float32)
    result[:ACTION_PHYSICAL_DIM] = target
    return result


def extract_target28(action32: np.ndarray) -> np.ndarray:
    action = _require_shape("Action32", action32, (ACTION_DIM,)).astype(np.float32)
    if not np.array_equal(action[ACTION_PHYSICAL_DIM:], np.zeros(ACTION_PADDING_DIM, dtype=np.float32)):
        raise ValueError("State56 Action32 pad4 must be exactly zero")
    return action[:ACTION_PHYSICAL_DIM].copy()


def build_state56(
    *,
    hand_qpos: np.ndarray,
    finger_contacts: np.ndarray,
    lift_height: float,
    fingertip_collision_box_xyz: np.ndarray,
    finger_log1p_force: np.ndarray,
    relative_vertical_velocity: float,
    multifinger_contact_age: float,
) -> np.ndarray:
    state = np.empty(STATE_DIM, dtype=np.float32)
    state[:HAND_QPOS_DIM] = _require_shape("hand_qpos", hand_qpos, (HAND_QPOS_DIM,))
    state[FINGER_CONTACT_SLICE] = _require_shape("finger_contacts", finger_contacts, (5,))
    state[LIFT_HEIGHT_INDEX] = float(lift_height)
    state[FINGERTIP_OBJECT_SLICE] = _require_shape(
        "fingertip_collision_box_xyz", fingertip_collision_box_xyz, (5, 3)
    ).reshape(-1)
    state[FINGER_FORCE_SLICE] = _require_shape("finger_log1p_force", finger_log1p_force, (5,))
    state[RELATIVE_VERTICAL_VELOCITY_INDEX] = float(relative_vertical_velocity)
    state[MULTIFINGER_CONTACT_AGE_INDEX] = float(multifinger_contact_age)
    if not np.all(np.isfinite(state)):
        raise ValueError("assembled State56 contains non-finite values")
    return state


def build_state56_window_from_features(
    *,
    hand_qpos: np.ndarray,
    finger_contacts: np.ndarray,
    finger_log1p_force: np.ndarray,
    fingertip_collision_box_xyz: np.ndarray,
    object_position_world: np.ndarray,
    window_start: int,
    window_end: int,
) -> np.ndarray:
    qpos = np.asarray(hand_qpos, dtype=np.float32)
    contacts = np.asarray(finger_contacts, dtype=np.float32)
    forces = np.asarray(finger_log1p_force, dtype=np.float32)
    tips = np.asarray(fingertip_collision_box_xyz, dtype=np.float32)
    positions = np.asarray(object_position_world, dtype=np.float64)
    frame_count = qpos.shape[0]
    expected = {
        "hand_qpos": (frame_count, 28),
        "finger_contacts": (frame_count, 5),
        "finger_log1p_force": (frame_count, 5),
        "fingertip_collision_box_xyz": (frame_count, 5, 3),
        "object_position_world": (frame_count, 3),
    }
    actual = {
        "hand_qpos": qpos.shape,
        "finger_contacts": contacts.shape,
        "finger_log1p_force": forces.shape,
        "fingertip_collision_box_xyz": tips.shape,
        "object_position_world": positions.shape,
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(f"State56 {name} must have shape {expected_shape}, got {actual[name]}")
    if not all(np.all(np.isfinite(value)) for value in (qpos, contacts, forces, tips, positions)):
        raise ValueError("State56 source features must all be finite")
    if np.any((contacts != 0.0) & (contacts != 1.0)):
        raise ValueError("State56 contacts must be binary")
    if np.any(forces < 0.0):
        raise ValueError("State56 log1p forces must be non-negative")
    start, end = int(window_start), int(window_end)
    if start < 0 or end < start or end >= frame_count:
        raise ValueError(f"invalid inclusive window [{start},{end}] for {frame_count} frames")
    lift = positions[start : end + 1, 2] - positions[0, 2]
    tracker = State56TemporalTracker()
    states = np.empty((end - start + 1, STATE_DIM), dtype=np.float32)
    for local_index, source_index in enumerate(range(start, end + 1)):
        velocity, age = tracker.update(
            object_z=positions[source_index, 2],
            palm_z=qpos[source_index, 2],
            finger_contacts=contacts[source_index],
        )
        states[local_index] = build_state56(
            hand_qpos=qpos[source_index],
            finger_contacts=contacts[source_index],
            lift_height=lift[local_index],
            fingertip_collision_box_xyz=tips[source_index],
            finger_log1p_force=forces[source_index],
            relative_vertical_velocity=velocity,
            multifinger_contact_age=age,
        )
    return states


def fingertip_world_from_mujoco(model: Any, data: Any) -> np.ndarray:
    import mujoco

    tips = np.empty((5, 3), dtype=np.float64)
    for index, (body_name, offset) in enumerate(
        zip(FINGERTIP_DISTAL_BODIES, FINGERTIP_LOCAL_OFFSETS, strict=True)
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"native28 model is missing distal body {body_name!r}")
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        tips[index] = np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ offset
    return tips.astype(np.float32)


def verify_locked_state56_norm_stats(
    norm_stats_dir: Path,
    *,
    expected_sha256: str,
    data_contract_path: Path,
) -> tuple[Path, str]:
    directory = Path(norm_stats_dir).expanduser().resolve()
    norm_path = directory / "norm_stats.json"
    if not norm_path.is_file():
        raise ValueError(f"State56 requires norm_stats.json at {norm_path}")
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"invalid State56 expected norm SHA: {expected!r}")
    actual = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"State56 norm SHA mismatch: expected {expected}, got {actual}")
    contract_path = Path(data_contract_path).expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"State56 requires explicit data contract at {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "state_contract": STATE_CONTRACT_ID,
        "action_contract": ACTION_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_source": "urdf_target_absolute",
        "action_physical_dim": ACTION_PHYSICAL_DIM,
        "action_padding_dim": ACTION_PADDING_DIM,
        "action_delta_mask": list(ACTION_DELTA_MASK),
        "norm_stats_sha256": actual,
        "geometry_contract_sha256": GEOMETRY_CONTRACT_SHA256,
        "source_interval_seconds": SOURCE_INTERVAL_SECONDS,
        "contact_age_clip_seconds": CONTACT_AGE_CLIP_SECONDS,
        "max_token_len": PROFILE_MAX_TOKEN_LEN,
        "train_trajectory_count": 4613,
        "validation_trajectory_count": 243,
        "held_out_trajectory_count": 0,
    }
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError(
                f"State56 data contract {key!r} mismatch: expected {value!r}, got {contract.get(key)!r}"
            )
    token_path_value = contract.get("token_audit")
    if not token_path_value:
        raise ValueError("State56 data contract must name token_audit")
    token_path = Path(token_path_value).expanduser().resolve()
    if not token_path.is_file():
        raise ValueError(f"State56 token audit is missing: {token_path}")
    token_sha = hashlib.sha256(token_path.read_bytes()).hexdigest()
    if contract.get("token_audit_sha256") != token_sha:
        raise ValueError("State56 token audit SHA mismatch")
    audit = json.loads(token_path.read_text(encoding="utf-8"))
    if not (
        audit.get("zero_truncation") is True
        and audit.get("overflow_count") == 0
        and int(audit.get("maximum_token_length", PROFILE_MAX_TOKEN_LEN + 1)) <= PROFILE_MAX_TOKEN_LEN
        and audit.get("norm_stats_sha256") == actual
    ):
        raise ValueError("State56 clean token audit is invalid")
    augmentation = audit.get("augmentation")
    if not (
        isinstance(augmentation, dict)
        and augmentation.get("zero_truncation") is True
        and augmentation.get("overflow_count") == 0
        and int(augmentation.get("maximum_token_length", PROFILE_MAX_TOKEN_LEN + 1)) <= PROFILE_MAX_TOKEN_LEN
        and augmentation.get("seed") == 43
        and float(augmentation.get("state_noise_std", -1.0)) == 0.05
        and float(augmentation.get("target_noise_std", -1.0)) == 0.0
    ):
        raise ValueError("State56 augmented token audit is invalid")
    return norm_path, actual
