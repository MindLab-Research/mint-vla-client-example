"""MANO 32-dim state input contract v1.

state[0:26]  = MANO hand qpos (unchanged)
state[26]    = index finger contact  (0/1 → -1/+1)
state[27]    = thumb contact
state[28]    = ring finger contact
state[29]    = middle finger contact
state[30]    = pinky contact
state[31]    = object_z[t] - object_z[0] (lift height)

Contact semantics (v1, user final decision):
- Training side: target-object contact record EXISTS in Lance contact[]
- Mode 4 side: contact pair EXISTS between target-object and 16 MANO keypoint
  collision geoms. NO force_norm filtering.
- Palm: participates in 16-keypoint completeness validation but NOT output.
- Finger order: index/thumb/ring/middle/pinky (fixed).

Contact normalization: raw 0/1 → -1/+1 via fixed q01=0/q99=1 mapping
(NOT quantile normalization, which would fail for sparse fingers like pinky).

EXPECTED_NORM_SHA256: SHA256 of the legacy locked 185-row gesture03 norm.
CUBE1_ALL_NORM_SHA256: SHA256 of the locked 1102-row cube1-all norm.
CUBE1_CUBE2_ALL_NORM_SHA256: SHA256 of the locked 1997-row combined norm.
Mode 4 extended-state MUST verify norm_stats.json is allowlisted and any new
population norm has a matching data_contract.json before loading or first query.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

STATE_CONTRACT_ID = "mano_five_finger_contact_lift_v1"
EXPECTED_NORM_SHA256 = "507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d"
CUBE1_ALL_NORM_SHA256 = "4d7ee78f34c293c4a6023a8980a0c8a614eae6f0c63b889984a5f9a45ce0a747"
CUBE1_CUBE2_ALL_NORM_SHA256 = "4f91eca8ee91d53426ea07faf28873ab98c3761ecb84d6374f4c0c439d51069a"
CONTACT_SEMANTICS = "record_or_keypoint_pair_presence_v1"
CONTACT_RULE = (
    "Training: target-object contact record exists in Lance contact[]. "
    "Mode4: contact pair exists between target-object geom and any of 16 MANO "
    "keypoint collision geoms (palm validated but not output). "
    "No force_norm threshold. Finger order: index/thumb/ring/middle/pinky."
)


def verify_locked_norm_stats(
    norm_stats_dir: Path, *, expected_sha256: str | None = None
) -> tuple[Path, str]:
    """Return an allowlisted, contract-authenticated norm path.

    ``expected_sha256`` is an optional consumer-side assertion used by the newer
    Mode4 launcher. It narrows the existing strict population allowlist; it does
    not replace the allowlist or ``data_contract.json`` validation.
    """
    path = Path(norm_stats_dir) / "norm_stats.json"
    if not path.is_file():
        raise ValueError(f"v1 extended-state requires norm_stats.json at {path}")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    supported = {
        EXPECTED_NORM_SHA256,
        CUBE1_ALL_NORM_SHA256,
        CUBE1_CUBE2_ALL_NORM_SHA256,
    }
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"expected norm SHA256 must be 64 hexadecimal characters, got {expected_sha256!r}")
        if expected not in supported:
            raise ValueError(f"expected norm SHA is not allowlisted: {expected}")
        if actual_sha != expected:
            raise ValueError(
                f"v1 extended-state norm SHA mismatch: expected {expected}, got {actual_sha}: {path}"
            )
    elif actual_sha not in supported:
        raise ValueError(
            "v1 extended-state norm SHA mismatch: "
            f"expected one of {sorted(supported)}, got {actual_sha}: {path}"
        )
    # The legacy gesture03 cache predates data_contract.json enforcement. New
    # population-specific norms must authenticate both their bytes and contract.
    if actual_sha != EXPECTED_NORM_SHA256:
        contract_path = Path(norm_stats_dir) / "data_contract.json"
        if not contract_path.is_file():
            raise ValueError(f"population norm requires data_contract.json at {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_fields = {
            "norm_stats_sha256": actual_sha,
            "state_contract": STATE_CONTRACT_ID,
            "extended_state": True,
            "action_source": "urdf_target_absolute",
        }
        for key, expected in expected_fields.items():
            if contract.get(key) != expected:
                raise ValueError(
                    f"population norm contract {key!r} mismatch: "
                    f"expected {expected!r}, got {contract.get(key)!r}: {contract_path}"
                )
    return path, actual_sha


FINGER_NAMES = ("index", "thumb", "ring", "middle", "pinky")
# state[26:31] order matches FINGER_NAMES
FINGER_STATE_OFFSET = 26
LIFT_HEIGHT_INDEX = 31
HAND_QPOS_DIM = 26
STATE_DIM = 32

# Finger contact normalization: fixed binary mapping, not quantile
CONTACT_NEGATIVE = -1.0
CONTACT_POSITIVE = 1.0


def aggregate_finger_contacts(
    frame_contacts: list[dict[str, Any]],
    object_name: str,
) -> np.ndarray:
    """Aggregate per-joint contact records into per-finger binary contacts.

    Args:
        frame_contacts: list of contact dicts for one frame, each with
            'joint_name' and 'object_name'.
        object_name: target object to filter contacts against.

    Returns:
        (5,) float32 array of 0.0/1.0 per finger in FINGER_NAMES order.
    """
    contacts = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    for record in frame_contacts:
        joint_name = record.get("joint_name", "")
        obj_name = record.get("object_name", "")
        if not obj_name or obj_name != object_name:
            continue
        for fi, finger in enumerate(FINGER_NAMES):
            if joint_name.startswith(finger):
                contacts[fi] = 1.0
                break
    return contacts


def build_extended_state(
    hand_qpos: np.ndarray,
    frame_contacts: list[dict[str, Any]],
    object_name: str,
    object_z: float,
    object_z_initial: float,
) -> np.ndarray:
    """Build the 32-dim extended state for one frame.

    Args:
        hand_qpos: (26,) or (32,) hand qpos. If (32,), first 26 dims used.
        frame_contacts: contact records for this frame.
        object_name: target object name.
        object_z: current object center z.
        object_z_initial: initial object center z (frame 0).

    Returns:
        (32,) float32 extended state.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:HAND_QPOS_DIM] = hand_qpos[:HAND_QPOS_DIM]
    contacts = aggregate_finger_contacts(frame_contacts, object_name)
    for fi in range(len(FINGER_NAMES)):
        state[FINGER_STATE_OFFSET + fi] = contacts[fi]
    state[LIFT_HEIGHT_INDEX] = np.float32(object_z - object_z_initial)
    return state


def normalize_extended_state(
    raw_state: np.ndarray,
    qpos_q01: np.ndarray,
    qpos_q99: np.ndarray,
    lift_q01: float,
    lift_q99: float,
) -> np.ndarray:
    """Normalize the 32-dim extended state.

    state[0:26]: quantile normalization with qpos q01/q99.
    state[26:31]: fixed binary mapping 0→-1, 1→+1 (NOT quantile).
    state[31]: quantile normalization with lift q01/q99.
    """
    result = raw_state.copy()
    # hand qpos: quantile norm
    for i in range(HAND_QPOS_DIM):
        rng = qpos_q99[i] - qpos_q01[i]
        if rng > 1e-8:
            result[i] = 2.0 * (raw_state[i] - qpos_q01[i]) / rng - 1.0
        else:
            result[i] = 0.0
    # finger contacts: fixed binary mapping
    for i in range(FINGER_STATE_OFFSET, FINGER_STATE_OFFSET + len(FINGER_NAMES)):
        result[i] = CONTACT_POSITIVE if raw_state[i] > 0.5 else CONTACT_NEGATIVE
    # lift height: quantile norm
    lift_rng = lift_q99 - lift_q01
    if lift_rng > 1e-8:
        result[LIFT_HEIGHT_INDEX] = 2.0 * (raw_state[LIFT_HEIGHT_INDEX] - lift_q01) / lift_rng - 1.0
    else:
        result[LIFT_HEIGHT_INDEX] = 0.0
    return result
