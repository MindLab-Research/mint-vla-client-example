"""Phase-aware State45 observation layered on immutable native State41.

state45[0:41]  unchanged mano_state41_native_sim_28d_v1
state45[41]    causal peak object lift so far (metres)
state45[42]    stable-lift latch (0/1)
state45[43]    causal filtered object vertical velocity (metres/second)
state45[44]    task phase: 0=ACQUIRE, 1=PLACE, 2=DONE

action remains the State41 Action32 target28+pad4 contract.
"""
from __future__ import annotations

import numpy as np

from scripts.mano_state41_contract import (
    ACTION32_CONTRACT_ID,
    ACTION_DIM,
    CONTACT_SLICE,
    FLOOR_SUPPORT_INDEX,
    LIFT_HEIGHT_INDEX,
    MANO_28D_DELTA_MASK_SEGMENTS,
    STATE41_CONTRACT_ID,
    STATE_DIM as STATE41_DIM,
)
from scripts.mano_task_phase import (
    PHASE_TRACKER_CONTRACT_ID,
    ManoTaskPhaseTracker,
    PhaseObservation,
    PhaseTrackerConfig,
    phase_feature_sequence,
)


STATE45_CONTRACT_ID = "mano_state45_phase_native_sim_28d_v1"
STATE_DIM = 45
PHASE_FEATURE_SLICE = slice(41, 45)
PEAK_LIFT_INDEX = 41
STABLE_LIFT_INDEX = 42
OBJECT_VERTICAL_VELOCITY_INDEX = 43
TASK_PHASE_INDEX = 44
FULL_TASK_PROMPT_TEMPLATE = (
    "pick up the {object} using gesture {gesture}, then place it back on the table"
)


def append_phase_to_state41_sequence(
    state41: np.ndarray,
    *,
    config: PhaseTrackerConfig | None = None,
) -> np.ndarray:
    source = np.asarray(state41, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != STATE41_DIM:
        raise ValueError(f"state41 sequence must have shape [T,{STATE41_DIM}], got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("state41 sequence contains non-finite values")
    features = phase_feature_sequence(
        object_lift_m=source[:, LIFT_HEIGHT_INDEX],
        hand_object_contact=np.any(source[:, CONTACT_SLICE] > 0.5, axis=1),
        object_floor_contact=source[:, FLOOR_SUPPORT_INDEX] > 0.5,
        config=config,
    )
    result = np.concatenate([source, features], axis=1, dtype=np.float32)
    if result.shape != (len(source), STATE_DIM) or not np.isfinite(result).all():
        raise FloatingPointError(f"invalid State45 sequence {result.shape}")
    return result


def assemble_live_state45(
    state41: np.ndarray,
    phase: PhaseObservation,
) -> np.ndarray:
    source = np.asarray(state41, dtype=np.float32)
    if source.shape != (STATE41_DIM,) or not np.isfinite(source).all():
        raise ValueError(f"live state41 must have shape {(STATE41_DIM,)}, got {source.shape}")
    result = np.empty(STATE_DIM, dtype=np.float32)
    result[:STATE41_DIM] = source
    result[PHASE_FEATURE_SLICE] = phase.as_array()
    if not np.isfinite(result).all():
        raise FloatingPointError("live State45 contains non-finite values")
    return result


__all__ = [
    "ACTION32_CONTRACT_ID",
    "ACTION_DIM",
    "FULL_TASK_PROMPT_TEMPLATE",
    "MANO_28D_DELTA_MASK_SEGMENTS",
    "ManoTaskPhaseTracker",
    "OBJECT_VERTICAL_VELOCITY_INDEX",
    "PEAK_LIFT_INDEX",
    "PHASE_FEATURE_SLICE",
    "PHASE_TRACKER_CONTRACT_ID",
    "PhaseObservation",
    "PhaseTrackerConfig",
    "STABLE_LIFT_INDEX",
    "STATE41_CONTRACT_ID",
    "STATE45_CONTRACT_ID",
    "STATE_DIM",
    "TASK_PHASE_INDEX",
    "append_phase_to_state41_sequence",
    "assemble_live_state45",
]
