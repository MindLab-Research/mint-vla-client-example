from __future__ import annotations

import numpy as np
import pytest

from scripts.mano_state41_contract import STATE_DIM as STATE41_DIM
from scripts.mano_state45_contract import (
    OBJECT_VERTICAL_VELOCITY_INDEX,
    PEAK_LIFT_INDEX,
    STABLE_LIFT_INDEX,
    STATE_DIM as STATE45_DIM,
    TASK_PHASE_INDEX,
    append_phase_to_state41_sequence,
    assemble_live_state45,
)
from scripts.mano_task_phase import (
    ManoTaskPhaseTracker,
    PhaseTrackerConfig,
    TaskPhase,
    phase_feature_sequence,
)


def _config() -> PhaseTrackerConfig:
    return PhaseTrackerConfig(
        stable_lift_frames=3,
        settle_frames=2,
        release_frames=2,
        vertical_velocity_alpha=1.0,
        settle_velocity_threshold_mps=0.02,
    )


def test_initial_floor_contact_never_completes_task() -> None:
    tracker = ManoTaskPhaseTracker(_config())
    for _ in range(20):
        obs = tracker.update(
            object_lift_m=0.0,
            hand_object_contact=False,
            object_floor_contact=True,
        )
    assert obs.task_phase == TaskPhase.ACQUIRE
    assert obs.stable_lift_achieved == 0.0


def test_failed_lift_stays_acquire_and_peak_is_latched() -> None:
    tracker = ManoTaskPhaseTracker(_config())
    observations = [
        tracker.update(
            object_lift_m=lift,
            hand_object_contact=True,
            object_floor_contact=lift <= 0.0,
        )
        for lift in (0.0, 0.06, 0.04, 0.0)
    ]
    assert all(obs.task_phase == TaskPhase.ACQUIRE for obs in observations)
    assert all(obs.stable_lift_achieved == 0.0 for obs in observations)
    assert observations[-1].peak_lift_so_far == pytest.approx(0.06)


def test_stable_lift_then_recontact_settle_and_release_reaches_done() -> None:
    tracker = ManoTaskPhaseTracker(_config())
    # Three consecutive lifted frames fire the stable latch and enter PLACE.
    tracker.update(object_lift_m=0.0, hand_object_contact=True, object_floor_contact=True)
    for lift in (0.06, 0.06, 0.06):
        obs = tracker.update(
            object_lift_m=lift,
            hand_object_contact=True,
            object_floor_contact=False,
        )
    assert obs.task_phase == TaskPhase.PLACE
    assert obs.stable_lift_achieved == 1.0

    # Recontact alone is insufficient while moving or while the hand holds it.
    obs = tracker.update(
        object_lift_m=0.02,
        hand_object_contact=True,
        object_floor_contact=True,
    )
    assert obs.task_phase == TaskPhase.PLACE

    # Two settled and released frames satisfy both causal counters.
    tracker.update(object_lift_m=0.02, hand_object_contact=False, object_floor_contact=True)
    obs = tracker.update(object_lift_m=0.02, hand_object_contact=False, object_floor_contact=True)
    assert obs.task_phase == TaskPhase.DONE
    assert obs.stable_lift_achieved == 1.0

    # DONE and stable-lift are absorbing even if the object moves again.
    obs = tracker.update(object_lift_m=0.0, hand_object_contact=True, object_floor_contact=False)
    assert obs.task_phase == TaskPhase.DONE
    assert obs.stable_lift_achieved == 1.0


def test_offline_sequence_exactly_matches_online_updates() -> None:
    lift = np.asarray([0.0, 0.06, 0.06, 0.06, 0.02, 0.02, 0.02], dtype=np.float32)
    hand = np.asarray([1, 1, 1, 1, 1, 0, 0], dtype=bool)
    floor = np.asarray([1, 0, 0, 0, 1, 1, 1], dtype=bool)
    config = _config()
    offline = phase_feature_sequence(
        object_lift_m=lift,
        hand_object_contact=hand,
        object_floor_contact=floor,
        config=config,
    )
    tracker = ManoTaskPhaseTracker(config)
    online = np.stack(
        [
            tracker.update(
                object_lift_m=float(lift[i]),
                hand_object_contact=bool(hand[i]),
                object_floor_contact=bool(floor[i]),
            ).as_array()
            for i in range(len(lift))
        ]
    )
    np.testing.assert_array_equal(offline, online)


def test_prefix_features_do_not_depend_on_future_frames() -> None:
    prefix_lift = np.asarray([0.0, 0.01, 0.06], dtype=np.float32)
    prefix_hand = np.asarray([0, 1, 1], dtype=bool)
    prefix_floor = np.asarray([1, 1, 0], dtype=bool)
    config = _config()
    prefix = phase_feature_sequence(
        object_lift_m=prefix_lift,
        hand_object_contact=prefix_hand,
        object_floor_contact=prefix_floor,
        config=config,
    )
    extended = phase_feature_sequence(
        object_lift_m=np.concatenate([prefix_lift, [0.06, 0.06, 0.0]]),
        hand_object_contact=np.concatenate([prefix_hand, [1, 1, 0]]),
        object_floor_contact=np.concatenate([prefix_floor, [0, 0, 1]]),
        config=config,
    )
    np.testing.assert_array_equal(prefix, extended[: len(prefix)])


def test_state45_sequence_preserves_state41_and_appends_phase() -> None:
    state41 = np.zeros((5, STATE41_DIM), dtype=np.float32)
    state41[:, 33] = [0.0, 0.01, 0.02, 0.03, 0.04]
    state41[:, 28] = 1.0
    state41[0, 39] = 1.0
    state45 = append_phase_to_state41_sequence(state41, config=_config())
    assert state45.shape == (5, STATE45_DIM)
    np.testing.assert_array_equal(state45[:, :STATE41_DIM], state41)
    assert np.all(np.diff(state45[:, PEAK_LIFT_INDEX]) >= 0)
    assert set(state45[:, STABLE_LIFT_INDEX]) <= {0.0, 1.0}
    assert set(state45[:, TASK_PHASE_INDEX]) <= {0.0, 1.0, 2.0}
    assert np.isfinite(state45[:, OBJECT_VERTICAL_VELOCITY_INDEX]).all()


def test_live_state45_rejects_wrong_state41_width() -> None:
    tracker = ManoTaskPhaseTracker(_config())
    phase = tracker.update(
        object_lift_m=0.0,
        hand_object_contact=False,
        object_floor_contact=True,
    )
    with pytest.raises(ValueError, match="live state41"):
        assemble_live_state45(np.zeros(STATE41_DIM - 1, dtype=np.float32), phase)
