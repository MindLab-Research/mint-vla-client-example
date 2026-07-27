"""Focused tests for the MANO extended 32-dim state contract."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.mano_state_contract import (
    CONTACT_NEGATIVE,
    CONTACT_POSITIVE,
    CONTACT_SEMANTICS,
    EXPECTED_NORM_SHA256,
    FINGER_NAMES,
    FINGER_STATE_OFFSET,
    HAND_QPOS_DIM,
    LIFT_HEIGHT_INDEX,
    STATE_CONTRACT_ID,
    STATE_DIM,
    aggregate_finger_contacts,
    build_extended_state,
)


class TestContractIdentity:
    def test_v1_identity_and_locked_norm(self):
        assert STATE_CONTRACT_ID == "mano_five_finger_contact_lift_v1"
        assert CONTACT_SEMANTICS == "record_or_keypoint_pair_presence_v1"
        assert EXPECTED_NORM_SHA256 == (
            "507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d"
        )


class TestAggregateFingerContacts:
    def test_no_contacts(self):
        result = aggregate_finger_contacts([], "cube1")
        assert result.shape == (5,)
        assert np.all(result == 0)

    def test_single_finger(self):
        contacts = [{"joint_name": "index2", "object_name": "cube1"}]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert result[0] == 1.0  # index
        assert result[1] == 0.0  # thumb
        assert result[2] == 0.0  # ring
        assert result[3] == 0.0  # middle
        assert result[4] == 0.0  # pinky

    def test_multiple_fingers(self):
        contacts = [
            {"joint_name": "index2", "object_name": "cube1"},
            {"joint_name": "index3", "object_name": "cube1"},
            {"joint_name": "thumb3", "object_name": "cube1"},
        ]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert result[0] == 1.0  # index
        assert result[1] == 1.0  # thumb
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0

    def test_wrong_object_ignored(self):
        contacts = [{"joint_name": "index2", "object_name": "banana"}]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert np.all(result == 0)

    def test_empty_object_name_ignored(self):
        contacts = [{"joint_name": "index2", "object_name": ""}]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert np.all(result == 0)

    def test_palm_ignored(self):
        contacts = [{"joint_name": "palm1", "object_name": "cube1"}]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert np.all(result == 0)

    def test_all_fingers(self):
        contacts = [
            {"joint_name": f"{f}{j}", "object_name": "cube1"}
            for f in FINGER_NAMES for j in [1, 2, 3]
        ]
        result = aggregate_finger_contacts(contacts, "cube1")
        assert np.all(result == 1.0)


class TestBuildExtendedState:
    def test_shape(self):
        state = build_extended_state(
            np.zeros(26, dtype=np.float32), [], "cube1", 0.05, 0.04
        )
        assert state.shape == (STATE_DIM,)

    def test_hand_qpos_preserved(self):
        qpos = np.arange(26, dtype=np.float32) * 0.1
        state = build_extended_state(qpos, [], "cube1", 0.05, 0.04)
        np.testing.assert_allclose(state[:HAND_QPOS_DIM], qpos)

    def test_contacts_at_correct_positions(self):
        contacts = [{"joint_name": "index2", "object_name": "cube1"}]
        state = build_extended_state(
            np.zeros(26, dtype=np.float32), contacts, "cube1", 0.05, 0.04
        )
        assert state[FINGER_STATE_OFFSET] == 1.0  # index
        assert state[FINGER_STATE_OFFSET + 1] == 0.0  # thumb

    def test_lift_height(self):
        state = build_extended_state(
            np.zeros(26, dtype=np.float32), [], "cube1", 0.15, 0.04
        )
        assert state[LIFT_HEIGHT_INDEX] == pytest.approx(0.11, abs=1e-6)

    def test_lift_height_zero_initial(self):
        state = build_extended_state(
            np.zeros(26, dtype=np.float32), [], "cube1", 0.04, 0.04
        )
        assert state[LIFT_HEIGHT_INDEX] == pytest.approx(0.0, abs=1e-6)

    def test_32d_input(self):
        """32-dim input uses first 26 dims."""
        qpos32 = np.zeros(32, dtype=np.float32)
        qpos32[:26] = 1.0
        qpos32[26:] = 99.0  # should be overwritten
        state = build_extended_state(qpos32, [], "cube1", 0.05, 0.04)
        np.testing.assert_allclose(state[:HAND_QPOS_DIM], 1.0)
        assert state[26] == 0.0  # not 99


class TestContactNormalization:
    def test_binary_mapping(self):
        """q01=0, q99=1 maps 0→-1, 1→+1."""
        q01, q99 = 0.0, 1.0
        for raw, expected in [(0.0, CONTACT_NEGATIVE), (1.0, CONTACT_POSITIVE)]:
            norm = (raw - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            assert norm == pytest.approx(expected, abs=1e-3)
