from __future__ import annotations

import sys

import numpy as np

from scripts.eval import infer_mano_mode4_state45 as state45
from scripts.mano_state41_contract import STATE_DIM as STATE41_DIM
from scripts.mano_state45_contract import STATE_DIM as STATE45_DIM
from scripts.mano_task_phase import ManoTaskPhaseTracker, TaskPhase


def _required_argv() -> list[str]:
    return [
        "infer_mano_mode4_state45.py",
        "--base-url", "http://127.0.0.1:1",
        "--model", state45.MODEL,
        "--model-path", "mint://sampler",
        "--owner-id", "owner",
        "--lance-dataset", "dataset.lance",
        "--row-indices", "7",
        "--normalization-row-indices", "7",
        "--state-contract", "state45",
        "--norm-stats-dir", "norm",
        "--norm-sha-expected", "0" * 64,
        "--output-dir", "output",
        "--language-conditioning", "gesture",
        "--contact-window-manifest", "windows.json",
    ]


def test_parse_defaults_to_persistent_15_seconds_and_stride1(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _required_argv())
    args = state45.parse_args()
    assert args.frame_window == "persistent_task"
    assert args.max_control_seconds == 15.0
    assert args.chunk_stride == 1
    assert args.forced_first_failure is False
    assert args.forced_retry_config.trigger_lift_m == 0.03
    assert args.forced_retry_config.floor_contact_frames == 10
    assert args.forced_retry_config.no_hand_contact_frames == 20
    assert args.forced_retry_config.max_forced_release_frames == 150


def test_parse_accepts_explicit_forced_first_failure_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _required_argv()
        + [
            "--forced-first-failure",
            "--forced-release-trigger-lift-m", "0.025",
            "--forced-release-floor-frames", "8",
            "--forced-release-no-contact-frames", "16",
            "--forced-release-max-frames", "120",
        ],
    )
    args = state45.parse_args()
    assert args.forced_first_failure is True
    assert args.forced_retry_config.trigger_lift_m == 0.025
    assert args.forced_retry_config.floor_contact_frames == 8
    assert args.forced_retry_config.no_hand_contact_frames == 16
    assert args.forced_retry_config.max_forced_release_frames == 120


def test_parse_rejects_forced_release_trigger_at_stable_lift(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _required_argv() + ["--forced-release-trigger-lift-m", "0.05"],
    )
    with np.testing.assert_raises_regex(ValueError, "below"):
        state45.parse_args()


def test_persistent_termination_prioritizes_done_then_timeout() -> None:
    assert state45.persistent_termination_reason(
        phase=TaskPhase.ACQUIRE, control_steps=2999, max_control_frames=3000
    ) is None
    assert state45.persistent_termination_reason(
        phase=TaskPhase.ACQUIRE, control_steps=3000, max_control_frames=3000
    ) == "timeout"
    assert state45.persistent_termination_reason(
        phase=TaskPhase.DONE, control_steps=3000, max_control_frames=3000
    ) == "done"


def test_state45_full_task_prompt_is_formal_and_fixed() -> None:
    row = {
        "index": {"object": "cube1", "gesture": "03"},
        "trajectory_metadata": {"object_names": ["cube1"]},
        "prompt": "legacy",
    }
    conditioned = state45.condition_state45_language(row, "gesture")
    assert conditioned["prompt"] == (
        "pick up the cube1 using gesture 03, then place it back on the table"
    )


def test_live_state41_then_state45_has_exact_widths() -> None:
    state41 = state45.assemble_live_state41(
        hand_qpos=np.zeros(28, dtype=np.float32),
        contacts=np.zeros(5, dtype=np.float32),
        object_lift=0.0,
        signed_surface_distances=np.ones(5, dtype=np.float32),
        floor_support=1.0,
        persistence=0.0,
    )
    phase = ManoTaskPhaseTracker().update(
        object_lift_m=0.0,
        hand_object_contact=False,
        object_floor_contact=True,
    )
    combined = state45.assemble_live_state45(state41, phase)
    assert state41.shape == (STATE41_DIM,)
    assert combined.shape == (STATE45_DIM,)
    np.testing.assert_array_equal(combined[:STATE41_DIM], state41)
