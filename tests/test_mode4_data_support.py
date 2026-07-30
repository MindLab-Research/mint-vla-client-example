from __future__ import annotations

from scripts.eval import mode4_data_support as support


def make_row(frame_count: int = 6) -> dict:
    return {
        "state": [[0.0] * 32 for _ in range(frame_count)],
        "actions": [[0.0] * 32 for _ in range(frame_count)],
        "image": [b"x"] * frame_count,
        "wrist_image": [b"y"] * frame_count,
        "timestamp": [0.005 * i for i in range(frame_count)],
        "objects": [
            {
                "pos": [[0.0, 0.0, 0.0] for _ in range(frame_count)],
                "rot_aa": [[0.0, 0.0, 0.0] for _ in range(frame_count)],
            }
        ],
        "episode_metadata": {"total_frames": frame_count},
        "trajectory_metadata": {"object_names": ["cube1"]},
    }


def test_contact_window_uses_absolute_manifest_frames():
    row = make_row()
    window = support.resolve_row_window(
        row,
        row_index=943,
        frame_window="contact",
        contact_context_frames=100,
        missing_contact_policy="error",
        manifest_entry={
            "row_index": 943,
            "object_name": "cube1",
            "total_frames": 6,
            "first_contact_frame": 2,
            "last_contact_frame": 3,
            "start_frame": 1,
            "end_frame": 4,
            "status": "contact_window",
            "context_frames": 100,
        },
    )
    assert window is not None
    assert (window.start_frame, window.end_frame, window.frame_count) == (1, 4, 4)
    assert window.first_contact_frame == 2
    assert window.last_contact_frame == 3


def test_full_window_remains_explicit_stress_test():
    row = make_row()
    window = support.resolve_row_window(
        row,
        row_index=943,
        frame_window="full",
        contact_context_frames=100,
        missing_contact_policy="error",
    )
    assert window is not None
    assert (window.start_frame, window.end_frame, window.frame_count) == (0, 5, 6)
