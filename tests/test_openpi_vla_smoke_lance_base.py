from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.train.openpi_vla_smoke_lance_base import LanceViewpi05Dataset


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_getitem_copies_resident_numpy_action_window() -> None:
    dataset = object.__new__(LanceViewpi05Dataset)
    dataset._index = [(0, 0)]
    dataset._row_windows = {0: SimpleNamespace(end_frame=11)}
    dataset._action_horizon = 10
    dataset._extended_state = False

    resident_actions = np.arange(12 * 32, dtype=np.float32).reshape(12, 32)
    original = resident_actions.copy()
    image = _jpeg()
    row = {
        "actions": resident_actions,
        "state": np.zeros((12, 32), dtype=np.float32),
        "image": [image] * 12,
        "wrist_image": [image] * 12,
        "prompt": "pick up the cylinder1 using gesture 01",
    }
    dataset._get_row = lambda _: row

    sample = dataset[0]
    sample["actions"][:, :26] -= 1.0  # mirrors the in-place DeltaActions transform

    np.testing.assert_array_equal(resident_actions, original)
    assert not np.shares_memory(sample["actions"], resident_actions)
