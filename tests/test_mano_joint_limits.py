from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval.mano_joint_limits import clip_hand_state, compiled_hand_joint_limits


def model(*, limited=(True, False, True), ranges=((-1, 1), (-9, 9), (0, 2))):
    return SimpleNamespace(
        jnt_type=np.asarray([3, 2, 3]),
        jnt_limited=np.asarray(limited),
        jnt_range=np.asarray(ranges, dtype=np.float64),
    )


def test_clips_below_inside_and_above_without_mutating_input() -> None:
    source = np.asarray([-3.0, 123.0, 5.0], dtype=np.float32)
    original = source.copy()
    limits = compiled_hand_joint_limits(model(), ("below", "unlimited", "above"), (0, 1, 2))

    bounded, diagnostic = clip_hand_state(source, limits)

    np.testing.assert_allclose(bounded, [-1.0, 123.0, 2.0])
    np.testing.assert_array_equal(source, original)
    assert diagnostic["limited_joint_count"] == 2
    assert diagnostic["clipped_values"] == 2
    assert diagnostic["max_correction"] == 3.0
    assert diagnostic["per_joint"]["below"]["range"] == [-1.0, 1.0]


def test_inside_limit_and_unlimited_joint_are_unchanged() -> None:
    limits = compiled_hand_joint_limits(model(), ("a", "b", "c"), (0, 1, 2))
    bounded, diagnostic = clip_hand_state(np.asarray([0.5, 999.0, 1.5]), limits)
    np.testing.assert_allclose(bounded, [0.5, 999.0, 1.5])
    assert diagnostic["clipped_values"] == 0


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (np.zeros((3, 1)), "shape"),
        (np.zeros(2), "shape"),
        (np.asarray([0.0, np.nan, 1.0]), "finite"),
    ],
)
def test_rejects_invalid_state_shape_or_values(state: np.ndarray, message: str) -> None:
    limits = compiled_hand_joint_limits(model(), ("a", "b", "c"), (0, 1, 2))
    with pytest.raises(ValueError, match=message):
        clip_hand_state(state, limits)


def test_rejects_invalid_limited_range_and_non_scalar_joint() -> None:
    with pytest.raises(ValueError, match="invalid compiled ranges"):
        compiled_hand_joint_limits(model(ranges=((1, -1), (-9, 9), (0, 2))), ("a", "b", "c"), (0, 1, 2))
    invalid_type_model = model()
    invalid_type_model.jnt_type[1] = 1
    with pytest.raises(ValueError, match="cannot represent"):
        compiled_hand_joint_limits(invalid_type_model, ("a", "ball", "c"), (0, 1, 2))
