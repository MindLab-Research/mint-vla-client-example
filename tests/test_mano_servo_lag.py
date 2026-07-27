import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import mano_servo_lag as lag


def test_fit_first_order_gains_recovers_known_response():
    rng = np.random.default_rng(3)
    q = np.cumsum(rng.normal(scale=0.01, size=(30, lag.HAND_DIM)), axis=0)
    expected = np.linspace(0.01, 0.08, lag.HAND_DIM)
    target = q.copy()
    target[:-1] = q[:-1] + (q[1:] - q[:-1]) / expected
    target[-1] = target[-2]
    gains, raw, count = lag.fit_first_order_gains([(q, target)])
    np.testing.assert_allclose(gains, expected, atol=1e-10)
    np.testing.assert_allclose(raw, expected, atol=1e-10)
    assert count == 29


def test_wrap_euler_target_near_current_preserves_non_euler_coordinates():
    q = np.arange(lag.HAND_DIM, dtype=np.float64) / 10
    target = q + 1
    target[3] = q[3] + 2 * np.pi - 0.2
    wrapped = lag.wrap_euler_target_near_current(target, q)
    assert wrapped[3] == pytest.approx(q[3] - 0.2)
    np.testing.assert_allclose(wrapped[:3], target[:3])
    np.testing.assert_allclose(wrapped[6:], target[6:])


def test_servo_lag_step_uses_shortest_euler_branch():
    q = np.zeros(lag.HAND_DIM); target = np.ones(lag.HAND_DIM); gains = np.full(lag.HAND_DIM, 0.1)
    q[3] = np.pi - 0.01; target[3] = -np.pi + 0.01
    result = lag.servo_lag_step(q, target, gains)
    assert result[3] == pytest.approx(q[3] + 0.002)
    np.testing.assert_allclose(result[6:], 0.1)


def test_load_gain_file_fails_closed_and_reports_sha(tmp_path: Path):
    path = tmp_path / "gains.json"
    payload = {
        "contract_id": lag.CONTRACT_ID,
        "source_interval_seconds": 0.005,
        "row_count": 185,
        "transition_count": 1000,
        "gains": [0.05] * lag.HAND_DIM,
    }
    path.write_text(json.dumps(payload) + "\n")
    gains, loaded, sha = lag.load_gain_file(path)
    np.testing.assert_allclose(gains, 0.05)
    assert loaded == payload
    assert sha == lag.file_sha256(path)

    payload["source_interval_seconds"] = 0.01
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="exactly 0.005"):
        lag.load_gain_file(path)


def test_fit_rejects_missing_or_invalid_trajectories():
    with pytest.raises(ValueError, match="at least one"):
        lag.fit_first_order_gains([])
    with pytest.raises(ValueError, match="aligned"):
        lag.trajectory_error_and_step(np.zeros((1, 26)), np.zeros((1, 26)))
