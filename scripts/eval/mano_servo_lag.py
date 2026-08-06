"""Calibrated one-source-step response for kinematic MANO diagnostics.

The policy predicts ``urdf_dof_target`` setpoints.  A kinematic evaluator must
not treat those setpoints as already-achieved qpos.  This module fits and
applies a first-order response from recorded ``urdf_dof_target`` and
``urdf_dof`` trajectories while keeping the historical Mode3 path available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

HAND_DIM = 26
EULER_SLICE = slice(3, 6)
CONTRACT_ID = "mano_urdf_setpoint_first_order_response_v1"
SOURCE_INTERVAL_SECONDS = 0.005


def wrap_angle_error(values: np.ndarray) -> np.ndarray:
    """Map angle differences to the shortest branch in ``[-pi, pi)``."""
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_error_and_step(qpos: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != HAND_DIM or t.shape != q.shape or len(q) < 2:
        raise ValueError(f"qpos/targets must be aligned (T,{HAND_DIM}) arrays with T>=2")
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("qpos/targets must be finite")
    error = t[:-1] - q[:-1]
    step = q[1:] - q[:-1]
    error[:, EULER_SLICE] = wrap_angle_error(error[:, EULER_SLICE])
    step[:, EULER_SLICE] = wrap_angle_error(step[:, EULER_SLICE])
    return error, step


def fit_first_order_gains(trajectories: Iterable[tuple[np.ndarray, np.ndarray]], *,
                          lower: float = 0.0, upper: float = 0.2) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit ``dq = alpha * (target-q)`` independently per DoF through the origin."""
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("gain clipping bounds must satisfy 0 <= lower <= upper <= 1")
    errors, steps = [], []
    for qpos, targets in trajectories:
        error, step = trajectory_error_and_step(qpos, targets)
        errors.append(error); steps.append(step)
    if not errors:
        raise ValueError("at least one trajectory is required")
    error = np.concatenate(errors, axis=0)
    step = np.concatenate(steps, axis=0)
    denominator = np.sum(error * error, axis=0)
    if np.any(denominator <= 0):
        raise ValueError("cannot fit a DoF with zero setpoint error energy")
    raw = np.sum(error * step, axis=0) / denominator
    gains = np.clip(raw, lower, upper)
    if not np.isfinite(gains).all():
        raise ValueError("fitted gains are not finite")
    return gains.astype(np.float64), raw.astype(np.float64), int(error.shape[0])


def wrap_euler_target_near_current(target: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Return an equivalent target whose Euler coordinates use the nearest branch."""
    t = np.asarray(target, dtype=np.float64)
    q = np.asarray(qpos, dtype=np.float64)
    if t.shape != (HAND_DIM,) or q.shape != (HAND_DIM,):
        raise ValueError(f"target and qpos must have shape ({HAND_DIM},)")
    if not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("target and qpos must be finite")
    result = t.copy()
    result[EULER_SLICE] = q[EULER_SLICE] + wrap_angle_error(
        t[EULER_SLICE] - q[EULER_SLICE]
    )
    return result


def servo_lag_step(qpos: np.ndarray, target: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Advance one source interval toward a setpoint using calibrated gains."""
    q = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    a = np.asarray(gains, dtype=np.float64)
    if q.shape != (HAND_DIM,) or t.shape != (HAND_DIM,) or a.shape != (HAND_DIM,):
        raise ValueError(f"qpos, target, and gains must have shape ({HAND_DIM},)")
    if not np.isfinite(q).all() or not np.isfinite(t).all() or not np.isfinite(a).all():
        raise ValueError("servo transition inputs must be finite")
    if np.any((a < 0) | (a > 1)):
        raise ValueError("servo gains must be in [0,1]")
    error = t - q
    error[EULER_SLICE] = wrap_angle_error(error[EULER_SLICE])
    return q + a * error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gain_file(path: Path) -> tuple[np.ndarray, dict, str]:
    p = Path(path)
    payload = json.loads(p.read_text())
    if payload.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"unexpected servo gain contract: {payload.get('contract_id')!r}")
    if float(payload.get("source_interval_seconds", -1)) != SOURCE_INTERVAL_SECONDS:
        raise ValueError("servo gain source interval must be exactly 0.005 seconds")
    gains = np.asarray(payload.get("gains"), dtype=np.float64)
    if gains.shape != (HAND_DIM,) or not np.isfinite(gains).all() or np.any((gains < 0) | (gains > 1)):
        raise ValueError(f"servo gains must be {HAND_DIM} finite values in [0,1]")
    if int(payload.get("row_count", 0)) <= 0 or int(payload.get("transition_count", 0)) <= 0:
        raise ValueError("servo gain provenance is missing row/transition counts")
    return gains, payload, file_sha256(p)


def _parse_rows(text: str) -> list[int]:
    rows = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not rows or len(rows) != len(set(rows)) or min(rows) < 0:
        raise ValueError("row indices must be a non-empty unique non-negative CSV")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="fit one-step MANO setpoint response gains")
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--row-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain-lower", type=float, default=0.0)
    parser.add_argument("--gain-upper", type=float, default=0.2)
    args = parser.parse_args()

    import lance

    rows = _parse_rows(args.row_indices)
    records = lance.dataset(str(args.lance_dataset)).take(rows, columns=["hands", "timestamp"]).to_pylist()
    trajectories = []
    transition_count = 0
    for row_index, row in zip(rows, records, strict=True):
        timestamps = np.asarray(row["timestamp"], dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2 or not np.allclose(
            np.diff(timestamps), SOURCE_INTERVAL_SECONDS, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"row {row_index} does not have exact 0.005-second intervals")
        hand = row["hands"][0]
        qpos = np.asarray(hand["urdf_dof"], dtype=np.float64)
        targets = np.asarray(hand["urdf_dof_target"], dtype=np.float64)
        trajectories.append((qpos, targets)); transition_count += len(qpos) - 1
    gains, raw, fitted_count = fit_first_order_gains(
        trajectories, lower=args.gain_lower, upper=args.gain_upper
    )
    if fitted_count != transition_count:
        raise RuntimeError("transition accounting mismatch")
    payload = {
        "contract_id": CONTRACT_ID,
        "source_interval_seconds": SOURCE_INTERVAL_SECONDS,
        "equation": "qpos[t+1] = qpos[t] + gain * shortest_branch(target[t] - qpos[t])",
        "fit": "independent per-DoF least squares through origin; Euler residual/step wrapped",
        "gain_clip": [args.gain_lower, args.gain_upper],
        "row_indices": rows,
        "row_count": len(rows),
        "transition_count": transition_count,
        "gains": gains.tolist(),
        "raw_gains": raw.tolist(),
        "mean_gain": {
            "base_translation": float(np.mean(gains[0:3])),
            "euler": float(np.mean(gains[3:6])),
            "fingers": float(np.mean(gains[6:26])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output), **payload["mean_gain"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
