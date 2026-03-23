from __future__ import annotations

import math

from scripts.tools.validate_issue_193_194_loss_spike_a2ui import (
    analyze_metrics,
    detect_loss_spikes,
    pearson_correlation,
)


def test_pearson_correlation_returns_none_for_constant_series() -> None:
    assert pearson_correlation([1.0, 1.0], [2.0, 3.0]) is None


def test_detect_loss_spikes_flags_large_jump() -> None:
    rows = [
        {"step": 1, "loss": 0.10, "step_time_sec": 1.0, "avg_step_time_sec": 1.0},
        {"step": 2, "loss": 0.11, "step_time_sec": 1.1, "avg_step_time_sec": 1.05},
        {"step": 3, "loss": 0.12, "step_time_sec": 1.2, "avg_step_time_sec": 1.1},
        {"step": 4, "loss": 1.20, "step_time_sec": 5.0, "avg_step_time_sec": 2.0},
    ]

    spikes = detect_loss_spikes(
        rows,
        baseline_window=3,
        loss_spike_factor=3.0,
        loss_spike_abs=0.5,
    )

    assert len(spikes) == 1
    assert spikes[0].step == 4
    assert math.isclose(spikes[0].prev_mean_loss, 0.11, rel_tol=1e-6)
    assert math.isclose(spikes[0].step_time_sec or 0.0, 5.0, rel_tol=1e-6)


def test_analyze_metrics_reports_positive_loss_time_correlation(tmp_path) -> None:
    metrics_path = tmp_path / "train_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                '{"step": 1, "loss": 0.10, "step_time_sec": 1.0, "avg_step_time_sec": 1.0}',
                '{"step": 2, "loss": 0.20, "step_time_sec": 2.0, "avg_step_time_sec": 1.5}',
                '{"step": 3, "loss": 0.30, "step_time_sec": 3.0, "avg_step_time_sec": 2.0}',
                '{"step": 4, "loss": 0.90, "step_time_sec": 6.0, "avg_step_time_sec": 3.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze_metrics(
        metrics_path,
        baseline_window=3,
        loss_spike_factor=2.5,
        loss_spike_abs=0.4,
    )

    assert summary.rows == 4
    assert summary.spike_count == 1
    assert summary.loss_step_time_corr is not None
    assert summary.loss_avg_step_time_corr is not None
    assert summary.loss_step_time_corr > 0.9
    assert summary.loss_avg_step_time_corr > 0.9
