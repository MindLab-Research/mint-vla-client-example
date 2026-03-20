#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOSS_KEY_CANDIDATES = ("loss:mean", "loss", "train/loss", "cross_entropy_loss")
TIME_KEY_CANDIDATES = ("step_time_sec", "time/step", "avg_step_time_sec", "time/total")


@dataclass(frozen=True)
class Spike:
    step: int
    value: float
    baseline: float
    factor: float
    abs_increase: float


@dataclass(frozen=True)
class PerturbationWindow:
    start_step: int
    end_step: int
    total_sessions: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze loss/time spikes in A2UI or issue193/194-style train_metrics.jsonl logs."
    )
    parser.add_argument("--metrics-jsonl", required=True, help="Path to train_metrics.jsonl")
    parser.add_argument("--loss-key", default=None, help="Override loss key")
    parser.add_argument("--time-key", default=None, help="Override time key")
    parser.add_argument("--baseline-window", type=int, default=5)
    parser.add_argument("--loss-spike-factor", type=float, default=2.0)
    parser.add_argument("--loss-spike-abs", type=float, default=0.08)
    parser.add_argument("--time-spike-factor", type=float, default=2.0)
    parser.add_argument("--time-spike-abs", type=float, default=20.0)
    parser.add_argument("--overlap-tolerance", type=int, default=1)
    parser.add_argument("--manifest-json", default=None, help="Optional orchestrator manifest.json for perturbation windows")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument(
        "--title",
        default="Training loss & step_time_sec (dedup by step, keep last)",
        help="Plot title when --plot-output is set",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _dedup_keep_last(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        raw_step = row.get("step")
        if raw_step is None:
            continue
        step = int(raw_step)
        by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def _pick_key(rows: list[dict[str, Any]], override: str | None, candidates: tuple[str, ...], label: str) -> str:
    if override:
        return override
    for key in candidates:
        if any(row.get(key) is not None for row in rows):
            return key
    raise KeyError(f"could not infer {label} key from candidates={candidates}")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _series(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for row in rows:
        if row.get("step") is None:
            continue
        value = _to_float(row.get(key))
        if value is None or math.isnan(value) or math.isinf(value):
            continue
        out.append((int(row["step"]), value))
    return out


def _detect_spikes(
    series: list[tuple[int, float]],
    *,
    baseline_window: int,
    spike_factor: float,
    min_abs_increase: float,
) -> list[Spike]:
    findings: list[Spike] = []
    if baseline_window <= 0:
        raise ValueError("baseline_window must be positive")
    if len(series) <= baseline_window:
        return findings
    for idx in range(baseline_window, len(series)):
        step, value = series[idx]
        history = [v for _, v in series[idx - baseline_window : idx]]
        baseline = float(statistics.median(history))
        if baseline <= 0:
            continue
        factor = value / baseline
        abs_increase = value - baseline
        if factor >= spike_factor and abs_increase >= min_abs_increase:
            findings.append(
                Spike(
                    step=step,
                    value=value,
                    baseline=baseline,
                    factor=factor,
                    abs_increase=abs_increase,
                )
            )
    return findings


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return None
    return num / den


def _overlap_steps(a: list[Spike], b: list[Spike], tolerance: int) -> list[int]:
    out: set[int] = set()
    b_steps = [spike.step for spike in b]
    for spike in a:
        if any(abs(spike.step - other) <= tolerance for other in b_steps):
            out.add(spike.step)
    return sorted(out)


def _load_perturbation_windows(path: Path | None) -> list[PerturbationWindow]:
    if path is None or not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw_windows = obj.get("window_spec") or obj.get("window_specs") or []
    windows: list[PerturbationWindow] = []
    for idx, row in enumerate(raw_windows, start=1):
        if not isinstance(row, dict):
            continue
        start_step = int(row["start_step"])
        end_step = int(row["end_step"])
        total_sessions = int(row.get("total_sessions") or row.get("window_total_sessions") or 0)
        if total_sessions <= 0:
            continue
        windows.append(
            PerturbationWindow(
                start_step=start_step,
                end_step=end_step,
                total_sessions=total_sessions,
                label=f"W{idx}: {start_step}-{end_step} ({total_sessions} sessions)",
            )
        )
    return windows


def _maybe_plot(
    *,
    steps: list[int],
    losses: list[float],
    times: list[float],
    loss_spikes: list[Spike],
    time_spikes: list[Spike],
    perturbation_windows: list[PerturbationWindow],
    output_path: Path,
    title: str,
    loss_label: str,
    time_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        if output_path.suffix.lower() == ".svg":
            _write_svg_plot(
                steps=steps,
                losses=losses,
                times=times,
                loss_spikes=loss_spikes,
                time_spikes=time_spikes,
                perturbation_windows=perturbation_windows,
                output_path=output_path,
                title=title,
                loss_label=loss_label,
                time_label=time_label,
            )
            return
        raise RuntimeError("matplotlib is required for raster --plot-output; use .svg for no-deps output") from exc

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    for idx, window in enumerate(perturbation_windows):
        label = "perturbation window" if idx == 0 else None
        ax1.axvspan(window.start_step, window.end_step, color="#f59e0b", alpha=0.11, label=label)
        ax1.axvline(window.start_step, color="#b45309", linestyle="--", linewidth=1.4)
        ax1.axvline(window.end_step, color="#b45309", linestyle="--", linewidth=1.4)
        y_top = max(losses) if losses else 0.0
        ax1.text(
            (window.start_step + window.end_step) / 2,
            y_top,
            window.label,
            fontsize=10,
            color="#92400e",
            ha="center",
            va="bottom",
        )

    ax1.plot(steps, losses, marker="o", linewidth=1.4, markersize=4, label=loss_label, color="#1f77b4")
    ax2.plot(steps, times, marker="o", linewidth=1.4, markersize=4, label=time_label, color="#ff7f0e")

    if loss_spikes:
        ax1.scatter(
            [s.step for s in loss_spikes],
            [s.value for s in loss_spikes],
            color="#d62728",
            s=45,
            label="loss_spike",
            zorder=5,
        )
    if time_spikes:
        ax2.scatter(
            [s.step for s in time_spikes],
            [s.value for s in time_spikes],
            color="#8c564b",
            s=45,
            label="time_spike",
            zorder=5,
        )

    ax1.set_title(title)
    ax1.set_xlabel("step")
    ax1.set_ylabel(loss_label, color="#1f77b4")
    ax2.set_ylabel(time_label, color="#ff7f0e")
    ax1.grid(alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _scale_points(
    steps: list[int],
    values: list[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    if len(steps) != len(values):
        raise ValueError("steps and values must have same length")
    if not steps:
        return []
    min_step = min(steps)
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)
    step_span = max(max_step - min_step, 1)
    value_span = max(max_value - min_value, 1e-9)
    points: list[tuple[float, float]] = []
    for step, value in zip(steps, values, strict=True):
        x = x0 + ((step - min_step) / step_span) * width
        y = y0 + height - ((value - min_value) / value_span) * height
        points.append((x, y))
    return points


def _write_svg_plot(
    *,
    steps: list[int],
    losses: list[float],
    times: list[float],
    loss_spikes: list[Spike],
    time_spikes: list[Spike],
    perturbation_windows: list[PerturbationWindow],
    output_path: Path,
    title: str,
    loss_label: str,
    time_label: str,
) -> None:
    width = 1400
    height = 760
    left = 90
    right = 90
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    loss_points = _scale_points(steps, losses, x0=left, y0=top, width=plot_width, height=plot_height)
    time_points = _scale_points(steps, times, x0=left, y0=top, width=plot_width, height=plot_height)
    loss_point_by_step = {step: point for step, point in zip(steps, loss_points, strict=True)}
    time_point_by_step = {step: point for step, point in zip(steps, time_points, strict=True)}
    min_step = min(steps) if steps else 0
    max_step = max(steps) if steps else 1
    step_span = max(max_step - min_step, 1)

    def x_for_step(step: int) -> float:
        return left + ((step - min_step) / step_span) * plot_width

    def polyline(points: list[tuple[float, float]], color: str) -> str:
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}" />'

    def circles(spikes: list[Spike], point_map: dict[int, tuple[float, float]], color: str) -> str:
        out: list[str] = []
        for spike in spikes:
            point = point_map.get(spike.step)
            if point is None:
                continue
            x, y = point
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" />')
        return "\n".join(out)

    def perturbation_markup() -> str:
        out: list[str] = []
        for window in perturbation_windows:
            start_x = x_for_step(window.start_step)
            end_x = x_for_step(window.end_step)
            out.append(
                f'<rect x="{start_x:.2f}" y="{top:.2f}" width="{max(end_x - start_x, 2):.2f}" height="{plot_height:.2f}" '
                'fill="#f59e0b" fill-opacity="0.10" />'
            )
            out.append(
                f'<line x1="{start_x:.2f}" y1="{top:.2f}" x2="{start_x:.2f}" y2="{top + plot_height:.2f}" '
                'stroke="#b45309" stroke-dasharray="6 4" stroke-width="1.5" />'
            )
            out.append(
                f'<line x1="{end_x:.2f}" y1="{top:.2f}" x2="{end_x:.2f}" y2="{top + plot_height:.2f}" '
                'stroke="#b45309" stroke-dasharray="6 4" stroke-width="1.5" />'
            )
            out.append(
                f'<text x="{(start_x + end_x) / 2:.2f}" y="{top + 18:.2f}" font-size="12" text-anchor="middle" fill="#92400e">{html.escape(window.label)}</text>'
            )
        return "\n".join(out)

    x_ticks: list[str] = []
    for idx, step in enumerate(steps):
        if idx not in {0, len(steps) - 1} and idx % max(len(steps) // 10, 1) != 0:
            continue
        x, _ = loss_points[idx]
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}" stroke="#555" />'
            f'<text x="{x:.2f}" y="{top + plot_height + 24:.2f}" font-size="14" text-anchor="middle" fill="#333">{step}</text>'
        )

    def y_ticks(values: list[float], x: float, anchor: str) -> str:
        min_value = min(values)
        max_value = max(values)
        out: list[str] = []
        for i in range(6):
            frac = i / 5
            value = max_value - frac * (max_value - min_value)
            y = top + frac * plot_height
            out.append(
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_width:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />'
                f'<text x="{x:.2f}" y="{y + 5:.2f}" font-size="13" text-anchor="{anchor}" fill="#333">{value:.2f}</text>'
            )
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.2f}" y="36" font-size="26" text-anchor="middle" fill="#111">{html.escape(title)}</text>
  <text x="{width / 2:.2f}" y="{height - 18:.2f}" font-size="16" text-anchor="middle" fill="#333">step</text>
  <text x="28" y="{top + plot_height / 2:.2f}" font-size="16" text-anchor="middle" fill="#1f77b4" transform="rotate(-90 28 {top + plot_height / 2:.2f})">{html.escape(loss_label)}</text>
  <text x="{width - 22:.2f}" y="{top + plot_height / 2:.2f}" font-size="16" text-anchor="middle" fill="#ff7f0e" transform="rotate(90 {width - 22:.2f} {top + plot_height / 2:.2f})">{html.escape(time_label)}</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#555" />
  {perturbation_markup()}
  {y_ticks(losses, left - 10, "end")}
  {y_ticks(times, left + plot_width + 10, "start")}
  {"".join(x_ticks)}
  {polyline(loss_points, "#1f77b4")}
  {polyline(time_points, "#ff7f0e")}
  {circles(loss_spikes, loss_point_by_step, "#d62728")}
  {circles(time_spikes, time_point_by_step, "#8c564b")}
  <g font-size="14" fill="#333">
    <rect x="{width - 310}" y="{top + 12}" width="260" height="116" fill="white" stroke="#ddd" />
    <line x1="{width - 252}" y1="{top + 32}" x2="{width - 220}" y2="{top + 32}" stroke="#1f77b4" stroke-width="3" />
    <text x="{width - 210}" y="{top + 37}" fill="#1f77b4">{html.escape(loss_label)}</text>
    <line x1="{width - 252}" y1="{top + 56}" x2="{width - 220}" y2="{top + 56}" stroke="#ff7f0e" stroke-width="3" />
    <text x="{width - 210}" y="{top + 61}" fill="#ff7f0e">{html.escape(time_label)}</text>
    <circle cx="{width - 236}" cy="{top + 80}" r="5" fill="#d62728" />
    <text x="{width - 210}" y="{top + 85}" fill="#d62728">loss_spike</text>
    <line x1="{width - 252}" y1="{top + 104}" x2="{width - 220}" y2="{top + 104}" stroke="#b45309" stroke-dasharray="6 4" stroke-width="2" />
    <text x="{width - 210}" y="{top + 109}" fill="#92400e">perturbation window</text>
  </g>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_jsonl).expanduser().resolve()
    rows = _dedup_keep_last(_load_rows(metrics_path))
    perturbation_windows = _load_perturbation_windows(
        Path(args.manifest_json).expanduser().resolve() if args.manifest_json else None
    )

    loss_key = _pick_key(rows, args.loss_key, LOSS_KEY_CANDIDATES, "loss")
    time_key = _pick_key(rows, args.time_key, TIME_KEY_CANDIDATES, "time")

    loss_series = _series(rows, loss_key)
    time_series = _series(rows, time_key)
    if not loss_series:
        raise ValueError(f"no usable values for loss_key={loss_key!r}")
    if not time_series:
        raise ValueError(f"no usable values for time_key={time_key!r}")

    loss_by_step = dict(loss_series)
    time_by_step = dict(time_series)
    common_steps = sorted(set(loss_by_step) & set(time_by_step))
    losses = [loss_by_step[step] for step in common_steps]
    times = [time_by_step[step] for step in common_steps]

    loss_spikes = _detect_spikes(
        list(zip(common_steps, losses, strict=True)),
        baseline_window=args.baseline_window,
        spike_factor=args.loss_spike_factor,
        min_abs_increase=args.loss_spike_abs,
    )
    time_spikes = _detect_spikes(
        list(zip(common_steps, times, strict=True)),
        baseline_window=args.baseline_window,
        spike_factor=args.time_spike_factor,
        min_abs_increase=args.time_spike_abs,
    )
    overlapping_loss_steps = _overlap_steps(loss_spikes, time_spikes, args.overlap_tolerance)
    overlapping_time_steps = _overlap_steps(time_spikes, loss_spikes, args.overlap_tolerance)

    summary = {
        "metrics_jsonl": str(metrics_path),
        "row_count_raw": len(_load_rows(metrics_path)),
        "row_count_dedup": len(rows),
        "loss_key": loss_key,
        "time_key": time_key,
        "step_count_with_both": len(common_steps),
        "baseline_window": args.baseline_window,
        "loss_spike_factor": args.loss_spike_factor,
        "loss_spike_abs": args.loss_spike_abs,
        "time_spike_factor": args.time_spike_factor,
        "time_spike_abs": args.time_spike_abs,
        "loss_spike_count": len(loss_spikes),
        "time_spike_count": len(time_spikes),
        "loss_spikes": [asdict(item) for item in loss_spikes],
        "time_spikes": [asdict(item) for item in time_spikes],
        "loss_spikes_near_time_spikes": overlapping_loss_steps,
        "time_spikes_near_loss_spikes": overlapping_time_steps,
        "pearson_loss_time": _pearson(losses, times),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "time_min": min(times),
        "time_max": max(times),
        "perturbation_windows": [asdict(item) for item in perturbation_windows],
    }

    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else metrics_path.parent / "issue193_194_loss_spike_summary.json"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plot_output:
        _maybe_plot(
            steps=common_steps,
            losses=losses,
            times=times,
            loss_spikes=loss_spikes,
            time_spikes=time_spikes,
            perturbation_windows=perturbation_windows,
            output_path=Path(args.plot_output).expanduser().resolve(),
            title=args.title,
            loss_label=loss_key,
            time_label=time_key,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
