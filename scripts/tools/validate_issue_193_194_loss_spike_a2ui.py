#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_A2UI_REPO = Path("/vePFS-Mindverse/user/intern/nolanho/code/a2ui_training")
FALLBACK_COVER_ROOT = Path("/vePFS-Mindverse/user/intern/nolanho/code/mint/cover/loss_spike")


@dataclass(frozen=True)
class SpikeEvent:
    step: int
    loss: float
    prev_mean_loss: float
    spike_factor: float
    abs_increase: float
    step_time_sec: float | None
    avg_step_time_sec: float | None


@dataclass(frozen=True)
class AnalysisSummary:
    metrics_path: str
    rows: int
    rows_with_loss: int
    rows_with_step_time: int
    first_loss: float | None
    last_loss: float | None
    min_loss: float | None
    max_loss: float | None
    mean_loss: float | None
    loss_step_time_corr: float | None
    loss_avg_step_time_corr: float | None
    spike_count: int
    spikes: list[SpikeEvent]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cover_path(filename: str) -> Path:
    candidate = _repo_root() / "cover" / "loss_spike" / filename
    if candidate.exists():
        return candidate
    return FALLBACK_COVER_ROOT / filename


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{metrics_path}:{line_no} is not a JSON object")
            rows.append(data)
    return rows


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(denom_x * denom_y)
    if denom == 0.0:
        return None
    return numer / denom


def detect_loss_spikes(
    rows: list[dict[str, Any]],
    *,
    baseline_window: int,
    loss_spike_factor: float,
    loss_spike_abs: float,
) -> list[SpikeEvent]:
    if baseline_window <= 0:
        raise ValueError("baseline_window must be positive")
    loss_rows = [row for row in rows if _float_or_none(row.get("loss")) is not None]
    spikes: list[SpikeEvent] = []
    for idx in range(baseline_window, len(loss_rows)):
        prev_rows = loss_rows[idx - baseline_window : idx]
        prev_losses = [_float_or_none(row.get("loss")) for row in prev_rows]
        baseline = [loss for loss in prev_losses if loss is not None]
        if len(baseline) != baseline_window:
            continue
        prev_mean = statistics.fmean(baseline)
        current = loss_rows[idx]
        loss = _float_or_none(current.get("loss"))
        if loss is None:
            continue
        abs_increase = loss - prev_mean
        if abs_increase < loss_spike_abs:
            continue
        if prev_mean <= 0.0:
            spike_factor = math.inf
            crossed = True
        else:
            spike_factor = loss / prev_mean
            crossed = spike_factor >= loss_spike_factor
        if not crossed:
            continue
        spikes.append(
            SpikeEvent(
                step=int(current.get("step", idx + 1)),
                loss=loss,
                prev_mean_loss=prev_mean,
                spike_factor=spike_factor,
                abs_increase=abs_increase,
                step_time_sec=_float_or_none(current.get("step_time_sec")),
                avg_step_time_sec=_float_or_none(current.get("avg_step_time_sec")),
            )
        )
    return spikes


def analyze_metrics(
    metrics_path: Path,
    *,
    baseline_window: int,
    loss_spike_factor: float,
    loss_spike_abs: float,
) -> AnalysisSummary:
    rows = load_metrics_rows(metrics_path)
    losses = [_float_or_none(row.get("loss")) for row in rows]
    valid_losses = [loss for loss in losses if loss is not None]
    step_pairs = [
        (_float_or_none(row.get("loss")), _float_or_none(row.get("step_time_sec")))
        for row in rows
    ]
    avg_step_pairs = [
        (_float_or_none(row.get("loss")), _float_or_none(row.get("avg_step_time_sec")))
        for row in rows
    ]
    loss_step_corr = pearson_correlation(
        [loss for loss, step_time in step_pairs if loss is not None and step_time is not None],
        [step_time for loss, step_time in step_pairs if loss is not None and step_time is not None],
    )
    loss_avg_step_corr = pearson_correlation(
        [loss for loss, avg_step_time in avg_step_pairs if loss is not None and avg_step_time is not None],
        [avg_step_time for loss, avg_step_time in avg_step_pairs if loss is not None and avg_step_time is not None],
    )
    spikes = detect_loss_spikes(
        rows,
        baseline_window=baseline_window,
        loss_spike_factor=loss_spike_factor,
        loss_spike_abs=loss_spike_abs,
    )
    return AnalysisSummary(
        metrics_path=str(metrics_path),
        rows=len(rows),
        rows_with_loss=len(valid_losses),
        rows_with_step_time=sum(
            1 for row in rows if _float_or_none(row.get("step_time_sec")) is not None
        ),
        first_loss=valid_losses[0] if valid_losses else None,
        last_loss=valid_losses[-1] if valid_losses else None,
        min_loss=min(valid_losses) if valid_losses else None,
        max_loss=max(valid_losses) if valid_losses else None,
        mean_loss=statistics.fmean(valid_losses) if valid_losses else None,
        loss_step_time_corr=loss_step_corr,
        loss_avg_step_time_corr=loss_avg_step_corr,
        spike_count=len(spikes),
        spikes=spikes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A2UI SFT against the issue193/194 branch and analyze loss spikes."
    )
    parser.add_argument("--a2ui-repo-path", type=Path, default=DEFAULT_A2UI_REPO)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch a2ui_training.",
    )
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=_default_cover_path("train_v4.json"))
    parser.add_argument("--eval-data", type=Path, default=_default_cover_path("eval_v4.json"))
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "dummy"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--renderer-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-preview-count", type=int, default=100)
    parser.add_argument("--save-preview-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--retry-max", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--retry-jitter", type=float, default=1.0)
    parser.add_argument("--baseline-window", type=int, default=5)
    parser.add_argument("--loss-spike-factor", type=float, default=3.0)
    parser.add_argument("--loss-spike-abs", type=float, default=0.5)
    parser.add_argument("--print-top-spikes", type=int, default=5)
    parser.add_argument(
        "--fail-on-spike",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when spike_count > 0.",
    )
    parser.add_argument(
        "--run-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch A2UI training before analysis. Disable to analyze an existing log dir.",
    )
    return parser.parse_args()


def build_training_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "experiments.run_sft_single_turn_mint",
        "--log-path",
        str(args.log_path),
        "--model-name",
        args.model_name,
        "--train-data",
        str(args.train_data),
        "--preview-data",
        str(args.eval_data),
        "--batch-size",
        str(args.batch_size),
        "--num-epochs",
        str(args.num_epochs),
        "--lora-rank",
        str(args.lora_rank),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--save-every",
        str(args.save_every),
        "--save-preview-count",
        str(args.save_preview_count),
        "--save-preview-every",
        str(args.save_preview_every),
        "--log-every",
        str(args.log_every),
        "--retry-max",
        str(args.retry_max),
        "--retry-base-delay",
        str(args.retry_base_delay),
        "--retry-jitter",
        str(args.retry_jitter),
        "--base-url",
        args.base_url,
    ]
    if args.renderer_name:
        command.extend(["--renderer-name", args.renderer_name])
    if args.train_limit is not None:
        command.extend(["--train-limit", str(args.train_limit)])
    return command


def write_summary(path: Path, summary: AnalysisSummary) -> None:
    payload = asdict(summary)
    payload["spikes"] = [asdict(spike) for spike in summary.spikes]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.log_path.mkdir(parents=True, exist_ok=True)

    if args.run_training:
        if not args.a2ui_repo_path.exists():
            raise FileNotFoundError(f"a2ui repo not found: {args.a2ui_repo_path}")
        command = build_training_command(args)
        print(f"[run] cwd={args.a2ui_repo_path}")
        print(f"[run] command={' '.join(command)}")
        env = os.environ.copy()
        env["TINKER_API_KEY"] = args.api_key
        completed = subprocess.run(command, cwd=args.a2ui_repo_path, env=env, check=False)
        if completed.returncode != 0:
            print(f"[error] training exited with code {completed.returncode}", file=sys.stderr)
            return completed.returncode

    metrics_path = args.log_path / "train_metrics.jsonl"
    if not metrics_path.exists():
        print(f"[error] missing metrics file: {metrics_path}", file=sys.stderr)
        return 2

    summary = analyze_metrics(
        metrics_path,
        baseline_window=args.baseline_window,
        loss_spike_factor=args.loss_spike_factor,
        loss_spike_abs=args.loss_spike_abs,
    )
    summary_path = args.log_path / "issue193_194_loss_spike_analysis.json"
    write_summary(summary_path, summary)

    print(
        "[analysis] "
        f"rows={summary.rows} rows_with_loss={summary.rows_with_loss} "
        f"first_loss={summary.first_loss} last_loss={summary.last_loss} "
        f"max_loss={summary.max_loss} mean_loss={summary.mean_loss}"
    )
    print(
        "[analysis] "
        f"loss_step_time_corr={summary.loss_step_time_corr} "
        f"loss_avg_step_time_corr={summary.loss_avg_step_time_corr} "
        f"spike_count={summary.spike_count}"
    )
    for spike in summary.spikes[: max(args.print_top_spikes, 0)]:
        print(
            "[spike] "
            f"step={spike.step} loss={spike.loss:.6f} prev_mean={spike.prev_mean_loss:.6f} "
            f"factor={spike.spike_factor:.3f} abs_increase={spike.abs_increase:.6f} "
            f"step_time={spike.step_time_sec} avg_step_time={spike.avg_step_time_sec}"
        )
    print(f"[analysis] wrote {summary_path}")

    if args.fail_on_spike and summary.spike_count > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
