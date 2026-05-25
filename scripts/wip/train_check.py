#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://mint.macaron.xin"
DEFAULT_RESULTS_ROOT = "/root/run_results/mint"
FEISHU_TITLE = "MinT sanity-check report"
DEFAULT_MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "4b": "Qwen/Qwen3-4B-Instruct-2507",
    "4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}
ALL_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]
DEFAULT_DEGRADATION_THRESHOLDS = {
    "Qwen/Qwen3-0.6B": {"wall_clock_s": 180.0, "slowest_max_s": 90.0},
    "Qwen/Qwen3-4B-Instruct-2507": {"wall_clock_s": 180.0, "slowest_max_s": 90.0},
    "Qwen/Qwen3-4B-Thinking-2507": {"wall_clock_s": 180.0, "slowest_max_s": 90.0},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"wall_clock_s": 1800.0, "slowest_max_s": 900.0},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"wall_clock_s": 1500.0, "slowest_max_s": 900.0},
}
RUNNER = Path(".claude/skills/sanity-check/mint_rl_test_long.py")
REQUEST_RE = re.compile(r"request_type=([A-Za-z0-9_]+)\s+request_id=([A-Za-z0-9_:-]+)")
KV_RE = {
    "client_session_id": re.compile(r"client_session_id='([^']+)'"),
    "tokenizer_training_model_id": re.compile(r"tokenizer_training_model_id='([^']+)'"),
    "training_model_id": re.compile(r"training_model_id='([^']+)'"),
    "sampling_session_id": re.compile(r"sampling_session_id='([^']+)'"),
    "final_sampling_session_id": re.compile(r"final_sampling_session_id='([^']+)'"),
    "experiment_directory": re.compile(r"Experiment directory:\s+(.+)$"),
}


@dataclass
class ModelRun:
    model: str
    slug: str
    run_dir: Path
    command: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str | None
    detail: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MinT production RL sanity checks with aligned params and artifacts."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model aliases or full model names. Aliases: 0.6b, 4b, 4b-instruct, 4b-thinking, 30b, 235b.",
    )
    parser.add_argument(
        "--model",
        dest="models_flag",
        action="append",
        default=[],
        help="Additional model alias or full model name. Repeatable.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the standard 5-model production matrix.",
    )
    parser.add_argument("--num-rl-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for wrapper artifacts.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed run directory name under results root.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution even when multiple models are requested.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Allow parallel execution. Refused for --all-models because production sanity must run in order.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip SDK and HTTP preflight checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit path for summary JSON.",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Optional explicit path for summary Markdown.",
    )
    parser.add_argument(
        "--feishu",
        action="store_true",
        default=None,
        help="Send the final Feishu report after the run.",
    )
    parser.add_argument(
        "--no-feishu",
        dest="feishu",
        action="store_false",
        help="Do not send the final Feishu report.",
    )
    return parser.parse_args()


def resolve_model(name: str) -> str:
    lowered = name.strip().lower()
    return DEFAULT_MODELS.get(lowered, name.strip())


def selected_models(args: argparse.Namespace) -> list[str]:
    requested = []
    if args.all_models:
        requested.extend(ALL_MODELS)
    requested.extend(resolve_model(m) for m in args.models)
    requested.extend(resolve_model(m) for m in args.models_flag)
    if not requested:
        requested = [DEFAULT_MODELS["0.6b"]]

    deduped = []
    seen = set()
    for model in requested:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def model_slug(model: str) -> str:
    return (
        model.lower()
        .replace("/", "__")
        .replace(":", "__")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "-")
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def validate_production_env(base_url: str) -> str:
    if base_url != DEFAULT_BASE_URL:
        raise SystemExit(f"train-check expects base URL {DEFAULT_BASE_URL}, got: {base_url}")

    api_key = os.environ.get("MINT_API_KEY")
    if not api_key:
        raise SystemExit("missing MINT_API_KEY in env/.secrets.env")
    os.environ.setdefault("MINT_API_KEY", api_key)

    owner = os.environ.get("MINT_TEST_CHECKPOINT_OWNER_ID", "")
    if not re.fullmatch(r"[0-9a-fA-F]{24}", owner):
        raise SystemExit(
            "MINT_TEST_CHECKPOINT_OWNER_ID must be a 24-character production owner ObjectId"
        )
    return api_key


def preflight(base_url: str, api_key: str | None) -> None:
    print(f"[preflight] python={sys.executable}")
    print("[preflight] production URL and owner id are set; API key is present (redacted)")

    req = Request(f"{base_url}/api/v1/healthz")
    with urlopen(req, timeout=30) as resp:
        print(f"[preflight] /api/v1/healthz -> {resp.status}")

    if api_key:
        req = Request(f"{base_url}/internal/actors", headers={"X-API-Key": api_key})
        try:
            with urlopen(req, timeout=30) as resp:
                print(f"[preflight] /internal/actors -> {resp.status}")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[preflight] /internal/actors probe skipped: {exc}", file=sys.stderr)


def ensure_runner_exists() -> None:
    if not RUNNER.exists():
        raise FileNotFoundError(f"runner not found: {RUNNER}")


def make_run_root(results_root: Path, run_name: str | None, *, create: bool = True) -> Path:
    timestamp = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_root = results_root / timestamp
    if create:
        run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def build_runs(args: argparse.Namespace, run_root: Path, *, create_dirs: bool = True) -> list[ModelRun]:
    base_env = os.environ.copy()
    base_env["MINT_BASE_URL"] = args.base_url
    base_env.setdefault("PYTHONUNBUFFERED", "1")
    base_env["MINT_TEST_CHECKPOINT_OWNER_ID"] = os.environ["MINT_TEST_CHECKPOINT_OWNER_ID"]
    base_env["MINT_API_KEY"] = os.environ["MINT_API_KEY"]

    runs = []
    for model in selected_models(args):
        slug = model_slug(model)
        model_dir = run_root / slug
        if create_dirs:
            model_dir.mkdir(parents=True, exist_ok=True)
        env = base_env.copy()
        env["MINT_TEST_EXPERIMENT_ROOT"] = str(model_dir)
        cmd = [
            sys.executable,
            str(RUNNER),
            f"--model={model}",
            f"--num-rl-steps={args.num_rl_steps}",
            f"--batch-size={args.batch_size}",
            f"--group-size={args.group_size}",
            f"--max-tokens={args.max_tokens}",
            f"--timeout-s={args.timeout_s}",
        ]
        runs.append(ModelRun(model=model, slug=slug, run_dir=model_dir, command=cmd, env=env))
    return runs


def stream_pipe(pipe: IO[str], sink: IO[str], console: IO[str], prefix: str) -> None:
    try:
        for line in pipe:
            sink.write(line)
            sink.flush()
            console.write(f"[{prefix}] {line}")
            console.flush()
    finally:
        pipe.close()


def flatten_experiment_dir(run_dir: Path, stdout_text: str) -> Path | None:
    match = KV_RE["experiment_directory"].search(stdout_text)
    if not match:
        return None
    experiment_dir = Path(match.group(1).strip())
    if not experiment_dir.exists() or experiment_dir == run_dir:
        return experiment_dir if experiment_dir.exists() else None

    for item in experiment_dir.iterdir():
        dest = run_dir / item.name
        if dest.exists():
            continue
        shutil.move(str(item), dest)
    try:
        experiment_dir.rmdir()
    except OSError:
        pass
    return run_dir


def parse_log_metadata(stdout_path: Path) -> dict[str, object]:
    text = stdout_path.read_text() if stdout_path.exists() else ""
    info: dict[str, object] = {"request_ids": {}, "session_ids": {}}
    request_ids: dict[str, list[str]] = {}
    for request_type, request_id in REQUEST_RE.findall(text):
        request_ids.setdefault(request_type, [])
        if request_id not in request_ids[request_type]:
            request_ids[request_type].append(request_id)
    info["request_ids"] = request_ids

    session_ids: dict[str, object] = {}
    for key, pattern in KV_RE.items():
        if key == "experiment_directory":
            continue
        matches = pattern.findall(text)
        if not matches:
            continue
        if len(matches) == 1:
            session_ids[key] = matches[0]
        else:
            ordered = []
            for match in matches:
                if match not in ordered:
                    ordered.append(match)
            session_ids[key] = ordered
    info["session_ids"] = session_ids
    return info


def load_timing_summary(run_dir: Path) -> dict[str, object] | None:
    timing_path = find_latest_artifact(run_dir, "timing_summary.json")
    if timing_path is None:
        return None
    return json.loads(timing_path.read_text())


def find_latest_artifact(run_dir: Path, name: str) -> Path | None:
    matches = sorted(run_dir.glob(f"**/{name}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def extract_slowest_stage(timing: dict[str, object] | None) -> tuple[str | None, float | None]:
    if not timing:
        return None, None
    stages = timing.get("stages", {})
    if isinstance(stages, list):
        iterable = []
        for stage in stages:
            if isinstance(stage, dict):
                iterable.append((str(stage.get("stage") or stage.get("name") or "unknown"), stage))
    elif isinstance(stages, dict):
        iterable = [(str(name), stats) for name, stats in stages.items()]
    else:
        return None, None
    best_name = None
    best_max = None
    for name, stats in iterable:
        if not isinstance(stats, dict):
            continue
        max_s = stats.get("max_s")
        if not isinstance(max_s, (int, float)):
            continue
        if best_max is None or max_s > best_max:
            best_name = name
            best_max = float(max_s)
    return best_name, best_max


def _load_degradation_thresholds() -> dict[str, dict[str, float]]:
    raw = os.environ.get("MINT_SANITY_DEGRADATION_THRESHOLDS_JSON", "").strip()
    if not raw:
        return DEFAULT_DEGRADATION_THRESHOLDS
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_DEGRADATION_THRESHOLDS
    if not isinstance(payload, dict):
        return DEFAULT_DEGRADATION_THRESHOLDS
    out: dict[str, dict[str, float]] = {}
    for model, thresholds in payload.items():
        if not isinstance(thresholds, dict):
            continue
        model_thresholds: dict[str, float] = {}
        for key in ("wall_clock_s", "slowest_max_s"):
            value = thresholds.get(key)
            if isinstance(value, (int, float)) and float(value) > 0:
                model_thresholds[key] = float(value)
        if model_thresholds:
            out[str(model)] = model_thresholds
    return {**DEFAULT_DEGRADATION_THRESHOLDS, **out}


def classify_timing_degradation(
    *,
    model: str,
    status: str,
    wall_clock_s: object,
    slowest_max_s: object,
) -> str | None:
    if status != "ok":
        return None
    thresholds = _load_degradation_thresholds().get(model)
    if not thresholds:
        return None
    reasons = []
    if isinstance(wall_clock_s, (int, float)):
        limit = thresholds.get("wall_clock_s")
        if limit is not None and float(wall_clock_s) > float(limit):
            reasons.append(f"wall {_fmt_duration(wall_clock_s)} > {_fmt_duration(limit)}")
    if isinstance(slowest_max_s, (int, float)):
        limit = thresholds.get("slowest_max_s")
        if limit is not None and float(slowest_max_s) > float(limit):
            reasons.append(f"slowest max {_fmt_duration(slowest_max_s)} > {_fmt_duration(limit)}")
    return "; ".join(reasons) if reasons else None


def extract_generation_summary(timing: dict[str, object] | None) -> dict[str, object] | None:
    if not timing:
        return None
    stages = timing.get("stages", {})
    if isinstance(stages, dict):
        iterable = stages.values()
    elif isinstance(stages, list):
        iterable = stages
    else:
        return None
    output_tokens = 0
    elapsed_s = 0.0
    hit_max_count = 0
    by_stage: dict[str, dict[str, object]] = {}
    for raw_stage in iterable:
        if not isinstance(raw_stage, dict):
            continue
        stage_name = str(raw_stage.get("stage") or raw_stage.get("name") or "unknown")
        tokens = raw_stage.get("output_tokens")
        stage_elapsed = raw_stage.get("total_s")
        if not isinstance(tokens, (int, float)) or float(tokens) <= 0:
            continue
        if not isinstance(stage_elapsed, (int, float)) or float(stage_elapsed) <= 0:
            continue
        tokens_i = int(tokens)
        elapsed_f = float(stage_elapsed)
        hits = raw_stage.get("hit_max_count")
        hits_i = int(hits) if isinstance(hits, (int, float)) else 0
        output_tokens += tokens_i
        elapsed_s += elapsed_f
        hit_max_count += hits_i
        by_stage[stage_name] = {
            "output_tokens": tokens_i,
            "elapsed_s": round(elapsed_f, 3),
            "tokens_per_s": round(tokens_i / elapsed_f, 3),
            "hit_max_count": hits_i,
        }
    if output_tokens <= 0 or elapsed_s <= 0:
        return None
    return {
        "output_tokens": output_tokens,
        "elapsed_s": round(elapsed_s, 3),
        "tokens_per_s": round(output_tokens / elapsed_s, 3),
        "hit_max_count": hit_max_count,
        "by_stage": by_stage,
    }


def run_one(run: ModelRun) -> dict[str, object]:
    stdout_path = run.run_dir / "stdout.log"
    stderr_path = run.run_dir / "stderr.log"
    started_at = time.time()
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        proc = subprocess.Popen(
            run.command,
            cwd=Path.cwd(),
            env=run.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        t_out = threading.Thread(
            target=stream_pipe, args=(proc.stdout, stdout_file, sys.stdout, run.slug), daemon=True
        )
        t_err = threading.Thread(
            target=stream_pipe, args=(proc.stderr, stderr_file, sys.stderr, run.slug), daemon=True
        )
        t_out.start()
        t_err.start()
        rc = proc.wait()
        t_out.join()
        t_err.join()

    stdout_text = stdout_path.read_text() if stdout_path.exists() else ""
    experiment_dir = flatten_experiment_dir(run.run_dir, stdout_text)
    stderr_text = stderr_path.read_text() if stderr_path.exists() else ""
    combined_text = stdout_text + "\n" + stderr_text
    meta = parse_log_metadata(stdout_path)
    timing = load_timing_summary(run.run_dir)
    timing_json_path = find_latest_artifact(run.run_dir, "timing_summary.json")
    timing_md_path = find_latest_artifact(run.run_dir, "timing_summary.md")
    timing_events_path = find_latest_artifact(run.run_dir, "timing_events.jsonl")
    slowest_stage, slowest_max_s = extract_slowest_stage(timing)
    wall_clock_s = None
    if timing and isinstance(timing.get("wall_clock_s"), (int, float)):
        wall_clock_s = float(timing["wall_clock_s"])
    status = "ok" if rc == 0 else "fail"
    classification = classify_failure_detail(combined_text, exit_code=rc)
    degradation_reason = classify_timing_degradation(
        model=run.model,
        status=status,
        wall_clock_s=wall_clock_s,
        slowest_max_s=slowest_max_s,
    )
    return {
        "model": run.model,
        "slug": run.slug,
        "exit_code": rc,
        "status": status,
        "run_dir": str(run.run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "experiment_dir": str(experiment_dir) if experiment_dir else None,
        "request_ids": meta["request_ids"],
        "session_ids": meta["session_ids"],
        "failure_class": classification.failure_class,
        "failure_detail": classification.detail,
        "failure_surface": failure_surface_from_logs(combined_text),
        "timing_summary_json": str(timing_json_path) if timing_json_path else None,
        "timing_summary_md": str(timing_md_path) if timing_md_path else None,
        "timing_events_jsonl": str(timing_events_path) if timing_events_path else None,
        "wall_clock_s": wall_clock_s,
        "slowest_stage": slowest_stage,
        "slowest_max_s": slowest_max_s,
        "generation_summary": extract_generation_summary(timing),
        "timing_degraded": bool(degradation_reason),
        "degradation_reason": degradation_reason,
        "started_at_epoch_s": started_at,
        "finished_at_epoch_s": time.time(),
    }


def run_parallel(runs: list[ModelRun], sequential: bool) -> list[dict[str, object]]:
    if sequential or len(runs) <= 1:
        return [run_one(run) for run in runs]

    results: list[dict[str, object] | None] = [None] * len(runs)
    threads = []

    def worker(index: int, run: ModelRun) -> None:
        results[index] = run_one(run)

    for index, run in enumerate(runs):
        thread = threading.Thread(target=worker, args=(index, run), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return [result for result in results if result is not None]


def _fmt_duration(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}s"
    return "n/a"


def _compact_error_detail(text: str, *, max_chars: int = 260) -> str | None:
    candidates: list[str] = []
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in (
                "requestfailederror",
                "runtimeerror:",
                "valueerror:",
                "timeouterror:",
                "traceback",
                "fail in ",
                "error:",
                "exception",
                "terminated due to",
            )
        ):
            candidates.append(line)
        if len(candidates) >= 3:
            break
    if not candidates:
        for raw_line in reversed(text.splitlines()):
            line = raw_line.strip()
            if line:
                candidates.append(line)
                break
    if not candidates:
        return None
    detail = " | ".join(reversed(candidates))
    detail = re.sub(r"\s+", " ", detail).strip()
    detail = _sanitize_failure_detail(detail)
    if len(detail) > max_chars:
        return detail[: max_chars - 3].rstrip() + "..."
    return detail


def _sanitize_failure_detail(detail: str) -> str:
    detail = re.sub(r"/(?:root|vePFS-Mindverse|tmp|mnt)/\S+", "<local_path>", detail)
    detail = re.sub(
        r"https?://(?:localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?::\d+)?\S*",
        "<internal_url>",
        detail,
    )
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "<redacted_key>", detail)
    return detail


def classify_failure_detail(text: str, *, exit_code: int) -> FailureClassification:
    if exit_code == 0:
        return FailureClassification(None)
    lowered = text.lower()
    if "model_path must start with" in lowered or "checkpoint model_path must start with" in lowered:
        return FailureClassification(
            "client workflow",
            "checkpoint URI was rejected by the sanity-check client workflow",
        )
    if "valueerror" in lowered and "create_sampling_client" in lowered:
        return FailureClassification(
            "client workflow",
            _compact_error_detail(text) or "create_sampling_client raised ValueError",
        )
    if "sampling session terminated due to sampling inactivity" in lowered:
        return FailureClassification(
            "capacity/scheduling",
            "sampling session exceeded inactivity TTL while waiting for queued sampling work",
        )
    if "no_resources" in lowered or "placement group" in lowered or "scheduling" in lowered:
        return FailureClassification("capacity/scheduling", _compact_error_detail(text))
    if "cuda out of memory" in lowered or "actordiederror" in lowered or "enginedeaderror" in lowered:
        return FailureClassification("server exception", _compact_error_detail(text))
    if "worker failed" in lowered or "requestfailederror" in lowered or "traceback" in lowered:
        return FailureClassification("server exception", _compact_error_detail(text))
    if "api key" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        return FailureClassification("client env/auth", _compact_error_detail(text))
    return FailureClassification("unknown", _compact_error_detail(text) or "unmatched failure; inspect stdout/stderr artifacts")


def classify_failure(text: str, *, exit_code: int) -> str | None:
    return classify_failure_detail(text, exit_code=exit_code).failure_class


def failure_surface_from_logs(text: str) -> str | None:
    lowered = text.lower()
    if "rl sanity did not complete requested steps" in lowered or "fail in rl_step_not_completed" in lowered:
        return "rl_step_not_completed"
    if "model_path must start with" in lowered or "checkpoint model_path must start with" in lowered:
        return "create_sampling_client"
    if "valueerror" in lowered and "create_sampling_client" in lowered:
        return "create_sampling_client"
    patterns = [
        r"FAIL in `?([A-Za-z0-9_./:-]+)`?",
        r"Failure surface:\s*([^.\n]+)",
        r"waiting label=([A-Za-z_]+)",
        r"request_type=([A-Za-z0-9_]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).strip()
    if "preflight" in text.lower():
        return "preflight"
    return None


def _failure_surface(result: dict[str, object]) -> str:
    if result.get("status") == "ok":
        return "completed"
    if result.get("failure_surface"):
        return str(result["failure_surface"])
    if result.get("slowest_stage"):
        return f"after_{result['slowest_stage']}"
    if result.get("request_ids"):
        return "request_failed"
    return "unknown"


def preflight_failure_results(models: list[str], message: str) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "slug": model_slug(model),
            "exit_code": 2,
            "status": "fail",
            "run_dir": None,
            "stdout_log": None,
            "stderr_log": None,
            "experiment_dir": None,
            "request_ids": {},
            "session_ids": {},
            "failure_class": "client env/auth",
            "failure_surface": "preflight",
            "failure_detail": message,
            "timing_summary_json": None,
            "timing_summary_md": None,
            "timing_events_jsonl": None,
            "wall_clock_s": 0.0,
            "slowest_stage": "preflight",
            "slowest_max_s": 0.0,
            "generation_summary": None,
            "timing_degraded": False,
            "degradation_reason": None,
            "started_at_epoch_s": time.time(),
            "finished_at_epoch_s": time.time(),
        }
        for model in models
    ]


def build_feishu_report(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    any_failed = False
    ok_count = sum(1 for result in results if result.get("status") == "ok")
    degraded_count = sum(1 for result in results if result.get("status") == "ok" and result.get("timing_degraded"))
    status = "PASS" if ok_count == len(results) else "FAIL"
    if status == "PASS" and degraded_count:
        status = "PASS_WITH_DEGRADATION"
    lines.append(f"**Result:** {status} ({ok_count}/{len(results)} models passed)")
    lines.append("")
    lines.append("**Model timing**")
    for result in results:
        model = result["model"]
        slowest = result.get("slowest_stage") or "unknown"
        max_s = _fmt_duration(result.get("slowest_max_s"))
        wall = _fmt_duration(result.get("wall_clock_s"))
        if result.get("status") == "ok":
            label = "DEGRADED" if result.get("timing_degraded") else "PASS"
            generation = result.get("generation_summary")
            gen_text = ""
            if isinstance(generation, dict) and isinstance(generation.get("tokens_per_s"), (int, float)):
                gen_text = (
                    f" generated_tokens=`{generation.get('output_tokens')}`, "
                    f"gen_tok_s=`{float(generation['tokens_per_s']):.2f}`"
                )
                if generation.get("hit_max_count"):
                    gen_text += f", hit_max=`{generation['hit_max_count']}`"
            degraded_text = ""
            if result.get("degradation_reason"):
                degraded_text = f" degradation=`{result['degradation_reason']}`"
            lines.append(
                f"- {label} `{model}`: slowest=`{slowest}`, max=`{max_s}`, wall=`{wall}`"
                f"{gen_text}{degraded_text}."
            )
        else:
            any_failed = True
            failure_class = result.get("failure_class") or "unknown"
            line = (
                f"- FAIL `{model}`: failed=`{_failure_surface(result)}`, class=`{failure_class}`, "
                f"slowest_completed=`{slowest}`, max=`{max_s}`, wall=`{wall}`."
            )
            if result.get("failure_detail"):
                line += f" detail={result['failure_detail']}"
            lines.append(line)
    lines.append("")
    lines.append("**Ops:** none attempted by wrapper.")
    lines.append("**Issue:** not filed by wrapper.")
    if any_failed:
        details = [
            str(result.get("failure_detail"))
            for result in results
            if result.get("failure_detail")
        ]
        if details:
            lines.append(f"**Next:** {details[0]}")
        else:
            lines.append("**Next:** preserve artifacts, classify with logs/telemetry, remediate minimally, then rerun the full matrix.")
    else:
        lines.append("**Next:** no action required.")
    return "\n".join(lines)


def send_feishu_report(markdown: str) -> None:
    subprocess.run(
        [
            sys.executable,
            ".claude/skills/sanity-check/feishu_notify.py",
            "--title",
            FEISHU_TITLE,
            "--markdown",
            markdown,
        ],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        check=True,
    )


def write_summary(results: list[dict[str, object]], run_root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    json_path = Path(args.summary_json) if args.summary_json else run_root / "summary.json"
    md_path = Path(args.summary_md) if args.summary_md else run_root / "summary.md"

    payload = {
        "base_url": args.base_url,
        "num_models": len(results),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Train Check Summary",
        "",
        f"- base_url: `{args.base_url}`",
        f"- num_models: `{len(results)}`",
        "",
    ]
    for result in results:
        lines.append(f"## {result['model']}")
        lines.append(f"- status: `{result['status']}`")
        lines.append(f"- exit_code: `{result['exit_code']}`")
        if result.get("slowest_stage") is not None:
            lines.append(
                f"- timing: slowest stage=`{result['slowest_stage']}` max_s=`{result['slowest_max_s']}` wall_clock_s=`{result['wall_clock_s']}`"
            )
        if result.get("timing_degraded"):
            lines.append(f"- timing_degraded: `{result.get('degradation_reason')}`")
        if result.get("generation_summary"):
            lines.append(f"- generation_summary: `{json.dumps(result['generation_summary'], sort_keys=True)}`")
        session_ids = result.get("session_ids") or {}
        request_ids = result.get("request_ids") or {}
        lines.append(f"- session_ids: `{json.dumps(session_ids, sort_keys=True)}`")
        lines.append(f"- request_ids: `{json.dumps(request_ids, sort_keys=True)}`")
        lines.append(f"- stdout_log: `{result['stdout_log']}`")
        lines.append(f"- stderr_log: `{result['stderr_log']}`")
        if result.get("timing_summary_json"):
            lines.append(f"- timing_summary_json: `{result['timing_summary_json']}`")
        if result.get("failure_detail"):
            lines.append(f"- failure_detail: `{result['failure_detail']}`")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def write_outputs_and_maybe_notify(
    results: list[dict[str, object]],
    run_root: Path,
    args: argparse.Namespace,
    *,
    default_send_feishu: bool,
) -> bool:
    json_path, md_path = write_summary(results, run_root, args)
    feishu_report = build_feishu_report(results)
    feishu_report_path = run_root / "final_feishu_report.md"
    feishu_report_path.write_text(feishu_report + "\n")
    print(f"[summary] json={json_path}")
    print(f"[summary] md={md_path}")
    print(f"[summary] final_feishu_report={feishu_report_path}")

    should_send_feishu = bool(args.feishu) if args.feishu is not None else default_send_feishu
    if not should_send_feishu:
        return True
    try:
        send_feishu_report(feishu_report)
    except subprocess.CalledProcessError as exc:
        print(f"[summary] Feishu report failed: {exc}", file=sys.stderr)
        return False
    return True


def print_dry_run(runs: list[ModelRun], sequential: bool) -> None:
    mode = "sequential" if sequential or len(runs) <= 1 else "parallel"
    print(f"[dry-run] mode={mode} models={len(runs)}")
    for run in runs:
        print(f"[dry-run] model={run.model}")
        print(f"[dry-run] run_dir={run.run_dir}")
        print(f"[dry-run] MINT_TEST_EXPERIMENT_ROOT={run.env['MINT_TEST_EXPERIMENT_ROOT']}")
        print("[dry-run] command=" + " ".join(subprocess.list2cmdline([part]) for part in run.command))


def main() -> int:
    args = parse_args()
    ensure_runner_exists()
    load_env_file(Path(".secrets.env"))
    if args.all_models and args.parallel:
        raise SystemExit("--all-models must run sequentially for production sanity-check")

    run_root = make_run_root(Path(args.results_root), args.run_name, create=not args.dry_run)
    models = selected_models(args)
    sequential = bool(args.sequential or args.all_models or not args.parallel)

    try:
        api_key = validate_production_env(args.base_url)
    except SystemExit as exc:
        message = str(exc)
        if args.dry_run or not args.all_models:
            raise
        results = preflight_failure_results(models, message)
        if not write_outputs_and_maybe_notify(
            results, run_root, args, default_send_feishu=True
        ):
            return 3
        print(f"[summary] preflight failed: {message}", file=sys.stderr)
        return 2

    runs = build_runs(args, run_root, create_dirs=not args.dry_run)

    if args.dry_run:
        print_dry_run(runs, sequential)
        return 0

    if not args.skip_preflight:
        try:
            preflight(args.base_url, api_key)
        except (HTTPError, URLError, TimeoutError) as exc:
            if not args.all_models:
                raise SystemExit(f"HTTP preflight failed: {exc}") from exc
            results = preflight_failure_results(models, f"HTTP preflight failed: {exc}")
            if not write_outputs_and_maybe_notify(
                results, run_root, args, default_send_feishu=True
            ):
                return 3
            print(f"[summary] preflight failed: {exc}", file=sys.stderr)
            return 2

    results = run_parallel(runs, sequential)
    if not write_outputs_and_maybe_notify(
        results, run_root, args, default_send_feishu=bool(args.all_models)
    ):
        return 3

    failed = [result for result in results if result["exit_code"] != 0]
    if failed:
        print(f"[summary] failed_models={len(failed)}", file=sys.stderr)
        return 1
    print("[summary] all models passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
