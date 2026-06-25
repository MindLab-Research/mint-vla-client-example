#!/usr/bin/env python3
"""Run the RL sanity-check matrix against an explicit SGLang Mint server.

This is intentionally separate from scripts/wip/check.sh. The canonical
production sanity wrapper must keep forcing https://mint.macaron.xin; this
helper is for feature/issue-scoped SGLang servers that select
serving_backend="sglang" through server-side model config.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


PRODUCTION_URLS = {
    "https://mint.macaron.xin",
    "https://mint.macaron.im",
}
SDK_SHIM_PATH = Path(__file__).resolve().parents[1] / "wip" / "mint_sdk_shim"


def _load_train_check():
    # The shared RL sanity runner still lives beside the canonical production
    # wrapper. Keep this SGLang entry point small and avoid duplicating it.
    path = Path(__file__).resolve().parents[1] / "wip" / "train_check.py"
    spec = importlib.util.spec_from_file_location("mint_train_check_for_sglang", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load train_check module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    train_check = _load_train_check()
    parser = argparse.ArgumentParser(
        description="Run MinT RL sanity checks against a server configured for SGLang sampling."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model aliases or full model names. Aliases match the shared train_check runner.",
    )
    parser.add_argument("--model", dest="models_flag", action="append", default=[])
    parser.add_argument("--all-models", action="store_true", help="Run the standard 5-model matrix.")
    parser.add_argument("--num-rl-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Forward an explicit LoRA rank to the underlying train_check runner.",
    )
    parser.add_argument(
        "--train-mlp",
        dest="train_mlp",
        action="store_true",
        default=None,
        help="Forward --train-mlp to the underlying train_check runner.",
    )
    parser.add_argument(
        "--no-train-mlp",
        dest="train_mlp",
        action="store_false",
        help="Forward --no-train-mlp to the underlying train_check runner.",
    )
    parser.add_argument(
        "--train-attn",
        dest="train_attn",
        action="store_true",
        default=None,
        help="Forward --train-attn to the underlying train_check runner.",
    )
    parser.add_argument(
        "--no-train-attn",
        dest="train_attn",
        action="store_false",
        help="Forward --no-train-attn to the underlying train_check runner.",
    )
    parser.add_argument(
        "--train-unembed",
        dest="train_unembed",
        action="store_true",
        default=None,
        help="Forward --train-unembed to the underlying train_check runner.",
    )
    parser.add_argument(
        "--no-train-unembed",
        dest="train_unembed",
        action="store_false",
        help="Forward --no-train-unembed to the underlying train_check runner.",
    )
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MINT_BASE_URL", "http://localhost:8000"),
        help="Explicit SGLang-configured Mint server URL.",
    )
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "dummy"))
    parser.add_argument(
        "--checkpoint-owner-id",
        default=None,
        help="Optional owner ObjectId for production-style checkpoint URIs. Omitted by default for dev/issue servers.",
    )
    parser.add_argument(
        "--resolve-dev-sampler-uris",
        action="store_true",
        default=os.environ.get("MINT_TEST_RESOLVE_SAMPLER_TINKER_URI_TO_FILE") == "1",
        help="Resolve returned tinker:// sampler_weights URIs to dev runtime file:// adapter paths.",
    )
    parser.add_argument(
        "--runtime-checkpoint-root",
        default=os.environ.get(
            "MINT_TEST_RUNTIME_CHECKPOINT_ROOT",
            "/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints",
        ),
        help="Runtime checkpoint root used by --resolve-dev-sampler-uris.",
    )
    parser.add_argument("--results-root", default="/root/run_results/mint-sglang")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument(
        "--allow-production-url",
        action="store_true",
        help="Allow a production URL. This is off by default so feature validation cannot masquerade as canonical prod sanity.",
    )
    parser.set_defaults(_train_check=train_check)
    return parser.parse_args()


def _normal_base_url(raw: object) -> str:
    return str(raw or "").strip().rstrip("/")


def _sdk_compatible_api_key(raw: object) -> str:
    api_key = str(raw or "").strip()
    if api_key == "dummy":
        return "tml-dummy"
    return api_key


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"SDK compatibility shim not found: {path}")
    existing = env.get("PYTHONPATH", "")
    parts = [str(path)]
    parts.extend(part for part in existing.split(os.pathsep) if part and part != str(path))
    env["PYTHONPATH"] = os.pathsep.join(parts)


def validate_sglang_target(args: argparse.Namespace) -> str:
    base_url = _normal_base_url(args.base_url)
    if not base_url:
        raise SystemExit("missing --base-url or MINT_BASE_URL")
    if base_url in PRODUCTION_URLS and not bool(args.allow_production_url):
        raise SystemExit(
            f"{Path(__file__).name} refuses production URL {base_url}; "
            "use scripts/wip/check.sh for canonical production sanity, or pass "
            "--allow-production-url only for an explicitly configured SGLang production deployment."
        )
    return base_url


def selected_models(args: argparse.Namespace) -> list[str]:
    return args._train_check.selected_models(args)


def _is_qwen3_moe_model(model: str) -> bool:
    normalized = str(model or "").lower()
    return "qwen3-" in normalized and ("-a3b" in normalized or "-a22b" in normalized)


def _default_lora_switches_for_model(model: str) -> dict[str, bool]:
    if _is_qwen3_moe_model(model):
        return {
            "train_unembed": False,
        }
    return {}


def _append_optional_runner_args(command: list[str], args: argparse.Namespace, *, model: str) -> None:
    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is not None:
        command.append(f"--lora-rank={int(lora_rank)}")

    defaults = _default_lora_switches_for_model(model)
    for attr, positive_flag, negative_flag in (
        ("train_mlp", "--train-mlp", "--no-train-mlp"),
        ("train_attn", "--train-attn", "--no-train-attn"),
        ("train_unembed", "--train-unembed", "--no-train-unembed"),
    ):
        value = getattr(args, attr, None)
        if value is None:
            value = defaults.get(attr)
        if value is True:
            command.append(positive_flag)
        elif value is False:
            command.append(negative_flag)


def build_sglang_runs(args: argparse.Namespace, run_root: Path, *, create_dirs: bool = True):
    train_check = args._train_check
    base_url = validate_sglang_target(args)
    base_env = os.environ.copy()
    _prepend_pythonpath(base_env, SDK_SHIM_PATH)
    base_env["MINT_BASE_URL"] = base_url
    base_env["TINKER_BASE_URL"] = base_url
    base_env.setdefault("PYTHONUNBUFFERED", "1")
    api_key = _sdk_compatible_api_key(args.api_key)
    if api_key:
        base_env["MINT_API_KEY"] = api_key
        base_env["TINKER_API_KEY"] = api_key
    else:
        base_env.pop("MINT_API_KEY", None)
        base_env.pop("TINKER_API_KEY", None)
    if args.checkpoint_owner_id:
        base_env["MINT_TEST_CHECKPOINT_OWNER_ID"] = str(args.checkpoint_owner_id)
    else:
        base_env.pop("MINT_TEST_CHECKPOINT_OWNER_ID", None)
    if getattr(args, "resolve_dev_sampler_uris", False):
        base_env["MINT_TEST_RESOLVE_SAMPLER_TINKER_URI_TO_FILE"] = "1"
        base_env["MINT_TEST_RUNTIME_CHECKPOINT_ROOT"] = str(
            getattr(
                args,
                "runtime_checkpoint_root",
                "/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints",
            )
        )
    else:
        base_env.pop("MINT_TEST_RESOLVE_SAMPLER_TINKER_URI_TO_FILE", None)
    base_env["MINT_SANITY_TARGET_BACKEND"] = "sglang"

    runs = []
    for model in selected_models(args):
        slug = train_check.model_slug(model)
        model_dir = run_root / slug
        if create_dirs:
            model_dir.mkdir(parents=True, exist_ok=True)
        env = base_env.copy()
        env["MINT_TEST_EXPERIMENT_ROOT"] = str(model_dir)
        command = [
            sys.executable,
            str(train_check.RUNNER),
            f"--model={model}",
            f"--num-rl-steps={args.num_rl_steps}",
            f"--batch-size={args.batch_size}",
            f"--group-size={args.group_size}",
            f"--max-tokens={args.max_tokens}",
        ]
        _append_optional_runner_args(command, args, model=model)
        command.append(f"--timeout-s={args.timeout_s}")
        runs.append(train_check.ModelRun(model=model, slug=slug, run_dir=model_dir, command=command, env=env))
    return runs


def _report_with_backend(results: list[dict[str, object]], train_check) -> str:
    report = train_check.build_feishu_report(results)
    return "**Target backend:** sglang\n\n" + report


def _write_outputs(results: list[dict[str, object]], run_root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train_check = args._train_check
    summary_args = SimpleNamespace(
        base_url=validate_sglang_target(args),
        summary_json=args.summary_json,
        summary_md=args.summary_md,
    )
    json_path, md_path = train_check.write_summary(results, run_root, summary_args)

    payload = json.loads(json_path.read_text())
    payload["target_backend"] = "sglang"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    md_text = md_path.read_text()
    if "- target_backend:" not in md_text:
        md_lines = md_text.splitlines()
        insert_at = 1
        if len(md_lines) > 1 and md_lines[1] == "":
            insert_at = 2
        md_lines.insert(insert_at, "- target_backend: `sglang`")
        md_path.write_text("\n".join(md_lines) + "\n")

    report = _report_with_backend(results, train_check)
    report_path = run_root / "final_sglang_report.md"
    report_path.write_text(report + "\n")
    print(f"[summary] json={json_path}")
    print(f"[summary] md={md_path}")
    print(f"[summary] final_sglang_report={report_path}")
    print(report)
    return json_path, md_path, report_path


def _print_dry_run(runs: list[object], args: argparse.Namespace) -> None:
    print(f"[dry-run] target_backend=sglang base_url={validate_sglang_target(args)} models={len(runs)}")
    for run in runs:
        print(f"[dry-run] model={run.model}")
        print(f"[dry-run] run_dir={run.run_dir}")
        print("[dry-run] command=" + " ".join(args._train_check.subprocess.list2cmdline([part]) for part in run.command))


def main() -> int:
    args = _parse_args()
    train_check = args._train_check

    train_check.ensure_runner_exists()
    base_url = validate_sglang_target(args)
    os.environ["MINT_BASE_URL"] = base_url
    api_key = _sdk_compatible_api_key(args.api_key)
    if api_key:
        os.environ["MINT_API_KEY"] = api_key
        os.environ["TINKER_API_KEY"] = api_key

    run_root = train_check.make_run_root(Path(args.results_root), args.run_name, create=not args.dry_run)
    runs = build_sglang_runs(args, run_root, create_dirs=not args.dry_run)

    if args.dry_run:
        _print_dry_run(runs, args)
        return 0

    if not args.skip_preflight:
        train_check.preflight(base_url, api_key)

    results = train_check.run_parallel(runs, sequential=True)
    _write_outputs(results, run_root, args)

    failed = [result for result in results if result.get("exit_code") != 0]
    if failed:
        print(f"[summary] failed_models={len(failed)}", file=sys.stderr)
        return 1
    print("[summary] all SGLang-targeted models passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
