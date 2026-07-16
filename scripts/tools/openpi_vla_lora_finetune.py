#!/usr/bin/env python3
"""Production LoRA fine-tune driver for OpenPI pi0.5 VLA models over HTTP.

Given a Lance dataset path, runs the full mint-server flow against a
Ray-free (or Ray) mint-server instance:

    create_model -> train_step (x N) -> save_weights_for_sampler -> [optional inference check] -> cleanup

This is the *productized* counterpart to ``scripts/wip/openpi_vla_smoke_lance.py``.
It imports the reusable dataset/transform/HTTP plumbing from that smoke
script (dataset windowing, OpenPI transform pipeline, mint wire-format
conversion, future polling) rather than duplicating it, but fixes three
things that make the smoke script unsuitable as an end-user tool:

  1. ``--base-model`` is configurable (validated against MODEL_CONFIGS)
     instead of being locked to a single ``choices=[PI05_MODEL]`` value.
  2. LoRA rank/train-flags are configurable instead of hardcoded
     (rank=16, train_attn/mlp/unembed=True).
  3. "save the checkpoint" and "run a post-training inference smoke check"
     are independent flags (``--skip-save`` / ``--skip-inference-check``)
     instead of being bundled behind a single ``--skip-action`` switch.

It also adds two checks the smoke script does not perform:

  - ``validate_action_dim()``: hard-stops *before* any expensive work
    (dataset load, server round-trips) if the Lance dataset's per-frame
    state/action vector length does not exactly match the target
    base_model's configured ``action_dim`` in
    ``mint_server/backend/core/model_registry.py``. Zero-padding or
    masking a dimension mismatch is a documented dead end (see
    ``ActionHeadSummary.md`` in the repo root) -- do not "fix" this
    check by padding data to fit.
  - ``probe_lance_dataset()``: cheap read-probe before the full table
    load, so a dataset whose latest version has missing/corrupt
    fragments fails fast with a suggested ``--lance-dataset-version``
    instead of surfacing a confusing mid-run Arrow error.

Usage:
    python scripts/tools/openpi_vla_lora_finetune.py \\
        --lance-dataset /path/to/data.lance \\
        --base-model openpi/pi05-libero-low-mem-finetune \\
        --steps 400 --batch-size 2 --lora-rank 16 \\
        --save-checkpoint-name my_run_v1

    # Dry run: validate action_dim + dataset readability only, no network.
    python scripts/tools/openpi_vla_lora_finetune.py \\
        --lance-dataset /path/to/data.lance --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests

# --------------------------------------------------------------------------- #
# Import reusable pieces from the smoke script without copy-pasting them.
# The smoke script lives in scripts/wip/ and is not itself an importable
# package (no __init__.py in wip/), so we load it by file path.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "wip" / "openpi_vla_smoke_lance.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_EVAL_MSE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "tools" / "openpi_vla_eval_mse.py"
_INFER_TO_LANCE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "tools" / "openpi_vla_infer_to_lance.py"


def _load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_module():
    return _load_module_by_path("openpi_vla_smoke_lance", _SMOKE_SCRIPT_PATH)


_smoke = _load_smoke_module()
_eval_mse = _load_module_by_path("openpi_vla_eval_mse", _EVAL_MSE_SCRIPT_PATH)
run_mse_evaluation = _eval_mse.run_mse_evaluation

_infer_to_lance = _load_module_by_path("openpi_vla_infer_to_lance", _INFER_TO_LANCE_SCRIPT_PATH)
run_inference_and_merge_to_lance = _infer_to_lance.run_inference_and_merge_to_lance

# Reused directly, unmodified, from the smoke script.
LanceViewpi05Dataset = _smoke.LanceViewpi05Dataset
_headers = _smoke._headers
_post_json = _smoke._post_json
_get_json = _smoke._get_json
_await_result = _smoke._await_result
_build_model_config = _smoke._build_model_config
_make_data_config = _smoke._make_data_config
_compute_norm_stats = _smoke._compute_norm_stats
_build_batch = _smoke._build_batch
_delete_model = _smoke._delete_model

from mint_server.backend.core.model_registry import MODEL_CONFIGS  # noqa: E402
from mint_server.backend.openpi.openpi_pi05_training import (  # noqa: E402
    OPENPI_PI05_LORA_RANK,
)


# --------------------------------------------------------------------------- #
# LoRA config pre-flight warning (NOT a hard stop). As of this writing the
# server hard-rejects (HTTP 400 on create_model) any openpi_pi05 request whose
# LoRA rank isn't exactly OPENPI_PI05_LORA_RANK, or whose
# train_attn/train_mlp/train_unembed aren't all True (see
# validate_openpi_pi05_create_request in
# mint_server/backend/openpi/openpi_pi05_training.py). That constraint may be
# relaxed server-side in the future (tracked as an improvement target in
# RECURSIVE.md), so this function only *warns* and still sends whatever the
# user asked for -- it does not block the request. If the server still
# rejects it, main() surfaces the server's own error text (see
# _create_model_with_lora's error handling) rather than guessing.
# --------------------------------------------------------------------------- #
def validate_lora_config(
    base_model: str,
    *,
    lora_rank: int,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> None:
    model_cfg = MODEL_CONFIGS.get(base_model)
    if model_cfg is None or model_cfg.training_backend != "openpi_pi05":
        return  # constraint only applies to the openpi_pi05 backend

    if lora_rank != OPENPI_PI05_LORA_RANK:
        print(
            f"warning: --lora-rank={lora_rank} != the server's currently-enforced "
            f"rank={OPENPI_PI05_LORA_RANK} for openpi_pi05 (see "
            "validate_openpi_pi05_create_request in "
            "mint_server/backend/openpi/openpi_pi05_training.py). Sending it anyway -- "
            "if the server still rejects it, that constraint has not been relaxed yet."
        )
    for name, value in (("train_attn", train_attn), ("train_mlp", train_mlp), ("train_unembed", train_unembed)):
        if value is not True:
            print(
                f"warning: --lora-{name.replace('_', '-')}={value} != the server's "
                f"currently-enforced {name}=True for openpi_pi05. Sending it anyway -- "
                "if the server still rejects it, that constraint has not been relaxed yet."
            )


# --------------------------------------------------------------------------- #
# New: action_dim validation (fixes the gap where the smoke script checks
# action_horizon but not action_dim).
# --------------------------------------------------------------------------- #
def validate_action_dim(
    lance_dataset: str,
    base_model: str,
    *,
    action_horizon: int,
    lance_dataset_version: int | None = None,
) -> int:
    """Hard-stop if the dataset's state/action vector length doesn't match
    the model's configured action_dim.

    Returns the dataset's action_dim on success.
    """
    if base_model not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise SystemExit(f"unknown --base-model {base_model!r}. Known models: {known}")

    model_cfg = MODEL_CONFIGS[base_model]
    if model_cfg.action_dim is None:
        raise SystemExit(
            f"base_model {base_model!r} has no action_dim configured in model_registry.py; "
            "this driver only supports flow_action / actions-modality models"
        )

    dataset = _open_lance_dataset(lance_dataset, action_horizon=action_horizon, version=lance_dataset_version)
    sample = dataset[0]
    dataset_action_dim = int(sample["observation/state"].shape[0])

    if dataset_action_dim != model_cfg.action_dim:
        raise SystemExit(
            f"action_dim mismatch: dataset {lance_dataset!r} has {dataset_action_dim}-dim "
            f"state/actions, but base_model {base_model!r} is configured with "
            f"action_dim={model_cfg.action_dim} in model_registry.py.\n"
            "This is a hard stop, not a warning: zero-padding a mismatched dimension or "
            "masking it out of the loss does NOT work for this model (the Diffusion target "
            "on padded dims collapses to pure noise, and forward-pass attention still leaks "
            "the padding into every other dimension's hidden state regardless of masking). "
            "See ActionHeadSummary.md in the repo root for the experiments that established "
            "this.\n"
            f"Fix: either use a base_model with action_dim={dataset_action_dim}, or "
            "point --lance-dataset at data with the matching dimensionality."
        )

    if model_cfg.action_horizon is not None and action_horizon != model_cfg.action_horizon:
        print(
            f"warning: --action-horizon={action_horizon} != "
            f"model action_horizon={model_cfg.action_horizon}; the model's positional "
            "encodings were trained for its own horizon length."
        )

    return dataset_action_dim


# --------------------------------------------------------------------------- #
# New: cheap readability probe before the full table load, with automatic
# fallback-version discovery on failure.
# --------------------------------------------------------------------------- #
def probe_lance_dataset(lance_dataset: str, *, requested_version: int | None = None) -> None:
    """Read-probe a Lance dataset before doing any real work.

    If the requested (or latest) version fails to read, walks backwards
    through .versions() to find the newest version that actually round-trips,
    and raises with that version number as a suggested --lance-dataset-version.
    Does not silently fall back itself -- the caller must decide, since
    silently picking stale data could hide a real problem.
    """
    import lance

    try:
        ds = (
            lance.dataset(lance_dataset)
            if requested_version is None
            else lance.dataset(lance_dataset, version=requested_version)
        )
        ds.to_table(limit=1)
        return
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
        probe_error = exc

    version_desc = "latest" if requested_version is None else f"version {requested_version}"
    print(f"warning: failed to read {version_desc} of {lance_dataset!r}: {probe_error}")
    print("probing earlier versions for one that round-trips cleanly...")

    ds = lance.dataset(lance_dataset)
    versions = ds.versions()
    for v in reversed(versions):
        vnum = v["version"]
        if requested_version is not None and vnum >= requested_version:
            continue
        try:
            ds_v = lance.dataset(lance_dataset, version=vnum)
            ds_v.to_table(limit=1)
            raise SystemExit(
                f"dataset {lance_dataset!r} failed to read at {version_desc}, but version "
                f"{vnum} ({ds_v.count_rows()} rows) reads cleanly.\n"
                f"Re-run with: --lance-dataset-version {vnum}\n"
                "Do not assume this older version has the data you expect -- confirm the "
                "row count and content are what you intended before training on it."
            )
        except Exception:  # noqa: BLE001
            continue

    raise SystemExit(
        f"dataset {lance_dataset!r} has no readable version at all (tried {len(versions)} "
        "versions). The dataset is likely still being written/synced, or its fragment "
        "files were deleted/moved. This is not something this driver can fix -- resolve "
        "the underlying storage issue first."
    )


def _open_lance_dataset(
    lance_dataset: str,
    *,
    action_horizon: int,
    max_samples: int | None = None,
    version: int | None = None,
) -> "LanceViewpi05Dataset":
    """Open LanceViewpi05Dataset at a specific version.

    LanceViewpi05Dataset.__init__ always opens the latest version (it calls
    lance.dataset(path) with no version kwarg). To honor --lance-dataset-version
    without forking that class, monkeypatch lance.dataset for the duration of
    construction only, restoring it immediately after (success or failure).
    """
    if version is None:
        return LanceViewpi05Dataset(Path(lance_dataset), action_horizon=action_horizon, max_samples=max_samples)

    import lance

    original_dataset_fn = lance.dataset

    def _pinned_dataset(path, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        kwargs.setdefault("version", version)
        return original_dataset_fn(path, *args, **kwargs)

    lance.dataset = _pinned_dataset
    try:
        return LanceViewpi05Dataset(Path(lance_dataset), action_horizon=action_horizon, max_samples=max_samples)
    finally:
        lance.dataset = original_dataset_fn


# --------------------------------------------------------------------------- #
# New: configurable create_model (smoke script hardcodes lora_config to
# rank=16 / train_attn=train_mlp=train_unembed=True).
# --------------------------------------------------------------------------- #
def _create_model_with_lora(
    base_url: str,
    headers: dict[str, str],
    *,
    base_model: str,
    lora_rank: int,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "session_id": f"vla-lora-{uuid.uuid4().hex[:12]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {
            "rank": lora_rank,
            "train_attn": train_attn,
            "train_mlp": train_mlp,
            "train_unembed": train_unembed,
        },
    }
    try:
        post_result = _post_json(base_url, "/api/v1/create_model", headers, payload)
    except requests.exceptions.HTTPError as exc:
        # requests' default HTTPError message is just "400 Client Error: Bad
        # Request" -- the actually-useful text is the response body's
        # "detail" field (FastAPI's HTTPException shape). Surface that
        # instead of making the caller re-derive it from a bare status code.
        detail = None
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail")
            except Exception:  # noqa: BLE001
                pass
        if detail:
            raise SystemExit(
                f"create_model rejected by server for lora_config={payload['lora_config']}: {detail}"
            ) from exc
        raise
    create_result = _await_result(base_url, headers, post_result)
    # The server appends a suffix to session_id to form the real model_id
    # (e.g. "vla-lora-abc123" -> "vla-lora-abc123_0") -- always read it back
    # from the response rather than assuming it matches what we sent.
    model_id = create_result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {create_result!r}")
    return model_id, create_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument(
        "--base-model",
        default=_smoke.PI05_MODEL,
        help=f"any key present in model_registry.MODEL_CONFIGS (default: {_smoke.PI05_MODEL})",
    )
    parser.add_argument("--lance-dataset", required=True)
    parser.add_argument(
        "--lance-dataset-version",
        type=int,
        default=None,
        help="pin a specific Lance dataset version instead of the latest (see probe_lance_dataset)",
    )
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank. The openpi_pi05 server currently only accepts 16 (as of "
        "this writing) -- other values are sent anyway and will be rejected by the "
        "server with a 400 if that constraint hasn't been relaxed yet. See "
        "RECURSIVE.md for the improvement-target status of this constraint.",
    )
    parser.add_argument("--lora-train-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-train-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-train-unembed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-checkpoint-name",
        default=None,
        help="path/name for save_weights_for_sampler; omit for an auto-generated name",
    )
    parser.add_argument("--skip-save", action="store_true", help="skip save_weights_for_sampler entirely")
    parser.add_argument(
        "--skip-inference-check",
        action="store_true",
        help="skip the post-save action_session/act smoke check (independent of --skip-save; "
        "has no effect if --skip-save is also set, since there is no saved checkpoint to check)",
    )
    parser.add_argument(
        "--eval-mse",
        action="store_true",
        help="after save_weights_for_sampler, run a quantitative MSE/L1 evaluation "
        "(pred vs ground-truth actions, both in normalized space, plus a zero-prediction "
        "baseline for comparison) via openpi_vla_eval_mse.run_mse_evaluation. Has no effect "
        "if --skip-save is also set. Independent of --skip-inference-check (that's a single "
        "smoke-test act() call; this is a proper quantitative metric over --eval-mse-indices).",
    )
    parser.add_argument(
        "--eval-mse-indices",
        default="0,1,2,5,10,20,50,100",
        help="comma-separated dataset indices to evaluate when --eval-mse is set",
    )
    parser.add_argument(
        "--infer-to-lance",
        action="store_true",
        help="after save_weights_for_sampler, run inference on every frame in the dataset "
        "and write a new Lance dataset (--infer-to-lance-output) that keeps all original "
        "columns plus 4 appended per-frame columns: pred_actions (normalized), "
        "pred_actions_physical (unnormalized), pred_action_mse, pred_meta. Via "
        "openpi_vla_infer_to_lance.run_inference_and_merge_to_lance. This infers on the "
        "FULL dataset (not a sample), so it can take a while for large datasets. Has no "
        "effect if --skip-save is also set.",
    )
    parser.add_argument(
        "--infer-to-lance-output",
        default=None,
        help="output path for --infer-to-lance; required if --infer-to-lance is set",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate action_dim + dataset readability only; no network calls",
    )
    args = parser.parse_args()

    if args.infer_to_lance and not args.infer_to_lance_output:
        raise SystemExit("--infer-to-lance requires --infer-to-lance-output <path>")

    # --- Fail fast, before touching the network or loading the full table. --- #
    validate_lora_config(
        args.base_model,
        lora_rank=args.lora_rank,
        train_attn=args.lora_train_attn,
        train_mlp=args.lora_train_mlp,
        train_unembed=args.lora_train_unembed,
    )
    probe_lance_dataset(args.lance_dataset, requested_version=args.lance_dataset_version)
    dataset_action_dim = validate_action_dim(
        args.lance_dataset,
        args.base_model,
        action_horizon=args.action_horizon,
        lance_dataset_version=args.lance_dataset_version,
    )
    print(f"action_dim check passed: dataset and {args.base_model!r} both use {dataset_action_dim} dims")

    if args.dry_run:
        dataset = _open_lance_dataset(
            args.lance_dataset,
            action_horizon=args.action_horizon,
            max_samples=args.max_samples,
            version=args.lance_dataset_version,
        )
        print(f"lance_dataset: {args.lance_dataset}")
        print(f"samples(frame windows): {len(dataset)}  action_horizon={args.action_horizon}")
        sample = dataset[0]
        for key, value in sample.items():
            shape = getattr(value, "shape", None)
            print(f"  {key}: shape={shape}" if shape is not None else f"  {key}: {type(value).__name__}")
        print("dry-run OK (no network calls made)")
        return 0

    headers = _headers(args.api_key)
    dataset = _open_lance_dataset(
        args.lance_dataset,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        version=args.lance_dataset_version,
    )
    model_cfg = _build_model_config(args.action_horizon, action_dim=dataset_action_dim)
    norm_stats = _compute_norm_stats(dataset)
    data_config = _make_data_config(model_cfg, norm_stats)
    rng = np.random.default_rng(args.seed)

    def _sample_indices(n: int) -> list[int]:
        return [int(rng.integers(0, len(dataset))) for _ in range(n)]

    print(f"lance_dataset: {args.lance_dataset}")
    print(f"samples(frame windows): {len(dataset)}  action_horizon={args.action_horizon}")
    print(f"base_model: {args.base_model}  lora_rank: {args.lora_rank}")

    model_id, create_result = _create_model_with_lora(
        args.base_url,
        headers,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        train_attn=args.lora_train_attn,
        train_mlp=args.lora_train_mlp,
        train_unembed=args.lora_train_unembed,
    )
    print(f"model created: model_id={model_id}  result={create_result}")

    steps_log: list[dict[str, Any]] = []
    save_result: dict[str, Any] = {}
    action_session_id: str = ""
    action_result: dict[str, Any] = {}
    mse_evaluation: dict[str, Any] = {}
    infer_to_lance_result: dict[str, Any] = {}
    try:
        for step in range(1, args.steps + 1):
            batch = _build_batch(
                dataset, data_config, base_model=args.base_model, indices=_sample_indices(args.batch_size)
            )
            train_result = _await_result(
                args.base_url,
                headers,
                _post_json(
                    args.base_url,
                    "/api/v1/mint/vla/train_step",
                    headers,
                    {"model_id": model_id, "loss_fn": "flow_matching", "data": batch},
                ),
            )
            metrics = train_result.get("metrics", {}) if isinstance(train_result, dict) else {}
            loss = metrics.get("loss:mean")
            steps_log.append({"step": step, "loss": loss, "metrics": metrics})
            print(json.dumps({"step": step, "loss": loss}), flush=True)

        if not args.skip_save:
            checkpoint_name = args.save_checkpoint_name or f"vla_lora_sampler_{uuid.uuid4().hex[:8]}"
            save_result = _await_result(
                args.base_url,
                headers,
                _post_json(
                    args.base_url,
                    "/api/v1/save_weights_for_sampler",
                    headers,
                    {"model_id": model_id, "path": checkpoint_name},
                ),
            )
            model_path = save_result.get("path")
            if not isinstance(model_path, str) or not model_path:
                raise RuntimeError(f"save_weights_for_sampler missing path: {save_result!r}")
            print(f"save_weights_for_sampler: {save_result}")

            if not args.skip_inference_check:
                action_created = _post_json(
                    args.base_url,
                    "/api/v1/mint/action_sessions",
                    headers,
                    {
                        "session_id": f"vla-lora-infer-{uuid.uuid4().hex[:12]}",
                        "base_model": args.base_model,
                        "model_path": model_path,
                        "owner_id": save_result.get("owner_id"),
                    },
                )
                action_session_id = action_created["action_session_id"]
                obs = _build_batch(dataset, data_config, base_model=args.base_model, indices=[0])[0]["observation"]
                action_result = _await_result(
                    args.base_url,
                    headers,
                    _post_json(
                        args.base_url,
                        f"/api/v1/mint/action_sessions/{action_session_id}/act",
                        headers,
                        {"observation": obs},
                    ),
                )
                print(f"inference check: action_session_id={action_session_id}  result={action_result}")

            if args.eval_mse:
                eval_indices = [int(t) for t in args.eval_mse_indices.split(",") if t.strip()]
                print(f"running MSE evaluation on indices={eval_indices} ...")
                mse_evaluation = run_mse_evaluation(
                    args.base_url,
                    headers,
                    base_model=args.base_model,
                    model_path=model_path,
                    owner_id=save_result.get("owner_id"),
                    dataset=dataset,
                    data_config=data_config,
                    indices=eval_indices,
                )
                print(f"MSE evaluation aggregate: {mse_evaluation['aggregate']}")

            if args.infer_to_lance:
                print(f"running full-dataset inference and writing merged Lance to {args.infer_to_lance_output} ...")
                infer_to_lance_result = run_inference_and_merge_to_lance(
                    args.base_url,
                    headers,
                    base_model=args.base_model,
                    model_path=model_path,
                    owner_id=save_result.get("owner_id"),
                    dataset=dataset,
                    data_config=data_config,
                    action_horizon=args.action_horizon,
                    lance_dataset_path=args.lance_dataset,
                    output_lance_path=args.infer_to_lance_output,
                )
                print(f"infer_to_lance: {infer_to_lance_result}")
    finally:
        # This only deletes server-side session/training state for model_id.
        # It does NOT delete the checkpoint files save_weights_for_sampler already
        # wrote to disk -- those are persisted independently of this cleanup.
        _delete_model(args.base_url, headers, model_id)

    summary = {
        "base_model": args.base_model,
        "model_id": model_id,
        "lance_dataset": args.lance_dataset,
        "create_result": create_result,
        "steps": steps_log,
        "save_result": save_result,
        "action_session_id": action_session_id,
        "action_result": action_result,
        "mse_evaluation": mse_evaluation,
        "infer_to_lance_result": infer_to_lance_result,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"wrote summary to {args.output_json}")

    losses = [s["loss"] for s in steps_log if s.get("loss") is not None]
    if losses:
        print(f"final loss: {losses[-1]:.4f}  (first: {losses[0]:.4f})")
    print(f"done: model_id={model_id}  checkpoint_saved={not args.skip_save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
