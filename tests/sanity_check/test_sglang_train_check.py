from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_sglang_train_check():
    path = Path(__file__).resolve().parents[2] / "scripts/tools/sglang_train_check.py"
    spec = importlib.util.spec_from_file_location("mint_sglang_train_check_for_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sglang_train_check_refuses_production_url_by_default() -> None:
    mod = _load_sglang_train_check()

    with pytest.raises(SystemExit, match="refuses production URL"):
        mod.validate_sglang_target(
            SimpleNamespace(base_url="https://mint.macaron.xin", allow_production_url=False)
        )


def test_sglang_train_check_builds_standard_matrix_with_backend_marker(tmp_path) -> None:
    mod = _load_sglang_train_check()
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18084",
        api_key="dummy",
        checkpoint_owner_id=None,
        all_models=True,
        models=[],
        models_flag=[],
        num_rl_steps=1,
        batch_size=2,
        group_size=4,
        max_tokens=128,
        timeout_s=7200.0,
        allow_production_url=False,
        _train_check=mod._load_train_check(),
    )

    runs = mod.build_sglang_runs(args, tmp_path, create_dirs=False)

    assert [run.model for run in runs] == args._train_check.ALL_MODELS
    assert all(run.env["MINT_BASE_URL"] == "http://127.0.0.1:18084" for run in runs)
    assert all(run.env["TINKER_BASE_URL"] == "http://127.0.0.1:18084" for run in runs)
    assert all(run.env["MINT_API_KEY"] == "tml-dummy" for run in runs)
    assert all(run.env["TINKER_API_KEY"] == "tml-dummy" for run in runs)
    assert all(run.env["MINT_SANITY_TARGET_BACKEND"] == "sglang" for run in runs)
    expected_shim = str(mod.SDK_SHIM_PATH)
    assert all(run.env["PYTHONPATH"].split(os.pathsep)[0] == expected_shim for run in runs)
    assert all("MINT_TEST_CHECKPOINT_OWNER_ID" not in run.env for run in runs)
    assert runs[0].command[:2] == [
        sys.executable,
        str(args._train_check.RUNNER),
    ]
    commands_by_model = {run.model: run.command for run in runs}
    for model in (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ):
        assert "--no-train-mlp" not in commands_by_model[model]
        assert "--no-train-unembed" in commands_by_model[model]
    assert "--no-train-mlp" not in commands_by_model["Qwen/Qwen3-0.6B"]
    assert "--no-train-unembed" not in commands_by_model["Qwen/Qwen3-0.6B"]


def test_sglang_train_check_forwards_explicit_lora_target_switches(tmp_path) -> None:
    mod = _load_sglang_train_check()
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18084",
        api_key="dummy",
        checkpoint_owner_id=None,
        all_models=False,
        models=["30b"],
        models_flag=[],
        num_rl_steps=1,
        batch_size=2,
        group_size=4,
        max_tokens=128,
        lora_rank=32,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
        timeout_s=7200.0,
        allow_production_url=False,
        _train_check=mod._load_train_check(),
    )

    runs = mod.build_sglang_runs(args, tmp_path, create_dirs=False)

    assert len(runs) == 1
    assert runs[0].model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    command = runs[0].command
    assert "--lora-rank=32" in command
    assert "--no-train-mlp" in command
    assert "--train-attn" in command
    assert "--no-train-unembed" in command


def test_sglang_train_check_explicit_moe_lora_switches_override_defaults(tmp_path) -> None:
    mod = _load_sglang_train_check()
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18084",
        api_key="dummy",
        checkpoint_owner_id=None,
        all_models=False,
        models=["30b"],
        models_flag=[],
        num_rl_steps=1,
        batch_size=2,
        group_size=4,
        max_tokens=128,
        lora_rank=None,
        train_mlp=True,
        train_attn=None,
        train_unembed=True,
        timeout_s=7200.0,
        allow_production_url=False,
        _train_check=mod._load_train_check(),
    )

    runs = mod.build_sglang_runs(args, tmp_path, create_dirs=False)

    assert len(runs) == 1
    command = runs[0].command
    assert "--train-mlp" in command
    assert "--train-unembed" in command
    assert "--no-train-mlp" not in command
    assert "--no-train-unembed" not in command


def test_sglang_train_check_writes_backend_marked_outputs(tmp_path) -> None:
    mod = _load_sglang_train_check()
    train_check = mod._load_train_check()
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18084",
        summary_json=None,
        summary_md=None,
        allow_production_url=False,
        _train_check=train_check,
    )
    results = [
        {
            "model": "Qwen/Qwen3-0.6B",
            "slug": "qwen__qwen3-0_6b",
            "exit_code": 0,
            "status": "ok",
            "run_dir": str(tmp_path / "qwen__qwen3-0_6b"),
            "stdout_log": str(tmp_path / "stdout.log"),
            "stderr_log": str(tmp_path / "stderr.log"),
            "experiment_dir": None,
            "request_ids": {},
            "session_ids": {},
            "failure_class": None,
            "failure_detail": None,
            "failure_surface": "Sample",
            "timing_summary_json": None,
            "timing_summary_md": None,
            "timing_events_jsonl": None,
            "wall_clock_s": 42.7,
            "slowest_stage": "rl_step_total",
            "slowest_max_s": 33.9,
            "generation_summary": None,
            "queue_stage_attribution": None,
            "timing_degraded": False,
            "degradation_reason": None,
            "started_at_epoch_s": 0.0,
            "finished_at_epoch_s": 1.0,
        }
    ]

    json_path, md_path, report_path = mod._write_outputs(results, tmp_path, args)

    payload = json.loads(json_path.read_text())
    assert payload["target_backend"] == "sglang"
    assert payload["base_url"] == "http://127.0.0.1:18084"
    assert md_path.name == "summary.md"
    assert "- target_backend: `sglang`" in md_path.read_text()
    assert report_path.name == "final_sglang_report.md"
    report = report_path.read_text()
    assert report.startswith("**Target backend:** sglang")
    assert "**Result:** PASS (1/1 models passed)" in report
