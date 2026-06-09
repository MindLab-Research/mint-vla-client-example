from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_train_check():
    path = Path(__file__).resolve().parents[2] / "scripts/wip/train_check.py"
    spec = importlib.util.spec_from_file_location("mint_train_check_for_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mint_uri_owner_sampling_error_is_client_workflow():
    train_check = _load_train_check()
    text = """
    Traceback (most recent call last):
      File ".claude/skills/sanity-check/mint_rl_test_long.py", line 1432, in <lambda>
        lambda: _create_sampling_client_for_checkpoint(...)
      File ".claude/skills/sanity-check/mint_rl_test_long.py", line 598, in _create_sampling_client_for_checkpoint
        raise ValueError("model_path must start with 'mint://'")
    ValueError: model_path must start with 'mint://'
    """

    assert train_check.classify_failure(text, exit_code=1) == "client workflow"
    assert train_check.failure_surface_from_logs(text) == "create_sampling_client"


def test_incomplete_rl_loop_surfaces_as_step_not_completed():
    train_check = _load_train_check()
    text = """
    [step 1] Failed to create sampling client: RequestFailedError: checkpoint already uploading
    FAIL in rl_step_not_completed: RL sanity did not complete requested steps: completed=0/1;
    last_failure=[step 1] Failed to create sampling client: RequestFailedError: checkpoint already uploading
    RuntimeError: RL sanity did not complete requested steps: completed=0/1
    """

    assert train_check.classify_failure(text, exit_code=1) == "server exception"
    assert train_check.failure_surface_from_logs(text) == "rl_step_not_completed"


def test_sampling_inactivity_is_capacity_scheduling_with_detail():
    train_check = _load_train_check()
    text = """
    Request failed with non-retryable error: RequestFailedError: Request failed:
    Sampling session terminated due to sampling inactivity (> 1800.0s)
    for self.request_id='sample_fe75a3e6a65c15daf4642461ddf12776'
    """

    classification = train_check.classify_failure_detail(text, exit_code=1)

    assert classification.failure_class == "capacity/scheduling"
    assert "inactivity TTL" in classification.detail


def test_preflight_http_503_is_control_plane_health():
    train_check = _load_train_check()

    classification = train_check.classify_preflight_failure(
        "HTTP preflight failed: HTTP Error 503: Service Unavailable"
    )
    results = train_check.preflight_failure_results(["Qwen/Test"], classification.detail or "")

    assert classification.failure_class == "server health/control-plane"
    assert results[0]["failure_class"] == "server health/control-plane"
    assert results[0]["failure_surface"] == "preflight"


def test_preflight_auth_remains_client_env_auth():
    train_check = _load_train_check()

    classification = train_check.classify_preflight_failure(
        "MINT_API_KEY is required for production sanity checks"
    )

    assert classification.failure_class == "client env/auth"


def test_unknown_failure_includes_compact_error_detail():
    train_check = _load_train_check()
    text = """
    all previous stages looked normal
    FatalWidgetMelt: allocator returned nonsense
    """

    classification = train_check.classify_failure_detail(text, exit_code=1)

    assert classification.failure_class == "unknown"
    assert "FatalWidgetMelt" in classification.detail


def test_generation_summary_aggregates_sample_tokens():
    train_check = _load_train_check()
    timing = {
        "stages": [
            {
                "stage": "sample",
                "total_s": 10.0,
                "output_tokens": 80,
                "hit_max_count": 4,
            },
            {
                "stage": "eval_sample",
                "total_s": 5.0,
                "output_tokens": 10,
                "hit_max_count": 0,
            },
        ]
    }

    summary = train_check.extract_generation_summary(timing)

    assert summary["output_tokens"] == 90
    assert summary["tokens_per_s"] == 6.0
    assert summary["hit_max_count"] == 4
    assert summary["by_stage"]["sample"]["tokens_per_s"] == 8.0
    assert summary["by_stage"]["eval_sample"]["tokens_per_s"] == 2.0


def test_feishu_report_splits_sample_and_eval_throughput():
    train_check = _load_train_check()
    report = train_check.build_feishu_report(
        [
            {
                "model": "Qwen/Test",
                "status": "ok",
                "slowest_stage": "rl_step_total",
                "slowest_max_s": 10.0,
                "wall_clock_s": 20.0,
                "timing_degraded": False,
                "generation_summary": {
                    "output_tokens": 90,
                    "tokens_per_s": 6.0,
                    "hit_max_count": 4,
                    "by_stage": {
                        "sample": {
                            "output_tokens": 80,
                            "elapsed_s": 10.0,
                            "tokens_per_s": 8.0,
                            "hit_max_count": 4,
                        },
                        "eval_sample": {
                            "output_tokens": 10,
                            "elapsed_s": 5.0,
                            "tokens_per_s": 2.0,
                            "hit_max_count": 0,
                        },
                    },
                },
                "degradation_reason": None,
            }
        ]
    )

    assert "sample_e2e_tok_s=`8.00`" in report
    assert "eval_e2e_tok_s=`2.00`" in report
    assert "total_e2e_tok_s=`6.00`" in report
    assert "gen_tok_s" not in report


def test_feishu_report_includes_checkpoint_cleanup_counts():
    train_check = _load_train_check()
    report = train_check.build_feishu_report(
        [
            {
                "model": "Qwen/Test",
                "status": "ok",
                "slowest_stage": "rl_step_total",
                "slowest_max_s": 10.0,
                "wall_clock_s": 20.0,
                "timing_degraded": False,
                "generation_summary": None,
                "queue_stage_attribution": None,
                "degradation_reason": None,
            }
        ],
        train_check.CheckpointCleanupResult(
            attempted=True,
            listed=7,
            selected=3,
            deleted=3,
            failed=0,
            errors=[],
        ),
    )

    assert "Checkpoint cleanup" in report
    assert "selected=3, deleted=3, failed=0" in report


def test_queue_stage_attribution_aggregates_stable_buckets():
    train_check = _load_train_check()
    timing = {
        "stages": [
            {
                "stage": "sample",
                "queue_stage_timing_s": {
                    "scheduler_wait_s": 3.0,
                    "executor_wait_s": 1.0,
                    "lora_s": 4.0,
                    "vllm_generate_s": 9.0,
                    "finalization_s": 0.5,
                },
            }
        ],
    }

    attribution = train_check.extract_queue_stage_attribution(timing)

    assert attribution == {
        "scheduler_wait_s": 3.0,
        "executor_wait_s": 1.0,
        "lora_s": 4.0,
        "vllm_generate_s": 9.0,
        "finalization_s": 0.5,
    }


def test_queue_stage_attribution_prefers_summary_total_over_stage_breakdown():
    train_check = _load_train_check()
    timing = {
        "queue_stage_attribution_s": {"scheduler_wait_s": 2.0},
        "stages": [{"stage": "sample", "queue_stage_timing_s": {"scheduler_wait_s": 3.0}}],
    }

    assert train_check.extract_queue_stage_attribution(timing) == {"scheduler_wait_s": 2.0}


def test_feishu_report_includes_queue_stage_attribution():
    train_check = _load_train_check()
    report = train_check.build_feishu_report(
        [
            {
                "model": "Qwen/Test",
                "status": "ok",
                "slowest_stage": "sample",
                "slowest_max_s": 10.0,
                "wall_clock_s": 20.0,
                "timing_degraded": True,
                "generation_summary": None,
                "queue_stage_attribution": {
                    "scheduler_wait_s": 5.0,
                    "executor_wait_s": 1.0,
                    "lora_s": 4.0,
                    "vllm_generate_s": 9.0,
                    "finalization_s": 0.5,
                },
                "degradation_reason": "slowest max 10.0s > 5.0s",
            }
        ]
    )

    assert "scheduler_wait=`5.0s`" in report
    assert "executor_wait=`1.0s`" in report
    assert "LoRA=`4.0s`" in report
    assert "vLLM_generate=`9.0s`" in report
    assert "finalization=`0.5s`" in report


def test_timing_degradation_is_thresholded_for_success_only():
    train_check = _load_train_check()

    reason = train_check.classify_timing_degradation(
        model="Qwen/Qwen3-4B-Thinking-2507",
        status="ok",
        wall_clock_s=258.8,
        slowest_max_s=97.1,
    )

    assert "wall" in reason
    assert train_check.classify_timing_degradation(
        model="Qwen/Qwen3-4B-Thinking-2507",
        status="fail",
        wall_clock_s=258.8,
        slowest_max_s=97.1,
    ) is None


def test_main_cleans_up_after_full_matrix_pass(monkeypatch, tmp_path):
    train_check = _load_train_check()
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("MINT_API_KEY", "redacted")
    monkeypatch.setenv("MINT_TEST_CHECKPOINT_OWNER_ID", "0123456789abcdef01234567")
    monkeypatch.setattr(
        train_check,
        "parse_args",
        lambda: SimpleNamespace(
            models=[],
            models_flag=[],
            all_models=True,
            num_rl_steps=1,
            batch_size=2,
            group_size=4,
            max_tokens=128,
            timeout_s=7200.0,
            base_url=train_check.DEFAULT_BASE_URL,
            results_root=str(tmp_path),
            run_name="run",
            sequential=False,
            parallel=False,
            skip_preflight=True,
            dry_run=False,
            summary_json=None,
            summary_md=None,
            feishu=False,
            cleanup_pass_checkpoints=True,
        ),
    )
    monkeypatch.setattr(train_check, "ensure_runner_exists", lambda: None)
    monkeypatch.setattr(train_check, "load_env_file", lambda path: None)
    monkeypatch.setattr(train_check, "build_runs", lambda args, run_root, create_dirs=True: [])

    def ok_result(model):
        return {
            "model": model,
            "status": "ok",
            "exit_code": 0,
            "stdout_log": None,
            "stderr_log": None,
            "request_ids": {},
            "session_ids": {},
            "timing_degraded": False,
            "generation_summary": None,
            "queue_stage_attribution": None,
        }

    monkeypatch.setattr(
        train_check,
        "run_parallel",
        lambda runs, sequential: [ok_result(model) for model in train_check.ALL_MODELS],
    )
    calls = []

    def fake_cleanup(**kwargs):
        calls.append(kwargs)
        return train_check.CheckpointCleanupResult(
            attempted=True,
            listed=5,
            selected=5,
            deleted=5,
            failed=0,
            errors=[],
        )

    monkeypatch.setattr(train_check, "cleanup_pass_checkpoints", fake_cleanup)

    assert train_check.main() == 0
    assert len(calls) == 1
    assert calls[0]["models"] == train_check.ALL_MODELS


def test_main_returns_failure_when_checkpoint_cleanup_fails(monkeypatch, tmp_path):
    train_check = _load_train_check()
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("MINT_API_KEY", "redacted")
    monkeypatch.setenv("MINT_TEST_CHECKPOINT_OWNER_ID", "0123456789abcdef01234567")
    monkeypatch.setattr(
        train_check,
        "parse_args",
        lambda: SimpleNamespace(
            models=[],
            models_flag=[],
            all_models=True,
            num_rl_steps=1,
            batch_size=2,
            group_size=4,
            max_tokens=128,
            timeout_s=7200.0,
            base_url=train_check.DEFAULT_BASE_URL,
            results_root=str(tmp_path),
            run_name="run",
            sequential=False,
            parallel=False,
            skip_preflight=True,
            dry_run=False,
            summary_json=None,
            summary_md=None,
            feishu=False,
            cleanup_pass_checkpoints=True,
        ),
    )
    monkeypatch.setattr(train_check, "ensure_runner_exists", lambda: None)
    monkeypatch.setattr(train_check, "load_env_file", lambda path: None)
    monkeypatch.setattr(train_check, "build_runs", lambda args, run_root, create_dirs=True: [])

    def ok_result(model):
        return {
            "model": model,
            "status": "ok",
            "exit_code": 0,
            "stdout_log": None,
            "stderr_log": None,
            "request_ids": {},
            "session_ids": {},
            "timing_degraded": False,
            "generation_summary": None,
            "queue_stage_attribution": None,
        }

    monkeypatch.setattr(
        train_check,
        "run_parallel",
        lambda runs, sequential: [ok_result(model) for model in train_check.ALL_MODELS],
    )
    monkeypatch.setattr(
        train_check,
        "cleanup_pass_checkpoints",
        lambda **kwargs: train_check.CheckpointCleanupResult(
            attempted=True,
            listed=5,
            selected=5,
            deleted=4,
            failed=1,
            errors=["delete failed"],
        ),
    )

    assert train_check.main() == 4
