from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_unknown_failure_includes_compact_error_detail():
    train_check = _load_train_check()
    text = """
    all previous stages looked normal
    FatalWidgetMelt: allocator returned nonsense
    """

    classification = train_check.classify_failure_detail(text, exit_code=1)

    assert classification.failure_class == "unknown"
    assert "FatalWidgetMelt" in classification.detail
