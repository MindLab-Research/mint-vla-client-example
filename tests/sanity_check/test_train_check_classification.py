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


def test_tinker_uri_owner_sampling_error_is_client_workflow():
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
