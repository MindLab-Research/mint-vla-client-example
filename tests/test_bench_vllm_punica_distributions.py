from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tools"
        / "bench_vllm_punica_distributions.py"
    )
    spec = importlib.util.spec_from_file_location("bench_vllm_punica_distributions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_percentile_stays_in_sample_range_for_two_points():
    mod = _load_module()
    out = mod._percentile([1.0, 2.0], 0.95)
    assert out is not None
    assert 1.0 <= out <= 2.0
    assert out == 1.95


def test_percentile_handles_small_samples_without_extrapolation():
    mod = _load_module()
    vals = [1.0, 2.0, 4.0, 8.0]
    out = mod._percentile(vals, 0.95)
    assert out is not None
    assert min(vals) <= out <= max(vals)
