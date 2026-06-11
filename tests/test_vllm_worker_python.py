from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts" / "vllm_worker_python.py"
    spec = importlib.util.spec_from_file_location("mint_vllm_worker_python", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_worker_bootstrap_clears_driver_temp_hints(monkeypatch, tmp_path):
    wrapper = _load_wrapper()

    env_keys = (
        "MINT_RAY_TEMP_DIR",
        "MINT_RAY_NODE_IP_ADDRESS",
        "RAY_TMPDIR",
        "TMPDIR",
        "TMP",
        "TEMP",
    )
    address_keys = (
        "RAY_ADDRESS",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_CLIENT_ADDRESS",
    )
    for key in (*env_keys, *address_keys):
        monkeypatch.setenv(key, "/tmp/mph/t")
    monkeypatch.setattr(wrapper.runpy, "run_path", lambda *_args, **_kwargs: None)

    script = tmp_path / "ray" / "_private" / "workers" / "default_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    wrapper._run_as_python([str(script)])

    for key in env_keys:
        assert key not in os.environ
    for key in address_keys:
        assert os.environ[key] == "/tmp/mph/t"
