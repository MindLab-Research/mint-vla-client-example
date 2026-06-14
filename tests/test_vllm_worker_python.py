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
    attach_keys = (
        "RAY_ADDRESS",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_CLIENT_ADDRESS",
    )
    mint_gcs_key = "MINT_RAY_GCS_ADDRESS"
    for key in (*env_keys, *attach_keys):
        monkeypatch.setenv(key, "/tmp/mph/t")
    monkeypatch.setenv(mint_gcs_key, "192.168.40.99:6379")
    monkeypatch.setattr(wrapper.runpy, "run_path", lambda *_args, **_kwargs: None)

    script = tmp_path / "ray" / "_private" / "workers" / "default_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    wrapper._run_as_python([str(script)])

    for key in env_keys:
        assert key not in os.environ
    for key in attach_keys:
        assert key not in os.environ
    assert os.environ[mint_gcs_key] == "192.168.40.99:6379"


def test_worker_bootstrap_clears_attach_hints_before_module_execution(monkeypatch):
    wrapper = _load_wrapper()
    seen: dict[str, str | None] = {}

    for key in (
        "RAY_ADDRESS",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_CLIENT_ADDRESS",
    ):
        monkeypatch.setenv(key, "192.168.40.99:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.40.99:6379")

    def fake_run_module(*_args, **_kwargs):
        for key in (
            "RAY_ADDRESS",
            "RAY_CLIENT_ADDRESS",
            "MINT_RAY_CLIENT_ADDRESS",
            "MINT_RAY_GCS_ADDRESS",
        ):
            seen[key] = os.environ.get(key)

    monkeypatch.setattr(wrapper.runpy, "run_module", fake_run_module)

    wrapper._run_as_python(["-m", "ray._private.workers.default_worker"])

    assert seen == {
        "RAY_ADDRESS": None,
        "RAY_CLIENT_ADDRESS": None,
        "MINT_RAY_CLIENT_ADDRESS": None,
        "MINT_RAY_GCS_ADDRESS": "192.168.40.99:6379",
    }
