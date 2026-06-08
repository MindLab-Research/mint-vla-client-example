from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts" / "vllm_worker_python.py"
    spec = importlib.util.spec_from_file_location("mint_vllm_worker_python", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_worker_bootstrap_disables_ray_client_mode(monkeypatch, tmp_path):
    wrapper = _load_wrapper()

    calls: list[tuple[str, object]] = []
    hook = types.ModuleType("ray._private.client_mode_hook")

    def _explicitly_disable_client_mode():
        calls.append(("disable", None))

    def _set_client_hook_status(value):
        calls.append(("status", value))

    hook._explicitly_disable_client_mode = _explicitly_disable_client_mode
    hook._set_client_hook_status = _set_client_hook_status

    ray_mod = types.ModuleType("ray")
    private_mod = types.ModuleType("ray._private")
    ray_mod._private = private_mod
    private_mod.client_mode_hook = hook

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray._private", private_mod)
    monkeypatch.setitem(sys.modules, "ray._private.client_mode_hook", hook)
    monkeypatch.setenv("RAY_CLIENT_MODE", "1")

    script = tmp_path / "ray" / "_private" / "workers" / "default_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    monkeypatch.setattr(wrapper.runpy, "run_path", lambda *_args, **_kwargs: None)

    wrapper._run_as_python([str(script)])

    assert os.environ["RAY_CLIENT_MODE"] == "0"
    assert calls == [("disable", None), ("status", False)]
