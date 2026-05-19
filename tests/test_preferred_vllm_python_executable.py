from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _import_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MINT_CONFIG_PATH", raising=False)
    sys.modules.pop("mint_server.config", None)
    return importlib.import_module("mint_server.config")


def test_preferred_vllm_python_executable_uses_existing_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = tmp_path / "python"
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(worker))
    monkeypatch.delenv("MINT_CODE_ROOT", raising=False)

    cfg = _import_config(monkeypatch)

    assert cfg.preferred_vllm_python_executable() == str(worker)


def test_preferred_vllm_python_executable_rejects_missing_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    script = repo_root / "scripts" / "vllm_worker_python.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env python3\n")
    script.chmod(0o755)

    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(tmp_path / "missing-python"))
    monkeypatch.setenv("MINT_CODE_ROOT", str(repo_root))

    cfg = _import_config(monkeypatch)

    with pytest.raises(RuntimeError, match="MINT_VLLM_CHILD_PYTHON_EXECUTABLE does not exist"):
        cfg.preferred_vllm_python_executable()
