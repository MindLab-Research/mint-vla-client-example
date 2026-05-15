from __future__ import annotations

from tinker_server import config as cfg


def test_worker_visible_py_executable_keeps_absolute_without_job_working_dir(monkeypatch):
    monkeypatch.delenv("MINT_RAY_JOB_WORKING_DIR", raising=False)
    path = "/tmp/repo/scripts/vllm_worker_python.py"
    assert cfg._worker_visible_py_executable(path) == path


def test_worker_visible_py_executable_relativizes_into_uploaded_working_dir(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "scripts" / "vllm_worker_python.py"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(repo))
    assert cfg._worker_visible_py_executable(str(script)) == "python ./scripts/vllm_worker_python.py"


def test_preferred_vllm_python_executable_relativizes_explicit_env(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "scripts" / "vllm_worker_python.py"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(repo))
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(script))

    assert cfg.preferred_vllm_python_executable() == "python ./scripts/vllm_worker_python.py"


def test_preferred_vllm_python_executable_relativizes_before_existence_check(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "scripts" / "vllm_worker_python.py"

    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(repo))
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(script))

    assert cfg.preferred_vllm_python_executable() == "python ./scripts/vllm_worker_python.py"


def test_preferred_vllm_python_executable_uses_relative_wrapper_in_ray_client_mode(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    script = repo / "scripts" / "vllm_worker_python.py"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setattr(cfg, "PFS_TINKER_PATH", str(repo))
    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(repo))
    monkeypatch.delenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", raising=False)

    assert cfg.preferred_vllm_python_executable() == "python ./scripts/vllm_worker_python.py"
