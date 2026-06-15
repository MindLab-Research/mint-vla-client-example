from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import time
from queue import Empty
from types import SimpleNamespace

from mint_server import ray_utils


def _holder_proc(lock_path: str, acquired_evt: mp.synchronize.Event, release_evt: mp.synchronize.Event, out_q: mp.Queue) -> None:
    os.environ["MINT_RAY_INIT_LOCK_PATH"] = lock_path
    os.environ["MINT_RAY_INIT_LOCK_TIMEOUT_S"] = "5"
    with ray_utils._ray_init_interprocess_lock() as wait_s:
        out_q.put(("holder", float(wait_s)))
        acquired_evt.set()
        release_evt.wait(timeout=5)


def _contender_proc(lock_path: str, acquired_evt: mp.synchronize.Event, out_q: mp.Queue) -> None:
    os.environ["MINT_RAY_INIT_LOCK_PATH"] = lock_path
    os.environ["MINT_RAY_INIT_LOCK_TIMEOUT_S"] = "5"
    if not acquired_evt.wait(timeout=5):
        out_q.put(("contender", "failed_waiting_for_holder"))
        return
    with ray_utils._ray_init_interprocess_lock() as wait_s:
        out_q.put(("contender", float(wait_s)))


def test_ray_init_interprocess_lock_serializes_processes(tmp_path):
    lock_path = str(tmp_path / "ray-init.lock")

    ctx = mp.get_context("fork")
    acquired_evt = ctx.Event()
    release_evt = ctx.Event()
    out_q = ctx.Queue()

    holder = ctx.Process(target=_holder_proc, args=(lock_path, acquired_evt, release_evt, out_q))
    contender = ctx.Process(target=_contender_proc, args=(lock_path, acquired_evt, out_q))

    holder.start()
    contender.start()

    assert acquired_evt.wait(timeout=5), "holder did not acquire lock"
    time.sleep(0.3)
    release_evt.set()

    holder.join(timeout=10)
    contender.join(timeout=10)

    assert holder.exitcode == 0
    assert contender.exitcode == 0

    results = {}
    try:
        while True:
            role, value = out_q.get_nowait()
            results[role] = value
    except Empty:
        pass

    assert isinstance(results.get("holder"), float), results
    assert isinstance(results.get("contender"), float), results
    assert results["holder"] < 0.2
    assert results["contender"] >= 0.25


def test_init_ray_resolves_auto_address_from_explicit_mint_gcs(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    @contextlib.contextmanager
    def _no_lock():
        yield 0.0

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.setattr(ray_utils, "_ray_init_interprocess_lock", _no_lock)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(address="auto", namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["address"] == "192.168.38.184:6379"
    assert calls[0]["namespace"] == "mint"
    assert calls[0]["ignore_reinit_error"] is True
    assert calls[0]["log_to_driver"] is False


def test_init_ray_prefers_ray_client_over_gcs_address(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "ray://192.168.38.184:10001"
    assert ray_utils.preferred_ray_gcs_address() == "192.168.38.184:6379"


def test_gcs_address_helpers_ignore_legacy_ray_address(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_HEAD_ADDRESS_PATH", raising=False)

    assert ray_utils.preferred_ray_gcs_address() is None
    assert ray_utils.strict_ray_gcs_address() is None


def test_init_ray_ignores_legacy_ray_address_without_explicit_address(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_HEAD_ADDRESS_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    try:
        ray_utils.init_ray(address="auto", namespace="mint", ignore_reinit_error=True)
    except ray_utils.MissingRayAddressError as exc:
        assert "explicit Ray address is required" in str(exc)
    else:
        raise AssertionError("legacy RAY_ADDRESS must not satisfy Ray init address contract")
    assert calls == []


def test_init_ray_blanks_attach_hints_in_ray_client_job_runtime_env(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "ray://192.168.38.184:10001"
    env_vars = calls[0]["runtime_env"]["env_vars"]
    assert "RAY_ADDRESS" not in env_vars
    assert env_vars["RAY_CLIENT_ADDRESS"] == ""
    assert env_vars["MINT_RAY_CLIENT_ADDRESS"] == ""
    assert env_vars["MINT_RAY_GCS_ADDRESS"] == "192.168.38.184:6379"


def test_init_ray_does_not_promote_legacy_ray_address_to_job_gcs_hint(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_RAY_GCS_ADDRESS", raising=False)
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "ray://192.168.38.184:10001"
    env_vars = calls[0]["runtime_env"]["env_vars"]
    assert "RAY_ADDRESS" not in env_vars
    assert env_vars["MINT_RAY_CLIENT_ADDRESS"] == ""
    assert "MINT_RAY_GCS_ADDRESS" not in env_vars


def test_init_ray_sets_job_py_executable_for_ray_client_workers(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    wrapper = tmp_path / "repo" / "scripts" / "vllm_worker_python.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setenv("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", str(wrapper))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "ray://192.168.38.184:10001"
    assert calls[0]["runtime_env"]["py_executable"] == str(wrapper)


def test_init_ray_blanks_attach_hints_without_dropping_existing_ray_client_runtime_env(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(
        namespace="mint",
        runtime_env={"env_vars": {"PYTHONPATH": "/shared/code"}, "py_modules": ["s3://bucket/mint.zip"]},
    )

    assert out == {"ok": True}
    runtime_env = calls[0]["runtime_env"]
    assert runtime_env["py_modules"] == ["s3://bucket/mint.zip"]
    env_vars = runtime_env["env_vars"]
    assert env_vars["PYTHONPATH"] == "/shared/code"
    assert "RAY_ADDRESS" not in env_vars
    assert env_vars["MINT_RAY_CLIENT_ADDRESS"] == ""
    assert env_vars["MINT_RAY_GCS_ADDRESS"] == "192.168.38.184:6379"


def test_init_ray_skips_lock_when_attaching_to_existing_cluster(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    @contextlib.contextmanager
    def _unexpected_lock():
        raise AssertionError("interprocess lock should not run for remote ray attach")
        yield 0.0

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setattr(ray_utils, "_ray_init_interprocess_lock", _unexpected_lock)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "192.168.38.184:6379"


def test_init_ray_skips_when_already_initialized(monkeypatch):
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init should not run")),
    )

    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)
    assert out is None


def test_init_ray_reuses_existing_worker_context(monkeypatch):
    fake_worker = SimpleNamespace(mode=1)
    fake_ray = SimpleNamespace(
        WORKER_MODE=1,
        is_initialized=lambda: True,
        get_runtime_context=lambda: SimpleNamespace(worker=fake_worker, namespace="mint"),
        init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init should not run inside worker context")),
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("shutdown should not run inside worker context")),
    )

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setattr(ray_utils, "_RAY_LAST_INIT_ADDRESS", None)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="mint", ignore_reinit_error=True)

    assert out is None
    assert ray_utils._RAY_LAST_INIT_ADDRESS == "192.168.38.184:6379"
