from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import time
from queue import Empty
from types import SimpleNamespace

from tinker_server import ray_utils


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


def test_init_ray_resolves_auto_address(monkeypatch):
    calls: list[dict[str, object]] = []

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(dict(kwargs)) or {"ok": True},
    )

    @contextlib.contextmanager
    def _no_lock():
        yield 0.0

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.setattr(ray_utils, "_ray_init_interprocess_lock", _no_lock)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(address="auto", namespace="tinker", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["address"] == "192.168.38.184:6379"
    assert calls[0]["namespace"] == "tinker"
    assert calls[0]["ignore_reinit_error"] is True
    assert calls[0]["log_to_driver"] is False


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

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setattr(ray_utils, "_ray_init_interprocess_lock", _unexpected_lock)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="tinker", ignore_reinit_error=True)

    assert out == {"ok": True}
    assert calls[0]["address"] == "192.168.38.184:6379"


def test_init_ray_skips_when_already_initialized(monkeypatch):
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init should not run")),
    )

    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="tinker", ignore_reinit_error=True)
    assert out is None


def test_init_ray_reuses_existing_worker_context(monkeypatch):
    fake_worker = SimpleNamespace(mode=1)
    fake_ray = SimpleNamespace(
        WORKER_MODE=1,
        is_initialized=lambda: True,
        get_runtime_context=lambda: SimpleNamespace(worker=fake_worker, namespace="tinker"),
        init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init should not run inside worker context")),
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("shutdown should not run inside worker context")),
    )

    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setattr(ray_utils, "_RAY_LAST_INIT_ADDRESS", None)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    out = ray_utils.init_ray(namespace="tinker", ignore_reinit_error=True)

    assert out is None
    assert ray_utils._RAY_LAST_INIT_ADDRESS == "192.168.38.184:6379"
