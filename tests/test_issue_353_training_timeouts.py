from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from mint_server.backend import megatron_distributed as mg
from mint_server.backend.training_session_manager import TrainingSession
from mint_server.backend.verl_training import VerlTrainingEngine


class _FakeRemoteMethod:
    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return object()


class _FakeWorker:
    def __init__(self):
        self.optim_step = _FakeRemoteMethod()


def test_issue_353_engine_optim_step_uses_training_remote_call_timeout(monkeypatch):
    engine = VerlTrainingEngine()
    worker = _FakeWorker()
    session = TrainingSession(
        model_id="m1",
        session_id="s1",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    request = SimpleNamespace(adam_params=SimpleNamespace(learning_rate=1e-4))
    seen: dict[str, object] = {}

    monkeypatch.setattr("mint_server.backend.verl_training.server_config.training_remote_call_timeout_s", 123.0)

    async def fake_get_live_worker(session_arg, op):
        seen["get_live_worker"] = (session_arg.model_id, op)
        return worker

    async def fake_keepalive(awaitable, session_arg, interval_s=30.0, timeout_s=None):
        seen["awaitable"] = awaitable
        seen["session"] = session_arg.model_id
        seen["interval_s"] = interval_s
        seen["timeout_s"] = timeout_s
        return {"metrics": {}}

    def fake_touch_actor(session_arg):
        seen["touches"] = int(seen.get("touches", 0)) + 1

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_touch_actor", fake_touch_actor)

    result = asyncio.run(engine.optim_step(session, request))

    assert seen["get_live_worker"] == ("m1", "optim_step")
    assert seen["session"] == "m1"
    assert seen["interval_s"] == 30.0
    assert seen["timeout_s"] == 123.0
    assert seen["touches"] == 1
    assert worker.optim_step.calls[0][0][1] == "m1"
    assert result["metrics"]["step"] == 1


def test_issue_353_megatron_group_optim_step_times_out(monkeypatch):
    group_cls = mg.MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = [_FakeWorker(), _FakeWorker()]
    group._step_count = 0
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id or "s1"
    group._ensure_session_loaded = lambda *args, **kwargs: None
    group._start_slow_group_watchdog = lambda **kwargs: None
    group._stop_slow_group_watchdog = lambda token: None

    monkeypatch.setattr(mg.server_config, "training_remote_call_timeout_s", 12.0)

    seen: dict[str, object] = {}

    def fake_ray_get(futures, timeout=None):
        seen["timeout"] = timeout
        seen["count"] = len(futures)
        raise mg.ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(mg.ray, "get", fake_ray_get)

    with pytest.raises(RuntimeError, match="optim_step timed out after 12.0s"):
        group.optim_step(learning_rate=1e-4, session_id="s-timeout")

    assert seen == {"timeout": 12.0, "count": 2}
