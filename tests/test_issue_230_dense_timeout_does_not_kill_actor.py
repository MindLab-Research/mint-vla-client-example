import asyncio
import uuid

import pytest

pytest.importorskip("ray")

import ray

from tinker_server.backend.resource_pool import ActorType, get_resource_pool
from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import VerlTrainingEngine


def test_issue_230_timeout_does_not_kill_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
    model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()
    engine._resource_pool_actor_names[model_id] = actor_name

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    killed: list[dict] = []

    import tinker_server.backend.verl_training as verl_training

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    def _always_timeout(*args, **kwargs):
        raise ray.exceptions.GetTimeoutError("timeout")

    monkeypatch.setattr(ray, "get", _always_timeout)

    async def _run() -> None:
        await engine._await_with_keepalive(
            awaitable=object(),
            session=session,
            interval_s=0.01,
            timeout_s=0.05,
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())

    assert killed == []

    pool.unregister(actor_name)


def test_issue_230_keepalive_marks_dense_actor_inflight(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
    model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    entry = pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()
    engine._resource_pool_actor_names[model_id] = actor_name

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    observed_inflight: list[int] = []

    def _timeout_once_then_return(*args, **kwargs):
        observed_inflight.append(pool.get(actor_name).inflight_count)
        if len(observed_inflight) == 1:
            raise ray.exceptions.GetTimeoutError("timeout")
        return {"ok": True}

    monkeypatch.setattr(ray, "get", _timeout_once_then_return)

    async def _run() -> None:
        result = await engine._await_with_keepalive(
            awaitable=object(),
            session=session,
            interval_s=0.01,
            timeout_s=0.05,
        )
        assert result == {"ok": True}

    asyncio.run(_run())

    assert observed_inflight[0] == 1
    assert pool.get(actor_name).inflight_count == 0
    assert entry.current_session == model_id

    pool.unregister(actor_name)


def test_issue_230_unbind_session_keeps_shared_dense_actor_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
    model_id = f"model_{uuid.uuid4().hex}"
    other_model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    entry = pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()
    shared_worker = object()
    engine._resource_pool_actor_names[model_id] = actor_name
    engine._resource_pool_actor_names[other_model_id] = actor_name
    engine._workers[model_id] = shared_worker
    engine._workers[other_model_id] = shared_worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )

    import tinker_server.backend.verl_training as verl_training

    killed: list[dict] = []

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    asyncio.run(engine.unbind_session(session))

    assert killed == []
    assert model_id not in engine._resource_pool_actor_names
    assert other_model_id in engine._resource_pool_actor_names
    assert model_id not in engine._workers
    assert entry.current_session == other_model_id
    assert pool.get(actor_name).current_session == other_model_id

    pool.unregister(actor_name)


def test_issue_230_shutdown_session_keeps_protected_dense_actor_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
    model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    entry = pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
        protected=True,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()

    class _ShutdownRecorder:
        def __init__(self) -> None:
            self.calls = 0
            self.shutdown = self

        def remote(self):
            self.calls += 1
            return True

    worker = _ShutdownRecorder()
    engine._resource_pool_actor_names[model_id] = actor_name
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_protected",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )

    import tinker_server.backend.verl_training as verl_training

    killed: list[dict] = []

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    asyncio.run(engine.shutdown_session(session))

    assert worker.calls == 0
    assert killed == []
    assert model_id not in engine._resource_pool_actor_names
    assert model_id not in engine._workers
    assert entry.current_session is None
    assert pool.is_protected(actor_name) is True

    pool.unregister(actor_name)
