import asyncio
import uuid

import pytest

pytest.importorskip("ray")

from mint_server.backend.actors.model_actor_supervisor import ActorType, ModelActorSupervisor
from mint_server.backend.training.training_session_manager import TrainingSession
from mint_server.backend.training.verl.verl_training import VerlTrainingEngine


def _get_local_model_actor_inventory(monkeypatch: pytest.MonkeyPatch):
    pool = ModelActorSupervisor()
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.backend.actors.ray_keepalive as ray_keepalive
    import mint_server.backend.training.verl.verl_training as verl_training

    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", pool)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: pool)
    monkeypatch.setattr(ray_keepalive, "get_model_actor_supervisor", lambda: pool)
    monkeypatch.setattr(verl_training, "get_model_actor_supervisor", lambda: pool, raising=False)
    pool.clear(kill_actors=False)
    return pool


def test_issue_230_timeout_does_not_kill_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _get_local_model_actor_inventory(monkeypatch)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
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

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = actor_name
    session.namespace = "mint"

    killed: list[dict] = []

    import mint_server.backend.training.verl.verl_training as verl_training

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        await engine._await_with_keepalive(
            awaitable=fut,
            session=session,
            interval_s=0.01,
            timeout_s=0.05,
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())

    assert killed == []

    pool.unregister(actor_name)


def test_issue_230_keepalive_touches_dense_actor_without_inflight_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _get_local_model_actor_inventory(monkeypatch)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
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

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = actor_name
    session.namespace = "mint"

    observed_inflight: list[int] = []
    before_last_accessed = pool.get(actor_name).last_accessed

    original_touch_actor = engine._touch_actor

    def _touch_actor(target_session):
        observed_inflight.append(pool.get(actor_name).inflight_count)
        return original_touch_actor(target_session)

    monkeypatch.setattr(engine, "_touch_actor", _touch_actor)

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(0.025, fut.set_result, {"ok": True})
        result = await engine._await_with_keepalive(
            awaitable=fut,
            session=session,
            interval_s=0.01,
            timeout_s=0.05,
        )
        assert result == {"ok": True}

    asyncio.run(_run())

    assert observed_inflight[0] == 0
    assert pool.get(actor_name).inflight_count == 0
    assert pool.get(actor_name).last_accessed >= before_last_accessed
    assert entry.current_session == model_id

    pool.unregister(actor_name)


def test_issue_230_keepalive_cancellation_silences_late_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _get_local_model_actor_inventory(monkeypatch)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
    model_id = f"model_{uuid.uuid4().hex}"

    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = actor_name
    session.namespace = "mint"
    discarded: list[str] = []

    from mint_server.backend.ray_cluster import async_ray_control

    def _record_late_result(fut: asyncio.Future) -> None:
        try:
            fut.result()
        except RuntimeError as exc:
            discarded.append(str(exc))
        except BaseException as exc:
            discarded.append(type(exc).__name__)

    monkeypatch.setattr(async_ray_control, "_discard_late_result", _record_late_result)

    async def _run() -> None:
        fut = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            engine._await_with_keepalive(
                awaitable=fut,
                session=session,
                interval_s=1.0,
                timeout_s=60.0,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not fut.cancelled()

        fut.set_exception(RuntimeError("late boom"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert discarded == ["late boom"]

    asyncio.run(_run())
    pool.unregister(actor_name)


def test_issue_230_unbind_session_keeps_shared_dense_actor_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _get_local_model_actor_inventory(monkeypatch)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
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
    engine._workers[model_id] = shared_worker
    engine._workers[other_model_id] = shared_worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )
    session.actor_name = actor_name
    session.namespace = "mint"

    import mint_server.backend.training.verl.verl_training as verl_training

    killed: list[dict] = []

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)
    monkeypatch.setattr(
        "mint_server.backend.stores.training_session_store.list_training_sessions",
        lambda: [
            {"model_id": model_id, "actor_name": actor_name},
            {"model_id": other_model_id, "actor_name": actor_name},
        ],
    )

    asyncio.run(engine.shutdown_session(session))

    assert killed == []
    assert model_id not in engine._workers
    assert other_model_id in engine._workers
    assert entry.current_session == other_model_id
    assert pool.get(actor_name).current_session == other_model_id

    pool.unregister(actor_name)


def test_issue_230_shutdown_session_keeps_protected_dense_actor_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _get_local_model_actor_inventory(monkeypatch)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
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
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_protected",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )
    session.actor_name = actor_name
    session.namespace = "mint"

    import mint_server.backend.training.verl.verl_training as verl_training

    killed: list[dict] = []

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    asyncio.run(engine.delete_session(session))

    assert worker.calls == 0
    assert killed == []
    assert model_id not in engine._workers
    assert entry.current_session is None
    assert pool.is_protected(actor_name) is True

    pool.unregister(actor_name)
