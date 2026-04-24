# ruff: noqa: F403,F405
from tests.issue193_common import *


def test_issue_193_megatron_load_weights_marks_recycled_worker_loaded(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    live_workers = [dead_worker, recovered_worker]
    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_get_live_worker(*args, **kwargs):
        return live_workers.pop(0)

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    async def fake_run_with_recycle(*args, **kwargs):
        engine._workers[model_id] = recovered_worker
        return _megatron_load_meta(
            current_step=9,
            learning_rate=2e-4,
            actual_rank=7,
            checkpoint_path="/tmp/issue_193_megatron_load_recycle",
        )

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_run_worker_call_with_actor_recycle", fake_run_with_recycle)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_recycle",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert keepalive_calls == [
        ("fake-load-ready-ref", model_id, 30.0, 1800.0),
    ]
    assert dead_worker.mark_session_loaded.calls == []
    assert recovered_worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 9,
                "learning_rate": pytest.approx(2e-4),
                "actual_rank": 7,
                "actor_only_state_dirty": True,
                "checkpoint_path": "/tmp/issue_193_megatron_load_recycle",
                "optimizer_restored": True,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
        )
    ]


def test_issue_193_megatron_load_weights_recovers_when_ready_probe_actor_dies(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_ready_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    recovered_worker.__ray_ready__ = _RecordingRemoteMethod("recovered-ready-ref")
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_ready_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    live_workers = [dead_worker, recovered_worker, recovered_worker]
    recycle_calls: list[tuple[str, str]] = []
    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_get_live_worker(*args, **kwargs):
        return live_workers.pop(0)

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        if awaitable == "fake-load-ready-ref":
            raise ray.exceptions.ActorDiedError()
        return {"status": "ok"}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    async def fake_run_with_recycle(*args, **kwargs):
        return _megatron_load_meta(
            current_step=6,
            learning_rate=4e-4,
            actual_rank=3,
            checkpoint_path="/tmp/issue_193_megatron_ready_recycle",
        )

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_worker_after_failure", fake_recycle)
    monkeypatch.setattr(engine, "_run_worker_call_with_actor_recycle", fake_run_with_recycle)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_ready_recycle",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert recycle_calls == [("load_weights", "ActorDiedError")]
    assert keepalive_calls == [
        ("fake-load-ready-ref", model_id, 30.0, 1800.0),
        ("recovered-ready-ref", model_id, 30.0, 1800.0),
    ]
    assert recovered_worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 6,
                "learning_rate": pytest.approx(4e-4),
                "actual_rank": 3,
                "actor_only_state_dirty": True,
                "checkpoint_path": "/tmp/issue_193_megatron_ready_recycle",
                "optimizer_restored": True,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
        )
    ]


def test_issue_193_megatron_load_weights_missing_actor_can_recreate_from_checkpoint(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_missing_actor"
    worker = _FakeLoadWorker(ref="missing-actor-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_missing_actor",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    get_live_calls: list[tuple[str, bool]] = []

    async def fake_get_live_worker(_session, *, op, allow_recover=False):
        assert _session is session
        get_live_calls.append((op, allow_recover))
        return worker

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "missing-actor-load-ref"
        return _megatron_load_meta(
            current_step=4,
            learning_rate=3e-4,
            actual_rank=6,
            checkpoint_path="/tmp/issue_193_megatron_load_missing_actor",
        )

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_missing_actor",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert get_live_calls[0] == ("load_weights", False)
    assert get_live_calls[-1] == ("load_weights", False)
    assert len(get_live_calls) == 3
    assert worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 4,
                "learning_rate": pytest.approx(3e-4),
                "actual_rank": 6,
                "actor_only_state_dirty": True,
                "checkpoint_path": "/tmp/issue_193_megatron_load_missing_actor",
                "optimizer_restored": True,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
        )
    ]


def test_issue_193_megatron_load_weights_missing_actor_with_dirty_sibling_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_load_missing_actor_dirty_sibling"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_missing_actor_dirty_sibling",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _SiblingDirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return [model_id, "model_issue_193_dirty_sibling"]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 1, "lr": 1e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _SiblingDirtySessionManager,
    )

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_missing_actor_dirty_sibling",
            load_optimizer=True,
        )

    with pytest.raises(RuntimeError, match="dirty_sibling"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_recycle_fails_loud_when_live_state_was_only_in_memory(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_dead_dirty"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"
    engine._actor_loaded_sessions["shared-megatron-actor"] = model_id
    engine._actor_volatile_sessions["shared-megatron-actor"] = {model_id}

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_dead_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    keepalive_calls: list[object] = []
    recycle_calls: list[tuple[str, str]] = []

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_calls.append(awaitable)
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="optim_step",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert keepalive_calls == [dead_worker]
    assert recycle_calls == []
    assert "Reload the lost session from a checkpoint before continuing." in engine._poisoned_sessions[model_id]


def test_issue_193_megatron_recycle_retries_when_no_live_state_was_lost(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_dead_clean"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"
    engine._actor_loaded_sessions["shared-megatron-actor"] = model_id

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_dead_clean",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    submit_workers: list[object] = []
    recycle_calls: list[tuple[str, str]] = []
    keepalive_count = {"n": 0}

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_count["n"] += 1
        if keepalive_count["n"] == 1:
            raise ray.exceptions.ActorDiedError()
        return {"ok": True}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: submit_workers.append(worker) or worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert submit_workers == [dead_worker, recovered_worker]
    assert recycle_calls == [("forward", "ActorDiedError")]
    assert model_id not in engine._poisoned_sessions


def test_issue_193_megatron_switched_out_dirty_session_still_poisoned_on_actor_death(monkeypatch):
    engine = VerlTrainingEngine()
    actor_name = "shared-megatron-actor"
    session_a = TrainingSession(
        model_id="model_issue_193_megatron_session_a",
        session_id="session_a",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    session_b = TrainingSession(
        model_id="model_issue_193_megatron_session_b",
        session_id="session_b",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[session_a.model_id] = actor_name
    engine._resource_pool_actor_names[session_b.model_id] = actor_name

    engine._note_successful_worker_call(session_a, op="forward_backward")
    engine._note_successful_worker_call(session_b, op="forward")

    recycle_calls: list[tuple[str, str]] = []

    async def fake_recycle(recycle_session, *, op, cause):
        recycle_calls.append((recycle_session.model_id, op))
        return object()

    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)

    async def _run():
        await engine._recycle_worker_after_failure(
            session_b,
            op="forward",
            cause=ray.exceptions.ActorDiedError(),
        )

    with pytest.raises(RuntimeError, match="session\\(s\\) model_issue_193_megatron_session_a"):
        asyncio.run(_run())

    assert recycle_calls == []
    assert engine._actor_volatile_sessions.get(actor_name) is None
    assert session_a.model_id in engine._poisoned_sessions


def test_issue_193_megatron_adapter_only_load_restore_stays_recoverable_until_next_train_step(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_loaded_clean"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_loaded_clean",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    # load_optimizer=False path clears volatility because adapter shards plus
    # metadata are persisted to the Megatron session cache.
    engine._note_successful_worker_call(session, op="load_weights")

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    submit_workers: list[object] = []
    recycle_calls: list[tuple[str, str]] = []
    keepalive_count = {"n": 0}

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_count["n"] += 1
        if keepalive_count["n"] == 1:
            raise ray.exceptions.ActorDiedError()
        return {"ok": True}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: submit_workers.append(worker) or worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert submit_workers == [dead_worker, recovered_worker]
    assert recycle_calls == [("forward", "ActorDiedError")]
    assert engine._actor_volatile_sessions.get("shared-megatron-actor") is None
    assert model_id not in engine._poisoned_sessions


def test_issue_193_megatron_load_weights_with_optimizer_keeps_session_volatile(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_with_optimizer"
    worker = _FakeLoadWorker(ref="megatron-load-with-optimizer-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_with_optimizer",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "megatron-load-with-optimizer-ref"
        return _megatron_load_meta(
            current_step=4,
            learning_rate=7e-5,
            actual_rank=6,
            checkpoint_path="/tmp/issue_193_megatron_load_with_optimizer",
        )

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_with_optimizer",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert engine._actor_volatile_sessions["shared-megatron-actor"] == {model_id}


def test_issue_193_megatron_load_weights_keeps_session_volatile_until_mark_loaded_finishes(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_mark_gap"
    worker = _FakeLoadWorker(ref="megatron-load-mark-gap-ref")
    worker.mark_session_loaded = _RecordingRemoteMethod("mark-loaded-gap-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_mark_gap",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "megatron-load-mark-gap-ref"
        return _megatron_load_meta(
            current_step=8,
            learning_rate=1e-4,
            actual_rank=5,
            checkpoint_path="/tmp/issue_193_megatron_load_mark_gap",
        )

    def fake_ray_get(ref, timeout=None):
        if ref == "mark-loaded-gap-ref":
            raise ray.exceptions.ActorDiedError()
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_mark_gap",
            load_optimizer=True,
        )

    with pytest.raises(ray.exceptions.ActorDiedError):
        asyncio.run(_run())

    assert engine._actor_volatile_sessions["shared-megatron-actor"] == {model_id}


def test_issue_193_megatron_train_step_marks_session_volatile(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_train_step_dirty"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_train_step_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._note_successful_worker_call(session, op="train_step")

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert engine._actor_volatile_sessions.get("shared-megatron-actor") is None
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_sampler_save_does_not_clear_volatile_train_state(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_sampler_volatile"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_sampler_volatile",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name

    engine._note_successful_worker_call(session, op="forward_backward")
    engine._note_successful_worker_call(session, op="save_lora_weights_for_sampler")

    assert engine._actor_volatile_sessions[actor_name] == {model_id}


def test_issue_193_megatron_save_weights_does_not_clear_volatile_train_state():
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_actor_only_marker"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_actor_only_marker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name

    engine._note_successful_worker_call(session, op="forward_backward")
    engine._note_successful_worker_call(session, op="save_weights")

    assert engine._actor_volatile_sessions[actor_name] == {model_id}


def test_issue_193_megatron_missing_worker_rebinds_before_recycle(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_missing_worker"
    rebound_worker = object()
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_worker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[str] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        assert allow_create is False
        rebind_calls.append(reason)
        return rebound_worker

    async def fail_recycle(*args, **kwargs):
        raise AssertionError("missing megatron worker should rebind before recycle")

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_worker_after_failure", fail_recycle)
    monkeypatch.setattr(engine, "_await_with_keepalive", lambda awaitable, *_args, **_kwargs: asyncio.sleep(0, result={"ok": True}))
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert rebind_calls == ["forward:missing_worker"]


def test_issue_193_megatron_rebind_reuses_existing_actor_without_ready_probe(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_rebind_registers_pool"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_rebind_registers_pool",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    register_calls: list[tuple[tuple, dict]] = []
    mark_ready_calls: list[str] = []

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.is_persistent_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda actor_name, namespace=None: worker,
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(
            register=lambda *args, **kwargs: register_calls.append((args, kwargs)),
            mark_ready=lambda actor_name: mark_ready_calls.append(actor_name),
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
        ),
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    rebound = asyncio.run(
        engine._rebind_megatron_worker(
            session,
            reason="unit_test",
            allow_create=False,
        )
    )

    assert rebound is worker
    assert keepalive_calls == []
    assert len(register_calls) == 1
    assert register_calls[0][1]["session_id"] == model_id
    assert register_calls[0][1]["num_gpus"] == 1
    assert mark_ready_calls == ["megatron_qwen3_30b_a3b_instruct_2507"]


def test_issue_193_megatron_rebind_created_actor_ready_death_maps_to_missing_worker(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_rebind_ready_death"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_rebind_ready_death",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.is_persistent_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.async_get_or_create_megatron_worker_group",
        lambda **_kwargs: asyncio.sleep(0, result=worker),
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(
            register=lambda *args, **kwargs: None,
            mark_ready=lambda *_args, **_kwargs: None,
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
        ),
    )

    async def fake_keepalive(*_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    with pytest.raises(RuntimeError, match="missing worker for backend=megatron"):
        asyncio.run(
            engine._rebind_megatron_worker(
                session,
                reason="unit_test",
                allow_create=True,
            )
        )


def test_issue_193_megatron_missing_worker_with_live_state_still_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_missing_worker_dirty"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_worker_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name
    engine._actor_volatile_sessions[actor_name] = {model_id}

    async def fake_rebind(*_args, **_kwargs):
        raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        assert isinstance(cause, RuntimeError)
        return object()

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_without_cache_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_no_cache"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_no_cache",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    class _MissingCacheSessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return []

        def session_exists(self, session_id):
            assert session_id == model_id
            return False

        def get_metadata(self, session_id):
            assert session_id == model_id
            return None

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _MissingCacheSessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="has no persisted Megatron session cache"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_invalid_session_metadata_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_invalid_meta"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_invalid_meta",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    class _InvalidMetaSessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return []

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return None

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _InvalidMetaSessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="missing session_metadata.json"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_with_persisted_dirty_marker_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_dirty_marker"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_dirty_marker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _DirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return [model_id]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 7, "lr": 1e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _DirtySessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="actor-only training state that was never fully persisted"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_with_dirty_sibling_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_clean_session"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_clean_session",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _SiblingDirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return ["model_issue_193_megatron_dirty_sibling"]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 2, "lr": 2e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _SiblingDirtySessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="dirty_sibling"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions
