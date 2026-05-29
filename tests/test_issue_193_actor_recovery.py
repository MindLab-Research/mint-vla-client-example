# ruff: noqa: F403,F405
from tests.issue193_common import *


def test_issue_193_megatron_load_weights_marks_recycled_worker_loaded(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "megatron-actor"

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
    engine._model_actor_supervisor_actor_names[model_id] = "megatron-actor"

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

    async def fake_recycle(recycle_session, *, op, cause, explicit_checkpoint_path=None):
        assert recycle_session is session
        assert explicit_checkpoint_path == "/tmp/issue_193_megatron_ready_recycle"
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


def test_issue_670_bumblebee_explicit_load_retries_after_rank_worker_death(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_load_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_load_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    checkpoint_path = tmp_path / "bumblebee-checkpoint"
    checkpoint_path.mkdir()

    keepalive_calls: list[object] = []
    recycle_calls: list[tuple[str, str]] = []

    async def fake_get_live_worker(_session, *, op, allow_recover=False):
        assert _session is session
        assert op == "load_weights"
        assert allow_recover is False
        return engine._workers[model_id]

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert keepalive_session is session
        keepalive_calls.append(awaitable)
        if awaitable == "dead-load-ref":
            raise ray.exceptions.ActorDiedError()
        if awaitable == "recovered-load-ref":
            return {"current_step": 78, "learning_rate": 2e-5}
        raise AssertionError(f"unexpected awaitable: {awaitable!r}")

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    meta = asyncio.run(
        engine.load_weights(
            session=session,
            load_path=str(checkpoint_path),
            load_optimizer=True,
        )
    )

    assert meta == {"current_step": 78, "learning_rate": 2e-5}
    assert session.current_step == 78
    assert session.learning_rate == pytest.approx(2e-5)
    assert dead_worker.load_checkpoint.calls == [
        ((str(checkpoint_path), True), {"traceparent": None, "session_id": model_id})
    ]
    assert recovered_worker.load_checkpoint.calls == [
        ((str(checkpoint_path), True), {"traceparent": None, "session_id": model_id})
    ]
    assert recycle_calls == [("load_weights", "ActorDiedError")]
    assert keepalive_calls == ["dead-load-ref", "recovered-load-ref"]
    assert model_id not in engine._poisoned_sessions


def test_issue_670_bumblebee_explicit_load_keeps_session_poisoned_when_retry_load_fails(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_load_retry_fails"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_load_retry_fails",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    checkpoint_path = tmp_path / "bumblebee-checkpoint"
    checkpoint_path.mkdir()

    async def fake_get_live_worker(_session, *, op, allow_recover=False):
        assert _session is session
        assert op == "load_weights"
        return engine._workers[model_id]

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert keepalive_session is session
        if awaitable == "dead-load-ref":
            raise ray.exceptions.ActorDiedError()
        if awaitable == "recovered-load-ref":
            raise RuntimeError("corrupt checkpoint")
        raise AssertionError(f"unexpected awaitable: {awaitable!r}")

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    with pytest.raises(RuntimeError, match="corrupt checkpoint"):
        asyncio.run(
            engine.load_weights(
                session=session,
                load_path=str(checkpoint_path),
                load_optimizer=True,
            )
        )

    assert "checkpoint reload must complete successfully" in engine._poisoned_sessions[model_id]

    async def _train_after_failed_reload():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward_backward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="checkpoint reload must complete successfully"):
        asyncio.run(_train_after_failed_reload())


def test_issue_670_bumblebee_explicit_load_recycles_when_rank_liveness_probe_fails(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_load_probe_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_load_probe_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    checkpoint_path = tmp_path / "bumblebee-checkpoint"
    checkpoint_path.mkdir()

    heartbeat_refs: list[object] = []
    recycle_calls: list[tuple[str, str]] = []

    async def fake_async_get_ray_ref(awaitable, timeout_s=None):
        assert awaitable == "heartbeat-ref"
        assert timeout_s == 10
        heartbeat_refs.append(awaitable)
        if len(heartbeat_refs) == 1:
            raise ray.exceptions.ActorDiedError()
        return {"ok": True}

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert keepalive_session is session
        if awaitable == "recovered-load-ref":
            return {"current_step": 79, "learning_rate": 3e-5}
        raise AssertionError(f"unexpected awaitable: {awaitable!r}")

    async def fake_recycle(recycle_session, *, op, cause, request_started=False, explicit_checkpoint_path=None):
        assert recycle_session is session
        assert explicit_checkpoint_path == str(checkpoint_path)
        recycle_calls.append((op, type(cause).__name__))
        return await VerlTrainingEngine._recycle_worker_after_failure(
            engine,
            recycle_session,
            op=op,
            cause=cause,
            request_started=request_started,
            explicit_checkpoint_path=explicit_checkpoint_path,
        )

    async def fake_recycle_bumblebee(recycle_session, *, op, cause):
        assert recycle_session is session
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    monkeypatch.setattr("mint_server.backend.verl_training.async_get_ray_ref", fake_async_get_ray_ref)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_worker_after_failure", fake_recycle)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle_bumblebee)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    meta = asyncio.run(
        engine.load_weights(
            session=session,
            load_path=str(checkpoint_path),
            load_optimizer=True,
        )
    )

    assert meta == {"current_step": 79, "learning_rate": 3e-5}
    assert heartbeat_refs == ["heartbeat-ref", "heartbeat-ref"]
    assert recycle_calls == [("load_weights", "RuntimeError")]
    assert dead_worker.load_checkpoint.calls == []
    assert recovered_worker.load_checkpoint.calls == [
        ((str(checkpoint_path), True), {"traceparent": None, "session_id": model_id})
    ]
    assert model_id not in engine._poisoned_sessions


def test_issue_670_bumblebee_training_op_still_fails_closed_after_rank_worker_death(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_train_recycle"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_train_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )

    submit_workers: list[object] = []
    recycle_calls: list[tuple[str, str]] = []

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        assert awaitable is dead_worker
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward_backward",
            submit_fn=lambda worker: submit_workers.append(worker) or worker,
        )

    with pytest.raises(RuntimeError, match="bumblebee actor recycle detected after op=forward_backward"):
        asyncio.run(_run())

    assert submit_workers == [dead_worker]
    assert recycle_calls == [("forward_backward", "ActorDiedError")]
    assert "reload from checkpoint before retrying" in engine._poisoned_sessions[model_id]


def _issue_670_training_request() -> SimpleNamespace:
    return SimpleNamespace(
        forward_backward_input=SimpleNamespace(
            data=[SimpleNamespace(model_dump=lambda: {"model_input": {}, "loss_fn_inputs": {}})],
            loss_fn="cross_entropy",
            loss_fn_config={},
        ),
        adam_params=SimpleNamespace(learning_rate=1e-4),
    )


def test_issue_670_bumblebee_public_forward_backward_recycles_and_poisons(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_public_fb"
    dead_worker = _FakeWorker(ref="dead-fb-ref")
    recovered_worker = _FakeWorker(ref="recovered-fb-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_public_fb",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    request = _issue_670_training_request()
    recycle_calls: list[tuple[str, str]] = []

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        if awaitable == "heartbeat-ref":
            return {"ok": True}
        assert awaitable == "dead-fb-ref"
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    async def fake_async_get_ray_ref(awaitable, timeout_s=None):
        assert awaitable == "heartbeat-ref"
        assert timeout_s == 10
        return {"ok": True}

    monkeypatch.setattr("mint_server.backend.verl_training.async_get_ray_ref", fake_async_get_ray_ref)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    with pytest.raises(RuntimeError, match="bumblebee actor recycle detected after op=forward_backward"):
        asyncio.run(engine.forward_backward(session, request))

    assert dead_worker.forward_backward.calls == [
        (
            ([{"model_input": {}, "loss_fn_inputs": {}}], "cross_entropy", {}, None, model_id, None),
            {
                "traceparent": None,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
        )
    ]
    assert recovered_worker.forward_backward.calls == []
    assert recycle_calls == [("forward_backward", "ActorDiedError")]
    assert "reload from checkpoint before retrying" in engine._poisoned_sessions[model_id]


@pytest.mark.parametrize(
    ("method_name", "remote_method_name"),
    [
        ("optim_step", "optim_step"),
        ("train_step", "train_step"),
    ],
)
def test_issue_670_bumblebee_public_training_step_ops_recycle_and_poison(
    monkeypatch,
    method_name,
    remote_method_name,
):
    engine = VerlTrainingEngine()
    model_id = f"model_issue_670_bumblebee_public_{method_name}"
    ref = f"dead-{method_name}-ref"
    dead_worker = _FakeWorker(ref=ref)
    recovered_worker = _FakeWorker(ref=f"recovered-{method_name}-ref")
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id=f"session_issue_670_bumblebee_public_{method_name}",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    request = _issue_670_training_request()
    recycle_calls: list[tuple[str, str]] = []

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        assert awaitable == ref
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    async def fake_async_get_ray_ref(awaitable, timeout_s=None):
        assert awaitable == "heartbeat-ref"
        assert timeout_s == 10
        return {"ok": True}

    monkeypatch.setattr("mint_server.backend.verl_training.async_get_ray_ref", fake_async_get_ray_ref)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_bumblebee_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    with pytest.raises(RuntimeError, match=f"bumblebee actor recycle detected after op={method_name}"):
        asyncio.run(getattr(engine, method_name)(session, request))

    assert getattr(dead_worker, remote_method_name).calls
    assert getattr(recovered_worker, remote_method_name).calls == []
    assert recycle_calls == [(method_name, "ActorDiedError")]
    assert "reload from checkpoint before retrying" in engine._poisoned_sessions[model_id]


def test_issue_670_bumblebee_public_training_ops_block_poisoned_session(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_public_poison"
    worker = _FakeWorker(ref="unused-ref")
    engine._workers[model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"
    engine._poisoned_sessions[model_id] = (
        f"[{model_id}] bumblebee actor recycled before explicit load_weights; "
        "checkpoint reload must complete successfully before training can continue."
    )
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_public_poison",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    request = _issue_670_training_request()
    monkeypatch.setattr(
        "mint_server.backend.model_registry.get_model_config",
        lambda *_args, **_kwargs: SimpleNamespace(is_moe=True),
    )

    with pytest.raises(RuntimeError, match="checkpoint reload must complete successfully"):
        asyncio.run(engine.forward_backward(session, request))
    with pytest.raises(RuntimeError, match="checkpoint reload must complete successfully"):
        asyncio.run(engine.optim_step(session, request))
    with pytest.raises(RuntimeError, match="checkpoint reload must complete successfully"):
        asyncio.run(engine.train_step(session, request))

    assert worker.forward_backward.calls == []
    assert worker.optim_step.calls == []
    assert worker.train_step.calls == []


def test_issue_670_bumblebee_successful_explicit_load_unpoisons_public_training(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_670_bumblebee_load_unpoison"
    worker = _FakeLoadWorker(ref="load-ref")
    worker.forward_backward = _RecordingRemoteMethod("fb-ref")
    engine._workers[model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = "bumblebee-actor"
    engine._poisoned_sessions[model_id] = (
        f"[{model_id}] bumblebee actor recycled before explicit load_weights; "
        "checkpoint reload must complete successfully before training can continue."
    )
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_670_bumblebee_load_unpoison",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="bumblebee",
    )
    checkpoint_path = tmp_path / "bumblebee-checkpoint"
    checkpoint_path.mkdir()

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        if awaitable == "load-ref":
            return {"current_step": 80, "learning_rate": 4e-5}
        if awaitable == "fb-ref":
            return {"metrics": {"loss:mean": 0.1}}
        if awaitable == "heartbeat-ref":
            return {"ok": True}
        raise AssertionError(f"unexpected awaitable: {awaitable!r}")

    async def fake_async_get_ray_ref(awaitable, timeout_s=None):
        assert awaitable == "heartbeat-ref"
        assert timeout_s == 10
        return {"ok": True}

    monkeypatch.setattr("mint_server.backend.verl_training.async_get_ray_ref", fake_async_get_ray_ref)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    meta = asyncio.run(
        engine.load_weights(
            session=session,
            load_path=str(checkpoint_path),
            load_optimizer=True,
        )
    )
    assert meta == {"current_step": 80, "learning_rate": 4e-5}
    assert model_id not in engine._poisoned_sessions

    result = asyncio.run(engine.forward_backward(session, _issue_670_training_request()))
    assert result == {"metrics": {"loss:mean": 0.1}}
    assert worker.forward_backward.calls == [
        (
            ([{"model_input": {}, "loss_fn_inputs": {}}], "cross_entropy", {}, None, model_id, None),
            {
                "traceparent": None,
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
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
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
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"
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


def test_issue_651_explicit_load_recycle_defers_same_session_volatile_cleanup(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_651_megatron_explicit_load_dirty"
    actor_name = "shared-megatron-actor"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    recovered_worker = object()
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
    engine._actor_loaded_sessions[actor_name] = model_id
    engine._actor_volatile_sessions[actor_name] = {model_id}

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_651_megatron_explicit_load_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_recycle(recycle_session, *, op, cause, allow_create=True):
        assert recycle_session is session
        assert op == "load_weights"
        assert allow_create is True
        return recovered_worker

    class _CleanSessionManager:
        def list_actor_only_state_sessions(self, actor_name_arg):
            assert actor_name_arg == actor_name
            return []

    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
        _CleanSessionManager,
    )

    result = asyncio.run(
        engine._recycle_worker_after_failure(
            session,
            op="load_weights",
            cause=RuntimeError(f"[{model_id}] missing worker for backend=megatron"),
            explicit_checkpoint_path=str(checkpoint_path),
        )
    )

    assert result is recovered_worker
    assert engine._actor_loaded_sessions[actor_name] == model_id
    assert engine._actor_volatile_sessions[actor_name] == {model_id}
    assert model_id not in engine._poisoned_sessions


def test_issue_651_explicit_load_recycle_still_fails_after_request_started(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_651_megatron_explicit_load_started_dirty"
    actor_name = "shared-megatron-actor"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
    engine._actor_loaded_sessions[actor_name] = model_id
    engine._actor_volatile_sessions[actor_name] = {model_id}

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_651_megatron_explicit_load_started_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_recycle(*_args, **_kwargs):
        raise AssertionError("request-started actor death must fail before recycle")

    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)

    async def _run():
        await engine._recycle_worker_after_failure(
            session,
            op="load_weights",
            cause=ray.exceptions.ActorDiedError(),
            request_started=True,
            explicit_checkpoint_path=str(checkpoint_path),
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_recycle_retries_when_no_live_state_was_lost(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_dead_clean"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"
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
    engine._model_actor_supervisor_actor_names[session_a.model_id] = actor_name
    engine._model_actor_supervisor_actor_names[session_b.model_id] = actor_name

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
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"

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
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"

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
    worker.mark_session_loaded = _RecordingRemoteMethod(_failed_ray_ref(ray.exceptions.ActorDiedError()))
    engine._workers[model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"

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

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

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
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"

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
    engine._model_actor_supervisor_actor_names[model_id] = actor_name

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
    engine._model_actor_supervisor_actor_names[model_id] = actor_name

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
        "mint_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "mint_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "mint_server.backend.model_registry.is_topology_desired_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda actor_name, namespace=None: worker,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
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
    assert mark_ready_calls == ["mint_megatron_qwen3_30b_a3b_instruct_2507"]


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
        "mint_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "mint_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "mint_server.backend.model_registry.is_topology_desired_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "mint_server.backend.megatron_distributed.async_get_or_create_megatron_worker_group",
        lambda **_kwargs: asyncio.sleep(0, result=worker),
    )
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
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
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
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
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
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
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
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
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
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


def test_issue_651_explicit_load_defers_same_session_actor_only_marker_cleanup(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_651_explicit_load_defers_self_dirty"
    actor_name = "shared-megatron-actor"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_651_explicit_load_defers_self_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._model_actor_supervisor_actor_names[model_id] = actor_name

    class _DirtySelfSessionManager:
        def list_actor_only_state_sessions(self, actor_name_arg):
            assert actor_name_arg == actor_name
            return [model_id]

        def clear_actor_only_state(self, session_id):
            raise AssertionError("marker cleanup must wait until checkpoint load succeeds")

        def clear_persisted_actor_only_state(self, session_id):
            raise AssertionError("marker cleanup must wait until checkpoint load succeeds")

        def session_exists(self, session_id):
            raise AssertionError("explicit checkpoint reload should not require an existing session cache")

    monkeypatch.setattr(
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
        _DirtySelfSessionManager,
    )

    error = engine._megatron_missing_actor_recovery_error(
        session,
        op="load_weights",
        cause=RuntimeError(f"[{model_id}] missing worker for backend=megatron"),
        explicit_checkpoint_path=str(checkpoint_path),
    )

    assert error is None


def test_issue_651_explicit_load_preserves_dirty_sibling_fail_closed(monkeypatch, tmp_path):
    engine = VerlTrainingEngine()
    model_id = "model_issue_651_explicit_load_dirty_sibling"
    sibling_id = "model_issue_651_other_dirty_session"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_651_explicit_load_dirty_sibling",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    class _DirtySiblingSessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return [model_id, sibling_id]

        def clear_actor_only_state(self, session_id):
            raise AssertionError("dirty sibling must fail closed before clearing actor-only state")

        def clear_persisted_actor_only_state(self, session_id):
            raise AssertionError("dirty sibling must fail closed before clearing actor-only state")

    monkeypatch.setattr(
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
        _DirtySiblingSessionManager,
    )

    error = engine._megatron_missing_actor_recovery_error(
        session,
        op="load_weights",
        cause=RuntimeError(f"[{model_id}] missing worker for backend=megatron"),
        explicit_checkpoint_path=str(checkpoint_path),
    )

    assert error is not None
    assert "dirty_sibling" in error
    assert sibling_id in error


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
        "mint_server.backend.megatron_distributed.MegatronSessionStateManager",
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


class _Issue651ProbeRemoteMethod:
    def __init__(self, ref):
        self.ref = ref

    def remote(self):
        return self.ref


class _Issue651ProbeWorker:
    def __init__(self, ref):
        self.rank_liveness_probe = _Issue651ProbeRemoteMethod(ref)


def _issue651_worker_group_for_rank_health(world_size=2, refs=None):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = group_cls.__new__(group_cls)
    group.config = SimpleNamespace(world_size=world_size)
    group.workers = [_Issue651ProbeWorker(ref) for ref in (refs or range(world_size))]
    return group


def test_issue_651_rank_worker_health_reports_healthy(monkeypatch):
    group = _issue651_worker_group_for_rank_health(refs=["rank-0", "rank-1"])

    def fake_get(refs, timeout=None):
        assert refs == ["rank-0", "rank-1"]
        assert timeout == 5.0
        return [
            {"ok": True, "rank": 0, "world_size": 2},
            {"ok": True, "rank": 1, "world_size": 2},
        ]

    monkeypatch.setattr(ray, "get", fake_get)

    health = group._get_rank_worker_health_diagnostics()

    assert health["healthy"] is True
    assert health["responded_workers"] == 2
    assert health["ranks"] == [0, 1]


def test_issue_651_rank_worker_health_treats_timeout_as_unknown(monkeypatch):
    group = _issue651_worker_group_for_rank_health(refs=["rank-0", "rank-1"])

    def fake_get(refs, timeout=None):
        raise ray.exceptions.GetTimeoutError("rank worker probe timed out")

    monkeypatch.setattr(ray, "get", fake_get)

    health = group._get_rank_worker_health_diagnostics()

    assert health["healthy"] is None
    assert health["reason"] == "rank_probe_timeout"
    assert health["error_type"] == "GetTimeoutError"
    assert "rank worker probe timed out" in health["error"]
