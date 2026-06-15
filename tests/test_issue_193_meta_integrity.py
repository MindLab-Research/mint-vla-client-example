# ruff: noqa: F403,F405
from tests.issue193_common import *


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42", "learning_rate": "oops"},
        ["not-a-dict"],
    ],
    ids=["missing_current_step", "current_step_wrong_type", "meta_wrong_type"],
)
def test_issue_193_load_weights_invalid_meta_warns_without_pollution(monkeypatch, caplog, meta_payload):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_load"
    worker = _FakeLoadWorker(ref="invalid-meta-load-ref")
    engine._workers[model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = "shared-actor"
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.current_step = 77
    session.learning_rate = 1.5e-4

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert awaitable == "invalid-meta-load-ref"
        assert keepalive_session is session
        return meta_payload

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_invalid_meta_load",
            load_optimizer=True,
        )

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run())

    assert session.current_step == 77
    assert session.learning_rate == pytest.approx(1.5e-4)
    assert any("load_weights" in rec.getMessage() for rec in caplog.records)


def test_issue_193_megatron_load_weights_invalid_meta_fails_loud(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_megatron_load"
    worker = _FakeLoadWorker(ref="invalid-meta-megatron-load-ref")
    engine._workers[model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = "shared-megatron-actor"
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_megatron_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    session.current_step = 12
    session.learning_rate = 9e-5

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert keepalive_session is session
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "invalid-meta-megatron-load-ref"
        return ["not-a-dict"]

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_invalid_meta_megatron_load",
            load_optimizer=True,
        )

    with pytest.raises(RuntimeError, match="Megatron load_checkpoint returned invalid metadata"):
        asyncio.run(_run())

    assert session.current_step == 12
    assert session.learning_rate == pytest.approx(9e-5)
    assert worker.mark_session_loaded.calls == []


def test_issue_193_megatron_create_training_session_waits_for_ready_probe(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_create_ready"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_create_ready",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="peft",
    )

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "mint_server.backend.training.megatron.megatron_distributed.async_get_or_create_megatron_worker_group",
        lambda **kwargs: asyncio.sleep(0, result=worker),
    )
    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
            mark_ready=lambda *_args, **_kwargs: None,
            clear_session=lambda *_args, **_kwargs: None,
        ),
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.create_training_session(session)

    asyncio.run(_run())

    assert keepalive_calls == [("fake-load-ready-ref", model_id, 30.0, 3600.0)]
    assert engine._workers[model_id] is worker
    assert session.backend == "megatron"
    assert session.is_active is True


@pytest.mark.parametrize(
    "raised_error",
    [
        ray.exceptions.GetTimeoutError("save_lora_weights timed out"),
        RuntimeError("save_lora_weights failed"),
    ],
    ids=["timeout", "runtime_error"],
)
def test_issue_193_save_lora_weights_for_sampler_propagates_errors_without_step_pollution(
    monkeypatch,
    raised_error,
):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_sampler_error"
    worker = _FakeSamplerWorker(ref="fake-save-lora-ref-error")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_sampler_error",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 88

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert awaitable == "fake-save-lora-ref-error"
        assert keepalive_session is session
        assert timeout_s is not None
        raise raised_error

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_lora_error",
        )

    with pytest.raises(type(raised_error)):
        asyncio.run(_run())

    assert session.current_step == 88
    assert len(worker.save_lora_weights.calls) == 1
    _, kwargs = worker.save_lora_weights.calls[0]
    assert kwargs["session_id"] == model_id


def test_issue_193_save_lora_weights_for_sampler_retry_same_session_is_idempotent(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_sampler_retry"
    worker = _FakeSamplerWorker(ref="fake-save-lora-ref-retry")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_sampler_retry",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 15

    call_count = {"n": 0}

    def fake_ray_get(ref, timeout=None):
        assert ref == "fake-save-lora-ref-retry"
        assert timeout is not None
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient save_lora failure")
        return {"current_step": 66}

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run_save():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_lora_retry",
        )

    with pytest.raises(RuntimeError, match="transient save_lora failure"):
        asyncio.run(_run_save())
    assert session.current_step == 15

    saved_path = asyncio.run(_run_save())
    assert saved_path == str(Path("/tmp/issue_193_lora_retry").resolve())
    assert session.current_step == 66

    assert len(worker.save_lora_weights.calls) == 2
    first_args, first_kwargs = worker.save_lora_weights.calls[0]
    second_args, second_kwargs = worker.save_lora_weights.calls[1]
    assert first_args == second_args == (str(Path("/tmp/issue_193_lora_retry").resolve()),)
    assert first_kwargs["session_id"] == model_id
    assert second_kwargs["session_id"] == model_id


def test_issue_193_same_model_concurrent_save_weights_out_of_order_steps_do_not_regress(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_same_model_save"

    class _SequenceCheckpointRemote:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []
            self._count = 0

        def remote(self, *args, **kwargs):
            self._count += 1
            self.calls.append((args, kwargs))
            return _labeled_ray_ref(f"same-model-save-ref-{self._count}")

    remote = _SequenceCheckpointRemote()
    engine._workers[model_id] = SimpleNamespace(save_checkpoint=remote)

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_same_model_save",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 0

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "same-model-save-ref-1":
            time.sleep(0.01)
            return {"current_step": 101}
        if ref == "same-model-save-ref-2":
            time.sleep(0.02)
            return {"current_step": 99}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_weights(session, "/tmp/issue_193_same_model_ckpt_a"),
            engine.save_weights(session, "/tmp/issue_193_same_model_ckpt_b"),
        )

    _ = asyncio.run(_run())
    assert session.current_step == 101


def test_issue_193_same_model_concurrent_save_lora_out_of_order_steps_do_not_regress(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_same_model_lora"

    class _SequenceLoraRemote:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []
            self._count = 0

        def remote(self, *args, **kwargs):
            self._count += 1
            self.calls.append((args, kwargs))
            return _labeled_ray_ref(f"same-model-lora-ref-{self._count}")

    remote = _SequenceLoraRemote()
    engine._workers[model_id] = SimpleNamespace(save_lora_weights=remote)

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_same_model_lora",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 0

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "same-model-lora-ref-1":
            time.sleep(0.01)
            return {"current_step": 101}
        if ref == "same-model-lora-ref-2":
            time.sleep(0.02)
            return {"current_step": 99}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_lora_weights_for_sampler(session, "/tmp/issue_193_same_model_lora_a"),
            engine.save_lora_weights_for_sampler(session, "/tmp/issue_193_same_model_lora_b"),
        )

    _ = asyncio.run(_run())
    assert session.current_step == 101


def test_issue_193_save_weights_invalid_meta_fails_in_strict_megatron_mode(monkeypatch):
    meta_payload = {}
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_save"
    worker = _FakeWorker(ref="invalid-meta-save-ref")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_save",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-save-ref"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_weights(session, "/tmp/issue_193_invalid_meta_save")

    with pytest.raises(ValueError, match="save_weights"):
        asyncio.run(_run())

    assert session.current_step == 77


def test_issue_193_save_lora_invalid_meta_fails_in_strict_megatron_mode(monkeypatch):
    meta_payload = {"current_step": "42"}
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_lora"
    worker = _FakeSamplerWorker(ref="invalid-meta-lora-ref")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_lora",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-lora-ref"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_invalid_meta_lora",
        )

    with pytest.raises(ValueError, match="save_lora_weights_for_sampler"):
        asyncio.run(_run())

    assert session.current_step == 77


def test_issue_193_save_weights_invalid_meta_non_strict_mode_warns_without_pollution(
    monkeypatch,
    caplog,
):
    meta_payload = {}
    monkeypatch.setenv("MINT_MEGATRON_STRICT_SAVE_META", "0")

    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_save_non_strict"
    worker = _FakeWorker(ref="invalid-meta-save-ref-non-strict")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_save_non_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-save-ref-non-strict"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_weights(session, "/tmp/issue_193_invalid_meta_save_non_strict")

    with caplog.at_level(logging.WARNING):
        _ = asyncio.run(_run())

    assert session.current_step == 77
    assert any("save_weights" in rec.getMessage() for rec in caplog.records)


def test_issue_193_save_lora_invalid_meta_non_strict_mode_warns_without_pollution(
    monkeypatch,
    caplog,
):
    meta_payload = {"current_step": "42"}
    monkeypatch.setenv("MINT_MEGATRON_STRICT_SAVE_META", "0")

    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_lora_non_strict"
    worker = _FakeSamplerWorker(ref="invalid-meta-lora-ref-non-strict")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_lora_non_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-lora-ref-non-strict"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_invalid_meta_lora_non_strict",
        )

    with caplog.at_level(logging.WARNING):
        _ = asyncio.run(_run())

    assert session.current_step == 77
    assert any("save_lora_weights_for_sampler" in rec.getMessage() for rec in caplog.records)
