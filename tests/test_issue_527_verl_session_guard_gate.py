# ruff: noqa: F403,F405
from tests.issue193_common import *


class _FakeGuardedForwardWorker:
    def __init__(self, guard_ref: str, forward_ref: str):
        self.get_session_guard_state = _RecordingRemoteMethod(guard_ref)
        self.forward = _RecordingRemoteMethod(forward_ref)


def _make_forward_request() -> SimpleNamespace:
    item = SimpleNamespace(model_dump=lambda: {"model_input": {"chunks": []}})
    return SimpleNamespace(
        forward_input=SimpleNamespace(data=[item]),
    )


def test_issue_527_verl_forward_fails_closed_when_megatron_guard_is_contaminated(monkeypatch):
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="model_issue_527_guard_contaminated",
        session_id="session_issue_527_guard_contaminated",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    worker = _FakeGuardedForwardWorker(
        guard_ref="guard-ref",
        forward_ref="forward-ref",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        if awaitable == "guard-ref":
            return {
                "session_id": session.model_id,
                "contaminated": True,
                "blocked": False,
                "contamination_reason": "forward_backward:group_timeout:600s",
                "block_reason": None,
            }
        raise AssertionError("forward should not execute when guard is contaminated")

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_touch_actor", lambda *_args, **_kwargs: None)

    async def _run():
        await engine.forward(session, _make_forward_request())

    with pytest.raises(RuntimeError, match="requires clean reload"):
        asyncio.run(_run())


def test_issue_527_verl_forward_allows_clean_guard(monkeypatch):
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="model_issue_527_guard_clean",
        session_id="session_issue_527_guard_clean",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    worker = _FakeGuardedForwardWorker(
        guard_ref="guard-ref",
        forward_ref="forward-ref",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        if awaitable == "guard-ref":
            return {
                "session_id": session.model_id,
                "contaminated": False,
                "blocked": False,
                "contamination_reason": None,
                "block_reason": None,
            }
        if awaitable == "forward-ref":
            return {
                "loss_fn_outputs": [{"loss": {"data": [0.0]}}],
                "metrics": {"num_tokens": 0},
            }
        raise AssertionError(f"unexpected awaitable={awaitable!r}")

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_touch_actor", lambda *_args, **_kwargs: None)

    async def _run():
        return await engine.forward(session, _make_forward_request())

    result = asyncio.run(_run())
    assert result["loss_fn_outputs"][0]["loss"]["data"] == [0.0]
    assert len(worker.forward.calls) == 1
