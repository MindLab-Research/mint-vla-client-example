import asyncio

import pytest

from mint_server.backend.sessions.session_manager import SessionManager
from mint_server.routes import sampling as sampling_route


class _FakeEngine:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.remove_calls: list[str] = []

    async def remove_session(self, session_id: str) -> None:
        self.remove_calls.append(session_id)
        if self.error is not None:
            raise self.error


class _FakeMultiModelManager:
    def __init__(self, engine: _FakeEngine):
        self.engine = engine

    def get_engine_if_exists(self, _base_model: str) -> _FakeEngine:
        return self.engine


@pytest.fixture(autouse=True)
def _clear_lora_load_locks() -> None:
    sampling_route._lora_load_locks.clear()
    yield
    sampling_route._lora_load_locks.clear()


def test_issue_362_end_session_drops_lora_load_lock_after_engine_teardown() -> None:
    manager = SessionManager()
    engine = _FakeEngine()
    manager.set_multi_model_manager(_FakeMultiModelManager(engine))
    manager.register_multi_lora_session(
        session_id="sess-362",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=32,
        adapter_path=None,
        lora_loaded=False,
    )

    asyncio.run(sampling_route._get_lora_load_lock("sess-362"))
    assert "sess-362" in sampling_route._lora_load_locks

    ended = asyncio.run(manager.end_session("sess-362"))

    assert ended is True
    assert engine.remove_calls == ["sess-362"]
    assert "sess-362" not in sampling_route._lora_load_locks


def test_issue_362_end_session_still_drops_lora_load_lock_when_engine_remove_fails() -> None:
    manager = SessionManager()
    engine = _FakeEngine(error=RuntimeError("transient remove failure"))
    manager.set_multi_model_manager(_FakeMultiModelManager(engine))
    manager.register_multi_lora_session(
        session_id="sess-362-fail",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=32,
        adapter_path=None,
        lora_loaded=False,
    )

    asyncio.run(sampling_route._get_lora_load_lock("sess-362-fail"))
    assert "sess-362-fail" in sampling_route._lora_load_locks

    ended = asyncio.run(manager.end_session("sess-362-fail"))

    assert ended is True
    assert engine.remove_calls == ["sess-362-fail"]
    assert "sess-362-fail" not in sampling_route._lora_load_locks
