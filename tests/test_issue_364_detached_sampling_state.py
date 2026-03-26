from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest
from fastapi import HTTPException

from tinker_server import app as app_module
from tinker_server.backend.session_manager import SessionManager
from tinker_server.routes import service as service_route


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRegistry:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    async def get_lora_id(self, session_id: str) -> int | None:
        return self.ids.get(session_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.registry = _FakeRegistry()
        self.restore_calls: list[tuple[str, str, int]] = []

    async def restore_loaded_session(
        self,
        *,
        sampling_session_id: str,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        self.restore_calls.append((sampling_session_id, adapter_path, lora_int_id))
        self.registry.ids[sampling_session_id] = int(lora_int_id)
        return int(lora_int_id)


class _FakeMultiModelManager:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.calls: list[str] = []

    async def get_engine(self, model_name: str) -> _FakeEngine:
        self.calls.append(model_name)
        return self.engine


def _request_stub(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def test_issue_364_register_multi_lora_session_persists_detached_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: list[dict] = []

    monkeypatch.setattr(
        "tinker_server.backend.sampling_session_store.upsert_sampling_session",
        lambda info: persisted.append(dict(info)),
    )

    manager = SessionManager()
    manager.register_multi_lora_session(
        session_id="sess-364",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=64,
        adapter_path="/tmp/adapter-364",
        lora_loaded=False,
    )
    manager.mark_session_lora_loaded("sess-364", True, lora_int_id=17)

    assert persisted[0]["session_id"] == "sess-364"
    assert persisted[0]["lora_loaded"] is False
    assert persisted[-1]["lora_loaded"] is True
    assert persisted[-1]["lora_int_id"] == 17


def test_issue_364_get_engine_for_session_rehydrates_loaded_lora_mapping() -> None:
    manager = SessionManager()
    engine = _FakeEngine()
    multi_model_manager = _FakeMultiModelManager(engine)
    manager.set_multi_model_manager(multi_model_manager)
    manager.restore_sampling_session(
        {
            "session_id": "sess-364-restore",
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "lora_rank": 32,
            "adapter_path": "/shared/adapter-364",
            "lora_loaded": True,
            "lora_int_id": 9,
            "uses_base_model": False,
            "last_activity": 123.0,
        }
    )

    restored_engine = asyncio.run(manager.get_engine_for_session("sess-364-restore"))

    assert restored_engine is engine
    assert multi_model_manager.calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert engine.restore_calls == [("sess-364-restore", "/shared/adapter-364", 9)]
    assert manager.get_session_lora_int_id("sess-364-restore") == 9


@pytest.mark.anyio
async def test_issue_364_app_restore_sampling_sessions_reads_detached_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager()

    async def _async_list_sampling_sessions():
        return [
            {
                "session_id": "sess-364-base",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "uses_base_model": True,
                "last_activity": 1.0,
            },
            {
                "session_id": "sess-364-lora",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "lora_rank": 32,
                "adapter_path": "/shared/lora-364",
                "lora_loaded": True,
                "lora_int_id": 4,
                "uses_base_model": False,
                "last_activity": 2.0,
            },
        ]

    monkeypatch.setattr(
        "tinker_server.backend.sampling_session_store.async_list_sampling_sessions",
        _async_list_sampling_sessions,
    )

    restored = await app_module._restore_sampling_sessions(manager)

    assert restored == 2
    assert manager.is_base_model_session("sess-364-base") is True
    assert manager.get_session_base_model("sess-364-lora") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert manager.get_session_lora_int_id("sess-364-lora") == 4


def test_issue_364_get_session_no_longer_uses_process_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_get_session_index(_session_id: str):
        return None

    monkeypatch.setattr(
        "tinker_server.backend.session_index_store.async_get_session_index",
        _async_get_session_index,
    )
    monkeypatch.setattr(
        service_route,
        "sessions",
        {"sess-local-only": {"user_id": "admin", "created_at": "2026-03-25T00:00:00"}},
        raising=False,
    )

    with pytest.raises(HTTPException, match="not found"):
        anyio.run(service_route.get_session, "sess-local-only", _request_stub("admin"))


def test_issue_364_list_sessions_no_longer_uses_process_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_list_session_index():
        return []

    monkeypatch.setattr(
        "tinker_server.backend.session_index_store.async_list_session_index",
        _async_list_session_index,
    )
    monkeypatch.setattr(
        service_route,
        "sessions",
        {"sess-local-only": {"user_id": "admin", "created_at": "2026-03-25T00:00:00"}},
        raising=False,
    )

    out = anyio.run(service_route.list_sessions, 20, 0, _request_stub("admin"))

    assert out.sessions == []
