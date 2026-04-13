from __future__ import annotations

from types import SimpleNamespace
import time

import pytest

from tinker_server.backend.training_session_manager import TrainingSessionManager
from tinker_server.routes import training as training_route


@pytest.fixture

def anyio_backend() -> str:
    return "asyncio"


def test_issue_364_training_snapshot_updates_on_newer_version() -> None:
    manager = TrainingSessionManager()
    manager.create_session(
        model_id="model-364",
        session_id="session-364",
        model_seq_id=1,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        learning_rate=1e-4,
    )

    snapshot = manager.get_training_session_snapshot("model-364")
    assert snapshot is not None
    assert snapshot.metadata_version == 1

    manager.restore_training_session_info(
        {
            "model_id": "model-364",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "learning_rate": 5e-5,
            "user_metadata": {"tag": "v2"},
            "metadata_version": 2,
        }
    )

    snapshot = manager.get_training_session_snapshot("model-364")
    assert snapshot is not None
    assert snapshot.metadata_version == 2
    assert snapshot.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert snapshot.backend == "megatron"
    assert snapshot.current_step == 7


def test_issue_364_training_snapshot_ignores_stale_version() -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-stale",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 5,
            "metadata_version": 3,
        }
    )

    manager.restore_training_session_info(
        {
            "model_id": "model-364-stale",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "megatron",
            "current_step": 2,
            "metadata_version": 2,
        }
    )

    snapshot = manager.get_training_session_snapshot("model-364-stale")
    assert snapshot is not None
    assert snapshot.metadata_version == 3
    assert snapshot.base_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert snapshot.backend == "peft"
    assert snapshot.current_step == 5


@pytest.mark.anyio
async def test_issue_364_training_route_restores_snapshot_from_detached_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-restore"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 2,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 11,
            "learning_rate": 5e-5,
            "user_metadata": {"source": "detached"},
            "metadata_version": 4,
        }

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    session, snapshot = await training_route._get_training_session_for_request("model-364-restore")

    assert session is not None
    assert snapshot is not None
    assert snapshot.metadata_version == 4
    assert snapshot.current_step == 11
    assert snapshot.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert session.backend == "megatron"


@pytest.mark.anyio
async def test_issue_364_training_route_refreshes_step_without_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-step",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 3,
            "metadata_version": 7,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-step"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 9,
            "metadata_version": 7,
        }

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager._sessions.get(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(training_route, "_has_training_worker_binding", lambda _model_id: True)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    session, snapshot = await training_route._get_training_session_for_request("model-364-step")

    assert session is not None
    assert snapshot is not None
    assert snapshot.metadata_version == 7
    assert snapshot.current_step == 9
    assert session.current_step == 9


@pytest.mark.anyio
async def test_issue_364_owner_cleanup_respects_detached_training_last_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager(inactivity_timeout=1)
    manager.restore_training_session_info(
        {
            "model_id": "model-364-live-cleanup",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 2,
            "metadata_version": 3,
            "last_activity": 0.0,
        }
    )
    cleaned: list[str] = []

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-live-cleanup"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 2,
            "metadata_version": 3,
            "last_activity": time.time(),
        }

    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    async def _cleanup_session(model_id: str):
        cleaned.append(model_id)

    monkeypatch.setattr(manager, "_cleanup_session", _cleanup_session)

    await manager._cleanup_inactive()

    assert cleaned == []


@pytest.mark.anyio
async def test_issue_364_training_route_drops_stale_local_snapshot_when_store_entry_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-gone",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 4,
            "metadata_version": 2,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-gone"
        return None

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    session, snapshot = await training_route._get_training_session_for_request("model-364-gone")

    assert session is None
    assert snapshot is None
    assert manager.get_session("model-364-gone") is None


@pytest.mark.anyio
async def test_issue_364_get_model_info_rejects_stale_local_training_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-readonly",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 6,
            "metadata_version": 2,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-readonly"
        return None

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    with pytest.raises(Exception):
        await training_route.get_model_info("model-364-readonly")

    assert manager.get_session("model-364-readonly") is None


@pytest.mark.anyio
async def test_issue_364_list_training_runs_ignores_stale_local_training_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-stale-list",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 5,
            "metadata_version": 2,
        }
    )

    async def _async_list_training_sessions():
        return []

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_list_training_sessions",
        _async_list_training_sessions,
    )

    out = await training_route.list_training_runs(limit=20, offset=0, http_request=None)

    assert out.training_runs == []
    assert manager.get_session("model-364-stale-list") is None


@pytest.mark.anyio
async def test_issue_364_get_training_run_refreshes_read_only_metadata_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-read-refresh",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 2,
            "metadata_version": 1,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-read-refresh"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 9,
            "metadata_version": 3,
            "user_id": "owner-364",
            "created_at": "2026-03-26T00:00:00",
            "last_activity": 180.0,
        }

    monkeypatch.setattr(training_route.time, "time", lambda: 200.0)
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    out = await training_route.get_training_run("model-364-read-refresh", SimpleNamespace(state=SimpleNamespace(user_data=None)))

    assert out.training_run_id == "model-364-read-refresh"
    assert out.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert out.last_activity == 180.0
    assert out.idle_for_s == pytest.approx(20.0)


@pytest.mark.anyio
async def test_issue_364_list_training_runs_refreshes_local_stale_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-read-list-refresh",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 2,
            "metadata_version": 1,
        }
    )

    async def _async_list_training_sessions():
        return [
            {
                "model_id": "model-364-read-list-refresh",
                "session_id": "session-364",
                "model_seq_id": 1,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "current_step": 8,
                "metadata_version": 4,
                "created_at": "2026-03-26T00:00:00",
                "user_id": "owner-364",
                "last_activity": 190.0,
            }
        ]

    monkeypatch.setattr(training_route.time, "time", lambda: 200.0)
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_list_training_sessions",
        _async_list_training_sessions,
    )

    out = await training_route.list_training_runs(limit=20, offset=0, http_request=None)

    assert len(out.training_runs) == 1
    assert out.training_runs[0].training_run_id == "model-364-read-list-refresh"
    assert out.training_runs[0].base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert out.training_runs[0].last_activity == 190.0
    assert out.training_runs[0].idle_for_s == pytest.approx(10.0)


@pytest.mark.anyio
async def test_issue_364_get_model_info_refreshes_read_only_metadata_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-model-info",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 1,
            "metadata_version": 1,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-model-info"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "last_activity": 180.0,
            "metadata_version": 3,
        }

    monkeypatch.setattr(training_route.time, "time", lambda: 200.0)
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    out = await training_route.get_model_info("model-364-model-info")

    assert out["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert out["backend"] == "megatron"
    assert out["current_step"] == 7
    assert out["last_activity"] == 180.0
    assert out["idle_for_s"] == pytest.approx(20.0)


@pytest.mark.anyio
async def test_issue_364_get_info_refreshes_read_only_metadata_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-get-info",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 1,
            "metadata_version": 1,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-get-info"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": 3,
        }

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )
    monkeypatch.setattr(
        "tinker_server.gateway.async_remote_training_model",
        lambda _model_id: None,
    )

    out = await training_route.get_info(
        SimpleNamespace(model_id="model-364-get-info", model_dump=lambda: {"model_id": "model-364-get-info"}),
        SimpleNamespace(state=SimpleNamespace(user_data=None), headers={}),
    )

    assert out.model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert out.model_data.model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.mark.anyio
async def test_issue_364_list_models_refreshes_read_only_metadata_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-model-list",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 1,
            "metadata_version": 1,
        }
    )

    async def _async_list_training_sessions():
        return [
            {
                "model_id": "model-364-model-list",
                "session_id": "session-364",
                "model_seq_id": 1,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "current_step": 6,
                "last_activity": 190.0,
                "metadata_version": 3,
            }
        ]

    monkeypatch.setattr(training_route.time, "time", lambda: 200.0)
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_list_training_sessions",
        _async_list_training_sessions,
    )

    out = await training_route.list_models()

    assert out["total"] == 1
    assert out["models"][0]["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert out["models"][0]["current_step"] == 6
    assert out["models"][0]["last_activity"] == 190.0
    assert out["models"][0]["idle_for_s"] == pytest.approx(10.0)


@pytest.mark.anyio
async def test_issue_364_get_tokenizer_refreshes_read_only_metadata_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-tokenizer",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "current_step": 1,
            "metadata_version": 1,
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-tokenizer"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": 3,
        }

    async def _fake_get_tokenizer_info(session):
        return {"model_name": session.base_model}

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        training_route,
        "training_engine",
        SimpleNamespace(_workers={}, _resource_pool_actor_names={}, get_tokenizer_info=_fake_get_tokenizer_info),
    )
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    out = await training_route.get_tokenizer("model-364-tokenizer")

    assert out["tokenizer"]["model_name"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
