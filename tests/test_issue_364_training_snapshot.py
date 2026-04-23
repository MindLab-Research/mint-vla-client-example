from __future__ import annotations

from types import SimpleNamespace
import time

import pytest

from tinker_server.backend.training_session_manager import (
    MATERIALIZATION_STATE_READY,
    TRAINING_SESSION_METADATA_VERSION,
    TrainingSessionManager,
)
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
    assert snapshot.metadata_version == TRAINING_SESSION_METADATA_VERSION

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
            "metadata_version": TRAINING_SESSION_METADATA_VERSION + 1,
        }
    )

    snapshot = manager.get_training_session_snapshot("model-364")
    assert snapshot is not None
    assert snapshot.metadata_version == TRAINING_SESSION_METADATA_VERSION + 1
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
async def test_issue_364_get_tokenizer_uses_detached_store_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-tokenizer"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": TRAINING_SESSION_METADATA_VERSION,
            "materialization_state": MATERIALIZATION_STATE_READY,
            "tokenizer_info": {
                "vocab_size": 151936,
                "model_max_length": 32768,
                "pad_token": "<|endoftext|>",
                "pad_token_id": 151645,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 151645,
                "bos_token": "<|im_start|>",
                "bos_token_id": 151643,
                "unk_token": None,
                "unk_token_id": None,
            },
        }

    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )

    out = await training_route.get_tokenizer("model-364-tokenizer")

    assert out["tokenizer"] == {
        "vocab_size": 151936,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151645,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 151645,
        "bos_token": "<|im_start|>",
        "bos_token_id": 151643,
        "unk_token": None,
        "unk_token_id": None,
    }


@pytest.mark.anyio
async def test_issue_364_get_tokenizer_backfills_detached_store_without_training_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    persisted: dict[str, object] = {}

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-tokenizer-backfill"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": 1,
        }

    def _fake_build_local_tokenizer_metadata(base_model: str, backend: str) -> dict[str, object]:
        assert base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert backend == "megatron"
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3",
            "tokenizer_identity": "/hf/snapshots/qwen3#abc123",
            "tokenizer_info": {
                "vocab_size": 151936,
                "model_max_length": 32768,
                "pad_token": "<|endoftext|>",
                "pad_token_id": 151645,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 151645,
                "bos_token": "<|im_start|>",
                "bos_token_id": 151643,
                "unk_token": None,
                "unk_token_id": None,
            },
        }

    async def _async_upsert_training_session(info: dict[str, object]) -> None:
        persisted.update(info)

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "_build_local_tokenizer_metadata", _fake_build_local_tokenizer_metadata)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )

    out = await training_route.get_tokenizer("model-364-tokenizer-backfill")

    assert out["tokenizer"]["vocab_size"] == 151936
    assert persisted["tokenizer_identity"] == "/hf/snapshots/qwen3#abc123"
    assert persisted["tokenizer_source_path"] == "/hf/snapshots/qwen3"
    assert persisted["tokenizer_info"] == {
        "vocab_size": 151936,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151645,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 151645,
        "bos_token": "<|im_start|>",
        "bos_token_id": 151643,
        "unk_token": None,
        "unk_token_id": None,
    }
    assert persisted["metadata_version"] == TRAINING_SESSION_METADATA_VERSION


def test_issue_364_restore_training_session_info_defaults_new_materialization_metadata() -> None:
    manager = TrainingSessionManager()
    session = manager.restore_training_session_info(
        {
            "model_id": "model-364-state",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 1,
        }
    )

    assert session is not None
    assert session.materialization_state == MATERIALIZATION_STATE_READY
    assert session.tokenizer_info is None
    assert session.tokenizer_identity is None
    assert session.tokenizer_source_path is None


def test_issue_364_pending_local_session_stays_hidden_until_persisted() -> None:
    manager = TrainingSessionManager()
    manager.create_session(
        model_id="model-364-pending",
        session_id="session-364",
        model_seq_id=1,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    assert manager.get_session("model-364-pending") is None
    assert manager.get_local_session("model-364-pending") is not None


def test_issue_364_restore_training_session_updates_actor_binding_without_version_bump() -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-actor",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 7,
            "actor_name": "old-actor",
            "namespace": "old-ns",
        }
    )

    session = manager.restore_training_session_info(
        {
            "model_id": "model-364-actor",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 7,
            "actor_name": "new-actor",
            "namespace": "new-ns",
        }
    )

    assert session is not None
    assert session.actor_name == "new-actor"
    assert session.namespace == "new-ns"


def test_issue_364_refresh_training_session_drops_stale_worker_binding_when_actor_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-binding",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 7,
            "actor_name": "old-actor",
            "namespace": "old-ns",
        }
    )
    engine = SimpleNamespace(
        _workers={"model-364-binding": object()},
        _resource_pool_actor_names={"model-364-binding": "old-actor"},
    )

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)

    training_route._refresh_training_session_from_info_if_needed(
        "model-364-binding",
        {
            "model_id": "model-364-binding",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 7,
            "actor_name": "new-actor",
            "namespace": "new-ns",
        },
    )

    assert "model-364-binding" not in engine._workers
    assert engine._resource_pool_actor_names["model-364-binding"] == "new-actor"
    assert manager.get_local_session("model-364-binding").actor_name == "new-actor"


@pytest.mark.anyio
async def test_issue_364_cleanup_inactive_skips_session_when_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager(inactivity_timeout=1)
    manager.restore_training_session_info(
        {
            "model_id": "model-364-cleanup-store-down",
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

    async def _raise_store():
        raise RuntimeError("store unavailable")

    async def _cleanup_session(model_id: str):
        cleaned.append(model_id)

    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_list_training_sessions",
        _raise_store,
    )
    monkeypatch.setattr(manager, "_cleanup_session", _cleanup_session)

    await manager._cleanup_inactive()

    assert cleaned == []


@pytest.mark.anyio
async def test_issue_364_cleanup_inactive_restores_detached_sessions_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager(inactivity_timeout=1)
    cleaned: list[str] = []

    async def _async_list_training_sessions():
        return [
            {
                "model_id": "model-364-cleanup-restored",
                "session_id": "session-364",
                "model_seq_id": 1,
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "backend": "peft",
                "current_step": 2,
                "metadata_version": 3,
                "last_activity": 0.0,
            }
        ]

    async def _cleanup_session(model_id: str):
        cleaned.append(model_id)

    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_list_training_sessions",
        _async_list_training_sessions,
    )
    monkeypatch.setattr(manager, "_cleanup_session", _cleanup_session)

    await manager._cleanup_inactive()

    assert cleaned == ["model-364-cleanup-restored"]


@pytest.mark.anyio
async def test_issue_364_get_tokenizer_falls_back_to_local_metadata_when_worker_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "model-364-tokenizer-fallback",
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": TRAINING_SESSION_METADATA_VERSION,
            "actor_name": "actor-364",
            "namespace": "tinker",
        }
    )

    async def _async_get_training_session_info(model_id: str):
        assert model_id == "model-364-tokenizer-fallback"
        return {
            "model_id": model_id,
            "session_id": "session-364",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "current_step": 7,
            "metadata_version": TRAINING_SESSION_METADATA_VERSION,
            "actor_name": "actor-364",
            "namespace": "tinker",
        }

    async def _broken_get_tokenizer_info(_session):
        raise RuntimeError("worker missing")

    def _fake_build_local_tokenizer_metadata(base_model: str, backend: str) -> dict[str, object]:
        assert base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert backend == "megatron"
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3",
            "tokenizer_identity": "/hf/snapshots/qwen3#def456",
            "tokenizer_info": {
                "vocab_size": 151936,
                "model_max_length": 32768,
                "pad_token": "<|endoftext|>",
                "pad_token_id": 151645,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 151645,
                "bos_token": "<|im_start|>",
                "bos_token_id": 151643,
                "unk_token": None,
                "unk_token_id": None,
            },
        }

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        training_route,
        "training_engine",
        SimpleNamespace(_workers={}, _resource_pool_actor_names={}, get_tokenizer_info=_broken_get_tokenizer_info),
    )
    async def _restore_none(_model_id: str):
        return None

    async def _async_upsert_training_session(_info: dict[str, object]) -> None:
        return None

    monkeypatch.setattr(training_route, "_build_local_tokenizer_metadata", _fake_build_local_tokenizer_metadata)
    monkeypatch.setattr(training_route, "_restore_training_session", _restore_none)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_get_training_session_info",
        _async_get_training_session_info,
    )
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )

    out = await training_route.get_tokenizer("model-364-tokenizer-fallback")

    assert out["tokenizer"] == {
        "vocab_size": 151936,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151645,
        "eos_token": "<|endoftext|>",
        "eos_token_id": 151645,
        "bos_token": "<|im_start|>",
        "bos_token_id": 151643,
        "unk_token": None,
        "unk_token_id": None,
    }
