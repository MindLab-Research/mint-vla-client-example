from tinker_server.backend.session_heartbeat_store import SessionHeartbeatStore
from tinker_server.backend.session_manager import SessionManager


def test_issue_362_session_heartbeat_store_prunes_old_entries_on_update() -> None:
    store = SessionHeartbeatStore()
    store._max_age_s = 50.0
    store._prune_every = 2

    store.update("old", now=100.0)
    store.update("fresh", now=200.0)

    assert store.last_seen("old") is None
    assert store.last_seen("fresh") == 200.0


def test_issue_362_session_heartbeat_store_manual_prune_removes_stale_entries() -> None:
    store = SessionHeartbeatStore()
    store.update("keep", now=100.0)
    store.update("drop", now=10.0)

    with store._lock:
        removed = store._prune_locked(now=120.0, max_age_s=50.0)

    assert removed == 1
    assert store.last_seen("keep") == 100.0
    assert store.last_seen("drop") is None


def test_issue_362_observability_snapshot_excludes_base_model_sessions_from_lora_loaded() -> None:
    manager = SessionManager()
    manager.register_base_model_session("base", "Qwen/Qwen3-4B-Instruct-2507")
    manager.register_multi_lora_session(
        session_id="lora",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=32,
        adapter_path=None,
        lora_loaded=True,
    )

    snapshot = manager.observability_snapshot()

    assert snapshot["sampling_sessions_total"] == 2
    assert snapshot["sampling_sessions_base_model"] == 1
    assert snapshot["sampling_sessions_multi_lora"] == 2
    assert snapshot["sampling_sessions_lora_loaded"] == 1
