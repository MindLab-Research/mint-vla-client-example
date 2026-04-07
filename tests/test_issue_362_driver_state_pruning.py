import sys
from types import SimpleNamespace

from tinker_server.backend.session_heartbeat_store import SessionHeartbeatStore
from tinker_server.backend.session_manager import SessionManager


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, **kwargs):
        return self._fn(**kwargs)


class _FakeHeartbeatActor:
    def __init__(self) -> None:
        self._last_seen: dict[str, float] = {}
        self.update = _Remote(self._update)
        self.last_seen = _Remote(self._last_seen_value)
        self.prune = _Remote(self._prune)

    def _update(self, session_id: str, now: float | None = None) -> None:
        if not session_id:
            return
        self._last_seen[str(session_id)] = float(now) if now is not None else 0.0

    def _last_seen_value(self, session_id: str) -> float | None:
        return self._last_seen.get(str(session_id))

    def _prune(self, max_age_s: float) -> int:
        now = max(self._last_seen.values(), default=0.0)
        stale = [sid for sid, ts in self._last_seen.items() if (now - ts) > float(max_age_s)]
        for sid in stale:
            del self._last_seen[sid]
        return len(stale)


def test_issue_362_session_heartbeat_store_prunes_old_entries_on_update(monkeypatch) -> None:
    store = SessionHeartbeatStore()
    actor = _FakeHeartbeatActor()
    monkeypatch.setattr(store, "_get_actor", lambda: actor)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))

    store.update("old", now=100.0)
    store.update("fresh", now=200.0)
    removed = store.prune(50.0)

    assert removed == 1
    assert store.last_seen("old") is None
    assert store.last_seen("fresh") == 200.0


def test_issue_362_session_heartbeat_store_manual_prune_removes_stale_entries(monkeypatch) -> None:
    store = SessionHeartbeatStore()
    actor = _FakeHeartbeatActor()
    monkeypatch.setattr(store, "_get_actor", lambda: actor)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))

    store.update("keep", now=100.0)
    store.update("drop", now=10.0)
    removed = store.prune(50.0)

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
