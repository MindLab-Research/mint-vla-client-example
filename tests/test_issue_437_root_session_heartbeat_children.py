from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data))


def _stub_sampling_last_activity(monkeypatch) -> None:
    async def _noop_async_set_last_activity(_session_id: str, _last_activity: float):
        return None

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_set_sampling_session_last_activity",
        _noop_async_set_last_activity,
    )


def _async_callable(fn):
    async def _inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return _inner


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_touches_explicit_heartbeat_children(monkeypatch) -> None:
    from mint_server.models.types import SessionHeartbeatRequest
    from mint_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    _stub_sampling_last_activity(monkeypatch)

    import mint_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "async_get_session_index",
        _async_callable(
            lambda session_id: {
                "session_id": session_id,
                "user_id": "owner-a",
                "heartbeat_sampler_ids": ["sampler-a", "sampler-b", "sampler-a", "", None],
            }
        ),
    )

    resp = await service.session_heartbeat(
        SessionHeartbeatRequest(session_id="root-session"),
        _dummy_request("owner-a"),
    )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [
        ("root-session", 0),
        ("sampler-a", 0),
        ("sampler-b", 0),
    ]


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_derives_training_checkpoint_children_only(monkeypatch) -> None:
    from mint_server.models.types import SessionHeartbeatRequest
    from mint_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    _stub_sampling_last_activity(monkeypatch)

    import mint_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "async_get_session_index",
        _async_callable(
            lambda session_id: {
                "session_id": session_id,
                "user_id": "owner-a",
                "training_run_ids": ["train-a"],
                "sampler_ids": ["child-sampler", "other-checkpoint", "base-model"],
            }
        ),
    )
    monkeypatch.setattr(
        sis,
        "async_get_sampler_index",
        _async_callable(
            lambda sampler_id: {
                "child-sampler": {"source_type": "checkpoint", "model_id": "train-a"},
                "other-checkpoint": {"source_type": "checkpoint", "model_id": "train-b"},
                "base-model": {"source_type": "base_model"},
            }.get(sampler_id)
        ),
    )

    resp = await service.session_heartbeat(
        SessionHeartbeatRequest(session_id="root-session"),
        _dummy_request("owner-a"),
    )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [
        ("root-session", 0),
        ("child-sampler", 0),
    ]


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_ignores_missing_child_samplers(monkeypatch) -> None:
    from mint_server.models.types import SessionHeartbeatRequest
    from mint_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            if session_id == "stale-sampler":
                return
            touched.append((session_id, delta))

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    _stub_sampling_last_activity(monkeypatch)

    import mint_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "async_get_session_index",
        _async_callable(
            lambda session_id: {
                "session_id": session_id,
                "user_id": "owner-a",
                "heartbeat_sampler_ids": ["live-sampler", "stale-sampler"],
            }
        ),
    )

    resp = await service.session_heartbeat(
        SessionHeartbeatRequest(session_id="root-session"),
        _dummy_request("owner-a"),
    )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [
        ("root-session", 0),
        ("live-sampler", 0),
    ]


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_skips_child_fanout_for_owner_mismatch(monkeypatch, caplog) -> None:
    from mint_server.models.types import SessionHeartbeatRequest
    from mint_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    _stub_sampling_last_activity(monkeypatch)

    import mint_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "async_get_session_index",
        _async_callable(
            lambda session_id: {
                "session_id": session_id,
                "user_id": "owner-a",
                "heartbeat_sampler_ids": ["sampler-a"],
            }
        ),
    )

    with caplog.at_level("WARNING"):
        resp = await service.session_heartbeat(
            SessionHeartbeatRequest(session_id="root-session"),
            _dummy_request("other-user"),
        )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [("root-session", 0)]
    assert "child sampler propagation denied for root-session" in caplog.text


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_keeps_best_effort_on_index_failure(monkeypatch, caplog) -> None:
    from mint_server.models.types import SessionHeartbeatRequest
    from mint_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    _stub_sampling_last_activity(monkeypatch)

    import mint_server.backend.session_index_store as sis

    async def _boom(_session_id: str):
        raise RuntimeError("session index offline")

    monkeypatch.setattr(sis, "async_get_session_index", _boom)

    with caplog.at_level("WARNING"):
        resp = await service.session_heartbeat(
            SessionHeartbeatRequest(session_id="root-session"),
            _dummy_request("owner-a"),
        )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [("root-session", 0)]
    assert "session index lookup failed for root-session" in caplog.text


def test_issue_437_add_heartbeat_sampler_uses_task_state_store(monkeypatch) -> None:
    import mint_server.backend.session_index_store as sis

    calls: list[tuple[str, str, str | None, str | None]] = []
    monkeypatch.setattr(
        sis.task_state_store,
        "add_heartbeat_sampler_to_session_index",
        lambda *, session_id, sampler_id, user_id=None, created_at=None: calls.append(
            (session_id, sampler_id, user_id, created_at)
        ),
    )

    sis.add_heartbeat_sampler_to_session(
        session_id="root-session",
        sampler_id="sampler-new",
        user_id="owner-a",
        created_at="2026-04-01T00:00:00",
    )

    assert calls == [("root-session", "sampler-new", "owner-a", "2026-04-01T00:00:00")]
    assert not hasattr(sis, "_get_or_create_actor")
