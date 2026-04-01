from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data))


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_touches_explicit_heartbeat_children(monkeypatch) -> None:
    from tinker_server.models.types import SessionHeartbeatRequest
    from tinker_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    async def _fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    monkeypatch.setattr(service, "run_in_threadpool", _fake_run_in_threadpool)

    import tinker_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "get_session_index",
        lambda session_id: {
            "session_id": session_id,
            "user_id": "owner-a",
            "heartbeat_sampler_ids": ["sampler-a", "sampler-b", "sampler-a", "", None],
        },
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
async def test_issue_437_root_heartbeat_skips_children_without_explicit_heartbeat_ids(monkeypatch) -> None:
    from tinker_server.models.types import SessionHeartbeatRequest
    from tinker_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    async def _fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    monkeypatch.setattr(service, "run_in_threadpool", _fake_run_in_threadpool)

    import tinker_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "get_session_index",
        lambda session_id: {
            "session_id": session_id,
            "user_id": "owner-a",
            "training_run_ids": ["train-a"],
            "sampler_ids": ["child-sampler", "other-checkpoint", "base-model"],
        },
    )

    resp = await service.session_heartbeat(
        SessionHeartbeatRequest(session_id="root-session"),
        _dummy_request("owner-a"),
    )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [("root-session", 0)]


@pytest.mark.anyio
async def test_issue_437_root_heartbeat_ignores_missing_child_samplers(monkeypatch) -> None:
    from tinker_server.models.types import SessionHeartbeatRequest
    from tinker_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            if session_id == "stale-sampler":
                return
            touched.append((session_id, delta))

    async def _fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    monkeypatch.setattr(service, "run_in_threadpool", _fake_run_in_threadpool)

    import tinker_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "get_session_index",
        lambda session_id: {
            "session_id": session_id,
            "user_id": "owner-a",
            "heartbeat_sampler_ids": ["live-sampler", "stale-sampler"],
        },
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
    from tinker_server.models.types import SessionHeartbeatRequest
    from tinker_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    async def _fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    monkeypatch.setattr(service, "run_in_threadpool", _fake_run_in_threadpool)

    import tinker_server.backend.session_index_store as sis

    monkeypatch.setattr(
        sis,
        "get_session_index",
        lambda session_id: {
            "session_id": session_id,
            "user_id": "owner-a",
            "heartbeat_sampler_ids": ["sampler-a"],
        },
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
    from tinker_server.models.types import SessionHeartbeatRequest
    from tinker_server.routes import service

    touched: list[tuple[str, int]] = []
    updates: list[str] = []

    class _SessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    async def _fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(service, "session_manager", _SessionManager())
    monkeypatch.setattr(service, "session_heartbeat_store", SimpleNamespace(update=updates.append))
    monkeypatch.setattr(service, "run_in_threadpool", _fake_run_in_threadpool)

    import tinker_server.backend.session_index_store as sis

    def _boom(_session_id: str):
        raise RuntimeError("session index offline")

    monkeypatch.setattr(sis, "get_session_index", _boom)

    with caplog.at_level("WARNING"):
        resp = await service.session_heartbeat(
            SessionHeartbeatRequest(session_id="root-session"),
            _dummy_request("owner-a"),
        )

    assert resp.type == "session_heartbeat"
    assert updates == ["root-session"]
    assert touched == [("root-session", 0)]
    assert "session index lookup failed for root-session" in caplog.text


def test_issue_437_add_heartbeat_sampler_compat_upserts_when_actor_lacks_method(monkeypatch, caplog) -> None:
    import tinker_server.backend.session_index_store as sis

    sampler_adds: list[tuple[str, str, str | None, str | None]] = []
    upserts: list[tuple[str, dict]] = []

    actor = SimpleNamespace(
        add_sampler=SimpleNamespace(
            remote=lambda session_id, sampler_id, user_id, created_at: sampler_adds.append(
                (session_id, sampler_id, user_id, created_at)
            )
        ),
        get_session=SimpleNamespace(
            remote=lambda _session_id: {
                "session_id": "root-session",
                "heartbeat_sampler_ids": ["sampler-old"],
            }
        ),
        upsert_session=SimpleNamespace(remote=lambda session_id, info: upserts.append((session_id, info))),
    )

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True,
            get=lambda value: value,
        ),
    )
    monkeypatch.setattr(sis, "_get_or_create_actor", lambda: actor)

    with caplog.at_level("WARNING"):
        sis.add_heartbeat_sampler_to_session(
            session_id="root-session",
            sampler_id="sampler-new",
            user_id="owner-a",
            created_at="2026-04-01T00:00:00",
        )

    assert sampler_adds == [
        ("root-session", "sampler-new", "owner-a", "2026-04-01T00:00:00"),
    ]
    assert upserts == [
        (
            "root-session",
            {
                "session_id": "root-session",
                "heartbeat_sampler_ids": ["sampler-old", "sampler-new"],
            },
        )
    ]
    assert "actor missing add_heartbeat_sampler; using compatibility upsert" in caplog.text
