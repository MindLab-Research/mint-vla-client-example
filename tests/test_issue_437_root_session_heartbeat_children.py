from __future__ import annotations

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
async def test_issue_437_root_heartbeat_derives_training_checkpoint_children_only(monkeypatch) -> None:
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
    monkeypatch.setattr(
        sis,
        "get_sampler_index",
        lambda sampler_id: {
            "child-sampler": {"source_type": "checkpoint", "model_id": "train-a"},
            "other-checkpoint": {"source_type": "checkpoint", "model_id": "train-b"},
            "base-model": {"source_type": "base_model"},
        }.get(sampler_id),
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
