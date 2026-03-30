from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_364_queue_supervisor_claim_and_heartbeat(monkeypatch) -> None:
    import tinker_server.backend.queue_supervisor as qs

    class _FakeActor:
        class _ClaimRemote:
            def remote(self, **_kwargs):
                return {
                    "generation_id": 1,
                    "owner_id": qs.queue_supervisor.owner_id(),
                    "state": "starting",
                }

        class _HeartbeatRemote:
            def remote(self, **_kwargs):
                return True

        class _BeginRemote:
            def remote(self, **_kwargs):
                return True

        class _FinishRemote:
            def remote(self, **_kwargs):
                return True

        class _CurrentRemote:
            def remote(self, **_kwargs):
                return True

        class _FencedRemote:
            def remote(self, **_kwargs):
                return None

        class _SnapshotRemote:
            def remote(self):
                return {"generation_id": 1, "state": "active"}

        @property
        def claim_generation(self):
            return self._ClaimRemote()

        @property
        def heartbeat(self):
            return self._HeartbeatRemote()

        @property
        def begin_reconcile(self):
            return self._BeginRemote()

        @property
        def finish_reconcile(self):
            return self._FinishRemote()

        @property
        def is_generation_current(self):
            return self._CurrentRemote()

        @property
        def record_fenced_worker(self):
            return self._FencedRemote()

        @property
        def snapshot(self):
            return self._SnapshotRemote()

    async def _identity(ref):
        return ref

    monkeypatch.setattr(qs.queue_supervisor, "_get_ray_actor", lambda: _FakeActor())
    monkeypatch.setattr(qs, "_await_ray_ref", _identity)

    claimed = await qs.queue_supervisor.async_claim_generation()
    assert claimed["generation_id"] == 1
    assert claimed["owner_id"] == qs.queue_supervisor.owner_id()
    assert await qs.queue_supervisor.async_heartbeat(generation_id=1) is True
    assert await qs.queue_supervisor.async_is_generation_current(generation_id=1) is True


def test_issue_364_future_store_rejects_stale_generation(monkeypatch) -> None:
    future_store_module = importlib.import_module("tinker_server.backend.future_store")

    calls: list[tuple[str, str]] = []

    class _FakeActor:
        class _FailRemote:
            def remote(self, *, request_id: str, error: str):
                calls.append((request_id, error))

        class _GetMetaRemote:
            def remote(self, *, request_id: str):
                return {"request_id": request_id}

        @property
        def fail(self):
            return self._FailRemote()

        @property
        def get_meta(self):
            return self._GetMetaRemote()

        class _ResolveRefRemote:
            def remote(self, **_kwargs):
                raise AssertionError("resolve_ref should not run for stale generation")

        @property
        def resolve_ref(self):
            return self._ResolveRefRemote()

    class _FakeQueueSupervisor:
        def is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            assert generation_id == 7
            return False

        async def async_is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            assert generation_id == 7
            return False

    monkeypatch.setattr(future_store_module.future_store, "_get_ray_actor", lambda: _FakeActor())
    queue_supervisor_module = importlib.import_module("tinker_server.backend.queue_supervisor")
    monkeypatch.setattr(queue_supervisor_module, "queue_supervisor", _FakeQueueSupervisor())
    monkeypatch.setattr(future_store_module, "get_current_queue_generation_id", lambda: 7)

    future_store_module.future_store.resolve("rid-1", {"ok": True})

    assert calls == [("rid-1", "stale generation finalize rejected (generation_id=7)")]
