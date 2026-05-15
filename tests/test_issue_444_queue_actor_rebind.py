from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.mark.anyio
async def test_issue_444_api_work_queue_rebinds_active_job_id_when_actor_recreated(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    client = api_work_queue_module.ApiWorkQueueClient()
    client._consumer_job_id = "consumer-123"

    calls: list[tuple[str, str]] = []

    class _FakeActor:
        class _StatsRemote:
            def remote(self):
                return {
                    "ok": True,
                    "code_identity": api_work_queue_module.CURRENT_CODE_IDENTITY,
                    "runtime_contract_digest": api_work_queue_module._api_work_queue_runtime_contract_digest(),
                }

        class _DebugStateRemote:
            def remote(self):
                return {"active_job_id": None}

        class _SetActiveRemote:
            def remote(self, consumer_job_id: str):
                calls.append(("set_active_job_id", consumer_job_id))
                return True

        @property
        def stats(self):
            return self._StatsRemote()

        @property
        def debug_state(self):
            return self._DebugStateRemote()

        @property
        def set_active_job_id(self):
            return self._SetActiveRemote()

    fake_actor = _FakeActor()

    fake_ray = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(
            GetTimeoutError=type("GetTimeoutError", (Exception,), {}),
            ActorDiedError=type("ActorDiedError", (Exception,), {}),
            RayActorError=type("RayActorError", (Exception,), {}),
        ),
        is_initialized=lambda: True,
        get_actor=lambda name, namespace=None: fake_actor,
    )

    async def _await_ref(ref, *, timeout_s: float | None = None):
        _ = timeout_s
        return ref

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(client, "_await_ray_ref", _await_ref)
    client._ray_actor = fake_actor

    actor = await client._get_ray_actor_async()

    assert actor is fake_actor
    assert calls == [("set_active_job_id", "consumer-123")]


@pytest.mark.anyio
async def test_issue_444_api_work_queue_enqueue_reacquires_actor(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    client = api_work_queue_module.ApiWorkQueueClient()
    stale_actor = object()
    client._ray_actor = stale_actor

    enqueue_calls: list[tuple[dict[str, object], str | None]] = []

    class _FreshActor:
        class _EnqueueRemote:
            def remote(self, item: dict[str, object], producer_job_id: str | None):
                enqueue_calls.append((item, producer_job_id))
                return {"ok": True}

        @property
        def enqueue(self):
            return self._EnqueueRemote()

    fresh_actor = _FreshActor()
    reacquire_calls: list[str] = []

    async def _get_ray_actor_async():
        reacquire_calls.append("reacquired")
        return fresh_actor

    async def _await_ref(ref, *, timeout_s: float | None = None):
        _ = timeout_s
        return ref

    fake_ray = types.SimpleNamespace(
        get_runtime_context=lambda: types.SimpleNamespace(get_job_id=lambda: "producer-123")
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(client, "_get_ray_actor_async", _get_ray_actor_async)
    monkeypatch.setattr(client, "_await_ray_ref", _await_ref)
    monkeypatch.setattr(api_work_queue_module, "get_otel_tracer", lambda: None)

    await client.enqueue(
        request_id="rid-1",
        op="mint.action.act",
        request_json=b"{}",
        user_id=None,
        webhook_url=None,
    )

    assert reacquire_calls == ["reacquired"]
    assert len(enqueue_calls) == 1
    item, producer_job_id = enqueue_calls[0]
    assert item["request_id"] == "rid-1"
    assert item["op"] == "mint.action.act"
    assert isinstance(producer_job_id, str) and producer_job_id


@pytest.mark.anyio
async def test_issue_444_future_store_async_create_reacquires_actor(monkeypatch) -> None:
    future_store_module = importlib.import_module("tinker_server.backend.future_store")

    store = future_store_module.FutureStore()
    store._ray_actor = object()
    add_pending_calls: list[str] = []

    class _FreshActor:
        class _AddPendingRemote:
            def remote(self, request_id: str):
                add_pending_calls.append(request_id)
                return None

        @property
        def add_pending(self):
            return self._AddPendingRemote()

    fresh_actor = _FreshActor()
    reacquire_calls: list[str] = []

    async def _get_ray_actor_async():
        reacquire_calls.append("reacquired")
        return fresh_actor

    async def _await_ref(ref):
        return ref

    monkeypatch.setattr(store, "_get_ray_actor_async", _get_ray_actor_async)
    monkeypatch.setattr(future_store_module, "_await_ray_ref", _await_ref)

    request_id = await store.async_create_with_id("rid-fs-1")

    assert request_id == "rid-fs-1"
    assert reacquire_calls == ["reacquired"]
    assert add_pending_calls == ["rid-fs-1"]


def test_issue_444_queue_actor_name_prefers_env_overrides(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    monkeypatch.setattr(api_work_queue_module.server_config, "api_work_queue_actor_name", "from-config")
    monkeypatch.delenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    monkeypatch.delenv("MINT_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-config"

    monkeypatch.setenv("MINT_API_WORK_QUEUE_ACTOR_NAME", "from-mint-env")
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-mint-env"

    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "from-tinker-env")
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-tinker-env"


def test_issue_444_queue_actor_resources_prefers_pinned_node_ip(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    class _RayStub:
        @staticmethod
        def cluster_resources():
            return {"node:__internal_head__": 1.0}

    monkeypatch.setitem(sys.modules, "ray", _RayStub)
    monkeypatch.delenv("MINT_API_WORK_QUEUE_PINNED_NODE_IP", raising=False)
    assert api_work_queue_module._api_work_queue_actor_resources() == {"node:__internal_head__": 0.001}

    monkeypatch.setenv("MINT_API_WORK_QUEUE_PINNED_NODE_IP", "192.168.38.176")
    assert api_work_queue_module._api_work_queue_actor_resources() == {"node:192.168.38.176": 0.001}
