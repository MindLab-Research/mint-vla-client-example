from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from mint_server.backend.contracts.control_plane_contracts import ConflictReason
from mint_server.backend.stores.task_payload_store import TaskPayloadStore
from mint_server.backend.stores.task_state_store import (
    TaskStateConflictError,
    TaskFutureService,
    TaskStateStore,
    TaskStateStoreClient,
    _TaskStateStoreActor,
    billing_observations_from_input,
    build_billing_observation,
)


def _create_task(store: TaskStateStore, request_id: str = "req-1") -> None:
    created = store.create_task(
        request_id=request_id,
        op="sampling.asample",
        domain_key="vllm:Qwen/Qwen3-4B-Instruct-2507",
        request_json=b'{"prompt": "hi"}',
        payload_hash="hash-1",
        metadata={"queue_kind": "model_work_scheduler"},
        now=100.0,
    )
    assert created.ok is True
    assert created.created is True


def test_duplicate_create_task_preserves_model_work_append_owner_marker() -> None:
    store = TaskStateStore.in_memory()
    try:
        first = store.create_task(
            request_id="append-owner",
            op="sampling.asample",
            domain_key="vllm:model-a",
            request_json=b'{"prompt":"first"}',
            metadata={
                "model_work_scheduler_append_attempt_id": "attempt-a",
                "stage": "first",
            },
        )
        second = store.create_task(
            request_id="append-owner",
            op="sampling.asample",
            domain_key="vllm:model-a",
            request_json=b'{"prompt":"second"}',
            metadata={
                "model_work_scheduler_append_attempt_id": "attempt-b",
                "stage": "duplicate",
            },
        )

        assert first.created is True
        assert second.created is False
        assert second.record["request_json"] == b'{"prompt":"first"}'
        assert second.record["metadata"]["model_work_scheduler_append_attempt_id"] == "attempt-a"
        assert second.record["metadata"]["stage"] == "first"
    finally:
        store.close()


def test_task_state_store_client_async_ensure_ready_can_create_actor(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as module
    import ray

    calls: dict[str, object] = {}

    class _PingRemote:
        def remote(self) -> dict[str, object]:
            return {"ok": True, "actor_name": "mint_task_state_store"}

    class _Actor:
        ping = _PingRemote()

    async def _fake_async_get_ray_ref(ref, *, timeout_s=10.0):
        calls["timeout_s"] = timeout_s
        return ref

    def _fake_create_ray_actor_handle():
        calls["created_handle"] = True
        return _Actor()

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_actor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(module, "_create_ray_actor_handle", _fake_create_ray_actor_handle)
    monkeypatch.setattr(module, "async_get_ray_ref", _fake_async_get_ray_ref)

    client = TaskStateStoreClient()
    out = asyncio.run(client.async_ensure_ready(timeout_s=7.0, create_if_missing=True))

    assert out == {"ok": True, "actor_name": "mint_task_state_store"}
    assert calls == {"created_handle": True, "timeout_s": 7.0}


def test_task_state_store_ray_actor_ping_uses_health_concurrency_group(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as module
    import ray

    captured: dict[str, object] = {}

    class _OptionsProxy:
        def __init__(self, cls):
            self._cls = cls

        def remote(self, db_path: str):
            captured["db_path"] = db_path
            actor = self._cls(db_path)
            ping = actor.ping

            class _RemoteMethod:
                def remote(self):
                    return ping()

            actor.ping = _RemoteMethod()
            return actor

    class _RemoteClass:
        def __init__(self, cls):
            self._cls = cls

        def options(self, **options):
            captured["options"] = options
            return _OptionsProxy(self._cls)

    def _fake_remote(**remote_kwargs):
        captured["remote_kwargs"] = remote_kwargs

        def _decorator(cls):
            captured["actor_cls"] = cls
            return _RemoteClass(cls)

        return _decorator

    def _fake_method(**method_kwargs):
        captured["method_kwargs"] = method_kwargs

        def _decorator(fn):
            captured["method_name"] = fn.__name__
            return fn

        return _decorator

    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(ray, "method", _fake_method)
    def _fake_actor_runtime_env(**kwargs):
        captured["runtime_env_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(module, "actor_runtime_env", _fake_actor_runtime_env)
    monkeypatch.setattr(module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "sync_get_ray_ref", lambda ref, *, timeout_s=None: ref)
    monkeypatch.setattr(module, "_task_state_store_db_path", lambda: ":memory:")

    module._create_ray_actor(require_ready=True)

    assert captured["remote_kwargs"]["concurrency_groups"] == {"health": 8}
    assert captured["runtime_env_kwargs"]["include_ray_attach_hints"] is False
    assert captured["method_kwargs"] == {"concurrency_group": "health"}
    assert captured["method_name"] == "ping"


def test_training_session_inflight_is_durable_metadata(tmp_path: Path) -> None:
    store = TaskStateStore(str(tmp_path / "task-state-training-inflight.sqlite3"))
    try:
        store.upsert_training_session(
            model_id="model-inflight",
            info={
                "model_id": "model-inflight",
                "session_id": "session-inflight",
                "base_model": "Qwen/Qwen3-0.6B",
                "metadata_version": 1,
                "last_activity": 10.0,
            },
        )

        assert store.mark_training_session_inflight(model_id="model-inflight", delta=1) == 1
        info = store.get_training_session(model_id="model-inflight")
        assert info is not None
        assert info["inflight_ops"] == 1
        assert info["last_activity"] >= 10.0

        assert store.mark_training_session_inflight(model_id="model-inflight", delta=-3) == 0
        info = store.get_training_session(model_id="model-inflight")
        assert info is not None
        assert info["inflight_ops"] == 0
        assert store.mark_training_session_inflight(model_id="missing", delta=1) is None
    finally:
        store.close()


def test_task_state_store_client_async_cached_actor_survives_concurrent_reset(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as module
    import ray

    class _PingRemote:
        def remote(self) -> dict[str, object]:
            return {"ok": True}

    class _Actor:
        ping = _PingRemote()

    async def _fake_async_get_ray_ref(ref, *, timeout_s=10.0):
        client._reset_ray_actor()
        return ref

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(module, "async_get_ray_ref", _fake_async_get_ray_ref)

    client = TaskStateStoreClient()
    actor = _Actor()
    client._ray_actor = actor

    out = asyncio.run(client._get_ray_actor_async())

    assert out is actor
    assert client._ray_actor is None


def test_task_state_store_client_sync_cached_actor_survives_reset(monkeypatch) -> None:
    import ray

    class _Actor:
        pass

    client = TaskStateStoreClient()
    actor = _Actor()

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda *args, **kwargs: (client._reset_ray_actor(), actor)[1],
    )

    out = client._get_ray_actor_sync(require_ready=False, create_if_missing=False)

    assert out is actor
    assert client._ray_actor is actor


def test_task_state_store_actor_wait_status_change_notifies(tmp_path) -> None:
    async def _run() -> None:
        actor = _TaskStateStoreActor(str(tmp_path / "task-state-watch.sqlite3"))
        try:
            actor.create_task(
                request_id="req-watch",
                op="sampling.asample",
                domain_key="vllm:test",
                request_json=b"{}",
                now=1.0,
            )

            waiter = asyncio.create_task(
                asyncio.to_thread(
                    actor.wait_task_status_change,
                    request_id="req-watch",
                    timeout_s=1.0,
                )
            )
            await asyncio.sleep(0.01)
            actor.update_task_metadata(
                request_id="req-watch",
                metadata={"queue_state": "running"},
                status="running",
                now=2.0,
            )

            out = await waiter
            assert out["changed"] is True
            assert out["record"]["request_id"] == "req-watch"
            assert out["record"]["status"] == "running"
            assert out["record"]["metadata"]["queue_state"] == "running"
        finally:
            actor.close()

    asyncio.run(_run())


def test_task_state_store_actor_future_store_init_is_singleton_under_concurrency(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.backend.stores.future_state_store as future_state_store_module

    actor = _TaskStateStoreActor(str(tmp_path / "task-state-concurrency.sqlite3"))
    created: list[object] = []

    class _FakeFutureStateStore:
        def __init__(self, path: str) -> None:
            self.db_path = path
            created.append(self)
            time.sleep(0.05)

        def close(self) -> None:
            return None

    monkeypatch.setattr(future_state_store_module, "FutureStateStore", _FakeFutureStateStore)

    barrier = threading.Barrier(3)
    results: list[object] = []

    def _worker() -> None:
        barrier.wait()
        results.append(actor._future_store_or_create())

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1.0)
        assert all(not thread.is_alive() for thread in threads)
        assert len(created) == 1
        assert len(results) == 2
        assert results[0] is results[1]
        assert results[0] is created[0]
    finally:
        actor.close()


def test_issue_638_task_state_store_stats_include_future_dashboard_fields(tmp_path) -> None:
    import mint_server.backend.stores.task_state_store as task_state_store_module

    task_state_store_module._FUTURE_TIMEOUT_METRICS.clear()
    task_state_store_module._FUTURE_TIMEOUT_METRICS.update(
        {"queue": 0.0, "execution": 0.0, "total": 0.0, "by_op": {}}
    )
    task_state_store_module._BILLING_FLUSH_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.update(
        {"total": 0.0, "error": 0.0, "inflight": 0.0, "by_method": {}}
    )
    task_state_store_module._TASK_STATE_STATS_METRICS.clear()
    task_state_store_module._TASK_STATE_STATS_METRICS.update(
        {
            "calls": 0.0,
            "cache_hits": 0.0,
            "total_duration_ms": 0.0,
            "last_duration_ms": 0.0,
            "max_duration_ms": 0.0,
        }
    )
    actor = _TaskStateStoreActor(str(tmp_path / "task-state-metrics.sqlite3"))
    try:
        actor.create_task(
            request_id="req-pending",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            now=100.0,
        )
        actor.create_task(
            request_id="req-done",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            now=90.0,
        )
        actor.complete_task_success(
            request_id="req-done",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 101.0},
            now=101.0,
        )

        stats = actor.stats()

        assert stats["pending"] == 1
        assert stats["results"] == 1
        assert stats["refs"] == 1
        assert stats["by_op"]["sampling.asample"]["pending"] == 1
        assert stats["by_op"]["sampling.asample"]["results"] == 1
        assert stats["age_stats"]["oldest_pending_s"] >= 0
        assert stats["payload_stats"]["result_refs_count"] == 1
        assert stats["timeout_counts"]["total"] == 0.0
        assert stats["task_state_rpc"]["total"] == 0.0
        assert stats["task_state_stats"]["calls"] >= 1.0
        assert stats["task_state_stats"]["last_duration_ms"] >= 0.0

        expired = actor.expire_active_tasks(older_than_s=10.0, now=200.0)
        assert expired == ["req-pending"]
        stats = actor.stats()
        assert stats["timeout_counts"]["execution"] == 1.0
        assert stats["timeout_counts"]["total"] == 1.0
        assert stats["timeout_counts"]["by_op"]["sampling.asample"]["execution"] == 1.0
        assert stats["timeout_counts"]["by_op"]["sampling.asample"]["total"] == 1.0

        actor.record_billing_metrics({"flush_success": 1, "event_inserted": 2, "event_conflict": 1})
        stats = actor.stats()
        assert stats["billing_outbox"]["metrics"]["flush_success"] == 1.0
        assert stats["billing_outbox"]["metrics"]["event_inserted"] == 2.0
        assert stats["billing_outbox"]["metrics"]["event_conflict"] == 1.0
    finally:
        actor.close()


def test_issue_638_task_state_store_registers_otel_future_and_billing_gauges(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.stores.task_state_store as task_state_store_module
    import mint_server.logging_context as logging_context

    task_state_store_module._FUTURE_TIMEOUT_METRICS.clear()
    task_state_store_module._FUTURE_TIMEOUT_METRICS.update(
        {"queue": 0.0, "execution": 0.0, "total": 0.0, "by_op": {}}
    )
    task_state_store_module._BILLING_FLUSH_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.update(
        {"total": 0.0, "error": 0.0, "inflight": 0.0, "by_method": {}}
    )
    task_state_store_module._TASK_STATE_STATS_METRICS.clear()
    task_state_store_module._TASK_STATE_STATS_METRICS.update(
        {
            "calls": 0.0,
            "cache_hits": 0.0,
            "total_duration_ms": 0.0,
            "last_duration_ms": 0.0,
            "max_duration_ms": 0.0,
        }
    )

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setenv("MINT_DEPLOYMENT_ENV", "prod")
    monkeypatch.setenv("MINT_CLUSTER_ID", "volcano")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    actor = _TaskStateStoreActor(str(tmp_path / "task-state-otel.sqlite3"))
    try:
        actor.create_task(
            request_id="req-pending",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            now=100.0,
        )
        actor.record_billing_metrics(
            {
                "flush_success": 1,
                "flush_transient_error": 2,
                "event_inserted": 3,
                "skipped_missing_billing_context": 4,
            }
        )

        assert "mint_task_futures_pending" in gauges
        assert "mint_task_futures_oldest_pending_s" in gauges
        assert "mint_task_futures_timeouts_total" in gauges
        assert "mint_billing_outbox_rows" in gauges
        assert "mint_billing_outbox_flush_attempts_total" in gauges
        assert "mint_billing_observation_skipped_total" in gauges
        assert "mint_task_state_store_rpc_inflight" in gauges
        assert "mint_task_state_store_rpc_total" in gauges
        assert "mint_task_state_store_rpc_last_duration_ms" in gauges
        assert "mint_task_state_store_stats_last_duration_ms" in gauges

        pending_obs = gauges["mint_task_futures_pending"][0](None)
        assert any(obs.value == 1.0 and obs.attributes.get("op") is None for obs in pending_obs)
        assert any(obs.value == 1.0 and obs.attributes.get("op") == "sampling.asample" for obs in pending_obs)
        for obs in pending_obs:
            assert obs.attributes["deployment.env"] == "prod"
            assert obs.attributes["mint.cluster_id"] == "volcano"
            assert "request_id" not in obs.attributes
        timeout_obs = gauges["mint_task_futures_timeouts_total"][0](None)
        assert any(obs.value == 0.0 and obs.attributes.get("kind") == "queue" for obs in timeout_obs)
        assert any(obs.value == 0.0 and obs.attributes.get("kind") == "execution" for obs in timeout_obs)
        assert all("request_id" not in obs.attributes for obs in timeout_obs)
        billing_flush_obs = gauges["mint_billing_outbox_flush_attempts_total"][0](None)
        assert any(obs.value == 1.0 and obs.attributes.get("result") == "success" for obs in billing_flush_obs)
        assert any(obs.value == 2.0 and obs.attributes.get("result") == "transient_error" for obs in billing_flush_obs)
        billing_event_obs = gauges["mint_billing_outbox_events_total"][0](None)
        assert any(obs.value == 3.0 and obs.attributes.get("result") == "inserted" for obs in billing_event_obs)
        skipped_obs = gauges["mint_billing_observation_skipped_total"][0](None)
        assert skipped_obs[0].value == 4.0
        assert skipped_obs[0].attributes.get("reason") == "missing_billing_context"
        for observations in (billing_flush_obs, billing_event_obs, skipped_obs):
            for obs in observations:
                assert "request_id" not in obs.attributes
        stats_obs = gauges["mint_task_state_store_stats_last_duration_ms"][0](None)
        assert stats_obs[0].value >= 0.0
    finally:
        actor.close()


def test_task_state_store_client_rpc_metrics_record_success_and_failure(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as task_state_store_module

    task_state_store_module._TASK_STATE_RPC_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.update(
        {"total": 0.0, "error": 0.0, "inflight": 0.0, "by_method": {}}
    )

    class _Remote:
        def __init__(self, value=None, error: Exception | None = None) -> None:
            self.value = value
            self.error = error

        def remote(self, **kwargs):
            if self.error is not None:
                raise self.error
            return self.value

    class _Actor:
        future_get_task = _Remote({"request_id": "req-1"})
        future_mark_task_retrieved = _Remote(error=RuntimeError("boom"))

    async def _fake_get_ray_ref(ref, *, timeout_s=10.0):
        return ref

    monkeypatch.setattr(task_state_store_module, "async_get_ray_ref", _fake_get_ray_ref)
    client = TaskStateStoreClient()

    async def _fake_get_actor(*args, **kwargs):
        return _Actor()

    client._get_ray_actor_async = _fake_get_actor

    assert asyncio.run(client._call("future_get_task", request_id="req-1")) == {"request_id": "req-1"}
    with pytest.raises(RuntimeError):
        asyncio.run(client._call("future_mark_task_retrieved", request_id="req-1"))

    metrics = task_state_store_module.task_state_rpc_metrics_snapshot()
    assert metrics["total"] == 2.0
    assert metrics["error"] == 1.0
    assert metrics["inflight"] == 0.0
    assert metrics["by_method"]["future_get_task"]["total"] == 1.0
    assert metrics["by_method"]["future_mark_task_retrieved"]["error"] == 1.0


def test_task_state_store_client_stats_exposes_client_process_rpc_metrics(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as task_state_store_module

    task_state_store_module._TASK_STATE_RPC_METRICS.clear()
    task_state_store_module._TASK_STATE_RPC_METRICS.update(
        {
            "total": 7.0,
            "error": 1.0,
            "inflight": 0.0,
            "by_method": {
                "future_get_task": {
                    "total": 7.0,
                    "error": 1.0,
                    "total_duration_ms": 70.0,
                    "last_duration_ms": 10.0,
                    "max_duration_ms": 20.0,
                }
            },
        }
    )

    class _Remote:
        def remote(self, **kwargs):
            return {
                "pending": 0,
                "task_state_rpc": {"total": 0.0, "error": 0.0, "inflight": 0.0, "by_method": {}},
            }

    class _Actor:
        stats = _Remote()

    async def _fake_get_ray_ref(ref, *, timeout_s=10.0):
        return ref

    monkeypatch.setattr(task_state_store_module, "async_get_ray_ref", _fake_get_ray_ref)
    client = TaskStateStoreClient()

    async def _fake_get_actor(*args, **kwargs):
        return _Actor()

    client._get_ray_actor_async = _fake_get_actor

    out = asyncio.run(client.async_stats())

    assert out["pending"] == 0
    assert out["task_state_rpc"]["total"] == 8.0
    assert out["task_state_rpc"]["error"] == 1.0
    assert out["task_state_rpc"]["by_method"]["stats"]["total"] == 1.0
    assert out["task_state_rpc"]["by_method"]["future_get_task"]["total"] == 7.0


def test_task_state_store_actor_wait_status_change_times_out(tmp_path) -> None:
    async def _run() -> None:
        actor = _TaskStateStoreActor(str(tmp_path / "task-state-watch-timeout.sqlite3"))
        try:
            actor.create_task(
                request_id="req-watch-timeout",
                op="sampling.asample",
                domain_key="vllm:test",
                request_json=b"{}",
                now=1.0,
            )

            out = await asyncio.to_thread(
                actor.wait_task_status_change,
                request_id="req-watch-timeout",
                timeout_s=0.01,
            )

            assert out["changed"] is False
            assert out["timeout"] is True
            assert out["record"]["request_id"] == "req-watch-timeout"
            assert out["record"]["status"] == "pending"
        finally:
            actor.close()

    asyncio.run(_run())


def test_task_state_store_actor_wait_terminal_only_ignores_active_progress(tmp_path) -> None:
    async def _run() -> None:
        actor = _TaskStateStoreActor(str(tmp_path / "task-state-terminal-watch.sqlite3"))
        try:
            actor.create_task(
                request_id="req-terminal-watch",
                op="sampling.asample",
                domain_key="vllm:test",
                request_json=b"{}",
                now=1.0,
            )

            waiter = asyncio.create_task(
                asyncio.to_thread(
                    actor.wait_task_status_change,
                    request_id="req-terminal-watch",
                    timeout_s=1.0,
                    terminal_only=True,
                )
            )
            await asyncio.sleep(0.01)
            actor.update_task_metadata(
                request_id="req-terminal-watch",
                metadata={"queue_state": "running"},
                status="running",
                now=2.0,
            )
            await asyncio.sleep(0.01)
            assert waiter.done() is False
            actor.complete_task_failure(
                request_id="req-terminal-watch",
                error="boom",
                now=3.0,
            )

            out = await waiter
            assert out["changed"] is True
            assert out["record"]["status"] == "failed"
        finally:
            actor.close()

    asyncio.run(_run())


def _own_scheduler(store: TaskStateStore, owner_id: str = "scheduler-a") -> int:
    owner = store.acquire_scheduler_owner(owner_id=owner_id, ttl_s=30.0, now=101.0)
    assert owner.ok is True
    return int(owner.epoch)


def _leased_task(store: TaskStateStore) -> tuple[int, str, str]:
    _create_task(store)
    epoch = _own_scheduler(store)
    store.assign_task(
        request_id="req-1",
        subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
        scheduler_epoch=epoch,
        now=102.0,
    )
    lease_id = "lease-1"
    attempt_id = "attempt-1"
    claimed = store.claim_task(
        request_id="req-1",
        subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
        lease_id=lease_id,
        attempt_id=attempt_id,
        consumer_id="runtime-0",
        scheduler_epoch=epoch,
        runtime_generation=7,
        lease_ttl_s=30.0,
        now=103.0,
    )
    assert claimed.record["status"] == "leased"
    return epoch, lease_id, attempt_id


def test_owner_epoch_fences_stale_scheduler() -> None:
    store = TaskStateStore.in_memory()
    try:
        owner_a = store.acquire_scheduler_owner(owner_id="scheduler-a", ttl_s=30.0, now=100.0)
        assert owner_a.ok is True
        assert owner_a.epoch == 1

        blocked = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=110.0)
        assert blocked.ok is False
        assert blocked.reason == ConflictReason.OWNER_ACTIVE
        assert blocked.epoch == 1

        owner_b = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=131.0)
        assert owner_b.ok is True
        assert owner_b.epoch == 2
        stale_renew = store.renew_scheduler_owner(
            owner_id="scheduler-a",
            epoch=1,
            ttl_s=30.0,
            now=132.0,
        )
        assert stale_renew.ok is False
        assert stale_renew.reason == "stale_owner"
    finally:
        store.close()


def test_task_state_store_migrates_staged_payload_columns(tmp_path) -> None:
    db_path = tmp_path / "task_state.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (
                request_id TEXT PRIMARY KEY,
                op TEXT NOT NULL,
                status TEXT NOT NULL,
                domain_key TEXT NOT NULL,
                subqueue_id TEXT,
                lease_id TEXT,
                attempt_id TEXT,
                scheduler_epoch INTEGER,
                runtime_generation INTEGER,
                consumer_id TEXT,
                request_json BLOB NOT NULL,
                payload_hash TEXT,
                result_path TEXT,
                result_checksum TEXT,
                result_size_bytes INTEGER,
                error TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                assigned_at REAL,
                leased_at REAL,
                lease_expires_at REAL,
                finalizing_until REAL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = TaskStateStore(db_path)
    try:
        columns = {
            str(row[1])
            for row in store._conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        assert "staged_payload_path" in columns
        assert "staged_payload_checksum" in columns
        assert "staged_payload_size_bytes" in columns
    finally:
        store.close()


def test_task_state_store_active_load_and_claim_lifecycle() -> None:
    store = TaskStateStore.in_memory()
    try:
        _create_task(store)
        epoch = _own_scheduler(store)

        assigned = store.assign_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            scheduler_epoch=epoch,
            now=102.0,
        )
        assert assigned.record["status"] == "assigned"
        assert assigned.record["subqueue_id"] == "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0"

        claimed = store.claim_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            lease_id="lease-1",
            attempt_id="attempt-1",
            consumer_id="runtime-0",
            scheduler_epoch=epoch,
            runtime_generation=7,
            lease_ttl_s=30.0,
            now=103.0,
        )
        assert claimed.record["status"] == "leased"
        assert claimed.record["lease_id"] == "lease-1"
        assert claimed.record["attempt_id"] == "attempt-1"

        renewed = store.renew_lease(
            request_id="req-1",
            lease_id="lease-1",
            attempt_id="attempt-1",
            scheduler_epoch=epoch,
            runtime_generation=7,
            lease_ttl_s=60.0,
            now=104.0,
        )
        assert renewed.record["status"] == "leased"
        assert renewed.record["lease_expires_at"] == 164.0

        with pytest.raises(TaskStateConflictError):
            store.renew_lease(
                request_id="req-1",
                lease_id="lease-1",
                attempt_id="attempt-1",
                scheduler_epoch=epoch,
                runtime_generation=8,
                lease_ttl_s=60.0,
                now=105.0,
            )

        active = store.list_active_tasks()
        assert [record["request_id"] for record in active] == ["req-1"]
        assert active[0]["metadata"]["queue_kind"] == "model_work_scheduler"
    finally:
        store.close()


def test_finalize_success_is_cas_fenced_and_idempotent() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)

        with pytest.raises(TaskStateConflictError):
            store.begin_finalize(
                request_id="req-1",
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=epoch,
                runtime_generation=8,
                finalize_ttl_s=30.0,
                now=104.0,
            )

        finalizing = store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            now=104.0,
        )
        assert finalizing.record["status"] == "finalizing"
        assert finalizing.record["staged_payload_path"] == "/vePFS-Mindverse/share/mint-results/req-1.json"
        assert finalizing.record["result_path"] is None

        committed = store.commit_finalize_success(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            result_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            result_checksum="sha256:abc",
            result_size_bytes=123,
            now=105.0,
        )
        assert committed.ok is True
        assert committed.idempotent is False
        assert committed.record["status"] == "done"
        assert committed.record["result_path"] == "/vePFS-Mindverse/share/mint-results/req-1.json"
        assert committed.record["staged_payload_path"] is None
        assert store.list_active_tasks() == []

        repeated = store.commit_finalize_success(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            result_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            result_checksum="sha256:abc",
            result_size_bytes=123,
            now=106.0,
        )
        assert repeated.idempotent is True

        with pytest.raises(TaskStateConflictError):
            store.commit_finalize_success(
                request_id="req-1",
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=epoch,
                runtime_generation=7,
                result_path="/vePFS-Mindverse/share/mint-results/req-1-other.json",
                result_checksum="sha256:def",
                result_size_bytes=456,
                now=107.0,
            )
    finally:
        store.close()


def test_finalize_success_rejects_payload_path_that_does_not_match_staged_path() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            now=104.0,
        )

        with pytest.raises(TaskStateConflictError):
            store.commit_finalize_success(
                request_id="req-1",
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=epoch,
                runtime_generation=7,
                result_path="/vePFS-Mindverse/share/mint-results/other.json",
                result_checksum="sha256:abc",
                result_size_bytes=123,
                now=105.0,
            )

        record = store.get_task("req-1")
        assert record["status"] == "finalizing"
        assert record["staged_payload_path"] == "/vePFS-Mindverse/share/mint-results/req-1.json"
    finally:
        store.close()


def test_failure_commit_after_stage_preserves_abandoned_staged_payload() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="req-direct-fail",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="pending",
            now=1.0,
        )
        store.stage_payload(
            request_id="req-direct-fail",
            staged_payload_path="/tmp/payloads/re/req-direct-fail/future__stage-a.json",
            now=2.0,
        )

        failed = store.complete_task_failure(
            request_id="req-direct-fail",
            error="payload write failed",
            now=3.0,
        )

        assert failed.record["status"] == "failed"
        assert failed.record["staged_payload_path"] is None
        assert failed.record["metadata"]["abandoned_staged_payload_paths"] == [
            "/tmp/payloads/re/req-direct-fail/future__stage-a.json"
        ]
    finally:
        store.close()


def test_finalize_failure_after_stage_preserves_abandoned_staged_payload() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/payloads/re/req-1/attempt-1__lease-1.json",
            now=104.0,
        )

        failed = store.commit_finalize_failure(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            error="executor failed after staging",
            now=105.0,
        )

        assert failed.record["status"] == "failed"
        assert failed.record["staged_payload_path"] is None
        assert failed.record["metadata"]["abandoned_staged_payload_paths"] == [
            "/tmp/payloads/re/req-1/attempt-1__lease-1.json"
        ]
    finally:
        store.close()


def test_staged_payload_path_is_not_terminal_payload_until_commit() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/staged-req-1.json",
            now=104.0,
        )

        assert store.list_terminal_payloads_for_eviction(
            older_than_s=1.0,
            now=10_000.0,
            limit=1000,
        ) == []

        committed = store.commit_finalize_success(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            result_path="/tmp/staged-req-1.json",
            result_checksum="sha256:abc",
            result_size_bytes=123,
            now=105.0,
        )
        assert committed.record["staged_payload_path"] is None

        payloads = store.list_terminal_payloads_for_eviction(
            older_than_s=1.0,
            now=10_000.0,
            limit=1000,
        )
        assert [record["request_id"] for record in payloads] == ["req-1"]
        assert payloads[0]["result_path"] == "/tmp/staged-req-1.json"
    finally:
        store.close()


def test_runtime_commit_does_not_require_live_scheduler_owner() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            now=104.0,
        )
        owner_b = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=132.0)
        assert owner_b.ok is True
        assert owner_b.epoch == 2

        committed = store.commit_finalize_success(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            result_path="/vePFS-Mindverse/share/mint-results/req-1.json",
            result_checksum="sha256:abc",
            result_size_bytes=123,
            now=133.0,
        )

        assert committed.record["status"] == "done"
        assert committed.record["staged_payload_path"] is None
    finally:
        store.close()


def test_scheduler_leased_task_can_finalize_after_runtime_marks_running() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.update_task_metadata(
            request_id="req-1",
            metadata={"stage": "prefill"},
            status="running",
            now=104.0,
        )

        finalizing = store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            now=105.0,
        )

        assert finalizing.record["status"] == "finalizing"
        assert finalizing.record["lease_id"] == lease_id
        assert finalizing.record["metadata"]["stage"] == "prefill"
    finally:
        store.close()


def test_requeue_task_resets_active_record_for_reclaim() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/payloads/re/req-1/attempt-1__lease-1.json",
            now=103.5,
        )
        requeued = store.requeue_task(
            request_id="req-1",
            scheduler_epoch=epoch,
            reason="lease_expired",
            now=104.0,
        )
        assert requeued.record["status"] == "pending"
        assert requeued.record["lease_id"] is None
        assert requeued.record["attempt_id"] is None
        assert requeued.record["staged_payload_path"] is None
        assert requeued.record["metadata"]["abandoned_staged_payload_paths"] == [
            "/tmp/payloads/re/req-1/attempt-1__lease-1.json"
        ]

        store.assign_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            scheduler_epoch=epoch,
            now=105.0,
        )
        claimed = store.claim_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            lease_id=f"{lease_id}-retry",
            attempt_id=f"{attempt_id}-retry",
            consumer_id="runtime-0",
            scheduler_epoch=epoch,
            runtime_generation=7,
            lease_ttl_s=30.0,
            now=106.0,
        )
        assert claimed.record["status"] == "leased"
        assert claimed.record["lease_id"] == "lease-1-retry"
    finally:
        store.close()


def test_stale_finalizer_uses_distinct_payload_path_and_cannot_overwrite_retry(tmp_path) -> None:
    store = TaskStateStore.in_memory()
    payloads = TaskPayloadStore(tmp_path)
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        stale_path = payloads.payload_path(request_id="req-1", attempt_id=f"{attempt_id}__{lease_id}")
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path=str(stale_path),
            now=104.0,
        )
        store.requeue_task(
            request_id="req-1",
            scheduler_epoch=epoch,
            reason="lease_expired",
            now=105.0,
        )
        store.assign_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            scheduler_epoch=epoch,
            now=106.0,
        )
        store.claim_task(
            request_id="req-1",
            subqueue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",
            lease_id="lease-2",
            attempt_id=attempt_id,
            consumer_id="runtime-0",
            scheduler_epoch=epoch,
            runtime_generation=7,
            lease_ttl_s=30.0,
            now=107.0,
        )
        retry_path = payloads.payload_path(request_id="req-1", attempt_id=f"{attempt_id}__lease-2")
        store.begin_finalize(
            request_id="req-1",
            lease_id="lease-2",
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            staged_payload_path=str(retry_path),
            now=108.0,
        )
        retry_meta = payloads.write_json_payload(
            request_id="req-1",
            attempt_id=f"{attempt_id}__lease-2",
            payload={"winner": "retry"},
        )
        committed = store.commit_finalize_success(
            request_id="req-1",
            lease_id="lease-2",
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            result_path=str(retry_meta["path"]),
            result_checksum=str(retry_meta["checksum"]),
            result_size_bytes=int(retry_meta["size_bytes"]),
            now=109.0,
        )
        assert committed.record["status"] == "done"

        payloads.write_json_payload(
            request_id="req-1",
            attempt_id=f"{attempt_id}__{lease_id}",
            payload={"winner": "stale"},
        )
        with pytest.raises(TaskStateConflictError):
            store.commit_finalize_success(
                request_id="req-1",
                lease_id=lease_id,
                attempt_id=attempt_id,
                scheduler_epoch=epoch,
                runtime_generation=7,
                result_path=str(stale_path),
                result_checksum="sha256:stale",
                result_size_bytes=1,
                now=110.0,
            )
        assert payloads.read_json_payload(
            path=str(retry_meta["path"]),
            expected_checksum=str(retry_meta["checksum"]),
        ) == {"winner": "retry"}
        assert store.get_task("req-1")["metadata"]["abandoned_staged_payload_paths"] == [str(stale_path)]
    finally:
        store.close()


def test_future_style_task_lifecycle_and_metadata_lookup() -> None:
    store = TaskStateStore.in_memory()
    try:
        created = store.ensure_task(
            request_id="future-1",
            op="training.train_step",
            domain_key="future:default",
            metadata={"model_id": "model-a"},
            status="queued",
            now=100.0,
        )
        assert created["created"] is True
        assert created["record"]["status"] == "queued"

        running = store.update_task_metadata(
            request_id="future-1",
            metadata={"stage": "running"},
            status="running",
            now=101.0,
        )
        assert running.record["metadata"]["model_id"] == "model-a"
        assert running.record["metadata"]["stage"] == "running"

        completed = store.complete_task_success(
            request_id="future-1",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 102.0},
            now=102.0,
        )
        assert completed["record"]["status"] == "done"

        by_meta = store.list_tasks_by_metadata(
            filters={"model_id": "model-a"},
            statuses=["done"],
        )
        assert [record["request_id"] for record in by_meta] == ["future-1"]

        retrieved = store.mark_task_retrieved(request_id="future-1", now=103.0)
        assert retrieved["record"]["status"] == "retrieved"
        assert retrieved["record"]["metadata"]["terminal_status"] == "done"
    finally:
        store.close()


def test_billing_outbox_claim_delete_and_terminal_metadata() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="future-bill",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-bill",
            charge_item="sampling",
            quantity=7,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=101.0,
        )
        completed = store.complete_task_success(
            request_id="future-bill",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 102.0},
            billing_observations=[observation],
            now=102.0,
        )
        assert completed["record"]["metadata"]["billing_status"] == "outboxed"
        assert completed["record"]["metadata"]["billing_observation_count"] == 1

        stats = store.billing_outbox_stats(now=103.0)
        assert stats["by_status"]["pending"]["rows"] == 1

        claimed = store.claim_billing_outbox(claim_id="claim-1", limit=10, lease_ttl_s=30.0, now=104.0)
        assert len(claimed) == 1
        assert claimed[0]["event"]["charge_item"] == "sampling"
        assert claimed[0]["event"]["quantity"] == 7
        assert claimed[0]["event"]["label"] == (
            "model=Qwen/Test,route=sampling.asample,dimension=sample,unit=tokens"
        )

        stats = store.billing_outbox_stats(now=105.0)
        assert stats["by_status"]["flushing"]["rows"] == 1

        deleted = store.delete_billing_outbox_claim(
            claim_id="claim-1",
            outbox_ids=[claimed[0]["outbox_id"]],
        )
        assert deleted == {"ok": True, "deleted": 1}
        assert store.billing_outbox_stats(now=106.0)["by_status"] == {}
    finally:
        store.close()


def test_billing_outbox_failure_terminal_is_not_billed() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="future-fail",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        store.complete_task_failure(
            request_id="future-fail",
            error="boom",
            metadata={"failed_at": 102.0},
            now=102.0,
        )
        assert store.billing_outbox_stats(now=103.0)["by_status"] == {}
    finally:
        store.close()


def test_billing_outbox_not_written_when_terminal_success_rejected(monkeypatch) -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="future-reject",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        appended: list[dict] = []

        def _append(*args, **kwargs):
            appended.append({"args": args, "kwargs": kwargs})
            return {"ok": True, "inserted": 1, "duplicate": 0, "conflicts": 0, "errors": []}

        monkeypatch.setattr(store._hot_kv, "append_billing_outbox", _append)

        def _fail_record_event(*_args, **_kwargs):
            raise RuntimeError("terminal event write failed")

        monkeypatch.setattr(store, "_record_event", _fail_record_event)

        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-reject",
            charge_item="sampling",
            quantity=7,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=101.0,
        )
        with pytest.raises(RuntimeError, match="terminal event write failed"):
            store.complete_task_success(
                request_id="future-reject",
                result_path="/tmp/result.json",
                result_checksum="sha256:abc",
                result_size_bytes=17,
                metadata={"done_at": 102.0},
                billing_observations=[observation],
                now=102.0,
            )

        assert appended == []
        assert store.billing_outbox_stats(now=103.0)["by_status"] == {}
        assert store.get_task("future-reject")["status"] == "running"
    finally:
        store.close()


def test_billing_outbox_idempotent_terminal_retry_writes_missing_billing() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="future-idempotent-bill",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        store.complete_task_success(
            request_id="future-idempotent-bill",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 101.0},
            now=101.0,
        )
        assert store.billing_outbox_stats(now=102.0)["by_status"] == {}

        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-idempotent-bill",
            charge_item="sampling",
            quantity=7,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=103.0,
        )
        repeated = store.complete_task_success(
            request_id="future-idempotent-bill",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 101.0},
            billing_observations=[observation],
            now=104.0,
        )
        assert repeated["idempotent"] is True
        assert repeated["record"]["metadata"]["billing_status"] == "outboxed"
        claimed = store.claim_billing_outbox(claim_id="claim-idempotent", limit=10, lease_ttl_s=30.0)
        assert len(claimed) == 1
        assert claimed[0]["event"]["request_id"] == "future-idempotent-bill"
    finally:
        store.close()


def test_billing_observations_from_input_rejects_malformed_input() -> None:
    gateway_auth = {
        "user_id": "user-1",
        "user_role": "user",
        "account_id": "acct-1",
        "apikey_id": "key-1",
        "request_id": "gw-1",
    }
    with pytest.raises(ValueError, match="missing required keys"):
        billing_observations_from_input(
            gateway_auth=gateway_auth,
            request_id="req-bad",
            billing_input={
                "charge_item": "sampling",
                "quantity": 1,
                "unit": "tokens",
                "route": "sampling.asample",
            },
        )
    for malformed in ([], "bad", True):
        with pytest.raises(ValueError, match="must be a dict"):
            billing_observations_from_input(
                gateway_auth=gateway_auth,
                request_id="req-bad-type",
                billing_input=malformed,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="metadata"):
        billing_observations_from_input(
            gateway_auth=gateway_auth,
            request_id="req-bad-meta",
            billing_input={
                "charge_item": "sampling",
                "quantity": 1,
                "unit": "tokens",
                "route": "sampling.asample",
                "dimension": "sample",
                "metadata": "not-a-dict",
            },
        )


def test_billing_outbox_write_failure_sets_underbilling_signal(monkeypatch) -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="future-bill-drop",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-bill-drop",
            charge_item="sampling",
            quantity=7,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=101.0,
        )

        def _fail_append(*_args, **_kwargs):
            return {"ok": False, "errors": [{"error": "kv unavailable"}], "inserted": 0}

        monkeypatch.setattr(store._hot_kv, "append_billing_outbox", _fail_append)

        completed = store.complete_task_success(
            request_id="future-bill-drop",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 102.0},
            billing_observations=[observation],
            now=102.0,
        )
        assert completed["record"]["status"] == "done"
        assert completed["record"]["metadata"]["billing_status"] == "dropped"
        assert completed["record"]["metadata"]["billing_error"]["errors"][0]["error"] == "kv unavailable"
    finally:
        store.close()


def test_billing_outbox_hash_conflict_sets_underbilling_signal() -> None:
    store = TaskStateStore.in_memory()
    try:
        first = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-bill-conflict",
            charge_item="sampling",
            quantity=7,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=101.0,
        )
        assert store.append_billing_outbox(observations=[first], source="test", now=101.0)["ok"] is True
        store.ensure_task(
            request_id="future-bill-conflict",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=102.0,
        )
        conflicting = dict(first)
        conflicting["quantity"] = 9
        completed = store.complete_task_success(
            request_id="future-bill-conflict",
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 103.0},
            billing_observations=[conflicting],
            now=103.0,
        )
        assert completed["record"]["status"] == "done"
        assert completed["record"]["metadata"]["billing_status"] == "dropped"
        assert completed["record"]["metadata"]["billing_error"]["conflicts"] == 1
    finally:
        store.close()


def test_task_future_service_flushes_billing_outbox_and_deletes_conflicts(monkeypatch) -> None:
    store = TaskStateStore.in_memory()
    try:
        observations = [
            build_billing_observation(
                account_id="acct-1",
                apikey_id="key-1",
                request_id="req-bill-flush",
                charge_item="sampling",
                quantity=3,
                unit="tokens",
                route="sampling.asample",
                dimension="prefill",
                model="Qwen/Test",
                observed_at=101.0,
            ),
            build_billing_observation(
                account_id="acct-1",
                apikey_id="key-1",
                request_id="req-bill-flush",
                charge_item="sampling",
                quantity=5,
                unit="tokens",
                route="sampling.asample",
                dimension="sample",
                model="Qwen/Test",
                observed_at=101.0,
            ),
        ]
        store.append_billing_outbox(observations=observations, source="test", now=102.0)

        class _TaskState:
            async def async_claim_billing_outbox(self, **kwargs):
                return store.claim_billing_outbox(**kwargs)

            async def async_delete_billing_outbox_claim(self, **kwargs):
                return store.delete_billing_outbox_claim(**kwargs)

            async def async_mark_billing_outbox_claim_failed(self, **kwargs):
                return store.mark_billing_outbox_claim_failed(**kwargs)

        class _UsageStore:
            def __init__(self):
                self.events = []

            async def write_events(self, events):
                self.events.extend(list(events))
                return [self.events[0].event_id]

        usage_store = _UsageStore()

        async def _fake_get_usage_store():
            return usage_store

        monkeypatch.setattr("mint_server.usage_store.get_usage_store", _fake_get_usage_store)

        service = TaskFutureService(task_state_client=_TaskState())
        out = asyncio.run(service.async_flush_billing_outbox(limit=10, lease_ttl_s=60.0, claim_id="claim-flush"))

        assert out == {"ok": True, "claimed": 2, "inserted": 1, "conflict": 1, "failed": 0}
        assert len(usage_store.events) == 2
        assert store.billing_outbox_stats(now=110.0)["by_status"] == {}
    finally:
        store.close()


def test_task_future_service_flush_failure_releases_claim(monkeypatch) -> None:
    store = TaskStateStore.in_memory()
    try:
        store.append_billing_outbox(
            observations=[
                build_billing_observation(
                    account_id="acct-1",
                    apikey_id="key-1",
                    request_id="req-bill-fail",
                    charge_item="sampling",
                    quantity=3,
                    unit="tokens",
                    route="sampling.asample",
                    dimension="prefill",
                    model="Qwen/Test",
                    observed_at=101.0,
                )
            ],
            source="test",
            now=102.0,
        )

        class _TaskState:
            async def async_claim_billing_outbox(self, **kwargs):
                return store.claim_billing_outbox(**kwargs)

            async def async_delete_billing_outbox_claim(self, **kwargs):
                return store.delete_billing_outbox_claim(**kwargs)

            async def async_mark_billing_outbox_claim_failed(self, **kwargs):
                return store.mark_billing_outbox_claim_failed(**kwargs)

        class _UsageStore:
            async def write_events(self, _events):
                raise RuntimeError("pg unavailable")

        async def _fake_get_usage_store():
            return _UsageStore()

        monkeypatch.setattr("mint_server.usage_store.get_usage_store", _fake_get_usage_store)

        service = TaskFutureService(task_state_client=_TaskState())
        out = asyncio.run(service.async_flush_billing_outbox(limit=10, lease_ttl_s=60.0, claim_id="claim-fail"))

        assert out["ok"] is False
        assert out["failed"] == 1
        stats = store.billing_outbox_stats(now=110.0)
        assert stats["by_status"]["pending"]["rows"] == 1
    finally:
        store.close()


def test_task_future_service_permanent_usage_error_marks_outbox_failed(monkeypatch) -> None:
    store = TaskStateStore.in_memory()
    try:
        store.append_billing_outbox(
            observations=[
                build_billing_observation(
                    account_id="acct-1",
                    apikey_id="key-1",
                    request_id="req-bill-permanent",
                    charge_item="sampling",
                    quantity=3,
                    unit="tokens",
                    route="sampling.asample",
                    dimension="prefill",
                    model="Qwen/Test",
                    observed_at=101.0,
                )
            ],
            source="test",
            now=102.0,
        )

        class _TaskState:
            async def async_claim_billing_outbox(self, **kwargs):
                return store.claim_billing_outbox(**kwargs)

            async def async_delete_billing_outbox_claim(self, **kwargs):
                return store.delete_billing_outbox_claim(**kwargs)

            async def async_mark_billing_outbox_claim_failed(self, **kwargs):
                return store.mark_billing_outbox_claim_failed(**kwargs)

        class _UsageStore:
            async def write_events(self, _events):
                raise ValueError("unsupported usage_event charge_item: 'bad'")

        async def _fake_get_usage_store():
            return _UsageStore()

        monkeypatch.setattr("mint_server.usage_store.get_usage_store", _fake_get_usage_store)

        service = TaskFutureService(task_state_client=_TaskState())
        out = asyncio.run(service.async_flush_billing_outbox(limit=10, lease_ttl_s=60.0, claim_id="claim-permanent"))

        assert out["ok"] is False
        assert out["permanent"] is True
        stats = store.billing_outbox_stats(now=110.0)
        assert stats["by_status"]["failed"]["rows"] == 1
    finally:
        store.close()


def test_create_task_is_idempotent_for_precreated_scheduler_task() -> None:
    store = TaskStateStore.in_memory()
    try:
        precreated = store.ensure_task(
            request_id="req-precreated",
            op="sampling.asample",
            domain_key="vllm:Qwen/Test",
            request_json=b'{"prompt":"a"}',
            metadata={"stage": "queued"},
            status="queued",
            now=100.0,
        )
        assert precreated["created"] is True

        created = store.create_task(
            request_id="req-precreated",
            op="sampling.asample",
            domain_key="vllm:Qwen/Test",
            request_json=b'{"prompt":"b"}',
            payload_hash="hash-1",
            metadata={"model_work_scheduler": True},
            now=101.0,
        )

        assert created.ok is True
        assert created.created is False
        assert created.record["status"] == "queued"
        assert created.record["payload_hash"] == "hash-1"
        assert created.record["request_json"] == b'{"prompt":"b"}'
        assert created.record["metadata"]["stage"] == "queued"
        assert created.record["metadata"]["model_work_scheduler"] is True

        with pytest.raises(TaskStateConflictError):
            store.create_task(
                request_id="req-precreated",
                op="sampling.compute_logprobs",
                domain_key="vllm:Qwen/Test",
                request_json=b"{}",
            )
    finally:
        store.close()


def test_finalize_failure_records_terminal_error() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=30.0,
            now=104.0,
        )

        failed = store.commit_finalize_failure(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            error="executor failed",
            now=105.0,
        )

        assert failed.record["status"] == "failed"
        assert failed.record["error"] == "executor failed"
        assert store.get_task("req-1")["status"] == "failed"
    finally:
        store.close()


def test_task_state_store_reaper_expires_pending_payloads_and_tombstones() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="pending-old",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="pending",
            now=100.0,
        )
        store.ensure_task(
            request_id="queued-old",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="queued",
            now=101.0,
        )
        store.ensure_task(
            request_id="assigned-old",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="assigned",
            now=102.0,
        )
        store.ensure_task(
            request_id="running-old",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="running",
            now=100.0,
        )
        expired = store.expire_active_tasks(older_than_s=10.0, now=200.0, limit=1000)
        assert expired == ["pending-old", "queued-old", "assigned-old"]
        assert store.get_task("pending-old")["status"] == "expired"
        assert store.get_task("queued-old")["status"] == "expired"
        assert store.get_task("assigned-old")["status"] == "expired"
        assert store.get_task("running-old")["status"] == "running"

        store.ensure_task(
            request_id="done-old",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="pending",
            now=10.0,
        )
        store.complete_task_success(
            request_id="done-old",
            result_path="/tmp/done-old.json",
            result_checksum="sha256:abc",
            result_size_bytes=12,
            metadata={"done_at": 20.0},
            now=20.0,
        )
        payloads = store.list_terminal_payloads_for_eviction(older_than_s=100.0, now=200.0, limit=1000)
        assert [record["request_id"] for record in payloads] == ["done-old"]
        marked = store.mark_payload_evicted(
            request_id="done-old",
            expected_result_path="/tmp/done-old.json",
            now=201.0,
        )
        assert marked["record"]["result_path"] is None
        assert marked["record"]["metadata"]["payload_evicted_at"] == 201.0

        assert store.delete_expired_tombstones(older_than_s=1000.0, now=300.0, limit=1000) == []
        deleted = store.delete_expired_tombstones(older_than_s=100.0, now=200.0, limit=1000)
        assert deleted == ["done-old"]
        with pytest.raises(KeyError):
            store.get_task("done-old")
    finally:
        store.close()


def test_expired_leases_include_finalizing_deadline() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        assert store.list_expired_leases(now=110.0) == []

        store.begin_finalize(
            request_id="req-1",
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=epoch,
            runtime_generation=7,
            finalize_ttl_s=10.0,
            now=120.0,
        )

        assert store.list_expired_leases(now=129.0) == []
        expired = store.list_expired_leases(now=131.0)
        assert [record["request_id"] for record in expired] == ["req-1"]
        assert expired[0]["status"] == "finalizing"
    finally:
        store.close()


def test_task_state_store_actor_uses_single_db_path(tmp_path) -> None:
    db_path = tmp_path / "task_state.sqlite3"
    actor = _TaskStateStoreActor(str(db_path))
    try:
        owner = actor.acquire_scheduler_owner(owner_id="scheduler-a", ttl_s=30.0, now=100.0)
        assert owner["ok"] is True
        created = actor.create_task(
            request_id="req-actor",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            payload_hash="hash",
            metadata={"queue_kind": "model_work_scheduler"},
            now=101.0,
        )
        assert created["created"] is True
        stats = actor.stats()
        assert stats["db_path"] == str(db_path)
        assert stats["active_tasks"] == 1
        assert stats["active_by_status"] == {"pending": 1}
        assert actor.integrity_check() == "ok"
    finally:
        actor.close()

    reopened = _TaskStateStoreActor(str(db_path))
    try:
        assert [record["request_id"] for record in reopened.list_active_tasks()] == ["req-actor"]
    finally:
        reopened.close()


def test_task_state_store_actor_stats_uses_future_metrics_not_active_task_scan(tmp_path) -> None:
    actor = _TaskStateStoreActor(str(tmp_path / "task-state-stats-no-active-scan.sqlite3"))
    try:
        actor.create_task(
            request_id="req-actor",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            now=101.0,
        )

        def _unexpected_list_active_tasks(*_args, **_kwargs):
            raise AssertionError("stats must not scan active task rows")

        actor._store.list_active_tasks = _unexpected_list_active_tasks  # type: ignore[method-assign]

        stats = actor.stats()

        assert stats["active_tasks"] == 1
        assert stats["active_by_status"] == {"pending": 1}
    finally:
        actor.close()


def test_task_state_store_owns_sampling_session_metadata() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.upsert_sampling_session(
            session_id="sampler-a",
            info={
                "session_id": "sampler-a",
                "base_model": "Qwen/Test",
                "last_activity": 100.0,
                "metadata_version": 3,
            },
        )
        stale = {
            "session_id": "sampler-a",
            "base_model": "stale",
            "last_activity": 120.0,
            "metadata_version": 2,
        }
        store.upsert_sampling_session(session_id="sampler-a", info=stale)

        info = store.get_sampling_session(session_id="sampler-a")
        assert info is not None
        assert info["base_model"] == "Qwen/Test"
        assert info["last_activity"] == 120.0
        assert store.set_sampling_session_last_activity(session_id="sampler-a", last_activity=130.0) == 130.0
        assert store.list_sampling_sessions()[0]["last_activity"] == 130.0

        store.delete_sampling_session(session_id="sampler-a")
        assert store.get_sampling_session(session_id="sampler-a") is None
    finally:
        store.close()


def test_task_state_store_owns_session_and_sampler_indices() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.upsert_session_index(
            session_id="root-session",
            info={"session_id": "root-session", "user_id": "owner-a"},
        )
        store.add_training_run_to_session_index(
            session_id="root-session",
            training_run_id="train-1",
            user_id="owner-b",
            created_at="2026-04-01T00:00:00",
        )
        store.add_training_run_to_session_index(session_id="root-session", training_run_id="train-1")
        store.add_sampler_to_session_index(session_id="root-session", sampler_id="sampler-a")
        store.add_heartbeat_sampler_to_session_index(session_id="root-session", sampler_id="sampler-b")

        index = store.get_session_index(session_id="root-session")
        assert index is not None
        assert index["user_id"] == "owner-a"
        assert index["training_run_ids"] == ["train-1"]
        assert index["sampler_ids"] == ["sampler-a", "sampler-b"]
        assert index["heartbeat_sampler_ids"] == ["sampler-b"]

        store.upsert_sampler_index(sampler_id="sampler-b", info={"sampler_id": "sampler-b", "session_id": "root-session"})
        assert store.get_sampler_index(sampler_id="sampler-b") == {
            "sampler_id": "sampler-b",
            "session_id": "root-session",
        }

        store.remove_sampler_from_session_index(session_id="root-session", sampler_id="sampler-b")
        assert store.get_session_index(session_id="root-session")["sampler_ids"] == ["sampler-a"]
        assert store.get_session_index(session_id="root-session")["heartbeat_sampler_ids"] == []

        store.delete_sampler_index(sampler_id="sampler-b")
        assert store.get_sampler_index(sampler_id="sampler-b") is None
    finally:
        store.close()


def test_task_state_store_owns_session_heartbeats() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.update_session_heartbeat(session_id="session-old", now=10.0)
        store.update_session_heartbeat(session_id="session-fresh", now=100.0)

        assert store.session_heartbeat_size() == 2
        assert store.get_session_heartbeat(session_id="session-old") == 10.0
        assert store.is_session_heartbeat_stale(session_id="session-old", ttl_s=50.0, now=100.0) is True
        assert store.is_session_heartbeat_stale(session_id="missing", ttl_s=50.0, now=100.0) is False

        assert store.prune_session_heartbeats(max_age_s=50.0, now=100.0) == 1
        assert store.get_session_heartbeat(session_id="session-old") is None
        assert store.delete_session_heartbeat(session_id="session-fresh") is True
        assert store.session_heartbeat_size() == 0
    finally:
        store.close()


def test_task_state_store_owns_training_session_metadata() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.upsert_training_session(
            model_id="model-a",
            info={
                "model_id": "model-a",
                "session_id": "session-a",
                "current_step": 3,
                "last_activity": 100.0,
                "metadata_version": 3,
            },
        )
        store.upsert_training_session(
            model_id="model-a",
            info={
                "model_id": "model-a",
                "session_id": "stale",
                "current_step": 5,
                "last_activity": 120.0,
                "metadata_version": 2,
            },
        )

        info = store.get_training_session(model_id="model-a")
        assert info is not None
        assert info["session_id"] == "session-a"
        assert info["current_step"] == 5
        assert info["last_activity"] == 120.0
        assert store.bump_training_session_step(model_id="model-a") == 6
        assert store.set_training_session_step(model_id="model-a", step=4) == 6
        assert store.set_training_session_last_activity(model_id="model-a", last_activity=130.0) == 130.0
        assert store.list_training_sessions()[0]["current_step"] == 6

        store.delete_training_session(model_id="model-a")
        assert store.get_training_session(model_id="model-a") is None
    finally:
        store.close()


def test_task_future_service_stages_payload_metadata_before_direct_resolve(tmp_path) -> None:
    store = TaskStateStore.in_memory()
    observed_updates: list[dict] = []
    try:
        store.ensure_task(
            request_id="req-direct",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            metadata={"model_work_attempt_id": "stale-model-work-attempt"},
            status="pending",
            now=1.0,
        )

        class _LocalTaskStateClient:
            async def async_get_task(self, request_id: str) -> dict:
                return store.get_task(request_id)

            async def async_stage_payload(self, **kwargs):
                observed_updates.append(dict(kwargs))
                out = store.stage_payload(**kwargs)
                assert out["record"]["status"] == "pending"
                return out

            async def async_complete_task_success(self, **kwargs):
                return store.complete_task_success(**kwargs)

            async def async_append_billing_outbox(self, **_kwargs):
                raise AssertionError("billing must be committed through complete_task_success")

        service = TaskFutureService(
            task_state_client=_LocalTaskStateClient(),
            future_state_client=_LocalTaskStateClient(),
            payload_store=TaskPayloadStore(tmp_path),
        )

        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="req-direct",
            charge_item="sampling",
            quantity=3,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=2.0,
        )

        asyncio.run(service.async_resolve("req-direct", {"ok": True}, billing_observations=[observation]))

        assert observed_updates[0]["request_id"] == "req-direct"
        assert observed_updates[0]["staged_payload_path"].startswith(str(tmp_path / "re" / "req-direct" / "future__"))
        assert "stale-model-work-attempt" not in observed_updates[0]["staged_payload_path"]
        assert observed_updates[0]["metadata"] == {
            "staged_payload_path": observed_updates[0]["staged_payload_path"],
        }
        record = store.get_task("req-direct")
        assert record["status"] == "done"
        assert record["result_path"] == observed_updates[0]["staged_payload_path"]
        assert record["staged_payload_path"] is None
        assert record["metadata"]["payload_state"] == "committed"
        assert record["metadata"]["staged_payload_path"] is None
        assert record["metadata"]["billing_status"] == "outboxed"
        claimed = store.claim_billing_outbox(claim_id="claim-direct", limit=10, lease_ttl_s=30.0)
        assert len(claimed) == 1
        assert claimed[0]["event"]["request_id"] == "req-direct"
    finally:
        store.close()


def test_task_future_service_wait_status_falls_back_to_scheduler_task_state() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id="req-scheduler-pending",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            metadata={"queue_kind": "model_work_scheduler"},
            status="pending",
            now=1.0,
        )

        class _MissingFutureStateClient:
            async def async_wait_task_status_change(self, **_kwargs):
                return {"changed": True, "missing": True, "request_id": "req-scheduler-pending"}

            async def async_stats(self):
                return {"store": "missing-future-state"}

        class _LocalTaskStateClient:
            async def async_get_task(self, request_id: str) -> dict:
                return store.get_task(request_id)

            async def async_wait_task_status_change(self, **kwargs):
                record = store.get_task(str(kwargs["request_id"]))
                return {"changed": False, "timeout": True, "record": record}

            async def async_stats(self):
                return {"pending": 1}

        service = TaskFutureService(
            task_state_client=_LocalTaskStateClient(),
            future_state_client=_MissingFutureStateClient(),
        )

        status = asyncio.run(
            service.async_wait_status_change(
                "req-scheduler-pending",
                timeout_s=0.0,
                terminal_only=True,
            )
        )
        debug = asyncio.run(service.async_debug_snapshot())

        assert status is None
        assert debug["future_state_store"] == {"store": "missing-future-state"}
        assert debug["task_state_store"]["pending"] == 1
    finally:
        store.close()


def test_task_state_actor_future_success_preserves_billing_observations(tmp_path) -> None:
    from mint_server.backend.stores.future_state_store import FutureStateStore

    actor = _TaskStateStoreActor(str(tmp_path / "task-state.sqlite3"))
    actor._future_store = FutureStateStore.in_memory()
    try:
        actor.future_ensure_task(
            request_id="future-actor-bill",
            op="sampling.asample",
            domain_key="future:default",
            metadata={},
            status="running",
            now=100.0,
        )
        actor.future_stage_payload(
            request_id="future-actor-bill",
            staged_payload_path="/tmp/future-actor-result.json",
            metadata={"staged_payload_path": "/tmp/future-actor-result.json"},
            now=101.0,
        )
        observation = build_billing_observation(
            account_id="acct-1",
            apikey_id="key-1",
            request_id="future-actor-bill",
            charge_item="sampling",
            quantity=11,
            unit="tokens",
            route="sampling.asample",
            dimension="sample",
            model="Qwen/Test",
            observed_at=102.0,
        )
        out = actor.future_complete_task_success(
            request_id="future-actor-bill",
            result_path="/tmp/future-actor-result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 103.0},
            billing_observations=[observation],
            now=103.0,
        )

        assert out["record"]["status"] == "done"
        assert out["record"]["metadata"]["billing_status"] == "outboxed"
        claimed = actor.claim_billing_outbox(claim_id="claim-future-actor", limit=10, lease_ttl_s=30.0)
        assert len(claimed) == 1
        assert claimed[0]["event"]["request_id"] == "future-actor-bill"
        assert claimed[0]["event"]["quantity"] == 11
    finally:
        actor.close()


def test_task_state_store_owns_gateway_routes() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.upsert_gateway_sampling_session(
            sampling_session_id="sampler-a",
            upstream_alias="mint-prod-aliyun",
            base_model="Qwen/Test",
        )
        store.upsert_gateway_training_model(
            model_id="model-a",
            upstream_alias="mint-prod-aliyun",
            base_model="Qwen/Test",
            owner_id="owner-a",
        )

        assert store.get_gateway_sampling_session(sampling_session_id="sampler-a") == {
            "upstream_alias": "mint-prod-aliyun",
            "base_model": "Qwen/Test",
        }
        assert store.get_gateway_training_model(model_id="model-a") == {
            "upstream_alias": "mint-prod-aliyun",
            "base_model": "Qwen/Test",
            "owner_id": "owner-a",
        }
        snapshot = store.list_gateway_routes()
        assert sorted(snapshot) == ["sampling_sessions", "training_models"]

        store.delete_gateway_sampling_session(sampling_session_id="sampler-a")
        store.delete_gateway_training_model(model_id="model-a")
        assert store.get_gateway_sampling_session(sampling_session_id="sampler-a") is None
        assert store.get_gateway_training_model(model_id="model-a") is None
    finally:
        store.close()


def test_task_state_store_actor_exposes_session_metadata_methods(tmp_path) -> None:
    actor = _TaskStateStoreActor(str(tmp_path / "task_state.sqlite3"))
    try:
        actor.upsert_sampling_session(session_id="sampler-a", info={"session_id": "sampler-a"})
        assert actor.get_sampling_session(session_id="sampler-a")["session_id"] == "sampler-a"

        actor.upsert_training_session(model_id="model-a", info={"model_id": "model-a"})
        assert actor.get_training_session(model_id="model-a")["model_id"] == "model-a"

        actor.upsert_gateway_sampling_session(
            sampling_session_id="sampler-a",
            upstream_alias="upstream-a",
            base_model="Qwen/Test",
        )
        assert actor.get_gateway_sampling_session(sampling_session_id="sampler-a")["upstream_alias"] == "upstream-a"

        actor.add_heartbeat_sampler_to_session_index(session_id="root-session", sampler_id="sampler-a")
        assert actor.get_session_index(session_id="root-session")["heartbeat_sampler_ids"] == ["sampler-a"]

        actor.update_session_heartbeat(session_id="root-session", now=12.0)
        assert actor.get_session_heartbeat(session_id="root-session") == 12.0
    finally:
        actor.close()


def test_task_future_service_reaper_retries_payload_delete_failures(tmp_path, monkeypatch) -> None:
    from mint_server import config as config_module

    store = TaskStateStore.in_memory()

    class _FailingPayloadStore:
        async def async_delete_json_payload(self, *, path):
            raise RuntimeError("delete failed")

    class _WorkingPayloadStore:
        async def async_delete_json_payload(self, *, path):
            from pathlib import Path

            Path(path).unlink()
            return True

    class _LocalTaskStateClient:
        async def async_ensure_task(self, **kwargs):
            return store.ensure_task(**kwargs)

        async def async_complete_task_success(self, **kwargs):
            return store.complete_task_success(**kwargs)

        async def async_expire_active_tasks(self, **kwargs):
            return store.expire_active_tasks(**kwargs)

        async def async_list_terminal_payloads_for_eviction(self, **kwargs):
            return store.list_terminal_payloads_for_eviction(**kwargs)

        async def async_mark_payload_evicted(self, **kwargs):
            return store.mark_payload_evicted(**kwargs)

        async def async_delete_expired_tombstones(self, **kwargs):
            return store.delete_expired_tombstones(**kwargs)

        async def async_record_payload_evict_error(self, **kwargs):
            return store.record_payload_evict_error(**kwargs)

        async def async_list_staged_payloads_for_gc(self, **kwargs):
            return store.list_staged_payloads_for_gc(**kwargs)

        async def async_mark_staged_payload_gc_deleted(self, **kwargs):
            return store.mark_staged_payload_gc_deleted(**kwargs)

    try:
        store.ensure_task(
            request_id="req-fail-delete",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="pending",
            now=1.0,
        )
        result_path = tmp_path / "payload.json"
        result_path.write_text("{}", encoding="utf-8")
        store.complete_task_success(
            request_id="req-fail-delete",
            result_path=str(result_path),
            result_checksum="sha256:abc",
            result_size_bytes=2,
            metadata={"done_at": 10.0},
            now=10.0,
        )
        monkeypatch.setattr(config_module.config, "task_pending_ttl_s", 86400.0, raising=False)
        monkeypatch.setattr(config_module.config, "task_result_ttl_s", 1.0, raising=False)
        monkeypatch.setattr(config_module.config, "task_tombstone_ttl_s", 10**12, raising=False)

        service = TaskFutureService(
            task_state_client=_LocalTaskStateClient(),
            future_state_client=_LocalTaskStateClient(),
            payload_store=_FailingPayloadStore(),
        )
        out = asyncio.run(service.async_reap())

        assert out["payload_evicted"] == []
        assert out["payload_evict_errors"][0]["request_id"] == "req-fail-delete"
        record = store.get_task("req-fail-delete")
        assert record["result_path"] == str(result_path)
        assert "payload_evicted_at" not in record["metadata"]

        service = TaskFutureService(
            task_state_client=_LocalTaskStateClient(),
            future_state_client=_LocalTaskStateClient(),
            payload_store=_WorkingPayloadStore(),
        )
        out = asyncio.run(service.async_reap())

        assert out["payload_evicted"] == ["req-fail-delete"]
        record = store.get_task("req-fail-delete")
        assert record["result_path"] is None
        assert record["metadata"]["payload_evicted_at"] > 0
        assert not result_path.exists()
    finally:
        store.close()


def test_task_future_service_reaper_deletes_abandoned_staged_payload(tmp_path, monkeypatch) -> None:
    from mint_server import config as config_module

    store = TaskStateStore.in_memory()

    class _LocalTaskStateClient:
        async def async_expire_active_tasks(self, **kwargs):
            return store.expire_active_tasks(**kwargs)

        async def async_list_terminal_payloads_for_eviction(self, **kwargs):
            return store.list_terminal_payloads_for_eviction(**kwargs)

        async def async_mark_payload_evicted(self, **kwargs):
            return store.mark_payload_evicted(**kwargs)

        async def async_delete_expired_tombstones(self, **kwargs):
            return store.delete_expired_tombstones(**kwargs)

        async def async_record_payload_evict_error(self, **kwargs):
            return store.record_payload_evict_error(**kwargs)

        async def async_list_staged_payloads_for_gc(self, **kwargs):
            return store.list_staged_payloads_for_gc(**kwargs)

        async def async_mark_staged_payload_gc_deleted(self, **kwargs):
            return store.mark_staged_payload_gc_deleted(**kwargs)

    try:
        store.ensure_task(
            request_id="req-stage-gc",
            op="sampling.asample",
            domain_key="vllm:test",
            request_json=b"{}",
            status="pending",
            now=1.0,
        )
        payloads = TaskPayloadStore(tmp_path)
        staged_meta = payloads.write_json_payload(
            request_id="req-stage-gc",
            attempt_id="future__old-stage",
            payload={"stale": True},
        )
        store.stage_payload(
            request_id="req-stage-gc",
            staged_payload_path=str(staged_meta["path"]),
            now=2.0,
        )
        store.complete_task_failure(
            request_id="req-stage-gc",
            error="failed after staging",
            now=3.0,
        )
        monkeypatch.setattr(config_module.config, "task_pending_ttl_s", 86400.0, raising=False)
        monkeypatch.setattr(config_module.config, "task_result_ttl_s", 1.0, raising=False)
        monkeypatch.setattr(config_module.config, "task_tombstone_ttl_s", 10**12, raising=False)

        out = asyncio.run(
            TaskFutureService(
                task_state_client=_LocalTaskStateClient(),
                future_state_client=_LocalTaskStateClient(),
                payload_store=payloads,
            ).async_reap()
        )

        assert out["staged_payload_gc_deleted"] == ["req-stage-gc"]
        assert not Path(staged_meta["path"]).exists()
        assert store.get_task("req-stage-gc")["metadata"]["abandoned_staged_payload_paths"] == []
    finally:
        store.close()
