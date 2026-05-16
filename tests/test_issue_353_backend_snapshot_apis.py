from __future__ import annotations

import asyncio
import sys
import time

import tinker_server.backend.resource_pool as resource_pool_mod
from tinker_server.backend.api_work_queue import ApiWorkQueueClient
from tinker_server.backend.task_state_store import FutureStatus, TaskStateFutureStore
from tinker_server.backend.resource_pool import ActorType, get_resource_pool


def _local_resource_pool(monkeypatch):
    monkeypatch.setattr(resource_pool_mod, "_detached_enabled", lambda: False)
    monkeypatch.setattr(resource_pool_mod.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(resource_pool_mod.ResourcePool, "_instance", None)
    return get_resource_pool()


def test_actor_observability_metadata_preserves_recent_latency_and_gpu_fields(monkeypatch) -> None:
    class _Getter:
        def __call__(self):
            return None

        def remote(self):
            return "binding-ref"

    class _Handle:
        get_observability_binding = _Getter()

    monkeypatch.setattr(
        resource_pool_mod.ray,
        "get",
        lambda _ref, timeout=None: {
            "hostname": "host-a",
            "node_id": "node-a",
            "scheduler_waiting_requests": 4,
            "seq_slot_wait_s_p95_recent": 1.2,
            "generate_lock_wait_s_p50_recent": 0.1,
            "time_per_output_token_s_total": 0.96,
            "gpu_memory_allocated_bytes": 48000000000,
            "gpu_memory_reserved_bytes": 52000000000,
            "gpu_memory_fragmentation_bytes": 4000000000,
        },
    )

    out = resource_pool_mod.actor_observability_metadata(_Handle(), timeout_s=0.5)

    assert out == {
        "hostname": "host-a",
        "node_id": "node-a",
        "scheduler_waiting_requests": 4,
        "seq_slot_wait_s_p95_recent": 1.2,
        "generate_lock_wait_s_p50_recent": 0.1,
        "time_per_output_token_s_total": 0.96,
        "gpu_memory_allocated_bytes": 48000000000,
        "gpu_memory_reserved_bytes": 52000000000,
        "gpu_memory_fragmentation_bytes": 4000000000,
    }


def test_async_actor_observability_metadata_preserves_recent_latency_and_gpu_fields(monkeypatch) -> None:
    class _Getter:
        def __call__(self):
            return None

        def remote(self):
            return "binding-ref"

    class _Handle:
        get_observability_binding = _Getter()

    async def _async_get_ray_ref(ref, *, timeout_s=None):
        assert ref == "binding-ref"
        assert timeout_s == 0.5
        return {
            "hostname": "host-a",
            "node_id": "node-a",
            "scheduler_waiting_requests": 4,
            "seq_slot_wait_s_p95_recent": 1.2,
            "generate_lock_wait_s_p50_recent": 0.1,
            "time_per_output_token_s_total": 0.96,
            "gpu_memory_allocated_bytes": 48000000000,
            "gpu_memory_reserved_bytes": 52000000000,
            "gpu_memory_fragmentation_bytes": 4000000000,
        }

    monkeypatch.setattr(resource_pool_mod, "async_get_ray_ref", _async_get_ray_ref)

    out = asyncio.run(resource_pool_mod.async_actor_observability_metadata(_Handle(), timeout_s=0.5))

    assert out == {
        "hostname": "host-a",
        "node_id": "node-a",
        "scheduler_waiting_requests": 4,
        "seq_slot_wait_s_p95_recent": 1.2,
        "generate_lock_wait_s_p50_recent": 0.1,
        "time_per_output_token_s_total": 0.96,
        "gpu_memory_allocated_bytes": 48000000000,
        "gpu_memory_reserved_bytes": 52000000000,
        "gpu_memory_fragmentation_bytes": 4000000000,
    }


def test_api_work_queue_metrics_snapshot_tracks_local_state() -> None:
    q = ApiWorkQueueClient()
    now = time.time()

    q._snapshot_on_enqueue(
        {
            "request_id": "req-1",
            "op": "weights.save_weights",
            "created_at": now - 9.0,
        }
    )
    q._snapshot_on_enqueue(
        {
            "request_id": "req-2",
            "op": "sampling.asample",
            "throttle_principal": "user:a",
            "apikey_id": "key-a",
            "created_at": now - 3.0,
        }
    )

    snap = q.metrics_snapshot()
    assert snap["depth"] == 2
    assert snap["enqueued"] == 2
    assert snap["dequeued"] == 0
    assert snap["by_executor"] == {"weights.save_weights": 1, "sampling.asample": 1}
    assert snap["by_throttle_principal"] == {"user:a": 1}
    assert snap["by_apikey_id"] == {"key-a": 1}
    assert snap["age_stats"]["oldest_queued_s"] >= snap["age_stats"]["avg_queued_s"]
    assert snap["scheduler_metrics_ready"] is False
    assert "depth_scheduled" not in snap
    assert "scheduler_enabled" not in snap

    q._snapshot_on_dequeue({"request_id": "req-2"})
    snap2 = q.metrics_snapshot()
    assert snap2["depth"] == 1
    assert snap2["enqueued"] == 2
    assert snap2["dequeued"] == 1
    assert snap2["by_executor"] == {"weights.save_weights": 1}
    assert snap2["by_throttle_principal"] == {}
    assert snap2["by_apikey_id"] == {}


class _InlineTaskStateClient:
    def __init__(self, *, stats: dict | None = None) -> None:
        self.records: dict[str, dict] = {}
        self.stats_payload = stats or {"backend": "task_state_store", "records": 0}
        self.started = 0

    async def async_ensure_task(self, **kwargs):
        request_id = str(kwargs["request_id"])
        created = request_id not in self.records
        record = self.records.setdefault(
            request_id,
            {
                "request_id": request_id,
                "status": kwargs.get("status") or "pending",
                "op": kwargs.get("op") or "unknown",
                "domain_key": kwargs.get("domain_key") or "future:default",
                "metadata": dict(kwargs.get("metadata") or {}),
            },
        )
        if kwargs.get("metadata") is not None:
            record["metadata"] = dict(kwargs["metadata"])
        if kwargs.get("op") is not None:
            record["op"] = kwargs["op"]
        if kwargs.get("domain_key") is not None:
            record["domain_key"] = kwargs["domain_key"]
        if kwargs.get("status") is not None:
            record["status"] = kwargs["status"]
        return {"created": created, "record": dict(record)}

    async def async_get_task(self, request_id: str):
        return dict(self.records[str(request_id)])

    async def async_complete_task_success(self, **kwargs):
        record = self.records[str(kwargs["request_id"])]
        record.update(
            {
                "status": "done",
                "result_path": kwargs["result_path"],
                "result_checksum": kwargs["result_checksum"],
                "result_size_bytes": kwargs["result_size_bytes"],
                "metadata": {**record.get("metadata", {}), **dict(kwargs.get("metadata") or {})},
            }
        )
        return dict(record)

    async def async_mark_task_retrieved(self, **kwargs):
        record = self.records[str(kwargs["request_id"])]
        record["status"] = "retrieved"
        return dict(record)

    async def async_stats(self):
        return dict(self.stats_payload)

    async def async_ensure_started(self):
        self.started += 1


class _InlinePayloadStore:
    def __init__(self) -> None:
        self.payloads: dict[str, object] = {}

    def write_json_payload(self, *, request_id: str, attempt_id: str, payload):
        _ = attempt_id
        path = f"inline://{request_id}"
        self.payloads[path] = payload
        return {"path": path, "checksum": f"sha256:{request_id}", "size_bytes": 1}

    def read_json_payload(self, *, path: str, expected_checksum: str | None = None):
        _ = expected_checksum
        return self.payloads[path]


def test_task_state_future_store_metrics_snapshot_identifies_backend() -> None:
    assert TaskStateFutureStore().metrics_snapshot() == {"backend": "task_state_store"}


def test_task_state_future_store_round_trips_result_payload() -> None:
    task_state = _InlineTaskStateClient()
    payload_store = _InlinePayloadStore()
    fs = TaskStateFutureStore(task_state_client=task_state, payload_store=payload_store)

    async def _run():
        assert await fs.async_ensure_pending("req-done", meta={"op": "sampling.asample"}) == {
            "created": True,
            "meta": {"op": "sampling.asample"},
        }
        await fs.async_resolve("req-done", {"ok": True})
        assert await fs.async_get_status("req-done") is FutureStatus.DONE
        assert await fs.async_get_result("req-done") == {"ok": True}

    asyncio.run(_run())
    assert task_state.records["req-done"]["status"] == "retrieved"


def test_task_state_future_store_ensure_pending_syncs_existing_pending_without_meta() -> None:
    task_state = _InlineTaskStateClient()
    task_state.records["req-existing"] = {
        "request_id": "req-existing",
        "status": "pending",
        "op": "unknown",
        "domain_key": "future:default",
        "metadata": {},
    }
    fs = TaskStateFutureStore(task_state_client=task_state, payload_store=_InlinePayloadStore())

    out = asyncio.run(fs.async_ensure_pending("req-existing", meta=None))

    assert out == {"created": False, "meta": {}}


def test_api_work_queue_hydrate_metrics_snapshot_restores_restart_baseline(monkeypatch) -> None:
    class _StubRay:
        @staticmethod
        def get(ref, timeout=None):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self):
            async def _result():
                return self._fn()

            return _result()

    class _StubActor:
        def __init__(self):
            self.metrics_seed_snapshot = _StubMethod(
                lambda: {
                    "stats": {"enqueued": 9, "dequeued": 4},
                    "queued_items": [
                        {
                            "request_id": "rid-q-1",
                            "op": "sampling.asample",
                            "throttle_principal": "user:a",
                            "apikey_id": "key-a",
                            "created_at": time.time() - 7.0,
                        },
                        {
                            "request_id": "rid-q-2",
                            "op": "weights.save_weights",
                            "created_at": time.time() - 3.0,
                        },
                    ],
                }
            )

    monkeypatch.setitem(sys.modules, "ray", _StubRay)

    q = ApiWorkQueueClient()
    monkeypatch.setattr(q, "_get_ray_actor", lambda: _StubActor())

    assert q.hydrate_metrics_snapshot(force=True)
    snap = q.metrics_snapshot()
    assert snap["depth"] == 2
    assert snap["enqueued"] == 9
    assert snap["dequeued"] == 4
    assert snap["by_executor"] == {"sampling.asample": 1, "weights.save_weights": 1}
    assert snap["by_throttle_principal"] == {"user:a": 1}
    assert snap["by_apikey_id"] == {"key-a": 1}
    assert snap["age_stats"]["oldest_queued_s"] >= 7.0


def test_api_work_queue_scheduler_decisions_client_proxies_filters() -> None:
    class _StubSchedulerDecisions:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def remote(self, **kwargs):
            self.calls.append(dict(kwargs))
            return {
                "actor_name": "tinker_api_work_queue",
                "last_seq": 9,
                "items": [],
                "scheduler": {"enabled": True},
            }

    class _StubActor:
        def __init__(self):
            self.scheduler_decisions = _StubSchedulerDecisions()

    actor = _StubActor()
    client = ApiWorkQueueClient()

    async def _get_ray_actor_async():
        return actor

    async def _await_ray_ref(ref, *, timeout_s=None):
        return ref

    client._get_ray_actor_async = _get_ray_actor_async  # type: ignore[method-assign]
    client._await_ray_ref = _await_ray_ref  # type: ignore[method-assign]

    payload = asyncio.run(
        client.scheduler_decisions(
            limit=25,
            scheduler_domain="megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
            reason="sticky",
            since_seq=7,
            timeout_s=3.0,
        )
    )

    assert actor.scheduler_decisions.calls == [
        {
            "limit": 25,
            "scheduler_domain": "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
            "reason": "sticky",
            "since_seq": 7,
        }
    ]
    assert payload["last_seq"] == 9


def test_task_state_future_store_ensure_ready_returns_task_state_stats() -> None:
    task_state = _InlineTaskStateClient(stats={"backend": "task_state_store", "active": 3})
    fs = TaskStateFutureStore(task_state_client=task_state, payload_store=_InlinePayloadStore())

    assert asyncio.run(fs.async_ensure_ready()) == {"backend": "task_state_store", "active": 3}


def test_task_state_future_store_ensure_started_delegates_to_task_state_client() -> None:
    task_state = _InlineTaskStateClient()
    fs = TaskStateFutureStore(task_state_client=task_state, payload_store=_InlinePayloadStore())

    asyncio.run(fs.async_ensure_started())

    assert task_state.started == 1


def test_api_work_queue_start_workers_continues_when_hydration_baseline_missing(monkeypatch) -> None:
    class _StubRuntimeContext:
        def get_job_id(self):
            return "job-1"

    class _StubRay:
        @staticmethod
        def get_runtime_context():
            return _StubRuntimeContext()

        @staticmethod
        def get(ref, timeout=None):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return lambda: self._fn(*args, **kwargs)

    class _StubActor:
        def __init__(self):
            self.set_active_job_id = _StubMethod(lambda job_id: None)

    monkeypatch.setitem(sys.modules, "ray", _StubRay)

    q = ApiWorkQueueClient()
    monkeypatch.setattr(q, "_get_ray_actor", lambda: _StubActor())
    monkeypatch.setenv("MINT_API_WORK_QUEUE_METRICS_HYDRATE_STARTUP_RETRIES", "3")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_METRICS_HYDRATE_RETRY_DELAY_S", "0")

    attempts = {"count": 0}

    def _always_fail_hydrate(**kwargs) -> bool:
        attempts["count"] += 1
        return False

    monkeypatch.setattr(q, "hydrate_metrics_snapshot", _always_fail_hydrate)

    async def _noop_queue_supervisor_loop():
        return None

    q._queue_supervisor_loop = _noop_queue_supervisor_loop  # type: ignore[method-assign]

    asyncio.run(q.start_workers(num_workers=1))

    assert attempts["count"] == 3
    assert q._running is True
    assert q._consumer_job_id is None
    assert q._consumer_generation_id is None

    asyncio.run(q.shutdown())


def test_resource_pool_cached_snapshot_exposes_rss_cache_state(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.RSS_TTL_S

    try:
        pool.RSS_TTL_S = 10.0
        pool.register("actor-fresh", ActorType.VLLM, 1, actor_handle=None, base_model="m")
        pool.register("actor-stale", ActorType.VLLM, 1, actor_handle=None, base_model="m")
        pool.register("actor-unknown", ActorType.VLLM, 1, actor_handle=None, base_model="m")

        now = time.time()
        with pool._pool_lock:
            fresh = pool._entries["actor-fresh"]
            fresh.rss_bytes = 111
            fresh.rss_sample_time = now - 2.0
            fresh.rss_sample_source = "touch"

            stale = pool._entries["actor-stale"]
            stale.rss_bytes = 222
            stale.rss_sample_time = now - 90.0
            stale.rss_sample_source = "mark_ready"

        snapshot = pool.cached_snapshot()
        by_name = {rec["actor_name"]: rec for rec in snapshot}

        assert by_name["actor-fresh"]["rss_cache_state"] == "fresh"
        assert by_name["actor-fresh"]["rss_bytes"] == 111
        assert by_name["actor-fresh"]["rss_sample_source"] == "touch"

        assert by_name["actor-stale"]["rss_cache_state"] == "stale"
        assert "rss_bytes" not in by_name["actor-stale"]
        assert by_name["actor-stale"]["rss_sample_source"] == "mark_ready"
        assert by_name["actor-stale"]["rss_sample_age_s"] >= 80.0

        assert by_name["actor-unknown"]["rss_cache_state"] == "unknown"
        assert "rss_bytes" not in by_name["actor-unknown"]
        assert "rss_sample_age_s" not in by_name["actor-unknown"]
    finally:
        pool.RSS_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_cached_snapshot_refreshes_vllm_observability_on_ttl(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S
    old_timeout = pool.METADATA_TIMEOUT_S
    calls: list[tuple[object, float]] = []

    def _stub_observability(handle, *, timeout_s=5.0):
        calls.append((handle, float(timeout_s)))
        return {
            "scheduler_waiting_requests": 3,
            "scheduler_running_requests": 1,
            "time_per_output_token_s_total": 0.4,
        }

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", _stub_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.METADATA_TIMEOUT_S = 0.25
        handle = object()
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=handle, base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {"scheduler_waiting_requests": 99}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        first = {rec["actor_name"]: rec for rec in pool.cached_snapshot()}["actor-vllm"]
        assert first["metadata"]["scheduler_waiting_requests"] == 3
        assert first["metadata"]["scheduler_running_requests"] == 1
        assert first["metadata"]["time_per_output_token_s_total"] == 0.4
        assert calls == [(handle, 0.25)]

        second = {rec["actor_name"]: rec for rec in pool.cached_snapshot()}["actor-vllm"]
        assert second["metadata"]["scheduler_waiting_requests"] == 3
        assert len(calls) == 1

        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            assert entry.metadata_sample_source == "cached_snapshot"
            assert entry.metadata_sample_time is not None
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.METADATA_TIMEOUT_S = old_timeout
        pool.clear(kill_actors=False)


def test_resource_pool_list_actors_can_refresh_vllm_observability(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S
    calls: list[object] = []

    def _stub_observability(handle, *, timeout_s=5.0):
        calls.append(handle)
        return {
            "scheduler_waiting_requests": 7,
            "scheduler_running_requests": 2,
            "scheduler_kv_cache_usage_ratio": 0.5,
        }

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", _stub_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        handle = object()
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=handle, base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {"scheduler_waiting_requests": 0}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        rec = {item["actor_name"]: item for item in pool.list_actors(refresh_metadata=True)}["actor-vllm"]

        assert rec["metadata"]["scheduler_waiting_requests"] == 7
        assert rec["metadata"]["scheduler_running_requests"] == 2
        assert rec["metadata"]["scheduler_kv_cache_usage_ratio"] == 0.5
        assert rec["metadata_sample_source"] == "list_actors"
        assert rec["metadata_cache_state"] == "fresh"
        assert len(calls) == 1
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_async_list_actors_can_refresh_vllm_observability(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S
    calls: list[object] = []

    async def _stub_observability(handle, *, timeout_s=5.0):
        calls.append(handle)
        return {
            "scheduler_waiting_requests": 7,
            "scheduler_running_requests": 2,
            "scheduler_kv_cache_usage_ratio": 0.5,
        }

    monkeypatch.setattr(resource_pool_mod, "async_actor_observability_metadata", _stub_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        handle = object()
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=handle, base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {"scheduler_waiting_requests": 0}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        rec = {item["actor_name"]: item for item in asyncio.run(pool.async_list_actors(refresh_metadata=True))}[
            "actor-vllm"
        ]

        assert rec["metadata"]["scheduler_waiting_requests"] == 7
        assert rec["metadata"]["scheduler_running_requests"] == 2
        assert rec["metadata"]["scheduler_kv_cache_usage_ratio"] == 0.5
        assert rec["metadata_sample_source"] == "list_actors"
        assert rec["metadata_cache_state"] == "fresh"
        assert len(calls) == 1
        assert asyncio.run(pool.async_total_gpus_used()) == 1
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_async_list_actors_refreshes_metadata_with_bounded_parallelism(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S
    old_concurrency = pool.METADATA_REFRESH_CONCURRENCY
    active = 0
    max_active = 0

    class _Handle:
        def __init__(self, value: int) -> None:
            self.value = value

    async def _stub_observability(handle, *, timeout_s=5.0):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            return {"scheduler_waiting_requests": handle.value}
        finally:
            active -= 1

    monkeypatch.setattr(resource_pool_mod, "async_actor_observability_metadata", _stub_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.METADATA_REFRESH_CONCURRENCY = 2
        for idx in range(4):
            name = f"actor-vllm-{idx}"
            pool.register(name, ActorType.VLLM, 1, actor_handle=_Handle(idx), base_model="m")
            with pool._pool_lock:
                entry = pool._entries[name]
                entry.metadata = {"scheduler_waiting_requests": -1}
                entry.metadata_sample_time = time.time() - 120.0
                entry.metadata_sample_source = "stale"

        records = {item["actor_name"]: item for item in asyncio.run(pool.async_list_actors(refresh_metadata=True))}

        assert max_active == 2
        for idx in range(4):
            rec = records[f"actor-vllm-{idx}"]
            assert rec["metadata"]["scheduler_waiting_requests"] == idx
            assert rec["metadata_sample_source"] == "list_actors"
            assert rec["metadata_cache_state"] == "fresh"
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.METADATA_REFRESH_CONCURRENCY = old_concurrency
        pool.clear(kill_actors=False)


def test_resource_pool_async_list_actors_refreshes_detached_inventory(monkeypatch) -> None:
    monkeypatch.setattr(resource_pool_mod, "_detached_enabled", lambda: True)
    monkeypatch.setattr(resource_pool_mod.ResourcePool, "_instance", None)
    pool = get_resource_pool()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    handle = object()
    stale_sample_time = time.time() - 120.0

    async def _call_actor_async(method_name: str, *args, retry_on_actor_restart: bool = False, **kwargs):
        calls.append((method_name, args, dict(kwargs)))
        if method_name == "list_entries":
            assert args == ()
            assert kwargs == {"prune_stale": True}
            assert retry_on_actor_restart is True
            return [
                {
                    "actor_name": "actor-vllm",
                    "actor_type": ActorType.VLLM.value,
                    "num_gpus": 1,
                    "namespace": "ns",
                    "base_model": "m",
                    "metadata": {"scheduler_waiting_requests": 0},
                    "metadata_sample_time": stale_sample_time,
                    "metadata_sample_source": "stale",
                }
            ]
        if method_name == "update_metadata":
            assert args[0] == "actor-vllm"
            assert args[1] == {
                "scheduler_waiting_requests": 7,
                "scheduler_running_requests": 2,
            }
            assert args[3] == "list_actors"
            assert retry_on_actor_restart is True
            return True
        if method_name == "total_gpus_used":
            assert args == ()
            assert retry_on_actor_restart is True
            return 3
        raise AssertionError(f"unexpected ResourcePool actor method: {method_name}")

    async def _lookup_handle_async(self, actor_name: str, namespace: str):
        assert actor_name == "actor-vllm"
        assert namespace == "ns"
        return handle

    async def _observability(actor_handle, *, timeout_s=5.0):
        assert actor_handle is handle
        return {
            "scheduler_waiting_requests": 7,
            "scheduler_running_requests": 2,
        }

    monkeypatch.setattr(resource_pool_mod, "_call_actor_async", _call_actor_async)
    monkeypatch.setattr(resource_pool_mod.ResourcePool, "_lookup_handle_async", _lookup_handle_async)
    monkeypatch.setattr(resource_pool_mod, "async_actor_observability_metadata", _observability)

    rec = {item["actor_name"]: item for item in asyncio.run(pool.async_list_actors(refresh_metadata=True))}[
        "actor-vllm"
    ]

    assert rec["metadata"] == {
        "scheduler_waiting_requests": 7,
        "scheduler_running_requests": 2,
    }
    assert rec["metadata_sample_source"] == "list_actors"
    assert rec["metadata_cache_state"] == "fresh"
    assert asyncio.run(pool.async_total_gpus_used()) == 3
    assert [method_name for method_name, _args, _kwargs in calls] == [
        "list_entries",
        "update_metadata",
        "total_gpus_used",
    ]


def test_resource_pool_detached_gpu_usage_lists_entries_with_keyword_prune(monkeypatch) -> None:
    monkeypatch.setattr(resource_pool_mod, "_detached_enabled", lambda: True)
    monkeypatch.setattr(resource_pool_mod.ResourcePool, "_instance", None)
    pool = get_resource_pool()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _call_actor_sync(method_name: str, *args, retry_on_actor_restart: bool = False, **kwargs):
        calls.append((method_name, args, dict(kwargs)))
        if method_name == "list_entries":
            assert args == ()
            assert kwargs == {"prune_stale": False}
            assert retry_on_actor_restart is True
            return [
                {
                    "actor_name": "actor-vllm",
                    "actor_type": ActorType.VLLM.value,
                    "num_gpus": 4,
                    "namespace": "ns",
                    "base_model": "m",
                    "node_id": "node-1",
                }
            ]
        raise AssertionError(f"unexpected ResourcePool actor method: {method_name}")

    monkeypatch.setattr(resource_pool_mod, "_call_actor_sync", _call_actor_sync)

    assert pool.gpus_used_by_node() == {"node-1": 4}
    assert calls == [("list_entries", (), {"prune_stale": False})]


def test_resource_pool_list_actors_skips_refresh_when_requested(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S

    def _fail_observability(_handle, *, timeout_s=5.0):
        raise AssertionError("list_actors(refresh_metadata=False) must not refresh metadata")

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", _fail_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=object(), base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {"scheduler_waiting_requests": 0}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        rec = {item["actor_name"]: item for item in pool.list_actors(refresh_metadata=False)}["actor-vllm"]

        assert rec["metadata"]["scheduler_waiting_requests"] == 0
        assert rec["metadata_sample_source"] == "stale"
        assert rec["metadata_cache_state"] == "stale"
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_rss_snapshot_preserves_cached_metadata_when_collecting_rss(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S

    class _Method:
        def remote(self):
            return "rss-ref"

    class _Handle:
        get_rss_bytes = _Method()

    monkeypatch.setattr(resource_pool_mod.ray, "get", lambda _ref, timeout=None: 4096)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=_Handle(), base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {
                "scheduler_waiting_requests": 4,
                "scheduler_running_requests": 1,
            }
            entry.metadata_sample_time = time.time() - 5.0
            entry.metadata_sample_source = "cached_snapshot"

        rec = {item["actor_name"]: item for item in pool.rss_snapshot(timeout_s=0.1)}["actor-vllm"]

        assert rec["rss_bytes"] == 4096
        assert rec["metadata"]["scheduler_waiting_requests"] == 4
        assert rec["metadata"]["scheduler_running_requests"] == 1
        assert rec["metadata_sample_source"] == "cached_snapshot"
        assert rec["metadata_cache_state"] == "fresh"
        assert rec["metadata_sample_age_s"] >= 0.0
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_cached_snapshot_refreshes_megatron_observability_on_ttl(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S
    old_timeout = pool.METADATA_TIMEOUT_S
    calls: list[tuple[object, float]] = []

    def _stub_observability(handle, *, timeout_s=5.0):
        calls.append((handle, float(timeout_s)))
        return {
            "active_sessions": 1,
            "session_step": 17,
            "learning_rate": 5e-5,
        }

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", _stub_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.METADATA_TIMEOUT_S = 0.25
        handle = object()
        pool.register("actor-megatron", ActorType.MEGATRON, 8, actor_handle=handle, base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-megatron"]
            entry.metadata = {"active_sessions": 9}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        first = {rec["actor_name"]: rec for rec in pool.cached_snapshot()}["actor-megatron"]
        assert first["metadata"]["active_sessions"] == 1
        assert first["metadata"]["session_step"] == 17
        assert first["metadata"]["learning_rate"] == 5e-5
        assert calls == [(handle, 0.25)]

        second = {rec["actor_name"]: rec for rec in pool.cached_snapshot()}["actor-megatron"]
        assert second["metadata"]["active_sessions"] == 1
        assert len(calls) == 1

        with pool._pool_lock:
            entry = pool._entries["actor-megatron"]
            assert entry.metadata_sample_source == "cached_snapshot"
            assert entry.metadata_sample_time is not None
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.METADATA_TIMEOUT_S = old_timeout
        pool.clear(kill_actors=False)


def test_resource_pool_cached_snapshot_uses_fresh_metadata_without_refresh(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S

    def _fail_observability(_handle, *, timeout_s=5.0):
        raise AssertionError("fresh metadata must not trigger refresh")

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", _fail_observability)

    try:
        pool.METADATA_TTL_S = 30.0
        before = {row["actor_type"]: row for row in pool.metadata_cache_metrics_snapshot()}
        before_vllm = before.get("vllm", {})
        pool.register("actor-vllm", ActorType.VLLM, 1, actor_handle=object(), base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-vllm"]
            entry.metadata = {"scheduler_waiting_requests": 2}
            entry.metadata_sample_time = time.time()
            entry.metadata_sample_source = "register"

        rec = {item["actor_name"]: item for item in pool.cached_snapshot()}["actor-vllm"]
        assert rec["metadata"]["scheduler_waiting_requests"] == 2
        stats = {row["actor_type"]: row for row in pool.metadata_cache_metrics_snapshot()}
        assert stats["vllm"]["cache_hits_total"] >= int(before_vllm.get("cache_hits_total", 0)) + 1
        assert stats["vllm"]["refresh_failures_total"] == int(before_vllm.get("refresh_failures_total", 0))
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)


def test_resource_pool_cached_snapshot_tracks_refresh_failure(monkeypatch) -> None:
    pool = _local_resource_pool(monkeypatch)
    pool.clear(kill_actors=False)
    old_ttl = pool.METADATA_TTL_S

    monkeypatch.setattr(resource_pool_mod, "actor_observability_metadata", lambda *_args, **_kwargs: None)

    try:
        pool.METADATA_TTL_S = 30.0
        pool.register("actor-megatron", ActorType.MEGATRON, 8, actor_handle=object(), base_model="m")
        with pool._pool_lock:
            entry = pool._entries["actor-megatron"]
            entry.metadata = {"active_sessions": 3}
            entry.metadata_sample_time = time.time() - 120.0
            entry.metadata_sample_source = "stale"

        rec = {item["actor_name"]: item for item in pool.cached_snapshot()}["actor-megatron"]
        assert rec["metadata"]["active_sessions"] == 3
        stats = {row["actor_type"]: row for row in pool.metadata_cache_metrics_snapshot()}
        assert stats["megatron"]["cache_stale_total"] >= 1
        assert stats["megatron"]["refresh_failures_total"] >= 1
    finally:
        pool.METADATA_TTL_S = old_ttl
        pool.clear(kill_actors=False)
