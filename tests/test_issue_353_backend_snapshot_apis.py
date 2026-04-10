from __future__ import annotations

import asyncio
import sys
import time

import tinker_server.backend.resource_pool as resource_pool_mod
from tinker_server.backend.api_work_queue import ApiWorkQueueClient
from tinker_server.backend.future_store import FutureStatus, FutureStore, FutureStoreUnavailableError
from tinker_server.backend.resource_pool import ActorType, get_resource_pool


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


def test_future_store_metrics_snapshot_tracks_local_state() -> None:
    fs = FutureStore()

    fs._snapshot_ensure_pending("req-pending", meta={"op": "sampling.asample"}, has_ref=True)
    fs._snapshot_ensure_pending("req-done", meta={"op": "sampling.asample"}, has_ref=True)
    fs._snapshot_ensure_pending("req-failed", meta={"op": "weights.save_weights"}, has_ref=False)
    fs._snapshot_mark_terminal("req-done", status=FutureStatus.DONE.value)
    fs._snapshot_mark_terminal("req-failed", status=FutureStatus.FAILED.value)
    fs._snapshot_mark_terminal("req-expired", status=FutureStatus.EXPIRED.value)
    fs._snapshot_mark_terminal("req-retrieved", status=FutureStatus.RETRIEVED.value)

    snap = fs.metrics_snapshot()

    assert snap["pending"] == 1
    assert snap["results"] == 1
    assert snap["errors"] == 1
    assert snap["expired"] == 1
    assert snap["retrieved"] == 1
    assert snap["refs"] == 1
    assert snap["meta"] == 3

    assert snap["by_op"]["sampling.asample"]["pending"] == 1
    assert snap["by_op"]["sampling.asample"]["results"] == 1
    assert snap["by_op"]["weights.save_weights"]["errors"] == 1

    assert snap["age_stats"]["oldest_pending_s"] >= 0.0
    assert snap["age_stats"]["oldest_done_s"] >= 0.0
    assert snap["payload_stats"]["result_refs_count"] == 1
    assert snap["payload_stats"]["errors_count"] == 1
    assert snap["payload_stats"]["refs_count"] == 1


def test_future_store_payload_stats_use_backing_store_semantics() -> None:
    fs = FutureStore()
    now = time.time()
    with fs._snapshot_lock:
        fs._snapshot_requests = {
            "req-done-with-payload": {
                "status": FutureStatus.DONE.value,
                "created_at": now,
                "done_at": now,
                "op": "op.a",
                "has_meta": False,
                "has_ref": False,
                "has_result_ref": True,
                "has_error": False,
            },
            "req-done-no-payload": {
                "status": FutureStatus.DONE.value,
                "created_at": now,
                "done_at": now,
                "op": "op.a",
                "has_meta": False,
                "has_ref": False,
                "has_result_ref": False,
                "has_error": False,
            },
            "req-failed-with-error": {
                "status": FutureStatus.FAILED.value,
                "created_at": now,
                "done_at": now,
                "op": "op.b",
                "has_meta": False,
                "has_ref": False,
                "has_result_ref": False,
                "has_error": True,
            },
            "req-failed-no-error": {
                "status": FutureStatus.FAILED.value,
                "created_at": now,
                "done_at": now,
                "op": "op.b",
                "has_meta": False,
                "has_ref": True,
                "has_result_ref": False,
                "has_error": False,
            },
        }

    snap = fs.metrics_snapshot()

    assert snap["results"] == 2
    assert snap["errors"] == 2
    assert snap["refs"] == 1
    assert snap["payload_stats"]["result_refs_count"] == 1
    assert snap["payload_stats"]["errors_count"] == 1
    assert snap["payload_stats"]["refs_count"] == 1


def test_future_store_ensure_pending_syncs_existing_pending_without_meta(monkeypatch) -> None:
    class _StubRayExceptions:
        class ActorDiedError(Exception):
            pass

    class _StubRay:
        exceptions = _StubRayExceptions

        @staticmethod
        def get(ref):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, **kwargs):
            return lambda: self._fn(**kwargs)

    class _StubActor:
        def __init__(self):
            self.ensure_pending = _StubMethod(lambda request_id, meta: {"created": False, "meta": None})
            self.get_status = _StubMethod(lambda request_id: FutureStatus.PENDING.value)

    monkeypatch.setitem(sys.modules, "ray", _StubRay)

    fs = FutureStore()
    monkeypatch.setattr(fs, "_get_ray_actor", lambda: _StubActor())

    out = fs.ensure_pending("req-existing", meta=None)

    assert out == {"created": False, "meta": None}
    snap = fs.metrics_snapshot()
    assert snap["pending"] == 1
    assert snap["by_op"]["unknown"]["pending"] == 1


def test_api_work_queue_hydrate_metrics_snapshot_restores_restart_baseline(monkeypatch) -> None:
    class _StubRay:
        @staticmethod
        def get(ref, timeout=None):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self):
            return lambda: self._fn()

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


def test_future_store_hydrate_metrics_snapshot_restores_restart_baseline(monkeypatch) -> None:
    class _StubRay:
        @staticmethod
        def get(ref, timeout=None):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self):
            return lambda: self._fn()

    class _StubActor:
        def __init__(self):
            self.metrics_seed_snapshot = _StubMethod(
                lambda: {
                    "requests": [
                        {
                            "request_id": "rid-fs-1",
                            "status": FutureStatus.PENDING.value,
                            "created_at": time.time() - 5.0,
                            "done_at": None,
                            "op": "sampling.asample",
                            "has_meta": True,
                            "has_ref": True,
                            "has_result_ref": False,
                            "has_error": False,
                        },
                        {
                            "request_id": "rid-fs-2",
                            "status": FutureStatus.DONE.value,
                            "created_at": time.time() - 12.0,
                            "done_at": time.time() - 2.0,
                            "op": "sampling.asample",
                            "has_meta": True,
                            "has_ref": False,
                            "has_result_ref": True,
                            "has_error": False,
                        },
                        {
                            "request_id": "rid-fs-3",
                            "status": FutureStatus.FAILED.value,
                            "created_at": time.time() - 10.0,
                            "done_at": time.time() - 1.0,
                            "op": "weights.save_weights",
                            "has_meta": False,
                            "has_ref": False,
                            "has_result_ref": False,
                            "has_error": True,
                        },
                    ]
                }
            )

    monkeypatch.setitem(sys.modules, "ray", _StubRay)

    fs = FutureStore()
    monkeypatch.setattr(fs, "_get_ray_actor", lambda: _StubActor())

    assert fs.hydrate_metrics_snapshot(force=True)
    snap = fs.metrics_snapshot()
    assert snap["pending"] == 1
    assert snap["results"] == 1
    assert snap["errors"] == 1
    assert snap["refs"] == 1
    assert snap["meta"] == 2
    assert snap["payload_stats"]["result_refs_count"] == 1
    assert snap["payload_stats"]["errors_count"] == 1
    assert snap["by_op"]["sampling.asample"]["pending"] == 1
    assert snap["by_op"]["sampling.asample"]["results"] == 1
    assert snap["by_op"]["weights.save_weights"]["errors"] == 1


def test_future_store_ensure_ready_fails_when_hydration_baseline_required(monkeypatch) -> None:
    class _StubRay:
        @staticmethod
        def get(ref, timeout=None):
            return ref() if callable(ref) else ref

    class _StubMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self):
            return lambda: self._fn()

    class _StubActor:
        def __init__(self):
            self.stats = _StubMethod(lambda: {"pending": 3})

    monkeypatch.setitem(sys.modules, "ray", _StubRay)

    fs = FutureStore()
    monkeypatch.setattr(fs, "_get_ray_actor", lambda: _StubActor())
    monkeypatch.setenv("MINT_FUTURE_STORE_METRICS_HYDRATE_STARTUP_RETRIES", "3")
    monkeypatch.setenv("MINT_FUTURE_STORE_METRICS_HYDRATE_RETRY_DELAY_S", "0")

    attempts = {"count": 0}

    def _always_fail_hydrate(**kwargs) -> bool:
        attempts["count"] += 1
        return False

    monkeypatch.setattr(fs, "hydrate_metrics_snapshot", _always_fail_hydrate)

    try:
        fs.ensure_ready(require_hydrated_baseline=True)
        assert False, "expected ensure_ready to fail when hydration baseline is required"
    except FutureStoreUnavailableError as e:
        assert "metrics baseline hydration failed" in str(e)

    assert attempts["count"] == 3


def test_future_store_ensure_started_skips_ready_probe(monkeypatch) -> None:
    fs = FutureStore()
    calls: list[bool] = []

    def _fake_get_ray_actor(*, require_ready: bool = True):
        calls.append(bool(require_ready))
        return object()

    monkeypatch.setattr(fs, "_get_ray_actor", _fake_get_ray_actor)

    fs.ensure_started()

    assert calls == [False]


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


def test_resource_pool_cached_snapshot_exposes_rss_cache_state() -> None:
    pool = get_resource_pool()
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
    pool = get_resource_pool()
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


def test_resource_pool_cached_snapshot_refreshes_megatron_observability_on_ttl(monkeypatch) -> None:
    pool = get_resource_pool()
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
    pool = get_resource_pool()
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
    pool = get_resource_pool()
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
