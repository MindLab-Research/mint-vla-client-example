from __future__ import annotations

import asyncio

from mint_server.backend.scheduling.model_work_admission import enqueue_model_work
from mint_server.backend.stores.future_state_store import FutureStateStore
from mint_server.backend.stores.task_state_store import FutureStatus, TaskFutureService, TaskStateStore


class _LocalFutureStateClient:
    def __init__(self, store: FutureStateStore) -> None:
        self.store = store

    async def async_ensure_task(self, **kwargs):
        return self.store.ensure_task(**kwargs)

    async def async_create_task(self, **kwargs):
        return self.store.create_task(**kwargs)

    async def async_update_task_metadata(self, **kwargs):
        return self.store.update_task_metadata(**kwargs)

    async def async_stage_payload(self, **kwargs):
        return self.store.stage_payload(**kwargs)

    async def async_complete_task_success(self, **kwargs):
        return self.store.complete_task_success(**kwargs)

    async def async_complete_task_failure(self, **kwargs):
        return self.store.complete_task_failure(**kwargs)

    async def async_mark_task_retrieved(self, **kwargs):
        return self.store.mark_task_retrieved(**kwargs)

    async def async_get_task(self, request_id: str):
        return self.store.get_task(request_id)

    async def async_list_tasks_by_metadata(self, **kwargs):
        return self.store.list_tasks_by_metadata(**kwargs)

    async def async_forget_task(self, **kwargs):
        return self.store.forget_task(**kwargs)

    async def async_expire_active_tasks(self, **kwargs):
        return self.store.expire_active_tasks(**kwargs)

    async def async_list_terminal_payloads_for_eviction(self, **kwargs):
        return self.store.list_terminal_payloads_for_eviction(**kwargs)

    async def async_mark_payload_evicted(self, **kwargs):
        return self.store.mark_payload_evicted(**kwargs)

    async def async_record_payload_evict_error(self, **kwargs):
        return self.store.record_payload_evict_error(**kwargs)

    async def async_list_staged_payloads_for_gc(self, **kwargs):
        return self.store.list_staged_payloads_for_gc(**kwargs)

    async def async_mark_staged_payload_gc_deleted(self, **kwargs):
        return self.store.mark_staged_payload_gc_deleted(**kwargs)

    async def async_delete_expired_tombstones(self, **kwargs):
        return self.store.delete_expired_tombstones(**kwargs)

    async def async_wait_task_status_change(self, **kwargs):
        record = self.store.get_task(kwargs["request_id"])
        return {"changed": True, "record": record}


class _LocalTaskStateClient:
    def __init__(self, store: TaskStateStore) -> None:
        self.store = store
        self.outbox_calls = []

    async def async_append_billing_outbox(self, **kwargs):
        self.outbox_calls.append(kwargs)
        return self.store.append_billing_outbox(**kwargs)

    async def async_get_task(self, request_id: str):
        return self.store.get_task(request_id)

    async def async_mark_task_retrieved(self, **kwargs):
        return self.store.mark_task_retrieved(**kwargs)

    async def async_billing_outbox_stats(self):
        return self.store.billing_outbox_stats()

    async def async_stats(self):
        return {"ok": True}


def test_future_state_store_scheduler_lease_and_finalize() -> None:
    store = FutureStateStore.in_memory()
    owner = store.acquire_scheduler_owner(owner_id="scheduler-a", ttl_s=30.0, now=100.0)
    assert owner["ok"] is True

    store.create_task(
        request_id="req-1",
        op="sampling.asample",
        domain_key="vllm:model",
        request_json=b'{"x":1}',
        metadata={"op": "sampling.asample"},
        now=101.0,
    )
    store.assign_task(request_id="req-1", subqueue_id="q-1", scheduler_epoch=owner["epoch"], now=102.0)
    store.claim_task(
        request_id="req-1",
        subqueue_id="q-1",
        lease_id="lease-1",
        attempt_id="attempt-1",
        consumer_id="consumer-1",
        scheduler_epoch=owner["epoch"],
        runtime_generation=1,
        lease_ttl_s=30.0,
        now=103.0,
    )
    renewed = store.renew_lease(
        request_id="req-1",
        lease_id="lease-1",
        attempt_id="attempt-1",
        scheduler_epoch=owner["epoch"],
        runtime_generation=1,
        lease_ttl_s=60.0,
        now=104.0,
    )
    assert renewed["record"]["status"] == "leased"
    assert renewed["record"]["lease_expires_at"] == 164.0
    store.begin_finalize(
        request_id="req-1",
        lease_id="lease-1",
        attempt_id="attempt-1",
        scheduler_epoch=owner["epoch"],
        runtime_generation=1,
        finalize_ttl_s=30.0,
        staged_payload_path="/tmp/payload.json",
        now=105.0,
    )
    out = store.commit_finalize_success(
        request_id="req-1",
        lease_id="lease-1",
        attempt_id="attempt-1",
        scheduler_epoch=owner["epoch"],
        runtime_generation=1,
        result_path="/tmp/payload.json",
        result_checksum="sha256:abc",
        result_size_bytes=12,
        metadata={"billing_status": "outboxed"},
        now=106.0,
    )

    assert out["record"]["status"] == "done"
    assert out["record"]["metadata"]["billing_status"] == "outboxed"
    assert store.get_task("req-1")["request_json"] == b'{"x":1}'


def test_future_state_store_metrics_use_status_indexes_not_created_full_scan() -> None:
    store = FutureStateStore.in_memory()
    store.create_task(
        request_id="pending-1",
        op="sampling.asample",
        domain_key="vllm:model",
        request_json=b"{}",
        now=100.0,
    )
    store.create_task(
        request_id="done-1",
        op="sampling.asample",
        domain_key="vllm:model",
        request_json=b"{}",
        now=90.0,
    )
    store.complete_task_success(
        request_id="done-1",
        result_path="/tmp/done.json",
        result_checksum="sha256:abc",
        result_size_bytes=12,
        now=101.0,
    )
    store._kv.put("idx:created:000000000000000001.000000:missing-old", "missing-old")

    stats = store.future_metrics_stats(now=200.0)

    assert stats["pending"] == 1
    assert stats["results"] == 1
    assert stats["by_op"]["sampling.asample"]["pending"] == 1
    assert stats["by_op"]["sampling.asample"]["results"] == 1
    assert stats["payload_stats"]["records_scanned"] == 2


def test_future_state_store_list_active_tasks_limit_bounds_index_scans(monkeypatch) -> None:
    store = FutureStateStore.in_memory()
    for idx in range(3):
        store.create_task(
            request_id=f"pending-{idx}",
            op="sampling.asample",
            domain_key="vllm:model",
            request_json=b"{}",
            now=float(idx),
        )

    calls: list[int | None] = []
    original = store._ids_from_index_prefix

    def _tracked(prefix: str, *, limit: int | None = None):
        calls.append(limit)
        return original(prefix, limit=limit)

    monkeypatch.setattr(store, "_ids_from_index_prefix", _tracked)

    active = store.list_active_tasks(limit=1)

    assert [record["request_id"] for record in active] == ["pending-0"]
    assert calls == [1]


def test_task_future_service_writes_new_futures_to_future_state_store(tmp_path) -> None:
    future_store = FutureStateStore.in_memory()
    task_store = TaskStateStore.in_memory()
    service = TaskFutureService(
        future_state_client=_LocalFutureStateClient(future_store),
        task_state_client=_LocalTaskStateClient(task_store),
    )
    service._payload_store = __import__(
        "mint_server.backend.stores.task_payload_store",
        fromlist=["TaskPayloadStore"],
    ).TaskPayloadStore(root_dir=tmp_path)

    asyncio.run(service.async_create_model_work_with_id(
        "req-2",
        op="sampling.asample",
        domain_key="vllm:model",
        request_json=b"{}",
        meta={"op": "sampling.asample"},
    ))
    asyncio.run(service.async_resolve("req-2", {"ok": True}))

    assert future_store.get_task("req-2")["status"] == "done"
    assert asyncio.run(service.async_get_status("req-2")) == FutureStatus.DONE
    assert asyncio.run(service.async_get_result("req-2")) == {"ok": True}
    assert future_store.get_task("req-2")["status"] == "retrieved"


def test_model_work_admission_delegates_durable_create_to_scheduler_gateway(tmp_path) -> None:
    future_store = FutureStateStore.in_memory()
    task_store = TaskStateStore.in_memory()
    service = TaskFutureService(
        future_state_client=_LocalFutureStateClient(future_store),
        task_state_client=_LocalTaskStateClient(task_store),
    )
    service._payload_store = __import__(
        "mint_server.backend.stores.task_payload_store",
        fromlist=["TaskPayloadStore"],
    ).TaskPayloadStore(root_dir=tmp_path)

    class _Scheduler:
        def __init__(self) -> None:
            self.seen_status: FutureStatus | None = None
            self.seen_metadata: dict | None = None

        async def append_work(self, **kwargs):
            try:
                self.seen_status = await service.async_get_status(kwargs["request_id"])
            except KeyError:
                self.seen_status = None
            self.seen_metadata = dict(kwargs.get("extra") or {})
            return {"ok": True, "request_id": kwargs["request_id"]}

    scheduler = _Scheduler()

    asyncio.run(
        enqueue_model_work(
            request_id="req-admission",
            op="training.create_model",
            request_json=b'{"base_model":"Qwen/Test"}',
            domain_key="megatron:Qwen/Test",
            queued_meta={"op": "training.create_model"},
            scheduler_client=scheduler,
            future_service_client=service,
        )
    )

    assert scheduler.seen_status == FutureStatus.PENDING
    assert scheduler.seen_metadata is not None
    assert scheduler.seen_metadata["op"] == "training.create_model"
    assert asyncio.run(service.async_get_status("req-admission")) == FutureStatus.PENDING


def test_model_work_admission_does_not_cleanup_future_when_scheduler_append_fails(tmp_path) -> None:
    future_store = FutureStateStore.in_memory()
    task_store = TaskStateStore.in_memory()
    service = TaskFutureService(
        future_state_client=_LocalFutureStateClient(future_store),
        task_state_client=_LocalTaskStateClient(task_store),
    )
    service._payload_store = __import__(
        "mint_server.backend.stores.task_payload_store",
        fromlist=["TaskPayloadStore"],
    ).TaskPayloadStore(root_dir=tmp_path)

    class _Scheduler:
        async def append_work(self, **kwargs):
            raise RuntimeError("scheduler append failed")

    async def _run() -> None:
        try:
            await enqueue_model_work(
                request_id="req-admission-fail",
                op="training.create_model",
                request_json=b'{"base_model":"Qwen/Test"}',
                domain_key="megatron:Qwen/Test",
                queued_meta={"op": "training.create_model"},
                scheduler_client=_Scheduler(),
                future_service_client=service,
            )
        except RuntimeError as e:
            assert str(e) == "scheduler append failed"
        else:
            raise AssertionError("enqueue_model_work should have raised")

    asyncio.run(_run())
    try:
        future_store.get_task("req-admission-fail")
    except KeyError:
        pass
    else:
        raise AssertionError("failed scheduler append should clean up local future")


def test_task_future_service_reads_legacy_sqlite_terminal_rows(tmp_path) -> None:
    future_store = FutureStateStore.in_memory()
    task_store = TaskStateStore.in_memory()
    payload_store = __import__(
        "mint_server.backend.stores.task_payload_store",
        fromlist=["TaskPayloadStore"],
    ).TaskPayloadStore(root_dir=tmp_path)
    payload = payload_store.write_json_payload(request_id="legacy-1", attempt_id="a", payload={"legacy": True})
    task_store.ensure_task(request_id="legacy-1", metadata={"op": "sampling.asample"})
    task_store.stage_payload(request_id="legacy-1", staged_payload_path=str(payload["path"]))
    task_store.complete_task_success(
        request_id="legacy-1",
        result_path=str(payload["path"]),
        result_checksum=str(payload["checksum"]),
        result_size_bytes=int(payload["size_bytes"]),
        metadata={"done_at": 1.0},
    )
    service = TaskFutureService(
        future_state_client=_LocalFutureStateClient(future_store),
        task_state_client=_LocalTaskStateClient(task_store),
    )
    service._payload_store = payload_store

    assert asyncio.run(service.async_get_status("legacy-1")) == FutureStatus.DONE
    assert asyncio.run(service.async_get_result("legacy-1")) == {"legacy": True}
