from __future__ import annotations

import pytest

from tinker_server.backend.task_state_store import (
    TaskStateConflictError,
    TaskStateStore,
    _TaskStateStoreActor,
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
    assert created["ok"] is True
    assert created["created"] is True


def _own_scheduler(store: TaskStateStore, owner_id: str = "scheduler-a") -> int:
    owner = store.acquire_scheduler_owner(owner_id=owner_id, ttl_s=30.0, now=101.0)
    assert owner["ok"] is True
    return int(owner["epoch"])


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
    assert claimed["record"]["status"] == "leased"
    return epoch, lease_id, attempt_id


def test_owner_epoch_fences_stale_scheduler() -> None:
    store = TaskStateStore.in_memory()
    try:
        owner_a = store.acquire_scheduler_owner(owner_id="scheduler-a", ttl_s=30.0, now=100.0)
        assert owner_a["ok"] is True
        assert owner_a["epoch"] == 1

        blocked = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=110.0)
        assert blocked["ok"] is False
        assert blocked["reason"] == "owner_active"
        assert blocked["epoch"] == 1

        owner_b = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=131.0)
        assert owner_b["ok"] is True
        assert owner_b["epoch"] == 2
        assert store.renew_scheduler_owner(
            owner_id="scheduler-a",
            epoch=1,
            ttl_s=30.0,
            now=132.0,
        ) == {"ok": False, "reason": "stale_owner"}
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
        assert assigned["record"]["status"] == "assigned"
        assert assigned["record"]["subqueue_id"] == "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0"

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
        assert claimed["record"]["status"] == "leased"
        assert claimed["record"]["lease_id"] == "lease-1"
        assert claimed["record"]["attempt_id"] == "attempt-1"

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
            now=104.0,
        )
        assert finalizing["record"]["status"] == "finalizing"

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
        assert committed["ok"] is True
        assert committed["idempotent"] is False
        assert committed["record"]["status"] == "done"
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
        assert repeated["idempotent"] is True

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
            now=104.0,
        )
        owner_b = store.acquire_scheduler_owner(owner_id="scheduler-b", ttl_s=30.0, now=132.0)
        assert owner_b["ok"] is True
        assert owner_b["epoch"] == 2

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

        assert committed["record"]["status"] == "done"
    finally:
        store.close()


def test_requeue_task_resets_active_record_for_reclaim() -> None:
    store = TaskStateStore.in_memory()
    try:
        epoch, lease_id, attempt_id = _leased_task(store)
        requeued = store.requeue_task(
            request_id="req-1",
            scheduler_epoch=epoch,
            reason="lease_expired",
            now=104.0,
        )
        assert requeued["record"]["status"] == "pending"
        assert requeued["record"]["lease_id"] is None
        assert requeued["record"]["attempt_id"] is None

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
        assert claimed["record"]["status"] == "leased"
        assert claimed["record"]["lease_id"] == "lease-1-retry"
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

        assert failed["record"]["status"] == "failed"
        assert failed["record"]["error"] == "executor failed"
        assert store.get_task("req-1")["status"] == "failed"
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
