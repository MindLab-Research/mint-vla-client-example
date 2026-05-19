from __future__ import annotations

import asyncio

import pytest

from mint_server.backend.task_state_store import (
    TaskStateConflictError,
    TaskFutureService,
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

        assert finalizing["record"]["status"] == "finalizing"
        assert finalizing["record"]["lease_id"] == lease_id
        assert finalizing["record"]["metadata"]["stage"] == "prefill"
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
        assert running["record"]["metadata"]["model_id"] == "model-a"
        assert running["record"]["metadata"]["stage"] == "running"

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

        assert created["ok"] is True
        assert created["created"] is False
        assert created["record"]["status"] == "queued"
        assert created["record"]["payload_hash"] == "hash-1"
        assert created["record"]["request_json"] == b'{"prompt":"b"}'
        assert created["record"]["metadata"]["stage"] == "queued"
        assert created["record"]["metadata"]["model_work_scheduler"] is True

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

        assert failed["record"]["status"] == "failed"
        assert failed["record"]["error"] == "executor failed"
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

        service = TaskFutureService(task_state_client=_LocalTaskStateClient(), payload_store=_FailingPayloadStore())
        out = asyncio.run(service.async_reap())

        assert out["payload_evicted"] == []
        assert out["payload_evict_errors"][0]["request_id"] == "req-fail-delete"
        record = store.get_task("req-fail-delete")
        assert record["result_path"] == str(result_path)
        assert "payload_evicted_at" not in record["metadata"]

        service = TaskFutureService(task_state_client=_LocalTaskStateClient(), payload_store=_WorkingPayloadStore())
        out = asyncio.run(service.async_reap())

        assert out["payload_evicted"] == ["req-fail-delete"]
        record = store.get_task("req-fail-delete")
        assert record["result_path"] is None
        assert record["metadata"]["payload_evicted_at"] > 0
        assert not result_path.exists()
    finally:
        store.close()
