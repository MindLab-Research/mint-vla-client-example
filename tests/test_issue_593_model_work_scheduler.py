import asyncio
import time

import pytest

from mint_server.backend.model_work_scheduler import (
    ModelWorkSchedulerConflictError,
    _ModelWorkSchedulerActor,
    _ray_model_work_scheduler_actor_name,
)
from mint_server.backend.task_state_store import TaskStateStore


@pytest.fixture(autouse=True)
def disable_scheduler_assignment_loop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0")


def _work(
    request_id: str,
    *,
    domain_key: str = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    affinity_group: str | None = "lora:session-a:generation:1",
    token_cost: int = 1,
) -> dict:
    return {
        "request_id": request_id,
        "op": "sampling.asample",
        "request_json": b"{}",
        "user_id": "user-a",
        "apikey_id": "key-a",
        "throttle_principal": "apikey:key-a",
        "webhook_url": None,
        "extra": {},
        "created_at": 100.0,
        "domain_key": domain_key,
        "affinity_group": affinity_group,
        "ordering_key": "session:session-a",
        "token_cost": token_cost,
    }


def _replica(
    replica_id: str,
    *,
    domain_key: str = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    consumer_id: str | None = None,
    generation: int = 10,
    status: str = "healthy",
) -> dict:
    return {
        "domain_key": domain_key,
        "replica_id": replica_id,
        "consumer_id": consumer_id or f"consumer-{replica_id}",
        "generation": generation,
        "status": status,
        "queue_id": f"{domain_key}::{replica_id}",
        "capacity": 4,
        "actor_name": f"actor-{replica_id}",
        "node_pins": ["10.0.0.1"],
        "updated_at": 101.0,
    }


def test_scheduler_assigns_to_registered_replica_queue() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1")))["ok"] is True
        assert (await actor.sync_replicas([_replica("replica-0")]))["replicas"] == 1

        stats = actor.stats()
        queue = stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"]
        assert queue["depth"] == 1
        assert stats["backlog_depth"] == 0

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-1"]
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 0

    asyncio.run(_run())


def test_scheduler_default_actor_name_uses_mint_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME", raising=False)

    assert _ray_model_work_scheduler_actor_name() == "mint_model_work_scheduler"


def test_scheduler_append_can_assign_immediately() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert await actor.is_empty() is True
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(_work("req-1"), assign=True, assign_max_items=1)

        assert out["ok"] is True
        assert await actor.is_empty() is False
        assert out["scheduler_instance_id"]
        contains = await actor.contains_request(request_id="req-1")
        assert contains["present"] is True
        assert contains["location"] == "assigned"
        assert contains["scheduler_instance_id"] == out["scheduler_instance_id"]
        assert out["assigned"]["assigned"] == 1
        assert actor.stats()["backlog_depth"] == 0
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 1

    asyncio.run(_run())


def test_scheduler_assignment_loop_moves_backlog_to_replica_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0.01")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-loop")))["ok"] is True
        await actor.sync_replicas([_replica("replica-0")])
        for _ in range(20):
            stats = actor.stats()
            queue = stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"]
            if stats["backlog_depth"] == 0 and queue["depth"] == 1:
                return
            await asyncio.sleep(0.02)
        stats = actor.stats()
        raise AssertionError(f"assignment loop did not drain backlog: {stats!r}")

    asyncio.run(_run())


def test_scheduler_cancel_request_removes_assigned_work() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-1"), assign=True)

        out = await actor.cancel_request(request_id="req-1", reason="test_cancel")

        assert out["cancelled"] is True
        assert (await actor.contains_request(request_id="req-1"))["present"] is False
        assert await actor.is_empty() is True

    asyncio.run(_run())


def test_replica_can_claim_only_own_queue_and_generation() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.append(_work("req-1"))
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.assign_pending()

        with pytest.raises(ModelWorkSchedulerConflictError, match="consumer_id mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-1",
                consumer_generation=10,
            )
        with pytest.raises(ModelWorkSchedulerConflictError, match="generation mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-0",
                consumer_generation=9,
            )

    asyncio.run(_run())


def test_affinity_sticks_to_same_healthy_replica() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.append(_work("req-1", affinity_group="lora:a"))
        await actor.append(_work("req-2", affinity_group="lora:a"))
        await actor.append(_work("req-3", affinity_group="lora:b"))
        await actor.assign_pending()

        stats = actor.stats()["replica_queues"]
        depths = {
            queue["replica_id"]: queue["depth"]
            for queue in stats.values()
        }
        assert depths["replica-0"] == 2
        assert depths["replica-1"] == 1

    asyncio.run(_run())


def test_assignment_accounts_for_active_leases() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.append(_work("req-active", affinity_group="lora:a"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-active"]

        await actor.append(_work("req-next", affinity_group="lora:b"))
        await actor.assign_pending()

        stats = actor.stats()["replica_queues"]
        depths = {queue["replica_id"]: queue["depth"] for queue in stats.values()}
        assert depths["replica-0"] == 0
        assert depths["replica-1"] == 1

    asyncio.run(_run())


def test_unhealthy_replica_requeues_assigned_and_leased_work() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-assigned"))
        await actor.append(_work("req-leased"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-assigned"]

        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])
        assert sync["requeued"] == 2
        stats = actor.stats()
        assert stats["backlog_depth"] == 2
        assert stats["leases"] == []
        assert stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"][
            "status"
        ] == "unhealthy"

    asyncio.run(_run())


def test_lease_complete_fail_and_expiry() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _claim_one(request_id: str) -> str:
        await actor.append(_work(request_id))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=1.0,
        )
        return str(claimed["leases"][0]["lease_id"])

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])

        complete_lease = await _claim_one("req-complete")
        assert (await actor.complete_lease(
            lease_id=complete_lease,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        ))["ok"] is True
        assert actor.stats()["counters"]["completed"] == 1

        fail_lease = await _claim_one("req-fail")
        failed = await actor.fail_lease(
            lease_id=fail_lease,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=False,
            reason="runtime_error",
        )
        assert failed == {"ok": True, "request_id": "req-fail", "requeued": False}
        assert actor.stats()["counters"]["failed"] == 1

        expire_lease = await _claim_one("req-expire")
        assert (await actor.expire_leases(now=time.time() + 999.0))["expired"] == 1
        assert actor.stats()["backlog_depth"] == 1
        assert expire_lease not in {lease["lease_id"] for lease in actor.stats()["leases"]}

    asyncio.run(_run())


def test_validate_lease_rejects_requeued_or_stale_leases() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-validate"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )
        lease_id = str(claimed["leases"][0]["lease_id"])

        assert (await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        ))["ok"] is True
        assert (await actor.fail_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=True,
            reason="test_requeue",
        ))["requeued"] is True
        assert (await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )) == {"ok": False, "reason": "unknown_lease"}

    asyncio.run(_run())


def test_finalizing_lease_survives_replica_sync_until_finalize_ttl() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-finalizing"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            lease_ttl_s=1.0,
        )
        lease_id = str(claimed["leases"][0]["lease_id"])

        assert (await actor.begin_finalize_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=60.0,
        ))["ok"] is True
        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])

        assert sync["requeued"] == 0
        assert actor.stats()["leases"][0]["lease_id"] == lease_id

    asyncio.run(_run())


def test_scheduler_persists_append_assign_claim_and_begin_finalize_to_task_state_store() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(
            _work(
                "req-persisted",
                affinity_group="lora:persisted",
            ),
            assign=True,
        )
        assert out["ok"] is True

        record = store.get_task("req-persisted")
        assert record["status"] == "assigned"
        assert record["subqueue_id"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        assert record["scheduler_epoch"] == 1

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed["leases"][0]
        record = store.get_task("req-persisted")
        assert record["status"] == "leased"
        assert record["lease_id"] == lease["lease_id"]
        assert record["attempt_id"] == lease["attempt_id"]
        assert record["runtime_generation"] == 10

        finalizing = await actor.begin_finalize_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/req-persisted.json",
        )
        assert finalizing["ok"] is True
        record = store.get_task("req-persisted")
        assert record["status"] == "finalizing"
        assert record["staged_payload_path"] == "/tmp/req-persisted.json"
        assert actor.stats()["task_state_store_enabled"] is True
        assert actor.stats()["scheduler_epoch"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_hydrates_active_task_state_after_restart() -> None:
    store = TaskStateStore.in_memory()
    actor_a = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )
    actor_b = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor_a.sync_replicas([_replica("replica-0")])
        assert (await actor_a.append(_work("req-restart"), assign=True))["ok"] is True
        assert store.get_task("req-restart")["status"] == "assigned"

        await actor_b.sync_replicas([_replica("replica-0")])
        contains = await actor_b.contains_request(request_id="req-restart")
        assert contains["present"] is True
        assert contains["location"] == "assigned"
        claimed = await actor_b.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-restart"]
        assert store.get_task("req-restart")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_persists_requeue_before_reclaim() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-requeue"), assign=True))["ok"] is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed["leases"][0]
        failed = await actor.fail_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            reason="executor_failed",
            requeue=True,
        )
        assert failed["requeued"] is True
        assert store.get_task("req-requeue")["status"] == "pending"

        await actor.assign_pending(max_items=1)
        reclaimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [item["item"]["request_id"] for item in reclaimed["leases"]] == ["req-requeue"]
        assert store.get_task("req-requeue")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()
