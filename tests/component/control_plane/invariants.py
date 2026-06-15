from __future__ import annotations

from typing import Any

from mint_server.backend.stores.task_state_store import TERMINAL_TASK_STATUSES


async def assert_terminal_not_scheduled(world: Any, request_id: str) -> None:
    record = await world.observe_task(request_id)
    assert record["status"] in TERMINAL_TASK_STATUSES
    contains = await world.observe_scheduler(request_id)
    assert contains.present is False


async def assert_no_double_lease(world: Any) -> None:
    stats = await world.scheduler.stats()
    leases = stats.get("leases") or []
    assert isinstance(leases, list)

    lease_ids: set[str] = set()
    request_ids: set[str] = set()
    for lease in leases:
        assert isinstance(lease, dict)
        lease_id = str(lease.get("lease_id") or "")
        raw_item = lease.get("item")
        item = raw_item if isinstance(raw_item, dict) else {}
        request_id = str(item.get("request_id") or lease.get("request_id") or "")
        assert lease_id
        assert request_id
        assert lease_id not in lease_ids
        assert request_id not in request_ids
        lease_ids.add(lease_id)
        request_ids.add(request_id)


async def assert_lease_consistency(world: Any) -> None:
    stats = await world.scheduler.stats()
    leases = stats.get("leases") or []
    assert isinstance(leases, list)

    for lease in leases:
        assert isinstance(lease, dict)
        raw_item = lease.get("item")
        item = raw_item if isinstance(raw_item, dict) else {}
        request_id = str(item.get("request_id") or lease.get("request_id") or "")
        assert request_id
        record = await world.observe_task(request_id)
        assert record["status"] in {"leased", "finalizing"}
        assert str(record.get("lease_id") or "") == str(lease.get("lease_id") or "")
        assert str(record.get("attempt_id") or "") == str(lease.get("attempt_id") or "")
        assert int(record.get("scheduler_epoch") or 0) == int(lease.get("scheduler_epoch") or 0)


async def assert_every_terminal_has_payload_ref(world: Any) -> None:
    records = await world.task_state.async_list_tasks_by_metadata(
        statuses=["done"],
        limit=1000,
    )
    for record_obj in records:
        record = record_obj.to_wire() if hasattr(record_obj, "to_wire") else dict(record_obj)
        assert str(record.get("status") or "") == "done"
        assert record.get("result_path")
        assert record.get("result_checksum") is not None
        assert record.get("result_size_bytes") is not None


async def assert_no_orphan_assigned(world: Any) -> None:
    stats = await world.scheduler.stats()
    replica_queues = stats.get("replica_queues") or {}
    assert isinstance(replica_queues, dict)

    active = await world.task_ledger.list_active_tasks()
    active_by_request_id: dict[str, dict[str, Any]] = {}
    for record_obj in active:
        record = record_obj.to_wire() if hasattr(record_obj, "to_wire") else dict(record_obj)
        request_id = str(record.get("request_id") or "")
        if request_id:
            active_by_request_id[request_id] = record

    assigned_count_by_queue: dict[str, int] = {}
    for record in active_by_request_id.values():
        if str(record.get("status") or "") != "assigned":
            continue
        queue_id = str(record.get("subqueue_id") or "")
        assigned_count_by_queue[queue_id] = assigned_count_by_queue.get(queue_id, 0) + 1

    for queue_id, queue in replica_queues.items():
        assert isinstance(queue, dict)
        depth = int(queue.get("depth") or 0)
        assert assigned_count_by_queue.get(str(queue_id), 0) == depth
