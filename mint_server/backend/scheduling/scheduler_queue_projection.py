from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict
from typing import Any


class QueueProjection:
    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    def backlog(self, domain_key: str):
        return self._scheduler._domain_backlog.setdefault(str(domain_key), deque())

    def queue(self, domain_key: str, replica_id: str):
        return self._scheduler._replica_queues.setdefault(
            self._scheduler._queue_key(domain_key, replica_id),
            deque(),
        )

    def hot_projection_matches_work_item_locked(self, item: Any) -> bool:
        scheduler = self._scheduler
        lease_id = scheduler._lease_id_by_request_id.get(item.request_id)
        if lease_id is not None:
            lease = scheduler._leases_by_id.get(lease_id)
            if lease is not None:
                return scheduler._task_record_matches_work_item(lease.item.to_dict(), item)
        for queue in scheduler._replica_queues.values():
            for assigned in queue:
                if assigned.item.request_id == item.request_id:
                    return scheduler._task_record_matches_work_item(assigned.item.to_dict(), item)
        for backlog in scheduler._domain_backlog.values():
            for current in backlog:
                if current.request_id == item.request_id:
                    return scheduler._task_record_matches_work_item(current.to_dict(), item)
        return False

    def ordering_key_has_active_lease_locked(self, ordering_key: str | None) -> bool:
        if not ordering_key:
            return False
        return any(lease.item.ordering_key == ordering_key for lease in self._scheduler._leases_by_id.values())

    def cluster_queue_head_affinity(self, queue) -> None:
        if not queue:
            return
        affinity_group = queue[0].item.affinity_group
        if affinity_group is None:
            return
        same = deque()
        other = deque()
        while queue:
            assigned = queue.popleft()
            if assigned.item.affinity_group == affinity_group:
                same.append(assigned)
            else:
                other.append(assigned)
        queue.extend(same)
        queue.extend(other)

    def drop_empty_backlog(self, domain_key: str) -> None:
        backlog = self._scheduler._domain_backlog.get(domain_key)
        if backlog is not None and not backlog:
            self._scheduler._domain_backlog.pop(domain_key, None)

    def has_inflight_scheduler_transition_locked(self) -> bool:
        return any(
            location in {"assigning", "claiming", "requeueing", "finalizing"}
            for location in self._scheduler._request_locations.values()
        )

    def assigned_matches_locked(self, assigned: Any, *, location: str = "assigning") -> bool:
        scheduler = self._scheduler
        request_id = assigned.item.request_id
        if scheduler._request_locations.get(request_id) != location:
            return False
        if location != "assigning":
            for queue in scheduler._replica_queues.values():
                if any(current.item.request_id == request_id for current in queue):
                    return False
        return not any(
            item.request_id == request_id
            for backlog in scheduler._domain_backlog.values()
            for item in backlog
        )

    def commit_assigned_locked(self, assigned: Any) -> None:
        scheduler = self._scheduler
        queue = self.queue(assigned.item.domain_key, assigned.replica_id)
        if not any(current.item.request_id == assigned.item.request_id for current in queue):
            queue.append(assigned)
        scheduler._request_locations[assigned.item.request_id] = "assigned"
        scheduler._assigned += 1

    def restore_assigned_to_queue_locked(self, assigned: Any) -> None:
        queue = self.queue(assigned.item.domain_key, assigned.replica_id)
        queue.appendleft(assigned)
        self._scheduler._request_locations[assigned.item.request_id] = "assigned"

    def restore_assigning_to_backlog_locked(self, assigned: Any) -> bool:
        scheduler = self._scheduler
        if scheduler._request_locations.get(assigned.item.request_id) != "assigning":
            return False
        key = scheduler._queue_key(assigned.item.domain_key, assigned.replica_id)
        queue = self.queue(assigned.item.domain_key, assigned.replica_id)
        kept = deque(
            current
            for current in queue
            if current.item.request_id != assigned.item.request_id
        )
        scheduler._replica_queues[key] = kept
        self.backlog(assigned.item.domain_key).appendleft(assigned.item)
        scheduler._request_locations[assigned.item.request_id] = "backlog"
        return True

    def remove_request_from_memory_locked(self, request_id: str) -> bool:
        scheduler = self._scheduler
        request_id = str(request_id)
        removed = False
        for domain_key, backlog in list(scheduler._domain_backlog.items()):
            kept = deque(item for item in backlog if item.request_id != request_id)
            if len(kept) != len(backlog):
                removed = True
                scheduler._domain_backlog[domain_key] = kept
                self.drop_empty_backlog(domain_key)
        for key, queue in list(scheduler._replica_queues.items()):
            kept = deque(assigned for assigned in queue if assigned.item.request_id != request_id)
            if len(kept) != len(queue):
                removed = True
                scheduler._replica_queues[key] = kept
        lease_id = scheduler._lease_id_by_request_id.pop(request_id, None)
        if lease_id is not None:
            removed = scheduler._leases_by_id.pop(lease_id, None) is not None or removed
        scheduler._request_locations.pop(request_id, None)
        if removed:
            scheduler._untrack_sampling_inflight_locked(request_id)
        return removed

    def claimable_replicas(self, domain_key: str) -> list[Any]:
        scheduler = self._scheduler
        candidates = [
            replica
            for (replica_domain, _), replica in scheduler._replicas.items()
            if replica_domain == domain_key and replica.claimable
        ]
        active_by_replica: dict[str, int] = {}
        for lease in scheduler._leases_by_id.values():
            if lease.domain_key != domain_key:
                continue
            active_by_replica[lease.replica_id] = active_by_replica.get(lease.replica_id, 0) + 1
        replicas = [
            replica
            for replica in candidates
            if active_by_replica.get(replica.replica_id, 0)
            + len(self.queue(replica.domain_key, replica.replica_id))
            < max(1, int(replica.capacity))
        ]
        replicas.sort(
            key=lambda r: (
                active_by_replica.get(r.replica_id, 0) + len(self.queue(r.domain_key, r.replica_id)),
                active_by_replica.get(r.replica_id, 0),
                len(self.queue(r.domain_key, r.replica_id)),
                r.replica_id,
            )
        )
        return replicas

    def choose_replica(self, item: Any) -> Any | None:
        scheduler = self._scheduler
        replicas = self.claimable_replicas(item.domain_key)
        if not replicas:
            return None
        if item.affinity_group:
            affinity_key = (item.domain_key, item.affinity_group)
            sticky = scheduler._affinity_replica.get(affinity_key)
            if sticky is not None:
                for replica in replicas:
                    if replica.replica_id == sticky:
                        return replica
        replica = replicas[0]
        if item.affinity_group:
            scheduler._affinity_replica[(item.domain_key, item.affinity_group)] = replica.replica_id
        return replica

    def requeue_assigned(self, assigned: Any, *, reason: str) -> None:
        scheduler = self._scheduler
        item = assigned.item
        updated_extra = dict(item.extra)
        updated_extra["last_requeue_reason"] = str(reason)
        updated = type(item)(
            **{
                **asdict(item),
                "request_json": item.request_json,
                "extra": updated_extra,
            }
        )
        self.backlog(updated.domain_key).appendleft(updated)
        scheduler._request_locations[updated.request_id] = "backlog"
        scheduler._requeued += 1

    def remove_request_location(self, request_id: str) -> None:
        scheduler = self._scheduler
        scheduler._request_locations.pop(str(request_id), None)
        scheduler._lease_id_by_request_id.pop(str(request_id), None)
        scheduler._untrack_sampling_inflight_locked(str(request_id))

    def drop_claiming_request_locked(self, assigned: Any) -> bool:
        scheduler = self._scheduler
        if scheduler._request_locations.get(assigned.item.request_id) != "claiming":
            return False
        self.remove_request_location(assigned.item.request_id)
        scheduler._stale_dropped += 1
        return True

    def prepare_assignments_locked(self, *, max_items: int | None = None) -> tuple[list[Any], list[str]]:
        scheduler = self._scheduler
        pending: list[Any] = []
        skipped_domains: list[str] = []
        limit = None if max_items is None else max(0, int(max_items))
        for domain_key in sorted(list(scheduler._domain_backlog)):
            backlog = scheduler._domain_backlog.get(domain_key)
            while backlog:
                if limit is not None and len(pending) >= limit:
                    return pending, skipped_domains
                item = backlog[0]
                replica = self.choose_replica(item)
                if replica is None:
                    skipped_domains.append(domain_key)
                    break
                backlog.popleft()
                assigned = scheduler._assigned_work_type(
                    item=item,
                    replica_id=replica.replica_id,
                    queue_id=replica.effective_queue_id,
                    assigned_at=time.time(),
                    assignment_generation=replica.generation,
                    assignment_reason="least_loaded_affinity",
                )
                pending.append(assigned)
                queue = self.queue(replica.domain_key, replica.replica_id)
                queue.append(assigned)
                scheduler._request_locations[item.request_id] = "assigning"
            self.drop_empty_backlog(domain_key)
        return pending, skipped_domains
