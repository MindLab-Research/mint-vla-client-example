from __future__ import annotations

from typing import Any, Callable


class ActiveLeaseTracker:
    """Tracks currently executing scheduler leases inside a model engine host."""

    def __init__(self, *, lease_item_wire: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._lease_item_wire = lease_item_wire
        self._active_request_id: str | None = None
        self._active_lease_id: str | None = None
        self._active_leases: dict[str, dict[str, Any]] = {}

    @property
    def active_request_id(self) -> str | None:
        return self._active_request_id

    @property
    def active_lease_id(self) -> str | None:
        return self._active_lease_id

    def set_active(self, *, lease_id: str, request_id: str, lease: dict[str, Any]) -> None:
        self._active_leases[str(lease_id)] = lease
        self._active_request_id = str(request_id)
        self._active_lease_id = str(lease_id)

    def clear(self, lease_id: str) -> None:
        self._active_leases.pop(str(lease_id), None)
        next_lease = next(iter(self._active_leases.values()), None)
        self._active_request_id = (
            str(self._lease_item_wire(next_lease)["request_id"]) if isinstance(next_lease, dict) else None
        )
        self._active_lease_id = next(iter(self._active_leases.keys()), None)

    def leases(self) -> list[dict[str, Any]]:
        return list(self._active_leases.values())

    def snapshot_fields(self) -> dict[str, Any]:
        return {
            "active_request_id": self._active_request_id,
            "active_lease_id": self._active_lease_id,
            "active_request_ids": [
                str(self._lease_item_wire(lease)["request_id"]) for lease in self._active_leases.values()
            ],
            "active_lease_ids": list(self._active_leases.keys()),
            "active_lease_count": len(self._active_leases),
        }
