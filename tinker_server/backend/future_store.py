"""Storage for async operation results.

Maps request_id to results for the async polling pattern:
1. Client sends request, gets request_id
2. Server processes in background
3. Client polls with request_id until result ready
"""

import uuid
from enum import Enum
from typing import Any


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class FutureStore:
    """Thread-safe storage for async operation results."""

    def __init__(self):
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._pending: set[str] = set()

    def create(self) -> str:
        """Create a new pending future and return its request_id."""
        request_id = str(uuid.uuid4())
        self._pending.add(request_id)
        return request_id

    def resolve(self, request_id: str, result: Any) -> None:
        """Mark a future as completed with the given result."""
        self._pending.discard(request_id)
        self._results[request_id] = result

    def fail(self, request_id: str, error: str) -> None:
        """Mark a future as failed with the given error message."""
        self._pending.discard(request_id)
        self._errors[request_id] = error

    def get_status(self, request_id: str) -> FutureStatus:
        """Get the status of a future.

        Raises KeyError if request_id is unknown.
        """
        if request_id in self._results:
            return FutureStatus.DONE
        if request_id in self._errors:
            return FutureStatus.FAILED
        if request_id in self._pending:
            return FutureStatus.PENDING
        raise KeyError(f"Unknown request_id: {request_id}")

    def get_result(self, request_id: str) -> Any:
        """Get the result of a completed future, or None if not done."""
        return self._results.get(request_id)

    def get_error(self, request_id: str) -> str | None:
        """Get the error message of a failed future, or None if not failed."""
        return self._errors.get(request_id)

    def cleanup(self, request_id: str) -> None:
        """Remove a future from storage after retrieval."""
        self._pending.discard(request_id)
        self._results.pop(request_id, None)
        self._errors.pop(request_id, None)


# Global instance
future_store = FutureStore()
