from __future__ import annotations

from typing import Any

from mint_server.backend.task_state_store import TERMINAL_TASK_STATUSES


async def assert_terminal_not_scheduled(world: Any, request_id: str) -> None:
    record = world.task_store.get_task(request_id)
    assert record["status"] in TERMINAL_TASK_STATUSES
    contains = await world.scheduler.contains_request(request_id=request_id)
    assert contains["present"] is False
