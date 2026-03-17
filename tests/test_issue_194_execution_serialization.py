from __future__ import annotations

import asyncio

import pytest

from tinker_server.backend.api_work_queue import ApiWorkQueueClient, WorkItem


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _work_item(request_id: str, *, key: str, seq: int) -> WorkItem:
    return WorkItem(
        request_id=request_id,
        op="training.forward_backward",
        request_json=b"{}",
        user_id=None,
        apikey_id=None,
        throttle_principal=None,
        webhook_url=None,
        extra={
            "execution_serial_key": key,
            "execution_serial_seq": seq,
        },
        created_at=0.0,
    )


@pytest.mark.anyio
async def test_issue_194_same_key_execution_is_fifo_and_non_overlapping() -> None:
    client = ApiWorkQueueClient()
    item1 = _work_item("r1", key="training_session:model-a", seq=1)
    item2 = _work_item("r2", key="training_session:model-a", seq=2)

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    entered: list[str] = []
    active = 0
    max_active = 0

    async def _run(item: WorkItem) -> None:
        nonlocal active, max_active
        async with client._execution_serialized(item):
            entered.append(item.request_id)
            active += 1
            max_active = max(max_active, active)
            try:
                if item.request_id == "r1":
                    first_entered.set()
                    await release_first.wait()
                else:
                    second_entered.set()
            finally:
                active -= 1

    task1 = asyncio.create_task(_run(item1))
    await first_entered.wait()
    task2 = asyncio.create_task(_run(item2))
    await asyncio.sleep(0)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(task1, task2)

    assert entered == ["r1", "r2"]
    assert max_active == 1


@pytest.mark.anyio
async def test_issue_194_different_keys_can_execute_concurrently() -> None:
    client = ApiWorkQueueClient()
    item1 = _work_item("r1", key="training_session:model-a", seq=1)
    item2 = _work_item("r2", key="training_session:model-b", seq=1)

    release = asyncio.Event()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    active = 0
    max_active = 0

    async def _run(item: WorkItem, entered_event: asyncio.Event) -> None:
        nonlocal active, max_active
        async with client._execution_serialized(item):
            active += 1
            max_active = max(max_active, active)
            entered_event.set()
            try:
                await release.wait()
            finally:
                active -= 1

    task1 = asyncio.create_task(_run(item1, first_entered))
    task2 = asyncio.create_task(_run(item2, second_entered))

    await asyncio.wait_for(first_entered.wait(), timeout=1.0)
    await asyncio.wait_for(second_entered.wait(), timeout=1.0)

    release.set()
    await asyncio.gather(task1, task2)

    assert max_active == 2
