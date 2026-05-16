from __future__ import annotations

from typing import Any

from ..logging_context import run_async_with_otel_span
from . import model_work_dispatch
from .model_work_dispatch import KNOWN_MODEL_WORK_OPS, execute_model_work_item

KNOWN_QUEUE_OPS = KNOWN_MODEL_WORK_OPS


async def execute_work_item(item: Any) -> None:
    model_work_dispatch.run_async_with_otel_span = run_async_with_otel_span
    await execute_model_work_item(item, component="api_work_queue")


def register_api_work_queue_executors(api_work_queue: Any) -> None:
    for op in KNOWN_QUEUE_OPS:
        api_work_queue.set_executor(op, execute_work_item)
