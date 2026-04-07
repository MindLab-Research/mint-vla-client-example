from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_CURRENT_QUEUE_CONSUMER_ID: ContextVar[str | None] = ContextVar("mint_queue_consumer_id", default=None)
_CURRENT_QUEUE_GENERATION_ID: ContextVar[int | None] = ContextVar("mint_queue_generation_id", default=None)


def get_current_queue_consumer_id() -> str | None:
    return _CURRENT_QUEUE_CONSUMER_ID.get()


def get_current_queue_generation_id() -> int | None:
    return _CURRENT_QUEUE_GENERATION_ID.get()


@contextmanager
def queue_execution_context(*, consumer_id: str | None, generation_id: int | None) -> Iterator[None]:
    token_consumer = _CURRENT_QUEUE_CONSUMER_ID.set(None if consumer_id is None else str(consumer_id))
    token_generation = _CURRENT_QUEUE_GENERATION_ID.set(None if generation_id is None else int(generation_id))
    try:
        yield
    finally:
        _CURRENT_QUEUE_CONSUMER_ID.reset(token_consumer)
        _CURRENT_QUEUE_GENERATION_ID.reset(token_generation)
