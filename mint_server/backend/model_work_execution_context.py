from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

_CURRENT_MODEL_WORK_LEASE_ID: ContextVar[str | None] = ContextVar("mint_model_work_lease_id", default=None)
_CURRENT_MODEL_WORK_CONSUMER_ID: ContextVar[str | None] = ContextVar("mint_model_work_consumer_id", default=None)
_CURRENT_MODEL_WORK_CONSUMER_GENERATION: ContextVar[int | None] = ContextVar(
    "mint_model_work_consumer_generation",
    default=None,
)
_CURRENT_MODEL_WORK_FINALIZE_BUFFER: ContextVar["ModelWorkFinalizeBuffer | None"] = ContextVar(
    "mint_model_work_finalize_buffer",
    default=None,
)


@dataclass
class ModelWorkFinalize:
    kind: str
    request_id: str
    payload: object


@dataclass
class ModelWorkFinalizeBuffer:
    finalization: ModelWorkFinalize | None = None


def get_current_model_work_lease_id() -> str | None:
    return _CURRENT_MODEL_WORK_LEASE_ID.get()


def get_current_model_work_consumer_id() -> str | None:
    return _CURRENT_MODEL_WORK_CONSUMER_ID.get()


def get_current_model_work_consumer_generation() -> int | None:
    return _CURRENT_MODEL_WORK_CONSUMER_GENERATION.get()


def get_current_model_work_finalize_buffer() -> ModelWorkFinalizeBuffer | None:
    return _CURRENT_MODEL_WORK_FINALIZE_BUFFER.get()


@contextmanager
def model_work_execution_context(
    *,
    lease_id: str | None,
    consumer_id: str | None,
    consumer_generation: int | None,
    finalize_buffer: ModelWorkFinalizeBuffer | None = None,
) -> Iterator[None]:
    token_lease = _CURRENT_MODEL_WORK_LEASE_ID.set(None if lease_id is None else str(lease_id))
    token_consumer = _CURRENT_MODEL_WORK_CONSUMER_ID.set(
        None if consumer_id is None else str(consumer_id)
    )
    token_generation = _CURRENT_MODEL_WORK_CONSUMER_GENERATION.set(
        None if consumer_generation is None else int(consumer_generation)
    )
    token_finalize = _CURRENT_MODEL_WORK_FINALIZE_BUFFER.set(finalize_buffer)
    try:
        yield
    finally:
        _CURRENT_MODEL_WORK_FINALIZE_BUFFER.reset(token_finalize)
        _CURRENT_MODEL_WORK_LEASE_ID.reset(token_lease)
        _CURRENT_MODEL_WORK_CONSUMER_ID.reset(token_consumer)
        _CURRENT_MODEL_WORK_CONSUMER_GENERATION.reset(token_generation)
