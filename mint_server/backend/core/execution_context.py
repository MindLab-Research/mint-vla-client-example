from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime-actor-local execution handles for queued model work."""

    inference_manager: Any
    train_manager: Any
    train_engine: Any
    action_manager: Any
    multi_model_manager: Any = None
    restored_sampling_sessions: int = 0
    multi_model_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "inference_manager": self.inference_manager,
            "train_manager": self.train_manager,
            "train_engine": self.train_engine,
            "action_manager": self.action_manager,
            "multi_model_manager": self.multi_model_manager,
            "restored_sampling_sessions": int(self.restored_sampling_sessions),
            "multi_model_enabled": bool(self.multi_model_enabled),
        }


_CURRENT_EXECUTION_CONTEXT: ContextVar[ExecutionContext | None] = ContextVar(
    "mint_execution_context",
    default=None,
)


def current_execution_context() -> ExecutionContext | None:
    return _CURRENT_EXECUTION_CONTEXT.get()


@contextmanager
def bind_execution_context(context: ExecutionContext | None) -> Iterator[None]:
    if context is None:
        yield
        return
    token: Token[ExecutionContext | None] = _CURRENT_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_CONTEXT.reset(token)
