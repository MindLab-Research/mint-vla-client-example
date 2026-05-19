"""Backend components for mint-server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .task_state_store import FutureStatus, task_futures

if TYPE_CHECKING:
    from .verl_inference import VerlInferenceEngine

__all__ = ["FutureStatus", "task_futures", "VerlInferenceEngine"]


def __getattr__(name: str):
    if name == "VerlInferenceEngine":
        from .verl_inference import VerlInferenceEngine as _VerlInferenceEngine

        return _VerlInferenceEngine
    raise AttributeError(name)
