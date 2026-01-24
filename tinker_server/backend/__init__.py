"""Backend components for tinker-server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .future_store import FutureStatus, FutureStore, future_store

if TYPE_CHECKING:
    from .verl_inference import VerlInferenceEngine

__all__ = ["FutureStatus", "FutureStore", "future_store", "VerlInferenceEngine"]


def __getattr__(name: str):
    if name == "VerlInferenceEngine":
        from .verl_inference import VerlInferenceEngine as _VerlInferenceEngine

        return _VerlInferenceEngine
    raise AttributeError(name)
