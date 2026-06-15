"""Backend components for mint-server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mint_server.backend.stores.task_state_store import FutureStatus, task_futures

if TYPE_CHECKING:
    from mint_server.backend.training.verl.verl_inference import VerlInferenceEngine

__all__ = [
    "FutureStatus",
    "VerlInferenceEngine",
    "task_futures",
]


def __getattr__(name: str):
    if name == "VerlInferenceEngine":
        from mint_server.backend.training.verl.verl_inference import VerlInferenceEngine as _VerlInferenceEngine

        return _VerlInferenceEngine
    raise AttributeError(name)
