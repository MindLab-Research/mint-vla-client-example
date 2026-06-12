from __future__ import annotations

from types import SimpleNamespace

import pytest

from mint_server.backend.engine_lifecycle import ExecutionContextEngineLifecycle
from mint_server.backend.execution_context import ExecutionContext


class _ProbeEngine:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = bool(ready)
        self.restart_calls = 0

    async def is_ready(self) -> bool:
        return self.ready

    async def restart(self) -> None:
        self.restart_calls += 1
        self.ready = True

    async def get_observability_binding(self) -> dict:
        return {
            "kv_cache_capacity_tokens": 256,
            "gpu_performance": [{"device_index": 0, "utilization_percent": 12.5}],
        }


@pytest.mark.anyio
async def test_execution_context_engine_lifecycle_uses_nested_engine_probe() -> None:
    engine = _ProbeEngine(ready=False)
    manager = SimpleNamespace(_shared_engine=engine)
    context = ExecutionContext(
        inference_manager=manager,
        train_manager=object(),
        train_engine=object(),
        action_manager=object(),
    )
    lifecycle = ExecutionContextEngineLifecycle(lambda: _context(context))

    assert await lifecycle.is_ready() is False
    await lifecycle.restart()
    assert engine.restart_calls == 1
    assert await lifecycle.is_ready() is True
    observability = await lifecycle.get_observability_binding()
    assert observability.kv_cache_capacity_tokens == 256
    assert observability.gpu_performance[0].utilization_percent == 12.5


async def _context(context: ExecutionContext) -> ExecutionContext:
    return context
