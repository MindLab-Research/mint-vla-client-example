from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from mint_server.backend.contracts.engine_adapter import (
    EngineHealth,
    EngineHealthStatus,
    EngineObservability,
    GpuPerformanceSample,
    InferenceEngineAdapter,
    TrainingEngineAdapter,
)


class _FakeInferenceEngine:
    def __init__(self) -> None:
        self._status = EngineHealthStatus.READY
        self._error: str | None = None
        self.generated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def die(self, error: str = "engine died") -> None:
        self._status = EngineHealthStatus.UNHEALTHY
        self._error = error

    def resume(self) -> None:
        self._status = EngineHealthStatus.READY
        self._error = None

    async def is_ready(self) -> bool:
        return self._status is EngineHealthStatus.READY

    async def health(self) -> EngineHealth:
        return EngineHealth(status=self._status, reason=self._error)

    async def get_observability_binding(self) -> EngineObservability:
        return EngineObservability(
            gpu_performance=(
                GpuPerformanceSample(
                    device_index=0,
                    utilization_percent=73.5,
                    memory_used_bytes=21_000_000_000,
                    memory_total_bytes=80_000_000_000,
                    power_watts=312.0,
                    temperature_c=64.0,
                ),
            ),
            kv_cache_capacity_tokens=4096,
        )

    async def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not await self.is_ready():
            raise RuntimeError("engine is not ready")
        self.generated.append((args, kwargs))
        return {"text": "ok"}

    async def compute_logprobs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not await self.is_ready():
            raise RuntimeError("engine is not ready")
        return {"logprobs": [0.0]}


class _FakeTrainingEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def is_ready(self) -> bool:
        return True

    async def health(self) -> EngineHealth:
        return EngineHealth(status=EngineHealthStatus.READY)

    async def get_observability_binding(self) -> EngineObservability:
        return EngineObservability()

    async def create_training_session(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append("create_training_session")
        return "session"

    async def train_step(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.calls.append("train_step")
        return {"loss": 1.0}

    async def forward_backward(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.calls.append("forward_backward")
        return {"loss": 1.0}

    async def save_weights(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append("save_weights")
        return "/tmp/weights"

    async def load_weights(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("load_weights")


def test_engine_health_is_frozen_and_wire_round_trips() -> None:
    health = EngineHealth(
        status=EngineHealthStatus.DEGRADED,
        reason="warming",
        error_count=2,
    )

    with pytest.raises(FrozenInstanceError):
        health.status = EngineHealthStatus.READY  # type: ignore[misc]

    assert EngineHealth.from_wire(health.to_wire()) == health


def test_observability_records_gpu_performance_and_rejects_invalid_utilization() -> None:
    sample = GpuPerformanceSample(device_index=1, utilization_percent=99.9)
    observability = EngineObservability(gpu_performance=(sample,), active_requests=3)

    assert EngineObservability.from_wire(observability.to_wire()) == observability
    with pytest.raises(ValueError, match="utilization_percent"):
        GpuPerformanceSample(utilization_percent=101.0)


def test_inference_adapter_contract_supports_die_and_resume_double() -> None:
    async def run() -> None:
        engine = _FakeInferenceEngine()
        assert isinstance(engine, InferenceEngineAdapter)

        assert await engine.is_ready() is True
        assert await engine.generate(prompt="hi") == {"text": "ok"}

        engine.die()
        assert await engine.is_ready() is False
        assert (await engine.health()).status is EngineHealthStatus.UNHEALTHY
        with pytest.raises(RuntimeError, match="not ready"):
            await engine.generate(prompt="hi")

        engine.resume()
        assert await engine.is_ready() is True
        assert (await engine.get_observability_binding()).kv_cache_capacity_tokens == 4096

    asyncio.run(run())


def test_training_adapter_does_not_require_inference_surface() -> None:
    async def run() -> None:
        engine = _FakeTrainingEngine()
        assert isinstance(engine, TrainingEngineAdapter)
        assert not isinstance(engine, InferenceEngineAdapter)

        assert await engine.create_training_session(object()) == "session"
        assert await engine.train_step(object()) == {"loss": 1.0}
        assert await engine.forward_backward(object()) == {"loss": 1.0}
        assert await engine.save_weights(object()) == "/tmp/weights"
        await engine.load_weights(object())
        assert engine.calls == [
            "create_training_session",
            "train_step",
            "forward_backward",
            "save_weights",
            "load_weights",
        ]

    asyncio.run(run())
