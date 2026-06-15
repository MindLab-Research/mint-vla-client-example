from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class EngineHealthStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    STARTING = "starting"
    RESTARTING = "restarting"
    UNHEALTHY = "unhealthy"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class EngineHealth:
    status: EngineHealthStatus
    reason: str | None = None
    error_count: int = 0
    restart_count: int = 0
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status is EngineHealthStatus.READY

    def to_wire(self) -> dict[str, Any]:
        return {
            **dict(self.extra),
            "status": self.status.value,
            "reason": self.reason,
            "error_count": int(self.error_count),
            "restart_count": int(self.restart_count),
            "last_error": self.last_error,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "EngineHealth":
        known = {"status", "reason", "error_count", "restart_count", "last_error"}
        return cls(
            status=EngineHealthStatus(str(data["status"])),
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            error_count=int(data.get("error_count") or 0),
            restart_count=int(data.get("restart_count") or 0),
            last_error=str(data["last_error"]) if data.get("last_error") is not None else None,
            extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass(frozen=True)
class GpuPerformanceSample:
    device_index: int | None = None
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.utilization_percent is not None and not 0.0 <= float(self.utilization_percent) <= 100.0:
            raise ValueError("utilization_percent must be between 0 and 100")
        if self.memory_used_bytes is not None and int(self.memory_used_bytes) < 0:
            raise ValueError("memory_used_bytes must be non-negative")
        if self.memory_total_bytes is not None and int(self.memory_total_bytes) < 0:
            raise ValueError("memory_total_bytes must be non-negative")

    def to_wire(self) -> dict[str, Any]:
        return {
            **dict(self.extra),
            "device_index": self.device_index,
            "utilization_percent": self.utilization_percent,
            "memory_used_bytes": self.memory_used_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "power_watts": self.power_watts,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "GpuPerformanceSample":
        known = {
            "device_index",
            "utilization_percent",
            "memory_used_bytes",
            "memory_total_bytes",
            "power_watts",
            "temperature_c",
        }
        return cls(
            device_index=_optional_int(data.get("device_index")),
            utilization_percent=_optional_float(data.get("utilization_percent")),
            memory_used_bytes=_optional_int(data.get("memory_used_bytes")),
            memory_total_bytes=_optional_int(data.get("memory_total_bytes")),
            power_watts=_optional_float(data.get("power_watts")),
            temperature_c=_optional_float(data.get("temperature_c")),
            extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass(frozen=True)
class EngineObservability:
    gpu_performance: tuple[GpuPerformanceSample, ...] = ()
    kv_cache_capacity_tokens: int | None = None
    active_requests: int | None = None
    queued_requests: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kv_cache_capacity_tokens is not None and int(self.kv_cache_capacity_tokens) < 0:
            raise ValueError("kv_cache_capacity_tokens must be non-negative")
        if self.active_requests is not None and int(self.active_requests) < 0:
            raise ValueError("active_requests must be non-negative")
        if self.queued_requests is not None and int(self.queued_requests) < 0:
            raise ValueError("queued_requests must be non-negative")

    def to_wire(self) -> dict[str, Any]:
        return {
            **dict(self.extra),
            "gpu_performance": [sample.to_wire() for sample in self.gpu_performance],
            "kv_cache_capacity_tokens": self.kv_cache_capacity_tokens,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "EngineObservability":
        known = {
            "gpu_performance",
            "kv_cache_capacity_tokens",
            "active_requests",
            "queued_requests",
        }
        samples = data.get("gpu_performance") or ()
        return cls(
            gpu_performance=tuple(
                sample if isinstance(sample, GpuPerformanceSample) else GpuPerformanceSample.from_wire(dict(sample))
                for sample in samples
            ),
            kv_cache_capacity_tokens=_optional_int(data.get("kv_cache_capacity_tokens")),
            active_requests=_optional_int(data.get("active_requests")),
            queued_requests=_optional_int(data.get("queued_requests")),
            extra={key: value for key, value in data.items() if key not in known},
        )


@runtime_checkable
class _EngineLifecycle(Protocol):
    async def is_ready(self) -> bool: ...

    async def health(self) -> EngineHealth: ...

    async def get_observability_binding(self) -> EngineObservability: ...


@runtime_checkable
class InferenceEngineAdapter(_EngineLifecycle, Protocol):
    async def generate(self, *args: Any, **kwargs: Any) -> Any: ...

    async def compute_logprobs(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TrainingEngineAdapter(_EngineLifecycle, Protocol):
    async def create_training_session(self, *args: Any, **kwargs: Any) -> Any: ...

    async def train_step(self, *args: Any, **kwargs: Any) -> Any: ...

    async def forward_backward(self, *args: Any, **kwargs: Any) -> Any: ...

    async def save_weights(self, *args: Any, **kwargs: Any) -> Any: ...

    async def load_weights(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "EngineHealth",
    "EngineHealthStatus",
    "EngineObservability",
    "GpuPerformanceSample",
    "InferenceEngineAdapter",
    "TrainingEngineAdapter",
]
