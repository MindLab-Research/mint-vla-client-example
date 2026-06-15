from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine_adapter import EngineHealth, EngineObservability


@dataclass(frozen=True)
class EngineLivenessPush:
    actor_name: str
    domain_key: str
    replica_id: str
    consumer_id: str
    actor_generation: int
    running: bool
    engine_ready: bool
    engine_health: EngineHealth
    observability: EngineObservability = field(default_factory=EngineObservability)
    active_request_id: str | None = None
    active_lease_id: str | None = None
    active_lease_count: int = 0
    pushed_at: float | None = None
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.domain_key, self.replica_id

    def to_wire(self) -> dict[str, Any]:
        return {
            **dict(self.extra),
            "actor_name": self.actor_name,
            "domain_key": self.domain_key,
            "replica_id": self.replica_id,
            "consumer_id": self.consumer_id,
            "actor_generation": int(self.actor_generation),
            "running": bool(self.running),
            "engine_ready": bool(self.engine_ready),
            "engine_health": self.engine_health.to_wire(),
            "observability": self.observability.to_wire(),
            "active_request_id": self.active_request_id,
            "active_lease_id": self.active_lease_id,
            "active_lease_count": int(self.active_lease_count),
            "pushed_at": self.pushed_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "EngineLivenessPush":
        known = {
            "actor_name",
            "domain_key",
            "replica_id",
            "consumer_id",
            "actor_generation",
            "running",
            "engine_ready",
            "engine_health",
            "observability",
            "active_request_id",
            "active_lease_id",
            "active_lease_count",
            "pushed_at",
            "last_error",
        }
        health = data.get("engine_health") or {}
        observability = data.get("observability") or {}
        return cls(
            actor_name=str(data["actor_name"]),
            domain_key=str(data["domain_key"]),
            replica_id=str(data["replica_id"]),
            consumer_id=str(data["consumer_id"]),
            actor_generation=int(data["actor_generation"]),
            running=bool(data.get("running")),
            engine_ready=bool(data.get("engine_ready")),
            engine_health=(
                health
                if isinstance(health, EngineHealth)
                else EngineHealth.from_wire(dict(health))
            ),
            observability=(
                observability
                if isinstance(observability, EngineObservability)
                else EngineObservability.from_wire(dict(observability))
            ),
            active_request_id=(
                None if data.get("active_request_id") is None else str(data["active_request_id"])
            ),
            active_lease_id=None if data.get("active_lease_id") is None else str(data["active_lease_id"]),
            active_lease_count=int(data.get("active_lease_count") or 0),
            pushed_at=None if data.get("pushed_at") is None else float(data["pushed_at"]),
            last_error=None if data.get("last_error") is None else str(data["last_error"]),
            extra={key: value for key, value in data.items() if key not in known},
        )


__all__ = ["EngineLivenessPush"]
