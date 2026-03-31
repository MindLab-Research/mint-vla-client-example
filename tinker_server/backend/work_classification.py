from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _clean_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def scheduler_enabled_from_env() -> bool:
    return _to_bool(os.environ.get("MINT_SCHEDULER_ENABLE", "0"))


def infer_scheduler_capacity_owner(scheduler_domain: Any) -> str | None:
    domain = _clean_str(scheduler_domain)
    if domain is None or ":" not in domain:
        return None
    backend, domain_key = domain.split(":", 1)
    backend = backend.strip().lower()
    domain_key = domain_key.strip()
    if not backend or not domain_key:
        return None
    if backend == "vllm":
        if "::replica::" in domain_key:
            return "vllm_replica_single_worker"
        return "model_registry_inference_dp"
    if backend in ("megatron", "peft"):
        return "single_worker"
    return None


@dataclass(frozen=True)
class WorkClassification:
    queue_kind: str
    scheduler_enabled: bool
    scheduler_domain: str | None = None
    scheduler_session_key: str | None = None
    scheduler_domain_key_source: str | None = None
    scheduler_capacity_owner: str | None = None

    @classmethod
    def build(
        cls,
        *,
        scheduler_enabled: bool,
        scheduler_domain: Any = None,
        scheduler_session_key: Any = None,
        scheduler_domain_key_source: Any = None,
        scheduler_capacity_owner: Any = None,
    ) -> "WorkClassification":
        domain = _clean_str(scheduler_domain)
        session_key = _clean_str(scheduler_session_key)
        key_source = _clean_str(scheduler_domain_key_source)
        capacity_owner = _clean_str(scheduler_capacity_owner)
        if capacity_owner is None:
            capacity_owner = infer_scheduler_capacity_owner(domain)
        enabled = bool(scheduler_enabled)
        queue_kind = "scheduled" if enabled and domain and session_key and capacity_owner else "legacy"
        return cls(
            queue_kind=queue_kind,
            scheduler_enabled=enabled,
            scheduler_domain=domain,
            scheduler_session_key=session_key,
            scheduler_domain_key_source=key_source,
            scheduler_capacity_owner=capacity_owner,
        )

    @classmethod
    def from_queue_extra(
        cls,
        extra: Any,
        *,
        scheduler_enabled_default: bool = False,
    ) -> "WorkClassification":
        if not isinstance(extra, dict):
            return cls.build(scheduler_enabled=scheduler_enabled_default)
        scheduler_session_key = extra.get("scheduler_session_key")
        if scheduler_session_key is None:
            scheduler_session_key = extra.get("session_id")
        enabled = scheduler_enabled_default
        if "scheduler_enabled" in extra:
            enabled = _to_bool(extra.get("scheduler_enabled"))
        return cls.build(
            scheduler_enabled=enabled,
            scheduler_domain=extra.get("scheduler_domain"),
            scheduler_session_key=scheduler_session_key,
            scheduler_domain_key_source=extra.get("scheduler_domain_key_source"),
            scheduler_capacity_owner=extra.get("scheduler_capacity_owner"),
        )

    def queue_extra(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(extra or {})
        payload["scheduler_enabled"] = bool(self.scheduler_enabled)
        if self.scheduler_domain is not None:
            payload["scheduler_domain"] = str(self.scheduler_domain)
        if self.scheduler_session_key is not None:
            payload["scheduler_session_key"] = str(self.scheduler_session_key)
        if self.scheduler_domain_key_source is not None:
            payload["scheduler_domain_key_source"] = str(self.scheduler_domain_key_source)
        if self.scheduler_capacity_owner is not None:
            payload["scheduler_capacity_owner"] = str(self.scheduler_capacity_owner)
        return payload

    def queued_meta(
        self,
        *,
        op: str,
        queued_at: float,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = {
            "op": str(op),
            "queue_state": "queued",
            "queued_at": float(queued_at),
            "stage": "queued",
            "queue_kind": str(self.queue_kind),
            "scheduler_domain": self.scheduler_domain,
            "scheduler_session_id": self.scheduler_session_key,
            "scheduler_domain_key_source": self.scheduler_domain_key_source,
            "scheduler_capacity_owner": self.scheduler_capacity_owner,
        }
        if extra_meta:
            meta.update(extra_meta)
        return meta


def build_training_work_classification(
    *,
    session: Any,
    model_id: str,
) -> WorkClassification:
    backend = _clean_str(getattr(session, "backend", None)) or "unknown"
    base_model = _clean_str(getattr(session, "base_model", None))
    model_key = _clean_str(model_id)
    domain_key = base_model or model_key
    return WorkClassification.build(
        scheduler_enabled=scheduler_enabled_from_env(),
        scheduler_domain=None if domain_key is None else f"{backend}:{domain_key}",
        scheduler_session_key=model_key,
        scheduler_domain_key_source="backend_base_model",
    )


def build_sampling_work_classification(
    *,
    session_manager: Any,
    session_id: str,
) -> WorkClassification:
    if session_manager is None or not session_manager.is_multi_lora_session(session_id):
        return WorkClassification.build(scheduler_enabled=scheduler_enabled_from_env())
    get_replica_key = getattr(session_manager, "get_session_replica_key", None)
    replica_key = get_replica_key(session_id) if callable(get_replica_key) else None
    base_model = session_manager.get_session_base_model(session_id)
    domain_key = _clean_str(replica_key) or _clean_str(base_model)
    return WorkClassification.build(
        scheduler_enabled=scheduler_enabled_from_env(),
        scheduler_domain=None if domain_key is None else f"vllm:{domain_key}",
        scheduler_session_key=session_id,
        scheduler_domain_key_source="replica_key" if _clean_str(replica_key) else "base_model_fallback",
    )
