"""Sampling backend naming helpers.

This module keeps Mint sampling concepts separate from any specific serving
engine implementation. vLLM keeps its legacy domain and actor-name shape for
backward compatibility; new backends must use distinct names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

SamplingServingBackend = Literal["vllm", "sglang"]

SUPPORTED_SAMPLING_BACKENDS: tuple[SamplingServingBackend, ...] = ("vllm", "sglang")


@dataclass(frozen=True)
class SamplingBackendCapabilityDecision:
    """A backend support decision for one request-facing sampling feature."""

    backend: SamplingServingBackend
    feature: str
    supported: bool
    reason: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    adapter_path: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)


class SamplingBackendUnsupportedFeatureError(RuntimeError):
    """Raised when a sampling backend cannot safely satisfy a requested feature."""

    def __init__(self, decision: SamplingBackendCapabilityDecision, message: str | None = None) -> None:
        self.decision = decision
        detail = message or decision.reason or f"{decision.backend} does not support {decision.feature}"
        super().__init__(detail)


def normalize_sampling_backend(value: object | None) -> SamplingServingBackend:
    backend = str(value or "vllm").strip().lower()
    if backend not in SUPPORTED_SAMPLING_BACKENDS:
        raise ValueError(f"unsupported sampling serving backend: {value!r}")
    return cast(SamplingServingBackend, backend)


def domain_key_for_sampling_base_model(
    base_model: str,
    *,
    backend: object | None = "vllm",
) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    serving_backend = normalize_sampling_backend(backend)
    if serving_backend == "vllm":
        return f"vllm:{model}"
    return f"sglang:{model}"


def sampling_backend_from_domain_key(domain_key: str) -> SamplingServingBackend | None:
    domain = str(domain_key).strip()
    if domain.startswith("vllm:"):
        return "vllm"
    if domain.startswith("sglang:"):
        return "sglang"
    return None


def base_model_from_sampling_domain_key(domain_key: str) -> str | None:
    domain = str(domain_key).strip()
    backend = sampling_backend_from_domain_key(domain)
    if backend is None:
        return None
    base_model = domain.removeprefix(f"{backend}:").strip()
    return base_model or None


def actor_name_for_sampling_base_model(
    base_model: str,
    *,
    backend: object | None = "vllm",
) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    serving_backend = normalize_sampling_backend(backend)
    from mint_server.backend.ray_cluster.model_actor_names import sglang_actor_name, vllm_actor_name

    if serving_backend == "vllm":
        return vllm_actor_name(model)
    return sglang_actor_name(model)
