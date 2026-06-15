from __future__ import annotations

from typing import Any


DEFAULT_DOMAIN = "vllm:model-a"
DEFAULT_REPLICA = "replica-0"
DEFAULT_GENERATION = 3


def sampling_meta(domain_key: str = DEFAULT_DOMAIN) -> dict[str, Any]:
    return {
        "op": "sampling.asample",
        "domain_key": domain_key,
        "queue_kind": "model_work_scheduler",
        "queue_state": "queued",
    }
