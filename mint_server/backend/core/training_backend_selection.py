from __future__ import annotations

import structlog
import os

logger = structlog.get_logger(__name__)

_DISTRIBUTED_MOE_BACKENDS = {"megatron", "bumblebee"}


def _canonical_moe_training_backend(value: str | None) -> str:
    backend = str(value or "bumblebee").strip().lower()
    aliases = {
        "bb": "bumblebee",
        "bumblebee": "bumblebee",
        "megatron": "megatron",
    }
    if backend not in aliases:
        raise ValueError(
            f"unsupported MoE training backend {value!r}; expected one of {sorted(set(aliases.values()))}"
        )
    return aliases[backend]


def _is_qwen3_30b_model(model: str | None) -> bool:
    return "qwen3-30b-a3b" in str(model or "").lower()


def _is_qwen3_235b_model(model: str | None) -> bool:
    return "qwen3-235b-a22b" in str(model or "").lower()


def _is_qwen35_model(model: str | None) -> bool:
    return "qwen3.5-27b" in str(model or "").lower()


def _uses_distributed_training_backend(requested_model: str | None) -> bool:
    try:
        from mint_server.backend.core.model_registry import get_model_config

        if bool(getattr(get_model_config(requested_model or ""), "is_moe", False)):
            return True
    except Exception:
        logger.debug("distributed_training_model_config_lookup_failed_for__s", exc_info=True)
    return _is_qwen35_model(requested_model)


def _select_moe_training_backend(requested_model: str | None) -> str:
    """Select the resident MoE trainer for Mint text sessions."""
    env_name = None
    if _is_qwen3_30b_model(requested_model):
        env_name = "MINT_QWEN3_30B_TRAINING_BACKEND"
    elif _is_qwen3_235b_model(requested_model):
        env_name = "MINT_QWEN3_235B_TRAINING_BACKEND"
    elif _is_qwen35_model(requested_model):
        env_name = "MINT_QWEN35_TRAINING_BACKEND"
    if env_name and os.environ.get(env_name):
        return _canonical_moe_training_backend(os.environ.get(env_name))
    return _canonical_moe_training_backend(os.environ.get("MINT_MOE_TRAINING_BACKEND"))
