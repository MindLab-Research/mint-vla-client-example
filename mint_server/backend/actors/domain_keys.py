"""Single source of truth for model-actor domain keys.

A *domain key* identifies a logical model + backend pair used throughout the
scheduler, supervisor, and placement subsystems.  Every domain key has the
shape ``"<backend>:<base_model>"`` (or the special ``"internal:runtime"``).

This module centralises:

* **Prefix constants** — no other module should hard-code ``"vllm:"`` etc.
* **Construction** — ``domain_key_for_vllm()``, ``domain_key_for_training()``…
* **Parsing** — ``base_model_from_domain_key()``, ``domain_key_prefix()``, ``is_*_domain()``

Historically these patterns were duplicated across ``model_actor_supervisor``,
``model_actor_launchers``, ``model_actor_placement``, and
``cluster_placement_controller``, leading to missing-prefix bugs (e.g.
``verl_fsdp2_lora:`` was not recognised, blocking model actor creation).
"""

from __future__ import annotations

__all__ = [
    # Prefix constants
    "VLLM_PREFIX",
    "SGLANG_PREFIX",
    "TRAINING_PREFIX",
    "VERL_FSDP2_LORA_PREFIX",
    "MEGATRON_PREFIX",
    "BUMBLEBEE_PREFIX",
    "DENSE_PREFIX",
    "TRAINING_SESSION_PREFIX",
    "INTERNAL_RUNTIME_DOMAIN",
    # Construction
    "domain_key_for_vllm",
    "domain_key_for_training",
    "domain_key_for_internal_runtime",
    # Parsing
    "parse_domain_key",
    "domain_key_prefix",
    "base_model_from_domain_key",
    "launcher_key_for_domain",
    "is_vllm_domain",
    "is_sglang_domain",
    "is_training_domain",
    "is_megatron_domain",
    "is_bumblebee_domain",
    "is_training_like_domain",
]

# ── prefix constants ──────────────────────────────────────────────

VLLM_PREFIX = "vllm"
SGLANG_PREFIX = "sglang"
TRAINING_PREFIX = "training"
VERL_FSDP2_LORA_PREFIX = "verl_fsdp2_lora"
MEGATRON_PREFIX = "megatron"
BUMBLEBEE_PREFIX = "bumblebee"
DENSE_PREFIX = "dense"
TRAINING_SESSION_PREFIX = "training_session"

INTERNAL_RUNTIME_DOMAIN = "internal:runtime"

# Prefixes that use the generic "training" launcher (not "vllm").
_TRAINING_LAUNCHER_PREFIXES = frozenset({
    TRAINING_PREFIX,
    VERL_FSDP2_LORA_PREFIX,
    MEGATRON_PREFIX,
    BUMBLEBEE_PREFIX,
})

# All known domain-key prefixes (longest first so that e.g.
# "verl_fsdp2_lora" is matched before "v" when splitting on ":").
_KNOWN_PREFIXES = frozenset({
    VLLM_PREFIX,
    SGLANG_PREFIX,
    TRAINING_PREFIX,
    VERL_FSDP2_LORA_PREFIX,
    MEGATRON_PREFIX,
    BUMBLEBEE_PREFIX,
    DENSE_PREFIX,
    TRAINING_SESSION_PREFIX,
})


# ── construction ──────────────────────────────────────────────────

def domain_key_for_vllm(base_model: str) -> str:
    """``"Qwen/Qwen3-0.6B"`` → ``"vllm:Qwen/Qwen3-0.6B"``"""
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    return f"{VLLM_PREFIX}:{model}"


def domain_key_for_training(base_model: str, *, backend: str | None = None) -> str:
    """Construct a training-domain key.

    For MoE models the backend is resolved to ``megatron`` or ``bumblebee``
    via :func:`_select_moe_training_backend`; the domain key then uses that
    backend as the prefix (e.g. ``"megatron:mint_megatron_model"``).

    For dense models the prefix is ``"training:"``.
    """
    from mint_server.backend.actors.model_actor_supervisor import (
        _is_moe_training_model,
        _selected_moe_training_backend,
        _normalize_megatron_domain_key,
    )

    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    if _is_moe_training_model(model):
        moe_backend = _selected_moe_training_backend(model, backend=backend)
        return f"{moe_backend}:{_normalize_megatron_domain_key(model)}"
    return f"{TRAINING_PREFIX}:{model}"


def domain_key_for_internal_runtime() -> str:
    return INTERNAL_RUNTIME_DOMAIN


# ── parsing ──────────────────────────────────────────────────────

def parse_domain_key(domain_key: str) -> tuple[str, str]:
    """Split ``"vllm:Qwen/Qwen3-0.6B"`` → ``("vllm", "Qwen/Qwen3-0.6B")``.

    Returns ``("", domain_key)`` if no ``":"`` is found.
    """
    domain = str(domain_key or "").strip()
    if ":" not in domain:
        return ("", domain)
    prefix, model = domain.split(":", 1)
    return (prefix.strip(), model.strip())


def domain_key_prefix(domain_key: str) -> str:
    """Return the prefix part (before ``":"``)."""
    return parse_domain_key(domain_key)[0]


def base_model_from_domain_key(domain_key: str) -> str | None:
    """Extract the base-model portion from a domain key.

    Returns ``None`` for prefixes that don't carry a base model
    (``training_session:``, ``internal:``) or when the model part is empty.
    """
    prefix, model = parse_domain_key(domain_key)
    if not prefix or prefix in (TRAINING_SESSION_PREFIX, "internal"):
        return None
    return model or None


def launcher_key_for_domain(domain_key: str) -> str:
    """Map a domain key prefix to the launcher key used by the supervisor.

    * ``vllm:`` → ``"vllm"``
    * ``sglang:`` → ``"sglang"``
    * ``training:``, ``verl_fsdp2_lora:``, ``megatron:``, ``bumblebee:`` → ``"training"``
    * ``dense:`` → ``"dense"``
    * anything else → ``"training"`` (safe default)
    """
    prefix = domain_key_prefix(domain_key)
    if prefix == VLLM_PREFIX:
        return "vllm"
    if prefix == SGLANG_PREFIX:
        return "sglang"
    if prefix == DENSE_PREFIX:
        return "dense"
    return "training"


def is_vllm_domain(domain_key: str) -> bool:
    return domain_key_prefix(domain_key) == VLLM_PREFIX


def is_sglang_domain(domain_key: str) -> bool:
    return domain_key_prefix(domain_key) == SGLANG_PREFIX


def is_training_domain(domain_key: str) -> bool:
    """True for ``training:`` prefix only (not megatron/bumblebee/verl)."""
    return domain_key_prefix(domain_key) == TRAINING_PREFIX


def is_megatron_domain(domain_key: str) -> bool:
    return domain_key_prefix(domain_key) == MEGATRON_PREFIX


def is_bumblebee_domain(domain_key: str) -> bool:
    return domain_key_prefix(domain_key) == BUMBLEBEE_PREFIX


def is_training_like_domain(domain_key: str) -> bool:
    """True for any backend that uses the ``"training"`` launcher key."""
    return domain_key_prefix(domain_key) in _TRAINING_LAUNCHER_PREFIXES
