"""Single source of truth for model actor name construction.

All model-actor name generation must go through this module. No other module
should inline its own ``f"mint_vllm_{...}"`` string formatting.

Rules (applied uniformly to every backend):
  1. Extract model part from HF ID or resolved cache path.
  2. Lowercase.
  3. Every non-``[a-z0-9]`` character → ``_``.
  4. Collapse consecutive ``_``.
  5. Strip leading/trailing ``_``.
  6. Prepend ``mint_<backend>_``.

Result: one separator (``_``), one casing (lowercase), zero ambiguity.

  "Qwen/Qwen3-30B-A3B-Instruct-2507"
    → mint_vllm_qwen3_30b_a3b_instruct_2507
    → mint_megatron_qwen3_30b_a3b_instruct_2507
    → mint_bumblebee_qwen3_30b_a3b_instruct_2507

Placement-group names are derived from actor names via
:mod:`mint_server.backend.ray_cluster.model_actor_pg_names`.
"""

from __future__ import annotations

import re

__all__ = [
    "vllm_actor_name",
    "megatron_actor_name",
    "bumblebee_actor_name",
    "dense_actor_name",
    "default_model_actor_name",
    "sanitize_actor_name_part",
]

# models--Org--Model/snapshots/hash
_HF_CACHE_RE = re.compile(r"models--([^/]+)--([^/]+)/snapshots")


def _model_part(base_model: str) -> str:
    """Extract the model name from a base_model string.

    Handles both HuggingFace model IDs (``"Qwen/Qwen3-30B"``) and resolved
    cache paths (``"/vePFS/.../models--Qwen--Qwen3-30B/snapshots/abc"``).
    Both produce ``"Qwen3-30B"`` so downstream sanitization is identical.
    """
    match = _HF_CACHE_RE.search(str(base_model or ""))
    if match:
        return match.group(2)
    return str(base_model or "").split("/")[-1]


def _sanitize(value: str) -> str:
    """Lowercase, non-alphanumerics → ``_``, collapse runs, strip ends."""
    s = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())
    return s.strip("_") or "unknown"


def vllm_actor_name(base_model: str) -> str:
    """``"Qwen/Qwen3-30B-A3B-Instruct-2507"`` → ``"mint_vllm_qwen3_30b_a3b_instruct_2507"``"""
    return f"mint_vllm_{_sanitize(_model_part(base_model))}"


def megatron_actor_name(base_model: str) -> str:
    """``"Qwen/Qwen3-30B-A3B-Instruct-2507"`` → ``"mint_megatron_qwen3_30b_a3b_instruct_2507"``"""
    return f"mint_megatron_{_sanitize(_model_part(base_model))}"


def bumblebee_actor_name(base_model: str) -> str:
    """``"Qwen/Qwen3-30B-A3B-Instruct-2507"`` → ``"mint_bumblebee_qwen3_30b_a3b_instruct_2507"``"""
    return f"mint_bumblebee_{_sanitize(_model_part(base_model))}"


def dense_actor_name(base_model: str) -> str:
    """``"Qwen/Qwen3-0.6B"`` → ``"mint_dense_qwen3_0_6b"``"""
    return f"mint_dense_{_sanitize(_model_part(base_model))}"


def sanitize_actor_name_part(value: str) -> str:
    """Lowercase, non-alphanumerics → ``_``, collapse runs, strip ends."""
    return _sanitize(value)


def default_model_actor_name(domain_key: str, replica_id: str) -> str:
    """Supervisor wrapper name: ``mint_model_runtime_<domain>_<replica>``.

    ``"vllm:Qwen/Test"`` + ``"replica-0"``
      → ``"mint_model_runtime_vllm_qwen_test_replica_0"``
    """
    return f"mint_model_runtime_{_sanitize(domain_key)}_{_sanitize(replica_id)}"
