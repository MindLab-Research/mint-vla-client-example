"""Custom vLLM model class for Qwen3.5/3.6 Qwen3Next architecture.

Subclasses ``Qwen3NextForCausalLM`` to handle checkpoint weight-name mapping
(``model.language_model.`` → ``model.``) and linear-attention split-weight
packing (``in_proj_qkv`` + ``in_proj_z`` → ``in_proj_qkvz``).

This replaces the previous monkey-patch approach (``patch_vllm_qwen35_*``).
Because vLLM 0.23 forces ``spawn`` for worker subprocesses, monkey-patches
installed in the actor's ``__init__`` are lost.  A subclass + registry
approach survives ``spawn`` because the class methods are part of the class
definition, not process-local state.

The class is created lazily inside ``register()`` so that this module can be
imported in environments where vLLM is not installed (e.g. CI pyright gate).
Registration is triggered by appending an import of this module to vLLM's
``__init__.py`` in ``qwen36-vllm-deps``, so that both the API server process
and spawned worker subprocesses pick it up automatically.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Weight name mapping helpers (no vLLM dependency)
# ---------------------------------------------------------------------------

_LINEAR_ATTN_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+\.linear_attn)\."
    r"(?P<part>in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)\.weight$"
)

_LINEAR_ATTN_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"in_proj_qkv", "in_proj_z"}),
    frozenset({"in_proj_b", "in_proj_a"}),
)


def _pack_qkv_z(config: Any, qkv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack split ``in_proj_qkv`` + ``in_proj_z`` → ``in_proj_qkvz``."""
    hidden_size = qkv.shape[1]
    qk_heads = config.linear_num_key_heads
    qk_head_dim = config.linear_key_head_dim
    value_heads = config.linear_num_value_heads
    value_head_dim = config.linear_value_head_dim
    if value_heads % qk_heads != 0:
        raise ValueError(
            "Qwen3Next linear attention packing requires "
            f"linear_num_value_heads % linear_num_key_heads == 0, got "
            f"{value_heads} % {qk_heads}"
        )
    qk_dim = qk_heads * qk_head_dim
    value_dim = value_heads * value_head_dim
    expected_qkv = (qk_dim * 2 + value_dim, hidden_size)
    expected_z = (value_dim, hidden_size)
    if tuple(qkv.shape) != expected_qkv:
        raise ValueError(
            f"Unexpected in_proj_qkv shape: got={tuple(qkv.shape)} expected={expected_qkv}"
        )
    if tuple(z.shape) != expected_z:
        raise ValueError(
            f"Unexpected in_proj_z shape: got={tuple(z.shape)} expected={expected_z}"
        )
    q, k, v = torch.split(qkv, [qk_dim, qk_dim, value_dim], dim=0)
    q, k = [w.reshape(qk_heads, qk_head_dim, hidden_size) for w in (q, k)]
    v = v.reshape(qk_heads, value_heads // qk_heads * value_head_dim, hidden_size)
    z = z.reshape(qk_heads, value_heads // qk_heads * value_head_dim, hidden_size)
    return torch.cat([q, k, v, z], dim=1).reshape(-1, hidden_size)


def _pack_b_a(config: Any, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Pack split ``in_proj_b`` + ``in_proj_a`` → ``in_proj_ba``."""
    hidden_size = b.shape[1]
    qk_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    if value_heads % qk_heads != 0:
        raise ValueError(
            "Qwen3Next linear attention packing requires "
            f"linear_num_value_heads % linear_num_key_heads == 0, got "
            f"{value_heads} % {qk_heads}"
        )
    expected = (value_heads, hidden_size)
    if tuple(b.shape) != expected:
        raise ValueError(f"Unexpected in_proj_b shape: got={tuple(b.shape)} expected={expected}")
    if tuple(a.shape) != expected:
        raise ValueError(f"Unexpected in_proj_a shape: got={tuple(a.shape)} expected={expected}")
    b = b.reshape(qk_heads, value_heads // qk_heads, hidden_size)
    a = a.reshape(qk_heads, value_heads // qk_heads, hidden_size)
    return torch.cat([b, a], dim=1).reshape(-1, hidden_size)


def _map_checkpoint_weights(
    config: Any,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Map Qwen3Next checkpoint weight names to vLLM module parameter names.

    1. Skip visual encoder weights (``model.visual.*``).
    2. Strip ``model.language_model.`` → ``model.`` prefix.
    3. Pack linear-attention split weights into fused names.
    """
    pending: dict[str, dict[str, object]] = {}

    def _drain(prefix: str) -> Iterable[tuple[str, torch.Tensor]]:  # type: ignore[no-untyped-def]
        parts = pending.get(prefix)
        if not parts:
            return
        qkv = parts.get("in_proj_qkv")
        z = parts.get("in_proj_z")
        if qkv is not None and z is not None:
            packed = _pack_qkv_z(config, qkv, z)  # type: ignore[arg-type]
            parts.pop("in_proj_qkv", None)
            parts.pop("in_proj_z", None)
            yield f"{prefix}.in_proj_qkvz.weight", packed
        b = parts.get("in_proj_b")
        a = parts.get("in_proj_a")
        if b is not None and a is not None:
            packed = _pack_b_a(config, b, a)  # type: ignore[arg-type]
            parts.pop("in_proj_b", None)
            parts.pop("in_proj_a", None)
            yield f"{prefix}.in_proj_ba.weight", packed
        if not parts:
            pending.pop(prefix, None)

    for name, tensor in weights:
        # 1. Skip visual encoder
        if name.startswith("model.visual.") or name.startswith("visual."):
            continue

        # 2. Strip language_model prefix
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        elif name.startswith("language_model."):
            name = "model." + name[len("language_model."):]

        # 3. Intercept linear-attention split weights for packing
        match = _LINEAR_ATTN_SPLIT_RE.match(name)
        if match:
            prefix = match.group("prefix")
            part = match.group("part")
            bucket = pending.setdefault(prefix, {})
            if part in bucket:
                raise ValueError(f"Duplicate linear attention weight: {prefix}.{part}")
            bucket[part] = tensor
            yield from _drain(prefix)
            continue

        yield name, tensor

    # Flush any incomplete pairs (should not happen for valid checkpoints)
    if pending:
        descriptions = []
        for prefix, parts in sorted(pending.items()):
            present = frozenset(parts)
            missing: set[str] = set()
            for pair in _LINEAR_ATTN_PAIRS:
                if present & pair:
                    missing.update(pair - present)
            descriptions.append(f"{prefix} present={sorted(present)!r} missing={sorted(missing)!r}")
        raise ValueError(
            "Incomplete Qwen3Next linear attention split weights: " + "; ".join(descriptions)
        )


# ---------------------------------------------------------------------------
# Class creation (lazy — only when vLLM is importable)
# ---------------------------------------------------------------------------

def _create_mint_qwen3next_cls() -> type[nn.Module]:
    """Create MintQwen3NextForCausalLM by subclassing vLLM's Qwen3NextForCausalLM.

    Called lazily from ``register()`` so that importing this module does not
    require vLLM to be installed (CI pyright gate compatibility).
    """
    import importlib
    qwen3_next_mod = importlib.import_module("vllm.model_executor.models.qwen3_next")
    Qwen3NextForCausalLM = qwen3_next_mod.Qwen3NextForCausalLM

    class MintQwen3NextForCausalLM(Qwen3NextForCausalLM):  # type: ignore[misc, valid-type]
        """Qwen3NextForCausalLM with checkpoint weight-name mapping.

        Override ``load_weights`` to:
        - Strip ``model.language_model.`` → ``model.`` prefix
        - Pack split linear-attention weights (``in_proj_qkv`` + ``in_proj_z`` → ``in_proj_qkvz``)

        Override ``set_moe_parameters`` to handle dense models (``num_experts == 0``)
        gracefully instead of raising "No Qwen3Next layer found".
        """

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            mapped = _map_checkpoint_weights(self.config, weights)
            return super().load_weights(mapped)

        def set_moe_parameters(self) -> None:
            num_experts = int(getattr(self.config, "num_experts", 0) or 0)
            if num_experts > 0:
                return super().set_moe_parameters()
            # Dense model: no MoE layers to initialize. Set empty defaults
            # instead of letting the base class raise "No Qwen3Next layer found".
            self.expert_weights = []
            self.moe_layers = []
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0

    return MintQwen3NextForCausalLM


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register() -> None:
    """Register MintQwen3NextForCausalLM, replacing the original Qwen3NextForCausalLM."""
    try:
        import importlib
        cls = _create_mint_qwen3next_cls()
        registry = importlib.import_module("vllm.model_executor.models.registry")
        registry.ModelRegistry.register_model("Qwen3NextForCausalLM", cls)
    except Exception:
        # vLLM might not be importable in all contexts (e.g. API server
        # process without qwen36-vllm-deps, or CI pyright gate). Silently skip.
        pass


# Auto-register on import (triggered by import appended to vllm/__init__.py
# in qwen36-vllm-deps).
register()
