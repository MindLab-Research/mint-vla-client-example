from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .sampling_backend import SamplingBackendCapabilityDecision, SamplingBackendUnsupportedFeatureError

SGLANG_SELECTED_VERSION = os.environ.get("MINT_SGLANG_SELECTED_VERSION", "0.5.12.post1")
SGLANG_QWEN3_MOE_EXPERT_LORA_FEATURE = "qwen3_moe_per_expert_mlp_lora"


class SGLangUnsupportedFeatureError(SamplingBackendUnsupportedFeatureError):
    """The selected SGLang backend cannot safely satisfy this request."""

    def __init__(self, decision: SamplingBackendCapabilityDecision | str, message: str | None = None) -> None:
        if isinstance(decision, SamplingBackendCapabilityDecision):
            super().__init__(decision, message=message)
            return
        super().__init__(
            SamplingBackendCapabilityDecision(
                backend="sglang",
                feature="unknown",
                supported=False,
                reason=str(decision),
            ),
            message=message or str(decision),
        )


def canonical_peft_adapter_path(lora_path: str) -> str:
    root = Path(str(lora_path)).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"SGLang LoRA adapter path is not a directory: {root}")
    required = ("adapter_model.safetensors", "adapter_config.json")
    for filename in required:
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f"SGLang LoRA adapter missing {filename}: {path}")
        if not os.access(path, os.R_OK):
            raise RuntimeError(f"SGLang LoRA adapter file is not readable: {path}")
    return str(root.resolve())


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file {path}: {type(e).__name__}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _qwen3_moe_config(model_path: str) -> dict[str, Any] | None:
    config = _read_json_object(Path(str(model_path)) / "config.json")
    if not config:
        return None
    if str(config.get("model_type") or "").strip().lower() != "qwen3_moe":
        return None
    return config


def _first_moe_expert_lora_tensor(adapter_model_path: Path) -> tuple[str, tuple[int, ...]] | None:
    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(f"Failed to import safetensors for SGLang LoRA validation: {type(e).__name__}: {e}") from e

    with safe_open(str(adapter_model_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            lowered = str(key).lower()
            if ".mlp.experts." not in lowered or ".lora_" not in lowered:
                continue
            shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
            return str(key), shape
    return None


def check_sglang_lora_adapter_support(
    *,
    model_name: str,
    model_path: str,
    adapter_path: str,
) -> SamplingBackendCapabilityDecision:
    canonical_adapter_path = canonical_peft_adapter_path(adapter_path)
    model_config = _qwen3_moe_config(model_path)
    if model_config is None:
        return SamplingBackendCapabilityDecision(
            backend="sglang",
            feature="path_peft_lora",
            supported=True,
            model_name=str(model_name),
            model_path=str(model_path),
            adapter_path=canonical_adapter_path,
        )

    adapter_model_path = Path(canonical_adapter_path) / "adapter_model.safetensors"
    expert_tensor = _first_moe_expert_lora_tensor(adapter_model_path)
    if expert_tensor is None:
        return SamplingBackendCapabilityDecision(
            backend="sglang",
            feature="path_peft_lora",
            supported=True,
            model_name=str(model_name),
            model_path=str(model_path),
            adapter_path=canonical_adapter_path,
            evidence={"model_type": model_config.get("model_type")},
        )

    key, shape = expert_tensor
    return SamplingBackendCapabilityDecision(
        backend="sglang",
        feature=SGLANG_QWEN3_MOE_EXPERT_LORA_FEATURE,
        supported=True,
        model_name=str(model_name),
        model_path=str(model_path),
        adapter_path=canonical_adapter_path,
        evidence={
            "sglang_version": SGLANG_SELECTED_VERSION,
            "model_type": model_config.get("model_type"),
            "intermediate_size": model_config.get("intermediate_size"),
            "moe_intermediate_size": model_config.get("moe_intermediate_size"),
            "example_tensor": key,
            "example_shape": shape,
        },
    )


def validate_sglang_lora_adapter_supported(
    *,
    model_name: str,
    model_path: str,
    adapter_path: str,
) -> SamplingBackendCapabilityDecision:
    decision = check_sglang_lora_adapter_support(
        model_name=model_name,
        model_path=model_path,
        adapter_path=adapter_path,
    )
    if not decision.supported:
        raise SGLangUnsupportedFeatureError(decision)
    return decision
