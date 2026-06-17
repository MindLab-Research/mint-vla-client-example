"""Qwen3.5 text-only vLLM adapter.

This module formalizes Mint's current Qwen3.5 support boundary:
serve the language-model/text portion of Qwen3.5 checkpoints through vLLM's
Qwen3Next text implementation. It intentionally does not support visual
encoders, image/video tokens, multimodal preprocessing, or M-RoPE.
"""

from __future__ import annotations

import hashlib
import json
import structlog
import os
import re
from pathlib import Path
from typing import Any

logger = structlog.get_logger(__name__)

QWEN35_MODEL_TYPE = "qwen3_5"
QWEN35_TEXT_MODEL_TYPE = "qwen3_5_text"
QWEN35_VLLM_ARCHITECTURE = "Qwen3NextForCausalLM"
QWEN35_TEXT_ONLY_SHIM_MARKER = "mint_qwen35_text_only_shim"
QWEN35_BUMBLEBEE_TEXT_ONLY_SHIM_MARKER = "bumblebee_qwen35_text_only_shim"
QWEN35_SUPPORTED_MODALITY = "text_only"

_QWEN35_LINEAR_ATTN_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+\.linear_attn)\."
    r"(?P<part>in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)\.weight$"
)
_QWEN35_REQUIRED_TEXT_CONFIG_FIELDS = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "hidden_act",
        "rms_norm_eps",
        "head_dim",
        "max_position_embeddings",
        "rope_parameters",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "layer_types",
    }
)
_QWEN35_REQUIRED_POSITIVE_INT_TEXT_CONFIG_FIELDS = frozenset(
    _QWEN35_REQUIRED_TEXT_CONFIG_FIELDS
    - {"hidden_act", "rms_norm_eps", "rope_parameters", "layer_types"}
)
_QWEN35_LINEAR_ATTN_PAIR_PARTS = (
    frozenset({"in_proj_qkv", "in_proj_z"}),
    frozenset({"in_proj_b", "in_proj_a"}),
)


def normalize_hf_dtype_str(value: Any) -> str | None:
    if value is None:
        return None
    dtype_str = str(value).replace("torch.", "").strip().lower()
    if dtype_str in ("fp16", "float16", "half"):
        return "float16"
    if dtype_str in ("bf16", "bfloat16"):
        return "bfloat16"
    if dtype_str in ("fp32", "float32"):
        return "float32"
    return None


def read_hf_config_json(model_path: str) -> dict[str, Any] | None:
    config_dir = resolve_hf_config_dir(model_path)
    if config_dir is None:
        return None
    config_path = os.path.join(config_dir, "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = json.load(f)
    except OSError:
        return None
    except json.JSONDecodeError:
        logger.warning("unable_to_parse_hf_config_json___s")
        return None
    return raw_config if isinstance(raw_config, dict) else None


def resolve_hf_config_dir(model_path: str) -> str | None:
    """Resolve a local HF config directory for a path or cached repo id."""

    path = Path(model_path).expanduser()
    if (path / "config.json").is_file():
        return str(path)

    repo_id = str(model_path).strip()
    if not _looks_like_hf_repo_id(repo_id):
        return None

    revision = "main"
    if "@" in repo_id:
        repo_id, revision = repo_id.rsplit("@", 1)
    cache_name = "models--" + repo_id.replace("/", "--")
    for cache_root in _hf_cache_roots():
        repo_cache = cache_root / cache_name
        if not repo_cache.is_dir():
            continue
        snapshot = _resolve_hf_snapshot_dir(repo_cache, revision)
        if snapshot is not None and (snapshot / "config.json").is_file():
            return str(snapshot)
    return None


def infer_hf_torch_dtype_str(model_path: str) -> str | None:
    raw_config = read_hf_config_json(model_path)

    if isinstance(raw_config, dict):
        for key in ("torch_dtype", "dtype"):
            inferred = normalize_hf_dtype_str(raw_config.get(key))
            if inferred is not None:
                return inferred

        text_config = raw_config.get("text_config")
        if isinstance(text_config, dict):
            for key in ("torch_dtype", "dtype"):
                inferred = normalize_hf_dtype_str(text_config.get(key))
                if inferred is not None:
                    return inferred

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.debug(
            "AutoConfig dtype inference failed for model=%r: %s",
            model_path,
            exc,
            exc_info=True,
        )
        return None

    for attr in ("torch_dtype", "dtype"):
        inferred = normalize_hf_dtype_str(getattr(cfg, attr, None))
        if inferred is not None:
            return inferred
    text_config = getattr(cfg, "text_config", None)
    if text_config is not None:
        for attr in ("torch_dtype", "dtype"):
            inferred = normalize_hf_dtype_str(getattr(text_config, attr, None))
            if inferred is not None:
                return inferred
    return None


def is_qwen35_config(raw_config: dict[str, Any] | None) -> bool:
    return _qwen35_text_config(raw_config) is not None


def _qwen35_text_config(raw_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw_config, dict):
        return None
    if raw_config.get("model_type") != QWEN35_MODEL_TYPE:
        return None
    text_config = raw_config.get("text_config")
    if not isinstance(text_config, dict):
        return None
    if text_config.get("model_type") != QWEN35_TEXT_MODEL_TYPE:
        return None
    return text_config


def _looks_like_hf_repo_id(value: str) -> bool:
    if not value or value.startswith(("/", ".", "file:", "mint:", "s3:", "gs:")):
        return False
    if os.path.sep in value and value.count("/") != 1:
        return False
    namespace, sep, name = value.partition("/")
    return bool(sep and namespace and name)


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key, "").strip()
        if value:
            roots.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path("/vePFS-Mindverse/share/huggingface/hub"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root_str = str(root)
        if root_str not in seen:
            deduped.append(root)
            seen.add(root_str)
    return deduped


def _resolve_hf_snapshot_dir(repo_cache: Path, revision: str) -> Path | None:
    revision = revision.strip() or "main"
    ref_path = repo_cache / "refs" / revision
    if ref_path.is_file():
        commit = ref_path.read_text(encoding="utf-8").strip()
        if commit:
            return repo_cache / "snapshots" / commit

    direct_snapshot = repo_cache / "snapshots" / revision
    if direct_snapshot.is_dir():
        return direct_snapshot

    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _validate_qwen35_text_config(text_config: dict[str, Any]) -> None:
    missing = sorted(_QWEN35_REQUIRED_TEXT_CONFIG_FIELDS - set(text_config))
    if missing:
        raise ValueError(
            "Qwen3.5 text-only adapter requires text_config fields: "
            + ", ".join(missing)
        )

    for key in sorted(_QWEN35_REQUIRED_POSITIVE_INT_TEXT_CONFIG_FIELDS):
        value = text_config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(
                "Qwen3.5 text-only adapter requires positive integer "
                f"text_config.{key}, got {value!r}"
            )

    layer_types = text_config["layer_types"]
    if not isinstance(layer_types, list) or not all(isinstance(item, str) for item in layer_types):
        raise ValueError("Qwen3.5 text-only adapter requires text_config.layer_types to be a string list")
    unsupported_layer_types = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unsupported_layer_types:
        raise ValueError(
            "Qwen3.5 text-only adapter only supports linear_attention/full_attention layer_types, got "
            + ", ".join(unsupported_layer_types)
        )

    if len(layer_types) != text_config["num_hidden_layers"]:
        raise ValueError(
            "Qwen3.5 text-only adapter requires len(text_config.layer_types) "
            f"to match num_hidden_layers, got {len(layer_types)} != {text_config['num_hidden_layers']}"
        )

    if text_config["linear_num_value_heads"] % text_config["linear_num_key_heads"] != 0:
        raise ValueError(
            "Qwen3.5 text-only adapter requires "
            "linear_num_value_heads % linear_num_key_heads == 0, got "
            f"{text_config['linear_num_value_heads']} % {text_config['linear_num_key_heads']}"
        )

    if not isinstance(text_config["hidden_act"], str) or not text_config["hidden_act"].strip():
        raise ValueError("Qwen3.5 text-only adapter requires non-empty string text_config.hidden_act")
    if not isinstance(text_config["rms_norm_eps"], (int, float)) or text_config["rms_norm_eps"] <= 0:
        raise ValueError("Qwen3.5 text-only adapter requires positive text_config.rms_norm_eps")
    if not isinstance(text_config["rope_parameters"], dict):
        raise ValueError("Qwen3.5 text-only adapter requires dict text_config.rope_parameters")


def qwen35_text_as_qwen3next_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    text_config = _qwen35_text_config(raw_config)
    if text_config is None:
        raise ValueError(
            "Qwen3.5 text-only adapter requires outer model_type='qwen3_5' "
            "and text_config.model_type='qwen3_5_text'"
        )
    _validate_qwen35_text_config(text_config)

    config = dict(text_config)
    config["model_type"] = "qwen3_next"
    config["architectures"] = [QWEN35_VLLM_ARCHITECTURE]
    config[QWEN35_TEXT_ONLY_SHIM_MARKER] = True
    config[QWEN35_BUMBLEBEE_TEXT_ONLY_SHIM_MARKER] = True
    config["mint_source_model_type"] = QWEN35_MODEL_TYPE
    config["mint_supported_modality"] = QWEN35_SUPPORTED_MODALITY
    config["bumblebee_source_model_type"] = QWEN35_MODEL_TYPE
    config["bumblebee_supported_modality"] = QWEN35_SUPPORTED_MODALITY
    config["tie_word_embeddings"] = bool(raw_config.get("tie_word_embeddings", False))

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        rope_parameters = dict(rope_parameters)
        # Current support is text-only. Keeping M-RoPE fields makes vLLM route
        # into multimodal code paths that Qwen3NextForCausalLM does not support.
        rope_parameters.pop("mrope_section", None)
        rope_parameters.pop("mrope_interleaved", None)
        config["rope_parameters"] = rope_parameters

    # Qwen3.5-27B uses the Qwen3Next hybrid attention stack, but it is a dense
    # model. Current vLLM Qwen3Next code expects MoE metadata unless these are
    # explicitly dense/no-MoE.
    config["num_experts"] = 0
    config["num_experts_per_tok"] = 0
    config["moe_intermediate_size"] = 0
    config["shared_expert_intermediate_size"] = 0
    config.setdefault("decoder_sparse_step", 1)
    config.setdefault("mlp_only_layers", [])
    config.setdefault("norm_topk_prob", True)
    config.setdefault("output_router_logits", False)
    config.setdefault("router_aux_loss_coef", 0.001)

    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if key in raw_config and key not in config:
            config[key] = raw_config[key]
    return config


def materialize_qwen35_text_vllm_config(
    model_path: str,
    *,
    root_dir: str | None = None,
) -> str | None:
    config_source_dir = resolve_hf_config_dir(model_path)
    if config_source_dir is None:
        return None
    raw_config = read_hf_config_json(config_source_dir)
    if not isinstance(raw_config, dict):
        return None
    if raw_config.get("model_type") != QWEN35_MODEL_TYPE:
        return None
    vllm_config = qwen35_text_as_qwen3next_config(raw_config)
    digest = hashlib.sha1(
        json.dumps(vllm_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    if root_dir is None:
        root_dir = (
            os.environ.get("BUMBLEBEE_RUNTIME_CHECKPOINT_DIR")
            or os.environ.get("MINT_RUNTIME_CHECKPOINT_DIR")
            or "/tmp/mint-vllm-config"
        )
    config_dir = os.path.join(root_dir, "qwen35-text-vllm-config", digest)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vllm_config, f, indent=2, sort_keys=True)
        f.write("\n")
    return config_dir


def is_qwen35_text_only_shim_config(config: Any) -> bool:
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(
            config.get(QWEN35_TEXT_ONLY_SHIM_MARKER)
            or config.get(QWEN35_BUMBLEBEE_TEXT_ONLY_SHIM_MARKER)
        )
    return bool(
        getattr(config, QWEN35_TEXT_ONLY_SHIM_MARKER, False)
        or getattr(config, QWEN35_BUMBLEBEE_TEXT_ONLY_SHIM_MARKER, False)
    )


def is_qwen35_text_only_shim_model(model: Any) -> bool:
    return is_qwen35_text_only_shim_config(getattr(model, "config", None))


def install_vllm_qwen35_text_only_adapter_patches() -> None:
    """Install vLLM patches gated by the text-only shim marker."""

    patch_vllm_qwen3next_dense_moe_metadata()
    patch_vllm_qwen35_language_model_weight_prefix()


def patch_vllm_qwen3next_dense_moe_metadata() -> None:
    """Allow dense Qwen3Next initialization only for Qwen3.5 text shim configs."""

    try:
        import vllm.model_executor.models.qwen3_next as qwen3_next
    except Exception:
        logger.debug("Unable to import vLLM qwen3_next for dense metadata patch", exc_info=True)
        return

    mixin = getattr(qwen3_next, "QwenNextMixtureOfExperts", None)
    original = getattr(mixin, "set_moe_parameters", None)
    if not callable(original):
        logger.debug("vLLM qwen3_next has no QwenNextMixtureOfExperts.set_moe_parameters")
        return
    if getattr(original, "_mint_qwen35_text_dense_patch", False):
        return

    def _set_moe_parameters_allow_qwen35_text_dense(self):  # type: ignore[no-untyped-def]
        try:
            return original(self)
        except RuntimeError as exc:
            if "No Qwen3Next layer found" not in str(exc):
                raise
            if not is_qwen35_text_only_shim_model(self):
                raise
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

    _set_moe_parameters_allow_qwen35_text_dense._mint_qwen35_text_dense_patch = True  # type: ignore[attr-defined]
    mixin.set_moe_parameters = _set_moe_parameters_allow_qwen35_text_dense


def patch_vllm_qwen35_language_model_weight_prefix() -> None:
    """Map Qwen3.5 text checkpoint weights onto vLLM Qwen3Next names."""

    try:
        import vllm.model_executor.models.qwen3_next as qwen3_next
    except Exception:
        logger.debug("Unable to import vLLM qwen3_next for Qwen3.5 text weight patch", exc_info=True)
        return

    def _wrap_load_weights(cls, *, inner_model: bool = False):  # type: ignore[no-untyped-def]
        if cls is None:
            return
        original = getattr(cls, "load_weights", None)
        if not callable(original):
            logger.debug("vllm_qwen3_next_class_missing_load_weights", cls=cls)
            return
        if getattr(original, "_mint_qwen35_text_weight_patch", False):
            return

        def _load_weights_with_qwen35_text_adapter(self, weights):  # type: ignore[no-untyped-def]
            if not is_qwen35_text_only_shim_model(self):
                return original(self, weights)
            return original(self, _map_qwen35_text_weights(self.config, weights, inner_model=inner_model))

        _load_weights_with_qwen35_text_adapter._mint_qwen35_text_weight_patch = True  # type: ignore[attr-defined]
        cls.load_weights = _load_weights_with_qwen35_text_adapter

    _wrap_load_weights(getattr(qwen3_next, "Qwen3NextForCausalLM", None))
    _wrap_load_weights(getattr(qwen3_next, "Qwen3NextModel", None), inner_model=True)


def _map_qwen35_text_weights(config: Any, weights: Any, *, inner_model: bool):  # type: ignore[no-untyped-def]
    pending_linear_attn: dict[str, dict[str, object]] = {}

    def _pop_if_complete(prefix: str):  # type: ignore[no-untyped-def]
        parts = pending_linear_attn.get(prefix)
        if not parts:
            return

        qkv = parts.get("in_proj_qkv")
        z = parts.get("in_proj_z")
        if qkv is not None and z is not None:
            qkvz = _pack_qwen35_qkv_z(config, qkv, z)
            parts.pop("in_proj_qkv", None)
            parts.pop("in_proj_z", None)
            yield f"{prefix}.in_proj_qkvz.weight", qkvz

        b = parts.get("in_proj_b")
        a = parts.get("in_proj_a")
        if b is not None and a is not None:
            ba = _pack_qwen35_b_a(config, b, a)
            parts.pop("in_proj_b", None)
            parts.pop("in_proj_a", None)
            yield f"{prefix}.in_proj_ba.weight", ba

        if not parts:
            pending_linear_attn.pop(prefix, None)

    for name, loaded_weight in weights:
        if name.startswith("model.visual.") or name.startswith("visual."):
            continue
        if inner_model and name.startswith("model.language_model."):
            name = name[len("model.language_model.") :]
        elif inner_model and name.startswith("model."):
            name = name[len("model.") :]
        elif inner_model and name.startswith("language_model."):
            name = name[len("language_model.") :]
        elif name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith("language_model."):
            name = "model." + name[len("language_model.") :]

        if inner_model:
            match = _QWEN35_LINEAR_ATTN_SPLIT_RE.match(name)
            if match:
                prefix = match.group("prefix")
                part = match.group("part")
                parts = pending_linear_attn.setdefault(prefix, {})
                if part in parts:
                    raise ValueError(
                        "Duplicate Qwen3.5 linear attention split weight: "
                        f"{prefix}.{part}.weight"
                    )
                parts[part] = loaded_weight
                yield from _pop_if_complete(prefix)
                continue

        yield name, loaded_weight

    if pending_linear_attn:
        raise ValueError(
            "Incomplete Qwen3.5 linear attention split weights: "
            + "; ".join(_describe_incomplete_linear_attention_parts(pending_linear_attn))
        )


def _describe_incomplete_linear_attention_parts(
    pending_linear_attn: dict[str, dict[str, object]],
) -> list[str]:
    descriptions: list[str] = []
    for prefix, parts in sorted(pending_linear_attn.items()):
        present = frozenset(parts)
        missing: set[str] = set()
        for pair in _QWEN35_LINEAR_ATTN_PAIR_PARTS:
            if present & pair:
                missing.update(pair - present)
        descriptions.append(
            f"{prefix} present={sorted(present)!r} missing={sorted(missing)!r}"
        )
    return descriptions


def _pack_qwen35_qkv_z(config, qkv, z):  # type: ignore[no-untyped-def]
    import torch

    hidden_size = qkv.shape[1]
    qk_heads = config.linear_num_key_heads
    qk_head_dim = config.linear_key_head_dim
    value_heads = config.linear_num_value_heads
    value_head_dim = config.linear_value_head_dim
    if value_heads % qk_heads != 0:
        raise ValueError(
            "Qwen3.5 linear attention packing requires "
            f"linear_num_value_heads % linear_num_key_heads == 0, got "
            f"{value_heads} % {qk_heads}"
        )
    qk_dim = qk_heads * qk_head_dim
    value_dim = value_heads * value_head_dim
    expected_qkv_shape = (qk_dim * 2 + value_dim, hidden_size)
    expected_z_shape = (value_dim, hidden_size)
    if tuple(qkv.shape) != expected_qkv_shape:
        raise ValueError(
            "Unexpected Qwen3.5 in_proj_qkv weight shape: "
            f"got={tuple(qkv.shape)} expected={expected_qkv_shape}"
        )
    if tuple(z.shape) != expected_z_shape:
        raise ValueError(
            "Unexpected Qwen3.5 in_proj_z weight shape: "
            f"got={tuple(z.shape)} expected={expected_z_shape}"
        )
    q, k, v = torch.split(qkv, [qk_dim, qk_dim, value_dim], dim=0)
    q, k = [weight.reshape(qk_heads, qk_head_dim, hidden_size) for weight in (q, k)]
    v = v.reshape(qk_heads, value_heads // qk_heads * value_head_dim, hidden_size)
    z = z.reshape(qk_heads, value_heads // qk_heads * value_head_dim, hidden_size)
    return torch.cat([q, k, v, z], dim=1).reshape(-1, hidden_size)


def _pack_qwen35_b_a(config, b, a):  # type: ignore[no-untyped-def]
    import torch

    hidden_size = b.shape[1]
    qk_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    if value_heads % qk_heads != 0:
        raise ValueError(
            "Qwen3.5 linear attention packing requires "
            f"linear_num_value_heads % linear_num_key_heads == 0, got "
            f"{value_heads} % {qk_heads}"
        )
    expected_shape = (value_heads, hidden_size)
    if tuple(b.shape) != expected_shape:
        raise ValueError(
            "Unexpected Qwen3.5 in_proj_b weight shape: "
            f"got={tuple(b.shape)} expected={expected_shape}"
        )
    if tuple(a.shape) != expected_shape:
        raise ValueError(
            "Unexpected Qwen3.5 in_proj_a weight shape: "
            f"got={tuple(a.shape)} expected={expected_shape}"
        )
    b = b.reshape(qk_heads, value_heads // qk_heads, hidden_size)
    a = a.reshape(qk_heads, value_heads // qk_heads, hidden_size)
    return torch.cat([b, a], dim=1).reshape(-1, hidden_size)
