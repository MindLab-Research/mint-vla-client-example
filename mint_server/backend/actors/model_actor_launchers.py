from __future__ import annotations

import json
import os
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Protocol


class ModelActorSpecLike(Protocol):
    domain_key: str
    replica_id: str
    gpu_count: int | None
    placement_slices: tuple[tuple[str, str, int], ...]
    node_pin: str | None
    node_pins: tuple[str, ...]

    def normalized_actor_name(self) -> str: ...

    def normalized_node_pins(self) -> list[str]: ...


ModelActorLauncher = Callable[..., Any | Awaitable[Any]]
DEFAULT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET = 262144
MODEL_RUNTIME_LAUNCHER_ENV_KEYS = (
    "MINT_MODEL_RUNTIME_MAX_CLAIM",
    "MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM",
    "MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM",
    "MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM",
    "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
    "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET",
    "MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET",
    "MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET",
    "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
)
SGLANG_RUNTIME_ENV_KEYS = (
    "MINT_SGLANG_MODEL_PLACEMENT_JSON",
    "MINT_SGLANG_PYTHONPATH",
    "MINT_SGLANG_PY_EXECUTABLE",
    "MINT_SGLANG_MEM_FRACTION_STATIC",
    "MINT_SGLANG_MAX_RUNNING_REQUESTS",
    "MINT_SGLANG_MAX_TOTAL_TOKENS",
    "MINT_SGLANG_MAX_PREFILL_TOKENS",
    "MINT_SGLANG_CHUNKED_PREFILL_SIZE",
    "MINT_SGLANG_DISABLE_CUDA_GRAPH",
    "MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH",
    "MINT_SGLANG_DISABLE_FLASHINFER_KERNELS",
    "MINT_SGLANG_DISABLE_FLASHINFER_AUTOTUNE",
    "MINT_SGLANG_DISABLE_CUSTOM_ALL_REDUCE",
    "MINT_SGLANG_SKIP_SERVER_WARMUP",
    "MINT_SGLANG_TRUST_REMOTE_CODE",
    "MINT_SGLANG_LOG_LEVEL",
    "MINT_SGLANG_SHOW_TIME_COST",
    "MINT_SGLANG_WATCHDOG_TIMEOUT_S",
    "MINT_SGLANG_ENGINE_READY_WAIT_S",
    "MINT_SGLANG_REQUEST_TIMEOUT_S",
    "MINT_SGLANG_GENERATE_TIMEOUT_S",
    "MINT_SGLANG_LOGPROBS_TIMEOUT_S",
    "MINT_SGLANG_TOPK_TIMEOUT_S",
    "MINT_SGLANG_SLOW_REQUEST_LOG_S",
    "MINT_SGLANG_PYCACHE_PREFIX",
    "MINT_SGLANG_ACTOR_MAX_CONCURRENCY",
    "MINT_SGLANG_ENABLE_LORA",
    "MINT_SGLANG_ENABLE_LORA_OVERLAP_LOADING",
    "MINT_SGLANG_MAX_LORA_RANK",
    "MINT_SGLANG_LORA_TARGET_MODULES",
    "MINT_SGLANG_MAX_LOADED_LORAS",
    "MINT_SGLANG_MAX_LORAS_PER_BATCH",
    "MINT_SGLANG_LORA_BACKEND",
)
BUMBLEBEE_RUNTIME_ENV_KEYS = (
    "MINT_BUMBLEBEE_REPO_PATH",
    "MINT_BUMBLEBEE_MODEL_NAME",
    "MINT_BUMBLEBEE_MEGATRON_LM_PATH",
    "MINT_BUMBLEBEE_IMPL",
    "MINT_BUMBLEBEE_OPTIMIZER",
    "MINT_BUMBLEBEE_SKIP_HF_LOAD",
    "MINT_BUMBLEBEE_ATTENTION_BACKEND",
    "MINT_BUMBLEBEE_FLASH_ATTN_OVERLAY_PATH",
    "MINT_BUMBLEBEE_LORA_ALPHA",
    "BUMBLEBEE_QWEN35_MEGATRON_VENDOR_PATH",
    "BUMBLEBEE_BUILD_TRACE",
    "BUMBLEBEE_CKPT_TRACE",
    "BUMBLEBEE_MEMORY_TRACE",
    "BUMBLEBEE_RL_DEBUG_METRICS",
    "BUMBLEBEE_LITE_TRACE",
    "BUMBLEBEE_LITE_TRACE_ALL_RANKS",
    "BUMBLEBEE_LITE_TRACE_RANKS",
    "BUMBLEBEE_LITE_TRACE_BY_STEP",
    "BUMBLEBEE_LITE_TRACE_MAX_SHAPES",
    "BUMBLEBEE_Q3MOE_GQA_PROBE",
    "BUMBLEBEE_Q3MOE_GQA_PROBE_ALL_RANKS",
    "MINT_BENCH_RECORD_LOGPROBS",
    "MINT_BENCH_RECORD_LOGITS",
    "MINT_BENCH_RECORD_TOPK",
    "MINT_BENCH_RECORD_INPUTS",
    "MINT_BENCH_RECORD_MODEL_STATE",
    "MINT_BENCH_OUTPUT_DIR",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_DISTRIBUTED_DEBUG",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_DEBUG",
    "NVTE_DEBUG_LEVEL",
    "BUMBLEBEE_TE_SDPA_FALLBACK",
    "MINT_VERL_LOCAL_ATTENTION_PATCHES",
)


@dataclass(frozen=True)
class ModelActorLauncherRegistry:
    _launchers: dict[str, ModelActorLauncher]

    def resolve(self, launcher_key: str) -> ModelActorLauncher:
        key = str(launcher_key or "").strip()
        if not key:
            raise ValueError("model actor launcher_key is required")
        try:
            return self._launchers[key]
        except KeyError as e:
            raise ValueError(f"unknown model actor launcher_key: {key!r}") from e

    async def launch(
        self,
        spec: ModelActorSpecLike,
        generation: int,
        *,
        launcher_key: str,
        ray_address: str | None = None,
    ) -> Any:
        launcher = self.resolve(launcher_key)
        try:
            value = launcher(spec, generation, ray_address=ray_address)
        except TypeError as exc:
            if "ray_address" not in str(exc):
                raise
            value = launcher(spec, generation)
        if isawaitable(value):
            return await value
        return value


def _replica_int(replica_id: str) -> int:
    raw = str(replica_id).strip()
    if raw.startswith("replica-"):
        raw = raw.removeprefix("replica-")
    try:
        return int(raw)
    except Exception:
        return 0


def _base_model_from_spec(spec: ModelActorSpecLike) -> str | None:
    base_model = getattr(spec, "base_model", None)
    if base_model:
        return str(base_model)
    from mint_server.backend.actors.domain_keys import base_model_from_domain_key

    return base_model_from_domain_key(str(getattr(spec, "domain_key", "") or ""))


def placement_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    base_model = _base_model_from_spec(spec)
    if not base_model or spec.gpu_count is None:
        return {}
    domain_key = str(getattr(spec, "domain_key", "") or "")
    is_sglang = domain_key.startswith("sglang:")
    is_bumblebee = domain_key.startswith("bumblebee:")
    if spec.placement_slices:
        placement_value = [
            {
                "replica": _replica_int(replica_id),
                "node_ip": node_ip,
                "gpu_count": int(gpu_count),
            }
            for replica_id, node_ip, gpu_count in spec.placement_slices
        ]
        placement_raw = json.dumps({base_model: placement_value}, sort_keys=True, separators=(",", ":"))
        node_pins = spec.normalized_node_pins()
        out = {
            "MINT_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        }
        if is_sglang:
            out["MINT_SGLANG_MODEL_PLACEMENT_JSON"] = placement_raw
        elif is_bumblebee:
            out["MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON"] = placement_raw
        else:
            out["MINT_VLLM_MODEL_PLACEMENT_JSON"] = placement_raw
        return out
    node_pins = spec.normalized_node_pins()
    if len(node_pins) > 1:
        placement_raw = json.dumps(
            {
                base_model: [
                    {
                        "replica": _replica_int(spec.replica_id),
                        "node_ip": node_ip,
                        "gpu_count": int(spec.gpu_count),
                    }
                    for node_ip in node_pins
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        out = {
            "MINT_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        }
        if is_sglang:
            out["MINT_SGLANG_MODEL_PLACEMENT_JSON"] = placement_raw
        elif is_bumblebee:
            out["MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON"] = placement_raw
        else:
            out["MINT_VLLM_MODEL_PLACEMENT_JSON"] = placement_raw
        return out
    if len(node_pins) != 1:
        return {}
    placement_raw = json.dumps(
        {
            base_model: {
                "replica": _replica_int(spec.replica_id),
                "node_ip": node_pins[0],
                "gpu_count": int(spec.gpu_count),
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    out = {
        "MINT_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
    }
    if is_sglang:
        out["MINT_SGLANG_MODEL_PLACEMENT_JSON"] = placement_raw
    elif is_bumblebee:
        out["MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON"] = placement_raw
    else:
        out["MINT_VLLM_MODEL_PLACEMENT_JSON"] = placement_raw
    return out


def megatron_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    from mint_server.backend.actors.domain_keys import is_megatron_domain

    if not is_megatron_domain(str(getattr(spec, "domain_key", "") or "")):
        return {}
    out: dict[str, str] = {}
    for key in (
        "MINT_MEGATRON_ATTENTION_BACKEND",
        "MINT_MEGATRON_DISABLE_WINDOW_SIZE",
        "NVTE_FLASH_ATTN",
        "NVTE_FUSED_ATTN",
        "NVTE_UNFUSED_ATTN",
    ):
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
    return out


def sglang_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    if not str(getattr(spec, "domain_key", "") or "").startswith("sglang:"):
        return {}
    out: dict[str, str] = {}
    for key in SGLANG_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
    return out


def bumblebee_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    if not str(getattr(spec, "domain_key", "") or "").startswith("bumblebee:"):
        return {}
    out: dict[str, str] = {}
    for key in BUMBLEBEE_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
    return out


def launcher_process_env() -> dict[str, str]:
    """Environment the supervisor needs in order to launch backend runtimes."""
    out: dict[str, str] = {}
    for key in (*MODEL_RUNTIME_LAUNCHER_ENV_KEYS, *SGLANG_RUNTIME_ENV_KEYS, *BUMBLEBEE_RUNTIME_ENV_KEYS):
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
    return out


def _model_runtime_max_claim_for_spec(spec: ModelActorSpecLike) -> int:
    from mint_server.backend.actors.domain_keys import is_vllm_domain, is_megatron_domain, is_bumblebee_domain

    domain_key = str(getattr(spec, "domain_key", "") or "")
    if is_vllm_domain(domain_key):
        return max(1, int(os.environ.get("MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM", "64")))
    if domain_key.startswith("sglang:"):
        return max(1, int(os.environ.get("MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM", "1")))
    if is_megatron_domain(domain_key) or is_bumblebee_domain(domain_key):
        return max(1, int(os.environ.get("MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM", "16")))
    return max(1, int(os.environ.get("MINT_MODEL_RUNTIME_MAX_CLAIM", "1")))


def _positive_env_int(*keys: str) -> int | None:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _model_runtime_token_budget_for_spec(spec: ModelActorSpecLike) -> int | None:
    from mint_server.backend.actors.domain_keys import is_vllm_domain, is_bumblebee_domain, is_megatron_domain

    domain_key = str(getattr(spec, "domain_key", "") or "")
    if is_vllm_domain(domain_key):
        return None
    if is_bumblebee_domain(domain_key):
        return _positive_env_int(
            "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
        ) or DEFAULT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET
    if is_megatron_domain(domain_key):
        return _positive_env_int(
            "MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
        )
    if domain_key.startswith("sglang:"):
        return _positive_env_int(
            "MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
        )
    return _positive_env_int("MINT_MODEL_RUNTIME_TOKEN_BUDGET")


async def launch_model_engine_host(
    spec: ModelActorSpecLike,
    generation: int,
    *,
    ray_address: str | None = None,
) -> Any:
    from mint_server.backend.actors.model_engine_host import get_or_create_model_engine_host

    runtime_env_extra: dict[str, str] = {
        **placement_env_for_spec(spec),
        **megatron_env_for_spec(spec),
        **sglang_env_for_spec(spec),
        **bumblebee_env_for_spec(spec),
    }

    # Auto-enable verl local-attention patches when the model's serving backend
    # is SGLang.  Training workers (bumblebee/megatron) need these patches to
    # produce position_ids and 4-D attention masks compatible with Megatron
    # local attention, which SGLang uses.  Non-SGLang models keep the original
    # 2-tuple return from preprocess_thd_no_padding and no BSHD mask transform.
    # Respect an explicit env var override if already set (from bumblebee_env_for_spec
    # for bumblebee: domain keys, or from os.environ for other domain keys).
    if "MINT_VERL_LOCAL_ATTENTION_PATCHES" not in runtime_env_extra:
        explicit = os.environ.get("MINT_VERL_LOCAL_ATTENTION_PATCHES")
        if explicit is not None:
            runtime_env_extra["MINT_VERL_LOCAL_ATTENTION_PATCHES"] = explicit
        else:
            base_model = _base_model_from_spec(spec)
            if base_model:
                try:
                    from mint_server.backend.core.model_registry import get_model_config
                    cfg = get_model_config(base_model)
                    if getattr(cfg, "serving_backend", "vllm") == "sglang":
                        runtime_env_extra["MINT_VERL_LOCAL_ATTENTION_PATCHES"] = "1"
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "could_not_auto_enable_verl_local_attention_patches model=%s error=%s: %s",
                        base_model, type(e).__name__, e,
                    )

    return get_or_create_model_engine_host(
        domain_key=spec.domain_key,
        replica_id=spec.replica_id,
        actor_name=spec.normalized_actor_name(),
        actor_generation=int(generation),
        base_model=_base_model_from_spec(spec),
        max_claim=_model_runtime_max_claim_for_spec(spec),
        token_budget=_model_runtime_token_budget_for_spec(spec),
        ray_address=ray_address,
        runtime_env_extra=runtime_env_extra,
    )


def default_model_actor_launcher_registry() -> ModelActorLauncherRegistry:
    launchers = {
        "cpu_runtime": launch_model_engine_host,
        "training": launch_model_engine_host,
        "vllm": launch_model_engine_host,
        "sglang": launch_model_engine_host,
        "model_runtime": launch_model_engine_host,
    }
    return ModelActorLauncherRegistry(launchers)
