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
    domain_key = str(getattr(spec, "domain_key", "") or "")
    if domain_key.startswith("vllm:"):
        model = domain_key.removeprefix("vllm:").strip()
        return model or None
    if domain_key.startswith("training:"):
        model = domain_key.removeprefix("training:").strip()
        return model or None
    return None


def placement_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    base_model = _base_model_from_spec(spec)
    if not base_model or spec.gpu_count is None:
        return {}
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
        return {
            "MINT_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_VLLM_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        }
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
        return {
            "MINT_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_VLLM_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
            "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        }
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
    return {
        "MINT_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_VLLM_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_DENSE_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
    }


def megatron_env_for_spec(spec: ModelActorSpecLike) -> dict[str, str]:
    if not str(getattr(spec, "domain_key", "") or "").startswith("megatron:"):
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


def _model_runtime_max_claim_for_spec(spec: ModelActorSpecLike) -> int:
    domain_key = str(getattr(spec, "domain_key", "") or "")
    if domain_key.startswith("vllm:"):
        return max(1, int(os.environ.get("MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM", "64")))
    if domain_key.startswith(("bumblebee:", "megatron:")):
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
    domain_key = str(getattr(spec, "domain_key", "") or "")
    if domain_key.startswith("vllm:"):
        return None
    if domain_key.startswith("bumblebee:"):
        return _positive_env_int(
            "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
        ) or DEFAULT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET
    if domain_key.startswith("megatron:"):
        return _positive_env_int(
            "MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET",
            "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
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

    return get_or_create_model_engine_host(
        domain_key=spec.domain_key,
        replica_id=spec.replica_id,
        actor_name=spec.normalized_actor_name(),
        actor_generation=int(generation),
        base_model=_base_model_from_spec(spec),
        max_claim=_model_runtime_max_claim_for_spec(spec),
        token_budget=_model_runtime_token_budget_for_spec(spec),
        ray_address=ray_address,
        runtime_env_extra={
            **placement_env_for_spec(spec),
            **megatron_env_for_spec(spec),
        },
    )


def default_model_actor_launcher_registry() -> ModelActorLauncherRegistry:
    launchers = {
        "cpu_runtime": launch_model_engine_host,
        "training": launch_model_engine_host,
        "vllm": launch_model_engine_host,
        "model_runtime": launch_model_engine_host,
    }
    return ModelActorLauncherRegistry(launchers)
