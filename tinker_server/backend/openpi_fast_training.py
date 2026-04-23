from __future__ import annotations

import base64
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..models.types import AdamParams
from .model_registry import ModelConfig, get_model_config
from .openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from .openpi_ray_runtime import ensure_openpi_ray_initialized
from .openpi_shared_ray_runtime import start_openpi_shared_ray_runtime


logger = logging.getLogger(__name__)

OPENPI_FAST_TRAINING_BACKEND = "openpi_fast"
OPENPI_FAST_LORA_RANK = 16
OPENPI_FAST_CONFIG_NAMES = {
    "openpi/pi0-fast-libero-low-mem-finetune": "pi0_fast_libero_low_mem_finetune",
}


def _is_openpi_fast_model(base_model: str) -> bool:
    try:
        return get_model_config(base_model).training_backend == OPENPI_FAST_TRAINING_BACKEND
    except Exception:
        return False


def get_openpi_fast_config_name(base_model: str) -> str:
    try:
        return OPENPI_FAST_CONFIG_NAMES[base_model]
    except KeyError as exc:
        raise ValueError(f"No OpenPI FAST config mapping registered for {base_model!r}") from exc


def validate_openpi_fast_create_request(request: Any) -> None:
    base_model = str(getattr(request, "base_model", "") or "")
    if not _is_openpi_fast_model(base_model):
        return

    lora_config = getattr(request, "lora_config", None)
    if lora_config is None:
        raise ValueError("OpenPI FAST training requires lora_config")
    if int(lora_config.rank) != OPENPI_FAST_LORA_RANK:
        raise ValueError(
            "OpenPI FAST training only supports the upstream LoRA rank "
            f"{OPENPI_FAST_LORA_RANK}"
        )

    for field in ("train_attn", "train_mlp", "train_unembed"):
        if getattr(lora_config, field, True) is not True:
            raise ValueError(
                "OpenPI FAST training does not support partial LoRA toggle mapping; "
                f"expected {field}=True"
            )

    if getattr(request, "rollout_correction_config", None) is not None:
        raise ValueError("OpenPI FAST ST-02 only supports SFT create_model requests")


def _tensor_values(loss_fn_inputs: dict[str, Any], key: str) -> list[Any]:
    value = loss_fn_inputs.get(key)
    if value is None:
        raise ValueError(f"Missing loss_fn_inputs[{key!r}]")

    if hasattr(value, "data"):
        data = value.data
    elif isinstance(value, dict) and "data" in value:
        data = value["data"]
    else:
        raise ValueError(f"loss_fn_inputs[{key!r}] must provide tensor-style data")

    if isinstance(data, list):
        return data
    return [data]


def _binary_loss_mask(weights: list[Any]) -> list[bool]:
    mask: list[bool] = []
    for weight in weights:
        value = float(weight)
        if value not in (0.0, 1.0):
            raise ValueError("OpenPI FAST SFT weights must be binary 0/1 values")
        mask.append(bool(value))
    return mask


def _build_openpi_fast_common_payload(
    *,
    datum: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    if model_config.training_backend != OPENPI_FAST_TRAINING_BACKEND:
        raise ValueError(
            "build_openpi_fast_sft_runtime_payload only supports OpenPI FAST model configs"
        )

    image_chunks = [chunk for chunk in datum.model_input.chunks if chunk.type == "image"]
    text_chunks = [chunk for chunk in datum.model_input.chunks if chunk.type == "encoded_text"]
    other_chunks = [
        chunk.type
        for chunk in datum.model_input.chunks
        if chunk.type not in {"image", "encoded_text"}
    ]
    if other_chunks:
        raise ValueError(f"Unsupported OpenPI FAST chunk types: {other_chunks}")
    if len(text_chunks) != 1:
        raise ValueError("OpenPI FAST SFT expects exactly one encoded_text prefix chunk")

    camera_layout = tuple(model_config.camera_layout)
    if len(image_chunks) != len(camera_layout):
        raise ValueError(
            f"OpenPI FAST SFT expects {len(camera_layout)} image chunks, got {len(image_chunks)}"
        )

    target_tokens = [int(token) for token in _tensor_values(datum.loss_fn_inputs, "target_tokens")]
    weights = _tensor_values(datum.loss_fn_inputs, "weights")
    token_ar_mask = [int(token) for token in _tensor_values(datum.loss_fn_inputs, "token_ar_mask")]
    state = [float(value) for value in _tensor_values(datum.loss_fn_inputs, "state")]

    if len(target_tokens) != len(weights) or len(target_tokens) != len(token_ar_mask):
        raise ValueError(
            "OpenPI FAST SFT requires target_tokens, weights, and token_ar_mask to share one length"
        )

    prefix_tokens = [int(token) for token in text_chunks[0].tokens]
    loss_mask = _binary_loss_mask(weights)

    image_bytes = {
        name: {
            "data": base64.b64encode(chunk.data).decode("utf-8"),
            "format": chunk.format,
        }
        for name, chunk in zip(camera_layout, image_chunks, strict=True)
    }
    image_mask = {name: True for name in camera_layout}

    return {
        "image_bytes": image_bytes,
        "image_mask": image_mask,
        "state": state,
        "prefix_tokens": prefix_tokens,
        "target_tokens": target_tokens,
        "suffix_token_ar_mask": token_ar_mask,
        "suffix_loss_mask": loss_mask,
    }


def _build_openpi_fast_prompt_payload(common: dict[str, Any]) -> dict[str, Any]:
    prefix_tokens = list(common["prefix_tokens"])
    target_tokens = list(common["target_tokens"])
    suffix_token_ar_mask = list(common["suffix_token_ar_mask"])
    suffix_loss_mask = list(common["suffix_loss_mask"])

    return {
        "image_bytes": common["image_bytes"],
        "image_mask": common["image_mask"],
        "state": common["state"],
        "tokenized_prompt": prefix_tokens + target_tokens,
        "tokenized_prompt_mask": [True] * (len(prefix_tokens) + len(target_tokens)),
        "token_ar_mask": ([0] * len(prefix_tokens)) + suffix_token_ar_mask,
        "token_loss_mask": ([False] * len(prefix_tokens)) + suffix_loss_mask,
    }


def build_openpi_fast_sft_runtime_payload(
    *,
    datum: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    return _build_openpi_fast_prompt_payload(
        _build_openpi_fast_common_payload(datum=datum, model_config=model_config)
    )


def build_openpi_fast_rl_runtime_payload(
    *,
    datum: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    common = _build_openpi_fast_common_payload(datum=datum, model_config=model_config)
    target_len = len(common["target_tokens"])
    old_logprobs = [float(value) for value in _tensor_values(datum.loss_fn_inputs, "logprobs")]
    advantages = [float(value) for value in _tensor_values(datum.loss_fn_inputs, "advantages")]

    if len(old_logprobs) != target_len or len(advantages) != target_len:
        raise ValueError(
            "OpenPI FAST RL requires logprobs, advantages, and target_tokens to share one length"
        )

    payload = _build_openpi_fast_prompt_payload(common)
    payload["old_logprobs"] = old_logprobs
    payload["advantages"] = advantages
    return payload


async def _default_runtime_factory(
    *,
    session: Any,
    model_config: ModelConfig,
    config_name: str,
) -> Any:
    from .openpi_fast_runtime import OpenPIFastRuntimeSpec

    return await start_openpi_shared_ray_runtime(
        session=session,
        spec=OpenPIFastRuntimeSpec.from_env(),
        config_name=config_name,
        model_config=model_config,
    )


class OpenPIFastTrainingEngine:
    def __init__(
        self,
        *,
        runtime_factory: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._runtime_clients: dict[str, Any] = {}

    async def initialize(self) -> None:
        ensure_openpi_ray_initialized()
        return None

    def _runtime_for_session(self, session: Any) -> Any:
        try:
            return self._runtime_clients[session.model_id]
        except KeyError as exc:
            raise ValueError(
                f"OpenPI FAST runtime session is not initialized for {session.model_id!r}"
            ) from exc

    def _model_config(self, base_model: str) -> ModelConfig:
        config = get_model_config(base_model)
        if config.training_backend != OPENPI_FAST_TRAINING_BACKEND:
            raise ValueError(f"{base_model!r} is not configured for OpenPI FAST training")
        return config

    def _create_session_payload(self, *, session: Any, model_config: ModelConfig) -> dict[str, Any]:
        return {
            "model_id": session.model_id,
            "session_id": session.session_id,
            "base_model": session.base_model,
            "config_name": get_openpi_fast_config_name(session.base_model),
            "learning_rate": float(session.learning_rate),
            "action_dim": int(model_config.action_dim or 0),
            "action_horizon": int(model_config.action_horizon or 0),
            "max_token_len": int(model_config.max_model_len),
            "camera_layout": list(model_config.camera_layout),
        }

    def _optim_payload(self, adam_params: AdamParams | None) -> dict[str, float]:
        params = adam_params or AdamParams()
        return {
            "learning_rate": float(params.learning_rate),
            "beta1": float(params.beta1),
            "beta2": float(params.beta2),
            "eps": float(params.eps),
        }

    async def _request_runtime(
        self,
        runtime: Any,
        op: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        timeout_s = None
        timeout_getter = getattr(runtime, "timeout_for", None)
        if callable(timeout_getter):
            timeout_s = float(timeout_getter(op))
        return await runtime.request(op, payload, timeout_s=timeout_s)

    async def create_training_session(self, session: Any) -> None:
        model_config = self._model_config(session.base_model)
        config_name = get_openpi_fast_config_name(session.base_model)
        client = await self._runtime_factory(
            session=session,
            model_config=model_config,
            config_name=config_name,
        )
        try:
            await self._request_runtime(
                client,
                "create_session",
                self._create_session_payload(session=session, model_config=model_config),
            )
        except Exception:
            close = getattr(client, "close", None)
            if callable(close):
                await close()
            raise

        self._runtime_clients[session.model_id] = client
        session.backend = OPENPI_FAST_TRAINING_BACKEND
        session.is_active = True
        metadata = getattr(client, "metadata", None)
        if isinstance(metadata, dict):
            logger.info(
                "[%s] OpenPI FAST Ray runtime placed: actor_id=%s node_ip=%s cuda_visible_devices=%s pid=%s worker_module=%s",
                session.model_id,
                metadata.get("actor_id"),
                metadata.get("node_ip"),
                metadata.get("cuda_visible_devices"),
                metadata.get("pid"),
                metadata.get("worker_module"),
            )

    async def forward_backward(self, session: Any, request: Any) -> dict[str, Any]:
        loss_fn = str(request.forward_backward_input.loss_fn)
        if loss_fn == "cross_entropy":
            payload_builder = build_openpi_fast_sft_runtime_payload
        elif loss_fn in {"importance_sampling", "ppo"}:
            payload_builder = build_openpi_fast_rl_runtime_payload
        else:
            raise ValueError(
                "OpenPI FAST ST-03 only supports cross_entropy, importance_sampling, "
                "and ppo forward_backward requests"
            )
        model_config = self._model_config(session.base_model)
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(
            runtime,
            "forward_backward",
            {
                "loss_fn": loss_fn,
                "loss_fn_config": dict(request.forward_backward_input.loss_fn_config or {}),
                "batch": [
                    payload_builder(datum=datum, model_config=model_config)
                    for datum in request.forward_backward_input.data
                ],
            },
        )
        session.accumulated_gradients += 1
        return result

    async def forward(self, session: Any, request: Any) -> dict[str, Any]:
        del request
        raise NotImplementedError(
            f"OpenPI FAST ST-02 does not support forward-only requests for {session.base_model!r}"
        )

    async def get_tokenizer_info(self, session: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"OpenPI FAST tokenizer export is not wired yet for {session.base_model!r}"
        )

    async def optim_step(self, session: Any, request: Any) -> dict[str, Any]:
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(
            runtime,
            "optim_step",
            self._optim_payload(request.adam_params),
        )
        session.current_step += 1
        session.accumulated_gradients = 0
        metrics = dict(result.get("metrics") or {})
        metrics["step"] = session.current_step
        result["metrics"] = metrics
        if "learning_rate" in metrics:
            session.learning_rate = float(metrics["learning_rate"])
        return result

    async def train_step(self, session: Any, request: Any) -> dict[str, Any]:
        fb_result = await self.forward_backward(session, request)
        optim_request = type(
            "_OpenPIOptimRequest",
            (),
            {"adam_params": request.adam_params or AdamParams()},
        )()
        optim_result = await self.optim_step(session, optim_request)
        metrics = dict(fb_result.get("metrics") or {})
        metrics.update(dict(optim_result.get("metrics") or {}))
        fb_result["metrics"] = metrics
        return fb_result

    async def reset_expert_bias(self, session: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"OpenPI FAST does not expose expert-bias reset for {session.base_model!r}"
        )

    async def save_weights_for_sampler(
        self,
        *,
        session: Any,
        checkpoint_name: str,
        checkpoint_base_dir: str,
        use_per_expert_lora: bool = False,
        checkpoint_type: str | None = None,
    ) -> str:
        if use_per_expert_lora:
            raise ValueError("OpenPI FAST does not support per-expert LoRA sampler export")

        runtime = self._runtime_for_session(session)
        checkpoint_root = Path(checkpoint_base_dir).expanduser().resolve() / str(session.model_id)
        export_dir = checkpoint_root / checkpoint_name
        if checkpoint_type:
            export_dir = export_dir / str(checkpoint_type)
        if export_dir.exists():
            raise FileExistsError(f"OpenPI FAST sampler export path already exists: {export_dir}")

        temp_dir = checkpoint_root / f".openpi_fast_sampler_export_{checkpoint_name}_{uuid.uuid4().hex}"
        try:
            result = await self._request_runtime(runtime, "save_sampler_weights", {"save_path": str(temp_dir)})
            source_dir = find_openpi_policy_checkpoint_dir(result["path"])
            params_dir = source_dir / "params"
            assets_dir = source_dir / "assets"
            if not params_dir.is_dir():
                raise FileNotFoundError(f"OpenPI FAST sampler export missing params dir: {params_dir}")
            if not assets_dir.is_dir():
                raise FileNotFoundError(f"OpenPI FAST sampler export missing assets dir: {assets_dir}")

            export_dir.mkdir(parents=True, exist_ok=False)
            shutil.copytree(params_dir, export_dir / "params")
            shutil.copytree(assets_dir, export_dir / "assets")
            return str(export_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def save_weights(
        self,
        session: Any,
        save_path: str,
        use_per_expert_lora: bool = False,
    ) -> str:
        if use_per_expert_lora:
            raise ValueError("OpenPI FAST does not support per-expert LoRA export")
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(runtime, "save_weights", {"save_path": save_path})
        return str(result["path"])

    async def load_weights(
        self,
        session: Any,
        load_path: str,
        load_optimizer: bool = True,
    ) -> None:
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(
            runtime,
            "load_weights",
            {
                "load_path": load_path,
                "load_optimizer": bool(load_optimizer),
                "learning_rate": float(session.learning_rate),
            },
        )
        if "current_step" in result:
            session.current_step = int(result["current_step"])
        if "learning_rate" in result:
            session.learning_rate = float(result["learning_rate"])

    async def shutdown_session(self, session: Any) -> None:
        runtime = self._runtime_clients.pop(session.model_id, None)
        if runtime is not None:
            await self._request_runtime(runtime, "shutdown", {"model_id": session.model_id})
            close = getattr(runtime, "close", None)
            if callable(close):
                await close()
        session.is_active = False
