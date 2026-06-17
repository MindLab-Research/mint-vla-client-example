from __future__ import annotations

import base64
import structlog
import shutil
import uuid
from pathlib import Path
from typing import Any

from mint_server.models.types import AdamParams
from mint_server.backend.core.model_registry import ModelConfig, get_model_config
from mint_server.backend.openpi.openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from mint_server.backend.openpi.openpi_ray_runtime import ensure_openpi_ray_initialized
from mint_server.backend.openpi.openpi_shared_ray_runtime import start_openpi_shared_ray_runtime


logger = structlog.get_logger(__name__)

OPENPI_PI05_WORKER_MODULE = f"{__package__}.openpi_pi05_worker"
OPENPI_PI05_ACTION_WORKER_MODULE = f"{__package__}.openpi_pi05_action_worker"
OPENPI_PI05_TRAINING_BACKEND = "openpi_pi05"
OPENPI_PI05_LORA_RANK = 16
OPENPI_PI05_CONFIG_NAMES = {
    "openpi/pi05-libero-low-mem-finetune": "pi05_libero",
}


def _is_openpi_pi05_model(base_model: str) -> bool:
    try:
        return get_model_config(base_model).training_backend == OPENPI_PI05_TRAINING_BACKEND
    except Exception:
        return False


def get_openpi_pi05_config_name(base_model: str) -> str:
    try:
        return OPENPI_PI05_CONFIG_NAMES[base_model]
    except KeyError as exc:
        raise ValueError(f"No OpenPI pi0.5 config mapping registered for {base_model!r}") from exc


def validate_openpi_pi05_create_request(request: Any) -> None:
    base_model = str(getattr(request, "base_model", "") or "")
    if not _is_openpi_pi05_model(base_model):
        return

    lora_config = getattr(request, "lora_config", None)
    if lora_config is None:
        raise ValueError("OpenPI pi0.5 training requires lora_config")
    if int(lora_config.rank) != OPENPI_PI05_LORA_RANK:
        raise ValueError(
            "OpenPI pi0.5 training only supports the upstream LoRA rank "
            f"{OPENPI_PI05_LORA_RANK}"
        )

    for field in ("train_attn", "train_mlp", "train_unembed"):
        if getattr(lora_config, field, True) is not True:
            raise ValueError(
                "OpenPI pi0.5 training does not support partial LoRA toggle mapping; "
                f"expected {field}=True"
            )

    if getattr(request, "rollout_correction_config", None) is not None:
        raise ValueError("OpenPI pi0.5 does not support rollout_correction_config")


def _tensor_payload(value: Any, key: str) -> tuple[list[Any], list[int]]:
    if value is None:
        raise ValueError(f"Missing loss_fn_inputs[{key!r}]")

    data = getattr(value, "data", None)
    shape = getattr(value, "shape", None)
    if data is None and isinstance(value, dict):
        data = value.get("data")
        shape = value.get("shape")

    if data is None or shape is None:
        raise ValueError(f"loss_fn_inputs[{key!r}] must provide tensor-style data and shape")
    if not isinstance(shape, list):
        raise ValueError(f"loss_fn_inputs[{key!r}] shape must be a list")
    if isinstance(data, list):
        return data, shape
    return [data], shape


def _pad(values: list[float], target_dim: int, *, key: str) -> list[float]:
    if len(values) > target_dim:
        raise ValueError(f"{key} length {len(values)} exceeds action_dim {target_dim}")
    return values + ([0.0] * (target_dim - len(values)))


def _common_input_payload(
    *,
    model_input: Any,
    state_input: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    if model_config.training_backend != OPENPI_PI05_TRAINING_BACKEND:
        raise ValueError("OpenPI pi0.5 helpers only support OpenPI pi0.5 model configs")

    image_chunks = [chunk for chunk in model_input.chunks if chunk.type == "image"]
    text_chunks = [chunk for chunk in model_input.chunks if chunk.type == "encoded_text"]
    other_chunks = [
        chunk.type
        for chunk in model_input.chunks
        if chunk.type not in {"image", "encoded_text"}
    ]
    if other_chunks:
        raise ValueError(f"Unsupported OpenPI pi0.5 chunk types: {other_chunks}")
    if len(text_chunks) != 1:
        raise ValueError("OpenPI pi0.5 expects exactly one encoded_text prompt chunk")

    camera_layout = tuple(model_config.camera_layout)
    if len(image_chunks) != len(camera_layout):
        raise ValueError(
            f"OpenPI pi0.5 expects {len(camera_layout)} image chunks, got {len(image_chunks)}"
        )

    action_dim = int(model_config.action_dim or 0)
    if action_dim <= 0:
        raise ValueError("OpenPI pi0.5 model config must define a positive action_dim")

    state_data, state_shape = _tensor_payload(state_input, "state")
    if len(state_shape) != 1:
        raise ValueError("OpenPI pi0.5 state must be rank-1")
    state = _pad([float(value) for value in state_data], action_dim, key="state")

    image_bytes = {
        name: {
            "data": base64.b64encode(chunk.data).decode("utf-8"),
            "format": chunk.format,
        }
        for name, chunk in zip(camera_layout, image_chunks, strict=True)
    }
    prompt_tokens = [int(token) for token in text_chunks[0].tokens]
    return {
        "image_bytes": image_bytes,
        "image_mask": {name: True for name in camera_layout},
        "state": state,
        "tokenized_prompt": prompt_tokens,
        "tokenized_prompt_mask": [True] * len(prompt_tokens),
    }


def build_openpi_pi05_sft_runtime_payload(
    *,
    datum: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    payload = _common_input_payload(
        model_input=datum.model_input,
        state_input=datum.loss_fn_inputs.get("state"),
        model_config=model_config,
    )

    action_dim = int(model_config.action_dim or 0)
    action_horizon = int(model_config.action_horizon or 0)
    if action_horizon <= 0:
        raise ValueError("OpenPI pi0.5 model config must define positive action_dim and action_horizon")

    actions_data, actions_shape = _tensor_payload(datum.loss_fn_inputs.get("actions"), "actions")
    if len(actions_shape) != 2:
        raise ValueError("OpenPI pi0.5 actions must be rank-2")
    if int(actions_shape[0]) != action_horizon:
        raise ValueError(
            f"OpenPI pi0.5 actions action_horizon mismatch: expected {action_horizon}, got {actions_shape[0]}"
        )
    source_action_dim = int(actions_shape[1])
    if source_action_dim <= 0:
        raise ValueError("OpenPI pi0.5 actions must have a positive trailing dimension")
    if len(actions_data) != action_horizon * source_action_dim:
        raise ValueError("OpenPI pi0.5 actions data length does not match shape")

    actions: list[list[float]] = []
    for row_idx in range(action_horizon):
        start = row_idx * source_action_dim
        row = [float(value) for value in actions_data[start : start + source_action_dim]]
        actions.append(_pad(row, action_dim, key="actions"))

    return {**payload, "actions": actions}


def build_openpi_pi05_action_observation_payload(
    *,
    observation: Any,
    extra_inputs: dict[str, Any],
    model_config: ModelConfig,
) -> dict[str, Any]:
    return _common_input_payload(
        model_input=observation,
        state_input=extra_inputs.get("state"),
        model_config=model_config,
    )


async def _default_runtime_factory(*, session: Any, model_config: ModelConfig, config_name: str) -> Any:
    import dataclasses

    from mint_server.backend.openpi.openpi_fast_runtime import OpenPIFastRuntimeSpec

    spec = dataclasses.replace(
        OpenPIFastRuntimeSpec.from_env(),
        worker_module=OPENPI_PI05_WORKER_MODULE,
    )
    return await start_openpi_shared_ray_runtime(
        session=session,
        spec=spec,
        config_name=config_name,
        model_config=model_config,
    )


class OpenPIPi05TrainingEngine:
    def __init__(self, *, runtime_factory: Any | None = None) -> None:
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
                f"OpenPI pi0.5 runtime session is not initialized for {session.model_id!r}"
            ) from exc

    def _model_config(self, base_model: str) -> ModelConfig:
        config = get_model_config(base_model)
        if config.training_backend != OPENPI_PI05_TRAINING_BACKEND:
            raise ValueError(f"{base_model!r} is not configured for OpenPI pi0.5 training")
        return config

    def _create_session_payload(self, *, session: Any, model_config: ModelConfig) -> dict[str, Any]:
        return {
            "model_id": session.model_id,
            "session_id": session.session_id,
            "base_model": session.base_model,
            "config_name": get_openpi_pi05_config_name(session.base_model),
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

    async def _request_runtime(self, runtime: Any, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_s = None
        timeout_getter = getattr(runtime, "timeout_for", None)
        if callable(timeout_getter):
            timeout_s = float(timeout_getter(op))
        return await runtime.request(op, payload, timeout_s=timeout_s)

    async def create_training_session(self, session: Any) -> None:
        model_config = self._model_config(session.base_model)
        config_name = get_openpi_pi05_config_name(session.base_model)
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
        session.backend = OPENPI_PI05_TRAINING_BACKEND
        session.is_active = True
        metadata = getattr(client, "metadata", None)
        if isinstance(metadata, dict):
            logger.info(
                "[%s] OpenPI pi0.5 Ray runtime placed: actor_id=%s node_ip=%s cuda_visible_devices=%s pid=%s worker_module=%s",
                session.model_id,
                metadata.get("actor_id"),
                metadata.get("node_ip"),
                metadata.get("cuda_visible_devices"),
                metadata.get("pid"),
                metadata.get("worker_module"),
            )

    async def forward_backward(self, session: Any, request: Any) -> dict[str, Any]:
        loss_fn = str(request.forward_backward_input.loss_fn)
        if loss_fn != "flow_matching":
            raise ValueError("OpenPI pi0.5 only supports flow_matching forward_backward requests")

        model_config = self._model_config(session.base_model)
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(
            runtime,
            "forward_backward",
            {
                "loss_fn": loss_fn,
                "loss_fn_config": dict(request.forward_backward_input.loss_fn_config or {}),
                "batch": [
                    build_openpi_pi05_sft_runtime_payload(datum=datum, model_config=model_config)
                    for datum in request.forward_backward_input.data
                ],
            },
        )
        session.accumulated_gradients += 1
        return result

    async def forward(self, session: Any, request: Any) -> dict[str, Any]:
        del request
        raise NotImplementedError(
            f"OpenPI pi0.5 does not support forward-only requests for {session.base_model!r}"
        )

    async def get_tokenizer_info(self, session: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"OpenPI pi0.5 tokenizer export is not wired yet for {session.base_model!r}"
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
            "_OpenPIPi05OptimRequest",
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
            f"OpenPI pi0.5 does not expose expert-bias reset for {session.base_model!r}"
        )

    async def save_weights_for_sampler(
        self,
        *,
        session: Any,
        checkpoint_name: str,
        checkpoint_base_dir: str,
        checkpoint_type: str | None = None,
    ) -> str:
        runtime = self._runtime_for_session(session)
        checkpoint_root = Path(checkpoint_base_dir).expanduser().resolve() / str(session.model_id)
        export_dir = checkpoint_root / checkpoint_name
        if checkpoint_type:
            export_dir = export_dir / str(checkpoint_type)
        if export_dir.exists():
            raise FileExistsError(f"OpenPI pi0.5 sampler export path already exists: {export_dir}")

        temp_dir = checkpoint_root / f".openpi_pi05_sampler_export_{checkpoint_name}_{uuid.uuid4().hex}"
        try:
            result = await self._request_runtime(runtime, "save_sampler_weights", {"save_path": str(temp_dir)})
            source_dir = find_openpi_policy_checkpoint_dir(result["path"])
            params_dir = source_dir / "params"
            assets_dir = source_dir / "assets"
            if not params_dir.is_dir():
                raise FileNotFoundError(f"OpenPI pi0.5 sampler export missing params dir: {params_dir}")
            if not assets_dir.is_dir():
                raise FileNotFoundError(f"OpenPI pi0.5 sampler export missing assets dir: {assets_dir}")

            export_dir.mkdir(parents=True, exist_ok=False)
            shutil.copytree(params_dir, export_dir / "params")
            shutil.copytree(assets_dir, export_dir / "assets")
            return str(export_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def save_weights(self, session: Any, save_path: str) -> str:
        runtime = self._runtime_for_session(session)
        result = await self._request_runtime(runtime, "save_weights", {"save_path": save_path})
        return str(result["path"])

    async def load_weights(self, session: Any, load_path: str, load_optimizer: bool = True) -> None:
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
