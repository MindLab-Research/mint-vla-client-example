"""Pydantic models for tinker-server API.

These types match the tinker client API for compatibility.
"""

import base64
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


class EncodedTextChunk(BaseModel):
    """A chunk of encoded text tokens."""

    tokens: list[int]
    type: Literal["encoded_text"] = "encoded_text"

    @property
    def length(self) -> int:
        return len(self.tokens)


class ImageAssetPointerChunk(BaseModel):
    """A pointer to an image asset plus its expected token footprint."""

    format: Literal["png", "jpeg"]
    location: str
    expected_tokens: int | None = None
    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageAssetPointerChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens


class ImageChunk(BaseModel):
    """Inline image bytes plus their expected token footprint."""

    data: bytes
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = None
    type: Literal["image"] = "image"

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: bytes | str) -> bytes:
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("data")
    def serialize_data(self, value: bytes) -> str:
        return base64.b64encode(value).decode("utf-8")

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens


ModelInputChunk = Annotated[
    EncodedTextChunk | ImageAssetPointerChunk | ImageChunk,
    Field(discriminator="type"),
]


class ModelInput(BaseModel):
    """Input to the model as a list of chunks."""

    chunks: list[ModelInputChunk]

    @property
    def length(self) -> int:
        return sum(chunk.length for chunk in self.chunks)

    def to_ints(self) -> list[int]:
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(
                "to_ints only supported for ModelInput with EncodedTextChunks, "
                f"got {[type(chunk).__name__ for chunk in self.chunks]}"
            )
        return [token for chunk in self.chunks for token in chunk.tokens]

    def to_token_ids(self) -> list[int]:
        """Backward-compatible alias for strict text-only flattening."""
        return self.to_ints()

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        """Create ModelInput from a list of token IDs."""
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])


class SamplingParams(BaseModel):
    """Parameters for text generation."""

    max_tokens: int
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    stop: list[str] | list[int] | str | None = None
    seed: int | None = None


class SampledSequence(BaseModel):
    """A single generated sequence."""

    tokens: list[int]
    logprobs: list[float] | None
    routed_experts: list | None = None
    stop_reason: Literal["length", "stop", "eos"]


class SampleRequest(BaseModel):
    """Request to generate samples from the model.

    Accepts exactly one selector family:
    - `sampling_session_id` or `model_id`
    - `base_model`
    - `model_path`
    """

    sampling_session_id: str | None = None
    model_id: str | None = None  # Alias for sampling_session_id (Tinker SDK compat)
    base_model: str | None = None
    model_path: str | None = None
    seq_id: int | None = None
    num_samples: int
    prompt: ModelInput
    sampling_params: SamplingParams
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0
    include_prompt_logprobs: bool = False  # Alias for prompt_logprobs (Tinker SDK compat)

    @model_validator(mode="after")
    def validate_selectors(self) -> "SampleRequest":
        session_selector = self.sampling_session_id or self.model_id
        if self.sampling_session_id and self.model_id and self.sampling_session_id != self.model_id:
            raise ValueError("sampling_session_id and model_id must match when both are provided")
        selector_count = sum(
            [
                bool(session_selector),
                bool(self.base_model),
                bool(self.model_path),
            ]
        )
        if selector_count != 1:
            raise ValueError(
                "Exactly one selector must be provided: sampling_session_id/model_id, base_model, or model_path"
            )
        if self.seq_id is not None and not session_selector:
            raise ValueError("seq_id requires sampling_session_id or model_id")
        return self

    def has_session_selector(self) -> bool:
        return bool(self.sampling_session_id or self.model_id)

    def needs_session_creation(self) -> bool:
        return not self.has_session_selector()

    def get_session_id(self) -> str:
        """Get the session ID, preferring sampling_session_id over model_id."""
        if self.sampling_session_id:
            return self.sampling_session_id
        if self.model_id:
            return self.model_id
        raise ValueError("Either sampling_session_id or model_id must be provided")


class SampleResponse(BaseModel):
    """Response containing generated samples."""

    sequences: list[SampledSequence]
    # Tinker SDK: first entry is None (first token has no conditioning context).
    prompt_logprobs: list[float | None] | None = None
    # Tinker SDK: first entry is None (no prior context for token 0).
    # Each subsequent entry is a list of (token_id, logprob) pairs.
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None
    type: Literal["sample"] = "sample"


class ComputeLogprobsRequest(BaseModel):
    """Request to compute logprobs for a sequence.

    Returns a list of length len(sequence), where:
    - logprobs[0] is None (first token has no conditioning context)
    - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1
    """

    sampling_session_id: str
    seq_id: int
    sequence: ModelInput


class ComputeLogprobsResponse(BaseModel):
    """Response containing computed logprobs."""

    logprobs: list[float | None]
    type: Literal["compute_logprobs"] = "compute_logprobs"


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    tags: list[str] = []
    user_metadata: dict = {}
    sdk_version: str = ""
    type: Literal["create_session"] = "create_session"


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None
    type: Literal["create_session"] = "create_session"


class CreateSamplingSessionRequest(BaseModel):
    """Request to create a sampling session."""

    session_id: str
    sampling_session_seq_id: int | None = None  # Sequence within session
    base_model: str | None = None
    model_path: str | None = None  # tinker://, mint://, or file:// path to weights
    lora_rank: int = 32  # LoRA rank for per-session adapter


class CreateSamplingSessionResponse(BaseModel):
    """Response from sampling session creation."""

    sampling_session_id: str


class CreateActionSessionRequest(BaseModel):
    """Request to create an action inference session."""

    session_id: str
    action_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None


class CreateActionSessionResponse(BaseModel):
    """Response from action session creation."""

    action_session_id: str


class UntypedAPIFuture(BaseModel):
    """An async operation handle that can be polled for results."""

    request_id: str


class FutureRetrieveRequest(BaseModel):
    """Request to retrieve the result of an async operation."""

    request_id: str
    model_id: str | None = None


# =============================================================================
# Training Types
# =============================================================================


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True


class RolloutCorrectionConfig(BaseModel):
    """Session-level rollout correction policy for RL losses.

    This mirrors verl's canonical `policy_loss.rollout_correction` schema.
    """

    rollout_is: Literal["token", "sequence"] | None = None
    rollout_is_threshold: float | None = None
    rollout_is_batch_normalize: bool | None = None
    rollout_rs: str | None = None
    rollout_rs_threshold: str | float | None = None
    bypass_mode: bool | None = None
    loss_type: Literal["ppo_clip", "reinforce"] | None = None


class CreateModelRequest(BaseModel):
    """Request to create a new training model."""

    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoRAConfig | None = None
    rollout_correction_config: RolloutCorrectionConfig | None = None
    type: Literal["create_model"] = "create_model"


class CreateModelResponse(BaseModel):
    """Response from model creation."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str
    type: Literal["create_model"] = "create_model"
    backend: str | None = None  # "megatron" for MoE, "peft" for dense


class Cursor(BaseModel):
    """Pagination cursor information."""

    offset: int
    limit: int
    total_count: int


class TrainingRun(BaseModel):
    """Training run metadata."""

    training_run_id: str
    base_model: str
    model_owner: str | None = None
    is_lora: bool
    corrupted: bool
    lora_rank: int | None = None
    last_request_time: str | None = None
    last_checkpoint: Any | None = None
    last_sampler_checkpoint: Any | None = None
    user_metadata: dict[str, Any] | None = None


class TrainingRunsResponse(BaseModel):
    """List of training runs with pagination info."""

    training_runs: list[TrainingRun]
    cursor: Cursor | None = None


class GetSessionResponse(BaseModel):
    """Session metadata response."""

    training_run_ids: list[str]
    sampler_ids: list[str]


class ListSessionsResponse(BaseModel):
    """List of session IDs."""

    sessions: list[str]


class GetSamplerResponse(BaseModel):
    """Sampler metadata response."""

    sampler_id: str
    base_model: str
    model_path: str | None = None


class Datum(BaseModel):
    """A single data item for training."""

    model_config = ConfigDict(extra="allow")

    model_input: ModelInput
    loss_fn_inputs: dict[str, Any]  # Dict mapping field names to TensorData


class TensorData(BaseModel):
    """Tensor data for tinker compatibility."""

    data: list[float] | float
    shape: list[int]
    dtype: str = "float32"


class ActRequest(BaseModel):
    """Request to run action inference."""

    action_session_id: str
    seq_id: int | None = None
    observation: ModelInput
    extra_inputs: dict[str, TensorData] = {}
    temperature: float | None = None


class ActResponse(BaseModel):
    """Response containing action outputs."""

    actions: TensorData
    policy_timing: dict[str, float] | None = None
    type: Literal["act"] = "act"


class LossFnOutput(BaseModel):
    """Output from a loss function."""

    model_config = ConfigDict(extra="allow")

    loss: TensorData | None = None


class ForwardBackwardInput(BaseModel):
    """Input for forward/backward pass."""

    data: list[Datum]
    loss_fn: str
    loss_fn_config: dict[str, Any] | None = None


class ForwardBackwardOutput(BaseModel):
    """Output from forward/backward pass."""

    loss_fn_output_type: str
    loss_fn_outputs: list[LossFnOutput]
    metrics: dict[str, float]


class ForwardBackwardRequest(BaseModel):
    """Request to perform a forward-backward pass."""

    model_config = ConfigDict(protected_namespaces=())

    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardRequest(BaseModel):
    """Request to perform a forward-only pass (no backward).

    Uses forward_input field (not forward_backward_input) to match tinker client API.
    """

    model_config = ConfigDict(protected_namespaces=())

    forward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardBackwardResponse(BaseModel):
    """Response from a forward-backward pass."""

    output: ForwardBackwardOutput
    type: Literal["forward_backward"] = "forward_backward"


class AdamParams(BaseModel):
    """Parameters for Adam optimizer."""

    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12


class OptimStepRequest(BaseModel):
    """Request to perform an optimizer step."""

    model_config = ConfigDict(protected_namespaces=())

    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None
    type: Literal["optim_step"] = "optim_step"


class OptimStepResponse(BaseModel):
    """Response from an optimizer step."""

    metrics: dict[str, float] | None = None
    type: Literal["optim_step"] = "optim_step"


class TrainStepRequest(BaseModel):
    """Request to perform a combined train step (forward_backward + optim_step)."""

    model_config = ConfigDict(protected_namespaces=())

    forward_backward_input: ForwardBackwardInput
    adam_params: AdamParams | None = None
    model_id: str
    seq_id: int | None = None
    type: Literal["train_step"] = "train_step"


class ResetExpertBiasRequest(BaseModel):
    """Request to reset expert_bias buffers in MoE router modules.

    This is needed to ensure consistent behavior between Megatron (training)
    and vLLM (inference), as expert_bias accumulates during training but
    is not exported with LoRA weights.
    """

    model_id: str


class ResetExpertBiasResponse(BaseModel):
    """Response from reset_expert_bias."""

    model_id: str
    modules_reset: int = 0
    status: Literal["success", "not_applicable"] = "success"


class TelemetryRequest(BaseModel):
    """Telemetry data from tinker client (discarded)."""

    events: list[dict[str, Any]] = []
    platform: str = ""
    sdk_version: str = ""
    session_id: str = ""


class TelemetryResponse(BaseModel):
    """Response for telemetry submission."""

    status: Literal["accepted"] = "accepted"


class SessionHeartbeatRequest(BaseModel):
    """Heartbeat request to keep session alive."""

    session_id: str
    type: Literal["session_heartbeat"] = "session_heartbeat"


class SessionHeartbeatResponse(BaseModel):
    """Heartbeat response."""

    type: Literal["session_heartbeat"] = "session_heartbeat"


# =============================================================================
# OpenAI-Compatible API Types
# =============================================================================


class OAICompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1


class OAIFunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class OAIToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: OAIFunctionDefinition


class OAIFunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class OAIToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    type: Literal["function"] = "function"
    function: OAIFunctionCall


class OAIMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[OAIToolCall] | None = None

    @model_validator(mode="after")
    def validate_role_specific_fields(self) -> "OAIMessage":
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
            return self

        if self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")

        if self.role == "assistant":
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant messages require content or tool_calls")
            if self.tool_calls:
                missing_ids = [i for i, tc in enumerate(self.tool_calls) if tc.id is None]
                if missing_ids:
                    raise ValueError(f"tool_calls[{missing_ids[0]}].id is required")
            return self

        if self.tool_calls is not None:
            raise ValueError("tool_calls are only valid for assistant messages")
        if self.content is None:
            raise ValueError(f"{self.role} messages require content")
        return self


class OAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[OAIMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1
    tools: list[OAIToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def validate_tool_calling_fields(self) -> "OAIChatCompletionRequest":
        tool_names = [tool.function.name for tool in self.tools or []]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tools must have unique function names")

        if isinstance(self.tool_choice, str):
            if self.tool_choice not in {"none", "auto", "required"}:
                raise ValueError("tool_choice must be one of: none, auto, required")
            if self.tool_choice == "required" and not self.tools:
                raise ValueError("tool_choice='required' requires tools")
            return self

        if self.tool_choice is None:
            return self

        if not isinstance(self.tool_choice, dict):
            raise ValueError("tool_choice must be a string or a function selection object")

        if self.tool_choice.get("type") != "function":
            raise ValueError("tool_choice object must have type='function'")

        function = self.tool_choice.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool_choice.function must be an object")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.function.name must be a non-empty string")

        if not self.tools:
            raise ValueError("tool_choice function selection requires tools")
        if name not in set(tool_names):
            raise ValueError(f"tool_choice function {name!r} must be declared in tools")
        return self


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OAICompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Literal["stop", "length"]
    logprobs: None = None


class OAIChatMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[OAIToolCall] | None = None


class OAIChatCompletionChoice(BaseModel):
    index: int
    message: OAIChatMessageResponse
    finish_reason: Literal["stop", "length", "tool_calls"]
    logprobs: None = None


class OAICompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[OAICompletionChoice]
    usage: OAIUsage


class OAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OAIChatCompletionChoice]
    usage: OAIUsage


# =============================================================================
# Weight Sync Types
# =============================================================================


class SaveWeightsForSamplerRequest(BaseModel):
    """Request to save model weights for sampler use."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str | None = None  # checkpoint name for named save (None for ephemeral)
    ttl_seconds: int | None = None
    seq_id: int | None = None
    sampling_session_seq_id: int | None = None  # For ephemeral flow
    use_per_expert_lora: bool = False  # If True, expand shared MLP LoRA to per-expert format for MoE
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class SaveWeightsForSamplerResponse(BaseModel):
    """Response from save weights for sampler."""

    path: str | None = None  # tinker://, mint://, or file:// URI (None for ephemeral)
    sampling_session_id: str | None = None  # For ephemeral flow
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


# =============================================================================
# Checkpoint Types (save_weights, load_weights, list, delete)
# =============================================================================


class SaveStateRequest(BaseModel):
    """Request to save model state to checkpoint (save_weights endpoint)."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str | None = None  # checkpoint name, e.g. "checkpoint-100"
    ttl_seconds: int | None = None
    seq_id: int | None = None
    type: Literal["save_weights"] = "save_weights"


class SaveStateResponse(BaseModel):
    """Response from saving state."""

    path: str  # tinker:// or mint:// URI
    type: Literal["save_weights"] = "save_weights"


class LoadStateRequest(BaseModel):
    """Request to load model state from checkpoint (load_weights endpoint)."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str  # tinker://, mint://, or file:// path
    optimizer: bool = True  # whether to restore optimizer state
    seq_id: int | None = None
    type: Literal["load_weights"] = "load_weights"


class LoadStateResponse(BaseModel):
    """Response from loading state."""

    path: str
    type: Literal["load_weights"] = "load_weights"


class CheckpointInfo(BaseModel):
    """Information about a checkpoint."""

    checkpoint_id: str
    checkpoint_type: Literal["training", "sampler"]
    time: datetime
    tinker_path: str

    # Compatibility fields (ignored by Tinker clients; used by Mint tooling).
    path: str | None = None
    step: int | None = None  # parsed from checkpoint name if available
    created_at: str | None = None  # ISO timestamp
    size_bytes: int | None = None
    public: bool = False
    expires_at: datetime | None = None
    storage_tier: str | None = None
    mirror_status: str | None = None
    mirror_error: str | None = None


class CheckpointsListResponse(BaseModel):
    """Response listing checkpoints for a model."""

    model_id: str | None = None  # None for list-all endpoint
    checkpoints: list[CheckpointInfo]


class CheckpointUploadResponse(BaseModel):
    """Response from uploading a checkpoint archive."""

    checkpoint_id: str
    path: str


class CreateModelFromStateRequest(BaseModel):
    """Request to create a training model from existing checkpoint.

    Composes create_model + load_state into single operation.
    """

    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_seq_id: int
    base_model: str
    state_path: str  # mint:// or file:// path to checkpoint
    lora_config: LoRAConfig | None = None
    rollout_correction_config: RolloutCorrectionConfig | None = None
    load_optimizer: bool = True  # whether to restore optimizer state
    user_metadata: dict[str, Any] | None = None
    type: Literal["create_model_from_state"] = "create_model_from_state"


class CreateModelFromStateResponse(BaseModel):
    """Response from create model from state."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str
    type: Literal["create_model_from_state"] = "create_model_from_state"


# =============================================================================
# Model Info Types (get_info endpoint)
# =============================================================================


class GetInfoRequest(BaseModel):
    """Request to get model info."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    type: Literal["get_info"] = "get_info"


class ModelData(BaseModel):
    """Model architecture and tokenizer data."""

    model_config = ConfigDict(protected_namespaces=())

    arch: str | None = None
    model_name: str | None = None
    tokenizer_id: str | None = None


class GetInfoResponse(BaseModel):
    """Response from get_info endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_data: ModelData
    model_name: str | None = None
    is_lora: bool | None = None
    lora_rank: int | None = None
    type: Literal["get_info"] = "get_info"
