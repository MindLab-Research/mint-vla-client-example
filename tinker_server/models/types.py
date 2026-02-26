"""Pydantic models for tinker-server API.

These types match the tinker client API for compatibility.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class EncodedTextChunk(BaseModel):
    """A chunk of encoded text tokens."""

    tokens: list[int]
    type: Literal["encoded_text"] = "encoded_text"


class ModelInput(BaseModel):
    """Input to the model as a list of chunks."""

    chunks: list[EncodedTextChunk]

    def to_token_ids(self) -> list[int]:
        """Flatten all chunks into a single list of token IDs."""
        tokens = []
        for chunk in self.chunks:
            if chunk.type == "encoded_text":
                tokens.extend(chunk.tokens)
        return tokens

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        """Create ModelInput from a list of token IDs."""
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])


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

    Accepts either sampling_session_id or model_id for Tinker SDK compatibility.
    """

    sampling_session_id: str | None = None
    model_id: str | None = None  # Alias for sampling_session_id (Tinker SDK compat)
    seq_id: int = 0  # Default to 0 if not provided
    num_samples: int
    prompt: ModelInput
    sampling_params: SamplingParams
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0
    include_prompt_logprobs: bool = False  # Alias for prompt_logprobs (Tinker SDK compat)

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


class CreateModelRequest(BaseModel):
    """Request to create a new training model."""

    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoRAConfig | None = None
    type: Literal["create_model"] = "create_model"


class CreateModelResponse(BaseModel):
    """Response from model creation."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str
    type: Literal["create_model"] = "create_model"
    backend: str | None = None  # "megatron" for MoE, "peft" for dense


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


class LossFnOutput(BaseModel):
    """Output from a loss function."""

    model_config = ConfigDict(extra="allow")

    loss: TensorData | None = None


class ForwardBackwardInput(BaseModel):
    """Input for forward/backward pass."""

    data: list[Datum]
    loss_fn: str
    loss_fn_config: dict[str, float] | None = None


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
# Weight Sync Types
# =============================================================================


class SaveWeightsForSamplerRequest(BaseModel):
    """Request to save model weights for sampler use."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str | None = None  # checkpoint name for named save (None for ephemeral)
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

    checkpoint_id: str  # directory name, e.g. "checkpoint-100"
    path: str  # tinker://{model_id}/{checkpoint_id} or mint://{model_id}/{checkpoint_id}
    step: int | None = None  # parsed from checkpoint name if available
    created_at: str  # ISO timestamp


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
