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
    stop: list[str] = []
    seed: int | None = None


class SampledSequence(BaseModel):
    """A single generated sequence."""

    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: Literal["length", "stop", "eos"]


class SampleRequest(BaseModel):
    """Request to generate samples from the model."""

    sampling_session_id: str
    seq_id: int
    num_samples: int
    prompt: ModelInput
    sampling_params: SamplingParams
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0


class SampleResponse(BaseModel):
    """Response containing generated samples."""

    sequences: list[SampledSequence]
    prompt_logprobs: list[float] | None = None
    type: Literal["sample"] = "sample"


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
    base_model: str | None = None
    model_path: str | None = None
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


class Datum(BaseModel):
    """A single data item for training."""

    model_config = ConfigDict(extra="allow")


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


class TelemetryRequest(BaseModel):
    """Telemetry data from tinker client."""

    events: list[dict[str, Any]] = []


class TelemetryResponse(BaseModel):
    """Response for telemetry submission."""

    status: Literal["accepted"] = "accepted"
