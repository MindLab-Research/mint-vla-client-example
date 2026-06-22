from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .types import AdamParams, ModelInput, TensorData


class MintBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class InterpolateCheckpointsRequest(MintBaseModel):
    source_paths: list[str]
    coefficients: list[float]
    owner_id: str | None = None  # admin-only owner scope applied to all source checkpoint references
    output_path: str | None = None
    retry: bool = False
    output_checkpoint_type: Literal["sampler"] = "sampler"
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"


class InterpolateCheckpointsResponse(MintBaseModel):
    path: str
    checkpoint_type: Literal["sampler"] = "sampler"
    source_paths: list[str]
    coefficients: list[float]
    has_rank_shards: bool = False
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"


class ReverseKLDatum(MintBaseModel):
    student_input: ModelInput
    reference_input: ModelInput
    target_tokens: TensorData
    weights: TensorData


class ForwardBackwardReverseKLRequest(MintBaseModel):
    model_id: str
    reference_model_path: str
    owner_id: str | None = None  # admin-only owner scope for checkpoint references
    data: list[ReverseKLDatum]
    temperature: float = 1.0
    seq_id: int | None = None
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


class ReverseKLItemOutput(MintBaseModel):
    loss: TensorData


class ForwardBackwardReverseKLResponse(MintBaseModel):
    outputs: list[ReverseKLItemOutput]
    metrics: dict[str, float]
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


class VLAObservation(MintBaseModel):
    model_input: ModelInput
    state: TensorData


class VLADatum(MintBaseModel):
    observation: VLAObservation
    supervision: dict[str, TensorData]


class VLATrainStepRequest(MintBaseModel):
    model_id: str
    data: list[VLADatum]
    loss_fn: str
    loss_fn_config: dict[str, Any] | None = None
    adam_params: AdamParams | None = None
    seq_id: int | None = None
    type: Literal["mint_vla_train_step"] = "mint_vla_train_step"


class MintCreateActionSessionRequest(MintBaseModel):
    session_id: str
    action_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None
    owner_id: str | None = None  # admin-only owner scope for checkpoint references


class MintCreateActionSessionResponse(MintBaseModel):
    action_session_id: str


class VLAActRequest(MintBaseModel):
    seq_id: int | None = None
    observation: VLAObservation
    temperature: float | None = None
    return_rollout_trace: bool | None = None
    rollout_trace_config: dict[str, Any] | None = None


class MintDeleteActionSessionResponse(MintBaseModel):
    action_session_id: str
    status: Literal["deleted"] = "deleted"
