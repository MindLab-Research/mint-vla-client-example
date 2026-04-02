# MintX API Reference

Keep this file synchronized with the implementation in `tinker_server/routes/mint.py`, `tinker_server/models/mint_types.py`, and `mint.mint` client helpers.

## Server Namespace

All Mint-only server APIs live under `/api/v1/mint`.

### POST `/api/v1/mint/checkpoints/interpolate`

Create a new immutable sampler checkpoint by linearly interpolating existing checkpoints.

Request:

```python
class InterpolateCheckpointsRequest(BaseModel):
    source_paths: list[str]
    coefficients: list[float]
    output_path: str | None = None
    output_checkpoint_type: Literal["sampler"] = "sampler"
```

Response payload after `retrieve_future`:

```python
class InterpolateCheckpointsResponse(BaseModel):
    path: str
    checkpoint_type: Literal["sampler"]
    source_paths: list[str]
    coefficients: list[float]
    has_rank_shards: bool
```

Semantics:

- source checkpoints must share the same `model_name`, `backend`, `owner_id`, and `adapter_config.json`
- the output checkpoint reuses the first source checkpoint's `model_id`
- the operation is immutable and produces a new checkpoint id
- only sampler output is currently supported
- if source checkpoints contain Megatron rank adapter shards, those shards are interpolated too

### POST `/api/v1/mint/forward_backward_reverse_kl`

Run student forward/backward against a fixed reference checkpoint using reverse KL over the full vocabulary.

Request:

```python
class ReverseKLDatum(BaseModel):
    student_input: ModelInput
    reference_input: ModelInput
    target_tokens: TensorData
    weights: TensorData

class ForwardBackwardReverseKLRequest(BaseModel):
    model_id: str
    reference_model_path: str
    data: list[ReverseKLDatum]
    temperature: float = 1.0
    seq_id: int | None = None
```

Response payload after `retrieve_future`:

```python
class ReverseKLItemOutput(BaseModel):
    loss: TensorData

class ForwardBackwardReverseKLResponse(BaseModel):
    outputs: list[ReverseKLItemOutput]
    metrics: dict[str, float]
```

Semantics:

- `student_input`, `reference_input`, `target_tokens`, and `weights` are aligned token-by-token
- the reference checkpoint is immutable
- the server computes full-vocabulary reverse KL without exposing logits to the client
- gradients flow only through the student model
- temperature uses the standard scaled-distribution convention internally

### POST `/api/v1/mint/action_sessions`

Create a Mint-owned action inference session for a VLA checkpoint.

Request:

```python
class MintCreateActionSessionRequest(BaseModel):
    session_id: str
    action_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None
```

Response:

```python
class MintCreateActionSessionResponse(BaseModel):
    action_session_id: str
```

Semantics:

- `base_model` may be omitted only when it can be inferred from `model_path`
- `model_path`, when provided, is resolved and access-checked under the caller's user/admin context before runtime creation
- the returned `action_session_id` is the handle for subsequent `act` and `DELETE` requests

### POST `/api/v1/mint/action_sessions/{action_session_id}/act`

Queue one action-inference request against an existing MintX action session.

Request:

```python
class MintActRequest(BaseModel):
    seq_id: int | None = None
    observation: ModelInput
    extra_inputs: dict[str, TensorData] = {}
```

Immediate route response:

```python
class UntypedAPIFuture(BaseModel):
    request_id: str
```

Resolved payload after `retrieve_future`:

```python
class ActResponse(BaseModel):
    actions: TensorData
    policy_timing: dict[str, float] | None = None
    type: Literal["act"] = "act"
```

Semantics:

- `extra_inputs.state` is required
- the request is admitted through the normal Mint capacity and work-queue path under op `mint.action.act`
- the resolved payload reuses the existing action-output shape instead of defining a parallel Mint-only action result type

### DELETE `/api/v1/mint/action_sessions/{action_session_id}`

Release one MintX action session.

Response:

```python
class MintDeleteActionSessionResponse(BaseModel):
    action_session_id: str
    status: Literal["deleted"] = "deleted"
```

Semantics:

- this endpoint is synchronous and does not return a future
- success means the server has asked the action-session manager to shut down the bound runtime for `action_session_id`

## Client Namespace

All Mint-only client helpers live under `mint.mint`.

Current helpers:

- `mint.mint.ReverseKLDatum`
- `mint.mint.InterpolateCheckpointsRequest`
- `mint.mint.InterpolateCheckpointsResponse`
- `mint.mint.ForwardBackwardReverseKLRequest`
- `mint.mint.ForwardBackwardReverseKLResponse`
- `mint.mint.interpolate_checkpoints(...)`
- `mint.mint.interpolate_checkpoints_async(...)`
- `mint.mint.forward_backward_reverse_kl(...)`
- `mint.mint.forward_backward_reverse_kl_async(...)`

This repo does not vendor the `mint.mint` action-session helper implementation.
When those helpers are updated, they must mirror the server contract above for:

- `MintCreateActionSessionRequest`
- `MintCreateActionSessionResponse`
- `MintActRequest`
- `MintDeleteActionSessionResponse`
