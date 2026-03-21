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
