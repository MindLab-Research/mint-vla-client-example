# VLA Support Plan for Mint

This document describes how to add OpenPI-style VLA support to Mint while staying as close as possible to the Tinker API.

It covers two model families:

- autoregressive action-token models such as `pi0-fast`
- flow-matching action models such as `pi0.5`

## Goal

Expose VLA training and inference through the existing Mint control plane:

- `ServiceClient`
- `TrainingClient`
- async future protocol
- model/session identifiers
- weight save/load flows

The backend implementation can differ per model family. The client architecture should not.

## Non-goals

- Do not force continuous-action models through the text-only `SamplingClient`
- Do not claim that existing Tinker RL losses work unchanged for flow-matching models
- Do not depend on the original OpenPI websocket policy server as the long-term production interface

## Architecture decision

### Keep the training boundary

Keep `Datum` as the training unit.

Rationale:

- canonical Tinker `ModelInput` already supports multimodal chunks
- canonical `Datum.loss_fn_inputs` already supports arbitrary tensor payloads
- this is the smallest API change that still represents both model families honestly

### Add a sibling action inference surface

Do not overload token sampling for continuous actions.

Current canonical sampling types are token-oriented:

- request takes `prompt`
- response returns `SampledSequence.tokens`

That fits language models and could be stretched for `pi0-fast`, but it is the wrong contract for `pi0.5`.

Recommendation:

- add `ActionSamplingClient`
- add `create_action_sampling_session`
- add `act(...)`
- return action tensors and diagnostics, not token sequences

This keeps the outer Tinker shape while admitting continuous action models.

## Shared implementation work

These changes are shared by both model families.

### 1. Align Mint types with canonical Tinker

Mint's local `tinker_server/models/types.py` is currently narrower than canonical Tinker.

Required changes:

- make `ModelInput.chunks` support the canonical union of:
  - `EncodedTextChunk`
  - `ImageChunk`
  - `ImageAssetPointerChunk`
- keep `Datum.loss_fn_inputs` as a general tensor dictionary

This is prerequisite work. Otherwise the server-side contract cannot truthfully claim Tinker compatibility.

### 2. Add model-family metadata to the registry

Extend model metadata with at least:

- `policy_family`: `text_lm | ar_action_tokens | flow_action`
- `inference_modality`: `tokens | actions`
- `camera_layout`
- `action_dim`
- `action_horizon`
- `training_backend`

This removes scattered special-casing and lets route logic dispatch by model family.

### 3. Standardize observation reconstruction

The backend must reconstruct model-specific observations from Tinker inputs.

Needed decisions:

- camera ordering or camera naming
- default image masks
- state tensor key naming
- prompt token handling

The current OpenPI code expects fixed camera names such as:

- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

Tinker `ImageChunk` is positional. So Mint needs one explicit rule:

- either define canonical chunk order per model family
- or extend the image input contract with camera-role metadata

The first option is smaller. The second option is cleaner.

### 4. Add a persistent action inference actor type

Text models use vLLM. VLA models should not.

Needed actor type:

- one actor that owns a loaded OpenPI policy
- accepts structured observation input
- returns action chunk tensors

This actor should participate in:

- `ResourcePool`
- eviction
- detached actor reconciliation if made detached

## Plan for autoregressive models such as pi0-fast

## Why pi0-fast is the first target

`pi0-fast` is the best first integration target because:

- training is token-level cross-entropy over action tokens
- action generation is autoregressive
- the data contract already resembles Tinker's token training worldview

This means:

- SFT training can map cleanly onto `forward_backward(..., loss_fn="cross_entropy")`
- token-level RL losses are conceptually possible

## Data contract

Recommended `Datum` shape for `pi0-fast`:

- `model_input`
  - image chunks
  - text chunks for language prompt or task text
- `loss_fn_inputs`
  - `state`
  - `target_tokens`
  - `weights`
  - `token_ar_mask`
  - optionally `token_input_mask`

The backend adapter then reconstructs the OpenPI FAST observation format.

## Training backend

Add a dedicated backend, for example:

- `tinker_server/backend/openpi_fast_training.py`

Responsibilities:

- create and own OpenPI `pi0-fast` model state
- convert `Datum` into OpenPI observation tensors and targets
- run forward/backward and optimizer step
- save checkpoints in a Mint-owned format

Do not route this through the current language-model training worker. The supervision shape is different even if the loss name is the same.

## Inference backend

Even though `pi0-fast` is token-autoregressive internally, user-facing inference should still return action chunks.

Recommended behavior:

- action inference actor receives observation
- actor runs FAST decoding internally
- actor detokenizes to `[action_horizon, action_dim]`
- client receives action tensor result

Do not expose raw action tokens as the default user-facing inference result.

## RL support

`pi0-fast` is the only realistic first candidate for Tinker-style RL among the OpenPI families discussed here.

Why it is feasible:

- the model defines token probabilities
- canonical Tinker RL losses already operate on token-level `target_tokens`, `logprobs`, and `advantages`

What is still required:

- exact mapping from action rollout to FAST action tokens
- stable token masks so only action-token positions contribute to RL loss
- a trustworthy way to record sampling logprobs from the action inference path

Main caveat:

- user-facing inference should return action chunks, but RL still needs token logprobs internally
- the backend therefore needs a split representation:
  - internal action-token sequence for loss computation
  - external continuous action chunk for robot execution

## Weight export and serving caveat

OpenPI currently does not provide a native adapter-only serving path comparable to Mint's vLLM multi-LoRA stack.

That means the first implementation should assume:

- per-session policy checkpoints
- per-session action inference actors

Do not promise vLLM-style shared multi-LoRA serving for `pi0-fast` in the first version.

## Plan for flow-matching models such as pi0.5

## Why pi0.5 is different

`pi0.5` is currently supported in OpenPI as a flow-matching model, not an autoregressive action-token model.

Consequences:

- training target is continuous action chunks
- inference is iterative denoising / flow integration
- there is no natural token sequence output
- canonical Tinker RL losses do not directly apply

## Data contract

Recommended `Datum` shape for `pi0.5`:

- `model_input`
  - image chunks
  - optional prompt text chunks
- `loss_fn_inputs`
  - `state`
  - `actions`
  - optional masks and model-family-specific tensors

This keeps training close to Tinker:

- still `Datum`
- still `forward_backward`
- new loss function name, for example `flow_matching`

## Training backend

Add a dedicated backend, for example:

- `tinker_server/backend/openpi_flow_training.py`

Responsibilities:

- reconstruct OpenPI `Observation`
- reconstruct continuous target actions
- sample training noise and timesteps
- compute the flow-matching objective
- return diagnostics meaningful for this model family

The backend should not pretend the output is token logprobs. It should return metrics that actually matter, such as:

- flow loss
- action MSE surrogates if useful
- timing and memory diagnostics

## Inference backend

Flow inference needs its own serving path.

Recommended API shape:

- `create_action_sampling_session(...)`
- `ActionSamplingClient.act(...)`
- result contains:
  - `actions`
  - optional inference diagnostics such as `num_steps`, `infer_ms`

The actor implementation then:

- builds the observation
- initializes noise
- runs the configured denoising steps
- returns the final action chunk

## RL and custom loss challenges for flow-matching models

This is the main caveat.

### Built-in Tinker RL losses do not fit

Canonical Tinker RL losses such as:

- `importance_sampling`
- `ppo`
- `cispo`
- `dro`

all assume token logprobs.

That assumption breaks for flow-matching models because the model does not naturally expose exact token log probabilities or a simple action density in the current API shape.

### `forward_backward_custom` also does not fit well

Current Tinker custom loss flow assumes:

1. do a forward pass
2. obtain token logprobs
3. let user compute a differentiable custom scalar from those logprobs
4. backprop through that representation

That is a language-model custom-loss interface. It is not a good generic interface for flow-matching models.

If we want custom loss support for flow models, we need a new boundary. Options:

1. `forward_backward_custom_continuous`
   - expose continuous model outputs such as predicted velocity or denoising residual
   - user computes custom loss from those outputs

2. a more general `forward_with_outputs`
   - backend returns a typed family-specific output object
   - custom loss support becomes model-family aware

Option 2 is more honest but larger.

### Recommended first release policy

For flow-matching models:

- support SFT first
- defer RL
- defer generic custom-loss support unless there is a concrete use case

That is not a temporary excuse. It follows from the actual mathematical boundary of the existing Tinker RL interface.

## Backend choice caveats from upstream OpenPI

The upstream OpenPI repo has constraints that matter for implementation planning:

- the current JAX training script does not support multi-node training
- PyTorch support exists for `pi0` and `pi0.5`
- PyTorch support does not currently include `pi0-fast`
- PyTorch support does not currently include LoRA training

Implications:

- `pi0-fast` integration should start from the JAX path, not the PyTorch path
- `pi0.5` could use JAX or PyTorch for some workloads, but LoRA-focused integration still points back toward JAX
- we should not assume upstream OpenPI can drop directly into Mint's current distributed training patterns without adaptation

## Suggested rollout order

1. Align Mint server types with canonical Tinker multimodal `ModelInput`
2. Add action-sampling session and client types
3. Implement `pi0-fast` SFT training and action inference
4. Add `pi0-fast` token-logprob plumbing needed for RL
5. Implement `pi0.5` SFT training and action inference
6. Reassess whether flow-model custom loss is justified
7. Treat flow-model RL as a separate research project, not as an automatic extension of the existing Tinker RL stack

## Failure modes to watch

- claiming Tinker compatibility while keeping the narrowed local `ModelInput` type
- overloading token `SamplingClient` for continuous-action models
- pretending `ppo` is available for `pi0.5` just because `TrainingClient.forward_backward` accepts a loss string
- promising vLLM-style multi-LoRA economics for OpenPI checkpoints without a real adapter-serving path
- hiding camera-order assumptions in undocumented backend code

## Bottom line

The smallest honest design is:

- keep `Datum`
- keep `TrainingClient`
- add `flow_matching` as a new training loss family
- add a sibling action-inference client
- implement RL first only for autoregressive action-token models

That stays close to Tinker where the contract is genuinely reusable and splits only where the current token-sampling contract is mathematically the wrong abstraction.
