# Using OpenPI Models Through the Current Mint API

This file is background material, not the normative contract.

Historical source materials referenced by this guide were:

- the PR422 Mint OpenPI VLA target document
- the PR422 VLA docs README
- the PR422 VLA sub-target documents
- the verified stage-local scripts under `src/mindlab-toolkit/examples/`

This guide exists for readers who already know the upstream OpenPI workflow and want a concise map from that workflow to the Mint surface tested during the VLA work. Current Mint architecture guidance lives in this `references/` directory.

## What Mint changes and what it does not

Mint does not replace OpenPI model logic. It changes the service boundary around those models.

The stable user-side shape is now:

- `import mint`
- create a `ServiceClient`
- create a `TrainingClient` for `pi0-fast` or `pi0.5`
- send VLA `Datum` batches
- call `train_step(...)` as the default public training unit
- export weights into an `ActionSamplingClient`
- run action inference through Mint-owned action sessions

Mint keeps three things stable for the user:

- `Datum` remains the training unit
- async future semantics remain the long-running request contract
- action inference remains part of the same Mint service, not a separate upstream websocket service

Mint changes three things relative to old OpenPI-facing sketches:

- public training examples now use `train_step(...)`, not `forward_backward(...)` plus `optim_step(...)`
- public action inference now goes through Mint-owned `/api/v1/mint/action_sessions*`
- default action execution now goes through Mint queue, capacity, and Ray actor control-plane semantics

The older split-step examples were useful while the integration surface was still being discovered. They are no longer the public teaching path.

## Why the current Mint surface exists

The upstream OpenPI workflow is still appropriate when you want to work directly inside `src/openpi`:

- run upstream training scripts
- construct policies directly
- use the upstream remote-inference protocol as documented by OpenPI

That workflow is not the target production surface for this repo. The Mint surface exists because shared deployment needs a control plane that already handles:

- async futures and request tracking
- queueing and capacity control
- checkpoint lifecycle
- actor inventory and cleanup
- one client pattern across `pi0-fast` and `pi0.5`

The point is not abstraction for its own sake. The point is that callers should not need to rebuild service glue around each OpenPI model family.

## Current mental model

Treat Mint as the only primary service boundary.

From the user side:

1. build `Datum` values that contain images, text tokens, and tensor-valued supervision
2. submit one or more training steps through `TrainingClient.train_step(...)`
3. export a named checkpoint into an action client
4. call `act(...)`
5. clean up the action session and training model

From the server side:

- training stays on the standard Mint future path
- action inference stays on the Mint-owned action-session path
- action requests enter the same future, queue, and capacity control plane rather than bypassing it with host-local background tasks

## Current public training boundary

The public training boundary is now one atomic step:

```python
training_client.train_step(
    data=[datum],
    loss_fn="cross_entropy" | "flow_matching" | ...,
    adam_params=mint.types.AdamParams(...),
).result()
```

That is the default public contract for OpenPI training in this repo.

Internal service code still retains lower-level `forward`, `backward`, and optimizer-step semantics, but that lower-level split is no longer the public guide for shared deployment. The reason is multi-tenant scheduling: the public unit should be the whole step that Mint can enqueue, account for, and rotate fairly.

## Current public action boundary

The current action boundary is Mint-owned:

- `POST /api/v1/mint/action_sessions`
- `POST /api/v1/mint/action_sessions/{action_session_id}/act`
- `DELETE /api/v1/mint/action_sessions/{action_session_id}`

Callers do not need to construct those routes directly. They use:

- `ServiceClient.create_action_sampling_client(...)`
- `TrainingClient.save_weights_and_get_action_sampling_client(...)`
- `ActionSamplingClient.act(...)`
- `ActionSamplingClient.shutdown()`

The important change is operational, not cosmetic. `act(...)` is no longer a side path outside Mint control-plane semantics. It creates a future, passes through capacity and queue control, and lands on a Mint-managed Ray actor runtime.

## Example shape: pi0-fast

This is the current public shape. It is intentionally schematic. For concrete builders of `Datum` and `observation`, use the verified examples under `src/mindlab-toolkit/examples/`, especially:

- `st04_pi0_fast_action_inference_acceptance.py`
- `st06_mint_vla_minimal_closure.py`
- `st09a_mintx_action_boundary_acceptance.py`

```python
from __future__ import annotations

import mint

types = mint.types

service = mint.ServiceClient(
    base_url="REPLACE_WITH_BASE_URL",
    api_key="REPLACE_WITH_API_KEY",
)

training_client = service.create_lora_training_client(
    base_model="openpi/pi0-fast-libero-low-mem-finetune",
    rank=16,
    train_attn=True,
    train_mlp=True,
    train_unembed=True,
)

datum = build_pi0_fast_datum(mint_module=mint)
observation = build_pi0_fast_observation(mint_module=mint)

train_step = training_client.train_step(
    data=[datum],
    loss_fn="cross_entropy",
    adam_params=types.AdamParams(learning_rate=1e-4),
).result()

action_client = training_client.save_weights_and_get_action_sampling_client(
    "pi0-fast-example",
)

action_result = action_client.act(
    observation=observation,
    extra_inputs={
        "state": types.TensorData(
            data=[0.0] * 8,
            shape=[8],
            dtype="float32",
        ),
    },
).result()

action_client.shutdown().result()
training_client.delete_model().result()
```

The important part is not the exact datum builder. The important part is the control flow:

- `train_step(...)` is the public training unit
- weight export hands off directly into an action client
- action inference returns an action tensor payload, not a text-sampling payload
- cleanup is explicit

## Example shape: pi0.5

`pi0.5` uses the same control-plane shape, but the loss semantics are different.

For concrete builders of `Datum` and `observation`, use the verified examples under:

- `st05_pi05_sft_action_inference_acceptance.py`
- `st06_mint_vla_minimal_closure.py`
- `st09a_mintx_action_boundary_acceptance.py`

```python
from __future__ import annotations

import mint

types = mint.types

service = mint.ServiceClient(
    base_url="REPLACE_WITH_BASE_URL",
    api_key="REPLACE_WITH_API_KEY",
)

training_client = service.create_lora_training_client(
    base_model="openpi/pi05-libero-low-mem-finetune",
    rank=16,
    train_attn=True,
    train_mlp=True,
    train_unembed=True,
)

datum = build_pi05_datum(mint_module=mint)
observation = build_pi05_observation(mint_module=mint)

train_step = training_client.train_step(
    data=[datum],
    loss_fn="flow_matching",
    adam_params=types.AdamParams(learning_rate=1e-4),
).result()

action_client = training_client.save_weights_and_get_action_sampling_client(
    "pi05-example",
)

action_result = action_client.act(
    observation=observation,
    extra_inputs={
        "state": types.TensorData(
            data=[0.0] * 8,
            shape=[8],
            dtype="float32",
        ),
    },
).result()

action_client.shutdown().result()
training_client.delete_model().result()
```

What stays the same:

- one `ServiceClient`
- one `TrainingClient`
- one `train_step(...)` public training unit
- one action-session handoff
- one explicit cleanup sequence

What changes is the model semantics, not the service shape.

## Runtime and deployment notes

Do not interpret successful private development runs as proof that a shared runtime artifact is correct.

The current runtime contract is repo-owned:

- `src/mint/pyproject.toml` declares the canonical `tool.mint.runtime_env` sources and host requirements
- the `openpi` source contract includes both `src` and `packages/openpi-client/src`
- host-side requirements explicitly include the OpenPI worker stack such as `jax[cuda12]`, `flax`, `optax`, `orbax-checkpoint`, `ml_collections`, `jaxtyping`, `augmax`, `tqdm-loggable`, and `tyro`
- `src/mint/scripts/build_runtime_env.py --inspect --env-root ...` is the standard probe for manifest, layout, and host-python import checks

The current rule is therefore:

- do not rely on private workspace path guesses
- do not rely on `unison` code sync alone as proof that runtime artifacts are current
- do not use this guide as evidence that a shared runtime root is valid

Runtime validity must be proven separately by the runtime builder and inspect probes.

Current rollout status is narrower than "shared deployment already switched":

- the shared candidate runtime root has been verified privately
- the rollback baseline still exists
- `pi0.5` shared-service allowlist and repo fallback default list remain separate decisions
- 2026-04-01 read-only checks found that `mint-dev` currently has no live shared service on `8000`, `18000`, or `18080`, so public/shared root cutover is still an operational pending item rather than an already-executed step

## Relation to the upstream OpenPI workflow

Use the upstream OpenPI repo directly when your goal is:

- to reproduce upstream scripts exactly
- to experiment inside upstream policy/training code without Mint service boundaries
- to validate upstream behavior before integrating it into Mint

Use the Mint surface when your goal is:

- one service boundary for training and action inference
- async future semantics
- Mint-managed scheduling and task-state behavior
- Mint-managed actor lifecycle
- a stable client pattern across `pi0-fast` and `pi0.5`

## What this guide is and is not

This file is now aligned with the current public Mint shape:

- `train_step(...)` is the public training guide
- MintX `/api/v1/mint/action_sessions*` is the public action guide
- action inference is part of the Mint control plane

This file is still not the normative source.

At the time this guide was written, if there was any conflict between this file and:

- the PR422 Mint OpenPI VLA target document
- the PR422 VLA docs README
- the PR422 VLA sub-target documents
- the verified stage-local scripts

the then-normative docs and verified examples won. For current architecture, use this `references/` directory.
