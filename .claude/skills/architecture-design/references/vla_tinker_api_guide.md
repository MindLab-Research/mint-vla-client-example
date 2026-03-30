# Using OpenPI Models Through a Tinker-Style API

This file is a historical design sketch, not the current normative contract.

Current normative sources are:

- `docs/mint-openpi-vla-target.md`
- `docs/README.md`
- `docs/sub-targets/*.md`
- the verified toolkit stage-local scripts under `src/mindlab-toolkit/examples/`

This guide still contains pre-`train_step` and pre-MintX action-boundary examples. Until it is rewritten to match the current docs, do not use it as merge-readiness evidence or as the final interaction contract.

This document is for users who already know the OpenPI workflow and want to understand what Mint is trying to preserve and what it is trying to improve. The goal is not to replace OpenPI's model logic or claim that the original repo workflow is wrong. The goal is to expose the same model families through a service-style API that is easier to integrate into larger training and deployment systems.

The code examples below describe the intended Mint client surface (Tinker-style). The user entrypoint is `import mint` (provided by `src/mindlab-toolkit`), which re-exports a Tinker-compatible client plus a Mint-owned action inference client.

## Why prefer a Tinker-style API

If you are already using OpenPI directly, the current workflow is workable for local experimentation:

- train with repo-local scripts such as `scripts/train.py`
- run inference by constructing a policy or starting `scripts/serve_policy.py`
- connect your robot or eval loop to the OpenPI websocket policy server

That is a reasonable layout for model development inside the OpenPI repo. The friction appears when you want to operationalize the same models in a shared service environment.

Typical pain points are:

- training and inference use different transports and lifecycle assumptions
- queueing, retries, auth, and request tracking are outside the core OpenPI model interface
- moving from one model family to another leaks backend details into application code
- production systems often want asynchronous request handling and checkpoint lifecycle management, not just a local script plus a websocket server

A Tinker-style API is useful in that setting because it gives OpenPI users one control plane for:

- model creation and session management
- async futures and `request_id` tracking
- retries and queue-aware client behavior
- weight save/load and inference handoff
- a consistent client architecture across autoregressive and flow-matching VLA models

The point is not abstract simplicity. The point is that an OpenPI user can keep working with OpenPI model families while reducing the amount of service glue they have to build around them.

## Design goal

The design goal is to keep the parts OpenPI users care about stable:

- the underlying model family
- the semantics of the training target
- the observation and action structure

while replacing the operational surface with something more uniform:

- keep `ServiceClient`
- keep `TrainingClient`
- keep `Datum`
- keep async future and polling semantics
- add the smallest possible sibling to token sampling for continuous robot actions

The intended result is that changing model family should not force a rewrite of the surrounding application architecture.

## How the intended API should feel

From the user side, the workflow should feel like a service wrapper around OpenPI models rather than a new model stack:

1. create a training client
2. send batches of multimodal `Datum`
3. run `forward_backward` and `optim_step`
4. export weights for inference
5. create an action-sampling client and call `act(...)`

The backend can still do model-specific work such as FAST tokenization or flow-based denoising. The point is that the client code should stay focused on robot observations, training data, and deployment flow rather than on transport details or model-family-specific serving logic.

## Why this is better than the original OpenPI workflow

### Stable async semantics

Tinker already gives a consistent async contract:

- long-running work returns futures
- the client can overlap `forward_backward` and `optim_step`
- production deployments already understand this control flow

OpenPI's original workflow is more script-centric and less suited to shared, queued service operation.

### One training abstraction

Tinker `Datum` already supports multimodal `ModelInput` chunks and arbitrary tensor-valued `loss_fn_inputs`. That is a much better common boundary for serving multiple model families than exposing OpenPI's internal `Observation` struct directly.

### One production integration path

Mint already owns:

- auth
- queueing
- routing
- request lifecycle
- weight save/load semantics
- actor lifecycle

Reusing that control plane is better than teaching every application to talk to a separate OpenPI websocket server.

### Easier model-family switching

If `pi0-fast` and `pi0.5` are both exposed through the same Tinker-like client pattern, the user can switch model families without replacing their surrounding orchestration.

## Intended client example: pi0-fast

`pi0-fast` is the easiest first target because it is autoregressive over action tokens. Internally it can use FAST tokenization. Externally the user should still send `Datum` and receive action chunks.

```python
from __future__ import annotations

import numpy as np
import mint

types = mint.types


service = mint.ServiceClient(base_url="REPLACE_WITH_BASE_URL", api_key="REPLACE_WITH_API_KEY")

training_client = service.create_lora_training_client(
    base_model="openpi/pi0-fast-libero-low-mem-finetune",
    rank=16,
    train_attn=True,
    train_mlp=True,
    train_unembed=True,
)


def build_pi0_fast_datum(
    *,
    prompt_tokens: list[int],
    image_chunks: list[types.ImageChunk],
    state: np.ndarray,
    target_action_tokens: np.ndarray,
    weights: np.ndarray,
    token_ar_mask: np.ndarray,
) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput(
            chunks=[
                *image_chunks,
                types.EncodedTextChunk(tokens=prompt_tokens),
            ]
        ),
        loss_fn_inputs={
            "state": types.TensorData(
                data=state.astype(np.float32).reshape(-1).tolist(),
                shape=[int(state.size)],
                dtype="float32",
            ),
            "target_tokens": types.TensorData(
                data=target_action_tokens.astype(np.int64).reshape(-1).tolist(),
                shape=[int(target_action_tokens.size)],
                dtype="int64",
            ),
            "weights": types.TensorData(
                data=weights.astype(np.float32).reshape(-1).tolist(),
                shape=[int(weights.size)],
                dtype="float32",
            ),
            "token_ar_mask": types.TensorData(
                data=token_ar_mask.astype(np.int64).reshape(-1).tolist(),
                shape=[int(token_ar_mask.size)],
                dtype="int64",
            ),
        },
    )


batch = [
    build_pi0_fast_datum(
        prompt_tokens=[2, 314, 271, 99],
        image_chunks=[
            types.ImageChunk(data=open("base.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("left.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("right.png", "rb").read(), format="png", expected_tokens=256),
        ],
        state=np.zeros([8], dtype=np.float32),
        target_action_tokens=np.array([101, 102, 103, 104], dtype=np.int64),
        weights=np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        token_ar_mask=np.array([0, 1, 1, 1], dtype=np.int64),
    )
]

fwdbwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

fwdbwd_result = fwdbwd_future.result()
optim_result = optim_future.result()

save_name = "pi0-fast-example"
action_client = training_client.save_weights_and_get_action_sampling_client(save_name)

action_result = action_client.act(
    observation=types.ModelInput(
        chunks=[
            types.ImageChunk(data=open("base.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("left.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("right.png", "rb").read(), format="png", expected_tokens=256),
            types.EncodedTextChunk(tokens=[2, 314, 271, 99]),
        ]
    ),
    extra_inputs={
        "state": types.TensorData(
            data=np.zeros([8], dtype=np.float32).tolist(),
            shape=[8],
            dtype="float32",
        ),
    },
).result()

actions = np.asarray(action_result["actions"]["data"], dtype=np.float32).reshape(
    action_result["actions"]["shape"]
)
```

### Why this is a good fit

For `pi0-fast`, the training loop still looks like standard Tinker:

- `Datum`
- `forward_backward`
- `optim_step`

The main difference is that the backend interprets the target as action tokens instead of plain language tokens.

## Intended client example: pi0.5

`pi0.5` is a flow-matching model, not an autoregressive action-token model. The preferred client surface should still look similar, but the loss name and inference engine are different.

```python
from __future__ import annotations

import numpy as np
import mint

types = mint.types


service = mint.ServiceClient(base_url="REPLACE_WITH_BASE_URL", api_key="REPLACE_WITH_API_KEY")

training_client = service.create_lora_training_client(
    base_model="openpi/pi05-libero-low-mem-finetune",
    rank=16,
    train_attn=True,
    train_mlp=True,
    train_unembed=True,
)


def build_pi05_flow_datum(
    *,
    prompt_tokens: list[int],
    image_chunks: list[types.ImageChunk],
    state: np.ndarray,
    target_actions: np.ndarray,
) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput(
            chunks=[
                *image_chunks,
                types.EncodedTextChunk(tokens=prompt_tokens),
            ]
        ),
        loss_fn_inputs={
            "state": types.TensorData(
                data=state.astype(np.float32).reshape(-1).tolist(),
                shape=[int(state.size)],
                dtype="float32",
            ),
            "actions": types.TensorData(
                data=target_actions.astype(np.float32).reshape(-1).tolist(),
                shape=[int(target_actions.shape[0]), int(target_actions.shape[1])],
                dtype="float32",
            ),
        },
    )


batch = [
    build_pi05_flow_datum(
        prompt_tokens=[2, 314, 271, 99],
        image_chunks=[
            types.ImageChunk(data=open("base.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("left.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("right.png", "rb").read(), format="png", expected_tokens=256),
        ],
        state=np.zeros([8], dtype=np.float32),
        target_actions=np.zeros([10, 7], dtype=np.float32),
    )
]

fwdbwd_future = training_client.forward_backward(batch, loss_fn="flow_matching")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

fwdbwd_result = fwdbwd_future.result()
optim_result = optim_future.result()

save_name = "pi05-example"
action_client = training_client.save_weights_and_get_action_sampling_client(save_name)

action_result = action_client.act(
    observation=types.ModelInput(
        chunks=[
            types.ImageChunk(data=open("base.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("left.png", "rb").read(), format="png", expected_tokens=256),
            types.ImageChunk(data=open("right.png", "rb").read(), format="png", expected_tokens=256),
            types.EncodedTextChunk(tokens=[2, 314, 271, 99]),
        ]
    ),
    extra_inputs={
        "state": types.TensorData(
            data=np.zeros([8], dtype=np.float32).tolist(),
            shape=[8],
            dtype="float32",
        ),
    },
).result()

actions = np.asarray(action_result["actions"]["data"], dtype=np.float32).reshape(
    action_result["actions"]["shape"]
)
```

### Why this still feels like Tinker

The flow-matching model does not produce token samples, but the surrounding client flow still matches Tinker:

- one service client
- one training client
- one async future model
- one export-and-infer step

The difference is that inference returns action tensors instead of token sequences.

## What changes for the user

Very little should change in application structure:

- batching still happens client-side
- `Datum` is still the training unit
- weights are still saved and reused through the same service
- async futures still represent long-running work

The main new concept is that VLA inference should use an action-sampling client instead of the text-only `SamplingClient`.

## Scope and status

The main status distinctions are:

- `pi0-fast` is the best first fit for Tinker-style integration because it is autoregressive
- `pi0.5` can still fit the training API closely, but inference needs an action-output surface
- `pi0.5` does not have an autoregressive variant in the current OpenPI repo

## Recommended user mental model

Use Tinker-style APIs when you want:

- one client pattern across model families
- asynchronous training and inference requests
- production-grade routing and queue semantics
- stable checkpoint and session handling

Use the original OpenPI scripts when you want:

- direct experimentation inside the OpenPI repo
- local reproduction of their released examples
- the fastest path to matching the original repo workflow exactly
