# VLA runtime architecture

This file is the current Mint runtime reference for OpenPI/VLA models. It
describes the `mint_server` implementation on this branch, not the historical
PR422 sketches and not the upstream `tinker_server` runtime.

Historical VLA files in this directory remain background material only:

- `vla_mint_api_guide.md`
- `vla_implementation_plan.md`
- `vla_benchmark_demo_research.md`
- `vla_deterministic_startup_runbook.md`

## Scope

Mint treats VLA as an action-model backend family inside the same FastAPI,
TaskStateStore, ModelWorkScheduler, ModelActorSupervisor, and Ray actor system
used by text training/inference. The user-facing action API is Mint-owned, and
OpenPI model code runs inside Mint Ray actors.

Current supported model descriptors live in `mint_server/backend/model_registry.py`:

- `openpi/pi0-fast-libero-low-mem-finetune`
  - `training_backend="openpi_fast"`
  - `policy_family="ar_action_tokens"`
  - `inference_modality="actions"`
  - `camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`
  - `action_dim=7`
  - `action_horizon=10`
  - `action_token_budget=64`
- `openpi/pi05-libero-low-mem-finetune`
  - `training_backend="openpi_pi05"`
  - `policy_family="flow_action"`
  - `inference_modality="actions"`
  - same camera layout
  - `action_dim=32`
  - `action_horizon=10`

These model configs are capability descriptors and dispatch metadata. Deployment
exposure is still controlled by `MINT_SUPPORTED_MODELS` and model access policy.

## Runtime boundary

The current OpenPI runtime has no subprocess boundary and no stdout JSON-RPC
worker protocol.

The direct runtime stack is:

1. `OpenPIFastRuntimeSpec`
   - carries `worker_module`, `pythonpath`, timeouts, `cwd`, and `extra_env`
   - is shared by FAST, pi0.5, training, and action runtimes
   - does not carry a Python executable
2. `OpenPIDirectWorkerClient`
   - imports `spec.worker_module` inside the Ray actor process
   - creates one of the known worker session classes from that module
   - calls the module-level `_dispatch(session, op, payload)` directly
   - wraps Python exceptions as `OpenPIFastWorkerRemoteError`
3. Worker modules
   - training: `openpi_fast_worker.py`, `openpi_pi05_worker.py`
   - action: `openpi_fast_action_worker.py`, `openpi_pi05_action_worker.py`
   - each module owns OpenPI/JAX model construction, checkpoint IO, op
     semantics, and session state save/load

`OpenPIFastWorkerProtocolError` is still the local error class for invalid
direct-runtime payloads, Ray timeouts, and closed clients. The name is legacy;
it no longer means stdout protocol parsing exists.

Deleted legacy symbols must not be reintroduced:

- `OpenPIFastWorkerClient`
- `OpenPIFastActionWorkerClient`
- worker `main()` functions
- `_install_protocol_stdout_redirect`
- `_reply`
- `_dispatch_with_protocol_stdout`
- `OPENPI_FAST_WORKER_PROTOCOL_VERSION`
- subprocess-only `python_executable` / `build_env()`

## Training flow

Training starts on the normal Mint training routes:

- `POST /api/v1/create_model` validates OpenPI create requests through
  `validate_openpi_fast_create_request()` and
  `validate_openpi_pi05_create_request()`.
- `POST /api/v1/mint/vla/train_step` lowers VLA request payloads into the
  internal `TrainStepRequest` shape and enqueues work through
  `ModelWorkScheduler`.
- Scheduler metadata forces OpenPI train-step work through the scheduler and
  uses round-robin fairness with `scheduler_max_consecutive=1`.

Backend dispatch is selected from `ModelConfig.training_backend`:

- `openpi_fast` -> `OpenPIFastTrainingEngine`
- `openpi_pi05` -> `OpenPIPi05TrainingEngine`

Both training engines use `start_openpi_shared_ray_runtime()` by default. The
shared runtime creates or attaches to a named single-GPU Ray actor keyed by:

- base model
- worker module
- OpenPI config name
- action dimensions and horizon
- model token limit
- request/create/save/load timeouts

The shared actor runs `OpenPISharedRayRuntimeActor`, whose core owns a single
`OpenPIDirectWorkerClient`. It registers many Mint training sessions against the
same OpenPI worker by saving and loading per-session state around requests. The
shared runtime is therefore the GPU runtime boundary; the FastAPI worker and
`TrainingSessionManager` keep only process-local client handles and counters.

Training op mapping:

- `create_session`
  - builds OpenPI config/model state in the Ray actor
  - saves a reusable template state and the first session state
- `forward_backward`
  - FAST supports `cross_entropy`, `importance_sampling`, and `ppo`
  - pi0.5 supports `flow_matching`, `importance_sampling`, and `ppo`
  - payload builders convert Mint `Datum` images, encoded text, state tensors,
    action tensors, logprobs, and advantages into worker JSON-compatible dicts
- `optim_step`
  - applies Adam params and advances the Mint session step
- `train_step`
  - public Mint unit: `forward_backward` followed by `optim_step`
- `save_weights_for_sampler`
  - exports an OpenPI checkpoint directory for action inference
- `save_weights` / `load_weights`
  - use OpenPI checkpoint paths and may restore current step and learning rate
- `shutdown`
  - closes the runtime client and unregisters/clears session state as needed

OpenPI training does not use vLLM or text sampling sessions for action models.
When pi0.5 saves weights for sampler, the route returns the checkpoint path
instead of creating a text Multi-LoRA sampling session.

## Action inference flow

The current public action boundary is under `/api/v1/mint`:

- `POST /action_sessions`
- `POST /action_sessions/{action_session_id}/act`
- `DELETE /action_sessions/{action_session_id}`

`ActionSessionRouter` chooses the manager from `base_model`:

- FAST -> `OpenPIFastActionSessionManager`
- pi0.5 -> `OpenPIPi05ActionSessionManager`

Action session creation resolves `model_path` through Mint checkpoint resolution,
builds an OpenPI create payload, and obtains a runtime client. The runtime
client is either recovered from supervisor inventory or created by the configured
runtime factory.

Default factory behavior matters:

- FAST action runtime must be reconciled by `ModelActorSupervisor` before
  request handling. The default factory fails if a runtime actor is not already
  available.
- pi0.5 has the same supervisor-first behavior by default, but
  `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1` enables a direct actor creation
  path for that family.

Action request execution has two modes:

- Normal queued mode:
  - `/mint/action_sessions/{id}/act` creates a Mint future
  - request JSON is enqueued to `ModelWorkScheduler`
  - the work domain is the internal runtime domain
  - `action_session:{action_session_id}` is both affinity and ordering key
  - `routes/action_sampling.py` runs the actual manager `act(...)` call
- pi0.5 direct mode:
  - guarded by `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME`
  - route creates and resolves/fails the future inline
  - this is an explicit bypass and should not become the general action path

Runtime actor creation uses `OpenPIActionRayRuntimeActor` or the shared
`OpenPISharedRayRuntimeActor` family. Action actors are single-GPU actors and
publish `ActorType.OPENPI` entries through `ModelActorSupervisor` inventory.
Clients mark inflight, set the current session, touch the actor, and unregister
on close.

## State ownership

OpenPI/VLA state is split by authority:

- `TaskStateStore`
  - durable Mint task/future metadata
  - training-session metadata through compatibility facades
  - usage/billing observations
- `ModelWorkScheduler`
  - hot scheduling, leases, ordering, and fairness
- `ModelActorSupervisor`
  - OpenPI runtime actor inventory, readiness, inflight accounting, and recovery
  - actor registration is `ActorType.OPENPI`
- FastAPI process
  - transient `ActionSessionRouter` and training runtime-client caches
  - not a durable authority
- Ray actor process
  - OpenPI/JAX model objects
  - current worker session object
  - per-session train/action state save/load via worker modules
- filesystem
  - OpenPI checkpoints and exported sampler/action checkpoints
  - action session state roots under
    `checkpoints/openpi_action_session_state/<namespace>/<actor_name>`

API worker restart loses local client maps, but action managers can recover
detached actors by scanning `ModelActorSupervisor` entries and reconstructing
`OpenPIActionRayRuntimeClient` or `OpenPISharedRayRuntimeClient` handles when
inventory metadata matches the action session.

## Runtime environment

OpenPI Ray actors receive bootstrap env from `_openpi_runtime_env_vars()`:

- all `MINT_OPENPI_*` keys present in the API environment
- `XLA_FLAGS` from `MINT_OPENPI_XLA_FLAGS`
- `HF_HOME`, `HF_HUB_OFFLINE`, and `OPENPI_DATA_HOME` when set
- `PYTHONDONTWRITEBYTECODE=1`
- the standard actor runtime env vars built with `PFS_PYTHONPATH`

`OpenPIFastRuntimeSpec.from_env()` resolves:

- `MINT_OPENPI_FAST_PYTHONPATH`
  - explicit actor import path for OpenPI
  - when unset, uses `bootstrap_runtime_pythonpath(...)`
- `PFS_RUNTIME_ENV_ROOT`
  - validated with `require_host_python=True` when set
- `MINT_OPENPI_FAST_WORKER_MODULE`
- `MINT_OPENPI_FAST_STARTUP_TIMEOUT_S`
- `MINT_OPENPI_FAST_REQUEST_TIMEOUT_S`
- `MINT_OPENPI_FAST_CREATE_SESSION_TIMEOUT_S`
- `MINT_OPENPI_FAST_SAVE_TIMEOUT_S`
- `MINT_OPENPI_FAST_LOAD_TIMEOUT_S`
- `MINT_OPENPI_FAST_CWD`

Action runtime spec additionally resolves:

- `MINT_OPENPI_FAST_ACTION_WORKER_MODULE`
- `MINT_OPENPI_FAST_ACTION_STARTUP_TIMEOUT_S`
- `MINT_OPENPI_FAST_ACTION_REQUEST_TIMEOUT_S`
- `MINT_OPENPI_FAST_ACTION_CREATE_SESSION_TIMEOUT_S`
- `MINT_OPENPI_FAST_ACTION_CWD`

Other OpenPI runtime knobs consumed by workers or actor launchers include:

- `MINT_OPENPI_FAST_WEIGHTS_PATH`
- `MINT_OPENPI_FAST_RANDOM_INIT`
- `MINT_OPENPI_PI05_WEIGHTS_PATH`
- `MINT_OPENPI_PI05_RANDOM_INIT`
- `MINT_OPENPI_FAST_CHECKPOINT_BASE_DIR`
- `MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR`
- `MINT_OPENPI_FAST_ASSETS_BASE_DIR`
- `MINT_OPENPI_PI05_ASSETS_BASE_DIR`
- `MINT_OPENPI_FAST_TOKENIZER_PATH`
- `MINT_OPENPI_FAST_DEBUG_TOKENS`
- `MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT`
- `MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S`
- `MINT_OPENPI_ACTION_CAPACITY_RETRY_TIMEOUT_S`
- `MINT_OPENPI_ACTION_CAPACITY_RETRY_INTERVAL_S`
- `MINT_MODEL_PLACEMENT_JSON`
- `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME`
- `MINT_OPENPI_ALLOW_RUNTIME_NODE_SAMPLER_PATH`

Do not document or depend on subprocess-only OpenPI env vars as current
contract. If a key is only preserved in ConfigActor allowlists for compatibility
and no current OpenPI code reads it, treat it as compatibility residue rather
than runtime architecture.

## Placement and inventory

OpenPI runtime actors are single-GPU Ray actors. Placement can be pinned through
`MINT_MODEL_PLACEMENT_JSON`; both shared training/runtime actors and action
runtime actors validate that the selected placement is exactly one GPU on one
alive node before applying a `NodeAffinitySchedulingStrategy`.

Actor names:

- shared training/action pool actors: `mint_openpi_shared_<sha1-prefix>`
- dedicated action actors: `mint_openpi_action_<sha1-prefix>`

All live OpenPI actors must be published to `ModelActorSupervisor` inventory as
`ActorType.OPENPI`. Runtime launchers or runtime creation helpers are allowed to
create backend-specific Ray actors, but they must publish enough metadata for
supervisor inventory, health, inflight accounting, and API restart recovery.

## Observability and failure handling

Direct in-actor execution means OpenPI exceptions now appear as normal Ray task
errors or as `OpenPIFastWorkerRemoteError` with a captured traceback string. The
old failure mode where Ray/autoscaler stdout text corrupted a JSON-RPC channel
should not exist because stdout is no longer a control channel.

Operational checks should look at:

- `ModelActorSupervisor` inventory entries for `ActorType.OPENPI`
- actor metadata: `worker_module`, `node_ip`, `pid`, `cuda_visible_devices`
- task/future state in `TaskStateStore`
- scheduler domain queues and leases
- Ray actor stderr/logs for OpenPI/JAX exceptions

## Known open work

This document records the current runtime, including remaining non-final
constraints:

- PR #698 is not merge-ready until the external uvicorn multi-worker/control
  plane authority gate is fixed and the branch is rebased.
- Worker/state ownership cleanup is still in progress. The desired end state is
  that API workers remain stateless and OpenPI runtime/session authority is
  recoverable through detached control-plane state and supervisor inventory.
- Live validation is still required after the worker/state cleanup and rebase.
