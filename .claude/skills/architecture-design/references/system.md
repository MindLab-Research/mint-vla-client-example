# System boundary and code map

## System boundary

tinker-server ("MinT") is a FastAPI service that implements a Tinker-compatible REST API and brokers training/inference to Ray GPU actors.

- Clients: `tinker` SDK (ServiceClient/TrainingClient/SamplingClient) and `tinker-cookbook` recipes.
- API server: `tinker_server/app.py` (FastAPI). Owns HTTP, auth, request validation, and lightweight in-process state.
- Compute: Ray actors on GPU nodes (training workers, vLLM inference servers). Own model weights and do the heavy compute.
- Storage/transfer:
  - Ray object store for tensor transfer (small/medium payloads).
  - Shared filesystem (PFS) for artifacts that are too large to serialize through Ray.

High-level flow:

client -> HTTP -> tinker-server (FastAPI) -> Ray -> GPU actors

## Ray persistent (detached) actors

Some control-plane and GPU actors are created as detached Ray actors in a fixed namespace so they can be reused across API server restarts.

Control-plane detached actors:
- `tinker_future_store`: async future state and result refs (results stored in Ray object store).
- `tinker_capacity_manager`: admission control for async backlog (queue bytes and object store bytes).
- `tinker_api_work_queue`: stores request JSON for async operations; API workers dequeue and execute.

Implications:
- A server restart loses in-process mappings (sessions, registries), but detached actors may still exist and hold GPU memory.
- Startup reconciliation is required to rediscover or kill these actors (`tinker_server/app.py:_cleanup_stale_actors`).
- Changes to actor code require recreating the actor. Detached actors do not hot-reload.

## Code map (where to look)

- `tinker_server/app.py`
  - FastAPI startup/shutdown (lifespan), auth middleware, route registration.
  - Startup actor reconciliation: `_cleanup_stale_actors()` kills dead actors and registers alive detached actors with `ResourcePool`.

- `tinker_server/routes/*`
  - HTTP endpoints. Most heavy work is delegated to backend modules.
  - `sampling.py`: async sampling + backpressure; uses `SessionManager`.
  - `training.py`: training control plane; uses `TrainingSessionManager` + training engine.
  - `weights.py`: save/load weights and checkpoints; bridges training to inference.
  - `futures.py`: `request_id` polling.

- `tinker_server/models/types.py`
  - Pydantic request/response models intended to match the Tinker API.

- `tinker_server/backend/resource_pool.py`
  - Global GPU accounting and LRU eviction across actor types (training + inference).
  - Core design assumption: clients do not explicitly end sessions; the pool uses idle timeouts to evict actors.

- `tinker_server/backend/session_manager.py`
  - Sampling session bookkeeping and cleanup.
  - Supports legacy per-session engines and a shared-engine mode for faster ephemeral weight sync.

- `tinker_server/backend/multi_lora_engine.py`
  - Multi-tenant inference: one detached vLLM actor per base model, with many LoRA adapters loaded and selected by `lora_int_id`.
  - Selects between single-node vLLM and `MultiNodeInferenceEngine` (Ray distributed vLLM) based on model config.

- `tinker_server/backend/verl_inference.py`
  - vLLM-backed inference actor implementation and weight loading.

- `tinker_server/backend/verl_training.py`
  - Dense LoRA training backend via Ray actors.

- `tinker_server/backend/megatron_training.py`, `tinker_server/backend/megatron_distributed.py`
  - MoE LoRA training backend via Megatron (distributed actors + placement groups).

- `tinker_server/backend/model_registry.py`
  - Supported model allowlist and per-model parallelism/memory knobs (inference and training are specified separately).

- `tinker_server/config.py`
  - Server config + Ray runtime `PYTHONPATH` for actors (PFS package paths).
