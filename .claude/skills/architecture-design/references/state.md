# Identifiers and state ownership

This project has multiple identifiers that look similar but have different ownership and persistence characteristics.

## Identifiers

- `session_id`
  - Created by `POST /api/v1/create_session`.
  - Live request metadata is cached in server memory (`tinker_server/routes/service.py:sessions`).
  - Minimal index metadata is also mirrored into the detached session-index store for REST reads after API restart.
  - Used mainly for grouping/metadata; it does not own model weights.

- `model_id`
  - Created by `POST /api/v1/create_model` (training routes).
  - Tracks a live training session in server memory (`tinker_server/backend/training_session_manager.py`).
  - Minimal recovery metadata is mirrored into the detached training-session store.
  - The actual trainable weights/optimizer live in Ray actors (backend-dependent).
  - Automatically cleaned up after idle timeout (`MINT_TRAINING_INACTIVITY_TIMEOUT`, default 3600s).

- `sampling_session_id`
  - Created by `POST /api/v1/create_sampling_session`.
  - Used by sampling endpoints to select a base model and optional LoRA adapter.
  - In multi-LoRA mode, `sampling_session_id` is mapped (in-process) to a `lora_int_id` that vLLM uses to select frozen adapter weights.
  - In gateway mode, upstream routing metadata for remote sampling sessions is mirrored into the detached gateway-session store.

- `request_id`
  - Created by `FutureStore.create()` and returned by endpoints that run async work in the background.
  - Polled via `POST /api/v1/retrieve_future` (Tinker polling protocol).
  - Stored in detached `FutureStore`, not in API-process memory.

## Persistence and restart behavior

State that survives an API restart in detached control-plane actors:

- `FutureStore` entries (`request_id` status/result metadata)
- session and sampler index metadata
- training-session recovery metadata
- gateway routing metadata for remote sampling sessions and training models

State that is still in-process and lost on API restart:

- `sessions` dict request metadata
- `SessionManager` sampling-session bindings
- `sampling_session_id -> lora_int_id` mappings
- `TrainingSessionManager` live session objects
- engine registries and other API-host caches

Many Ray actors are detached and can survive server restarts. This is useful for reuse, but it also means you must reason about reconciliation between persisted control-plane metadata, in-process registries, and actors still running.

## Semantics recap

- `model_id` refers to mutable training state owned by trainer actors (weights, optimizer, step counters).
- `sampling_session_id` refers to frozen adapter state in inference (a LoRA loaded into a shared vLLM actor).
- `request_id` exists because GPU work is asynchronous and is polled to match the Tinker client contract.
