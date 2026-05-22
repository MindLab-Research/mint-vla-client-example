# Identifiers and state ownership

This project has multiple identifiers that look similar but have different ownership and persistence characteristics.

## Identifiers

- `session_id`
  - Created by `POST /api/v1/create_session`.
  - Live request metadata is cached in server memory (`mint_server/routes/service.py:sessions`).
  - Minimal index metadata is persisted through `TaskStateStore` session/index methods for REST reads after API restart.
  - Used mainly for grouping/metadata; it does not own model weights.

- `model_id`
  - Created by `POST /api/v1/create_model` (training routes).
  - Tracks a live training session in server memory (`mint_server/backend/training_session_manager.py`).
  - Minimal recovery metadata is persisted through `TaskStateStore` training-session methods.
  - The actual trainable weights/optimizer live in Ray actors (backend-dependent).
  - Automatically cleaned up after idle timeout (`MINT_TRAINING_INACTIVITY_TIMEOUT`, default 3600s).

- `sampling_session_id`
  - Created by `POST /api/v1/create_sampling_session`.
  - Used by sampling endpoints to select a base model and optional LoRA adapter.
  - In multi-LoRA mode, `sampling_session_id` is mapped (in-process) to a `lora_int_id` that vLLM uses to select frozen adapter weights.
  - In gateway mode, upstream routing metadata for remote sampling sessions is persisted through `TaskStateStore` gateway-session methods.

- `request_id`
  - Created through the `TaskFutureService` facade and returned by endpoints that run async work in the background.
  - Polled via `POST /api/v1/retrieve_future` (Mint polling protocol).
  - Stored in detached `TaskStateStore`, not in API-process memory. Terminal replay also uses this same task record plus `TaskPayloadStore`; there is no second future index.

## Persistence and restart behavior

State that survives an API restart in detached control-plane actors:

- `TaskStateStore` entries (`request_id` status/result metadata and terminal payload pointers)
- session and sampler index metadata
- training-session recovery metadata
- gateway routing metadata for remote sampling sessions and training models
- billing outbox rows awaiting PG flush
- session heartbeat metadata used by cleanup and REST metadata checks

`TaskStateStore` owns these through two local persistence components inside
the same detached actor:

- `FutureStateStore`: RocksDB-backed future/task KV keyed by `request_id`.
- `TaskHotKVStore`: RocksDB-backed hot metadata KV for billing outbox,
  sampling/training/gateway sessions, session/sampler indexes, and session
  heartbeats.

Neither component is a separate Ray actor. High-frequency point mutations use
striped per-key locks in the helper, while the detached `TaskStateStore` actor
is configured with bounded Ray concurrency so independent keys are not forced
through one Python mutex. Billing outbox claims and stats use explicit KV
status/event indexes rather than SQLite scans or process-local index sets.
SQLite remains available for legacy/non-hot task tables, but new future and hot
metadata writes must not depend on SQLite table scans or a process-local dict.

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
- `request_id` exists because GPU work is asynchronous and is polled to match the Mint client contract.
