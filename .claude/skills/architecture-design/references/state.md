# Identifiers and state ownership

This project has multiple identifiers that look similar but have different ownership and persistence characteristics.

## Identifiers

- `session_id`
  - Created by `POST /api/v1/create_session`.
  - Stored in server memory (`tinker_server/routes/service.py:sessions`).
  - Used mainly for grouping/metadata; it does not own model weights.

- `model_id`
  - Created by `POST /api/v1/create_model` (training routes).
  - Tracks a training session in server memory (`tinker_server/backend/training_session_manager.py`).
  - The actual trainable weights/optimizer live in Ray actors (backend-dependent).
  - Automatically cleaned up after idle timeout (`MINT_TRAINING_INACTIVITY_TIMEOUT`, default 3600s).

- `sampling_session_id`
  - Created by `POST /api/v1/create_sampling_session`.
  - Used by sampling endpoints to select a base model and optional LoRA adapter.
  - In multi-LoRA mode, `sampling_session_id` is mapped (in-process) to a `lora_int_id` that vLLM uses to select frozen adapter weights.

- `request_id`
  - Created by `FutureStore.create()` and returned by endpoints that run async work in the background.
  - Polled via `POST /api/v1/retrieve_future` (Tinker polling protocol).

## Persistence and restart behavior

- `FutureStore`, `sessions`, `training sessions`, and `sampling session` mappings are in-process. A server restart loses them.
- Many Ray actors are detached and can survive server restarts. This is useful for reuse, but it also means you must reason about reconciliation between "server memory state" and "actors still running".

## Semantics recap

- `model_id` refers to mutable training state owned by trainer actors (weights, optimizer, step counters).
- `sampling_session_id` refers to frozen adapter state in inference (a LoRA loaded into a shared vLLM actor).
- `request_id` exists because GPU work is asynchronous and is polled to match the Tinker client contract.
