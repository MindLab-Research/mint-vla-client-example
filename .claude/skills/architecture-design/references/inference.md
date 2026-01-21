# Inference architecture

## Primary path (multi-LoRA)

1. `POST /api/v1/create_sampling_session` validates access and chooses `base_model`.
2. `SessionManager.get_engine_for_model(base_model)` selects/creates a `MultiLoRAInferenceEngine` via `MultiModelInferenceManager`.
3. `MultiLoRAInferenceEngine.initialize()` connects to an existing detached vLLM actor or creates a new one:
   - `namespace="tinker"` so actors can be rediscovered across server restarts.
   - detached lifetime so actors survive API server restarts.
   - node-affinity scheduling to avoid co-locating with training actors that hold large CUDA allocations.
   - registers the actor in `ResourcePool` for LRU eviction and GPU accounting.
4. If a LoRA adapter is provided, the server loads weights and registers them for the session:
   - Small/medium adapters: tensors transferred via Ray object store (`add_lora_with_id`).
   - Very large adapters (MoE with many tensors): path-based load (`add_lora_from_path`) to avoid Ray serialization overhead.
5. `POST /api/v1/asample` uses `sampling_session_id` (or `model_id`, via SDK aliasing) to pick the right engine and then calls:
   - `generate_with_lora` when `sampling_session_id` resolves to a `lora_int_id`
   - `generate_base` when no LoRA is registered for the session

## Multi-node inference (TP > 8)

`MultiModelInferenceManager.get_engine()` selects a different engine implementation when a model requires more than one 8-GPU node:
- `MultiNodeInferenceEngine` (in `tinker_server/backend/multinode_inference.py`) for `config.total_gpus > 8`.
- `MultiLoRAInferenceEngine` otherwise.

The multi-node engine is still a detached actor, but it relies on vLLM's Ray distributed backend to manage GPU workers, so the actor itself is created with `num_gpus=0`.

## Why multi-LoRA is central

The Tinker contract expects that sampling sessions can hold "frozen weights" for inference. Multi-LoRA implements this by loading many LoRA adapters into a shared vLLM actor while keeping the base model fixed. The alternative (one vLLM engine per session) makes session creation expensive and scales poorly with the number of concurrent LoRAs.

## Non-obvious boundary: session mapping vs detached actors

- The mapping `sampling_session_id` to `lora_int_id` is stored in-process (`LoRARegistry`).
- A new server process can reconnect to a detached vLLM actor but will not automatically reconstruct `LoRARegistry`.
- After a server restart, existing detached vLLM actors may still have LoRAs loaded, but the API server will not know which `lora_int_id` corresponds to a previous `sampling_session_id` unless you design a reconciliation protocol.
