# Inference architecture

## Primary path (multi-LoRA)

1. `POST /api/v1/create_sampling_session` validates access and chooses `base_model`.
2. `SessionManager.get_engine_for_model(base_model)` selects/creates a `MultiLoRAInferenceEngine` via `MultiModelInferenceManager`.
3. `MultiLoRAInferenceEngine.initialize()` connects to an existing detached vLLM actor or creates a new one:
  - `namespace=tinker_server.config.RAY_NAMESPACE` (from `TINKER_RAY_NAMESPACE`) so actors can be rediscovered across server restarts.
  - detached lifetime so actors survive API server restarts.
  - registers the actor in `ResourcePool` for LRU eviction and GPU accounting.
4. If a LoRA adapter is provided, the server loads weights and registers them for the session:
   - Small/medium adapters: tensors transferred via Ray object store (`add_lora_with_id`).
   - Very large adapters (MoE with many tensors): path-based load (`add_lora_from_path`) to avoid Ray serialization overhead.
5. `POST /api/v1/asample` uses `sampling_session_id` (or `model_id`, via SDK aliasing) to pick the right engine and then calls:
   - `generate_with_lora` when `sampling_session_id` resolves to a `lora_int_id`
   - `generate_base` when no LoRA is registered for the session

## Multi-node inference (multi-node TP or MoE TP>=4)

`MultiModelInferenceManager.get_engine()` selects a different engine implementation when vLLM must run via Ray distributed execution:
- `MultiNodeInferenceEngine` (in `tinker_server/backend/multinode_inference.py`) for:
  - `config.total_gpus > 8` (true multi-node TP), or
  - `(config.is_moe and config.total_gpus >= 4)` (route MoE TP>=4 through the same engine even if it fits on one node).
- `MultiLoRAInferenceEngine` otherwise.

Mechanics (MultiNodeInferenceEngine):
- vLLM's Ray backend spawns 1-GPU worker actors plus a CPU-only controller actor.
- Mint creates:
  - a detached placement group with `total_required_gpus = worker_gpus` GPU bundles plus one CPU-only controller bundle (strategy `PACK`)
  - a detached controller actor with `num_gpus=0`, pinned to the controller bundle index
  - child vLLM workers captured into the same placement group (`placement_group_capture_child_tasks=True`)
- `ResourcePool` accounts for the full `total_required_gpus` so eviction decisions reflect the real cluster footprint.

## Why multi-LoRA is central

The Tinker contract expects that sampling sessions can hold "frozen weights" for inference. Multi-LoRA implements this by loading many LoRA adapters into a shared vLLM actor while keeping the base model fixed. The alternative (one vLLM engine per session) makes session creation expensive and scales poorly with the number of concurrent LoRAs.

## Non-obvious boundary: session mapping vs detached actors

- The mapping `sampling_session_id` to `lora_int_id` is stored in-process (`LoRARegistry`).
- A new server process can reconnect to a detached vLLM actor but will not automatically reconstruct `LoRARegistry`.
- After a server restart, existing detached vLLM actors may still have LoRAs loaded, but the API server will not know which `lora_int_id` corresponds to a previous `sampling_session_id` unless you design a reconciliation protocol.
