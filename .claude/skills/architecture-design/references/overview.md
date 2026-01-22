# Mint architecture overview

Mint (tinker-server) is a FastAPI service that implements the Tinker REST contract and brokers training and inference to Ray GPU actors. The server is a control plane plus request validation, not a compute engine. The design is shaped by two goals: keep compatibility with the Tinker API contract, and multiplex GPU resources across many LoRA sessions for both inference and training.

## Tinker API contract as the boundary

The API surface follows the Tinker SDK expectations: create sessions and models, submit work, poll for completion, and manage weights. The server owns HTTP, auth, and request validation. The GPU actors own model weights and do the compute. That split is the primary boundary.

Contract implications:
- Long-running work returns a `request_id` and is polled via `/api/v1/retrieve_future` (408 pending). This is the Tinker async protocol, not a server-specific choice.
- `session_id`, `model_id`, and `sampling_session_id` are identifiers tracked in server memory. They are not durable storage for weights or optimizer state.
- The server can restart while detached Ray actors remain alive, but in-process mappings are lost. This reduces server restart cost and increases reconciliation requirements.

## Multi-LoRA inference: one base model, many adapters

Inference uses a MultiLoRA design: one detached vLLM actor per base model, many LoRA adapters loaded into that actor, and requests select the adapter by `lora_int_id`.

Flow:
- `POST /api/v1/create_sampling_session` validates access and selects a base model.
- The server attaches to an existing detached vLLM actor or creates one.
- Adapter weights are loaded on demand, then requests call `generate_with_lora` or `generate_base`.

Boundaries and tradeoffs:
- The mapping `sampling_session_id` to `lora_int_id` is in server memory. After a server restart, the vLLM actor may still have LoRAs loaded, but the server no longer knows which session maps to which adapter without additional reconciliation.
- Small and medium adapters go through the Ray object store. Very large MoE adapters use path-based loading on a shared filesystem to avoid serializing thousands of tensors.
- Detached actors reduce warmup cost for repeated use but keep holding GPU memory until evicted.
- Inference engine selection is per-model. Models that require Ray-distributed vLLM execution use `MultiNodeInferenceEngine` and reserve an additional "controller GPU" beyond `tensor_parallel_size` (see `inference.md` and `placement-groups.md`).

## Multi-tenant training: time-sliced state swap

Training supports many concurrent sessions on fewer GPU trainers by swapping per-session state into the active trainer, while keeping session isolation of:
- LoRA weights
- accumulated gradients (for gradient accumulation across calls)
- optimizer state (momentum/variance)

Two distinct backends implement this:

1. Dense models: pooled detached TrainingWorker per base model, many sessions per actor
- `DenseTrainerPool` reuses a detached `TrainingWorker` keyed by `base_model` and configured with a `max_lora_rank`.
- Each call passes `session_id`, and the actor swaps LoRA weights, optimizer state, and gradients via disk-backed state on the trainer node.

2. MoE models: one MegatronWorkerGroup, many sessions
- A worker group owns a placement group and N rank workers (see `placement-groups.md`).
- On session switch, each rank swaps optimizer and gradient state in memory, while LoRA weights are saved and loaded from a shared filesystem path.

Boundaries and tradeoffs:
- The swap mechanism isolates sessions for time-slicing, but it is not a durable resume system. If an actor dies, in-memory optimizer and gradient state is lost.
- Dense swap state is stored on the trainer node (default `/tmp`). If the actor moves to another node, that state does not follow.
- MoE LoRA weights persist on shared storage, but optimizer state does not. After restart, a session resumes with a fresh optimizer unless restored externally.

## Weight formats and transfer constraints

Inference consumes LoRA adapters in PEFT format:
- `adapter_model.safetensors`
- `adapter_config.json`

This is a hard constraint. vLLM multi-LoRA expects adapter matrices separate from base weights. Any export path that merges LoRA into the base model is unusable for multi-LoRA inference.

Two transfer mechanisms exist because of size and serialization limits:
- Ray object store for smaller adapters.
- Path-based loading on PFS for large or highly sharded adapters (MoE).

MoE training uses Megatron and must export PEFT adapters by reconstructing full tensors across TP and EP sharding. The preferred path is a newer Megatron-Bridge adapter export API that returns adapter weights without merging.

## ResourcePool and eviction

ResourcePool owns global GPU accounting and eviction. Clients do not explicitly end sessions, so idle timeouts control eviction. This affects both training and inference:
- Detached inference actors can remain alive across server restarts and keep CUDA memory until evicted.
- Training actors can be evicted if idle, which can discard in-memory session state.

Eviction is a resource policy. It is not a fault-tolerance mechanism.

Mint can optionally pre-create and protect ("never evict") a set of persistent actors at startup (controlled by `MINT_PERSISTENT_MODELS`). This is a capacity planning knob, not a correctness requirement.

## Auth and access boundaries

Auth is enforced by middleware when keys are configured:
- Admin key (`TINKER_API_KEY`) for privileged access.
- Encrypted user tokens (`TINKER_TOKEN_SECRET_KEY`) for per-user identity.

Model access control is centralized and applied on session creation. The server hides detailed exception text unless the request is privileged.

## Non-goals

Mint does not aim to:
- persist session mappings across server restarts without explicit reconciliation
- migrate training state across GPU nodes automatically
- store full training state in the FastAPI process
- support LoRA formats that merge adapter weights into base weights
