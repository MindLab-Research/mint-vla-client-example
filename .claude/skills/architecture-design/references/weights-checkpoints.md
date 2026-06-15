# Weights and checkpoints

Two recurring constraints drive this code:

1. API server and GPU actors can have different local filesystems.
2. Some models produce adapter state that is too large/too fragmented to transfer as Python objects efficiently.

## Storage tiers

Async future result payloads and checkpoint/weight artifacts are separate
storage systems with separate reapers and metrics.

Async future results:

- Hot cache: per-API-process retrieve cache, default 300s.
- Durable metadata: `TaskStateStore` rows, including status, terminal metadata,
  active staged payload path, abandoned staged payload paths for GC attribution,
  terminal result path, checksum, and size.
- Durable payload: JSON files on vePFS written through the in-process
  `TaskPayloadStore` helper.

Checkpoint and weight artifacts:

- Metadata/catalog: PostgreSQL checkpoint index.
- Ephemeral cache: vePFS runtime checkpoint cache for temporary sampler or
  checkpoint materialization, default TTL 24h.
- Persistent upload cache: vePFS `persistent_cache` used as TOS mirror
  staging/cache. It is eligible for cleanup only after mirror completion and
  after its TTL.
- Persistent store: TOS checkpoint directory, the authoritative artifact store.

The future result payload reaper must not delete checkpoint artifacts. The
checkpoint reaper must not delete `TaskPayloadStore` JSON payloads.

`/api/v1/internal/healthz` does not scan PostgreSQL checkpoint rows against TOS
artifact existence. Checkpoint catalog/artifact consistency belongs to
checkpoint-specific validation or repair jobs, not lightweight health.

Checkpoint catalog publication is idempotent after artifact validation:

- save routes first claim a `checkpoint_staging` row, write artifacts to
  persistent cache, validate the checkpoint shape, then start the TOS mirror.
- if the same owner/model/checkpoint/type already has a failed staging row and
  no catalog row, a new save reuses that staging `ckpt_id` and resets it to
  `uploading`. A failed local publication attempt must not force users to pick
  a new checkpoint name.
- the mirror worker validates the mirrored TOS directory before publishing the
  catalog row.
- once mirrored artifacts are validated, publication may move either an
  `uploading` or `failed` staging row into `checkpoint_catalog`. This recovers
  races where a long save or mirror was marked failed even though the payload
  later completed successfully.
- missing staging rows are not recovered implicitly. If no staging row and no
  catalog row exist for the `ckpt_id`, publication fails and the checkpoint
  remains failed for operator inspection.
- successful catalog publication removes the staging row and writes one
  catalog row. Retrying publication for an already-published `ckpt_id` returns
  the existing catalog row.

## File formats and semantics

This repo uses "HuggingFace/PEFT LoRA adapter format" as the interchange format for inference:
- `adapter_model.safetensors` containing LoRA matrices (separate lora_A/lora_B weights, not merged into base weights)
- `adapter_config.json` describing LoRA rank and target modules

vLLM multi-LoRA serving expects this format.

Patterns used:
- Tensor transfer via Ray object store: simplest, but can become a bottleneck for MoE adapters with many tensors.
- Path-based loading on shared filesystem: used to avoid serializing 10k+ tensors through Ray.

Key knobs and locations:
- `MINT_CHECKPOINT_DIR` controls where server-side code expects checkpoints/adapters for `file://` and `mint://` URIs (see `mint_server/routes/service.py` and `mint_server/backend/sessions/session_manager.py`).
- External `tinker://` request payloads are accepted only at the API compatibility boundary and rewritten to `mint://` before route handlers run.

## Resume metadata lookup

The Tinker SDK `create_training_client_from_state(...)` and `create_training_client_from_state_with_optimizer(...)` first call `POST /api/v1/weights_info`. Mint implements that endpoint by resolving the checkpoint path, validating that it is a training checkpoint, then reading:
- base model from `metadata.json` `model_name`, or `adapter_config.json` `base_model_name_or_path`
- LoRA rank from `adapter_config.json` `r`
- training target flags from `adapter_config.json` `target_modules`

Sampler checkpoints are rejected for this endpoint because it recreates a training client, not a sampling client.

## Megatron session authority

Megatron session state has an explicit authority model. Code should not infer truth independently from the sidecars.

The authority record answers four questions:
- weights source: checkpoint path plus identity, or the internal session cache path plus identity
- optimizer source: checkpoint, live actor, actor snapshot manifest, or none
- gradient source: live actor, actor snapshot manifest, or none
- scheduler source: live actor, actor snapshot manifest, or none

The current storage layout remains:
- `{session_id}_checkpoint/`
- `{session_id}_checkpoint.session_metadata.json`
- `{session_id}_checkpoint.actor_only_state.json`
- `{session_id}_external_checkpoint.json`
- `{session_id}_checkpoint/actor_only_state_manifest.json`

`actor_only_state.json` stores independent source fields for `weights`, `optimizer`, `gradient`, and `scheduler`. Do not collapse those fields back into a single dirty reason. A later `forward_backward` can make gradients actor-local without changing optimizer authority; a later `optim_step` makes weights, optimizer, and scheduler actor-local and consumes gradients.

Actor snapshot manifests also store independent source fields. If a rank snapshot contains optimizer state but only a "consumed gradients" sentinel, the manifest records `optimizer=actor_snapshot`, `gradient=none`, and `scheduler=none` unless scheduler state was actually serialized.

`MegatronSessionStateManager.get_authority_record(...)` is the single place that interprets those files. Cache recycling and future session-state checks should consume that record instead of re-reading sidecars with separate precedence rules.

Cache recycling can delete an internal session cache only when the external checkpoint path still exists and contains optimizer state. If the external checkpoint disappears, the internal cache becomes the only known copy and must not be treated as cold-safe.

The external checkpoint must also match the `checkpoint_identity` recorded when the session cache was primed. This identity is a byte-content digest over checkpoint state files; route ownership metadata such as `metadata.json` is excluded. A reused path with different checkpoint contents is a different checkpoint, even if it still contains optimizer shards.

An external checkpoint marker without session metadata and checkpoint identity is not cold-safe. Treat the internal session cache as the only known copy until a validated metadata record proves an external checkpoint has the same bytes.

After `load_state(..., optimizer=True)`, the session cache is primed from the loaded checkpoint, but optimizer authority remains in the live actor while `actor_only_state.json` exists. Gradient and scheduler authority remain `none` unless a later operation or actor snapshot actually creates them. The marker prevents the cache from being treated as a cold durable checkpoint.

After `load_state(..., optimizer=False)`, `training_meta.json` is optional. If the file exists, `current_step` and `learning_rate` are validated strictly and used. If it is absent, step resets to `0`, the session learning rate is preserved, and optimizer, gradient, and scheduler authority are `none`.

The loaded checkpoint's LoRA rank and train-target flags become the session's active LoRA configuration. The `/load_state` route must persist those metadata-derived flags back through `TaskStateStore` training-session methods before resolving the future; otherwise an API restart can restore stale create-time defaults. Later Megatron operations must use those metadata-derived flags instead of the stale create-time request defaults.

If `/load_state` mutates the live actor successfully but the durable training-session metadata fails to persist afterward, the load future reports success with `metadata_persisted=false` and the error string. Reporting a failed load after actor state changed is a split-brain signal unless the implementation also rolls the actor back.

The same persistence rule applies to `/create_model_from_state`: after checkpoint load, durable training-session metadata must use the session's post-load LoRA configuration, not the raw request payload.

After `optim_step`, weights as well as optimizer state are actor-local until a later session switch or save writes them to a checkpoint/cache. The authority record should represent those live actor weights directly.

## Filesystem visibility contract

`file://` and `mint://` URIs are resolved to filesystem paths in the API server process.

For path-based loading (vLLM `add_lora_from_path`, Megatron adapter load), the resolved directory must be visible to the Ray worker process on its node. If the path is only present on the API server node, the only viable transfer path is to materialize tensors into memory and send them via Ray object store.

## Megatron to HuggingFace (PEFT) conversion for MoE training

MoE training runs in Megatron, but inference consumes PEFT-style adapter checkpoints. Converting is non-trivial because:
- LoRA parameters can be sharded across tensor-parallel (TP) ranks.
- Expert LoRA parameters can be sharded across expert-parallel (EP) ranks.
- Some "export weights" APIs merge LoRA into base weights, which is unusable for vLLM multi-LoRA.

Implementation locations:
- Extraction and conversion logic: `mint_server/backend/training/megatron/megatron_distributed.py` (MegatronRankWorker.get_lora_state_dict)
- Writing adapter checkpoint files: `mint_server/backend/training/megatron/megatron_distributed.py` (MegatronRankWorker.save_checkpoint)

Preferred path (default):
- Use Megatron-Bridge's newer adapter export API when available:
  - `export_adapter_weights(...)` returns adapter weights without merging into the base model.
  - The default Ray worker `PYTHONPATH` prefers a newer Megatron-Bridge variant (see `mint_server/config.py`), so this API is typically present.

Fallback paths:
- Custom extraction from `named_parameters()` plus explicit gathering:
  - TP gather to reconstruct full tensors from TP shards.
  - EP gather for routed experts so the exported adapter includes all experts.
- As a last resort, fallback to Megatron-Bridge `export_hf_weights()` and filter adapter params.
  - `export_weights()` is not acceptable because it merges LoRA into base weights.

Output:
- The conversion produces PEFT-style parameter names and saves `adapter_model.safetensors` and `adapter_config.json` for downstream vLLM loading.

## MoE per-expert LoRA export for inference

For MoE sessions that train MLP LoRA, Mint can export adapters in a per-expert format that vLLM expects.

`POST /api/v1/save_weights_for_sampler` behavior (`mint_server/routes/training.py`):
- If `session.backend == "megatron"` and the request does not explicitly set `use_per_expert_lora`, the server defaults `use_per_expert_lora=True` when the session's LoRA config indicates MLP LoRA training.
