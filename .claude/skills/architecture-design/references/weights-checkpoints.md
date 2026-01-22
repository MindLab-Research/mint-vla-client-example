# Weights and checkpoints

Two recurring constraints drive this code:

1. API server and GPU actors can have different local filesystems.
2. Some models produce adapter state that is too large/too fragmented to transfer as Python objects efficiently.

## File formats and semantics

This repo uses "HuggingFace/PEFT LoRA adapter format" as the interchange format for inference:
- `adapter_model.safetensors` containing LoRA matrices (separate lora_A/lora_B weights, not merged into base weights)
- `adapter_config.json` describing LoRA rank and target modules

vLLM multi-LoRA serving expects this format.

Patterns used:
- Tensor transfer via Ray object store: simplest, but can become a bottleneck for MoE adapters with many tensors.
- Path-based loading on shared filesystem: used to avoid serializing 10k+ tensors through Ray.

Key knobs and locations:
- `TINKER_CHECKPOINT_DIR` controls where server-side code expects checkpoints/adapters for `file://` and `tinker://` URIs (see `tinker_server/routes/service.py` and `tinker_server/backend/session_manager.py`).

## Filesystem visibility contract

`file://` and `tinker://` URIs are resolved to filesystem paths in the API server process.

For path-based loading (vLLM `add_lora_from_path`, Megatron adapter load), the resolved directory must be visible to the Ray worker process on its node. If the path is only present on the API server node, the only viable transfer path is to materialize tensors into memory and send them via Ray object store.

## Megatron to HuggingFace (PEFT) conversion for MoE training

MoE training runs in Megatron, but inference consumes PEFT-style adapter checkpoints. Converting is non-trivial because:
- LoRA parameters can be sharded across tensor-parallel (TP) ranks.
- Expert LoRA parameters can be sharded across expert-parallel (EP) ranks.
- Some "export weights" APIs merge LoRA into base weights, which is unusable for vLLM multi-LoRA.

Implementation locations:
- Extraction and conversion logic: `tinker_server/backend/megatron_distributed.py` (MegatronRankWorker.get_lora_state_dict)
- Writing adapter checkpoint files: `tinker_server/backend/megatron_distributed.py` (MegatronRankWorker.save_checkpoint)

Preferred path (default):
- Use Megatron-Bridge's newer adapter export API when available:
  - `export_adapter_weights(...)` returns adapter weights without merging into the base model.
  - The default Ray worker `PYTHONPATH` prefers a newer Megatron-Bridge variant (see `tinker_server/config.py`), so this API is typically present.

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

`POST /api/v1/save_weights_for_sampler` behavior (`tinker_server/routes/training.py`):
- If `session.backend == "megatron"` and the request does not explicitly set `use_per_expert_lora`, the server defaults `use_per_expert_lora=True` when the session's LoRA config indicates MLP LoRA training.
