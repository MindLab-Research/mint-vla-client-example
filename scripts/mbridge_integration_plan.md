# Megatron-Bridge Integration Plan

## Goal
Replace custom `get_lora_state_dict()` (~650 lines) with HollowMan fork's `export_adapter_weights()` API.

## Integration Steps

### 1. Update config.py

Add HollowMan fork path (prepend to use instead of current megatron-bridge):

```python
# HollowMan fork with export_adapter_weights API
PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH = "/vePFS-Mindverse/share/code/megatron-bridge-hollowman/src"

# Toggle between implementations
USE_MBRIDGE_LORA_EXPORT = os.environ.get("USE_MBRIDGE_LORA_EXPORT", "false").lower() in ("true", "1", "yes")

# PYTHONPATH: if USE_MBRIDGE_LORA_EXPORT, prepend HollowMan fork
if USE_MBRIDGE_LORA_EXPORT:
    PFS_PYTHONPATH = f"{PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH}:{PFS_VERL_PATH}:{PFS_TINKER_PATH}:{PFS_HF_MODULES_PATH}"
else:
    PFS_PYTHONPATH = f"{PFS_MEGATRON_BRIDGE_PATH}:{PFS_VERL_PATH}:{PFS_TINKER_PATH}:{PFS_HF_MODULES_PATH}"
```

### 2. Add alternative implementation in megatron_distributed.py

Add at the START of `get_lora_state_dict()`:

```python
def get_lora_state_dict(self, use_per_expert_lora: bool = False) -> dict:
    # Try HollowMan Megatron-Bridge export_adapter_weights first
    bridge = getattr(self.engine, 'bridge', None)
    if bridge is not None and hasattr(bridge, 'export_adapter_weights'):
        logger.info(f"[Rank {self.rank}] Using Megatron-Bridge export_adapter_weights API")
        adapter_state = {}

        with self.engine.eval_mode():
            for name, tensor in bridge.export_adapter_weights(self.engine.module, cpu=True, show_progress=False):
                if self.rank == 0:
                    adapter_state[name] = tensor.clone()

        # Non-rank-0 workers return empty dict
        if self.rank != 0:
            logger.info(f"[Rank {self.rank}] get_lora_state_dict: returning empty dict (non-rank-0)")
            return {}

        logger.info(f"[Rank 0] export_adapter_weights returned {len(adapter_state)} params")
        return adapter_state

    # Fall back to custom implementation
    logger.info(f"[Rank {self.rank}] Using custom LoRA extraction (export_adapter_weights not available)")
    # ... rest of existing implementation ...
```

### 3. Server restart required

After code changes:
1. Sync to volcano via unison
2. Restart server: `ssh volcano 'pkill -f run_server && sleep 2 && cd /vePFS-Mindverse/share/code/tinker-server && nohup python -m tinker_server.run_server > /tmp/tinker_server.log 2>&1 &'`
3. Kill existing Megatron actor to pick up new PYTHONPATH

### 4. Environment variable to toggle

```bash
# For experiment 3 and 4: use Megatron-Bridge export
export USE_MBRIDGE_LORA_EXPORT=true

# For experiment 1 and 2: use existing implementation
export USE_MBRIDGE_LORA_EXPORT=false
```

## Comparison Points

| Aspect | Custom Implementation | Megatron-Bridge API |
|--------|----------------------|---------------------|
| Lines of code | ~650 | ~10 |
| TP gathering | Manual `_gather_tensor_across_tp()` | Handled internally |
| EP gathering | Manual `_gather_expert_lora_across_ep()` | `_gather_expert_adapter_weight()` |
| Fused gate/up split | Manual half split | `_get_fused_adapter_linear_out_slices()` |
| Per-expert expansion | Manual `_expand_shared_to_per_expert()` | Handled by streaming |
| Maintenance | We maintain | HollowMan6 maintains |

## Testing

After integration:
1. Run `scripts/test_logprob_after_training.py` to verify logprob alignment
2. Compare KL divergence metrics between implementations
3. Monitor for any weight shape mismatches or export errors

## Expected Results

Both implementations should produce identical LoRA weights in PEFT format.
KL divergence at step 0 should be ~0.4-0.5 (baseline).
No change to training behavior, only to weight export path.
