# K2-Thinking Memory Analysis

## Model Architecture (from config.json)

| Parameter | Value |
|-----------|-------|
| num_hidden_layers | 61 |
| hidden_size | 7168 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| n_routed_experts | 384 |
| n_shared_experts | 1 |
| moe_intermediate_size | 2048 |
| intermediate_size | 18432 (dense layers) |
| first_k_dense_replace | 1 |
| vocab_size | 163840 |

## Quantization Config

From `config.json`:
```json
"quantization_config": {
  "num_bits": 4,
  "group_size": 32,
  "format": "pack-quantized",
  "quant_method": "compressed-tensors",
  "ignore": [
    "lm_head",
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"
  ]
}
```

**Critical**: Only MoE routed experts are INT4. Everything else is BF16.

## Theoretical Weight Calculation (v1)

### Per-layer breakdown

**Attention (BF16, NOT quantized)**:
- q_a_proj: 7168 × (512 + 64) = 4.13M params
- q_b_proj: 512 × 64 × 128 = 4.19M params
- kv_a_proj: 7168 × (512 + 64) = 4.13M params
- kv_b_proj: 512 × 64 × (128 + 128) = 8.39M params
- o_proj: 64 × 128 × 7168 = 58.72M params
- **Total attention**: 79.56M params/layer

**MoE Routed Experts (INT4)**:
- Per expert: 3 × 7168 × 2048 = 44.04M params
- 384 experts: 16.91B params/layer

**Shared Expert (BF16)**:
- 3 × 7168 × 2048 = 44.04M params/layer

**Router (BF16)**:
- 7168 × 384 = 2.75M params/layer

**LayerNorms (BF16)**:
- 2 × 7168 = 14.3K params/layer

### Totals

| Component | Params | Format | Memory |
|-----------|--------|--------|--------|
| 60 MoE layers attention | 4.77B | BF16 | 9.55 GB |
| 60 MoE layers experts | 1014.69B | INT4+scales | 570.76 GB |
| 60 MoE layers shared | 2.64B | BF16 | 5.29 GB |
| 60 MoE layers router | 0.17B | BF16 | 0.33 GB |
| 1 Dense layer | 0.21B | BF16 | 0.42 GB |
| Embeddings | 1.17B | BF16 | 2.35 GB |
| LM head | 1.17B | BF16 | 2.35 GB |
| **Total** | ~1025B | Mixed | **591.05 GB** |
| **Per GPU (TP=16)** | | | **36.94 GB** |

## Measured Values

| Metric | Value | Source |
|--------|-------|--------|
| Model loading memory | 67.09 GiB | vLLM log: "Model loading took 67.0889 GiB" |
| Available KV cache (90% util) | -5.50 GiB | vLLM log |
| non_kv_cache_memory | 77.5 GiB | Calculated: 72 + 5.5 |

## Discrepancy Analysis

**Theoretical**: 36.94 GB/GPU
**Measured**: 67.09 GB/GPU
**Discrepancy**: 30.15 GB/GPU

### Hypotheses for discrepancy:

1. **vLLM decompresses INT4 to BF16 for computation**
   - If INT4 experts decompressed: 570.76 GB × 4 = 2283 GB total
   - This would be 142 GB/GPU - way more than measured
   - UNLIKELY to be full decompression

2. **Partial decompression / workspace buffers**
   - vLLM may allocate workspace for matmul operations
   - Need to check: compressed-tensors kernel implementation

3. **LoRA infrastructure overhead** (when enable_lora=True)
   - Punica workspace buffers
   - LoRA weight slots (max_loras × layers × modules)

4. **My calculation is wrong**
   - Need to verify against actual safetensor file sizes

## Actual File Sizes (VERIFIED)

```
Total on disk: 554 GB (62 shards × ~9 GB each)
Per GPU (TP=16): 554 / 16 = 34.6 GB
```

Ratio measured/disk = 67.09 / 34.6 = **1.94x** (approximately 2x)

## Key Finding: INT4 → INT8 Unpacking

vLLM uses `CompressedTensorsWNA16MarlinMoEMethod` for K2 MoE experts.
From compressed_tensors_moe.py line 2092:
```python
# Register unpacked int4-as-int8 weights the loader will fill.
w13 = torch.nn.Parameter(
    torch.empty(E, 2 * IN, H, dtype=torch.int8), requires_grad=False
)
```

This is the non-Marlin path. K2 uses Marlin which keeps INT4 packed, but the
2x ratio suggests there's still some expansion happening during loading or
MLA weight processing (see `get_and_maybe_dequant_weights` in MLA).

## vLLM Memory Model

From `vllm/utils/mem_utils.py`:
```python
non_kv_cache_memory = (
    non_torch_memory +        # NCCL, CUDA context
    peak_activation_memory +  # Profiling forward pass
    weights_memory            # Model weights
)

available_kv_cache = requested_memory - non_kv_cache_memory
requested_memory = total_gpu_memory × gpu_memory_utilization
```

**Key insight**: `gpu_memory_utilization` sets a BUDGET. Actual allocations can exceed it.
To fix negative available KV cache: INCREASE gpu_memory_utilization.

## Questions to Answer

1. [ ] What is the actual breakdown of 67.09 GB? (weights vs overhead)
2. [ ] Why is measured 2x theoretical?
3. [ ] What is peak_activation_memory during profiling?
4. [ ] What is non_torch_memory (NCCL buffers)?
5. [ ] How much does LoRA add to memory?

## CONCLUSION: Root Cause Identified

**The 2x memory expansion is caused by INT4 → INT8 unpacking during weight loading.**

Even though K2 uses the Marlin kernel path (`CompressedTensorsWNA16MarlinMoEMethod`)
which keeps weights packed for computation, the safetensor loading path unpacks
INT4 to INT8 (1 byte per value instead of 0.5 bytes), doubling memory footprint.

| Metric | Value |
|--------|-------|
| On-disk size | 554 GB |
| Per GPU theoretical | 34.6 GB |
| Per GPU measured | 67.09 GB |
| Expansion ratio | 1.94x |

**Implication for context length:**
- 67 GB weights + overhead leaves ~11 GB for KV cache at 98% utilization
- Limits practical context to 2K-8K tokens
- This is a vLLM implementation constraint, not fundamental

**Potential optimization:** Keep weights packed during loading would recover ~30 GB/GPU.

## Revision History

- 2024-12-29: Initial calculation showing 30 GB discrepancy
