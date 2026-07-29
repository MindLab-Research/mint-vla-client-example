# Action-LoRA GPU supply-path optimization

## Result

The formal MANO Action-LoRA path was limited by two coupled mechanisms:

1. batch 8 did too little work per server step;
2. the client serialized coverage-slate row loads behind one row-cache miss lock,
   so its prefetch queue periodically emptied for 4–8 seconds.

The optimized path raises warm throughput from **9.01 to 23.37 samples/s
(2.59×)** on GPUs 0–3 while preserving the MANO training contract: cube1+cube2
contact windows, B actions, gesture prompts, state32, StateAug 0.05, locked norm,
CoverageSampler, LR 5e-5, and ordered optimizer batches.

The throughput point is:

```text
--batch-size 128
--batch-producers 1
--batch-build-workers 16
--prefetch-batches 1
```

The balanced point is:

```text
--batch-size 64
--batch-producers 2
--batch-build-workers 16
--prefetch-batches 2
```

Batch 128 was only 1.6% faster than batch 64. Batch 64 has lower request latency
(2.24 s versus 4.24 s), slightly lower host RSS, and uses two concurrent batch
materializers, so it is the recommended long-run setting unless maximum sample
throughput is the sole objective.

## Measurements

Warm statistics exclude the first cold-compile step. Every row below used the
same 1,997 trajectories, normalization SHA
`4f91eca8ee91d53426ea07faf28873ab98c3761ecb84d6374f4c0c439d51069a`,
sample seed 42, augmentation seed 43, and StateAug sigma 0.05.

| Batch / supply setting | Mean batch wait | Mean server request | Samples/s | Max client RSS | DATA_AXIS |
|---|---:|---:|---:|---:|---|
| 8, 1 producer, 4 workers, prefetch 2 (baseline) | 0.420 s | 0.468 s | 9.01 | not sampled | 4 devices inferred from fixed server contract |
| 32, 1 producer, 8 workers, prefetch 8 | 1.100 s | 1.055 s | 14.85 | not sampled | divisible by 4 |
| 32, 4 producers, 16 workers, prefetch 4 | 0.195 s | 1.368 s | 20.47 | 14.14 GiB | 4 devices, 8 samples/device |
| 64, 2 producers, 16 workers, prefetch 2 | 0.546 s | 2.235 s | 23.01 | 13.79 GiB | 4 devices, 16 samples/device |
| 128, 1 producer, 16 workers, prefetch 1 | 1.233 s | 4.243 s | **23.37** | 14.65 GiB | 4 devices, 32 samples/device |
| 128, 1 producer, 32 workers, prefetch 1 | 1.263 s | 4.396 s | 22.62 | 20.41 GiB | 4 devices, 32 samples/device |

The patched server reported finite loss, gradient norm, and parameter norm for
all accepted probes. It explicitly reported `device_count:sum=4` and
`used_data_sharding:mean=1`. Realized StateAug sigma was approximately 0.0499.
GPU SM reached 100% during compute bursts. Full-run mean SM, which includes
client setup and cold compilation, rose from about 8.8% at batch 8 to 24.6% at
the batch-128 point. It is therefore accurate to claim a 2.59× supply/throughput
improvement, not continuous 100% GPU occupancy.

Artifacts are under
`results/benchmarks/action_lora_gpu_saturation_20260729/runs/` in the integration
worktree. Key runs are `baseline_bs8_w4_p2`, `final_bs32_p4_w16_p4`,
`bs64_p2_w16_p2`, and `bs128_p1_w16_p1`.

## Implemented mechanism

### Concurrent row loading

`SelectedLanceDataset` now coalesces duplicate misses per row but releases the
global cache lock while an immutable Lance row is loaded. Distinct rows can be
read concurrently; threads requesting the same row receive one shared result.
This removes the old coverage-slate barrier without duplicating the 1.16-million
frame index or full row payloads.

### Deterministic multi-producer materialization

Multi-producer mode uses one planner and ordered materializers:

- the planner alone advances CoverageSampler, sampling RNG, StateAug RNG, and
  batch ordinal;
- augmentation arrays are drawn before dispatch;
- workers materialize immutable plans using a shared bounded cache;
- completed batches are buffered and sent to MINT strictly by ordinal;
- worker errors propagate and shutdown unblocks the bounded planner.

Tests compare single-thread and multi-producer sample/noise sequences and planned
wire batches exactly. Thread timing cannot reorder optimizer inputs.

### Coverage-slate guard

A standard coverage slate contains `16 rows × 8 anchors = 128 samples`. Parallel
prefetch must satisfy:

```text
batch_size * prefetch_batches <= 128
```

Batch 64 with prefetch 4 violated this invariant. Four materializers spanned two
slates while the row LRU held one, causing repeated eviction/reload: p95 wait was
31.0 s and throughput collapsed to 7.50 samples/s. Reducing prefetch to 2 cut
p95 wait to 3.50 s. The client now rejects cross-slate multi-producer settings
instead of silently entering this failure mode.

## Repository impact

- **Client:** owns the concurrent row cache, deterministic planner/materializer,
  timing metrics, probe-only no-save switch, and documented settings.
- **MINT:** only adds observability for device count, DATA_AXIS selection, and
  per-device batch size. Existing batched JIT and `PartitionSpec(DATA_AXIS)` do
  the GPU work.
- **OpenPI:** unchanged. Its existing Action-Expert A-LoRA and batched loss path
  already support these batch sizes.

Defaults remain backward compatible: `--batch-producers 1`. Production runs
must opt into a measured setting and keep checkpoint, norm, release, and
provenance contracts unchanged.
