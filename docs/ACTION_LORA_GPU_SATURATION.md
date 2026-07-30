# Action-LoRA GPU supply-path optimization

## Recommended production defaults

Pass these settings explicitly; the CLI keeps conservative legacy defaults for
backward compatibility.

| Server GPUs | Batch | Producers | Prefetch | Total build workers | Datum cache | Row cache |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 128 | **2** | **2** | 16 | 256 | Complete selected population, preloaded |
| 8 | 128 | **8** | **8** | 16 | 256 | Complete selected population, preloaded |

For the full Cylinder1 population (`rows 3514–4552`, 1,039 trajectories), use:

```text
--batch-size 128
--batch-build-workers 16
--datum-cache-size 256
--row-cache-size 1039
--preload-selected-rows
```

Add `--batch-producers 2 --prefetch-batches 2` for four GPUs, or
`--batch-producers 8 --prefetch-batches 8` for eight GPUs. Keep
`--slate-size 16 --coverage-anchors-per-row 8` in both cases.

Set `--row-cache-size` to the number of selected trajectory rows, not to the GPU
count. Full-population residency is what makes cross-slate prefetch valid. If the
population cannot fit host RAM, use the low-memory fallback
`batch64 / producers2 / prefetch2 / workers16` without cross-slate prefetch.

## Eight-GPU population-resident result

The eight-GPU bottleneck was full-trajectory Lance I/O, not GPU communication.
A canonical Coverage batch128 advances through 16 new rows per step; each row
contains an entire image/wrist trajectory and costs about 6.05 seconds to load.
The external prototype appeared faster because each producer reused a 16-row
slate for 250 sampling calls, which changes the row-coverage schedule.

The formal path now ports the reusable mechanism—row residency—without porting
that sampling bias:

```text
--batch-size 128
--batch-producers 8
--batch-build-workers 16
--prefetch-batches 8
--row-cache-size 1039
--preload-selected-rows
```

On Cylinder1, preloading all 1,039 selected rows took 433.15 seconds and raised
client RSS to 232.85 GiB. Warm throughput reached **60.19 samples/s**, versus
16.77 samples/s when only the P8/prefetch8 numbers were copied. Mean batch wait
fell from 4.238 seconds to 1.076 milliseconds; warm SM rose from 14.58% to
52.22%. The 100-step Coverage schedule hash and all StateAug diagnostics match
the previous formal run exactly. The preload breaks even after about 79 steps,
so this is the recommended eight-GPU setting only when the selected population
fits host memory and the run is long enough to amortize startup.

Evidence is in
`results/benchmarks/eight_gpu_efficiency_port_20260730/`.

## Four-GPU population-resident result

A simultaneous four-GPU A/B test used the same model, batch128, 128-row
resident population, norm, Coverage schedule, and StateAug draws:

| Four-GPU supply setting | Samples/s | Mean batch wait | Mean server route | Warm SM |
|---|---:|---:|---:|---:|
| P8 / prefetch8 / workers16 | 42.35 | 1.530 ms | 2.913 s | 71.56% |
| **P2 / prefetch2 / workers16** | **41.49** | **1.415 ms** | 2.972 s | 70.74% |

P2 retains 97.96% of P8 throughput and does not starve the GPUs. The small
throughput difference is explained by the two GPU groups' server-route times,
not by batch supply. Four-GPU production therefore uses P2/prefetch2; P8 is
valid but over-provisions the host queue. Evidence is in
`results/benchmarks/four_gpu_resident_param_ab_20260730/summary.json`.

## Historical four-GPU bounded-cache result

Before full-population residency, the formal MANO Action-LoRA path was limited
by two coupled mechanisms:

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

Batch 128 was only 1.6% faster than batch 64 in this historical bounded-cache
test. Batch64/P2/prefetch2 remains the low-memory fallback because it stays
within one 128-sample Coverage slate. The population-resident defaults above
supersede it on hosts that can retain the selected rows.

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

Artifacts are under the formal client root at
`results/benchmarks/action_lora_gpu_saturation_20260729/`. Key development runs
are `baseline_bs8_w4_p2`, `final_bs32_p4_w16_p4`, `bs64_p2_w16_p2`, and
`bs128_p1_w16_p1`. The authoritative post-merge reproduction is
`postmerge_bs64_p2_w16_p2_20_retry1`; `summary.json` preserves the provenance
boundary between its committed source and the earlier tuning worktrees.

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

A standard coverage slate contains `16 rows × 8 anchors = 128 samples`. With a
bounded slate-sized row cache, parallel prefetch must satisfy:

```text
batch_size * prefetch_batches <= 128
```

Batch 64 with prefetch 4 violated this invariant. Four materializers spanned two
slates while the row LRU held one, causing repeated eviction/reload: p95 wait was
31.0 s and throughput collapsed to 7.50 samples/s. Reducing prefetch to 2 cut
p95 wait to 3.50 s.

Cross-slate prefetch is accepted only after `--preload-selected-rows` has filled
an explicit row cache large enough for the complete selected population. This
state removes eviction by construction; the result records
`prefetch_contract.status=population_resident`. The client still rejects every
cross-slate setting backed by a partial cache.

## Repository impact

- **Client:** owns the concurrent row cache, optional full-population preload,
  deterministic planner/materializer, timing metrics, probe-only no-save switch,
  and documented settings.
- **MINT:** only adds observability for device count, DATA_AXIS selection, and
  per-device batch size. Existing batched JIT and `PartitionSpec(DATA_AXIS)` do
  the GPU work.
- **OpenPI:** unchanged. Its existing Action-Expert A-LoRA and batched loss path
  already support these batch sizes.

Defaults remain backward compatible: `--batch-producers 1`. Production runs
must opt into a measured setting and keep checkpoint, norm, release, and
provenance contracts unchanged.
