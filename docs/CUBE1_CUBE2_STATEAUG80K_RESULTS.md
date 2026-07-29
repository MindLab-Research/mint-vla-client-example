# cube1+cube2 StateAug A-LoRA 80K results

## Scope

The user stopped the planned 150K run after the complete step80K sampler was saved. Training logs reached step80,307 while processes unwound, but every evaluation in this report loads the immutable step80,000 sampler.

Canonical generated evidence remains under `results/` and is intentionally Git-ignored. The reproducible report is:

```text
results/reports/cube1_cube2_stateaug80k_training_inference_summary_20260729/
```

It contains `summary.json`, `behavior_rows.csv`, `lift_candidates.csv`, a recomputation script, and artifact validation.

## Training contract

| Item | Value |
|---|---|
| Population | cube1 rows507–1608 + cube2 rows1609–2503 |
| Trajectories | 1,997 (cube1=1,102; cube2=895) |
| Frames | 1,743,849 source; 1,160,274 contact-window |
| Model | π0.5 Action-Expert A-LoRA rank16; 13,224,992 trainable parameters |
| State/action | 32D contact/lift state; query-anchored B32D horizon10 |
| Augmentation | normalized `state[0:26]` Gaussian sigma0.05; absolute target preserved |
| Optimization | batch8; constant learning rate5e-5; CoverageSampler |
| Adopted checkpoint | step80,000 sampler |
| Effective consumption | 40.06 coverage epochs; 640,000 anchors; 6,400,000 action labels |
| Norm SHA256 | `4f91eca8ee91d53426ea07faf28873ab98c3761ecb84d6374f4c0c439d51069a` |

Sampler identifier:

```text
mint://lance-smoke-3f744c2fd57c_0/sampler_weights/cube1_cube2_all_32d_stateaug005_alora_b_lr5e5_bs8_150k_20260728_step80000
```

The sampler contains 21 files and 5,302,789,301 bytes. Its metadata SHA256 is `0f33bea4bc8019143a1981378ba4713de930f79bde9d00fc01bad073f5aa8895`.

Final loss windows remained finite and slowly improved:

| Window | Median | Mean | P95 |
|---|---:|---:|---:|
| Last5K | 0.0707 | 0.0896 | 0.2063 |
| Last10K | 0.0711 | 0.0897 | 0.2091 |
| Last20K | 0.0722 | 0.0908 | 0.2088 |
| Last40K | 0.0750 | 0.0939 | 0.2153 |

Flow-matching loss is optimization evidence, not grasp-success evidence.

## Mode4 evaluation contract

The user selected cube1+cube2 recorded-target physics Grade A rows. Grade A means the recorded target replay has a full-trajectory maximum object-translation error below3cm; it is not a policy-success label.

The original plan contained168 Grade A rows. Three complete32-row batches were accepted before the user canceled the remainder. The evaluation boundary is therefore96 unique rows; a partial fourth batch was deleted rather than mixed with synchronized complete-batch evidence.

- Population completed: cube1=40, cube2=56.
- Gestures:01=15,02=13,03=15,04=11,09=15,10=14,11=13.
- Protocol: canonical contact-window, `chunk_stride=1`, `act_batch_size=4`, four independent MuJoCo trajectories in lockstep.
- Artifacts:96 results,288 videos,1,440 finite arrays, zero validation errors.
- Runtime:15,657 batch requests,55,319 real observations,7,309 padding,88.33% slot utilization,76.58 minutes summed wall time.

## Behavior

| Metric | Result |
|---|---:|
| Any physics hand-object contact | 95/96 |
| At least2 simultaneous fingers | 49/96 |
| At least3 simultaneous fingers | 11/96 |
| At least4 simultaneous fingers | 2/96 |
| Peak lift above20mm | 7/96 |
| At least0.5s continuously above20mm | 2/96 |
| Final lift above20mm | 0/96 |
| Any3-finger/lift20 overlap | 1/96 |
| Strict stable grasp | **0/96** |

The only3-finger/lift20 overlap was row2233 for four frames (20ms); it ended below the lift origin. The largest elevations, rows1621 (183.15mm) and1639 (87.69mm), had at most two simultaneous fingers. Frame-level visual inspection showed exterior-hand/palm-root scoop or wedge contact rather than finger enclosure. Rows944 and2340 were brief threshold-crossing bump events.

## Mechanistic conclusion

The policy reliably reaches and contacts the object. Multi-finger closure appears in some trajectories, and exterior-hand collisions can produce large vertical motion. The unresolved mechanism is coordinated closure under load and retention: lift and three/four-finger closure almost never coincide, and no trajectory maintains the object above20mm at the end.

The user canceled further training and rollout. GPU0–3, the retained action session, and the owned inference server were released. The accepted80K sampler and96-row evidence are the final state of this experiment.
