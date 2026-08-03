# MINT VLA client example

This is the formal Git repository for fine-tuning and evaluating OpenPI pi0.5
through a MINT server. It owns client-side data projection, normalization,
training requests, MuJoCo Mode4 rollout, and result metadata. MINT and OpenPI
remain independent server/model repositories.

## Architecture and ownership

```text
GitHub: MindLab-Research/mint-vla-client-example
                       |
                       | git pull --ff-only
                       v
GPU-host client checkout (this repository)
/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
                       |
                       | HTTP: 32D state -> action prediction
                       v
MINT server: /vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16
                       |
                       | imports OpenPI model implementation
                       v
OpenPI: /vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16
```

The client usually runs on the GPU host because the Lance dataset and runtime
packages live on PFS. It is still a separate HTTP client. Mode4 contact and
physics state are computed here, before the request reaches MINT.

### Source-of-truth rule

- `config/datasets/mano_dataset_release.json` is the sole machine-readable source of truth for MANO data, language/contact sidecars, producer commits, assets, runtime contracts, norms, and physics evidence. Launchers resolve canonical roles through `scripts/mano_dataset_release.py`; docs and local env files cannot redefine the release.
- This Git repository is the source of truth for client code.
- The checkout at `/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example`
  is both a real Git checkout and the supported execution directory.
- Update it with reviewed commits and `git pull --ff-only`; do not overlay it
  with rsync or maintain a separate `.vla_mint_commit` marker.
- The former non-Git execution copy
  `/vePFS-Mindverse/user/intern/wenxi/vla_mint-parallel-preprocess` is retired
  in place while pre-existing shells finish. It is rollback-only and must not
  receive new work.
- Never modify the shared legacy copy
  `/vePFS-Mindverse/user/intern/wenxi/vla_mint`.

## Current MANO 32D v1 contract

`mano_five_finger_contact_lift_v1` is the existing-checkpoint state contract:

```text
state[0:26]  = MANO hand qpos
state[26:31] = index / thumb / ring / middle / pinky contact
state[31]    = current_object_z - initial_object_z

action[26:32] = physical zero
```

Training contact is target-object Lance contact-record presence. Mode4 contact
is target-object × MANO-keypoint MuJoCo contact-pair presence. Mode4 does not
call `mj_contactForce`, compute `force_norm`, or apply a `0.01 N` threshold.
Palm remains part of the 16-keypoint integrity check and has no output channel.

The locked gesture03 v1 norm SHA256 is:

```text
507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d
```

## MANO state44 v2 profile

`openpi/pi05-action-lora-r16-state44-finetune` uses profile
`pi05_action_lora_r16_state44_v1`. It separates a 44D observation from the
unchanged B-schema action `[10,32]`:

```text
state[0:26]  = MANO qpos26
state[26:31] = index / thumb / ring / middle / pinky target-object contact
state[31]    = object lift from trajectory or rollout initialization
state[32:37] = signed fingertip-sphere to target collision-surface distance (m)
state[37:42] = causal 25 ms fingertip-to-palm radial rate (m/s; + closing)
state[42]    = object-floor contact-pair presence
state[43]    = elapsed duration of the current >=2-finger contact run (s)

action[0:32] = existing query-anchored B schema; action[26:32] stays physical zero
```

Finger order is fixed to `index/thumb/ring/middle/pinky`. Training source poses
and Mode4 both use the same MuJoCo URDF fingertip markers and target-object
collision geoms. Source timestamps must advance by exactly 5 ms; the rate uses
the current sample and the sample five steps in the past. Persistence resets to
zero below two simultaneous finger contacts and has no hard physical clip; its
authenticated population quantiles provide normalization. The profile ends at
index 43 and contains no opposition/load-geometry score.

StateAug perturbs qpos only. It recomputes the five surface distances from the
perturbed qpos and current object pose, while preserving the clean radial-rate
history because an independently perturbed frame does not define a valid 25 ms
trajectory. State normalization must have width 44 and action normalization
must have width 32; state32 norms are rejected.

Build a versioned norm and perform the full raw-token audit before training:

```bash
./scripts/remote/run_client.sh scripts/train/prepare_mano_state44_profile.py \
  --lance-dataset "$DATASET" --target-lance-dataset "$DATASET" \
  --row-indices "$ROWS" --frame-window contact --contact-context-frames 100 \
  --contact-window-manifest "$CONTACT_MANIFEST" \
  --gesture-index config/datasets/new_all_generated_mano.index.json \
  --norm-output-dir "$OUTPUT/norm" --report-json "$OUTPUT/report.json"
```

The command writes the norm only when every selected sample is at or below the
immutable 200-token prefix limit. Training and Mode4 must pass both the output
norm SHA and `--state-contract state44`; the state44 model identity and contract
are rejected if selected independently.

The certified rows507–2503 contact-ctx100 population contains 1,997 rows and
1,160,274 selected frames. Its full audit had zero overflows: raw token lengths
were min143, p50=163, p95=172, p99=175, max182. The authenticated norm is:

```text
cd916feee01138f957ca400fad25d02ebb18029e8fc4844c8019f3814caf622a
```

Artifacts are under
`/vePFS-Mindverse/user/intern/wenxi/results/training/state44_profile_population_507_2503_20260730/`;
`validation_manifest.json` binds the population, norm, report, and three feature
branch commits. This norm is valid only for that exact row/window population.

## Current deployment map

| Component | Path / branch |
| --- | --- |
| Formal client checkout | `/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example`, `main` |
| MINT server | `/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16`, `action-lora-r16` |
| OpenPI model worktree | `/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16`, `action-lora-r16` |
| Runtime | `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl` |
| MANO release resolver | `config/datasets/mano_dataset_release.json` |
| Lance dataset (`training_dataset` role) | `/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance` |
| Gesture03 v1 norm | `/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726` |

Ports and GPUs are allocated per run. A different port does not imply a
separate GPU allocation; verify both before starting a server.

## What the current stack implements

The checked-in client and the paired `action-lora-r16` MINT/OpenPI worktrees
implement the following validated path:

- OpenPI pi0.5 action-expert LoRA, rank 16, trained through the MINT HTTP API;
- B-exact `urdf_target_absolute` supervision and query-anchored action
  reconstruction, with `action[26:32]` fixed to physical zero;
- backward-compatible `mano_five_finger_contact_lift_v1` state32 and the
  separately versioned `mano_five_finger_contact_geom_rate_v2` state44 profile;
- clean and state-augmented training, with StateAug noise restricted to MANO
  qpos and state44 surface geometry recomputed from the perturbed qpos;
- exact-byte normalization locking and state/action provenance metadata;
- cooperative deadline checkpoint saving;
- Mode4 closed-loop MuJoCo rollout with native position-servo control,
  sim-owned object motion, and fixed-shape `act_batch` requests;
- Mode3 historical kinematic diagnostic for B-exact v1 checkpoints: predicted
  MANO qpos and sim cameras are evaluated against the reference object
  trajectory with `mj_forward` only, never physics integration;
- five-finger Mode4 contact from object-keypoint pair presence, with no
  `mj_contactForce` or `0.01 N` filter;
- focused training, inference, contract, migration, and production-path tests.

Existing clean/StateAug checkpoints use this v1 training contract and do not
require retraining for the pair-presence Mode4 alignment.

## Recommended multi-user usage

| User activity | Client checkout | MINT/OpenPI worktrees |
| --- | --- | --- |
| Call an existing managed MINT endpoint | Use the user's own clone of this repository | Not needed |
| Launch an independent MINT server without source edits | Use the user's own clone | A dedicated MINT + OpenPI pair is recommended for provenance and isolation |
| Modify MINT backend or OpenPI model code | Use the user's own clone | A dedicated MINT + OpenPI pair is required |

Do not modify the project-owned worktrees
`mint-action-lora-r16` or `openpi-action-lora-r16` for another user's
experiment. Creating a worktree isolates source and branch state; it does not
isolate ports, GPUs, runtime/checkpoint roots, temporary directories, session
state, or logs. Allocate all of those independently. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#recommended-worktree-pattern) for
the exact commands.

## Prepare the formal checkout

On the GPU host:

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
git switch main
git pull --ff-only origin main
cp -n config/remote.env.example config/remote.env
```

`config/remote.env` is intentionally ignored. Keep machine-specific paths,
credentials, and per-run ports there; never commit it.

## Run the client

The launcher constructs the production `PYTHONPATH`, keeps client JAX on CPU,
and checks the MINT endpoint before non-dry runs:

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example

./scripts/remote/run_client.sh scripts/train/train_cube1_01_compare.py \
  --model openpi/pi05-action-lora-r16-finetune \
  --row-indices 810,811 \
  --missing-contact-policy error \
  --action-source urdf_target_absolute \
  --extended-state \
  --norm-stats-dir /vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726 \
  --norm-sha-expected 507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d \
  --dry-run \
  --save-path client_preflight \
  --output-json /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/training/client_preflight/result.json
```

For every new Mode4 checkpoint/row evaluation, use the single parameterized
launcher rather than copying an experiment script. It can attach to an existing
endpoint or own a dedicated server and records the effective stride, ensemble,
server, source provenance, and per-phase timing in the output root. Generated
inference defaults to this checkout's ignored `results/inference/` directory;
use `--run-name NAME` for a stable client-local directory or `--output-dir PATH`
only when an explicit override is required. Mode4 defaults to contact-window
initialization using the canonical ctx100 manifest. `dataset_reference.mp4`
keeps the full demonstration, while head/wrist physics-comparison videos and
all rollout arrays cover only the synchronized contact window. Use
`--frame-window full` only for an explicitly named full-trajectory stress test.
Multi-row Mode4 defaults to `--row-execution lockstep --row-batch-size 4`:
four independent MuJoCo trajectories contribute four real observations to each
fixed-shape `act_batch`, while only the final partial group/window tail is
padded:

```bash
./scripts/remote/run_mode4_eval.sh --help
```

For standalone dedicated-server inspection outside the unified Mode4 launcher,
use the reviewed operator helper only after GPU ownership is confirmed:

```bash
./scripts/remote/run_action_lora_server.sh \
  --runtime-root /vePFS-Mindverse/user/intern/wenxi/results/training/<run>/server \
  --port 30532 --gpus 0,1,2,3 --print-config
```

Remove `--print-config` only when the printed allocation is correct.

## Recommended Action-LoRA training defaults

| Server GPUs | Batch | Producers | Prefetch | Build workers | Datum cache |
|---:|---:|---:|---:|---:|---:|
| 4 | **64** | **2** | **2** | 16 | 256 |
| 8 | 128 | **8** | **8** | 16 | 256 |

The validated four-GPU production profile is batch64/P2/prefetch2 with the
complete selected population resident. The eight-GPU row remains the measured
throughput profile and also requires the corrected resident-action client.
Pass `--learning-rate 5e-5` explicitly; the CLI's legacy `1e-4` default is not
the MANO production setting. Both use `--row-cache-size N` and
`--preload-selected-rows`, where `N` is the number
of selected trajectory rows. For full Cylinder1, `N=1039`. See
[`docs/CLIENT_FINETUNE.md`](docs/CLIENT_FINETUNE.md#choose-the-gpu-count-defaults)
for a copy-paste command and the low-memory fallback.

## Maintained entrypoints

- `scripts/mano_dataset_release.py`: the only resolver for canonical MANO release roles.
- `scripts/tools/validate_mano_dataset_release.py`: fail-closed fast/deep release validation for paths, hashes, Lance population/schema, producer commits, assets, and physics evidence.
- `scripts/train/train_cube1_01_compare.py`: selected-row clean/StateAug
  training client, deterministic multi-producer materialization,
  population-resident row caching, checkpointing, deadline handling, and locked
  norm metadata.
- `docs/ACTION_LORA_GPU_SATURATION.md`: measured four- and eight-GPU defaults,
  sharding evidence, throughput results, and population-resident prefetch limits.
- `scripts/train/openpi_vla_smoke_lance_base.py`: Lance projection and MINT wire
  format.
- `scripts/eval/infer_mano_mode4.py`: maintained MANO closed-loop real-physics
  inference entrypoint.
- `scripts/eval/infer_mano_mode3.py`: dedicated historical kinematic Mode3
  diagnostic; it is intentionally not a Mode4 alternative or a generic
  numbered-mode multiplexer.
- `scripts/eval/mano_physics_core.py`: historical 26D MuJoCo contract retained for
  pre-migration evidence only.
- `scripts/eval/manorl_native_physics.py`: pinned ManoRL 28D native-model adapter.
  Physics comes from `compile_model(...)`; render models reuse the previous Client
  head/wrist cameras, floor, background, lights, and hand color treatment, and
  are rejected if those visual additions change collision or dynamics arrays.
- `scripts/eval/replay_mano_target_physics.py`: maintained recorded-target
  physics-quality producer, migrated in place to metadata-resolved right-hand
  28D targets while preserving object locking, resume checksums, grading,
  multiprocessing, and Lance aggregation. Its production population is the
  verified 5,425-row filtered Lance; startup binds every filtered UUID to the
  accepted-row manifest and records both filtered and original merged indices.
- `tools/render_mano_native_trace_video.py`: renders saved native traces without
  dynamics steps and verifies the MP4 before publication. Head rendering accepts
  `--head-camera-preset current|legacy`; `current` (elevated 65°) is the default,
  while `legacy` preserves the original 75° view. The selected preset and its
  expanded camera parameters are recorded in the video manifest.
- `scripts/mano_state_contract.py`: shared v1 contract identity, contact rule,
  and norm SHA verifier.
- `scripts/remote/run_client.sh`: client runtime and server preflight.
- `scripts/remote/run_mode4_eval.sh`: parameterized Mode4 evaluation launcher
  for an existing MINT endpoint or an explicitly owned dedicated server; it
  records effective configuration, source/normalization provenance, phase
  timings, and supports explicit `--video-mode none` diagnostic sweeps and
  retained action-session reuse through `--keep-server` / `--reuse-server-info`.
- `scripts/remote/stop_owned_mode4_server.sh`: ownership-checked retained-session
  cleanup and graceful stop for a server handed off with `--keep-server`.
- `scripts/remote/run_action_lora_server.sh`: dedicated action-LoRA server
  launcher; not a shared-server lifecycle command. Persistent JAX executable
  serialization is disabled unless explicitly enabled.
- `docs/ARCHITECTURE.md`: client/MINT/OpenPI ownership and local-context policy.
- `docs/MANO_DATA_PIPELINE.md`: canonical map of raw/derived MANO data, language labels, kinematic image rendering, target-DOF physics replay, assets, sidecars, and the current LoRA task boundary.
- `docs/CUBE1_CUBE2_STATEAUG80K_RESULTS.md`: final training and 96-row Mode4 evaluation evidence for the cube1+cube2 StateAug80K experiment.
- `docs/CLIENT_FINETUNE.md`: detailed client workflow.
- `Tutorial.md`: historical MINT no-Ray protocol reference.

## Repository hygiene

Commit source, tests, portable configuration templates, and user-facing docs.
Do not commit:

- `config/remote.env`;
- `.memory/` or agent transcripts;
- results, videos, logs, caches, bytecode, or runtime checkpoints;
- `.vla_mint_commit` or migration markers;
- secrets or credential-bearing Git URLs.

The MINT and OpenPI repositories are read-only dependencies for client work.
Server changes require their own reviewed commits in their respective
`action-lora-r16` branches.
