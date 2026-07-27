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

## Current deployment map

| Component | Path / branch |
| --- | --- |
| Formal client checkout | `/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example`, `main` |
| MINT server | `/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16`, `action-lora-r16` |
| OpenPI model worktree | `/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16`, `action-lora-r16` |
| Runtime | `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl` |
| Lance dataset | `/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance` |
| Gesture03 v1 norm | `/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726` |

Ports and GPUs are allocated per run. A different port does not imply a
separate GPU allocation; verify both before starting a server.

## What the current stack implements

The checked-in client and the paired `action-lora-r16` MINT/OpenPI worktrees
implement the following validated path:

- OpenPI pi0.5 action-expert LoRA, rank 16, trained through the MINT HTTP API;
- B-exact `urdf_target_absolute` supervision and query-anchored action
  reconstruction, with `action[26:32]` fixed to physical zero;
- `mano_five_finger_contact_lift_v1` 32D observations;
- clean and state-augmented training, with StateAug restricted to MANO qpos;
- exact-byte normalization locking and state/action provenance metadata;
- cooperative deadline checkpoint saving;
- Mode4 closed-loop MuJoCo rollout with native position-servo control,
  sim-owned object motion, and fixed-shape `act_batch` requests;
- Mode3 kinematic diagnostic for B-exact v1 checkpoints: predicted MANO qpos
  and sim cameras are evaluated against the reference object trajectory with
  `mj_forward` only; historical direct-setpoint and calibrated one-step servo-lag
  transitions are explicit inference options;
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
  --lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --target-lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --row-indices 810,811 \
  --contact-window-manifest /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.contact_ctx100_error_v1.json \
  --missing-contact-policy error \
  --action-source urdf_target_absolute \
  --extended-state \
  --norm-stats-dir /vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726 \
  --dry-run \
  --save-path client_preflight \
  --output-json /vePFS-Mindverse/user/intern/wenxi/results/client_runs/client_preflight/result.json
```

For a dedicated rank-16 server, use the reviewed operator helper only after GPU
ownership is confirmed:

```bash
./scripts/remote/run_action_lora_server.sh \
  --runtime-root /vePFS-Mindverse/user/intern/wenxi/results/training/<run>/server \
  --port 30532 --gpus 0,1,2,3 --print-config
```

Remove `--print-config` only when the printed allocation is correct.

## Maintained entrypoints

- `scripts/train/train_cube1_01_compare.py`: selected-row clean/StateAug
  training client, checkpointing, deadline handling, and locked norm metadata.
- `scripts/train/openpi_vla_smoke_lance_base.py`: Lance projection and MINT wire
  format.
- `scripts/eval/infer_mano_mode4.py`: maintained MANO closed-loop real-physics
  inference entrypoint.
- `scripts/eval/infer_mano_mode3.py`: dedicated historical kinematic Mode3
  diagnostic; it is intentionally not a Mode4 alternative or a generic
  numbered-mode multiplexer.
- `scripts/eval/mano_physics_core.py`: MuJoCo scene, collision, five-finger
  pair-presence contact, servo, and timing.
- `scripts/mano_state_contract.py`: shared v1 contract identity, contact rule,
  and norm SHA verifier.
- `scripts/remote/run_client.sh`: client runtime and server preflight.
- `scripts/remote/run_action_lora_server.sh`: dedicated action-LoRA server
  launcher; not a shared-server lifecycle command.
- `docs/ARCHITECTURE.md`: client/MINT/OpenPI ownership and local-context policy.
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
