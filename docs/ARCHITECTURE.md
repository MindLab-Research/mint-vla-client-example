# Client, MINT, and OpenPI architecture

## Component boundary

```text
Developer clone
/home/jay/vla/mint-vla-client-example
            |
            | commit / push
            v
GitHub: MindLab-Research/mint-vla-client-example
            |
            | git pull --ff-only
            v
GPU-host client checkout
/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
            |
            | HTTP: normalized observation -> action prediction
            v
MINT: mint-action-lora-r16 / action-lora-r16
            |
            | imports model code
            v
OpenPI: openpi-action-lora-r16 / action-lora-r16
```

The GPU-host process is a client even though it runs beside the MINT server. It
owns Lance reads, image/state projection, normalization, MuJoCo rollout,
contact feedback, B-scheme action reconstruction, result metadata, and HTTP
request orchestration. MINT owns model/session/checkpoint lifecycle and train or
inference execution. OpenPI owns the model implementation.

## Repository ownership

| Repository | Role | Update rule |
| --- | --- | --- |
| `MindLab-Research/mint-vla-client-example` | Formal VLA client | Develop in a clone; push reviewed commits; GPU checkout uses `git pull --ff-only` |
| `mint-action-lora-r16`, branch `action-lora-r16` | MINT API/backend | Server changes are committed and reviewed here, never copied into the client |
| `openpi-action-lora-r16`, branch `action-lora-r16` | OpenPI A-LoRA model | Model changes are committed and reviewed here, never copied into the client |

The shared legacy directory `/vePFS-Mindverse/user/intern/wenxi/vla_mint` is not
an execution or modification target.

## Formal client locations

- GitHub: `git@github.com:MindLab-Research/mint-vla-client-example.git`
- local development/context clone: `/home/jay/vla/mint-vla-client-example`
- GPU execution checkout: `/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example`
- ignored runtime configuration: `config/remote.env` in each machine-specific checkout

The old local repository is retired in place at `/home/jay/vla/vla_mint`
while pre-existing shells finish. Its complete Git history is preserved in
`/home/jay/vla/_archive/vla_mint-pre-formal-20260727.bundle`; it may be renamed
after those processes exit. The old non-Git GPU execution tree remains retired
in place at `/vePFS-Mindverse/user/intern/wenxi/vla_mint-parallel-preprocess`
until its pre-existing shells exit. Neither legacy tree is an independent
source or a destination for new work.

## MANO 32D v1 state contract

```text
state[0:26]  = MANO hand qpos
state[26:31] = index / thumb / ring / middle / pinky contact
state[31]    = current_object_z - initial_object_z

action[26:32] = physical zero
```

Training contact is target-object Lance contact-record presence. Mode4 contact
is target-object × MANO-keypoint MuJoCo contact-pair presence. Palm participates
in the 16-keypoint integrity check but has no five-finger output. V1 does not
call `mj_contactForce`, compute `force_norm`, or apply a `0.01 N` threshold.

The locked gesture03 v1 norm SHA256 is
`507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d`.

## Recommended worktree pattern

A branch isolates Git history; a worktree also isolates the filesystem. Do not
check out a new branch inside the project-owned GPU worktrees while a service
may be importing code from them. That would change the source beneath the
running process. Create a new branch and worktree together from the validated
base instead.

### Client-only development

A user who only calls an existing managed MINT endpoint needs an independent
clone of this repository and a normal feature branch:

```bash
git clone git@github.com:MindLab-Research/mint-vla-client-example.git
cd mint-vla-client-example
git switch -c users/<user>/<task> origin/main
```

On a shared machine, a second client worktree is also valid when concurrent
experiments need different client commits:

```bash
git -C /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example \
  worktree add \
  -b users/<user>/<task> \
  /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-<user>-<task> \
  main
```

### Independent server or backend/model development

Create a paired MINT and OpenPI worktree from the currently validated
`action-lora-r16` branches:

```bash
# MINT
MINT_COMMON=/vePFS-Mindverse/user/intern/wenxi/mint
MINT_WORKTREE=/vePFS-Mindverse/user/intern/wenxi/mint-<user>-<task>
git -C "$MINT_COMMON" worktree add \
  -b users/<user>/<task> \
  "$MINT_WORKTREE" \
  action-lora-r16

# OpenPI
OPENPI_COMMON=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/src/openpi
OPENPI_WORKTREE=/vePFS-Mindverse/user/intern/wenxi/openpi-<user>-<task>
git -C "$OPENPI_COMMON" worktree add \
  -b users/<user>/<task> \
  "$OPENPI_WORKTREE" \
  action-lora-r16
```

Point the user's private client configuration or server launcher to the pair:

```bash
MINT_CODE_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint-<user>-<task>
MINT_OPENPI_ROOT=/vePFS-Mindverse/user/intern/wenxi/openpi-<user>-<task>
```

Worktrees do not isolate runtime state. Every independently launched server
also needs a unique TCP port, confirmed GPU set, runtime/checkpoint root,
temporary root, action-session state root, and log path. Do not remove a
worktree until its server has stopped and its branch is clean and pushed.

## Local context policy

The local formal clone may contain an ignored `.memory/` tree used by Pi:

- `.memory/project/`: stable architecture, contracts, and invariants;
- `.memory/local/`: machine-specific paths, ports, runtime and delivery state;
- `.memory/tasks/`: experiment evidence and current epistemic models.

Team-facing behavior belongs in tracked `README.md` and `docs/`. Client source
belongs in Git. Server/model source belongs in MINT/OpenPI. `.memory/` points to
those sources and must not become a second codebase.

## Operational invariant

Before a run, record the client, MINT, and OpenPI commits independently. A
client commit does not identify backend/model source. Ports and GPU allocations
are runtime facts and must be checked per run.
