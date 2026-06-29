# Plan.md — VLA (OpenPI) Local Dev Plan

Branch: `dev-vla-wenxi` (cut from `develop`)
Owner: wenxi
Purpose: living plan + understanding log for bringing up MinT and running pi0.5 (and pi0-fast) VLA on the dev cluster.

> This file records WHAT we are doing and WHY (understanding the project).
> `Excute.md` records HOW (exact runnable commands per goal).
> Update both as we learn. Never delete goals; only mark done or add.

---

## 0. Project mental model (fog of war cleared)

MinT (`mint-server`) is a **FastAPI control plane**, not a compute engine. It:
- owns HTTP, auth, request validation, async-future polling
- brokers all GPU work to **detached Ray actors** on worker nodes

```
local client ──HTTP──> mint-server (FastAPI, CPU driver) ──Ray──> GPU worker actors
                                                                  ├─ vLLM (inference, multi-LoRA)
                                                                  ├─ Megatron / dense (text training)
                                                                  └─ OpenPI (VLA train + action inference)
```

Detached control-plane actors that must exist before/around the API:
`mint_config` → `mint_model_actor_supervisor` → (it ensures) `mint_task_state_store`,
`mint_model_work_scheduler`, `mint_maintenance_cron`. Server restart only loses
per-process caches; detached actors survive.

**Async contract:** long work returns `{"request_id": ...}`; poll
`POST /api/v1/retrieve_future` (HTTP 408 = pending, 200 = done).

### VLA specifics (this is our focus)

VLA = Vision-Language-Action. Implemented via the **OpenPI** backend in
`mint_server/backend/openpi/`. Two model families:

| Model (base_model id) | family | loss_fn | train backend | action_dim | notes |
|---|---|---|---|---|---|
| `openpi/pi0-fast-libero-low-mem-finetune` | `ar_action_tokens` | `cross_entropy` | `openpi_fast` | 7 | autoregressive action tokens |
| `openpi/pi05-libero-low-mem-finetune` | `flow_action` | `flow_matching` | `openpi_pi05` | 32 | flow-matching, **our primary target** |

Registry: `mint_server/backend/core/model_registry.py:132-164`.

VLA routes live under `/api/v1/mint/*` (`mint_server/routes/mint.py`), guarded by
`MINT_DISABLE_MINT_ROUTE`. The public surface:
- `POST /api/v1/create_model` — create LoRA training model
- `POST /api/v1/mint/vla/train_step` — one training step (data = list of VLA `Datum`)
- `POST /api/v1/save_weights_for_sampler` — materialize inference checkpoint
- `POST /api/v1/mint/action_sessions` — create action (sampling) session
- `POST /api/v1/mint/action_sessions/{id}/act` — run action inference
- `DELETE /api/v1/mint/action_sessions/{id}` — cleanup

OpenPI workers need **JAX + openpi** in their runtime, which the default dev
runtime (`/vePFS-Mindverse/share/mint/dev/runtime/cpu/site-packages`) does NOT
have. The openpi-capable runtime root is
`/vePFS-Mindverse/share/code/mint-runtime-py31213-openpi-candidate-20260331-203300`
(verified: `jax 0.5.3`, `openpi` importable via its `host-venv` python 3.12).

### IMPORTANT — base updated to develop `d86e1487` (VLA runtime rollup #698)

On 2026-06-22 we rebased `dev-vla-wenxi` onto develop `d86e1487`
(`fix(openpi): consolidate VLA runtime rollup (#698)`). This changes the runtime
model. Authoritative doc is now
`.claude/skills/architecture-design/references/vla-runtime.md` (the older
`vla_*` docs are explicitly background-only). Key deltas vs our earlier notes:

- **No subprocess / stdout JSON-RPC worker protocol.** OpenPI workers now run
  **directly inside the Ray actor process** via `OpenPIDirectWorkerClient`
  (`mint_server/backend/openpi/openpi_direct_runtime.py`). There is no separate
  Python executable; actors import the worker module and call
  `_dispatch(session, op, payload)` directly. Deleted symbols (e.g.
  `OpenPIFastWorkerClient`, worker `main()`, `python_executable`/`build_env()`)
  must NOT be reintroduced.
- **Runtime env** for actors comes from `_openpi_runtime_env_vars()`: all
  `MINT_OPENPI_*` keys + `XLA_FLAGS` + HF/OPENPI cache vars + standard actor env
  built with `PFS_PYTHONPATH`. Import path resolves from
  `MINT_OPENPI_FAST_PYTHONPATH` or `PFS_RUNTIME_ENV_ROOT` (validated with
  `require_host_python=True`). So the runtime root still matters — point it at
  the openpi-capable runtime.
- **pi0.5 action sessions are supervisor-first by default**: the default factory
  *fails* if no runtime actor is already reconciled. For a quick smoke without
  waiting on supervisor reconcile, set `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1`
  (explicit bypass; not the general path).
- Shared GPU runtime actor (`OpenPISharedRayRuntimeActor`, name
  `mint_openpi_shared_<sha1>`) is keyed by base model + worker module + config +
  action dims/horizon + token limit + timeouts; it multiplexes many Mint sessions
  by save/load of per-session state. Single-GPU actors, published to supervisor
  inventory as `ActorType.OPENPI`. Placement pinnable via `MINT_MODEL_PLACEMENT_JSON`.
- **#698 caveat (from the doc's "Known open work"):** not merge-ready until the
  external uvicorn multi-worker control-plane authority gate is fixed; keep
  `MINT_UVICORN_WORKERS=1`. Live validation still pending upstream — expect rough
  edges; this is exactly what we're validating.
- Registry path is `mint_server/backend/core/model_registry.py` (the new doc says
  `backend/model_registry.py` — minor doc inaccuracy, code is under `core/`).

---

## Deployment environment findings (2026-06-22, verified)

### A. Runtime tiers — `gpu_vla` is NOT required for pi0.5 (CORRECTED)

> CORRECTION (2026-06-22): An earlier draft of this section claimed pi0.5 is
> blocked on a missing `gpu_vla` tier. That was WRONG. Verified below: no runtime
> code path requests `gpu_vla`; OpenPI runs from the `gpu_rl` tier, which already
> contains openpi. The default dev runtime is sufficient — no symlink root, no
> rebuild needed.

#698 made the runtime root tiered: code reads `<env_root>/<tier>/manifest.json`
(`mint_server/ray/runtime_env.py:206-209`). Tiers: `cpu`, `gpu_rl`, `gpu_vla`
(cumulative: `gpu_vla` = `gpu_rl` + openpi, `_tiers_for` `:190-203`).

**What actually requests which tier:**
- API host process: `cpu` tier (`start_dev_server.sh:325,373` →
  `cpu/base-python/bin/python3.12`, `cpu/site-packages`). No torch/jax/openpi.
- OpenPI GPU worker: uses `OpenPIFastRuntimeSpec.from_env()`
  (`openpi_fast_runtime.py:87-115`, shared by FAST/pi0.5/training/action). With no
  env override it calls `validate_runtime_env_layout(...)` and
  `bootstrap_runtime_pythonpath(...)`, **both defaulting to `tier=gpu_rl`**
  (`runtime_env.py:217,321`).
- `TIER_GPU_VLA` is defined but **no runtime path requests it** (grep across
  `mint_server/` finds only the definition + `_tiers_for` mapping). It is
  forward-looking infra, consistent with "#698 not merge-ready".

**Verified empirically:** the live `gpu_rl` build at
`/vePFS-Mindverse/share/mint/dev/runtime/gpu_rl` imports `jax` 0.5.3 and `openpi`
(`.../gpu_rl/src/openpi/src/openpi/__init__.py`, commit `e6b0441` = pyproject pin);
its `host-venv` has jax/flax/jax_cuda12 installed.

**Implication:** start pi0.5 with the DEFAULT runtime root
(`PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/mint/dev/runtime`, the script's
default). Do NOT build a `gpu_vla` tier and do NOT use the flat candidate root.
(Re-verify the worker can `import openpi` at runtime when we actually launch — the
`from_env` default tier is the thing to confirm live.)

### B. Dev host access (SSH) — partial blocker

- New model (per user, 2026-06-22): **no more Ray Client mode** (it crashed the
  GCS server). The dev cluster now has a dedicated driver
  **`mint-dev-driver` = 192.168.42.106**; run the dev API server there (not on the
  Ray head). Ray head is `192.168.42.141`.
- Our shell is `192.168.42.153` (same subnet as driver/head; IP-reachable).
- `mint-dev-driver` sshd listens on **port 2222** (`22` is refused; `2222` open).
  Head `.141` uses `22`.
- Auth: keys go into the shared file
  `/vePFS-Mindverse/share/mint/runtime/ssh/authorized_keys`; a sync mechanism
  copies it into each node's `~/.ssh/authorized_keys`. Our pubkey
  (`~/.ssh/ssh_worker_rsa_key.pub`) was appended (backup made; existing keys
  untouched), but ssh to `106:2222` still returns `Permission denied (publickey)`
  → the key has not yet propagated to the node. **Blocked until propagation or a
  manual push of the key into 106's `~/.ssh/authorized_keys`.**

### C. New runtime knobs (per user, develop reworked 2026-06-22)

`origin/develop` was reworked to be easier to run. Knobs to set at start:
- `MINT_CODE_ROOT` (startup code root)
- namespace — defaults to `mint_<username>` if unset
- `MINT_PORT` — defaults to a hash of the namespace (avoids port collisions)
- TaskStore IN-MEMORY mode now available (non-persistent, pure in-memory)
- `PLACEMENT_JSON` — GPU worker placement
CI gate now exists but only covers type-check + the Scheduler component.

---

## Goal 1 — Deploy MinT so the service runs

**Status:** planned
What "deployed" means here: a dev API server answering `/api/v1/healthz`
`{"status":"ready"}`, attached to the dev Ray cluster, in our own namespace.

Deployment shape (from `mint-dev` skill):
1. Pick an explicit `MINT_CODE_ROOT` under `/vePFS-Mindverse/share/...` visible to
   all Ray nodes (NOT `/vePFS-Mindverse/user/...`, NOT the shared dev tree). We
   rsync our `dev-vla-wenxi` checkout there.
2. Generate placement config (`scripts/tools/gen_dev_placement.py`) — worker IPs
   change when the cluster is recreated.
3. Start via `scripts/start_dev_server.sh` with our namespace `mint_wenxi_dev`.
4. Wait for healthz, open SSH tunnel for local client access.

**Decisions to confirm with user (Goal 1):**
- Which `MINT_DEV_USER` / namespace (default derive `mint_wenxi_dev`).
- Is the dev Ray cluster already up with a GPU worker? (needs `volcano-cluster`
  skill if not — and we must NOT run `ray`/`volc` locally.)

Standard (non-VLA) bringup uses `MINT_DISABLE_MINT_ROUTE=1`. **For VLA we must NOT
set that flag** (so the `/api/v1/mint/*` routes load).

---

## Goal 2 — Get pi0.5 (flow) running end-to-end

**Status:** planned. Depends on Goal 1, plus VLA-specific overrides.

VLA-specific deployment deltas over Goal 1:
- Runtime root = openpi candidate runtime (has JAX/openpi).
- `MINT_SUPPORTED_MODELS` must include `openpi/pi05-libero-low-mem-finetune`
  (and pi0-fast). Default dev list does not.
- Do NOT set `MINT_DISABLE_MINT_ROUTE`.
- `MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params`
- `MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/models/openpi/pi05_base/assets`
- (pi0-fast equivalents if we also bring up fast — see runbook invariants.)

Bring-up ladder (smallest moving parts first):
1. **Synthetic smoke** — `scripts/wip/openpi_vla_smoke.py --model openpi/pi05-...`
   Validates: create_model → vla/train_step (flow_matching) → save_weights →
   action_session → act, with dummy 1×1 images + zero state/actions. No dataset
   dependency. This proves the *service path* works.
2. **Real-data SFT** — `scripts/wip/openpi_libero_sft.py` on one LIBERO task,
   expect a downward loss curve (reference: task 10 "put the bowl on the plate",
   12 steps, loss ~0.119 → ~0.063 from the 2026-04-08 validation report).

"pi0.5 runs" = synthetic smoke returns an action tensor AND a real LIBERO SFT
run produces a monotone-ish downward loss curve.

---

## Goal 3 — Understand pi0.5 data source + how data becomes a service

**Status:** planned (partly mapped already; deepen during Goal 2).

Data source (real): LIBERO dataset in LeRobot parquet format at
`/vePFS-Mindverse/share/code/conley/.hf-lerobot/physical-intelligence/libero`.
- `meta/tasks.jsonl` — task_index → instruction text
- `meta/episodes.jsonl` — episode_index → task, length
- `data/chunk-XXX/episode_XXXXXX.parquet` — per-step `image`, `wrist_image`,
  `state`, `actions`

Data → service pipeline (`openpi_libero_sft.py`):
1. Load OpenPI config (`pi05_libero`) + assets (norm_stats) via
   `openpi.training.config` and `openpi.transforms`.
2. Slide a window of `action_horizon` steps over an episode.
3. Apply the OpenPI transform chain (repack → data transforms → Normalize →
   model transforms) to produce tokenized prompt + normalized actions.
4. Lower into a **VLA `Datum`**: `observation` (state TensorData + model_input
   chunks: image chunks per camera + encoded_text tokens) and `supervision`
   (pi0.5: `actions` TensorData [horizon, action_dim]; pi0-fast:
   `target_tokens` + `weights` + `token_ar_mask`).
5. POST batches to `/api/v1/mint/vla/train_step`.

Server side: route `mint.py:vla_train_step` → dispatch
`mint.vla.train_step` → `OpenPIPi05TrainingEngine.forward_backward`
(`openpi_pi05_training.py:300`, asserts `loss_fn == "flow_matching"`) → OpenPI
pi0.5 worker (`openpi_pi05_worker.py`) on a `num_gpus=1` Ray actor.

Sampling/serving model (architecture note `vla_sampling_architecture_gap.md`):
one shared OpenPI action actor per base model; per-tenant isolation via
checkpoint-derived session state on vePFS (not yet a true shared-adapter surface
like vLLM multi-LoRA). Worth knowing but not a blocker for getting pi0.5 running.

Deliverable for Goal 3: a written walkthrough (in this file or a sibling) tracing
one LIBERO frame from parquet → transform → Datum → train_step → loss, with file
paths and line numbers.

---

## Goal 4 — Add more services (DEFERRED, not now)

Placeholder. Candidates from the VLA roadmap doc
(`vla_next_benchmarks_and_demos_20260408.md`): LIBERO suite expansion,
LIBERO-plus robustness, MinT-hosted LIBERO policy-server demo, DROID, Meta-World,
CALVIN. Also: meaningful pi0-fast RL (still "partial" per validation report),
mixed-client contamination-freedom, 30-client pressure pass. Revisit after
Goals 1-3 are solid.

---

## Open risks / watch-list

- **Cluster availability:** dev GPU worker must be up (Volcano pod). Don't run
  `ray`/`volc` locally — use the `volcano-cluster` skill.
- **Stale detached actors / placement groups:** Hall of Shame warns repeatedly.
  Pre-flight: list PGs/actors in our namespace; remove only owned stale ones
  before bringup.
- **Wrong runtime root:** if OpenPI actors crash on `import jax`/`import openpi`,
  the runtime root is wrong. Use the openpi candidate runtime.
- **Code not synced / server stale:** Python doesn't hot-reload. After any code
  change: rsync (no `--delete`) → kill server → clean stale control-plane actors
  → restart → verify new PID.
- **Don't substitute a toy task** for the real pi0.5 LIBERO run (Hall of Shame).

---

## Progress log

- 2026-06-22: Created branch `dev-vla-wenxi` from `develop`. Read architecture +
  VLA reference docs and key OpenPI code. Confirmed data/model paths exist on PFS
  and the openpi candidate runtime imports jax+openpi. Wrote Plan.md + Excute.md.
- 2026-06-22 (later): develop advanced `96216883` → `d86e1487`
  (`fix(openpi): consolidate VLA runtime rollup (#698)`). Fast-forwarded local
  `develop` and rebased `dev-vla-wenxi` onto it (no own commits; clean FF;
  `wenxi_dev_md/` preserved untracked). Read the new authoritative
  `vla-runtime.md`; updated Plan.md (direct in-actor runtime, supervisor-first
  pi0.5 action, `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME`, uvicorn-workers=1
  caveat) and Excute.md Goal 2 restart command. Both branches at `d86e1487`.
- 2026-06-22 (deployment recon): Confirmed `origin/develop` still at `d86e1487`.
  Found the `gpu_vla` runtime gap (see "Deployment environment findings A"):
  pi0.5 needs a `gpu_vla` tier that no PFS runtime root currently has, but the
  existing `gpu_rl` build already contains openpi+jax (commit matches pyproject),
  so a zero-network symlink root is the planned fix. SSH recon: dev driver is
  `192.168.42.106:2222`; appended our pubkey to the shared authorized_keys
  (backup made) but it has not propagated yet → ssh still blocked. Captured new
  runtime knobs and the no-Ray-Client/dedicated-driver model from the user.
  Wrote analysis into Plan.md; commands pending sign-off in Excute.md.
- 2026-06-22 (correction + startup study): Read `start_dev_server.sh` end-to-end
  and the OpenPI runtime spec. **Corrected a wrong earlier claim:** the missing
  `gpu_vla` tier is NOT a pi0.5 blocker — no runtime path requests `gpu_vla`;
  OpenPI's `from_env` defaults to the `gpu_rl` tier, which already imports openpi.
  The DEFAULT dev runtime is sufficient; dropped the symlink-root plan. SSH: host-
  key fingerprints proved `106:2222` is a system sshd (reads 106-local
  authorized_keys, our key not there), while the shared-file mint-sshd runs on the
  excluded head `141:22`. Real remaining blocker = getting our key into 106's
  local authorized_keys (or running driver commands via `!`). Updated Plan.md
  Findings A/B and Excute.md (removed Goal 0.2, default runtime in Goal 1/2).
