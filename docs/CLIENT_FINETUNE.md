# Client fine-tuning guide

This guide is the supported onboarding path for coworkers using this project.
It describes the current shared machine and directories, but keeps server
operation separate from experiment development.

## 1. Know which process you are running

There are two independent processes:

| Process | Responsibility | Typical location |
| --- | --- | --- |
| MINT server | Loads pi0.5 on GPUs, creates LoRA sessions, trains, saves checkpoints, runs action inference | GPU host, managed by the server owner |
| VLA client | Reads Lance, decodes images, computes normalization, builds HTTP payloads, records metrics | GPU host, launched by an experiment owner over SSH |

The client is normally executed on the same remote machine because the dataset
and Python environment live on PFS. Running on the same machine does not make
it part of the server. The boundary is the MINT HTTP API.

Do not edit the shared MINT checkout for an experiment. `MINT_OPENPI_ROOT` may select an isolated OpenPI worktree ahead of the shared source; `MINT_CODE_ROOT` remains the MINT code selector. These client changes do not start a new server. Do not kill a server
because GPU utilization is temporarily low: model weights can occupy most GPU
memory while requests are between steps.

## 2. Current connection and directory map

```text
Git:          git@github.com:MindLab-Research/mint-vla-client-example.git
SSH:          root@115.190.235.210 -p 1634
Client Git:   /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
MINT URL:     http://127.0.0.1:30532       (example only; confirm per run)
MINT code:    /vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16
OpenPI code:  /vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16
Environment:  /vePFS-Mindverse/user/intern/wenxi/mint_env
Assets/code:  /vePFS-Mindverse/user/intern/wenxi/pi-finetune
Dataset:      /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance
Run outputs:  /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/<run-name>
Inference:    /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/inference/<run-name>
Baselines:    /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/baselines/<baseline-name>
```

The committed defaults are in `config/remote.env.example`. Each coworker may
create an ignored `config/remote.env` for a different port, user, dataset, or
result root. Do not commit passwords, private keys, or production API keys.

## 3. Prepare and synchronize the client

The GPU-host directory is now the formal Git checkout rather than an rsync
copy:

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
git switch main
git pull --ff-only origin main
cp -n config/remote.env.example config/remote.env
$EDITOR config/remote.env
```

Develop on a clone of the same GitHub repository, push reviewed commits, then
fast-forward the GPU-host checkout. Do not rsync files over the checkout: that
would create an untraceable dirty execution tree. Never clone this repository
over a MINT/OpenPI checkout or the shared legacy client directory.

## 4. Check server allocation

Ask the MINT server owner which port and GPU set are assigned to the run. Then
check the HTTP endpoint from the GPU host:

```bash
ssh -p 1634 root@115.190.235.210
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:30532/openapi.json
```

Expected result is `200`. A no-Ray server may return `503` from `/healthz` even
when its API is usable, so the client launcher checks `/openapi.json` instead.

One TCP port is not one GPU allocation. Starting a second port on GPUs already
holding a full pi0.5 worker can cause OOM. Coordinate the GPU set, not only the
port number. `scripts/remote/run_client.sh` intentionally sets
`JAX_PLATFORMS=cpu` and does not allocate or change server GPUs.

## 5. Validate data without training

### Contact-centered frame windows

The default training population is the inclusive window around target-object
contact:

```text
start = max(0, first target-object contact frame - 100)
end   = min(total_frames - 1, last target-object contact frame + 100)
```

Contact comes from the Lance `contact` column, not
`trajectory_metadata.trajectory_info.object_move`. Build the cache once for a
new dataset:

```bash
./scripts/remote/run_client.sh scripts/tools/build_contact_windows.py \
  --lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance
```

The client automatically uses `<dataset>.contact_windows.json` afterward. The
resolved evidence and fallback status are written per row. Use
`--missing-contact-policy error` when a training run must reject unannotated
episodes.

### Mode4 physics inference

`scripts/eval/infer_mano_mode4.py` is the maintained MANO real-physics
inference entrypoint. It always initializes from source frame 0, reconstructs the model's
B output into absolute target DOFs, temporally ensembles absolute targets, and
executes them through MuJoCo's native position servo. The next policy state and
head/wrist images come from the integrated `MjData`; the object is never reset
to a later reference pose. Physics uses `dt=0.0025 s`, two steps per source
interval, and exactly `2*(T-1)` steps.

Mode4 defaults to fixed-shape `act_batch` requests. Closed-loop simulation
remains sequential; repeated observations only fill the server's fixed sharded
batch, and the evaluator consumes the first returned action. Timings include
HTTP completion and result materialization. The server must expose the matching
`act_batch` contract; failures are surfaced rather than silently falling back.
Each row result now records `timing.phase_seconds`, separating action requests,
query preparation, target processing, MuJoCo stepping, rendering/video, and
array finalization. `video_mode=full` is the canonical delivery setting;
`--video-mode none` preserves policy-observation rendering but skips the three
output video encoders for faster diagnostic sweeps.

For `mano_five_finger_contact_lift_v1`, training uses target-object Lance
contact-record presence and Mode4 uses target-object × MANO-keypoint MuJoCo
contact-pair presence. Mode4 does not query solved contact force or apply a
`0.01 N` threshold. Extended-state training and Mode4 both require the locked
norm SHA declared in `scripts/mano_state_contract.py`.

Use `scripts/remote/run_mode4_eval.sh` for every new checkpoint and row set.
This is the single operational interface; do not copy `deliver_mode4_eval.sh`.
It refuses an existing output directory unless `--overwrite-output` is supplied,
creates a clean replacement when that flag is explicit, writes
`effective_config.json`, and writes `run.completed.json` or `run.failed.json`.
Without an output option it creates a unique run below the formal checkout's
`results/inference/`. Prefer `--run-name NAME` for named comparisons;
`--output-dir PATH` remains an explicit override and cannot be combined with
`--run-name`.

The canonical Mode4 frame contract is `--frame-window contact`. The launcher
loads `<dataset-without-.lance>.contact_ctx100_error_v1.json` unless an explicit
manifest is supplied, initializes hand/object physics at the absolute window
start, and resets lift there. `dataset_reference.mp4` always preserves the full
source demonstration. `mode4_physics_vs_dataset_head.mp4`, the wrist comparison,
rollout arrays, and physical metrics cover only the synchronized contact window
and retain absolute source-frame labels. `--frame-window full` is a separate
out-of-support stress test; do not obtain a contact result by trimming a full
closed-loop video after the fact.

When using an existing endpoint, declare the backend/model commits reported by
the server owner. The launcher records them as operator-declared provenance; a
client cannot prove the source of a process it did not start:

```bash
ROWS=924,960
NORM_ROWS=$(seq -s, 810 994)
./scripts/remote/run_mode4_eval.sh \
  --model-path <checkpoint> \
  --dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --rows "${ROWS}" --normalization-rows "${NORM_ROWS}" \
  --norm-stats-dir /vePFS-Mindverse/user/intern/wenxi/results/training/<run>/norm \
  --run-name <run>-mode4 \
  --owner-id <owner> --chunk-stride 5 --temporal-decay 0.4 \
  --act-mode batch --act-batch-size 4 --video-mode full \
  --base-url http://127.0.0.1:30532 \
  --backend-commit <mint-commit> --model-commit <openpi-commit>
```

For a dedicated server, replace `--base-url` and the two declared commits with:

```bash
./scripts/remote/run_mode4_eval.sh \
  --model-path <checkpoint> --dataset <dataset> --rows 924,960 \
  --normalization-rows "$(seq -s, 810 994)" --norm-stats-dir <norm> \
  --run-name <run>-mode4 --owner-id <owner> \
  --own-server --server-runtime-root <checkpoint-bearing-server-root> \
  --server-port 30532 --server-gpus 4,5,6,7
```

For a same-checkpoint parameter sweep, keep the owned server alive after the
first successful evaluation:

```bash
# First evaluation: compile and hand the server off.
./scripts/remote/run_mode4_eval.sh ... --own-server \
  --server-runtime-root <checkpoint-bearing-server-root> \
  --server-port 30532 --server-gpus 4,5,6,7 --keep-server

# Later evaluations: reuse the same server and compiled action session.
./scripts/remote/run_mode4_eval.sh ... \
  --reuse-server-info <first-output>/server.keepalive.json

# End the lifecycle using the marker from the first output root.
./scripts/remote/stop_owned_mode4_server.sh <first-output>/server.keepalive.json
```

The JIT executable is owned by the action-session policy instance. Keeping only
the uvicorn process alive is insufficient: deleting the session and creating a
new one recompiles the identical `jit(sample_fn)` when persistent serialization
is disabled. `--keep-server` therefore retains both the owned server and the
successful evaluator's action session. `--reuse-server-info` verifies the live
PID and marker, supplies the endpoint, source commits, owner, checkpoint/model,
batch shape, and retained session ID, and leaves lifecycle ownership with the
first run. The retained policy RNG stream continues across evaluations; these
runs avoid compilation but are not independently reseeded or deterministic
sample pairs. An unrelated existing endpoint is never stopped by the launcher.

Run either command once with `--print-config` before allocating the server or
creating outputs. Source worktrees must be clean by default;
`--allow-dirty-sources` is an explicit diagnostic override and the dirty state
is retained in `effective_config.json`.

Dedicated mode starts and stops exactly one server and reads the MINT/OpenPI
commits from the worktrees it launches. For a parameter sweep on the same
checkpoint, `--keep-server` hands the owned server and compiled action session
to the operator after a successful run; the launcher writes
`server.keepalive.json` and uses `nohup`. Subsequent runs must use
`--reuse-server-info` so they submit to that retained session rather than
creating a new one. Stop only that explicitly owned lifecycle with:

```bash
./scripts/remote/stop_owned_mode4_server.sh <output>/server.keepalive.json
```

The stop helper verifies the marker, PID command line, and endpoint, deletes
the retained action session, and then sends a graceful server signal. Persistent
JAX executable serialization is disabled
by default because this pi0.5 runtime cannot serialize the multi-GB executable;
`--enable-jax-persistent-cache` is an intentional opt-in after validation. The
historical `deliver_mode4_eval.sh` remains a delivery-specific record, not a
template for new evaluations.

Connect to the GPU host and run:

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example

./scripts/remote/run_client.sh scripts/train/train_cube1_01_compare.py \
  --lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --target-lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --row-indices 810,811 \
  --contact-window-manifest /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.contact_ctx100_error_v1.json \
  --missing-contact-policy error \
  --action-source urdf_target_absolute \
  --extended-state \
  --norm-stats-dir /vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726 \
  --batch-size 4 \
  --seed 42 \
  --state-noise-std 0 \
  --save-path coworker_cube1_dry_run \
  --output-json /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/training/coworker_cube1_dry_run/result.json \
  --dry-run
```

Dry-run mode does not contact MINT. It verifies Lance access, selected rows,
state/action dimensions, normalization, image preprocessing, and batch shape.

### Mode3 historical kinematic diagnostic

`scripts/eval/infer_mano_mode3.py` restores the historical `sim_no_smooth`
kinematic contract for current B-exact 32D v1 checkpoints. It queries
non-overlapping 10-frame chunks through a fixed batch-4 request, reconstructs
B targets from the qpos at that query, and advances the predicted hand without
temporal ensembling. At every frame it forces the target object to the
reference trajectory pose, calls `mj_forward`, and renders head and wrist
observations. It never calls `mj_step`; its output is a visual/state diagnostic,
not a real-physics grasp result. Mode4 remains the maintained evaluator for
sim-owned object motion and native position-servo dynamics.

```bash
./scripts/remote/run_client.sh scripts/eval/infer_mano_mode3.py \
  --model-path <checkpoint> --owner-id <owner> \
  --lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --row-indices 810,811 --normalization-row-indices 810,811 \
  --contact-window-manifest "${MANIFEST}" --missing-contact-policy error \
  --extended-state --norm-stats-dir "${NORM}"
```

The required extended state is `[predicted_hand_qpos(26), contact(index,thumb,
ring,middle,pinky), reference_object_z[t]-reference_object_z[0]]`. Contacts are
target-object × MANO-keypoint pair presence after the frame's forward pass;
palm is checked while resolving the 16 keypoint geoms but is not emitted. The
locked norm SHA is checked before it is loaded or queried. Add `--backend-commit`
and `--model-commit` when those paired source SHAs are available; result metadata
also records the client commit.

The current pi0.5 setup expects:

- base model `openpi/pi05-action-lora-r16-finetune`;
- 32-dimensional state and action vectors;
- action horizon 10;
- LoRA rank 16, controlled by the server;
- two image observations: head and wrist cameras.

Do not silently pad a dataset with different action semantics to 32 dimensions.
Dimension equality is necessary but not sufficient: ordering and units must
also match the simulator and inference client.

## 6. Start a fine-tuning run

Use one parameterized command for the clean and StateAug arms so every setting
except state noise remains identical:

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example

ARM=clean                 # use stateaug005 for the augmented arm
STATE_NOISE=0             # use 0.05 for stateaug005
RUN_NAME="gesture03_32d_${ARM}_30k"
ROWS=$(seq -s, 810 994)
MANIFEST=/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.contact_ctx100_error_v1.json
NORM=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726

MINT_BASE_URL=http://127.0.0.1:30532 \
./scripts/remote/run_client.sh scripts/train/train_cube1_01_compare.py \
  --model openpi/pi05-action-lora-r16-finetune \
  --lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --target-lance-dataset /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
  --row-indices "${ROWS}" \
  --contact-window-manifest "${MANIFEST}" \
  --missing-contact-policy error \
  --action-source urdf_target_absolute \
  --language-conditioning gesture \
  --extended-state --norm-stats-dir "${NORM}" \
  --sampling-strategy coverage --slate-size 16 --coverage-anchors-per-row 8 \
  --steps 30000 --batch-size 8 --seed 42 --augmentation-seed 43 \
  --state-noise-std "${STATE_NOISE}" \
  --save-path "${RUN_NAME}" \
  --output-json "/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/training/${RUN_NAME}/result.json"
```

For the A-LoRA discrete-state profile, state augmentation is injected **after
normalization and before prompt tokenization**, so the state tokens the model
receives are actually augmented. The v1 extended-state path refuses computed or
wrong-hash norm fallback.

The augmentation changes only normalized state and its derived encoded prompt;
images, action labels, task text, and normalization statistics remain clean.
Inference stays unaugmented. The
client uses independent sample and augmentation RNG streams: `--seed 42` selects
samples and `--augmentation-seed 43` generates fresh state noise.

The client training loop also uses a bounded transformed-datum cache and an
ordered batch prefetch queue by default:

- `--datum-cache-size 4096` bounds transformed frame memory; use `0` to disable
  caching.
- `--prefetch-batches 2` builds the next two batches while the server executes
  the current request; use `0` for synchronous debugging.

Prefetch owns both RNG streams in one producer, so enabling it preserves the
sample and augmentation sequence. The output JSON records cache statistics and
aggregate client build/request timings. For checkpoint workflows, the resume
barrier remains unbounded by default; set
`--checkpoint-resume-timeout-seconds` to make a failed orchestrator surface as
an error instead of waiting forever. The launcher exports `JAX_PLATFORMS=cpu`.
That is intentional: the client does CPU preprocessing and HTTP I/O, while the
MINT server owns the assigned GPUs.

## 7. Use a different dataset

For a basic full-dataset smoke run, use the generic client:

```bash
./scripts/remote/run_client.sh scripts/train/openpi_vla_smoke_lance_base.py \
  --lance-dataset /path/on/pfs/my_dataset.lance \
  --steps 4 \
  --batch-size 4 \
  --seed 42 \
  --output-json /vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example/results/training/my_smoke/result.json
```

For a real experiment, add a dedicated entry point under `scripts/train/` that
imports the shared preprocessing/API helper and makes dataset selection,
normalization population, augmentation, seed, checkpoint name, and output path
explicit. Commit that entry point before the long run.

## 8. Outputs and provenance

Every run should preserve:

- Git commit hash of this client repository;
- MINT base URL and server owner/allocation;
- dataset path and selected row indices;
- base model, steps, batch size, seed, and augmentation values;
- loss log and elapsed time;
- `save_weights_for_sampler` response, especially the `mint://...` URI;
- later evaluation metrics and video paths.

`results/` is ignored by Git because videos, arrays, and logs are generated
artifacts, but it is the canonical filesystem home for this client's outputs.
Mode3/Mode4 default to `results/inference/<mode>_<UTC>_<pid>`; use
`run_mode4_eval.sh --run-name NAME` for a readable stable name. Training and
smoke metadata should use `results/training/`. User-selected behavioral
references belong in `results/baselines/<baseline-name>` with a manifest that
records source checkpoint, inference protocol, hashes, and interpretation
boundary. Explicit `--output-dir` remains available for intentional overrides.

## 9. Troubleshooting boundaries

- `openapi.json` is not `200`: confirm the allocated port; do not start or kill
  a shared server without the owner.
- GPU utilization is low but memory is high: the model is loaded and waiting
  between requests; those GPUs are not free.
- `ModuleNotFoundError`: use `scripts/remote/run_client.sh`; it assembles the
  shared MINT, Lance, OpenPI, GPU-runtime, and CPU-runtime paths.
- First step is slow: model load and XLA compilation dominate the cold start.
- HTTP returns a failed future: inspect the full error payload and server log;
  do not treat a missing loss as a successful step.
- State/action shape matches but rollout is wrong: verify dimension ordering,
  units, normalization population, image/state timestamp alignment, and action
  interpretation before increasing training steps.

`client_train_test.sh` is a developer-provided external script. It may be used
as upstream reference, but this project does not claim authorship or maintain a
fork of it. `Tutorial.md` contains deeper server/API history; it is not required
for the standard coworker client workflow.
