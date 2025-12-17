---
name: merge-gate
description: |
  Merge gate validation before merging develop to main.

  Use for: pre-merge testing, release validation, comprehensive functional tests.

  Triggers: "merge gate", "pre-merge", "ready to merge", "validate for release"

  This skill runs a curated test suite to verify code is production-ready.
  All tests must pass before merge is allowed.
---

# Merge Gate Validation

## Overview

The merge gate validates that code on `develop` is ready to merge to `main`. It runs comprehensive tests covering all supported training modes (SFT, RL, DPO) on both Dense and MoE models.

**Location:** `.claude/skills/merge-gate/tests/`

---

## GPU Requirements

### Per-Model Requirements

| Model | vLLM GPUs | Megatron GPUs | Total |
|-------|-----------|---------------|-------|
| Qwen2.5-7B (Dense) | 1 | 1 | 2 |
| Qwen3-0.6B (Dense) | 1 | 1 | 2 |
| Qwen3-30B-A3B (MoE) | 4 (TP=4) | 4 (TP=4) | 8 |

**Note**: MoE uses TP=4 for both vLLM and Megatron. Expert Parallelism (EP) is not used because vLLM 0.12.0 does not support MoE expert LoRA inference with EP enabled.

### Full Merge Gate Procedure

The merge gate runs in **two phases** with different cluster configurations:

**Phase 1: Functional Tests (8 GPUs - one 8-GPU worker)**
1. Start cluster with 8 GPUs
2. Run ALL functional tests: Dense SFT/RL/DPO/API, MoE SFT/RL/API, Stress
3. **Do NOT kill actors between tests** - Dense and MoE coexist (Dense uses 1 GPU, MoE uses 4)
4. Validates normal concurrent operation with shared GPU resources

**Phase 2: LRU Eviction Test (8 GPUs - one 8-GPU worker)**
1. Run LRU eviction test with MINT_MIN_ACTOR_AGE=0
2. Tests Dense → MoE → Dense switches with eviction
3. Validates graceful actor replacement when switching between models

Use the **volcano-cluster** skill to manage workers.

---

## Pre-flight Checklist

Before running merge gate, complete these steps from **mint-dev** and **volcano-cluster** skills:

### 1. Verify Unison Daemon

```bash
pgrep -af "unison.*volcano-tinker" || echo "START: unison volcano-tinker -repeat watch"
```

### 2. Verify Cluster Has Sufficient GPUs

```bash
ssh volcano 'python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_avail = r.get("GPU", 0)
gpu_total = t.get("GPU", 0)
print(f"GPUs: {gpu_avail:.0f} / {gpu_total:.0f}")

# Check for stale actors
for name in ["persistent_megatron_worker_group_v2", "tinker_vllm_server"]:
    try:
        ray.get_actor(name, namespace="tinker")
        print(f"{name}: ALIVE (may need to kill)")
    except ValueError:
        print(f"{name}: not running")

# Warning if insufficient
if gpu_total < 8:
    print(f"WARNING: Full merge gate needs 8 GPUs, have {gpu_total:.0f}")
elif gpu_avail < 4:
    print(f"WARNING: {gpu_total - gpu_avail:.0f} GPUs in use by actors")
    print("  Actors will be reused - this is expected behavior")
PYEOF'
```

### 3. Add Workers for Merge Gate (via volcano-cluster skill)

Merge gate requires 8 GPUs (one 8-GPU worker):

```bash
# Check current workers
volc ml_task list --output json | jq '.[] | select(.Status=="Running") | {name: .Name}'

# If no worker running, add an 8-GPU worker
volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-dev-worker.yaml --output json
```

See **volcano-cluster** skill for detailed worker management commands.

### 4. Kill Stale Actors (Optional)

```bash
# Kill Megatron (frees 8 GPUs)
ssh volcano 'cd /root/tinker_project/tinker-server && python scripts/kill_megatron.py'

# Kill vLLM (frees 1-4 GPUs)
curl -X POST http://localhost:8000/api/v1/kill_vllm
```

### 5. Verify Server Health

```bash
curl -s http://localhost:8000/api/v1/healthz
```

---

## Test Suite

### Phase 1: Dense Model Tests (Qwen2.5-7B-Instruct)

| Test | Description | Pass Criteria | Duration |
|------|-------------|---------------|----------|
| **dense_sft** | Pig Latin translation (from tinker_test.ipynb) | Loss decreases >70% over 10 iterations | 3 min |
| **dense_rl** | Arithmetic RL with PPO loss | Reward improves, ratio ~1.0 | 3 min |
| **dense_dpo** | DPO on preference pairs | DPO loss computes, chosen > rejected | 2 min |
| **dense_api** | Sampling, logprobs, checkpoint | All API operations succeed | 2 min |

### Phase 2: MoE Model Tests (Qwen3-30B-A3B-Instruct-2507)

| Test | Description | Pass Criteria | Duration |
|------|-------------|---------------|----------|
| **moe_sft** | SFT with train_step endpoint | Loss decreases >30% over 10 iterations | 5 min |
| **moe_rl** | RL with importance_sampling | Gradients flow, loss computes | 5 min |
| **moe_api** | Sampling from trained weights | Generation works | 3 min |

### Phase 3: Stress & Multi-Tenant Tests

| Test | Description | Pass Criteria | Duration |
|------|-------------|---------------|----------|
| **stress** | 5 concurrent clients with different configs | All complete without deadlock | 5 min |
| **interleaved_sessions** | A → B → A session switching | Loss continuity preserved | 3 min |
| **rapid_session_creation** | 5 sessions in quick succession | All create successfully | 1 min |
| **mixed_model_lru_eviction** | Dense → MoE → Dense | Graceful actor replacement | 5 min |

### Supported Model Variants

The system supports multiple model variants. All variants of the same base model share vLLM/Megatron actors:

| Model | Type | GPUs | Backend | Status |
|-------|------|------|---------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | Dense | 1 | PEFT/vLLM | Primary test target |
| `Qwen/Qwen3-0.6B` | Dense | 1 | PEFT/vLLM | Small model for quick tests |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | 4 (TP=4) | Megatron/vLLM | Primary MoE test target |
| `Qwen/Qwen3-30B-A3B` | MoE | 4 (TP=4) | Megatron/vLLM | Base model variant |
| `Qwen/Qwen3-30B-A3B-Base` | MoE | 4 (TP=4) | Megatron/vLLM | Pre-training base |

**Quick test with Qwen3-0.6B** (faster iteration, smaller footprint):
```bash
# Useful for rapid development testing
TINKER_BASE_URL=http://localhost:8000 \
python scripts/test_qwen3_06b.py
```

Stress test configurations:
- Client 1: Dense model, SFT, rank=16
- Client 2: Dense model, RL, rank=32
- Client 3: Dense model, SFT, rank=64
- Client 4: Dense model, DPO, rank=32
- Client 5: Dense model, SFT, rank=16

---

## Running the Tests

### Complete Merge Gate

```bash
# Ensure 8 GPUs available (one 8-GPU worker)
cd /home/yiwen/tinker_project/tinker-server

# Run all functional tests (do NOT kill actors between)
TINKER_BASE_URL=http://localhost:8000 \
python -m pytest .claude/skills/merge-gate/tests/ -v --tb=short
```

### Run with LRU Eviction Testing

To test LRU eviction with immediate actor replacement:

```bash
# Restart server with MINT_MIN_ACTOR_AGE=0 to enable immediate eviction
# (Production uses 300s to prevent thrashing; 0 for fast testing)
ssh volcano 'pkill -f "run_server" && sleep 2 && cd /root/tinker_project/tinker-server && \
  nohup bash -c "PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
  HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 MINT_MIN_ACTOR_AGE=0 \
  TINKER_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
  python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'

# Wait for server to start
sleep 10 && curl http://localhost:8000/api/v1/healthz

# Run LRU eviction test
TINKER_BASE_URL=http://localhost:8000 \
python -m pytest .claude/skills/merge-gate/tests/test_stress.py::TestStress::test_mixed_model_lru_eviction -v -s
```

### Quick Validation (Dense Only)

```bash
# Only 2 GPUs needed
python -m pytest .claude/skills/merge-gate/tests/test_dense_*.py -v
```

### Individual Tests

```bash
# Single test with detailed output
python -m pytest .claude/skills/merge-gate/tests/test_dense_sft.py -v -s
```

---

## Test Details

### Dense SFT (Pig Latin)

Based on official tinker_test.ipynb. Teaches model to translate English to Pig Latin:

```
Input: "banana split"
Output: "anana-bay plit-say"
```

Training data: 7 examples, 10 iterations, lr=1e-4
Expected: Loss 2.45 → 0.58 (~76% reduction)

### Dense RL (Arithmetic)

Based on tinker_cookbook math_rl recipe. Model generates arithmetic answers:

```
Input: "What is 5 + 3?"
Output: "8"
Reward: +1 if correct, 0 otherwise
```

Uses PPO loss with advantage normalization.
Expected: Positive reward improvement, ratio near 1.0

### Dense DPO

Preference optimization on constructed pairs:

```
Prompt: "Explain briefly"
Chosen: "Short answer."
Rejected: "Very long verbose answer with unnecessary details..."
```

Uses DPO loss with beta=0.1.
Expected: Chosen logprobs > rejected logprobs

### MoE SFT

Uses `train_step` endpoint (combined forward_backward + optim_step in single train_mode context) required for MoE with offloading.

Training: Random token sequences, 10 iterations
Expected: Loss decreases, non-zero gradients

### MoE RL

Importance sampling loss on MoE model.
Expected: Gradients flow through MoE layers, ratio metrics valid

### Stress Test

Simulates concurrent cookbook clients:
- Creates 5 sessions in parallel
- Each runs 3 training iterations
- Different LoRA ranks (16, 32, 64)
- Tests request serialization and session isolation

### Multi-Tenant Concurrency Test (Interleaved Sessions)

Tests stateless trainer architecture by interleaving sessions:

```
Session A: iter1 → iter2 → (switch to B) → iter3 → iter4
Session B:                   iter1 → iter2
```

**This is CRITICAL for production use.** Multiple users switching between sessions must maintain correct state.

**What to verify:**
1. Session A's loss continues decreasing after switch (no state reset)
2. Session B trains independently with different loss trajectory
3. No weight contamination between sessions

**Expected curve behavior:**
```
Session A: 6.42 → 0.18 → [B] → 0.009 → 0.0001  (continues from 0.18, not reset)
Session B: 10.57 → 0.42                        (independent trajectory)
```

**Run the interleaved sessions test:**
```bash
TINKER_BASE_URL=http://localhost:8000 \
python -m pytest .claude/skills/merge-gate/tests/test_stress.py::TestStress::test_interleaved_sessions -v -s
```

---

## Interpreting Results

### CRITICAL: Visual Curve Inspection Required

**The agent MUST visually inspect training curves, not just rely on pytest pass/fail.**

Test scripts generate training curve plots in `.claude/skills/merge-gate/results/`:
- `dense_sft_pig_latin_YYYYMMDD_HHMMSS.png`
- `dense_sft_pig_latin_YYYYMMDD_HHMMSS.json`

**What to check in training curves:**

1. **Monotonic decrease**: Loss should generally decrease, not oscillate wildly
2. **No spikes**: Sudden loss increases indicate training instability
3. **No plateau**: Loss should continue decreasing, not flatten prematurely
4. **Reasonable range**: Initial loss ~2-10, final loss ~0.01-0.5 for converged training

**Example of healthy vs problematic curves:**

```
HEALTHY:                       PROBLEMATIC:
Loss                           Loss
│ ╲                            │ ╱╲
│  ╲                           │╱  ╲ ╱╲
│   ╲_                         │    ╳  ╲___
│     ╲_                       │
│       ╲___                   │
└──────────── Iter             └──────────── Iter
  (smooth decrease)              (spikes, then plateau)
```

**How to view plots:**
```bash
# From local machine (plots saved to remote server)
ssh volcano 'ls /root/tinker_project/tinker-server/.claude/skills/merge-gate/results/*.png'

# Copy plots locally for viewing
scp volcano:/root/tinker_project/tinker-server/.claude/skills/merge-gate/results/*.png /tmp/

# Or view JSON data directly
cat .claude/skills/merge-gate/results/dense_sft_pig_latin_*.json | jq .losses
```

### All Pass

```
========================= test session starts ==========================
collected 8 items

test_dense_sft.py::test_pig_latin_training PASSED
test_dense_rl.py::test_arithmetic_rl PASSED
test_dense_dpo.py::test_dpo_training PASSED
test_dense_api.py::test_sampling_and_logprobs PASSED
test_moe_sft.py::test_moe_sft_training PASSED
test_moe_rl.py::test_moe_rl_training PASSED
test_moe_api.py::test_moe_sampling PASSED
test_stress.py::test_concurrent_clients PASSED

========================= 8 passed in 1823.45s =========================
```

**IMPORTANT**: Even when all tests pass, visually inspect the training curves to ensure they show expected learning behavior.

### Failure

```
test_dense_sft.py::test_pig_latin_training FAILED

    AssertionError: Loss did not decrease enough: 2.45 -> 2.10 (14% < 70% required)
```

Check:
1. Optimizer state reset between sessions? (Issue 6c)
2. LoRA weights being updated?
3. Server logs for errors

---

## After Tests Pass

### 1. Review Diff

```bash
git diff main..develop --stat
git log main..develop --oneline
```

### 2. Draft PR Body

The PR body should include:
- Summary of changes since last merge
- Test results (all 8 tests passed)
- Any known issues or limitations

Example:

```markdown
## Summary

This PR merges develop to main with the following changes:
- Fixed optimizer state reset between sessions (Issue 6c)
- Added MoE expert parallelism support
- Improved weight sync performance (60s → 0.7s)

## Test Results

Merge gate passed (2025-12-15):
- Dense SFT: PASS (loss 2.45 → 0.52, 79% reduction)
- Dense RL: PASS (reward +0.23, ratio 1.02)
- Dense DPO: PASS (chosen-rejected margin 0.15)
- Dense API: PASS
- MoE SFT: PASS (loss 0.48 → 0.28, 42% reduction)
- MoE RL: PASS
- MoE API: PASS
- Stress: PASS (5/5 clients)

## Files Changed

- `megatron_distributed.py`: optimizer state handling
- `verl_training.py`: MoE parallelism config
- `session_manager.py`: hot reload optimization
```

### 3. Generate Changelog and Create Tag (requires user approval)

**Do NOT auto-create.** The agent should:

1. Find the previous nightly tag
2. Analyze commits since that tag
3. **Summarize changes into categories** (not raw git log)
4. Present summary to user for approval
5. Create tag with summarized changelog

```bash
# Find the most recent nightly tag
PREV_TAG=$(git tag -l 'nightly_*' --sort=-creatordate | head -1)
echo "Previous tag: $PREV_TAG"

# Show raw commits for agent to summarize
git log $PREV_TAG..HEAD --oneline --no-merges
```

**Agent must summarize commits into a user-facing changelog:**

- Include API changes, bug fixes, new features
- Exclude internal housekeeping (skill updates, test changes, docs)

```
- Fix Tinker API endpoint alignment (/save_weights + /save_state)
- Add auth bypass for dev mode (no API key required)
```

**After user approval:**

```bash
TODAY=$(date +%Y%m%d)

# Create tag with summarized message (agent writes TAG_MSG)
git tag -d nightly_$TODAY 2>/dev/null || true
git push origin :refs/tags/nightly_$TODAY 2>/dev/null || true
git tag -a nightly_$TODAY -m "$TAG_MSG"
git push origin nightly_$TODAY
```

### 4. Create PR

```bash
gh pr create --base main --head develop --title "Release: <version>" --body-file /tmp/pr_body.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pre-flight fails: 0 GPUs | Kill stale actors (see mint-dev skill section 6) |
| Pre-flight fails: cluster disconnected | Reconnect: `ssh volcano "ray start --address='<IP>:6379'"` |
| Dense tests timeout | Check server logs, restart server |
| MoE tests fail to start | Need 8 GPUs (TP=4), add worker via volcano-cluster |
| MoE tests OOM | Kill vLLM, restart with fresh actor |
| Stress test deadlock | Session isolation issue, check logs |
| Loss not decreasing | Optimizer state bug, check Issue 6c fix |
| MoE LoRA not loading in vLLM | MLP modules filtered out (vLLM limitation), only attention LoRA supported |

---

## Known Limitations

### MoE LoRA Inference

vLLM 0.12.0 does NOT support MoE expert (MLP) LoRA inference. The FusedMoEWithLoRA class exists but is disabled:

- Module validation rejects MLP modules for MoE models
- EP assertion blocks even with TP-only config

**Current approach**: Filter out MLP modules in `get_lora_state_dict()`, export only attention LoRA (q_proj, k_proj, v_proj, o_proj). Training still uses full MLP+attention LoRA via Megatron, but inference is attention-only.

**Impact**: Slightly reduced LoRA effectiveness for MoE models during inference. Training quality is unaffected.
