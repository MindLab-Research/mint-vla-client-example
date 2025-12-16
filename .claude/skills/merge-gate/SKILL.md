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
| Qwen3-30B-A3B (MoE) | 4 (TP=1, DP=4) | 8 (TP=4, EP=2) | 12 |

### Full Merge Gate Procedure

The merge gate runs in **two phases** with different cluster configurations:

**Phase 1: Functional Tests (16 GPUs - two 8-GPU workers)**
1. Start cluster with 16 GPUs
2. Run ALL functional tests: Dense SFT/RL/DPO/API, MoE SFT, Stress
3. **Do NOT kill actors between tests** - Dense and MoE coexist
4. Validates normal concurrent operation

**Phase 2: LRU Eviction Test (12 GPUs - one 8-GPU + one 4-GPU)**
1. **Tear down** Phase 1 cluster completely
2. Start new cluster with only 12 GPUs
3. Run LRU eviction test (Dense -> MoE -> Dense switches)
4. Validates graceful actor replacement when resources insufficient

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
if gpu_total < 16:
    print(f"WARNING: Full concurrent tests need 16 GPUs, have {gpu_total:.0f}")
    if gpu_total >= 12:
        print("  Can run LRU eviction test (12 GPU config)")
elif gpu_avail < 14:
    print(f"WARNING: {gpu_total - gpu_avail:.0f} GPUs in use by actors")
    print("  Actors will be reused - this is expected behavior")
PYEOF'
```

### 3. Add Workers for Phase 1 (via volcano-cluster skill)

Phase 1 requires 16 GPUs (two 8-GPU workers):

```bash
# Check current workers
volc ml_task list --output json | jq '.[] | select(.Status=="Running") | {name: .Name}'

# If only one worker running, add another 8-GPU worker
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

### Phase 3: Stress Test

| Test | Description | Pass Criteria | Duration |
|------|-------------|---------------|----------|
| **stress** | 5 concurrent clients with different configs | All complete without deadlock | 5 min |

Stress test configurations:
- Client 1: Dense model, SFT, rank=16
- Client 2: Dense model, RL, rank=32
- Client 3: Dense model, SFT, rank=64
- Client 4: Dense model, DPO, rank=32
- Client 5: Dense model, SFT, rank=16

---

## Running the Tests

### Complete Merge Gate (Two Phases)

**Phase 1: Functional Tests (16 GPUs)**

```bash
# Ensure 16 GPUs available (two 8-GPU workers)
cd /home/yiwen/tinker_project/tinker-server

# Run all functional tests (do NOT kill actors between)
TINKER_BASE_URL=http://localhost:8000 \
python -m pytest .claude/skills/merge-gate/tests/ \
    --ignore=.claude/skills/merge-gate/tests/test_stress.py::TestStress::test_mixed_model_lru_eviction \
    -v --tb=short
```

**Phase 2: LRU Eviction Test (12 GPUs)**

```bash
# 1. Tear down Phase 1 cluster via volcano-cluster skill
# 2. Start new cluster with 12 GPUs (one 8-GPU + one 4-GPU worker)
volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-dev-worker.yaml --output json
volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-dev-worker-4gpu.yaml --output json

# 3. Wait for workers to join Ray cluster

# 4. Run LRU eviction test
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

---

## Interpreting Results

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
| MoE tests fail to start | Need 12 GPUs, add workers via volcano-cluster |
| MoE tests OOM | Kill vLLM, restart with fresh actor |
| Stress test deadlock | Session isolation issue, check logs |
| Loss not decreasing | Optimizer state bug, check Issue 6c fix |
