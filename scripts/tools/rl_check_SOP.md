# RL Check SOP

End-to-end RL validation for mint-server after code changes that affect
actor naming, placement, training, or inference flow.

## Prerequisites

- `mindlab-toolkit` installed (provides `import mint`)
- PFS mounted (for model weights)
- Network access to the target mint-server

## Tool

```bash
/vePFS-Mindverse/share/mint/tool/rl_check.py
```

## Environment

| Variable | Purpose | Dev | Prod |
|----------|---------|-----|------|
| `MINT_BASE_URL` | Server URL | `http://localhost:8000` | `https://mint.macaron.xin` |
| `MINT_API_KEY` | Auth key | not needed (dev=no-auth) | required |
| `MINT_TEST_TIMEOUT_S` | Per-request timeout | 600 | 3600+ |

## Quick Start

### Dev server (issue-scoped)

```bash
# 1. Deploy code to dev (see mint-dev skill)
MINT_CODE_ROOT=/vePFS-Mindverse/share/<your-path>/mint-server \
MINT_RAY_NAMESPACE=mint_<you>_pr736 \
MINT_PORT=10416 \
scripts/start_dev_server.sh

# 2. Run RL check
MINT_BASE_URL=http://localhost:10416 \
python /vePFS-Mindverse/share/mint/tool/rl_check.py \
  --model Qwen/Qwen3-0.6B \
  --steps 10 \
  --group-size 4
```

### Production

```bash
MINT_BASE_URL=https://mint.macaron.xin \
MINT_API_KEY=<key> \
python /vePFS-Mindverse/share/mint/tool/rl_check.py \
  --model Qwen/Qwen3-0.6B \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --steps 10
```

## What It Does

Per model, per step:

1. **Sample** — generate countdown completions (group_size samples)
2. **Reward** — 1.0 if correct countdown sequence, else 0.0
3. **forward_backward** — importance_sampling loss on generated tokens
4. **optim_step** — Adam update
5. **save_weights** — every 5 steps, save LoRA weights and refresh sampling client

## Output

```
/tmp/rl_check/<timestamp>/
  summary.json          # all models, status, reward trajectory
  Qwen_Qwen3-0.6B.json  # per-model detail
```

## Pass Criteria

- All steps complete without timeout or server error
- `num_datums > 0` for every step (training actually happened)
- `final_reward` is non-zero (model can produce correct countdown)

## Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | required | Repeatable for multiple models |
| `--steps` | 10 | RL steps per model |
| `--group-size` | 4 | Samples per step |
| `--lr` | 1e-4 | Learning rate |
| `--max-tokens` | 64 | Max generation length |
| `--lora-rank` | 16 | LoRA rank (use 64 for K2) |
| `--timeout-s` | 600 (env: `MINT_TEST_TIMEOUT_S`) | Per-request timeout |
| `--output-dir` | `/tmp/rl_check/<ts>` | Results directory |

## After Actor Naming Changes

When testing changes to actor naming (like PR #736):

1. Start with a clean namespace — old actors with old names become orphans
2. Kill stale actors in the old namespace before starting:
   ```bash
   HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
   export MINT_RAY_CLIENT_ADDRESS="ray://${HEAD_IP}:10001"
   export MINT_RAY_NAMESPACE="mint_<you>_pr736"
   /vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python - <<'PY'
   import os, ray
   ray.init(address=os.environ["MINT_RAY_CLIENT_ADDRESS"],
            namespace=os.environ["MINT_RAY_NAMESPACE"],
            ignore_reinit_error=True, log_to_driver=False)
   for actor in ray.util.list_named_actors(all_namespaces=True):
       ns = str(actor.get("namespace") or "")
       name = str(actor.get("name") or "")
       if ns == os.environ["MINT_RAY_NAMESPACE"] and name:
           ray.kill(ray.get_actor(name, namespace=ns), no_restart=True)
           print(f"killed {name} in {ns}")
   ray.shutdown()
   PY
   ```
3. Start issue-scoped dev server
4. Run RL check
5. Verify reward trajectory in summary.json
6. Clean up namespace after done
