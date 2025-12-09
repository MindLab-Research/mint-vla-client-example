# tinker-server

## Code Synchronization

Code sync handled by background `unison` process. **NEVER** manually sync.

## Quick Start

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

For server deployment and cluster management, use the `deployment-maintenance` skill.

## Fixed Issues

| Issue | Fix | Location |
|-------|-----|----------|
| max_tokens ignored | Override `generate()` to respect user param | `verl_inference.py:138-213` |
| EOS not detected | Add `stop_token_ids=[151645, 151643]` | `verl_inference.py:283`, `sampling.py:72-76` |
| Slow weight sync (60s) | Hot LoRA reload via shared engine (0.68s) | `session_manager.py:103-184`, `verl_inference.py:77-132` |

## Architecture

```
Client Machine              API Server (volcano)           GPU Workers
──────────────              ────────────────────           ───────────
tinker-cookbook  ──HTTP──>  tinker-server:8000  ──Ray──>  TrainingWorker
                                   |                       vLLM Engine
                            SSH tunnel (8000)
```

**Data transfer:** Weights via Ray object store, not file paths.

### Inference Modes

- **Named Sessions:** `create_sampling_session(model_path=...)` - dedicated engine (~60s init)
- **Ephemeral Sessions:** `save_weights_and_get_sampling_client()` - hot-reload (~0.7s after first)

### vLLM Actor

Detached Ray actor surviving server restarts. First start ~80s, subsequent ~2s.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/healthz` | GET | Health check |
| `/api/v1/vllm_status` | GET | vLLM actor status |
| `/api/v1/kill_vllm` | POST | Kill vLLM actor |
| `/api/v1/create_session` | POST | Create session |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Submit async sample |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending, 200=ready) |

## Cookbook Tests

```bash
cd /home/yiwen/tinker_project/tinker-cookbook

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

Note: `renderer_name="qwen3_instruct"` bypasses model lookup.
