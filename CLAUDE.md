# tinker-server

## Tinker API Reference

When needing details about the official Tinker API (types, methods, loss functions, data formats), read `tinker_official_reference.txt` in the project root. This contains complete SDK documentation and type definitions.

## Code Synchronization

Code sync handled by background `unison` process. **NEVER** manually sync.

## Remote Commands

**NEVER** run `ray` or `volc` commands locally. Use the `deployment-maintenance` skill for remote server operations.

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
Local Machine               API Server (volcano)           GPU Workers (Ray)
─────────────               ────────────────────           ─────────────────
tinker-cookbook  ──HTTP──>  tinker-server:8000  ──Ray──>  MegatronWorker
(Python 3.11+)         ↑                                   vLLM Engine
                       │
                 SSH tunnel (localhost:8000 → volcano:8000)
```

**Key points:**
- **tinker-cookbook**: Runs LOCALLY on your workstation. Requires Python 3.11+ (for `chz` package).
- **tinker-server**: Runs on volcano API server. Receives HTTP requests, dispatches to Ray workers.
- **GPU Workers**: Run on Ray cluster nodes. Execute training (Megatron) and inference (vLLM).
- **Data transfer**: Weights via Ray object store, not file paths.

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

## Cookbook Tests (Run Locally)

tinker-cookbook runs on your LOCAL machine, not on the server. Ensure SSH tunnel is active.

```bash
# Run from LOCAL machine (requires Python 3.11+, chz package)
cd /home/yiwen/tinker_project/tinker-cookbook

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

Note: `renderer_name="qwen3_instruct"` bypasses model lookup.

---

## Hall of Shame

Agent mistakes that wasted time. Read this before starting any task.

| Date | Mistake | What Should Have Been Done |
|------|---------|---------------------------|
| 2024-12-18 | Ran merge gate tests when user explicitly asked for K2 work | Focus on the requested task. Do not divert to "system health checks" unless specifically asked. |
| 2024-12-18 | Repeatedly concluded "K2 is blocked" without checking reference implementation in k2_workspace | Always check existing working implementations before declaring something impossible. |
| 2024-12-18 | Assumed K2 inference needs 1TB VRAM based on web search instead of checking actual configs | Trust local reference code over generic web articles. |
| 2024-12-23 | After K2 RL loop succeeded, ran Qwen2.5-7B cookbook instead of K2 12-hour training | User said "run K2 for 12 hours". Run K2, not a different model. Read the request literally. |
