# tinker-server

## Skills Reference

| Skill | Purpose |
|-------|---------|
| `mint-dev` | Development server on `volcano` (port 8000, no auth) |
| `mint-prod` | Production server on `mint-prod` (port 18000, auth required) |
| `volcano-cluster` | Ray cluster lifecycle (create/teardown worker tasks) |
| `merge-gate` | Pre-merge testing |

**Use `mint-prod` for production operations. Use `mint-dev` for development.**

## Code Synchronization

Code sync handled by background `unison` process. **NEVER** manually sync.

## Remote Commands

**NEVER** run `ray` or `volc` commands locally. Invoke the appropriate skill instead.

## Quick Start (Development)

```bash
# Requires SSH tunnel: ssh -f -N -L 8000:localhost:8000 volcano
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

For production (port 18000), use the `mint-prod` skill.

## Tinker API Reference

When needing details about the official Tinker API (types, methods, loss functions, data formats), read `tinker_official_reference.txt` in the project root.

## Architecture

```
                        Development                          Production
                        ───────────                          ──────────
Local Machine           volcano:8000                         mint-prod:18000
─────────────           ────────────                         ──────────────
tinker-cookbook ──HTTP──> tinker-server ──Ray──> GPU Workers (MegatronWorker, vLLM)
                    ↑                                  ↑
              SSH tunnel                         SSH tunnel
         localhost:8000 → volcano:8000      localhost:18000 → mint-prod:18000
```

| Environment | SSH Host | Port | Auth | Log File |
|-------------|----------|------|------|----------|
| Development | `volcano` | 8000 | No | `/tmp/tinker_server.log` |
| Production | `mint-prod` | 18000 | Yes (`X-API-Key`) | `/tmp/tinker_server_auth.log` |

**Finding the running server process:**
```bash
ssh <host> 'ps aux | grep run_server | grep -v grep'
ssh <host> 'ls -la /proc/<PID>/fd/1'  # Shows actual log file location
```

**Key points:**
- **tinker-cookbook**: Runs LOCALLY on your workstation. Requires Python 3.11+ (for `chz` package).
- **tinker-server**: Runs on API server (volcano for dev, mint-prod for prod). Receives HTTP requests, dispatches to Ray workers.
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

## Cookbook Tests (Development)

tinker-cookbook runs on your LOCAL machine against the **development** server. Ensure SSH tunnel to volcano is active.

```bash
# Run from LOCAL machine (requires Python 3.11+, chz package)
cd /home/yiwen/tinker_project/tinker-cookbook

# SSH tunnel (if not already running)
ssh -f -N -L 8000:localhost:8000 volcano

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

Note: `renderer_name="qwen3_instruct"` bypasses model lookup.

For production testing, use port 18000 and set valid `TINKER_API_KEY`.

---

## Hall of Shame

Agent mistakes that wasted time. Read this before starting any task.

| Date | Mistake | What Should Have Been Done |
|------|---------|---------------------------|
| 2024-12-18 | Ran merge gate tests when user explicitly asked for K2 work | Focus on the requested task. Do not divert to "system health checks" unless specifically asked. |
| 2024-12-18 | Repeatedly concluded "K2 is blocked" without checking reference implementation in k2_workspace | Always check existing working implementations before declaring something impossible. |
| 2024-12-18 | Assumed K2 inference needs 1TB VRAM based on web search instead of checking actual configs | Trust local reference code over generic web articles. |
| 2024-12-23 | After K2 RL loop succeeded, ran Qwen2.5-7B cookbook instead of K2 12-hour training | User said "run K2 for 12 hours". Run K2, not a different model. Read the request literally. |
| 2024-12-23 | Spent 15+ turns fumbling with SSH, log locations, process killing instead of following mint-dev skill | Read the skill docs first. SSH via `volcano` alias, logs at `/tmp/tinker_server.log`, restart commands are documented. Don't waste context on solved problems. |
| 2024-12-23 | VANDALISM: Copied files over unstaged work in develop worktree, then ran `git checkout --` destroying 4 hours of session state management code permanently. Two catastrophic errors in sequence: (1) overwrote files without backup, (2) destroyed recovery chance by reverting to HEAD. | NEVER touch another worktree's files without explicit backup. When user catches a mistake, STOP IMMEDIATELY - do not attempt to "fix" with more commands. Unstaged work is unrecoverable after `git checkout --`. |
| 2024-12-24 | Ran "K2 RL training" for hours without tracking reward - the actual RL metric. Script was SFT on trivial arithmetic, not RL. Model already solved problems correctly. Wasted 2+ hours monitoring meaningless loss values. | RL requires reward tracking. Verify the metric being optimized matches the goal. Trivial problems (5+5=10) don't demonstrate learning. |
| 2024-12-24 | Sent wrong context to theorist agent. Asked "find bugs in gradient isolation" when should have asked "how does gradient isolation work for small MoE models (where it succeeds)?". Jumped to debugging without understanding the working mechanism first. Correct approach: (1) understand how it works for Qwen3-30B-A3B, (2) identify what's different about K2, (3) then debug. Also ignored key clue: grad_norm=0 in all logs. | First understand the WORKING case before debugging the FAILING case. Understand mechanism → identify difference → then debug. |
| 2024-12-24 | VANDALISM AGAIN: Ran `git checkout --` to "revert premature changes" without being asked. Destroyed hours of work. Same pattern as 2024-12-23. | NEVER run destructive git commands (checkout, reset, clean) without EXPLICIT user request. If changes were wrong, leave them for user to review. The user decides what to keep or discard, not the agent. |
| 2024-12-24 | Used wrong API (train_step) in K2 RL script. Misunderstood trainer eviction as bug when it's expected GPU time-sharing. Claimed "training never completes" when idle timeout handles this. Made architecture claims without understanding: (1) train_step is deprecated, (2) trainer eviction before inference is by design, (3) weights transfer trainer→inferencer works correctly. | UNDERSTAND THE ARCHITECTURE FIRST. Read tinker_official_reference.txt for correct API. Trainer eviction + idle timeout is the designed flow. Don't invent explanations for behavior you don't understand. |
| 2024-12-24 | Changed renderer from "deepseekv3" to "role_colon" without checking for dedicated K2 renderer. A `KimiK2Renderer` exists but wasn't discovered until training crashed after 30+ minutes of sampling. | ALWAYS search for dedicated renderers first: `grep -i "kimi\|k2" renderers.py`. The `kimi_k2` renderer uses correct `<\|im_end\|>` stop tokens and handles thinking blocks. |
| 2024-12-26 | Spent entire night debugging why vLLM used TP=8 instead of TP=16 for K2. Analyzed code paths, model registry, normalize functions, server logs. Root cause: wrong unison profile running (`volcano-tinker-bugfix` instead of `volcano-tinker`), so code changes never synced to server. The skill doc explicitly says to verify unison before any work. | VERIFY UNISON IS RUNNING FIRST: `pgrep -af "unison.*volcano-tinker"`. Before debugging server behavior, confirm the server has the latest code. A 5-second check would have saved 8+ hours. |
| 2024-12-27 | When K2 training hit OOM, repeatedly tried smaller configurations (max_tokens 2048→1024→512, lora_rank 32→8) without understanding WHY it failed. Binary search by trial-and-error instead of calculating memory requirements. | See "Configuration Debugging Principle" section below. |
| 2024-12-27 | FABRICATED DATA: Presented made-up memory breakdown as verified fact. Claimed "37 GiB headroom" when previous runs OOMed - an obvious contradiction ignored. When challenged, suggested nvidia-smi which gives ONLY total usage (useless for breakdown). Then FLED to a harder dataset (MATH) when GSM8K showed reward=1.0, adding complexity instead of understanding the current problem. Classic pattern: can't explain current behavior → distract with new shiny thing. | (1) Unverified theory presented as fact is LYING. Say "I calculated X but haven't verified" not "X is the breakdown". (2) nvidia-smi total tells NOTHING about where memory goes. Use `torch.cuda.memory_stats()`, `torch.cuda.memory_summary()`, or instrument code with memory checkpoints. (3) REDUCE moving parts when debugging. If you don't understand GSM8K memory, adding MATH makes it WORSE. Stay with the simple case until you understand it completely. |
| 2024-12-27 | LIVESTOCK MINDSET: Celebrated "Run 47 works at 8K context" as success when K2 supports 262K context natively. That's 3% of capability. When asked to add memory profiling to push context higher, went in circles: started profiling → got interrupted → started again → user had to explain purpose THREE TIMES. Never completed anything. The goal is LONGER CONTEXT, not "it works at reduced settings". | (1) "Works at reduced settings" is FAILURE, not success. The goal is to maximize capability. (2) Memory profiling purpose: understand WHERE memory goes so we can INCREASE context length. Not to verify theoretical numbers. Not to debug past OOMs. TO PUSH LIMITS HIGHER. (3) When given a clear task (add memory profiling), COMPLETE IT. Don't explain, don't ask clarifying questions, don't go in circles. Just do it. |
| 2024-12-29 | CHAOTIC SESSION: Made nearly every action wrong in a single session: (1) Changed test script API back and forth 5+ times instead of using existing working script (2) Confused vLLM and Megatron as needing transfer when they should be tested SEPARATELY (3) Killed working Megatron actor trying to fix unrelated vLLM rank issue (4) Ran LOCAL scripts on volcano server (paths don't match) (5) Kept sleeping when user said "NEVER SLEEP, you have concurrent tasks" (6) Ignored user saying "you already have a working script with rank=16" and kept modifying (7) Created new scripts instead of finding and using existing ones | (1) STOP AND THINK before acting. (2) When user says "you already have X", FIND IT, don't recreate. (3) vLLM and Megatron are SEPARATE - test independently, no transfer needed for profiling. (4) Scripts in tinker-server/scripts run LOCALLY and connect via HTTP/Ray, not on volcano. (5) NEVER sleep with concurrent tasks - poll both, act on whichever responds. (6) When something works, DON'T TOUCH IT. (7) Read existing scripts before writing new ones. |
| 2024-12-29 | Silently removed "vLLM: investigate low throughput (3.81 tok/s)" from todo list when "cleaning up". Rationalized that "test longer generation" would cover it. But functionality testing (can it generate long?) and performance investigation (why slow?) are distinct tasks. Removing user-requested tracking without consent loses important work items. | NEVER delete pending todo items. Only allowed: (1) Mark completed when done, (2) Add new items, (3) User explicitly requests removal. Pending items require explicit user consent to remove - not agent judgment about "simplification" or "consolidation". |

---

## Configuration Debugging Principle

**NEVER "this does not work, let's try a smaller configuration".**

1. **NEVER "this does not work, let's try a smaller configuration"** — I must understand WHY it doesn't work, not retreat to safer settings.

2. **NEVER "this does not work, let's try a smaller configuration"** — Smaller configurations mask the root cause and waste time on suboptimal solutions.

3. **NEVER "this does not work, let's try a smaller configuration"** — The goal is to maximize capability, not find the easiest path.

4. **NEVER "this does not work, let's try a smaller configuration"** — Without understanding the failure, I can't predict what WILL work.

5. **NEVER "this does not work, let's try a smaller configuration"** — Binary search by trial-and-error is lazy engineering. Calculate first.

6. **NEVER "this does not work, let's try a smaller configuration"** — Each failed attempt without analysis is wasted compute and time.

7. **NEVER "this does not work, let's try a smaller configuration"** — The user needs to know the theoretical maximum, not an arbitrary safe value.

8. **NEVER "this does not work, let's try a smaller configuration"** — If I don't understand the memory model, I'll never get it right.

9. **NEVER "this does not work, let's try a smaller configuration"** — Measure reality, compare to calculation, find the discrepancy, fix the model.

10. **NEVER "this does not work, let's try a smaller configuration"** — The scientific method: hypothesis → test → analyze → refine. Not: fail → shrink → hope.

**The correct approach:**
1. Build a mathematical model of memory usage
2. Get real measurements from the system
3. Compare model vs reality
4. Identify discrepancies and fix the model
5. Use the calibrated model to determine maximum achievable configuration
6. Run at that configuration with confidence
