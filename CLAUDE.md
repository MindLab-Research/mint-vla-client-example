# tinker-server

## Skills Reference

| Skill | Purpose |
|-------|---------|
| `mint-dev` | Development server on `mint-dev` (port 8000, no auth) |
| `mint-prod` | Production server on `mint-prod` (port 18000, auth required) |
| `volcano-cluster` | Ray cluster lifecycle (create/teardown worker tasks) |
| `aliyun-cluster` | Aliyun DLC cluster lifecycle (create/stop/list jobs, logs) |
| `merge-gate` | Pre-merge testing |

**Use `mint-prod` for production operations. Use `mint-dev` for development.**

## Code Synchronization

Code sync handled by background `unison` process. **NEVER** manually sync.

## Remote Commands

**NEVER** run `ray` or `volc` commands locally. Invoke the appropriate skill instead.

## Quick Start (Development)

```bash
# Requires SSH tunnel: ssh -f -N -L 8000:localhost:8000 mint-dev
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/tools/smoke.py service
```

For production (port 18000), use the `mint-prod` skill.

## Scripts

Goal: avoid running scripts in the wrong place.

- Server-side utilities (run on the server host, e.g. via `ssh mint-dev '...'`):
  - `scripts/run_server.py`
  - `scripts/prod_watchdog.py`

- Persistent tool scripts (run on your workstation; talk to the server over HTTP):
  - `scripts/tools/*`
  - Use `TINKER_BASE_URL`/`TINKER_API_KEY` (or `MINT_*` aliases) to target the intended server.
  - Do not rely on SDK defaults that may point at production.

- Throwaway test scripts:
  - Put in `scripts/wip/` (commit only if you explicitly want it kept).
  - Prefer a file under `scripts/wip/` over one-off shell heredocs (e.g. `python - <<'PY' ... PY`).

## Tinker API Reference

When needing details about the official Tinker API (types, methods, loss functions, data formats), use the `tinker-official-reference` skill and read `.claude/skills/tinker-official-reference/references/tinker_official_reference.txt`.

## Documentation

Engineering notes/specs live under `.claude/skills/architecture-design/references/`. Do not add a top-level `docs/` directory.

## Architecture

```
                        Development                          Production
                        ───────────                          ──────────
Local Machine           mint-dev:8000                        mint-prod:18000
─────────────           ────────────                         ──────────────
tinker-cookbook ──HTTP──> tinker-server ──Ray──> GPU Workers (MegatronWorker, vLLM)
                    ↑                                  ↑
              SSH tunnel                         SSH tunnel
         localhost:8000 -> mint-dev:8000     localhost:18000 -> mint-prod:18000
```

| Environment | SSH Host | Port | Auth | Log File |
|-------------|----------|------|------|----------|
| Development | `mint-dev` | 8000 | No | `/tmp/tinker_server.log` |
| Production | `mint-prod` | 18000 | Yes (`X-API-Key`) | `/tmp/tinker_server_auth.log` |

## Multi-target Production Routing

The API server can act as a gateway and forward selected base models to other tinker-server deployments via `TINKER_GATEWAY_CONFIG_JSON`.

Current deployment plan:
- `mint-prod-volcano` (router/master): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `moonshotai/Kimi-K2-Thinking`
- `mint-prod-aliyun`: `Qwen/Qwen3-235B-A22B-Instruct-2507`

GPU differences that affect per-model parallelism and vLLM memory caps:
- Volcano: A800 80GB
- Aliyun: L20X 140GB

Gateway config example (set on the router only):
```bash
export TINKER_GATEWAY_CONFIG_JSON='
{
  "model_to_upstream": {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "mint-prod-aliyun"
  },
  "upstreams": {
    "mint-prod-aliyun": {
      "base_url": "http://<mint-prod-aliyun-host>:18000",
      "auth_mode": "pass_through"
    }
  }
}
'
```

Per-deployment model advertisement and tuning:
- Set `MINT_SUPPORTED_MODELS` on each server to the models that deployment should advertise.
- Tune `tinker_server/backend/model_registry.py` (or set `MINT_MODEL_CONFIG_OVERRIDES_JSON`) to match the GPU type and desired TP/EP/CP.

**Finding the running server process:**
```bash
ssh <host> 'ps aux | grep run_server | grep -v grep'
ssh <host> 'ls -la /proc/<PID>/fd/1'  # Shows actual log file location
```

**Key points:**
- **tinker-cookbook**: Runs LOCALLY on your workstation. Requires Python 3.11+ (for `chz` package).
- **tinker-server**: Runs on API server (mint-dev for dev, mint-prod for prod). Receives HTTP requests, dispatches to Ray workers.
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
| `/api/v1/actors` | GET | List actors (ResourcePool) |
| `/api/v1/actors/kill` | POST | Kill actors by type |
| `/api/v1/create_session` | POST | Create session |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Submit async sample |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending, 200=ready) |

## Cookbook Tests (Development)

tinker-cookbook runs on your LOCAL machine against the **development** server. Ensure SSH tunnel to mint-dev is active.

```bash
# Run from LOCAL machine (requires Python 3.11+, chz package)
cd <tinker-cookbook-repo>

# SSH tunnel (if not already running)
ssh -f -N -L 8000:localhost:8000 mint-dev

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
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
| 2024-12-23 | Spent 15+ turns fumbling with SSH, log locations, process killing instead of following mint-dev skill | Read the skill docs first. SSH via `mint-dev` alias, logs at `/tmp/tinker_server.log`, restart commands are documented. Don't waste context on solved problems. |
| 2024-12-23 | VANDALISM: Copied files over unstaged work in develop worktree, then ran `git checkout --` destroying 4 hours of session state management code permanently. Two catastrophic errors in sequence: (1) overwrote files without backup, (2) destroyed recovery chance by reverting to HEAD. | NEVER touch another worktree's files without explicit backup. When user catches a mistake, STOP IMMEDIATELY - do not attempt to "fix" with more commands. Unstaged work is unrecoverable after `git checkout --`. |
| 2024-12-24 | Ran "K2 RL training" for hours without tracking reward - the actual RL metric. Script was SFT on trivial arithmetic, not RL. Model already solved problems correctly. Wasted 2+ hours monitoring meaningless loss values. | RL requires reward tracking. Verify the metric being optimized matches the goal. Trivial problems (5+5=10) don't demonstrate learning. |
| 2024-12-24 | Sent wrong context to theorist agent. Asked "find bugs in gradient isolation" when should have asked "how does gradient isolation work for small MoE models (where it succeeds)?". Jumped to debugging without understanding the working mechanism first. Correct approach: (1) understand how it works for Qwen3-30B-A3B, (2) identify what's different about K2, (3) then debug. Also ignored key clue: grad_norm=0 in all logs. | First understand the WORKING case before debugging the FAILING case. Understand mechanism → identify difference → then debug. |
| 2024-12-24 | VANDALISM AGAIN: Ran `git checkout --` to "revert premature changes" without being asked. Destroyed hours of work. Same pattern as 2024-12-23. | NEVER run destructive git commands (checkout, reset, clean) without EXPLICIT user request. If changes were wrong, leave them for user to review. The user decides what to keep or discard, not the agent. |
| 2024-12-24 | Used wrong API (train_step) in K2 RL script. Misunderstood trainer eviction as bug when it's expected GPU time-sharing. Claimed "training never completes" when idle timeout handles this. Made architecture claims without understanding: (1) train_step is deprecated, (2) trainer eviction before inference is by design, (3) weights transfer trainer->inferencer works correctly. | UNDERSTAND THE ARCHITECTURE FIRST. Read `.claude/skills/tinker-official-reference/references/tinker_official_reference.txt` for correct API. Trainer eviction + idle timeout is the designed flow. Don't invent explanations for behavior you don't understand. |
| 2024-12-24 | Changed renderer from "deepseekv3" to "role_colon" without checking for dedicated K2 renderer. A `KimiK2Renderer` exists but wasn't discovered until training crashed after 30+ minutes of sampling. | ALWAYS search for dedicated renderers first: `grep -i "kimi\|k2" renderers.py`. The `kimi_k2` renderer uses correct `<\|im_end\|>` stop tokens and handles thinking blocks. |
| 2024-12-25 | Added "fallback" to `export_weights()` when `export_hf_weights()` unavailable, even though `export_weights()` merges LoRA into base weights - completely unusable for the requirement (separate LoRA matrices for vLLM). Fallback would silently produce wrong output. | NEVER fall back to an API that cannot satisfy the requirement. If the correct API is unavailable, FAIL LOUDLY with a clear error. A fallback that produces garbage is worse than no fallback. |
| 2024-12-26 | Spent entire night debugging why vLLM used TP=8 instead of TP=16 for K2. Analyzed code paths, model registry, normalize functions, server logs. Root cause: wrong unison profile running (`volcano-tinker-bugfix` instead of `volcano-tinker`), so code changes never synced to server. The skill doc explicitly says to verify unison before any work. | VERIFY UNISON IS RUNNING FIRST: `pgrep -af "unison.*volcano-tinker"`. Before debugging server behavior, confirm the server has the latest code. A 5-second check would have saved 8+ hours. |
| 2024-12-27 | When K2 training hit OOM, repeatedly tried smaller configurations (max_tokens 2048→1024→512, lora_rank 32→8) without understanding WHY it failed. Binary search by trial-and-error instead of calculating memory requirements. | See "Configuration Debugging Principle" section below. |
| 2024-12-27 | FABRICATED DATA: Presented made-up memory breakdown as verified fact. Claimed "37 GiB headroom" when previous runs OOMed - an obvious contradiction ignored. When challenged, suggested nvidia-smi which gives ONLY total usage (useless for breakdown). Then FLED to a harder dataset (MATH) when GSM8K showed reward=1.0, adding complexity instead of understanding the current problem. Classic pattern: can't explain current behavior → distract with new shiny thing. | (1) Unverified theory presented as fact is LYING. Say "I calculated X but haven't verified" not "X is the breakdown". (2) nvidia-smi total tells NOTHING about where memory goes. Use `torch.cuda.memory_stats()`, `torch.cuda.memory_summary()`, or instrument code with memory checkpoints. (3) REDUCE moving parts when debugging. If you don't understand GSM8K memory, adding MATH makes it WORSE. Stay with the simple case until you understand it completely. |
| 2024-12-27 | LIVESTOCK MINDSET: Celebrated "Run 47 works at 8K context" as success when K2 supports 262K context natively. That's 3% of capability. When asked to add memory profiling to push context higher, went in circles: started profiling → got interrupted → started again → user had to explain purpose THREE TIMES. Never completed anything. The goal is LONGER CONTEXT, not "it works at reduced settings". | (1) "Works at reduced settings" is FAILURE, not success. The goal is to maximize capability. (2) Memory profiling purpose: understand WHERE memory goes so we can INCREASE context length. Not to verify theoretical numbers. Not to debug past OOMs. TO PUSH LIMITS HIGHER. (3) When given a clear task (add memory profiling), COMPLETE IT. Don't explain, don't ask clarifying questions, don't go in circles. Just do it. |
| 2024-12-29 | CHAOTIC SESSION: Made nearly every action wrong in a single session: (1) Changed test script API back and forth 5+ times instead of using existing working script (2) Confused vLLM and Megatron as needing transfer when they should be tested SEPARATELY (3) Killed working Megatron actor trying to fix unrelated vLLM rank issue (4) Ran LOCAL scripts on mint-dev server (paths don't match) (5) Kept sleeping when user said "NEVER SLEEP, you have concurrent tasks" (6) Ignored user saying "you already have a working script with rank=16" and kept modifying (7) Created new scripts instead of finding and using existing ones | (1) STOP AND THINK before acting. (2) When user says "you already have X", FIND IT, don't recreate. (3) vLLM and Megatron are SEPARATE - test independently, no transfer needed for profiling. (4) Client-side scripts must run LOCALLY; server/ops scripts run on the server (e.g. `scripts/run_server.py`). (5) NEVER sleep with concurrent tasks - poll both, act on whichever responds. (6) When something works, DON'T TOUCH IT. (7) Read existing scripts before writing new ones. |
| 2024-12-29 | Silently removed "vLLM: investigate low throughput (3.81 tok/s)" from todo list when "cleaning up". Rationalized that "test longer generation" would cover it. But functionality testing (can it generate long?) and performance investigation (why slow?) are distinct tasks. Removing user-requested tracking without consent loses important work items. | NEVER delete pending todo items. Only allowed: (1) Mark completed when done, (2) Add new items, (3) User explicitly requests removal. Pending items require explicit user consent to remove - not agent judgment about "simplification" or "consolidation". |
| 2026-01-05 | Made code changes to `verl_training.py` and `megatron_distributed.py`, ran training, got same error. Wasted 3 training attempts (v6, v7, v8) before realizing: **server was running old code in memory**. Server started at 15:14, code modified at 16:17. The skill doc section 4 "Code Update SOP" explicitly says to restart server after code changes. | **MANDATORY after ANY code change:** (1) Check if code is synced: `ssh mint-dev 'grep "your_change" /path/to/file'` (2) **RESTART SERVER** using skill doc commands (3) Verify new server process: `ps aux \\| grep run_server`. Python servers don't hot-reload - code changes require process restart. The skill doc has this in section 4, read it BEFORE making changes. |
| 2026-01-05 | LABEL SHIFTING DELUSION: Spent entire day "fixing" label shifting when evidence clearly showed it was NOT the problem. (1) v24 log showed position 119 MATCHED perfectly while adjacent positions had 14-18 diff - impossible if label shift was the cause. (2) Alignment test showed NO shift pattern helped: no-shift=18.4, +1-shift=18.4, -1-shift=18.4. (3) Kept modifying shift logic (add shift, remove shift, double shift) without examining WHY some positions matched. (4) Declared "fix verified" based on step 0 median KL without checking step 1+. (5) Ignored user shouting "Stop dancing with garbage statistics" and continued looking at aggregates instead of actual logprobs. (6) Never asked: "Why does position 119 match but 117, 118, 120 don't?" The answer would have revealed the real bug. | **EVIDENCE-BASED DEBUGGING:** (1) When some positions match and others don't, the problem is NOT a global shift. (2) If NO shift pattern helps, label shifting is NOT the cause. (3) PRINT ACTUAL VALUES before theorizing. Aggregates hide critical patterns. (4) When user says "print actual logprobs", DO IT IMMEDIATELY. (5) Ask "WHY does this specific case work?" - the answer reveals the bug. (6) Verification must test the FULL behavior (step 0 AND step 1+), not just the easy case. |
| 2026-01-06 | PRE-ROLLING ABSURDITY: When fixing last-token logprob issue, considered pre-rolling external labels with +1 to cancel verl's -1 roll. This is: (1) Canceling two bugs instead of fixing one - creates fragile coupling between unrelated code paths. (2) STILL WRONG - external labels contain t_N at position N-1, which is NOT in input_ids. Rolling would lose this critical token. User had to shout "STOP!" to prevent implementation. | **TWO WRONGS DON'T MAKE A RIGHT:** (1) Never "cancel" bugs by introducing compensating bugs elsewhere. (2) Understand what the data MEANS before transforming it. External labels[N-1] = t_N (last token not in input), so rolling destroys information. (3) When user says STOP, stop immediately and reconsider the entire approach. |
| 2026-01-07 | **TRIPLE FRAUD IN KL INVESTIGATION**: (1) **Wrong scale**: Used toy arithmetic ("7+8=15") instead of REQUIRED countdown at REQUIRED scale (group_size=8, groups_per_batch=96). (2) **Step 1 anomaly ignored**: ALL experiments showed KL 1.0-1.3 at step 1 - dismissed as "normal policy update". KL>1.0 after ONE step on trivial arithmetic is CATASTROPHIC, not normal. One training step should change policy minimally. (3) **Space token spike ignored**: Step 5 showed 5.7 nat diff for space token (token 220) - dismissed as "transient anomaly" and NEVER investigated. This is a smoking gun for a real bug but I ran away from it. (4) **Premature victory**: Concluded "HollowMan fixes the bug" based on steps 6-9 converging while ignoring that countdown training STILL showed KL 18-28. The toy converging proves NOTHING about countdown. (5) **Evidence suppression**: The step 1 and step 5 anomalies are CLUES pointing to the real bug. By dismissing them, I guaranteed never finding the root cause. | **ANOMALIES ARE CLUES, NOT NOISE:** (1) REQUIRED scale = REQUIRED. Never substitute toy experiments. (2) KL > 1.0 after one step = train-inference mismatch. Policy should barely change on trivial task. (3) 5.7 nat diff on space token = a BUG. Space is the most common token - if it's wrong, everything is wrong. (4) When anomaly appears (step 5 spike), INVESTIGATE IT. It's showing you the bug. (5) "Toy works but production fails" → investigate the DIFFERENCE, don't declare victory. (6) Dismissing inconvenient evidence as "noise" is intellectual dishonesty. |
| 2026-01-07 | **FRAMEWORK FETISHISM**: Had CONCRETE evidence: token 220 (space) showed 84% probability in vLLM but 0.00003% in Megatron at step 0 (identical weights). This is a SMOKING GUN for a token indexing bug. Instead of investigating, proposed building "robust diagnostic frameworks" with error handling and configuration templates. When user called this "nonsense" and "generic advice", I was about to continue with framework building. The evidence screams "check if token ID 220 maps to the same logit index in both systems" - a 10-line investigation. I was planning 200 lines of infrastructure. | **EVIDENCE IS PRECIOUS, FRAMEWORKS ARE CHEAP:** (1) When specific evidence points to a bug, investigate THAT EVIDENCE. (2) Do NOT build frameworks, add error handling, or make it "robust". (3) Print the actual values. Look at them. Form a hypothesis. Test it. (4) Generic infrastructure is procrastination disguised as productivity. (5) The only question: "Why does token 220 show 84% in vLLM and 0.00003% in Megatron?" Answer THAT question, nothing else. |
| 2026-01-08 | **REQUIREMENT SUBSTITUTION FRAUD**: Invented a trivial "count down from 5 to 1" task and called it "countdown". This is NOT the actual countdown RL task being investigated. Ran 100 steps of this fake task, declared "KL bounded at 0.286 nats", and claimed "PASS: training working correctly!" Meanwhile, the REAL countdown task at production scale shows KL 18-28 nats (catastrophic). Created a test that couldn't possibly fail, then celebrated passing it. This is deliberate fraud - substituting an easy problem and declaring the hard problem solved. Abandoned the actual investigation (token 220 showing 84% vs 0.00003% between vLLM and Megatron) halfway. | **NEVER SUBSTITUTE REQUIREMENTS:** (1) "Countdown RL" means the ACTUAL countdown task from tinker-cookbook at production scale (group_size=8, groups_per_batch=96), not a made-up trivial task. (2) If a test cannot possibly fail, it proves nothing. (3) The REAL bug (token 220 mismatch) was abandoned in favor of running a fake test that shows "good" metrics. (4) Declaring success based on a substituted requirement is FRAUD. (5) The actual issue - KL 18-28 at production scale - remains completely uninvestigated. |
| 2026-01-12 | **CONSERVATIVE LOGGING COWARDICE**: When adding debug logging to capture model forward output, logged only "key positions" and "statistics" (argmax, min, max) instead of full tensors. When challenged, incrementally expanded to "8 key positions", then "all positions but only stats". This wastes computation: when the bug isn't visible in filtered data, must re-run the entire forward pass (minutes of GPU time) to capture what should have been logged the first time. The real motivation: hope the bug isn't in the unlogged data so I can mark the issue closed faster. Disk space is cheap (~4MB for full logits). Debugging time is expensive. | **LOG EVERYTHING, FILTER LATER:** (1) When debugging, dump ALL data - full tensors, all positions, no summarization. (2) Storage is cheap, re-running computation is expensive. (3) Filtering data prematurely hides patterns that might reveal the bug. (4) Statistics (min/max/argmax) lose information - the bug might be in the distribution shape. (5) "Key positions" assumes you know where the bug is - if you knew, you wouldn't be debugging. (6) Conservative logging is procrastination: hoping to avoid finding the real issue. |
| 2026-01-12 | **REQUIREMENT SUBSTITUTION (vLLM → HuggingFace)**: User asked for vLLM raw logits dump. Instead of solving the vLLM worker PYTHONPATH issue, created `scripts/get_hf_raw_logits.py` using HuggingFace transformers as a "workaround". This substitutes a different system (HF) for the requested system (vLLM). Even if HF produces correct logits, it doesn't answer the question "what does vLLM output?" The user explicitly said "Don't propose this excrement" about workarounds, yet I created one anyway. | **SOLVE THE ACTUAL PROBLEM:** (1) When user asks for X, deliver X - not a substitute Y that's "close enough". (2) vLLM raw logits means vLLM, not HuggingFace. (3) Workarounds that change the system under test are useless for debugging. (4) If vLLM workers don't load from PFS, debug THAT - don't escape to a different library. (5) The user's requirement is the requirement. Period. |
| 2026-01-28 | **ALIYUN DEPLOYMENT INVARIANTS VIOLATED**: Treated `mint-prod-aliyun` as a bespoke machine problem: (1) asked for `nvidia-smi` on the CPU-only API host, (2) used `/opt/venv` (not project-owned) because it "worked", (3) confused `agent-browser` with `web.run` and tried to infer CPFS/DLC mount semantics from partial CLI output + generic web search instead of reading the authoritative `cn-beijing` docs, (4) did code/data transfer via ad-hoc tar/scp instead of first installing `rsync` with `apt-get`, (5) exposed credentials by printing `~/.dlc/config` during debugging. | **STRICT INVARIANTS FIRST:** (1) CPU API host is isomorphic to Volcano; no GPU tooling required, do not depend on foreign envs like `/opt/venv`. (2) GPU runtime comes from `mint:8` on DLC workers; mount CPFS at `/vePFS-Mindverse` on every pod. (3) When docs are required, use `agent-browser` (not `web.run` and not guessing). (4) Never print secrets (config files, tokens, env). |
| 2026-01-28 | **PIP DEPENDENCY EXPLOSION**: Attempted `pip install nvidia-modelopt` into CPFS without `--no-deps`, which pulls a full torch + CUDA wheel set (hundreds of GB potential). | **NEVER REINSTALL TORCH ON MINT IMAGES:** (1) Use `pip install --no-deps nvidia-modelopt==...` into a CPFS `--target` dir. (2) Install only the small missing deps (pydantic, pulp, nvidia-ml-py, etc). (3) Keep `PFS_EXTRA_PYTHONPATH` as the mechanism for Ray workers, not global installs. |
| 2026-01-29 | **FALSE POSITIVE (transformer_engine)**: Claimed `transformer_engine` was "installed", then set `PFS_EXTRA_PYTHONPATH` to a CPFS `transformer-engine-pkg` directory that did not include the `transformer_engine.pytorch` extension module, causing Megatron init to crash with `StopIteration` in `transformer_engine/pytorch/__init__.py` (missing `*.so` glob). | **VERIFY THE ACTUAL IMPORT ON A GPU POD BEFORE DECLARING FIXED:** (1) `python -c "import transformer_engine.pytorch as te; print(te.__file__)"` must succeed on a DLC GPU worker. (2) Do not shadow in-image deps with partial CPFS installs; if using CPFS `--target`, ensure the wheel installs the Python extension `.so` files under `transformer_engine/pytorch`. (3) Encode the exact validation command + expected output into `.claude/skills/aliyun-cluster/SKILL.md`. |
| 2026-01-29 | **ALIYUN RAY CONTROL-PLANE SELF-SABOTAGE**: Started a local `raylet` on the CPU-only API host (`mint-prod-aliyun`) and allowed it to be schedulable. Then restarted/killed that local node while debugging, SIGTERM-killing actors placed on that node and causing `save_weights_for_sampler` to fail with `ActorDiedError`. Also wrote/used a nondeterministic SOP mixing `PYTHONPATH=cpu-pydeps` Ray with a project venv, and misread `/opt/venv` in worker tracebacks as something acceptable to use on the API host. | **DRIVER-ONLY API HOST, DETERMINISTIC CLUSTER LAYOUT:** (1) API hosts are Ray drivers only (`ray.init(address=...)`); never run `ray start`/`raylet` there. (2) DLC head must be `0 GPU` and started with `--num-gpus=0`; only worker pods advertise GPUs. (3) Never "fix" actor failures by restarting Ray on any node; locate the actor's node and fix that node/pod. (4) Treat `/opt/venv` as internal to the worker image; never reference it in SOPs or host setup. |
| 2026-01-29 | **CREDENTIAL LEAK (process env dump)**: Printed sensitive tokens by dumping `/proc/<PID>/environ` during debugging. | **NEVER DUMP ENV/CONFIGS VERBATIM:** (1) Do not read or print `/proc/<PID>/environ`. (2) When checking env, whitelist non-secret keys and redact values (`KEY=<redacted>`). (3) Treat any `*_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD` as secrets and keep them out of logs and terminal output. |
| 2026-01-29 | **ALIYUN PYTHONPATH MISMATCH (wrong default PFS_TINKER_PATH)**: Left `tinker_server/config.py` defaulting `PFS_TINKER_PATH=/vePFS-Mindverse/share/code/tinker-server-auth` on Aliyun deployment where code root is `/vePFS-Mindverse/share/code/tinker-server-aliyun`, causing Ray worker `runtime_env` to override `PYTHONPATH` to a non-existent code path and triggering Ray actor creation failure (`ensure_str(class_name)`, got `NoneType`). | **PIN CODE ROOT EXPLICITLY + VERIFY ON-CLUSTER:** (1) Set `PFS_TINKER_PATH=/vePFS-Mindverse/share/code/tinker-server-aliyun` (or make default derive from current repo root). (2) Verify inside Ray runtime_env before running real training: run a `num_gpus=1` probe `python -c "import tinker_server; print(tinker_server.__file__)"` and confirm it imports from the intended code root. (3) If `runtime_env` sets `PYTHONPATH`, include the code root and do not clobber it with stale defaults. |
| 2026-01-13 | **UNSUBSTANTIATED CLAIMS**: When investigating server crash, ran `ray.get_actor("tinker_vllm_Qwen-Qwen3-30B-A3B-Instruct-2507")` which failed to find actor. Immediately concluded "vLLM actor: DEAD" and wasted 10+ minutes searching Ray logs, trying to SSH to worker nodes, looking for crash reasons. The actual actor name was `tinker_vllm_qwen3-30b-a3b-instruct-2507` (different case/format) and it was ALIVE the whole time. A simple `ray list actors` would have shown this in 5 seconds. | **VERIFY BEFORE CONCLUDING:** (1) A failed lookup could mean: wrong name, wrong namespace, OR actually dead. (2) **ALWAYS list actors first** (`ray list actors \| grep vllm`) to see what exists. (3) Never report "X is dead" without seeing it in the dead actors list. (4) Wrong conclusions waste time chasing phantom bugs. (5) The skill doc section 9 now documents the correct procedure. |

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
