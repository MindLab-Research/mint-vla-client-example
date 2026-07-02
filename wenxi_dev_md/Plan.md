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

## Goal 1 — Deploy MinT so the service runs ✅ DONE

**Status:** completed (2026-07-02)

单机 Ray 集群（head 0-GPU + worker 8-GPU）+ mint-server :30496 `{"status":"ready"}`。
`step0_ray_up.sh` → `step1.sh`（rsync 到 `/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi`）→
`step3_start.sh`（namespace `mint_wenxi_dev`，port 30496）→ `step4_health.sh` 通过。
`MINT_DISABLE_MINT_ROUTE=0`，VLA 路由已加载。

---

## Goal 2 — Get pi0.5 (flow) running end-to-end ✅ DONE

**Status:** completed (2026-07-02)

两条验证链路均通过：
1. **合成数据冒烟**（`PI05check.sh` / `openpi_vla_smoke.py`）：create_model → train_step(flow_matching) → save_weights → action_session → act，虚拟 1×1 图像 + 零动作，全链路返回 action tensor。见 memory `mint-openpi-pi05-smoke`。
2. **Lance 真实数据**（`PI05lance.sh` / `openpi_vla_smoke_lance.py`）：从 `pi_video_streams_lance_smoke.lance` 读入真实图像/state/actions，走完整 openpi transform（LiberoInputs → Normalize → PaliGemma 分词 → Pad），train_step 返回真实 flow-matching loss（0.169 → 0.225），act 返回 `[10,7]` float32 动作张量。

---

## Goal 3 — Understand pi0.5 data source + how data becomes a service ✅ DONE

**Status:** completed (2026-07-02)

完整数据链路文档已写出：`wenxi_dev_md/Openpi05_dataflow.md`（766 行），覆盖：
- Lance / LeRobot parquet → `LancePi05Dataset` / `_iter_windows_for_task` 取窗口
- openpi transform 流水线（LiberoInputs → Normalize → TokenizePrompt/PaliGemma → PadStatesAndActions）
- Datum 降格（`_lower_vla_datum`，observation/supervision → loss_fn_inputs）
- 服务端 5 层流水线（路由 → 队列/派发 → 引擎路由 → 后端引擎 → Ray actor/JAX GPU）
- actor 生命周期（共享 pool、detached、GPU 放置、JIT 编译）
- 结果回传路径（async_resolve → retrieve_future 200）
每个环节附代码位置（文件:行号）。

---

## Goal 5 — 并行训练策略：最大化训练吞吐

**Status:** planned (2026-07-02)

### 5.0 现状诊断（从代码读出来的基线）

| 维度 | 现状 | 代码位置 |
|---|---|---|
| Ray actor 并发 | `max_concurrency=1`：串行，forward_backward 与 optim_step 不能并发 | `openpi_ray_runtime.py:119` |
| GPU 数 / actor | `num_gpus=1`：单 GPU，单 actor | 同上 |
| JAX FSDP mesh | `fsdp_devices` 参数已存在，调 `sharding.make_mesh(fsdp_devices)` | `openpi_pi05_worker.py:276` |
| 梯度累积 | `session.accumulated_gradients` 计数器已有；`train_step` = 1×fb + 1×optim，无累积 | `openpi_pi05_training.py:424,446` |
| 数据并行 | 无；`batch` 维度完全在单 actor 内处理 | — |

结论：**三个正交的并行轴都有提升空间**，且 JAX FSDP 基础设施已内置，是最低成本的第一步。

---

### 5.1 轴 A — 多 GPU FSDP（模型并行 + ZeRO sharding）

**原理：** openpi worker 内部用 `jax.sharding.NamedSharding` + `openpi.training.sharding.fsdp_sharding` 做 FSDP，`fsdp_devices` 控制参与卡数。目前 `create_session` payload 里 `fsdp_devices` 来自 `OpenPIFastRuntimeSpec`，默认为 1。

**计划：**
1. 在 `MINT_MODEL_PLACEMENT_JSON`（或 GPU placement spec）里为 pi0.5 actor 分配 N 卡（`num_gpus=N`）。
2. 在 `create_session` 的 `_create_session_payload` 里透传 `fsdp_devices=N`（`openpi_pi05_training.py:334`）。
3. worker 的 `make_mesh(N)` 自动创建 FSDP mesh；参数 shard 到 N 卡，梯度 all-reduce。
4. **需验证的约束：** pi0.5 是小模型（~3B，LoRA 冻结后可训参数更少），FSDP 的通信开销可能在 4 卡以上才值回来；先测 1 vs 2 vs 4 的 step time。

**预期收益：** 允许更大 batch 而不 OOM；梯度 all-reduce 替代单卡梯度累积，数值等价但延迟更低。

---

### 5.2 轴 B — 梯度累积（已有 API，还没用）

**原理：** mint server 已把 `forward_backward` 和 `optim_step` 拆开暴露：
```
POST /api/v1/mint/vla/train_step   →  1×fb + 1×optim（当前用法）

# 也可以：
N×POST /api/v1/mint/vla/forward_backward  → 梯度累积 N 步
1×POST /api/v1/mint/vla/optim_step        → 一次参数更新
```
`session.accumulated_gradients` 记录已累积次数；worker 端 JAX grad 在 `forward_backward` 里原地累积（`openpi_pi05_worker.py` 的 `_accumulate_grads`）。

**计划：**
1. 把 `openpi_vla_smoke_lance.py` 的训练循环改为支持 `--grad-accum N`：N 个 batch 各调一次 `forward_backward`，最后统一调 `optim_step`。
2. effective batch size = `batch_size × N`，显存占用不变（单次 fb 仍是 `batch_size`）。
3. 在 lance smoke 脚本里量化：grad_accum=1/2/4/8 时的 step time + loss 曲线，找到最优 effective batch。

**关键：** 梯度累积在 JAX 里是原地加，不是 Python list append；N 次 fb 后 optim_step 会归零 `accumulated_gradients`；client 侧需按序串行调用（不能并发）。

---

### 5.3 轴 C — 数据并行多 actor（多租户同时训练）

**原理：** mint supervisor 可以管理多个 OpenPI actor，每个 actor 独占 1-N 张 GPU。多个用户/任务可以同时 `create_model`，各自占一个 actor slot，并发训练互不干扰。

**计划：**
1. 在 `MINT_MODEL_PLACEMENT_JSON` 里为 pi0.5 配置多个 placement slice（每个 slice = 一组 GPU）。
2. `gen_dev_placement.py` 生成 placement JSON，pi0.5 key 对应多条 `node_ip:device_ids` 记录。
3. 在 lance smoke 脚本基础上写一个「多 client 并发」压测：同时起 K 个 `create_model` session，各跑 N 步，统计总吞吐（samples/sec）。

**注意：** 这是多租户数据并行（每个 session 独立），不是同一训练任务的梯度同步 DP。真正的同步 DP 需要多 actor 梯度聚合，目前 mint 没有这个接口，属于更长期工作。

---

### 5.4 优先级排序与里程碑

| 优先级 | 方向 | 工作量 | 预期收益 |
|---|---|---|---|
| ★★★ 先做 | 梯度累积（轴 B） | 改 client 脚本 ~30 行 | effective batch 扩大 N 倍，无需改 server |
| ★★☆ 次之 | 多 GPU FSDP（轴 A） | placement JSON + 透传 fsdp_devices | 允许更大 batch；解锁 2-4 卡训练 |
| ★☆☆ 后做 | 多 actor 并发（轴 C） | placement + 压测脚本 | 多任务并发，吞吐线性扩展 |

**Done 标准（Goal 5）：** `grad_accum=4` 的 lance smoke 跑通且 loss 曲线与 `grad_accum=1` 等价；多 GPU FSDP 在 2 卡下 step time < 单卡；有数据说明何时 FSDP 通信开销超过收益。

---

## Goal 6 — 推理加速：RTC 与低延迟 action serving

**Status:** planned (2026-07-02)

### 6.0 现状诊断

| 指标 | 现状 | 来源 |
|---|---|---|
| 首次推理延迟 | ~15 s（含 JAX JIT 编译） | 实测 `infer_ms=15441` |
| 热推理延迟 | ~2–3 s（JIT 缓存命中后） | 预估，需实测 |
| 去噪步数 | `num_steps=10`（flow ODE linspace 1→0） | `openpi_pi05_action_worker.py:319` |
| 输出 | 全 `[10, 7]` 张量一次返回，无流式 | `act route` |
| 温度采样 | 不支持（temperature != 0 直接报错） | `action_worker.py:370` |

**机器人控制的实时性要求：** 典型控制频率 10–50 Hz → 单次推理预算 20–100 ms。当前 ~2–3 s 的热延迟远超预算，是闭环控制的主要瓶颈。

---

### 6.1 RTC（Receding Horizon / Action Chunking）

**原理：** 不等当前 action chunk 执行完再推理下一个，而是提前触发下一次 act 请求，用执行时间掩盖推理延迟（overlapping inference with execution）。

**实现方案（client 侧，无需改 server）：**
```python
# 伪代码：action chunking + 异步预取
execute_every = 3          # 每 3 帧执行一次新推理
horizon = 10               # server 返回 10 步动作

pending_future = None
for t in count():
    if t % execute_every == 0:
        if pending_future:
            actions = retrieve_future(pending_future)   # 这时推理应已完成
        pending_future = post_act_async(observation)    # 提前触发下次推理
    robot.execute(actions[t % execute_every])
```
关键：`execute_every < infer_time / control_period`，让推理在执行期间完成。mint 的异步 future 机制（408 polling）天然支持这个模式。

**计划：**
1. 在 `scripts/wip/` 写一个 `openpi_closedloop_sim.py` 演示 action chunking + 异步预取模式（先用仿真时间代替真实控制周期）。
2. 实测不同 `execute_every`（1/3/5）下的等效帧率与延迟分布。

---

### 6.2 减少去噪步数（Consistency / Few-step Flow）

**原理：** flow-matching 的去噪步数直接决定推理延迟；从 10 步降到 1–3 步可降低延迟 3–10×，代价是动作质量略降（视任务而定）。

**现有 API：** `trace_config` 参数已可传 `num_steps`（`openpi_pi05_action_worker.py:294`）：
```python
# act payload 里透传 trace_config
{"observation": ..., "trace_config": {"num_steps": 3}}
```
当前 act route 是否透传 `trace_config` 到 action worker 需确认（读 `routes/mint.py:500`）。

**计划：**
1. 确认 `act` route 是否已透传 `trace_config`；如未透传，在 `openpi_pi05_action_worker.py` 的 `act` op 里加一行读取。
2. 在 lance smoke 基础上实测 `num_steps=10/5/3/1` 的推理延迟 vs 动作质量（LIBERO 成功率 or 仿真误差）。
3. 目标：`num_steps=3` 下热推理 < 1 s，`num_steps=1`（一步 flow）< 500 ms。

---

### 6.3 JIT 预热 + 权重预加载

**现状：** 首次推理 ~15 s 包含 JAX JIT 编译；action session 创建后的第一次 `act` 总是慢。

**计划：**
1. 在 `create_action_session` 之后、进入控制循环之前，发送一次**虚拟 act**（用零观测）触发 JIT 编译 + 权重预加载，丢弃结果。
2. 在 `PI05lance.sh` 风格的 shell wrapper 里集成这个 warm-up 步骤。
3. 量化 warm-up 后的 P50/P95 推理延迟。

---

### 6.4 Done 标准（Goal 6）

- `num_steps=3` 下热推理 P50 < 1 s，有实测数据支撑。
- action chunking demo 脚本能以 ≥ 5 Hz 等效帧率运行（inference 被 execution 掩盖）。
- warm-up 后首次"正式"推理延迟 < 2 s。

---

## Goal 7 — 闭环控制 Best Practice：仿真 + 实机

**Status:** planned (2026-07-02)

### 7.0 架构概述

```
仿真器 / 实机
  └── 观测获取 (camera + state)
        │
        ▼  HTTP (异步 future)
  mint-server ──→ OpenPI action actor ──→ [10, 7] actions
        │
        ▼
  client 侧 action executor
  (chunking / receding horizon / temporal ensemble)
        │
        ▼
  机器人执行器 (关节角 / 末端速度)
```

核心接口：**一个持久 action session** 跑整个 episode；每个控制步调 `act`，拿回 `[horizon, 7]` 动作张量，执行其中 `execute_every` 帧，然后继续。

---

### 7.1 仿真闭环（LIBERO / MuJoCo）

**目标：** 用训练好的 LoRA 权重跑 LIBERO 任务评测，验证微调是否有效。

**步骤：**
1. 训练完成后调 `save_weights_for_sampler` 拿到 `mint://` 路径。
2. `create_action_session(model_path=<mint_path>, base_model="openpi/pi05-libero-low-mem-finetune")`。
3. 在 `pi-finetune` repo 里（或新写一个 `scripts/wip/libero_eval_mint.py`）：
   - 起 LIBERO 环境（MuJoCo-based），每步获取 `image / wrist_image / state / prompt`；
   - 组成 observation（走相同的 `LiberoInputs → Normalize → TokenizePrompt` transform，**inference 路径只需 data_transforms + model_transforms，不需 Normalize 的 actions 部分**）；
   - 调 `act`，取前 `execute_every` 帧动作执行，滑窗；
   - 统计 **成功率**（task_success）、**平均 episode 长度**、**推理延迟 P50/P95**。
4. 基线对比：未微调的 pi05_base 权重在同任务的成功率。

**关键 Gotcha：**
- Inference 时的 observation 预处理必须与训练完全一致（同一 norm_stats、同一 `LiberoInputs` camera 映射），否则 distributional shift 导致策略失效。
- lance 数据的 norm_stats 是从 MuJoCo/MANO 分布算的，与 LIBERO 分布不同；inference 时应用哪个 norm_stats 需明确对齐（最好用同一个数据集的 norm_stats 训练 + eval）。

---

### 7.2 实机闭环（best practice 规范）

**控制循环规范：**

```python
# 伪代码：实机控制循环
session_id = create_action_session(model_path=..., base_model=...)
warm_up_act(session_id, dummy_obs)           # 触发 JIT，丢弃结果

EXECUTE_EVERY = 3                             # 每推理结果执行 3 帧
pending_req = submit_act(session_id, get_obs())

for t in count():
    obs = get_obs()
    if t % EXECUTE_EVERY == 0:
        actions = await_result(pending_req)   # 阻塞等上次推理完成
        pending_req = submit_act(session_id, obs)   # 立即提交下次推理
    robot.step(actions[t % EXECUTE_EVERY])   # 执行当前 chunk 的第 k 帧
    time.sleep(1 / CONTROL_HZ)

delete_action_session(session_id)            # finally 里清理
```

**安全约束：**
- 每次 `act` 之前做 obs 合法性检查（图像非全黑、state 在范围内）；
- 推理超时（>2s）时原地停止（发送零动作或保持当前关节角），不等待；
- `create_action_session / delete_action_session` 必须在 `try/finally` 里，避免占用 GPU actor；
- 实机首次运行时 `EXECUTE_EVERY=1`（每帧都更新），保守；稳定后调大。

---

### 7.3 Temporal Ensemble（可选，提升平滑性）

**原理（来自 ACT/π0 论文）：** 每 k 帧触发一次推理，拿到的 `[horizon, 7]` 与之前推理结果的 overlap 部分做指数加权平均，减少关节抖动：
```
action_t = Σ_{i} w_i * action_t_from_query_i    (w_i = exp(-λ*i))
```
**现状：** server 端无 temporal ensemble，需在 client 端实现（维护一个 `deque` 存最近 K 个推理结果）。

**计划：** 在 `openpi_closedloop_sim.py` 里实现并对比有无 ensemble 时的关节轨迹平滑度。

---

### 7.4 Done 标准（Goal 7）

- LIBERO 仿真评测脚本跑通，能输出 success rate；微调后 success rate > 未微调基线。
- 闭环控制 wrapper 有明确的 warm-up / chunking / timeout / cleanup 规范，封装进可复用的 `MintVLAController` 类（`scripts/wip/mint_vla_controller.py`）。
- Temporal ensemble 实现，有轨迹对比图。

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
