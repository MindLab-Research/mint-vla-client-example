# OpenPI_Separate — pi0.5 从 mint 拆分的开发目标

> 分支：`Openpi-LoRA-Separate`（从 `dev-vla-wenxi` 切出）。
> 本文是**开发目标与方向的对齐稿**，记录已定方向 + 待讨论项，不含实现细节。
> 起草：2026-07-08。内容来自与 wenxi 的讨论要点，逐条整理，**待确认后再动代码**。
>
> **状态更新（2026-07-13）**：Ray 拆除工作已完成。openpi（pi05 + fast）已彻底脱离 Ray，
> 在 API 进程内联执行。训练收敛验证（400 step，loss 0.042→0.006）与推理质量验证
>（逐帧 MSE 与 Ray 版本等价）均通过。删除 3 个 openpi 专属 Ray 文件，其他服务不受影响。

---

## 0. 一句话目标

把 pi0.5（openpi VLA）从 mint 的训练/推理 infra 里**拆出来独立开发**，
但**对外保留 Tinker / Mint API 契约**——把 mint 这一侧当作 RL 环境的接入面。
拆分后，训练三件套（算梯度 / 取权重 / 更新）改为**分别独立调用的 API**。

---

## 1. 为什么要拆（动机）

现阶段 pi0.5 **不适合继续在 mint 上开发**，原因：

- **架构根本不同**：pi0.5 是 JAX + flow-matching 的 VLA，mint 的核心是 Qwen 系
  文本 / MoE 栈（Megatron / verl / vLLM）。两者训练范式、权重格式、并行方式都不一样。
- **没有可复用的 Qwen 相关组件**：mint 里那套（tokenizer 元数据、文本 loss、MoE 调度、
  vLLM sampler 等）对 pi0.5 **推不动、也用不上**，硬塞进 mint 只会互相牵制。
- 结论：pi0.5 的 **infra 应当独立**，不再寄生在 mint 的训练后端分派里
  （即现在的 `training_backend == "openpi_pi05"` 这条路径）。

---

## 2. 拆分边界：保留什么 / 剥离什么

### 2.1 保留（不拆）

- **Tinker API 要保留**：外部契约（`forward_backward` / `optim_step` /
  `save_weights` / `sample` 等原语语义）继续对齐 Tinker，客户端接入方式不变。
- **Mint API 保留，定位改为「RL 环境」**：mint 这一侧当作 RL 环境的接口层
  （环境交互 / rollout / reward 入口），而不是 pi0.5 的训练执行引擎。
  - ✅ **已定**：保留 **HTTP 端点形态**——`/api/v1` 那套 Tinker 兼容端点继续对外，
    客户端接入方式完全不变；只把**底层执行**从 mint 的后端分派换成独立的 pi0.5 infra。

### 2.2 剥离（本分支核心目标）

- **🎯 从 Ray 模式拆出、不再依赖 Ray**（这就是分支名 `Separate` 的含义，本分支的核心）：
  pi0.5 不再走 mint 的 Ray actor / 调度 / 准入
  （`model_work_scheduler`、detached actor、runtime_env、direct-runtime future 那一套）。
  pi0.5 的进程管理、设备管理、分布式全部独立出来，用非 Ray 的方式跑。
  - ✅ **第二步已完成（2026-07-09）**：openpi（pi05 + fast）已彻底脱 Ray，
    删除三个 openpi 专属 Ray 文件，训练/推理全在 API 进程内联执行。见 §3.7。
- **微调 infra 待定**：拆出后 pi0.5 用什么做微调 infra **尚未定**
  （是否继续 FSDP-on-JAX、是否换编排、权重如何持久化），见 §5 Q2。

- **物理形态：本仓库内解耦目录**（✅ 已定）——不新建独立 repo，先在 mint 仓库内
  把 pi0.5 拆成独立目录/模块，与 mint 的 Qwen 训练后端在代码层面解耦，
  但物理上不搬出仓库。

---

## 3. 训练 API 分解（核心设计方向）

拆分后，把训练循环拆成**分别独立调用的 API**，与 Tinker 原语一一对应：

| 步骤 | 语义 | 对应 Tinker 原语 |
|---|---|---|
| **算梯度** | 前向 + 反向，产出/累积梯度 | `forward_backward` |
| **取权重** | 读取当前（或落盘的）权重 | `save_weights` / weights 读取 |
| **更新** | 用累积梯度做 optimizer step | `optim_step` |

要点：
- 三步**解耦、可分别调用**（不再是 mint 里 `train_step` 那种合并原语一把梭）。
- 目的：让 RL 环境（mint 侧）与训练执行（独立 pi0.5 infra）之间以这三个
  显式 API 交互 —— 环境给数据 → 算梯度 → 更新 → 取权重回推理，形成可控闭环。

---

## 3.5 拆 Ray 的关键发现（决定拆分策略）

读代码确认了当前 Ray 版 pi0.5 的真实形态，这直接简化了拆分路径：

- **训练逻辑本身与 Ray 无关**：`OpenPIPi05WorkerSession`（`openpi_pi05_worker.py`）
  是一个**自包含的纯 JAX 类**，持有 `_train_state`（params + opt_state），
  只有训练算子（`forward_backward` / `optim_step` / `save_sampler_weights`），
  通过 stdin/stdout 或直接调用的 `_dispatch(op, payload)` 与外界通信，**不 import ray**。
- **不是边训边推**：训练 worker **没有** `sample_actions`；推理是**另一个** worker
  （`OpenPIPi05ActionSession`，只有 `act()`），靠**落盘 checkpoint 单向传递**权重。
  训练 → `save_weights_for_sampler`（导出 params+assets 目录）→ 推理另起 session `load`。
- **Ray 耦合只在「runtime 装配层」**，两处：
  1. `OpenPIPi05TrainingEngine.initialize()` → `ensure_openpi_ray_initialized()`
  2. `_default_runtime_factory` → `start_openpi_shared_ray_runtime(...)`（把 worker 塞进 Ray actor）
- **已有现成接缝**：目录里 `OpenPIDirectWorkerClient`（`openpi_direct_runtime.py`）
  已经能"进程内直接跑 worker、不 fork 子进程"（原本用于 Ray actor 内部）；
  且引擎 `__init__` 接受 `runtime_factory` 注入。
  → **拆 Ray = 换 runtime_factory**，worker 与引擎主体一行不改。

---

## 3.6 进展：第一步 ✅ 已完成（脱离 Ray 训练跑通）

**目标**：证明同一套已验证正确的 LoRA 训练逻辑，经非 Ray 的 direct runtime，
脱离 Ray 能跑通训练三件套。不碰 server / HTTP / worker / 引擎主体。

**改动**（最小）：
- 新增 `mint_server/backend/openpi/openpi_pi05_local_runtime.py`：
  非 Ray 的 runtime factory `make_local_pi05_runtime`，直接
  `OpenPIDirectWorkerClient.start(spec)` 在本进程跑 worker。
  **关键**：不用 `OpenPIFastRuntimeSpec.from_env()`（那会触发 PFS runtime-env
  manifest / tier 的 Ray-infra bootstrap），改为直接构造最小 spec，做到真正 Ray-free。
- 新增验证脚本 `scripts/wip/openpi_pi05_local_train_check.py`：构造真
  `TrainingSession` + 真 `OpenPIPi05TrainingEngine`（注入本地 factory），
  数据构造复用已验证的 `openpi_vla_smoke_lance.py`，wire dict 经 `/mint/vla` 路由同款
  `_lower_vla_datum` 转 `Datum`——payload 与 server 路径逐字节一致，唯一差异是 runtime。
  注意：**不调 `engine.initialize()`**（那是 Ray init 路径），本地 factory 无需 Ray init。

**实测结果**（4 步 / batch=1 / full lance，2×A800 单进程，无 Ray）：
- create_session 33s ready；三件套全跑通；sampler 导出 params+assets 完整 5.0G 目录
  （结构与 Ray 路径导出一致）。
- loss:mean 0.111 → 0.095 → 0.498 → 0.057，grad_norm:mean 0.12~0.60 非零，
  param_norm 逐步变化 —— 梯度真实回传、LoRA 参数确在更新。
- 日志 `results/logs/pi05_local_train_check.log`，结果 `results/datas/pi05_local_train_check.json`。
  （5.0G 冒烟 checkpoint 已删，仅留日志与 JSON。）

**结论**：Ray 对 pi0.5 训练是可剥离的外层编排；训练执行本身在单进程 JAX 下即可跑通。
第一步只换了 runtime 装配层，没动任何已验证逻辑。

**下一步（第二步）**：把这个本地 runtime 接到 mint 引擎的开关上（如
`MINT_OPENPI_PI05_LOCAL_RUNTIME=1`），让 HTTP 端点也能走非 Ray 路径；
默认仍走 Ray，不影响现状。待与 wenxi 确认后进行。

---

## 3.7 进展：第二步 ✅ 已完成（openpi 彻底删除 Ray）

**决策**（与 wenxi 对齐）：不做开关、直接放弃 Ray。openpi（pi05 + fast）请求
在 API 进程内联执行、串行（单租户 A），不再进共享 Ray scheduler / ModelEngineHost /
task_state_store actor。其他服务（megatron/verl/qwen）的共享 Ray 基础设施物理保留、不动。

**关键更正**：原以为可复用现有 `_do_*` + `task_futures` 内联。实测发现 `task_futures`
坐在 `TaskStateStoreClient`（detached Ray actor 代理）上、`async_upsert_training_session`
同理——复用它们仍需 Ray。故 openpi 改用**自包含的进程内 future/session 存储**。

**新增**：
- `mint_server/backend/openpi/openpi_local_execution.py`：自包含 Ray-free 执行层。
  持有进程内 openpi 引擎（本地 factory）+ 进程内 session dict + 进程内 future dict
  （openpi 内联执行，算完即 ready）。提供 `handle_create_model / handle_train_step /
  handle_save_weights_for_sampler / handle_create_action_session / handle_act /
  delete_local_training_session` 及 `get_future`。`_exec_lock` 串行化所有 stateful op。
- `openpi_pi05_local_runtime.py` 增 `make_local_pi05_action_runtime`（动作推理本地 factory，
  并设 `MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT`，取代 Ray runtime_env 注入）。

**路由改动**（只加 openpi 分支，非 openpi 一字不动）：
- `routes/training.py`：`create_model` / `save_weights_for_sampler` / `delete_model`
  在 openpi 后端时走本地 handler，绕开 `enqueue_model_work`。
- `routes/mint.py`：`vla/train_step` / `create_action_session` / `act` /
  `delete_action_session` 同理。
- `routes/futures.py`：`retrieve_future` 顶部先查 openpi 进程内 future store。

**引擎/action 去 Ray + 删文件**：
- `openpi_pi05_training.py` / `openpi_fast_training.py`：默认 factory 改本地、
  `initialize()` 去掉 `ensure_openpi_ray_initialized()`。
- `action_session_manager.py`：去掉 `import ray` 与 Ray runtime 导入；默认 action
  factory 改本地 direct client；detached-actor 恢复逻辑（`_recover_detached_action_runtime_client`
  / `_recover_manager_for_session`）neuter 为 no-op（进程内无 detached actor 可恢复）。
- **删除**：`openpi_ray_runtime.py`、`openpi_shared_ray_runtime.py`、`openpi_action_ray_runtime.py`。
- **保留**：所有 worker、`openpi_direct_runtime.py`、`openpi_fast_runtime.py`、
  `openpi_fast_action_runtime.py`、`openpi_orbax_compat.py`；共享 `mint_server/ray`、
  `scheduling`、`actors` 一字未动。

**验证**：
- `mint_server.app` 删文件后仍能 import（server 可启动，其他服务不受影响）。
- `scripts/wip/openpi_pi05_local_route_check.py`：**Ray 全程未初始化**，按路由同款逻辑
  跑通 create_model → train_step×3 → save_weights_for_sampler → create_action_session
  → act：loss:mean 0.111→0.095→0.497（与第一步逐字节一致）、grad_norm 非零、
  save 产出 `mint://` URI + 完整 checkpoint、act 返回 `[10,32]` 动作块。
  日志 `results/logs/pi05_local_route_check*.log`。

**尚未做（诚实标注）**：
- `openpi_fast` 只做到 import-clean + 本地 factory 就位，未跑 fast 的端到端（本轮聚焦 pi05）。

---

## 3.8 进展：完整 HTTP 端到端 ✅ 已跑通（2026-07-10）

§3.7 曾标注"完整 HTTP-server 端到端未跑"。本轮补上：**真起 FastAPI、经真实
tinker HTTP API 完成 create_model → train_step → save_weights_for_sampler →
create_action_session → act 全链路，server 全程 Ray 零初始化。**

**Ray-free 起 server 的关键**：
- 启动器 `scripts/wip/_run_local_openpi_server.py`：直接对 `mint_server.app` 起
  uvicorn，单 worker，**必须用 gpu_rl host-venv 的 python**（API 进程内联跑
  JAX/GPU 训练，CPU base-python 不行）。
- `app.py` lifespan 里 `init_ray` 包 try/except，由新开关 `MINT_ALLOW_NO_RAY=1`
  控制：连不上 Ray 时降级（degraded）而非崩溃。
- 必需 env：`MINT_ALLOW_NO_RAY=1`、`MINT_USAGE_BACKEND=disabled`（跳过 postgres）、
  `MINT_SKIP_SUPERVISOR=1`、`MINT_UVICORN_WORKERS=1`（默认 8，会切分进程内 store！）、
  `MINT_SUPPORTED_MODELS=openpi/pi05-...`，加 OPENPI 权重/assets/HF 路径。
- healthz 返回 `unhealthy`（degraded：无 Ray / 无 pg），但 openpi 端点照常工作。

**一键脚本**：`scripts/vla/PI05lance_local_norray.sh` —— 停旧 server → 起
Ray-free server（带全部 env）→ 等就绪 → 跑 driver N 步 → 汇总 loss/save-uri/act
到 log。env 调参：`MINT_PI05_STEPS`（默认 400）、`MINT_PI05_BATCH`（2）、
`MINT_PI05_SKIP_SERVER=1`（复用已起 server）、`MINT_CUDA_DEVICES`（默认 3,4,5,6）。
带时间戳的 log 写到 `results/logs/pi05_norray_{server,run}_*.log`、json 到 `results/datas/`。

**跑 full-lance 400 step 时发现并修复的真实 bug**：

1. **【根因 OOM】变长 prompt → 每步重编译 XLA、编译产物不释放 → 显存爬升到
   RESOURCE_EXHAUSTED。** 先崩在第 17 步，加 command-buffer flag 后推到第 33 步
   （只是拖延）。根因：`openpi_pi05_worker._observation_from_payload` 把变长的
   `tokenized_prompt`（driver 按 mask 裁成真实长度）直接喂进 JAX，**没 pad 到
   `max_token_len`**。每个不同 prompt 长度 = 一个新 traced shape = 一次新编译。
   **修复：worker 加 `_padded_prompt()`，把 token+mask pad/截断到固定
   `max_token_len`（pad token 0、mask False）；action worker 同款修复。** 现在 shape
   恒定 → 只编译一次 → 显存全程平稳（~61GB，第 11→37 步几乎不变），干净越过第
   17、33 步，0 OOM。这与 openpi 官方管线一致（tokenizer 本就 pad 到 max_token_len）。
2. **【加固】command-buffer flag + 空闲卡绑定**：脚本设
   `XLA_FLAGS=--xla_gpu_enable_command_buffer=` 与 `CUDA_VISIBLE_DEVICES=3,4,5,6`。
   本机 **GPU 共享**——0/1/2 被别人占了 ~75%，pi05 mesh（fsdp_devices=1，跨全部可见卡）
   落上去没余量。绑定空闲卡给每卡留 ~20GB。（`make_mesh` 要求 device_count %
   fsdp_devices == 0，4 卡安全。）
3. **【driver bug】静默吞掉失败 future**：`_await_result` 把失败返回的
   `{"error":...}`（routes/futures.py `_failed_payload`，HTTP 200）当空 metrics，
   导致第一版 400-run 表面"跑完"实则第 17 步后全是 null loss。**修复：失败 future
   直接抛 RuntimeError 停止。** 另把 driver `_post_json` 超时 120→900s（冷启动首个
   train_step POST 要等权重加载+首编译），并在启动器静默 JAX/absl DEBUG（~9.5万行刷屏）。

**验证结果**：smoke（3 step/40 样本）loss 与进程内 route-check 逐字节一致、act
返回 `[10,32]`、save 产出 `mint://` URI + checkpoint；full-lance 400 step 稳定推进、
显存平稳、越过全部历史崩溃点。日志见 `results/logs/pi05_norray_*`。

**尚未做**：`openpi_fast` 的 HTTP E2E（本轮仍聚焦 pi05）。

**后续推理验证（2026-07-10）**：使用训练后权重对 full-lance 数据集（4 episodes / 887 帧）
进行逐帧推理，产出合并 Lance 数据集（`pi05_replay_merged_noray_20260710_114616.lance`），
包含真值与预测动作（归一化 + 物理空间）。与 Ray 时代同结构数据集对比，逐 episode MSE 等价：

| episode | Ray (2026-07-03) | no-Ray (2026-07-10) |
|---------|------------------|---------------------|
| ep0     | 0.3997           | 0.4013              |
| ep1     | 0.0903           | 0.0922              |
| ep2     | 0.1369           | 0.1391              |
| ep3     | 0.0347           | 0.0360              |

**结论：Ray 拆除无退化，训练收敛与推理质量与 Ray 版本一致。** 详见 `PI05infer_noray.md`。

---

## 4. 已知的模型侧问题（拆分后要处理）

这两条是 pi0.5 模型本身的改动需求，拆出独立开发后一并解决：

**✅ 已定：这两条是同一件事。** 「VLM 改 head」与「32 维不够」是一体两面——
当前 `action_dim=32`（前 26 真实动作 + 后 6 力控维，见 `Openpi_usage.md` §7）
维度不够用，需要改 VLM 的输出 head 来支持更大的动作维度。

> 待展开（不阻塞主方向）：目标维度是多少？改 head 的具体结构、对 flow-matching
> loss 形态与归一化的影响。见 §5 Q3。

---

## 5. 待确认 / 待讨论（下一步对齐这些再动手）

已定（见上文）：
- ✅ Mint API 保留 **HTTP 端点形态**（§2.1）
- ✅ VLM 改 head 与 32 维扩展是**同一件事**（§4）
- ✅ 拆分物理形态 = **本仓库内解耦目录**（§2.2）

已定（补充）：
- ✅ **分支名 "Separate" 的含义**：就是**把 pi0.5 从 Ray 模式里拆出来、不再依赖 Ray**。
  与 LoRA 权重的持久化方式无关。pi0.5 的执行不再走 mint 的 Ray actor / 调度 /
  runtime_env，改用独立的（非 Ray）进程/编排方式跑。

仍待展开：

1. **微调 infra 选型**：pi0.5 脱离 Ray 后用什么跑训练——单进程 / 直接多卡
   （FSDP-on-JAX 或 jax pmap/jit sharding）/ 其它编排？权重如何持久化？
2. **改 head 细节**（不阻塞主方向）：目标动作维度、head 结构、对 flow-matching
   loss 与归一化的影响。
3. **RL 环境职责边界**：mint 侧作为 RL 环境具体承担哪些（rollout / reward /
   数据供给），与独立训练侧通过 §3 三个 API 如何交互形成闭环。

---

## 6. 参考

- 现状全链路与已实测结论：`Openpi_usage.md`（训练 / 推理 / 评估 / 真机接入）。
- 数据与产物存储约定：`wenxi_dev_md/Data_Log.md`。
- mint 如何集成 Tinker 模式（拆分前的架构基线）：见本次讨论——
  mint = Tinker 兼容服务端，靠 `training_backend` 把原语分派到多后端，
  pi0.5 现为其中 `openpi_pi05` 一支。
