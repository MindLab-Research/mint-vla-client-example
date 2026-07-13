# 第二步计划：openpi 彻底脱离 Ray（进程内联执行）

## 采用的默认决策（基于"放弃 Ray、其他服务暂不考虑"）
- **范围**：只让 openpi 脱 Ray。删 openpi 专属的 3 个 Ray runtime 文件；共享的
  scheduler / ModelEngineHost / ModelActorSupervisor **物理保留**，openpi 不再走它们
  （不动共享系统，零成本地不破坏 megatron/verl/qwen）。
- **执行位置**：openpi 训练/推理引擎**在 API 进程内联跑**（单进程、串行），契合 A 单租户。
- **future**：保留 `/retrieve_future` 轮询契约，内部极简（算完立即 resolve）。

## 关键发现（决定方案形态）
- worker 会话代码（pi05 + fast，训练 + 推理）**已 100% 无 Ray**，只说 `_dispatch(op,payload)`。
- Ray 只在两层：① openpi 的 3 个 `*_ray_runtime.py`（把 worker 塞进 actor）；
  ② 生产请求路径 `enqueue_model_work → model_work_scheduler(Ray actor) → ModelEngineHost(Ray actor) → engine`。
- `app.py:359-366` 在 API 进程把 `training_engine=None`，引擎目前只活在 Ray actor 里。
- **但** `_do_forward_backward`（training.py:3015）已是完整的内联逻辑：取 session →
  materialize → `engine.forward_backward(session, request)` → `task_futures.async_resolve`。
  它跑在 Ray actor 内只是因为 `_current_training_engine()` 返回 actor-local 引擎。
- `_current_training_engine/manager()` 无 execution context 时**回退到模块全局**
  `training_engine/training_manager`——这正是内联注入点。

## 实施步骤

### A. 进程内注入 openpi 引擎（不动共享 infra）
1. 新增一个"本地 openpi 运行时"装配模块，在 API 进程启动时构造：
   - `OpenPIPi05TrainingEngine(runtime_factory=make_local_pi05_runtime)`（第一步已验证）
   - openpi fast 引擎同法用 `OpenPIDirectWorkerClient` 的本地 factory
   - 一个进程内 `TrainingSession` 注册表（dict）+ 动作会话注册表，取代 detached actor 持久化
2. `app.py`：仅当模型是 openpi 后端时，把 `training.training_engine/manager` 绑到这个本地引擎；
   其余服务保持 `None`（仍走 Ray）。

### B. 路由内联分支（openpi 绕开调度器）— 已纠正
**关键更正（2026-07-09）**：不能复用现有 `_do_*` + `task_futures`。因为
`task_futures` 坐在 `task_state_store = TaskStateStoreClient()` 上，后者是 **detached
Ray actor 的代理**；`async_upsert_training_session` 同一个 actor。复用它们仍需 Ray。

正确做法：openpi 走**自包含的进程内路径**，不碰共享 Ray store：
- 新模块 `openpi_local_execution.py`：进程内 openpi 引擎（本地 factory）+
  **进程内 session dict** + **进程内 future dict**（极简，算完即 ready）。
- create_model / forward_backward / optim_step / train_step / save_weights_for_sampler
  / vla_train_step / act 的 handler：`backend ∈ {openpi_pi05, openpi_fast}` 时走这套
  进程内路径，future 存进程内 dict，**不进** `enqueue_model_work` / `task_futures`。
- `routes/futures.py` 的 `/retrieve_future` 加 openpi 分支：先查进程内 future dict。
- 非 openpi 后端一字不动（仍走 Ray scheduler + task_futures）。
- 只加分支、不动原有 billing/auth/gateway 逻辑。

### C. 引擎与 session 管理去 Ray
1. `openpi_pi05_training.py`：`_default_runtime_factory` 改为本地 factory；
   `initialize()` 去掉 `ensure_openpi_ray_initialized()`。
2. `action_session_manager.py`：pi05/fast 动作 factory 默认走 `OpenPIDirectWorkerClient`
   （去掉 "must be reconciled by supervisor" 的 Ray 前置），动作 session 存进程内 dict。
3. session 持久化：openpi 不再依赖 detached actor；用进程内注册表 +（可选）落盘元数据。

### D. 删除 openpi 专属 Ray 文件
- `openpi_ray_runtime.py`、`openpi_shared_ray_runtime.py`、`openpi_action_ray_runtime.py`
- 清理对它们的 import（training.py 引擎侧、action_session_manager 侧）。
- **保留** `openpi_direct_runtime.py`、`openpi_fast_runtime.py`、`openpi_fast_action_runtime.py`、
  所有 worker、`openpi_orbax_compat.py`。
- **不删** `mint_server/ray/`、`scheduling/`、`actors/` 等共享模块（其他服务在用）。

### E. 验证
1. 复用第一步脚本确认引擎级三件套仍通过。
2. 起 server，用 `openpi_vla_smoke_lance.py`（真 HTTP 路径）跑
   create_model → train_step → save_weights_for_sampler → action session → act，
   全程无 Ray（可 `ray stop` 或不 init Ray 验证）。
3. 确认 loss 下降、sampler 导出结构与 Ray 路径一致、act 返回动作。

## 风险与边界
- 内联执行会阻塞 API worker（单租户 A 可接受；并发靠以后再加本地队列）。
- 共享 scheduler 里 `mint.vla.train_step`/`mint.action.act` 两个 op 名保留无害（不再触发）。
- 不触碰其他后端的任何路径。

## 产物文档
完成后更新 `OpenPI_Separate.md`：把 §2.2 从"从 Ray 拆出"升级为"已删除 Ray"，
记录内联执行架构、删除的文件清单、验证结果。
