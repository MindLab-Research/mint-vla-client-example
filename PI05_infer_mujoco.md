# PI05_infer_mujoco — MuJoCo 闭环真机推理模式(计划)

> 状态：**计划稿，待你确认后再执行**。本文只描述要做什么、为什么、涉及哪些文件与风险。
> 尚未写任何可运行脚本。

## 0. 目标（你的需求）

在 MuJoCo 仿真里模拟"真机推理"闭环：

```
渲染 obs(头/腕相机 + 手 state)
   → 送 fine-tune 的 pi0.5 模型推理(经 mint HTTP)
   → 得到 action_chunk [H, action_dim]
   → 执行完整个 chunk(而非只执行第 0 步)
   → 再渲染新 obs → 循环
```

即 **块内开环、块间闭环**（replan_steps = action_horizon），与 LIBERO eval 的
replan 模式同思路，但动作块整块执行完再重新观测。

## 1. 现状：不是从零开始

仓库里已经有一个高度相关的闭环脚本，本计划是在它基础上**换推理后端 + 改执行粒度**：

| 已有 | 路径 | 与目标的差距 |
|---|---|---|
| MuJoCo 闭环脚本 | `pi-finetune/case/04_closed_loop_mujoco_pi05/closed_loop_mujoco_pi05.py` | ① 用**进程内 JAX policy**(`create_policy`)而非 mint HTTP；② **replan=1**：每帧只执行 `action_chunk[0]` 就重新观测 |
| 单帧 obs→mint 推理 | `mint/scripts/wip/openpi_vla_infer_obs.py` | 已打通"原始图像+state+prompt → transform → mint `/act` → 反归一化"，但**没有仿真闭环**，只推一帧 |
| MuJoCo 场景/渲染 | `pi-finetune/case/01_export_video/export_mano_sim_video.py`（`MjcfBuilder`、`Renderer`、相机、位置执行器） | 直接复用 |
| 物理手/物体/接触 helper | `pi-finetune/case/03_verify_ckpt/verify_lance_hand_physics_mujoco.py`（`add_physical_hand`、`add_position_actuators`、`render_frame`、`load_lance_row` 等） | 直接复用 |

**结论**：新逻辑 = `case/04` 的仿真骨架 + `infer_obs.py` 的 mint HTTP 客户端 + 整块执行。
你已选择：**后端走 mint HTTP API**、**写全新独立脚本**（不改 `case/04`）。

## 1.5 运行环境（已实测确认，决定整个方案形态）

按你的指示对齐 `PI05lance_infer.sh` / `PI05lance.sh` 的用法。实测结论：

- **解释器**：`/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python`（CPython 3.13）。
- **PYTHONPATH**（与两个 PI05lance 脚本完全一致）：
  ```
  ${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src
  # CODE_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint
  # EXTRA_PYDEPS=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
  # GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
  ```
- **这个 venv 实测已具备**：numpy 2.4.6 / requests / PIL / lance 8.0.0 / jax 0.7.2 / openpi（含
  `openpi.training.config`、`openpi.policies.libero_policy`、`openpi.shared.normalize`、`openpi.transforms`）。
- **唯一缺**：`mujoco`、`imageio`（渲染要用）。→ run 脚本里 `uv pip install mujoco imageio` 补进该 venv。
- **主机**：2×A800-80GB + `libEGL_nvidia.so` 齐全 → **无头 EGL 渲染可直接在主机跑，不需要 docker**。
- **mint server**：`curl :30496/api/v1/healthz` = `{"status":"ready"}`，**已就绪、localhost 可达**。

> ⚠️ 这推翻了初版 MD 的"pi-finetune docker 跨网络连 mint"设想。**正确形态：整个闭环在
> gpu_rl host-venv 单进程里跑**——渲染 + transform + HTTP 全在一个 python，`MINT_BASE_URL=http://localhost:30496`。
> §6 的网络风险因此消除。

## 2. 推理后端：mint HTTP（已选定）

fine-tune 权重以 `mint://<model_id>/sampler_weights/<name>` 形式存在（就是你 merged lance
里 `pred_meta.model_path` 那个）。调用链（复用 `infer_obs.py` 已验证的方式）：

1. `POST /api/v1/mint/action_sessions`  body: `{session_id, base_model, model_path(mint://...), owner_id}` → 拿 `action_session_id`
2. 每帧：`POST /api/v1/mint/action_sessions/{id}/act`  body: `{observation: <datum.observation>}` → 返回 `{actions:{data,shape}, policy_timing}`
3. 结束：`DELETE /api/v1/mint/action_sessions/{id}`

**前置**：mint dev server 必须在跑且容器能网络访问到它（见 §6 风险）。
`base_model = "openpi/pi05-libero-low-mem-finetune"`（`PI05_MODEL`）。

### 关键：worker 返回的是**归一化空间**动作
`openpi_pi05_action_worker.act()` 直接返回 `model.sample_actions()` 的原始采样值，**不做输出反归一化**。
所以客户端必须：
- **送入前**：对 obs 做完整 transform（图像 resize 224、state 归一化+pad 到 32、PaliGemma 分词）——即 `_transform_sample` + `_pi05_datum_from_transformed`。
- **拿到后**：自己反归一化 `_unnormalize_actions()`（分位数公式，pi0.5 `use_quantile_norm=True`）得到**物理动作**。

这与进程内 `policy.infer()`（自带输出反归一化）不同，是走 HTTP 必须自己补的一步。

## 3. 动作语义（已用 merged lance 数值验证）

- 训练集 `actions[t] = state[t+1] - state[t]`，是**逐帧 delta**；一个 chunk 是连续 delta。
- 反归一化后的物理动作 = delta，与真值 `actions` 同空间。
- 执行方式（整块，逐步加到**当前实时 qpos**）：
  ```
  for k in range(replan_steps):          # 默认 replan_steps = action_horizon = 10
      current = hand_qpos(data)           # 每步重新读，qpos 在演化
      target  = current[:26] + phys_action[k][:26]
      # 在 substeps 内把 ctrl 线性 ramp 到 target,逐 substep mj_step
  # replan_steps 步执行完 → 重新渲染 obs → 重新推理
  ```
  为什么"加到当前 qpos"对：执行完 action[k-1] 后 qpos≈state[t+k]，再加 action[k]=state[t+k+1]−state[t+k]
  ≈得到 state[t+k+1]。k=0 时 `state[t]+action[0]≈hand_urdf_dof[t+1]`（merged lance 实测误差 0.0016）。
- 维度：`action_dim=32`，有效 `valid_dim=26`（MANO DOF），26:32 是零填充忽略。
  **注意**：不要设 `MINT_OPENPI_PI05_ACTION_OUT_DIM=7`（那是 LIBERO 的），本 case 要保留 32 维。

## 4. 计划产出物

### 4.1 脚本（放 pi-finetune case/05，你已确认保持默认）
`pi-finetune/case/05_closed_loop_mujoco_mint/closed_loop_mujoco_mint.py`

以 `case/04` 的 `run_closed_loop()` 为骨架，**替换/新增**：
- **删**：`load_policy` / `create_policy` / `policy.infer`（进程内 JAX）。
- **加**：mint HTTP 客户端 + 客户端 transform + 反归一化。为避免跨仓库脆弱 import，
  **复制**这几个小函数进新脚本（源自 `mint/scripts/wip/openpi_vla_smoke_lance.py` 与 `openpi_vla_infer_obs.py`）：
  `_headers`、`_post_json`、`_await_result`、`_build_model_config`、`_make_data_config`、
  `_transform_sample`、`_pi05_datum_from_transformed`、`_compute_norm_stats`、`_unnormalize_actions`。
- **改**：执行循环从"只执行 action[0]"→"执行 replan_steps 步"（§3）。
- **保留**：`case/04` 全部 MuJoCo 场景构建、双相机渲染、接触/抬升指标、mp4 与 metrics.json 输出。

**norm_stats 来源（简化）**：不用外部 json。像 `openpi_vla_infer_lance.py` 那样，
**从同一份 full lance 在线 `_compute_norm_stats` 算一次**（RunningStats 遍历 state/actions），
保证与 mint 训练/推理侧分布一致，也免掉 `--norm-stats` 文件依赖。

新增/变化的 CLI 参数（相对 case/04）：
```
--base-url(默认 http://localhost:30496) --api-key(默认 tml-dummy)
--model-path(mint://...) --owner-id       # mint 接入,同 PI05lance_infer.sh
--replan-steps <int>                       # 默认 = action_horizon(整块执行);可设 1 复现 case/04
# 复用: --lance-dataset(默认 full lance,取初始位姿+prompt) --row-index --start-frame
#       --max-steps --substeps --fps --width --height --action-dim 32 --valid-dim 26 --action-horizon 10
```
> 说明：仍需一个 lance row 提供**初始手/物体位姿**和 **prompt**（首帧 reset 用）；这是"仿真起点"
> = `pi_video_streams_full_lance.lance`（你指定），之后 obs 全部来自 MuJoCo 实时渲染，不再回放 lance。

### 4.2 运行 wrapper
`pi-finetune/case/05_closed_loop_mujoco_mint/run_closed_loop_mujoco_mint.sh`
**仿 `PI05lance_infer.sh`**（不是 case/04 的 docker 方式）：
- 用 gpu_rl host-venv 解释器 + 同款 PYTHONPATH（见 §1.5），额外把 `case/05`、`case/01`、`case/03`
  加进 PYTHONPATH 以 import 渲染 helper。
- 首次运行 `uv pip install mujoco imageio` 补进该 venv（或装进 `${EXTRA_PYDEPS}`）。
- 设 `MUJOCO_GL=egl`、`MINT_BASE_URL=http://localhost:30496`、`MINT_API_KEY=tml-dummy`，
  `--model-path` 由 `MINT_SAMPLER_PATH` 或从上次训练输出 json 的 `save_result.path` 解析（同 infer 脚本）。
- 先 `curl healthz` 探活再跑。

### 4.3 本文档
`mint/PI05_infer_mujoco.md`（即本文）——记录设计、接口、语义、风险，供另一 Claude 同步。

## 5. 数据流（一次 replan 周期）

```
MuJoCo data.qpos ──► current_hand(26) ──► pad→state(32)
MuJoCo Renderer  ──► head_img, wrist_img (HWC uint8)
        │
        ├─ _transform_sample: LiberoInputs(2→3相机,右腕零填充&mask) → Normalize(state) → resize224 → PaliGemma分词 → pad32
        ├─ _pi05_datum_from_transformed → observation payload(3 image chunks + encoded_text + state)
        ▼
   POST /act ──► actions[H,32] (归一化)
        ▼
   _unnormalize_actions ──► phys_action[H,32] (物理 delta)
        ▼
   for k in replan_steps: target=qpos_now+phys_action[k][:26]; ramp ctrl; mj_step×substeps
        ▼
   重新渲染 → 下一周期
```

## 6. 风险 / 前置条件（更新：多数已消除）

1. ~~mint server 可达~~ ✅ **已解除**：闭环在 gpu_rl host-venv 单进程跑，`localhost:30496` healthz=ready，无跨网络问题。
2. **权重形式**：`--model-path` 用 `mint://...`（产 merged lance 那次的同一 sampler）；沿用 `PI05lance_infer.sh`
   的解析方式（`MINT_SAMPLER_PATH` 或训练输出 json 的 `save_result.path`）。**执行前需确认该权重当前仍能被 action_sessions 加载。**
3. ~~norm_stats 一致性~~ ✅ **已解除**：改为从同一份 full lance 在线 `_compute_norm_stats`，与训练侧同源。
4. ~~EGL 无头渲染~~ ✅ **已确认**：主机 2×A800 + libEGL_nvidia 齐全，`MUJOCO_GL=egl` 可无头渲染。
5. **缺依赖**：host-venv 缺 `mujoco`、`imageio` → run 脚本首次 `uv pip install` 补上（其余 numpy/requests/PIL/lance/jax/openpi 已齐）。
6. **性能**：每 replan 一次 HTTP 往返 + flow-matching 采样；整块执行(replan=action_horizon=10)已把调用次数降到最低。
7. **numpy 2.x**：host-venv 是 numpy 2.4.6；渲染 helper 与 openpi transform 实测可 import，注意别混入 numpy<2 的包。

## 7. 你的决定（已确认）

1. **脚本落位**：`pi-finetune/case/05_closed_loop_mujoco_mint/`（保持默认）。✅
2. **仿真起点**：`/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance`
   （已确认存在），默认 `--row-index 0 --start-frame 0`。✅
3. **mint 用法**：对齐 `PI05lance_infer.sh` / `PI05lance.sh`——host-venv 解释器、同款 PYTHONPATH、
   `MINT_BASE_URL=http://localhost:30496`、`MINT_API_KEY=tml-dummy`、`model=openpi/pi05-libero-low-mem-finetune`、
   `--model-path` 走 sampler 解析。✅

### 仍需你在执行前给我的一个值
- **`--model-path` (mint://...)**：用哪个 sampler 权重？
  - 选项 A：设 `MINT_SAMPLER_PATH=mint://...` 显式给我；
  - 选项 B：让脚本从训练输出 json 自动解析（`/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json`
    的 `save_result.path`）——需确认该 json 存在且指向你要的权重。
  （其余全部用默认，无需再定。）

确认 model-path 来源后，我落地 §4.1 / §4.2 的脚本。
