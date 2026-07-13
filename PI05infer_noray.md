# PI05infer_noray — no-Ray pi0.5 推理产物（供渲染回放 pred_action）

> 面向读取本文的另一个项目：这里给出**训练后权重**、**逐帧推理写回的合并
> Lance 数据集**、以及产出这些的**脚本与流程**。目标是让你直接读取这个 Lance，
> 对每一帧渲染回放「真值 action vs 预测 action」。
> 最后更新：2026-07-10。本次训练与推理**全程不依赖 Ray**（openpi 进程内联执行）。
>
> **另有一份 Ray 时代的同结构对照数据集**（§7），schema 完全一致，可与本 no-Ray
> 版并排做 A/B 回放对比。实测两者逐 episode MSE 等价（拆 Ray 无退化）。

---

## 0. TL;DR（要用就用这些）

- **合并后的 Lance 数据集（读它做回放）**：
  ```
  /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged_noray_20260710_114616.lance
  ```
  4 episodes / 887 帧 / 15 列（原 11 + 预测 4）。97 MB。

- **训练后的权重（sampler）**：
  - 逻辑地址（mint server 内部解析）：
    `mint://lance-smoke-81cc60d6da8d_0/sampler_weights/lance_smoke_sampler_cd0abb48`
  - 落盘真实路径：
    `/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints/persistent_cache/000000000000000000000001/lance-smoke-81cc60d6da8d_0/lance_smoke_sampler_cd0abb48/sampler`
  - owner_id：`000000000000000000000001`

- **归一化统计（反归一化必须用这份同源统计）**：
  ```
  /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats_full_noray.json
  ```

- **源数据集（推理输入 / 真值来源）**：
  ```
  /vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance
  ```

**回放要用的两列**：`actions`（真值，物理空间）对比 `pred_actions_physical`
（预测，已反归一化到与 `actions` 同一物理空间）。

---

## 1. 这份数据是怎么来的

pi0.5（openpi flow-matching VLA）在 mint 上做了 **LoRA 微调，400 step**（无 Ray，
进程内联跑），loss 从 ~0.08 收敛到 ~0.03。然后用训练后的 sampler 权重，对源
数据集**逐帧推理**：站在第 f 帧的当前观测（图像+state+prompt），一次预测未来
`action_horizon=10` 步、每步 `action_dim=32` 维的动作块，写进 `pred_actions[f]`。

> ⚠️ 诚实提醒：推理用的是**训练集本身**（无 train/val split），所以这验证的是
> 「模型是否学会拟合训练轨迹」，不是泛化能力。要看泛化需用留出集。

---

## 2. Lance 列结构（渲染回放要读的）

原始 11 列原样保留：
`index, episode_metadata, camera, prompt, timestamp, frame_index, image,
wrist_image, state, actions, mujoco`

追加 4 列（外层长度 = 该 episode 的 total_frames，与 `actions` 逐帧对齐）：

| 列 | pyarrow 类型 | 含义 |
|---|---|---|
| `pred_actions` | `list<list<list<float>>>` → `[frame][10][32]` | 每帧预测的未来 10 步动作块（**归一化空间**） |
| `pred_actions_physical` | `list<list<list<float>>>` → `[frame][10][32]` | 同上**反归一化物理量**，与源 `actions` 同空间，**回放用这个** |
| `pred_action_mse` | `list<float>` → `[frame]` | 每帧 pred vs 真值窗口的 MSE（归一化空间） |
| `pred_meta` | `struct{model_path:str, action_horizon:int32=10, action_dim:int32=32}` | episode 级元信息 |

对第 f 帧：`actions[f]` 是 32 维真值当前动作；`pred_actions_physical[f]` 是
`[10,32]`，其中 `[0]` 是模型对「下一步」的预测，`[k]` 是往后第 k 步。回放对比时
最常用「预测首步 `pred_actions_physical[f][0]` vs 真值 `actions[f]`」，或整块
10 步展开做 open-loop 回放。

> 动作维度语义：`action_dim=32` = 前 26 维真实动作（MANO DOF）+ 后 6 维力控维。
> 有效自由度是前 26 维，26:32 多为零/常量。渲染时按你的机器人/手模型取对应维。

---

## 3. 最简读取示例

参考脚本 `scripts/wip/read_replay_lance.py`（逐帧打印真值+预测首步）。核心读法：

```python
import lance, numpy as np
path = "/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged_noray_20260710_114616.lance"
rows = lance.dataset(path).to_table().to_pylist()
for ep, r in enumerate(rows):
    frames = int(r["episode_metadata"]["total_frames"])
    for f in range(frames):
        gt   = np.asarray(r["actions"][f], dtype=np.float32)                 # (32,) 真值
        pred = np.asarray(r["pred_actions_physical"][f], dtype=np.float32)   # (10,32) 预测块(物理)
        mse  = r["pred_action_mse"][f]
        img  = r["image"][f]         # 主相机帧(渲染用)
        # 渲染回放:用 pred[0][:26] 或整块 pred[:, :26] 驱动手模型,与 gt[:26] 对比
```

> `lance` 库在本环境从 PYTHONPATH 提供（`mint_env/extra-pydeps/lance`，pylance 8.0.0）。
> 用 gpu_rl host-venv 解释器 + 该 PYTHONPATH，或你项目里自带的 pylance 均可。

---

## 4. 逐 episode 质量（本次实测）

`pred_action_mse`（归一化空间）逐 episode 统计：

| episode | frames | mse min / mean / max |
|---|---|---|
| ep0 | 221 | 0.013 / **0.401** / 6.169 |
| ep1 | 222 | 0.016 / 0.092 / 0.438 |
| ep2 | 222 | 0.018 / 0.139 / 0.546 |
| ep3 | 222 | 0.014 / **0.036** / 0.157 |

- ep1/ep2/ep3 的 mean 都在 0.04–0.14，模型追踪良好；ep3 最干净。
- **ep0 的 mean 被少数「脏帧」拉高**：这些帧归一化真值里 **dim20 = -34.79**（正常应在
  [-1,1]）。根因是 norm_stats 在该维 std 极小（~3.6e-5，某些维甚至 std=0），把微小
  偏差放大成几十——是**数据/归一化固有问题，非训练 bug**（Ray 时代同一批帧同样异常）。
  渲染时若遇到 mse 特别大的帧（对照 `pred_action_mse[f]`）可视作离群、单独标注。

---

## 5. 产出流程与脚本（复现 / 追溯）

全部在仓库 `/vePFS-Mindverse/user/intern/wenxi/mint`，全程 Ray-free（openpi 进程内联）。

| 步骤 | 脚本 | 产物 |
|---|---|---|
| ① 训练 400 step | `scripts/vla/PI05lance_local_norray.sh` | 权重 `mint://...cd0abb48` + `results/datas/pi05_norray_run_20260710_031310.json` |
| ② 导出 norm_stats | `scripts/wip/openpi_export_norm_stats.py` | `results/datas/pi05_norm_stats_full_noray.json` |
| ③ 逐帧推理写回 | `scripts/vla/PI05lance_local_merge_infer.sh` → driver `scripts/wip/openpi_vla_merge_infer_lance.py` | 本合并 Lance |
| （量化评估，旁证） | `scripts/vla/PI05lance_local_eval.sh` → `scripts/wip/openpi_vla_eval_lance.py` | `results/datas/pi05_eval_run_*.json` |
| 读取参考 | `scripts/wip/read_replay_lance.py` | — |

底层执行说明（供追溯，不影响读数据）：openpi 训练/推理不走 Ray，由
`mint_server/backend/openpi/openpi_local_execution.py` 在 API 进程内联执行；
server 由 `scripts/wip/_run_local_openpi_server.py` 单 worker 启动。设计与验证
详见 `OpenPI_Separate.md` §3.7–3.8。

一键复现推理写回（若要重跑；约 15–100 分钟，887 帧）：
```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint
bash scripts/vla/PI05lance_local_merge_infer.sh
# 自动取最新训练 json 的权重 + full lance,输出 results/datas/pi05_replay_merged_noray_<stamp>.lance
```

---

## 6. 日志（排查用）

- 推理运行日志：`results/logs/pi05_merge_noray_run_20260710_114616.log`（逐帧 mse）
- server 日志：`results/logs/pi05_merge_noray_server_20260710_114616.log`
- 训练日志：`results/logs/pi05_norray_run_20260710_031310.log`

---

## 7. Ray 时代对照数据集（A/B 回放用）

存在一份 **Ray 时代**产出的同结构合并数据集，可与本 no-Ray 版并排回放对比：

- **Ray 对照 Lance**：
  ```
  /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged.lance
  ```
  4 episodes / 887 帧 / **15 列，schema 与 no-Ray 版完全一致**（同样的
  `pred_actions / pred_actions_physical / pred_action_mse / pred_meta`）。97 MB，产于 2026-07-03。
- 它用的是 **Ray 时代训练的权重**（`pred_meta.model_path` =
  `mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9`；
  该权重的落盘目录现已不在，但**回放不需要权重**——预测已固化进 Lance）。

**读法与 no-Ray 版完全相同**（§3 的代码原样可用，只换 path）。两份都是对同一份
源数据集（`pi_video_streams_full_lance.lance`）逐帧推理，因此可逐 episode / 逐帧
并排对比「Ray 权重预测 vs no-Ray 权重预测 vs 真值」。

**实测逐 episode MSE（归一化空间）等价** —— 拆 Ray 后无退化：

| episode | Ray (7/3) mse mean | no-Ray (7/10) mse mean |
|---|---|---|
| ep0 | 0.3997 | 0.4013 |
| ep1 | 0.0903 | 0.0922 |
| ep2 | 0.1369 | 0.1391 |
| ep3 | 0.0347 | 0.0360 |

（ep0 mean 偏高同样来自 dim20 脏帧，见 §4；两版一致。）

> 另有更早的 `results/datas/pi05_replay.lance`（2026-07-03，**18 列的旧 schema**：
> 逐样本一行，含 gt/pred 的 norm+physical，2 rows）。它与本文的 15 列回放 schema
> **不同**，已被 `pi05_replay_merged.lance` 取代，回放对比请用上面的 merged 版。
