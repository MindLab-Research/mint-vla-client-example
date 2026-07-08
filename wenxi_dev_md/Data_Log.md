# Data_Log.md — pi0.5 VLA 数据与日志目录说明

> 面向读取本文的 agent：本文档描述 wenxi 的 pi0.5（openpi VLA）实验产物的**存储约定**、
> 每个文件的**含义与来源**、以及**哪个是当前最新有效产物**。若你要修改脚本的输出路径，
> 请遵守下面的目录约定。最后更新：2026-07-07。

---

## 0. 存储约定（新规定，必须遵守）

- **所有数据和日志的主目录 = `/vePFS-Mindverse/user/intern/wenxi/results`**
- **不要再往 `/tmp` 写产物。** 历史 /tmp 产物已迁移至此。
- 两个固定子目录：
  | 目录 | 放什么 |
  |---|---|
  | `results/datas/` | 数据产物：`.json`（训练/推理/评估结果、norm_stats）、`.lance`（数据集）、图像目录、`.zip` |
  | `results/logs/`  | 运行日志：各脚本 stdout/stderr 重定向的 `.log` |

- 相关代码仓库在 `/vePFS-Mindverse/user/intern/wenxi/mint`（脚本 `PI05*.sh`、`scripts/wip/openpi_vla_*.py`）。
  这些脚本的**默认输出路径已改为指向本 results 目录**（见第 3 节）。

---

## 1. 当前最新有效产物（要用就用这些）

| 产物 | 路径 | 说明 |
|---|---|---|
| **训练权重**（sampler） | `mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9` | 887帧/32维数据集, 400步, batch=2, loss 0.042→0.006。落盘于 `/vePFS-Mindverse/share/mint/dev/data/checkpoints/anonymous/lance-smoke-440c1d1c7107_0/...`（不在 results 下，由 server 管理） |
| **训练结果 JSON** | `datas/pi05_lance_smoke.json` | 上次训练的完整记录：model_id、`save_result`（含权重 mint:// 路径）、400 步逐步 metrics |
| **归一化统计** | `datas/pi05_norm_stats.json` | 从训练数据集导出的 state/actions mean/std/q01/q99（各 32 维）。**推理反归一化必须用这份同源统计** |
| **合并回放数据集** | `datas/pi05_replay_merged.lance` | 887 帧逐帧推理 + 合并回原 lance。15 列 = 原 11 + 预测 4（见第 2 节）。**replay 用这个** |
| 源训练数据集 | `/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance` | 4 episodes / 887 帧 / 32 维（前26真实动作+后6力控维）。不在 results 下 |

---

## 2. datas/ 详细清单

### 最新主产物
- **`pi05_replay_merged.lance`** — 合并回放数据集（当前主产物）。
  - 原始列：`index, episode_metadata, camera, prompt, timestamp, frame_index, image, wrist_image, state, actions, mujoco`
  - 追加预测列：
    | 列 | 结构 | 含义 |
    |---|---|---|
    | `pred_actions` | [frame][10,32] | 每帧预测的未来 10 步动作块（归一化空间） |
    | `pred_actions_physical` | [frame][10,32] | 反归一化物理量，**与源 `actions` 同空间，replay 用这个** |
    | `pred_action_mse` | [frame] | 每帧 pred vs 真值窗口 MSE（归一化空间） |
    | `pred_meta` | struct | `{model_path, action_horizon=10, action_dim=32}` |
  - 质量：整体 MSE mean=0.165 / median=0.062；每个 episode 前 8 帧偏高（均值 1.45，起始状态歧义），第 8 帧起降到 0.117。
- **`pi05_lance_smoke.json`** — 训练结果（含权重路径、逐步 loss）。
- **`pi05_norm_stats.json`** — 归一化统计（推理必需）。

### 其它历史产物（保留供参考）
- `pi05_infer.json` — 纯推理输出（`results` 列表，每条含 action chunk）。来自 `PI05lance_infer.sh`。
- `pi05_eval.json` — 验证集 MSE 量化评估（`per_sample` + `aggregate`）。来自 `openpi_vla_eval_lance.py`。
  **当前权重**（`440...c982d5e9`）8 样本结果：稳定帧 mean_mse=0.111 / median=0.077（约零基线 40%）；
  起始帧 idx0=6.17（初始状态歧义，已知现象，不代表质量）。运行日志 `logs/pi05_eval_440.log`。
- `pi05_obs_infer.json` — 单帧 obs 推理（`action_chunk`，真机接入用），含 `policy_timing`。
- `pi05_obs/` — 单帧推理用的样例观测：`main.png`、`wrist.png`、`state.json`、`prompt.txt`。
- `pi05_replay.lance` — 早期版本的合并数据集（已被 `pi05_replay_merged.lance` 取代）。
- `replay_frames/` — 导出的每帧主相机图（来自 `read_replay_lance.py --save-frames`）。
- `pi05_replay_merged.zip` — merged lance 的打包快照。
- `pi05_check.json` / `pi05_check_result.json` — `PI05check.sh` 端到端冒烟结果。
- `pi05_log_boundary.txt` — 调试用日志边界标记（可忽略）。

---

## 3. logs/ 清单

各脚本运行日志（stdout/stderr）：
- `pi05_merge.log` — 887 帧合并推理的逐帧日志（含每帧 MSE）。
- `pi05_lance_smoke.log` / `pi05_train400.log` / `pi05_full_train.log` — 训练运行日志。
- `pi05_check.log` / `pi05_8gpu_run.log` / `pi05_cachetest.log` / `pi05_probe.log` — 各类冒烟/探针日志。
- `pi05_run*.log` / `pi05_final.log` — 早期运行日志。
- `mint_dev_launch_wenxi.snapshot.log` — **server launcher 日志的快照副本**（截至 2026-07-07）。
  ⚠️ 活动日志原件仍在 `/tmp/mint_dev_launch_wenxi.log`，由 server 进程实时写入
  （`step3.sh`/`step3_start.sh` 里硬编码）。要让 server 也写到 results 下，需改这两个
  启动脚本的重定向路径并**重启 server**（Python server 不热加载）。本次未动，避免打断运行中的 server。

---

## 4. 脚本默认输出路径（已改为 results/，供你核对/修改）

改代码时如需调整输出位置，相关默认值在：
| 脚本 | 变量/参数 | 现默认值 |
|---|---|---|
| `PI05lance.sh` | `MINT_PI05_OUTPUT_JSON` | `results/datas/pi05_lance_smoke.json` |
| `PI05lance_infer.sh` | `MINT_PI05_OUTPUT_JSON` / `MINT_PI05_INFER_JSON` | `results/datas/pi05_lance_smoke.json` / `results/datas/pi05_infer.json` |
| `PI05check.sh` | `MINT_PI05_OUTPUT_JSON` | `results/datas/pi05_check.json` |
| `openpi_export_norm_stats.py` | `--output` | `results/datas/pi05_norm_stats.json` |
| `read_replay_lance.py` | argv[1] 默认 | `results/datas/pi05_replay_merged.lance` |
| `openpi_vla_merge_infer_lance.py` | `--output-lance`（无默认，示例已更新） | 传参时请指向 `results/datas/` |

> 仍属 /tmp 的 `mint_dev_run.env`、`mint_dev_launch_wenxi.log` 是 **server 基础设施**（不是实验产物），
> 由 `step2*.sh`/`step3*.sh` 管理，改动需重启 server，本次未迁移。

---

## 5. 快速复现命令

环境（所有命令通用）：
```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
export PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/code/conley/.openpi_cache
export HF_HOME=/vePFS-Mindverse/share/huggingface
export MINT_BASE_URL=http://localhost:30496 MINT_API_KEY=tml-dummy
PY="${GRB}/host-venv/bin/python"
```

读回放（最常用）：
```bash
$PY scripts/wip/read_replay_lance.py \
  /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged.lance
```

重新生成合并数据集（逐帧推理，约 25 分钟 / 887 帧）：
```bash
DS=/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance
$PY scripts/wip/openpi_export_norm_stats.py --lance-dataset "$DS"   # -> results/datas/pi05_norm_stats.json
$PY scripts/wip/openpi_vla_merge_infer_lance.py \
  --model-path mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9 \
  --norm-stats /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json \
  --lance-dataset "$DS" \
  --output-lance /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged.lance
```

> 训练/推理/真机接入的完整说明见仓库根的 `Openpi_usage.md`。本文档只管数据与日志的存储位置。
