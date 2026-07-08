# OpenPI pi0.5 使用指南（训练 + 推理 + 真机接入）

> 面向接手的人：读完本文你能跑通 pi0.5 的训练、推理、评估，并把训练好的
> 权重接到真实机器人。所有路径、脚本、命令均已在 dev 环境实测通过。
> 最后更新：2026-07-07。数据/日志产物统一存放于
> `/vePFS-Mindverse/user/intern/wenxi/results`（见 `Data_Log.md`），不再用 /tmp。

---

## 0. TL;DR（我只想快速用）

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint

# ① 确认 server 就绪（没起就看第 2 节启动流程）
curl -s http://localhost:30496/api/v1/healthz   # 期望 {"status":"ready"}

# ② 用【已训练好的权重】直接推理（不重训）
bash PI05lance_infer.sh                          # 对 lance 样本推理,输出 action chunk

# ③ 量化评估（证明训练有效:MSE vs 真值）
#   见第 5 节

# ④ 真机接入:见第 6 节(接口已留好,替换 obs 来源即可)
```

已训练权重（最新：887 帧 / 32 维数据集, 400 步, loss 0.042→0.006, 可直接用）：
```
mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9
```

---

## 1. 这是什么模型 / 训了什么

- **模型**：pi0.5（openpi），flow-matching VLA（视觉-语言-动作）。
  主干 = PaliGemma（Gemma 2B 语言 + SigLIP 视觉）+ action expert（gemma_300m）。
- **训练方式**：**LoRA 微调**，不是全量。
  - `paligemma_variant="gemma_2b_lora"` → 主干插 LoRA。
  - 冻结过滤器 `freeze_filter = Not(PathRegex(".*lora.*"))` → **只更新名字含 lora 的参数**。
  - ⚠️ **action expert 未被训练**：它是 `gemma_300m`（不带 lora），被上面的过滤器整个冻结。
    （openpi 官方 `get_freeze_filter` 默认会训 action expert 全量参数，但本 worker
    用了更严格的自定义过滤器，绕过了官方逻辑。如需训 action expert 见第 8 节。）
- **loss**：`flow_matching`。用样本里的 `actions`（归一化后）做扩散去噪回归目标。
- **输出**：一次推理产出 **action chunk**，shape `[action_horizon=10, action_dim=32]`
  —— 未来 10 步、每步 32 维的动作序列（前 26 维真实动作 + 后 6 维力控维，见第 7 节）。
  **输出在归一化空间**（见第 7 节反归一化）。

---

## 2. 环境与启动

### 2.1 关键坐标

| 项 | 值 |
|---|---|
| server 端口 | `30496`（namespace hash 派生） |
| Ray namespace | `mint_wenxi_dev` |
| server 代码根（worker 可见） | `/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi` |
| 本地仓库（改代码处） | `/vePFS-Mindverse/user/intern/wenxi/mint` |
| client 解释器（openpi+jax） | `.../mint_env/runtime/gpu_rl/host-venv/bin/python` |
| lance 数据集（当前默认） | `/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance`（4 ep / 887 帧 / 32 维） |
| 数据·日志产物根 | `/vePFS-Mindverse/user/intern/wenxi/results`（`datas/` + `logs/`，见 `Data_Log.md`） |
| server 日志 | `/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log` |
| launcher 日志（真异常在这） | `/tmp/mint_dev_launch_wenxi.log` |

### 2.2 冷启动流程（server 没起时）

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint
bash step0_ray_up.sh        # 单机起 Ray(head 0 GPU + worker 8 GPU)
bash step1.sh               # rsync 本地代码 -> server 代码根
bash step3.sh               # 启动 dev API server
bash step4_health.sh        # 健康检查,期望 {"status":"ready"}
```

### 2.3 干净重启（改了代码 / 状态乱了）

Python server **不热加载**，改完代码必须重启。且它依赖一批 detached Ray actor，
不随进程退出——必须先 kill 再重启：

```bash
bash step1.sh                    # 同步最新代码到 server 代码根
bash step2_clean_restart.sh      # 清 detached actor + 重启 + healthz
sleep 20                         # 等 scheduler lease 收敛
```

---

## 3. 训练（lance 数据集 → LoRA 微调）

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint
bash PI05lance.sh                                    # 默认 400 步,batch 2,full 数据集
MINT_PI05_STEPS=2000 MINT_PI05_BATCH=4 bash PI05lance.sh
MINT_LANCE_DATASET=/path/to/your.lance bash PI05lance.sh   # 换数据集
```

`PI05lance.sh` 是**一条龙**：`create_model → 训 N 步 → save_weights_for_sampler
→ 建 action_session → act(推理一次) → 清理`。driver 是
`scripts/wip/openpi_vla_smoke_lance.py`，走完整 openpi transform
（`LiberoInputs → Normalize → PaliGemma 分词 → Pad`）。

结果写到 `/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json`。

> ⚠️ **两个坑**（务必知道）：
> 1. driver 结束时会在 `finally` 里 **删掉 model_id 和 action_session**。
>    所以训完「活模型」没了，但 **LoRA 权重已落盘**（见第 4 节），推理复用落盘权重。
> 2. `PI05lance.sh` 的 "OK" 判定不严谨（只看 save 有没有 path，不看 train）。
>    **务必严格核对**：
>    ```bash
>    python3 -c "
>    import json; d=json.load(open('/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json'))
>    tr=d.get('train_result') or {}
>    print('train:', 'FAIL' if set(tr)<={'error','category'} else 'OK loss='+str((tr.get('metrics') or {}).get('loss:mean')))
>    "
>    # 且用 stat -c %y /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json 确认是新鲜写入,别读到旧数据
>    ```

---

## 4. 训练权重保存在哪里

`save_weights_for_sampler` 会把权重存成 **sampler checkpoint**，同一份内容落两处
（真拷贝，非软链）：

**① 运行时缓存（推理直接读这个）**
```
/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints/persistent_cache/
  anonymous/lance-smoke-440c1d1c7107_0/lance_smoke_sampler_c982d5e9/sampler/
```
**② 持久化镜像（备份）**
```
/vePFS-Mindverse/share/mint/dev/data/checkpoints/
  anonymous/lance-smoke-440c1d1c7107_0/lance_smoke_sampler_c982d5e9/sampler/
```

目录结构（每份 **5.0G**）：
```
sampler/
├── params/          ← 权重本体(Orbax/OCDBT),真实数据在 params/ocdbt.process_0/d/
├── assets/          ← 各机器人平台预置 norm_stats(libero 等,非本次微调统计)
└── metadata.json    ← model_id / step=400 / backend=openpi_pi05 / mirror_status
```

**关键说明**：
- 5.0G 是 **base+LoRA 合并后的完整权重**（不是几十 MB 的 LoRA 增量），
  这样 sampler 能独立加载推理，不依赖原始 base 文件。
- `optimizer_present: false` —— **不含优化器状态，不能续训**。续训需要训练 checkpoint
  （另一套，在 `MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR`）。
- 你在代码里引用它用逻辑地址（server 内部解析到位置①）：
  ```
  mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9
  ```

---

## 5. 推理与评估

### 5.1 用 lance 样本推理

```bash
bash PI05lance_infer.sh                          # 自动读上次训练的 sampler,推理 index 0
MINT_INFER_INDICES=0,1,2 bash PI05lance_infer.sh # 一次推多帧
MINT_SAMPLER_PATH=mint://... bash PI05lance_infer.sh  # 指定别的权重
```
driver = `scripts/wip/openpi_vla_infer_lance.py`：**只建 action_session + act，
不 create_model、不 train**（训推解耦）。结果写 `/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_infer.json`。

### 5.2 量化评估（证明训练有效）

driver = `scripts/wip/openpi_vla_eval_lance.py`：对样本推理，与 lance **真值动作**
比 MSE/L1，并和「全 0 预测」零基线对比。

```bash
SAMPLER=$(python3 -c "import json;print(json.load(open('/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json'))['save_result']['path'])")
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/code/conley/.openpi_cache
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
export MINT_BASE_URL=http://localhost:30496 MINT_API_KEY=tml-dummy
"${GRB}/host-venv/bin/python" scripts/wip/openpi_vla_eval_lance.py \
  --model-path "$SAMPLER" --indices 0,1 --output-json /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_eval.json
```

**实测结果**（当前权重 `440c1d1c7107...c982d5e9`, 8 样本 idx 0/100/220/300/443/600/700/886）：
- **稳定帧（7 样本，剔除起始帧 idx0）**：mean_mse=**0.111**，median=**0.077**，
  约为零基线的 **40%** → 训练确实起作用。
- **起始帧 idx0**：mse=**6.17**（≈零基线 6.42，几乎无改善）——每个 episode 第 0 帧
  初始状态歧义最大，是已知现象；含它的 8 样本整体 mean 被拉到 0.868，不代表模型质量。
> 结论：**按稳定帧看，当前权重 MSE 降到零基线的 ~40%**。评估起始帧无意义（obs 信息不足）。
> 数据在 `results/datas/pi05_eval.json`，日志 `results/logs/pi05_eval_440.log`。

> 前提：worker 的 `sample_actions` **不做反归一化**，driver 真值经 `Normalize`，
> 两者同处**归一化空间**，可直接比 MSE。评估用的是训练样本（自评），要证明
> **泛化**需用未参与训练的留出样本。

### 5.3 推理机制要点

- flow-matching：从高斯噪声出发，沿 `timesteps` 做 `num_steps=10` 步去噪 → action chunk。
- **`temperature` 必须为 0**（worker 会拒绝非 0），随机性只来自初始噪声 rng。
- `return_rollout_trace=True` 会额外返回去噪链 + logprobs（给 RL 用），纯推理不需要。

---

## 6. 真机接入（你最关心的）——接口已留好

**入口脚本：`scripts/wip/openpi_vla_infer_obs.py`**
把「原始 obs → action chunk」全过程封装好，你只提供原始输入，它内部自动跑
`LiberoInputs → Normalize → PaliGemma 分词 → Pad → 推理`，你不用碰预处理。

### 6.1 一次性准备：导出 norm_stats（脱离数据集）

训练用的归一化统计（你数据的 mean/std）没随权重落盘。先导出一次，之后推理只读它：
```bash
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
export PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
"${GRB}/host-venv/bin/python" scripts/wip/openpi_export_norm_stats.py \
  --output /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json
# 已实测:导出 state/actions 各 32 维,含 q01/q99
```

### 6.2 调用（当前从文件读，真机时替换来源）

```bash
python scripts/wip/openpi_vla_infer_obs.py \
  --model-path mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9 \
  --norm-stats /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json \
  --image      main.jpg \
  --wrist-image wrist.jpg \
  --state      '[...]'  \
  --prompt     "lift the object" \
  --unnormalize \
  --output-json /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_obs_infer.json
```
> 加 `--unnormalize` 输出物理动作（真机用）；不加则输出归一化 action。详见第 7 节。

### 6.3 接真机：唯一要改的地方

脚本里这几个 loader 是**接口留白点**，把「从文件读」换成「从你的机器人/仿真取实时数据」：

```python
_load_image(path)   # 现在:读图片文件  → 换成:接相机帧(numpy HxWx3 uint8)
_load_state(spec)   # 现在:读 json     → 换成:接机器人 state 向量(float32)
# prompt 直接传字符串
```

**唯一的数据契约**（`_build_raw_sample` 构造的 dict，凑出这个即可）：
```python
{
    "observation/image":       <numpy HxWx3 uint8>,   # 主相机(分辨率随意,transform 会 resize)
    "observation/wrist_image": <numpy HxWx3 uint8>,   # 腕部相机
    "observation/state":       <numpy float32 向量>,   # 原始 state(未归一化,transform 会归一化)
    "prompt":                  <str>,                  # 原始文本指令(未分词,transform 会分词)
    "actions":                 <推理占位即可,不参与>,
}
```

闭环用法（真机循环）：`读 obs → 构造上面的 dict → 推理得 [10,32] action chunk
→ 执行 → 读新 obs`。action session 可复用（不必每次重建），反复调 `/act` 即可。

---

## 7. 反归一化（驱动真实电机前必看）—— 已内置 `--unnormalize`

推理输出的 action chunk 默认在**归一化空间**（数值约 [-1, 1]）。要驱动真实电机，
加 `--unnormalize` 开关即可让脚本直接输出**物理动作**：

```bash
python scripts/wip/openpi_vla_infer_obs.py \
  --model-path mint://... --norm-stats /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json \
  --image main.jpg --wrist-image wrist.jpg --state '[...]' --prompt "..." \
  --unnormalize --output-json /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_obs_infer.json
```

开启后输出 JSON 里会多一个 `action_chunk_physical` 字段（物理量），
`action_chunk` 仍保留归一化值。反归一化公式（pi0.5 默认分位数归一化
`use_quantile_norm=True`，与 openpi 官方 `Unnormalize` 严格一致）：

```
physical = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01     # 分位数(pi0.5 默认)
physical = normalized * (std + 1e-6) + mean                     # z-score(非分位数时)
```

`q01/q99/std/mean` 取 `norm_stats.actions`，按输出动作维度切齐（当前输出完整
32 维：前 26 维是真实动作，后 6 维是力控维度，都保留）。
> ✅ 已实测：归一化↔反归一化往返可逆（max diff = 0.0），公式与官方对称，可放心用于真机。
> 实现见 `openpi_vla_infer_obs.py` 的 `_unnormalize_actions()`。

> ⚠️ **动作输出维度**：action worker 曾硬编码 `[:, :7]`（libero 7-DoF 遗留），会把
> 32 维动作切成 7 维、丢掉主自由度（dim 10-13）和力控维，导致回放几乎不动。已改为
> 输出完整 `action_dim`（默认 32），可用环境变量 `MINT_OPENPI_PI05_ACTION_OUT_DIM`
> 覆盖（设 7 复现旧行为）。改 server 代码后需 `bash step1.sh` + `step2_clean_restart.sh`。

---

## 7.5 推理结果合并回数据集（供回放对比）

把用数据集推理出的 action chunk **合并回原 Lance**，写成新数据集：保留原始所有列
（图像/state/真值 actions/mujoco/...），追加与 `actions` 平行的 per-frame 预测列。

```bash
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
export PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/code/conley/.openpi_cache
export HF_HOME=/vePFS-Mindverse/share/huggingface
export MINT_BASE_URL=http://localhost:30496 MINT_API_KEY=tml-dummy
DS=/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance

# ① 先导出【本次训练数据】的 norm_stats(反归一化必须用同一份,否则输出异常)
"${GRB}/host-venv/bin/python" scripts/wip/openpi_export_norm_stats.py \
  --lance-dataset "$DS" --output /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json

# ② 逐帧推理并合并回原 lance
"${GRB}/host-venv/bin/python" scripts/wip/openpi_vla_merge_infer_lance.py \
  --model-path mint://lance-smoke-440c1d1c7107_0/sampler_weights/lance_smoke_sampler_c982d5e9 \
  --norm-stats /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_norm_stats.json \
  --lance-dataset "$DS" \
  --output-lance /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged.lance
```

**推理机制（重要）**：脚本对 episode 里的**每一帧 f**，拿**第 f 帧的当前 obs**
（`image[f]`+`wrist_image[f]`+`state[f]`+`prompt`，单帧、不含未来），让模型
**一次性预测未来 `action_horizon`(=10) 步**的动作块 `[10, 32]`，写进 `pred_actions[f]`。
所以预测列逐帧与源 `actions` 对齐——第 f 个元素就是「站在第 f 帧往后看的 10 步」。
真值窗口取源 `actions[f : f+10]`（不足则重复末帧补齐），MSE = 两个 10 步窗口在
归一化空间的均方误差。

追加的列：

| 列 | 结构 | 含义 |
|---|---|---|
| `pred_actions` | [frame][H, dim] | 每帧预测的未来动作块（归一化空间） |
| `pred_actions_physical` | [frame][H, dim] | 同上反归一化物理量（与源 `actions` 同空间） |
| `pred_action_mse` | [frame] | 每帧 pred vs 真值窗口的 MSE |
| `pred_meta` | struct | `{model_path, action_horizon, action_dim}` |

**读取回放**（最简脚本）：

```bash
python scripts/wip/read_replay_lance.py /vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_replay_merged.lance \
  --save-frames /vePFS-Mindverse/user/intern/wenxi/results/datas/replay_frames        # 可选:导出每帧主相机图
```

对第 f 帧同时拿到：原图 `image[f]`、真值 `actions[f]`、预测 `pred_actions_physical[f]`。

---

## 8. 文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `PI05lance.sh` | 训练一条龙（lance→LoRA→save→act） | ✅ 实测 |
| `scripts/wip/openpi_vla_smoke_lance.py` | 上面的 driver（含 Lance→transform 流水线） | ✅ |
| `PI05lance_infer.sh` + `openpi_vla_infer_lance.py` | 用 lance 样本纯推理（不重训） | ✅ 实测 |
| `scripts/wip/openpi_vla_eval_lance.py` | 验证集 MSE 量化评估 | ✅ 实测(当前权重稳定帧 MSE ~40% 零基线) |
| `scripts/wip/openpi_export_norm_stats.py` | 导出归一化统计到 json | ✅ 实测 |
| **`scripts/wip/openpi_vla_infer_obs.py`** | **真机接入接口**（接原始 obs，含 `--unnormalize` 输出物理动作） | ⏸️ 接口就绪+反归一化已验证,真机数据来源待接 |
| `scripts/wip/openpi_vla_merge_infer_lance.py` | 逐帧推理并**合并回原数据集**（原列+预测列），供回放 | ✅ 实测 |
| `scripts/wip/read_replay_lance.py` | 最简读取脚本：逐帧打印原始+预测，演示回放读法 | ✅ 实测 |
| `step0/1/3/4_*.sh`, `step2_clean_restart.sh` | Ray/server 启动与重启 | ✅ |

---

## 9. 常见问题排查

- **act 返回 500 / TaskStateConflictError**：direct-runtime future 被 scheduler
  reaper 误领养的竞态，已修复（`model_work_scheduler._is_scheduler_owned_record`）。
  若复现：确认 server 跑的是最新代码（`bash step1.sh` 同步后 `step2_clean_restart.sh`）。
- **train 报 actor died / ray.kill**：叠跑竞争（前一次 delete_model 杀了共享 actor）。
  每次跑前 `step2_clean_restart.sh` + 等 20s，单跑不叠跑。
- **推理输出全 0 或异常**：多半是 norm_stats 不匹配。确认用的是**本次训练数据**导出的
  norm_stats，不是 sampler 自带的 libero 统计。
- **真异常不在结构化日志**：act/train 的真实堆栈在 `/tmp/mint_dev_launch_wenxi.log`
  （launcher 捕获的 stderr），不在 server 结构化日志、不在 Ray worker 日志。
- **改了代码不生效**：Python server 不热加载，必须 `step1.sh`(同步) + `step2_clean_restart.sh`(重启)。
