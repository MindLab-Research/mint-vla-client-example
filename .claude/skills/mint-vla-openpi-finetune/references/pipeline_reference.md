# OpenPI pi0.5 微调 Pipeline 全景参考

本文档梳理 mint 仓库中所有与 OpenPI pi0.5 VLA 微调相关的脚本、服务器端约束和已知坑点。
**在改动本 skill 的 driver 脚本或调试训练失败之前，先读完这份文档**，避免重复踩坑。

所有事实均经过本次开发过程中的**实际验证**（真实起服务器 + 真实跑训练），不是纯代码阅读推测。

---

## 1. 当前仓库唯一支持的 pi0.5 模型

`training_backend="openpi_pi05"` / `policy_family="flow_action"` 在
`mint_server/backend/core/model_registry.py` 中**只有一个模型条目**：

```python
"openpi/pi05-libero-low-mem-finetune": ModelConfig(
    num_parameters=3.0, is_moe=False,
    inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
    max_model_len=200,
    policy_family="flow_action", inference_modality="actions",
    training_backend="openpi_pi05",
    camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    action_dim=32, action_horizon=10,
)
```

- `action_dim=32`：与 `ActionHeadSummary.md` 的结论直接对应——32维是当前唯一验证过可行的配置。
- `camera_layout` 是 3 路相机，但 `openpi_vla_smoke_lance.py` 的 `LanceViewpi05Dataset` 只读 2 路
  （`image`/`wrist_image`），第 3 路相机是从 `MODEL_CONFIGS[base_model].camera_layout`
  静态定义里取的，不是从数据集读的——如果数据集不含第3路图像，这里可能有隐藏的不匹配（未在本次验证中触发，因为
  `pi_video_streams_full_lance.lance` 恰好是2路相机的历史遗留结构，需留意）。

同区域的姊妹条目 `openpi/pi0-fast-libero-low-mem-finetune`（`training_backend="openpi_fast"`,
`action_dim=7`, `action_token_budget=64`）是完全不同的 AR-token 解码路线，与本 skill 无关，不要混用。

---

## 2. 服务器端强制约束（真实验证过，不是猜测）

### 2.1 LoRA 配置被硬编码校验，不可自由配置 ⚠️

`mint_server/backend/openpi/openpi_pi05_training.py:40-62`（`validate_openpi_pi05_create_request`）：

```python
def validate_openpi_pi05_create_request(request: Any) -> None:
    base_model = str(getattr(request, "base_model", "") or "")
    if not _is_openpi_pi05_model(base_model):
        return

    lora_config = getattr(request, "lora_config", None)
    if lora_config is None:
        raise ValueError("OpenPI pi0.5 training requires lora_config")
    if int(lora_config.rank) != OPENPI_PI05_LORA_RANK:
        raise ValueError(
            "OpenPI pi0.5 training only supports the upstream LoRA rank "
            f"{OPENPI_PI05_LORA_RANK}"
        )

    for field in ("train_attn", "train_mlp", "train_unembed"):
        if getattr(lora_config, field, True) is not True:
            raise ValueError(
                "OpenPI pi0.5 training does not support partial LoRA toggle mapping; "
                f"expected {field}=True"
            )

    if getattr(request, "rollout_correction_config", None) is not None:
        raise ValueError("OpenPI pi0.5 does not support rollout_correction_config")
```

**三条硬性规则**（`create_model` 违反即返回 HTTP 400）：

1. `lora_config.rank` 必须**精确等于** `OPENPI_PI05_LORA_RANK = 16`
   （定义于 `openpi_pi05_training.py:20`，**没有配置化开关**，改只能改代码）。
2. `train_attn` / `train_mlp` / `train_unembed` 必须**全部是 `True`**，不支持部分开关。
3. `rollout_correction_config` 必须是 `None`。

**这不是 smoke 脚本"偷懒硬编码"，是服务器强制要求。** 之前设计本 skill 的
driver 脚本时，曾误判"LoRA rank/train-flags 可配置"是修复目标，实际写了
`--lora-rank`/`--lora-train-*` 允许自由传参，结果在真实验证时 `create_model`
返回 400（`--lora-rank 8` 触发规则1）。**修复方式**：保留这些 CLI 参数（用户可能仍需要知道/显式声明这些值），
但在发请求前先用 `validate_lora_config()` 本地校验，不合规直接 `SystemExit`，
不要指望服务器的 400 来提示用户——那条报错链路是纯 HTTP 状态码，没有像 `create_model` 缺
`lora_config` 那种详细 detail 文本会经由 future 轮询丢失细节。

另有一条**优先级更高**的通用校验会先拦截 pi0.5 请求（`mint_server/routes/training.py:1761-1780`，
`_validate_rollout_correction_config_or_400`）：如果 `rollout_correction_config` 非 `None`，
会先检查 `base_model` 是不是 MoE（`is_moe` 字段），pi0.5 (`is_moe=False`) 会先在这里被拦截
（报错文案是"only supported on Megatron backend (MoE models)"），根本走不到上面那条 pi0.5 专属检查。
两条规则效果重叠但报错文案不同，调试 400 时留意实际命中的是哪一条。

### 2.2 create_model 返回的 model_id 与你发送的 session_id 不是同一个字符串 ⚠️

服务器会给 `session_id` 追加后缀形成真正的 `model_id`（例如发送
`session_id="vla-lora-abc123"`，服务器返回 `model_id="vla-lora-abc123_0"`）。

**必须从 `create_model` 的响应体里读取 `model_id`**，不能假设它等于你发送的 `session_id`。
这是本次开发中真实踩到的 bug：driver 脚本最初把本地生成的 `session_id` 直接当 `model_id`
用于后续 `train_step` 请求，导致服务器 `has_local_training_session(model_id)` 查找失败，
`train_step` 返回 **HTTP 503**（不是 400，容易误判为服务不可用/GPU问题，实际是 model_id 对不上）。

正确写法（参考 `scripts/wip/openpi_vla_smoke_lance.py::_create_model`）：
```python
create_result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model", headers, payload))
model_id = create_result.get("model_id")
if not isinstance(model_id, str) or not model_id:
    raise RuntimeError(f"create_model missing model_id: {create_result!r}")
```

### 2.3 序列长度超限会在 train_step 返回 400

`mint_server/routes/mint.py` 附近逻辑（引自 `wenxi_dev_md/Openpi05_dataflow.md:164-165`）：
若 tokenized prompt 的 `max_seq_len > max_model_len`（pi0.5 是 200，见 model_registry），
`vla_train_step` 会直接 400。这与 LoRA rank 400 是不同触发点，报错文案不同。

---

## 3. no-Ray 模式的真实行为（纠正一处文档矛盾）⚠️

`OpenPI_Separate.md:190-199` 描述"no-Ray 模式完全绕开 Ray"，但**本次两次独立的真实起服务器验证**
（`skill_verify_server_20260715_090537.log` 和 `PI05lance_local_norray.sh` 跑通训练时的
`pi05_norray_server_20260715_091818.log`）都观察到服务器启动日志里出现：

```
2026-07-15 09:05:43  ray_init_skipped_no_ray_mode   error='explicit Ray address is required...'
2026-07-15 09:05:48  Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
2026-07-15 09:05:50  control_plane_startup_check_degraded___s
```

时间线证实：`init_ray()` 先按预期失败降级（5秒前的第一条日志），**但紧接着独立地起了一个本地单机 Ray 实例**。

**根因**（已定位到具体代码）：`mint_server/app.py` 的 lifespan 里，即使
`MINT_SKIP_SUPERVISOR=1` 跳过了 `model_actor_supervisor` 检查，健康检查列表里还有一项
`("config_actor", async_ping_config_actor(timeout_s=5.0))`（`app.py:117-120`）不受这个开关影响。
它调用 `mint_server/backend/core/config_actor.py` 里的 `ray.get_actor(actor_name, namespace=...)`
（第112/130行）。Ray SDK 的已知行为：当 `ray.is_initialized()` 为 `False` 时调用
`ray.get_actor()` 等 API，会自动 fallback 到不带地址的 `ray.init()`，从而起一个本地单机集群。

**结论/如何看待这件事**：
- 这个本地 Ray 实例**不影响 openpi Ray-free 训练本身**——两次验证里，无论这个本地 Ray 是否存在，
  pi0.5 的 `train_step`/`create_model`/`save_weights_for_sampler` 全部走的是
  `openpi_local_execution.py` 的进程内直连路径（`has_local_training_session()` 分支），
  不经过这个本地 Ray 集群。**今天成功跑通训练的两次日志里，这行"Started a local Ray instance"都存在**，
  证明它是无害的已知副作用，不是需要修的 bug，也不需要手动 kill 它。
- healthz 返回 `unhealthy`/503 是**预期**的降级状态标记（来自 `set_startup_degraded_state`），
  与"是否起了本地 Ray"无关，503 不代表训练会失败。
- 不要把清理这个本地 Ray 进程当作"修复 no-Ray 模式"的必要步骤；也不要因为看到这行日志就怀疑
  no-Ray 配置出了问题。如果需要杜绝这个副作用（比如担心多个本地 Ray 实例互相冲突），
  那是服务器端 `config_actor.py` 的改动范围，不属于本 skill 的职责。

---

## 4. 脚本地图：谁依赖谁

### 4.1 核心可复用模块（几乎所有 lance 系脚本的基础）

**`scripts/wip/openpi_vla_smoke_lance.py`**——定义：
- `LanceViewpi05Dataset`：Lance episode 表 → 按帧窗口切片的样本视图。
- `_build_model_config` / `_make_data_config` / `_compute_norm_stats` / `_transform_sample`：
  OpenPI 官方 transform 链（`LiberoInputs → Normalize → PaliGemma分词 → Pad`）。
- `_pi05_datum_from_transformed`：转成 mint-server wire format
  （`observation.model_input.chunks` + `observation.state` + `supervision.actions`）。
- `_headers` / `_post_json` / `_get_json` / `_await_result` / `_poll_future`：HTTP + future 轮询封装。
- `PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"` 常量。

被以下脚本 `import openpi_vla_smoke_lance as L` 直接复用：
`openpi_export_norm_stats.py`、`openpi_vla_eval_lance.py`、`openpi_vla_infer_lance.py`、
`openpi_vla_infer_obs.py`、`openpi_vla_infer_to_lance.py`、`openpi_vla_merge_infer_lance.py`、
`openpi_pi05_local_route_check.py`、`openpi_pi05_local_train_check.py`，
以及本 skill 的 `scripts/tools/openpi_vla_lora_finetune.py`。

### 4.2 服务器启动器

**`scripts/wip/_run_local_openpi_server.py`**：Ray-free uvicorn 启动器，直接对
`mint_server.app` 起单 worker server。会在 import app 之前把 `jax`/`jax._src`/`absl`
的 logger 级别设为 WARNING（否则默认 DEBUG 级别刷屏近10万行日志，拖慢首次编译）。

必须用 `gpu_rl` host-venv 的 python（API 进程内联跑 JAX/GPU 训练，系统 python 不行）。

### 4.3 已验证的 Ray-free 全链路脚本

**`scripts/vla/PI05lance_local_norray.sh`**：起 Ray-free server + 调用
`openpi_vla_smoke_lance.py` 跑 `create_model → train_step×N → save_weights_for_sampler →
action_session → act → cleanup` 全链路。**本次开发中用它验证过真实训练可以走通**
（4步训练，loss正常输出，save/act均成功）。用法：

```bash
MINT_PI05_STEPS=4 MINT_PI05_BATCH=2 MINT_PORT=<port> \
  bash scripts/vla/PI05lance_local_norray.sh
```

环境变量全集（均有默认值）：`MINT_PORT`(30510) `MINT_PI05_STEPS`(400) `MINT_PI05_BATCH`(2)
`MINT_PI05_MODEL` `MINT_CODE_ROOT` `MINT_GPU_RL_ROOT` `MINT_EXTRA_PYDEPS` `MINT_LANCE_DATASET`
`MINT_PI05_SKIP_SERVER`(=1 复用已在跑的server)。

配套脚本：
- **`PI05lance_local_eval.sh`**：对已训练的 sampler 做量化评估（MSE/L1 vs 零基线）。
  **关键约束**（脚本注释原文）：eval 必须用**与训练相同**的 lance 数据集，否则归一化统计不同、
  MSE 无意义。
- **`PI05lance_local_merge_infer.sh`**：把推理预测合并回原 lance 结构（保留原列+追加4个预测列）。
  **关键约束**：`--norm-stats` 必须是从**同一个** lance 用 `openpi_export_norm_stats.py`
  导出的（训练时统计现算不落盘）。

### 4.4 与 Ray-free 无关的路线（不要混用）

- `PI05lance.sh` / `PI05lance_infer.sh`：**依赖 Ray 控制面**（走 Ray actor supervisor），
  与 no-Ray 路线是两条并行方案，不要在同一次验证里混用两套环境变量。
- `openpi_libero_sft.py` 及其家族（`openpi_libero_resume_sft.py`、
  `openpi_libero_fast_group_rl.py`、`openpi_libero_fast_real_eval*.py`）：走**原始 LIBERO
  HuggingFace 数据集**（`DATASET_ROOT=/vePFS-Mindverse/share/code/conley/.hf-lerobot/...`），
  不是 lance 格式，是完全独立的数据管线。`openpi_libero_sft.py` 还会在 import 时 monkeypatch
  全局 `requests.post`/`requests.delete` 注入认证 header——import 这个模块会有跨模块副作用，
  本 skill 的 driver 脚本不应 import 它。

### 4.5 分层验证脚本（调试用，理解 Ray-free 架构分层时有用）

- `openpi_pi05_local_train_check.py`（Step 1）：绕过 HTTP + Route 层，直接驱动
  `OpenPIPi05TrainingEngine`，验证 Engine 层本身。
- `openpi_pi05_local_route_check.py`（Step 2）：绕过 HTTP，直接调用
  `openpi_local_execution` 的 handler 函数，验证 Route 逻辑本身。
- `PI05lance_local_norray.sh`（Step 3）：全 HTTP 链路，即本 skill 依赖的验证层级。

若未来调试遇到"HTTP 链路失败但不确定是 Route 层还是 Engine 层的问题"，可以借用 Step 1/2
脚本二分定位，不必每次都用全 HTTP 链路调试。

---

## 5. OOM 历史与真正的根因（不要只套用脚本注释里的缓解措施）

`scripts/vla/PI05lance_local_norray.sh` 头部注释只提到缓解措施：

> "OOM fix: pi05 train_step recompiles an XLA graph per step (variable padded shapes);
> CUDA command buffers accumulate and exhaust VRAM around step ~17.
> XLA_FLAGS=--xla_gpu_enable_command_buffer= disables command buffers so graphs don't pile up."

**这只是缓解措施，不是根因修复**——加了这个 flag 只是把崩溃点从第17步推迟到第33步。

真正的根因记录在 `OpenPI_Separate.md:207-227`：变长 prompt 导致每步都是不同的 traced shape，
每次都要重新编译 XLA 图、编译产物不释放，显存持续爬升直到 `RESOURCE_EXHAUSTED`。
**真正的修复**是在 worker 代码（`mint_server/backend/openpi/openpi_pi05_worker.py` 的
`_observation_from_payload`）里加了 `_padded_prompt()`，把 token/mask pad 到固定的
`max_token_len`，让 shape 恒定、只编译一次。修复后显存稳定在约 61GB，从第11步到第37步几乎不变。

**如果未来又遇到训练中途 OOM**：先确认这个 worker 端 padding 修复是否还在（没被回退），
而不是第一反应去调整 `XLA_FLAGS` 或减小 batch size——那只是治标。

同一批修复还包含一个独立的 driver bug：`_await_result` 曾经静默吞掉失败的 future
（`routes/futures.py` 的 `_failed_payload` 返回 `{"error":...}` 但 HTTP 200），导致
"表面跑完但loss全是null"的假成功。现在的 `_await_result` 实现（见
`scripts/wip/openpi_vla_smoke_lance.py:111-122`）已经会对这种情况 `raise RuntimeError`，
本 skill 的 driver 脚本复用的正是这个已修复版本。

---

## 6. 已知的数据集格式陷阱

### 6.1 `LanceViewpi05Dataset` 期望的 schema

每一行是一个 episode，必须包含：
`episode_metadata.total_frames`(int) `image`(每帧一个jpeg bytes的list)
`wrist_image`(同) `state`(每帧一个float向量的list) `actions`(同)
`prompt`(标量字符串)。

**不是所有 lance 数据集都是这个 schema。** 本次开发中发现
`/vePFS-Mindverse/share/ylang/datasets/Dataset_B/new_all_generated_mano.lance`
完全是另一种 schema（`hands`/`objects`/`contact`/`trajectory_metadata`，纯运动学捕捉数据，
**没有 image/wrist_image 字段**），需要先经过 MuJoCo 渲染管道转换才能喂给这套 pipeline——
这个渲染转换步骤在本仓库内没有找到，属于未来待办，本 skill 当前**只处理已经是
`image`/`wrist_image`/`state`/`actions`/`prompt` schema 的 lance 数据集**。

### 6.2 action_dim 必须与模型完全匹配，不能填充/mask 补救

详见仓库根目录 `ActionHeadSummary.md`（10个实验的完整结论）。核心结论：
- 零填充必然失败（Diffusion 目标在填充维度上收敛为纯噪声，且 forward pass 通过 attention
  污染其他维度的隐藏状态，mask 只能屏蔽 loss/梯度，无法阻止这两种污染）。
- 维度可以向下收缩（16维模型可用但性能略弱），但向上扩展必须有真实数据支撑。
- 当前 `pi05-libero-low-mem-finetune` 的 `action_dim=32` 正是这份研究验证过的最佳配置。

### 6.3 Lance 多版本数据集可能只有部分版本可读

本次开发中发现 `new_all_generated_mano.lance` 有236个历史版本，但只有 version 6 能实际读出数据
（其余版本因分片文件缺失，`to_table()` 抛 `ArrowInvalid: Not found`）。这类问题应该用一次性的
`lance.dataset(path).to_table(limit=1)` 轻量探测在早期发现，而不是等整表加载到一半才失败。

---

## 7. 环境依赖速查

训练脚本需要能同时 import `openpi`（+ jax + sentencepiece）和 `lance`，系统 python 做不到，
必须用专门配置的 venv：

```bash
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
EXTRA_PYDEPS=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
PY="${GRB}/host-venv/bin/python"
export PYTHONPATH="<repo_root>:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi
export HF_HOME=/vePFS-Mindverse/share/huggingface
```

`lance` 模块**只存在于** `EXTRA_PYDEPS` 目录，不在 `gpu_rl/site-packages` 下——如果
`PYTHONPATH` 漏了 `EXTRA_PYDEPS`，会报 `ModuleNotFoundError: No module named 'lance'`
（历史上多次因为漏配这一项而排查半天）。
