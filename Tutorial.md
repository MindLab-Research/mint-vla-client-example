# Tutorial: mint no-ray 模式下用 Tinker 兼容 API 微调 OpenPI pi0.5

> 本文保留为 MINT 服务端和 HTTP 协议的历史参考，包含上游开发仓库中的路径与脚本名。
> 同事使用本项目做微调时，请从 `README.md` 和
> `docs/CLIENT_FINETUNE.md` 开始；客户端实验不需要修改共享 MINT checkout。
> `client_train_test.sh` 是开发者提供的外部参考脚本，不由本项目编写或维护。

本文档按**两种角色**分别说明：

- **服务端角色**（你）：怎么启动一个 no-ray（Ray-free）的 mint-server，供别人训练用。
- **用户角色**（其他人）：拿到你的 server 地址后，怎么写一个简单脚本训练自己的数据集。

全文内容均来自本仓库已经**真实验证过**的记录（真实起服务器、真实跑训练），出处在
文末列出，不是理论推测。

**⚠️ 范围限定：no-ray 只针对 OpenPI pi0.5 (VLA) 这一条路径，不是整个 mint-server
都不依赖 Ray 了。** 见第0.3节。

---

## 0. 先确认：这是不是 no-ray 模式？

**是的，但只针对 OpenPI pi0.5 (VLA) 这一条路径。** mint-server 支持一种完全不依赖
Ray 控制面的运行模式（下文称 no-ray /
Ray-free 模式），OpenPI pi0.5 的训练和推理在这种模式下走的是进程内直连路径
（`openpi_local_execution.py`），不经过 Ray actor supervisor。这条路径已经被
真实起服务器、真实跑训练验证过（4步/50步训练均成功，loss 正常下降，checkpoint
正常落盘）。

### 0.1 如何识别一个 server 是不是 no-ray 模式启动的

看启动这个 server 时是否设置了这几个环境变量（缺一不可）：

```bash
MINT_ALLOW_NO_RAY=1       # init_ray() 失败时降级而不是直接崩溃退出
MINT_SKIP_SUPERVISOR=1    # 跳过 Ray 模型actor supervisor 的健康检查
MINT_UVICORN_WORKERS=1    # 单进程 worker（引擎/会话/future 状态全在进程内，不能多进程）
MINT_USAGE_BACKEND=disabled  # 跳过 postgres 用量存储
```

### 0.2 如何从日志确认（避免误判"环境有问题"）

no-ray 模式下服务器启动日志会依次出现下面几条，**这是预期行为，不是故障**：

```
ray_init_skipped_no_ray_mode   error='explicit Ray address is required...'
Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
control_plane_startup_check_degraded___...
```

有两个反直觉的地方需要提前知道：

- **即使是 no-ray 模式，日志里仍会出现"Started a local Ray instance"。**
  这不是矛盾，也不是 bug：`app.py` 的健康检查列表里有一项
  `config_actor` 健康检查不受 `MINT_SKIP_SUPERVISOR` 影响，它调用了
  `ray.get_actor()`，触发 Ray SDK 的默认行为——`ray.is_initialized()` 为
  `False` 时自动 fallback 起一个本地单机 Ray 集群。这个本地 Ray 实例**不参与**
  openpi pi0.5 的训练/推理，`train_step`/`create_model`/`save_weights_for_sampler`
  走的是完全独立的进程内直连路径。不需要手动 kill 它。
- **`GET /api/v1/healthz` 返回 `503`（"unhealthy"）是 no-ray 降级模式下的预期状态标记**，
  不代表服务不可用。判断服务器是否就绪应该看返回码是 `200` 或 `503`，两者都算就绪。

### 0.3 范围限定：只有 OpenPI pi0.5 拆掉了 Ray，其他 backend 没有

**这不是给整个 mint-server 拆 Ray，只是给 OpenPI pi0.5 (VLA) 这一条路径拆的。**

依据：提交 `1e310445`（"feat(openpi): remove Ray dependency, enable in-process
execution"）明确写的是删除 `openpi_ray_runtime.py` / `openpi_shared_ray_runtime.py` /
`openpi_action_ray_runtime.py` 这3个 **openpi 专属**的 Ray runtime 文件，换成
`openpi_local_execution.py` 的进程内直连路径；改动范围限定在 `training.py`/`mint.py`/
`futures.py`/`app.py` 里"路由到 openpi backend 时走 local execution"的分支，以及
openpi 相关模块本身。

其他训练/推理 backend 依然完全依赖 Ray 控制面，没有被这次改动触碰，包括（截至本文档
写作时，`mint_server/` 下仍在 `import ray` 的模块）：

- 训练：`training/megatron/megatron_distributed.py`、`training/verl/verl_training.py`、
  `training/qwen36/*`、`training/dense/dense_trainer.py`、
  `training/bumblebee/bumblebee_distributed.py`
- 推理：`inference/multi_lora_engine.py`、`inference/multinode_inference.py`、
  `sglang_engine.py`（vLLM / SGLang）
- 控制面本身：`core/config_actor.py`、`actors/ray_keepalive.py`、
  `actors/model_actor_inventory.py`、`actors/node_placement.py`、`ray_cluster/*`

所以：**Qwen 系列、K2 等其他模型的训练/推理仍然需要一个完整的 Ray 集群
（Volcano/Aliyun 的 GPU worker + Ray GCS），本文档第1节的 no-ray 启动方式只适用于
OpenPI pi0.5 这一个 backend，不能套用到其他模型上。** 如果要给其他模型起服务，
应该用 `mint-dev`/`mint-prod` skill 文档里的标准 Ray 集群启动流程，不是本文档的
`MINT_ALLOW_NO_RAY=1` 那套。

---

## 1. 服务端角色：我怎么启动这个 server？

下面是启动脚本，改编自本仓库已验证过的 `scripts/vla/PI05lance_local_norray.sh`。
可以直接复制运行（按需替换路径/端口/GPU）。

```bash
# --- 基础路径 ---
CODE_ROOT=/path/to/your/mint/checkout      # 你自己的 worker-visible 代码checkout
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
EXTRA_PYDEPS=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
PY="${GRB}/host-venv/bin/python"   # 必须用这个venv，系统python没装openpi/jax/lance

PORT=30510
MODEL="openpi/pi05-libero-low-mem-finetune"   # 当前唯一支持的 openpi_pi05 backend 模型

# --- no-ray 相关（缺一不可）---
export MINT_CODE_ROOT="${CODE_ROOT}"
export MINT_PORT="${PORT}" MINT_HOST=0.0.0.0
export MINT_UVICORN_WORKERS=1 MINT_SKIP_SUPERVISOR=1 MINT_ALLOW_NO_RAY=1 MINT_USAGE_BACKEND=disabled
export MINT_RAY_NAMESPACE=mint_<your_name>_local
export MINT_SUPPORTED_MODELS="${MODEL}"

# --- openpi pi0.5 资产路径 ---
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi
export HF_HOME=/vePFS-Mindverse/share/huggingface
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR=/vePFS-Mindverse/share/mint/dev/data/<you>/openpi-pi05-checkpoints
export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
export MINT_RUNTIME_CHECKPOINT_DIR=/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints

# --- GPU / CUDA ---
# XLA CUDA command buffer 会在训练中途导致显存爆炸（历史真实踩过的 OOM 根因之一的缓解措施）
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
# 绑定到确认空闲的卡（共享GPU box上，其他用户可能占用某些卡）
export CUDA_VISIBLE_DEVICES=3,4,5,6
# 关键：这台机器的显卡驱动(535.129.03)只原生支持到CUDA 12.2，但jaxlib是CUDA13构建的。
# 没有这行，所有JAX GPU调用会报 cudaErrorInsufficientDriver，看起来像"GPU被占用/坏了"，
# 实际是驱动兼容问题，不是硬件问题。
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"

export PYTHONPATH="${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

# --- 启动 ---
nohup "${PY}" -u "${CODE_ROOT}/scripts/wip/_run_local_openpi_server.py" > server.log 2>&1 &
echo "server pid=$!"

# --- 等待就绪：200 或 503 都算就绪 ---
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://localhost:${PORT}/api/v1/healthz")
  if [ "${code}" = "200" ] || [ "${code}" = "503" ]; then echo "ready (http=${code})"; break; fi
  sleep 2
done
```

`PYTHONPATH` 顺序有讲究：`lance` 模块只存在于 `EXTRA_PYDEPS` 目录，不在
`gpu_rl/site-packages` 下——漏配会报 `ModuleNotFoundError: No module named 'lance'`。

### 1.1 要让别人连进来，需要注意什么

- **`MINT_HOST=0.0.0.0`** 是必须的——如果起成 `127.0.0.1`（默认值），只有本机能连，
  其他机器的用户完全连不上。上面的脚本已经设成了 `0.0.0.0`。
- 把 `http://<这台机器的IP>:${PORT}` 告诉要用你 server 的人。如果对方在同一局域网/VPC，
  直连即可；如果隔了网络边界，需要你帮对方配一条 SSH tunnel 或者确认端口在防火墙里放行。
- **鉴权**：no-ray 单机模式默认不做真实鉴权校验，`X-API-Key` 填任意非空字符串
  （比如 `tml-dummy`）就能通过。这意味着**任何能连上这个端口的人都能训练/删除模型**——
  如果 server 开在有其他人共享访问的网络上，这是需要你知道的安全边界，不是自动加固的。
- **模型必须提前在 `MINT_SUPPORTED_MODELS` 里声明**：如果你只想让别人用
  `openpi/pi05-libero-low-mem-finetune`，启动时就只放这一个模型名；对方请求
  一个不在这个列表里的 `base_model` 会被拒绝。
- **同一个 server 可以被多个用户同时使用**：每次 `create_model` 都会生成独立的
  `model_id`，互不冲突，可以有多个人同时训练各自的 checkpoint。但如果都在抢同一批
  GPU，显存/吞吐会互相影响——这是物理资源限制，不是 API 层的限制。
- 停止 server：`kill <server_pid>`（脚本打印过 pid），或者在共享机器上先确认没有
  其他人在用这个 server 再停止，避免误杀别人的训练会话。

---

## 2. 用户角色：拿到 server 地址后，我怎么写脚本训练自己的数据集？

### 2.1 前提条件

server 起好之后，训练请求需要把原始图像/state/actions 转换成服务器要的 wire format
（base64 PNG + 归一化后的 TensorData，见第4节细节）。**这个转换目前是在调用方
（客户端）这边跑的**，不是服务器帮你做的，所以用户这边需要：

1. 能访问 server 的 `host:port`（服务端角色的人需要告诉你这个地址）。
2. 能 import `openpi`/`jax`/`lance` 这几个 Python 依赖——好消息是这套依赖已经装在
   共享存储的一个 venv 里（`/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl`），
   只要你能访问这条 PFS 路径，不需要自己重新装环境。
3. 有一份 Lance 格式的数据集，schema 必须是
   `image`/`wrist_image`/`state`/`actions`/`prompt`（每个 episode 一行，每帧一个元素），
   且 `state`/`actions` 的向量维度必须精确等于目标模型的 `action_dim`
   （`openpi/pi05-libero-low-mem-finetune` 是32维，维度不匹配无法通过零填充/mask
   绕过，详见 `ActionHeadSummary.md`）。

### 2.2 最简单的脚本：直接用仓库自带的一键 driver

不需要自己写 HTTP 请求，仓库里已经有一个封装好的脚本
`scripts/tools/openpi_vla_lora_finetune.py`，把下面几个变量换成你自己的即可：

```bash
#!/usr/bin/env bash
# 只需要改这4个变量：
SERVER="http://<服务端告诉你的host>:<port>"     # 例如 http://192.168.1.50:30510
MY_DATASET="/path/to/my_own_dataset.lance"       # 你自己的 Lance 数据集
MY_CHECKPOINT_NAME="alice_run_v1"                # 给你这次训练起个名字
MY_BASE_MODEL="openpi/pi05-libero-low-mem-finetune"  # 必须是server的MINT_SUPPORTED_MODELS里有的

# 以下这几行不用改，是共享环境的固定路径：
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
EXTRA_PYDEPS=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
REPO_ROOT=/path/to/any/mint/checkout   # 只需要有 scripts/tools/ 目录，不必是服务端那份checkout

export PYTHONPATH="${REPO_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

# JAX_PLATFORMS=cpu：客户端进程只做数据预处理+发HTTP请求，真正的GPU计算在server进程里，
# 这行防止客户端进程去抢GPU。
MINT_BASE_URL="${SERVER}" MINT_API_KEY=tml-dummy JAX_PLATFORMS=cpu \
"${GRB}/host-venv/bin/python" "${REPO_ROOT}/scripts/tools/openpi_vla_lora_finetune.py" \
  --base-url "${SERVER}" \
  --lance-dataset "${MY_DATASET}" \
  --base-model "${MY_BASE_MODEL}" \
  --steps 400 --batch-size 2 \
  --save-checkpoint-name "${MY_CHECKPOINT_NAME}" \
  --output-json "result_${MY_CHECKPOINT_NAME}.json"
```

跑这一个脚本，会自动完成：
`create_model → train_step×400 → save_weights_for_sampler → 推理验证(act) → 清理会话`。
脚本自带两个前置检查，出问题会在发出第一个网络请求之前就报错退出：

- `validate_action_dim`：数据集维度和模型 `action_dim` 不匹配 → 直接报错，不会浪费
  训练时间，也不要尝试用零填充/mask 绕过（详见 `ActionHeadSummary.md`）。
- `probe_lance_dataset`：数据集最新版本读不出来（比如分片文件缺失）→ 自动扫描历史
  版本，报出一个能读通的版本号供你用 `--lance-dataset-version <N>` 重试。

训练完成后脚本会打印 checkpoint 路径（`mint://...` URI）和最终 loss，也会写一份
JSON 结果到 `--output-json` 指定的路径。

### 2.3 想要更多功能（可选）

同一个脚本还支持：

- `--eval-mse --eval-mse-indices 0,1,2,5,10`：训练完后做一次量化 MSE/L1 评估
  （预测 vs 真实 action，同时给一个"全零预测"基线做对比）。
- `--infer-to-lance --infer-to-lance-output /path/to/output.lance`：对整个数据集
  逐帧跑推理，把预测结果写回一份新的 Lance 数据集（保留原始列 + 追加预测列）。
  这个选项会对数据集的每一帧发一次请求，数据集较大时会比较慢，谨慎在大数据集上开启。
- `--dry-run`：只做前置检查（数据集能不能读、维度是否匹配），不发任何网络请求，
  适合先确认自己的数据集能不能用。

完整参数列表和更多注意事项见
`.claude/skills/mint-vla-openpi-finetune/SKILL.md`。

---

## 3. 双方都要知道的注意事项

- **`--lora-rank` 不要改**：服务器对 `openpi/pi05-libero-low-mem-finetune` 硬性要求
  `rank=16` 且三个 `train_attn`/`train_mlp`/`train_unembed` 全部为 `True`，这是
  服务器代码里的硬编码常量，不是配置项。脚本默认值已经满足，用户不需要传这个参数。
- **`--batch-size` 建议是 server 可见 GPU 数量的倍数**：worker 端已经支持批量数据
  并行（`jax.sharding`），如果 `--batch-size` 恰好是 server 启动时
  `CUDA_VISIBLE_DEVICES` 里卡数的倍数，训练会真正把 batch 切分到多卡上跑，
  实测吞吐提升明显；不是倍数时会优雅退化（每张卡都算一份完整 batch，不报错，但没有
  额外的多卡切分收益）。服务端角色的人应该把自己起 server 用了几张卡告知用户，
  用户角色的人据此选 `--batch-size`。
- **checkpoint 文件独立于训练会话**：`save_weights_for_sampler` 写的 checkpoint
  文件在磁盘上持久化，即使后续调用 `DELETE /api/v1/models/{model_id}` 清理了训练
  会话，checkpoint 文件依然存在，可以用它继续做推理。

---

## 4. 进阶：手写 HTTP 请求（想理解每一步在做什么，或想用非Python客户端）

如果不想依赖仓库自带脚本，也可以直接手写 HTTP 请求驱动整条链路。这一节按真实调用
顺序逐步给出每个请求体/响应体的精确字段（字段名均来自源码，不是意译）。

### 4.0 认证与 future 轮询模式

- 请求头：`X-API-Key: <your-key>`（no-ray dev 场景下随便填一个非空字符串即可，
  比如 `tml-dummy`，因为 no-ray 单机模式不做真实鉴权校验）。
- **所有写操作**（`create_model`、`vla/train_step`、`save_weights_for_sampler`）都是
  future-based：POST 立刻返回 `{"request_id": "..."}`，真正的结果需要轮询：

  ```bash
  curl -s -X POST "http://localhost:${PORT}/api/v1/retrieve_future" \
    -H "X-API-Key: tml-dummy" -H "Content-Type: application/json" \
    -d '{"request_id": "<上一步返回的request_id>"}'
  ```

  `408`/`503` 表示还没算完，继续轮询；拿到非408/503的响应后，还要检查 body 里
  是否有 `{"error": ...}` 字段——服务器对失败操作可能返回 HTTP 200 但 body 带
  error，只看 HTTP 状态码会把失败误判成"结果是空的"。

### 4.1 创建模型：`POST /api/v1/create_model`

```bash
curl -s -X POST "http://localhost:${PORT}/api/v1/create_model" \
  -H "X-API-Key: tml-dummy" -H "Content-Type: application/json" \
  -d '{
    "session_id": "vla-lora-demo001",
    "model_seq_id": 0,
    "base_model": "openpi/pi05-libero-low-mem-finetune",
    "lora_config": {
      "rank": 16,
      "train_attn": true,
      "train_mlp": true,
      "train_unembed": true
    }
  }'
```

**openpi_pi05 backend 的三条硬性约束**（违反会返回 HTTP 400）：

1. `lora_config.rank` 必须**精确等于 16**（服务器代码硬编码常量，不是配置项）。
2. `train_attn` / `train_mlp` / `train_unembed` 必须**全部是 `true`**，不支持部分开关。
3. 不要传 `rollout_correction_config` 字段（省略，或显式传 `null`）——pi0.5 不是 MoE
   模型，传非 null 值会被拦截，报错文案是"only supported on Megatron backend (MoE models)"。

轮询 `retrieve_future` 拿到的响应示例：

```json
{
  "request_id": "...",
  "model_id": "vla-lora-demo001_0",
  "type": "create_model",
  "backend": "openpi_pi05"
}
```

**⚠️ 最容易踩的坑：`model_id` 不等于你发送的 `session_id`。**
服务器会给 `session_id` 追加后缀（这里是 `_0`）形成真正的 `model_id`。后续所有请求
必须用响应体里的 `model_id`，不能假设它等于发送时的 `session_id`——用错会导致
下一步 `train_step` 返回 HTTP 503（不是 400，容易误判成"服务不可用/GPU问题"，
实际是 `model_id` 对不上导致服务器找不到这个训练会话）。

### 4.2 跑一步训练：`POST /api/v1/mint/vla/train_step`

这是 mint 独有的 API 扩展（不在标准 tinker 协议里），定义在
`mint_server/models/mint_types.py`。请求体结构：

```json
{
  "model_id": "vla-lora-demo001_0",
  "loss_fn": "flow_matching",
  "data": [
    {
      "observation": {
        "model_input": {
          "chunks": [
            {"type": "image", "data": "<base64 PNG>", "format": "png", "expected_tokens": 256},
            {"type": "image", "data": "<base64 PNG>", "format": "png", "expected_tokens": 256},
            {"type": "encoded_text", "tokens": [1, 2, 3, ...]}
          ]
        },
        "state": {"data": [0.1, 0.2, ...], "shape": [32], "dtype": "float32"}
      },
      "supervision": {
        "actions": {"data": [...], "shape": [10, 32], "dtype": "float32"}
      }
    }
  ]
}
```

字段说明：

- `loss_fn` 对 pi0.5 固定传 `"flow_matching"`。
- `observation.model_input.chunks`：图像 chunk 数量按模型的 `camera_layout`
  逐个生成（`pi05-libero-low-mem-finetune` 是3路相机布局，但目前仓库里的
  Lance 数据集/transform 只实际填充 `image`/`wrist_image` 2路——第3路是从
  模型静态配置取的，注意这里如果换数据集可能出现隐藏的不匹配）。
- `observation.state` 是 rank-1 `TensorData`（`shape=[action_dim]`，pi0.5 是32维）。
- `supervision.actions` 是 rank-2 `TensorData`（`shape=[action_horizon, action_dim]`，
  即 `[10, 32]`）。

**这一步不建议手写复现。** 从原始图像/state/actions 到上面这个 wire format 需要过一遍
OpenPI 官方 transform 链（`LiberoInputs → Normalize → PaliGemma分词 → Pad`），
参考已验证过的实现：`scripts/wip/openpi_vla_smoke_lance.py` 里的
`_pi05_datum_from_transformed` 函数（结合 `_build_model_config`/`_make_data_config`/
`_compute_norm_stats`/`_build_batch`）。手写这段转换逻辑很容易在归一化统计或图像
编码上出错，且已经有验证过的现成实现，没必要重新造——这也是第2节推荐用现成脚本而
不是手写请求的原因。

发出请求后轮询 `retrieve_future`，响应示例：

```json
{"metrics": {"loss:mean": 0.1462, "...": "..."}}
```

**⚠️ loss 在 `result["metrics"]["loss:mean"]` 里，不是 `result["loss"]`。**

对每个 batch 重复这个请求 N 次（`model_id` 不变，`data` 换成新的采样批次）即为
"训练N步"。

另一个序列长度限制：如果 tokenized prompt 长度超过模型的 `max_model_len`
（pi0.5 是200），这一步会直接返回 HTTP 400。

### 4.3 保存权重给采样器用：`POST /api/v1/save_weights_for_sampler`

```bash
curl -s -X POST "http://localhost:${PORT}/api/v1/save_weights_for_sampler" \
  -H "X-API-Key: tml-dummy" -H "Content-Type: application/json" \
  -d '{"model_id": "vla-lora-demo001_0", "path": "my_run_v1"}'
```

`path` 传字符串是**命名保存**（推荐——可复现，方便以后引用）；传 `null` 是临时
（ephemeral）保存。轮询后的响应示例：

```json
{
  "path": "mint://vla-lora-demo001_0/sampler_weights/my_run_v1",
  "owner_id": "000000000000000000000001",
  "filesystem_path": "/vePFS-Mindverse/share/mint/dev/data/.../my_run_v1/sampler",
  "storage_tier": "persistent_cache"
}
```

`path` 字段是 `mint://` URI，后续做推理验证时要用这个值当 `model_path`。
`owner_id` 需要原样透传给下一步的 `action_sessions` 请求，否则会因为 owner_id
缺失返回 400。

### 4.4 （可选）推理验证：确认训练出的模型能正常出 action

```bash
# 建 action session
curl -s -X POST "http://localhost:${PORT}/api/v1/mint/action_sessions" \
  -H "X-API-Key: tml-dummy" -H "Content-Type: application/json" \
  -d '{
    "session_id": "vla-lora-infer-demo001",
    "base_model": "openpi/pi05-libero-low-mem-finetune",
    "model_path": "mint://vla-lora-demo001_0/sampler_weights/my_run_v1",
    "owner_id": "000000000000000000000001"
  }'
# -> {"action_session_id": "<uuid>"}

# 调用一次 act（observation 结构与 4.2 的 observation 字段完全一致）
curl -s -X POST "http://localhost:${PORT}/api/v1/mint/action_sessions/<uuid>/act" \
  -H "X-API-Key: tml-dummy" -H "Content-Type: application/json" \
  -d '{"observation": {"model_input": {...}, "state": {...}}}'
```

轮询后的响应示例：

```json
{
  "actions": {"data": [...], "shape": [10, 32], "dtype": "float32"},
  "policy_timing": {"infer_ms": 13517.9, "temperature": 0.0}
}
```

`actions.shape` 应该是 `[action_horizon, action_dim]`（这里是 `[10, 32]`）。

### 4.5 清理

```bash
curl -s -X DELETE "http://localhost:${PORT}/api/v1/models/vla-lora-demo001_0" \
  -H "X-API-Key: tml-dummy"
```

这只删除服务器端的会话/训练状态，**不会删除** 4.3 步已经写到磁盘的 checkpoint
文件——那些是独立持久化的，删除会话不影响之后继续用这个 checkpoint 做推理。

---

## 5. 常见问题速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `create_model` 返回400，提到"LoRA rank"或"partial LoRA toggle" | `lora_config.rank` 不是16，或某个train开关传了false | 用默认值：rank=16，三个train开关全true |
| `create_model` 返回400，提到"only supported on Megatron backend (MoE models)" | 传了非null的`rollout_correction_config` | 省略这个字段，pi0.5不是MoE模型 |
| `create_model` 成功，紧接着 `train_step` 返回503 | 几乎肯定是`model_id`提取错误——把发送的`session_id`当成了`model_id` | 从`create_model`响应体读`model_id`，不要用本地生成的`session_id` |
| `train_step` 返回400，提到序列长度 | tokenized prompt超过`max_model_len`（pi0.5是200） | 检查数据集里的prompt字段长度 |
| 训练"跑完"了但loss全是null | future轮询代码只看HTTP状态码没检查body里的`error`字段 | 轮询逻辑要主动检查`result.get("error")`并raise |
| 日志出现"Started a local Ray instance"，即使是no-ray模式 | `config_actor`健康检查触发Ray SDK自动fallback，与openpi训练路径无关 | 无害副作用，不需要处理，不要去kill它 |
| `healthz`返回503/"unhealthy" | no-ray降级模式下的预期状态标记 | 判断就绪应该看200或503都算就绪 |
| `ModuleNotFoundError: No module named 'lance'` | `PYTHONPATH`漏配了`EXTRA_PYDEPS` | 按第1节的`PYTHONPATH`顺序拼接 |
| 训练中途OOM（`RESOURCE_EXHAUSTED`） | 变长prompt导致每步都是不同的JAX traced shape，反复重新编译显存不释放 | 确认worker端的prompt padding修复还在，不要第一反应就调小batch size |
| 所有GPU的JAX调用报`cudaErrorInsufficientDriver` | CUDA驱动(535.129.03)与jaxlib(CUDA13构建)版本不匹配，不是GPU被占用 | 设置`LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH` |
| 用户连不上服务端的server | server起成了`127.0.0.1`而不是`0.0.0.0`，或者端口被防火墙挡住 | 服务端确认`MINT_HOST=0.0.0.0`，并确认端口对用户网络可达 |
| 用户请求的`base_model`被拒绝 | server启动时`MINT_SUPPORTED_MODELS`没包含这个模型名 | 服务端在启动前把要开放的模型名加进`MINT_SUPPORTED_MODELS` |

每条的完整原理分析和真实验证过程见
`.claude/skills/mint-vla-openpi-finetune/references/troubleshooting.md`，
本表只摘录 no-ray/API 路径最相关的部分。

---

## 6. 参考资料

- `.claude/skills/mint-vla-openpi-finetune/SKILL.md` —— 完整的自动化流程（含用户
  交互问答、MSE评估、推理写回lance等进阶功能）
- `.claude/skills/mint-vla-openpi-finetune/references/api_contracts.md` —— 每个
  endpoint的精确字段契约、真实响应样例
- `.claude/skills/mint-vla-openpi-finetune/references/pipeline_reference.md` ——
  no-ray真实行为的完整验证记录、LoRA/model_id/OOM根因分析
- `.claude/skills/mint-vla-openpi-finetune/references/troubleshooting.md` ——
  症状→原因→修复完整速查表
- `ActionHeadSummary.md`（仓库根目录）—— action_dim必须与模型精确匹配的10个实验结论
- `scripts/vla/PI05lance_local_norray.sh` —— 第1节服务端启动脚本的原始出处
- `scripts/tools/openpi_vla_lora_finetune.py` —— 第2节用户一键脚本的原始出处
