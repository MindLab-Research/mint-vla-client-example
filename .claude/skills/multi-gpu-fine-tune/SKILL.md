---
name: multi-gpu-fine-tune
description: 纯 HTTP 多卡 LoRA 微调 OpenPI pi0.5 — 不 import mint_server、不调 mint driver, 仅用 requests 打 HTTP. 8 卡 bs=128+8 生产者 ~52 samples/s (SM busy_mean 71%, 忙窗 ~90%); 4 卡限 CUDA_VISIBLE_DEVICES 起 server, bs=128+4 生产者 ~52 samples/s (SM busy_mean 75%). 用于在 client 侧跑通端到端训练并避免性能/启动陷阱.
---

# multi-gpu-fine-tune (纯 client 多卡微调)

通过**纯 HTTP** 在 8 卡 mint-server 上对 OpenPI pi0.5 做 LoRA 微调, **不 import `mint_server`、不 subprocess 调 mint driver**。多生产者预取内置, bs=128 + 8 生产者即可占满 8 卡 GPU (稳态 ~52 samples/s, SM busy_mean 71%/忙窗 ~90%, 与 mint driver 一致)。

## 何时用

- 想在不接触 mint 代码层级的前提下, 享受 8 卡数据并行训练的吞吐。
- 已有一个 fresh、Ray-free、`MINT_SUPPORTED_MODELS` 含 `openpi/pi05-libero-low-mem-finetune` 的 mint-server 在已知端口。
- Lance 数据集已是 `image/wrist_image/state/actions/prompt` schema (MANO/raw-capture → 渲染图像不在本 skill 范围)。

**不要用**于: pi0-fast 模型 (不同 policy family); 需要 Megatron 后端的模型; 尚未渲染成图像的原始捕获数据。

## 依赖 (重要: 为什么 client 要跑在 GPU host)

本 skill 的训练脚本是**纯 HTTP** 的, 但 client 侧仍需 `openpi` 包 (做数据 transforms: PaliGemma 分词、归一化)。openpi 包在 `MINT_GRB_ROOT/src/openpi/src`。因此 client 必须跑在能访问该 PFS 路径的机器 (通常就是 GPU host)。**这不是依赖 mint 代码** —— openpi 与 mint 是两个独立仓库, mint driver 自己也 import 同一份 openpi。

启动器 `scripts/remote/run_client.sh` 已把 PYTHONPATH 配好, 直接用它即可。

## 必备前置: fresh server

**最关键的一条**: 性能数字 (sm%、throughput) 只在 **fresh server** 上成立。

- server 老化 → sm% 从 ~47% 掉到 ~23%, 吞吐腰斩到 ~25 samples/s。
- 这曾被误诊为 "client HTTP 往返瓶颈", 实测证伪: mint driver 与 client **共用同一份 HTTP 代码** (driver 第 105-107 行 `_post_json = _smoke._post_json` import 的就是 client 的 helper), 两者发请求逐字节相同, HTTP 不可能是 client 独有瓶颈。
- **看到 client 吞吐低于 driver 时, 第一反应是重启 server, 不是改 client 代码。**

如果你没有 fresh server, 先用 mint skill `mint-vla-openpi-finetune` 的 `PI05lance_local_norray.sh` 启动模式起一个 (ununused port), 或确认用户已有的 server 端口 + `MINT_SUPPORTED_MODELS`。

## 输入 (向用户确认, 一次性问清)

| 问题 | 建议默认 | 说明 |
|---|---|---|
| Lance 数据集路径 | (必填) | `image/wrist_image/state/actions/prompt` schema |
| 训练步数 | 400 | |
| batch size | 128 (8 卡) | **必须是可见 GPU 数 (8) 的倍数**, 否则 batch 不会数据并行分片 (退化为每卡重复算同一份, 仍能跑但拿不到多卡加速) |
| 生产者数 | auto (bs=128→8) | 1=可复现串行; ≥2 每个 producer 独立 dataset/rng |
| checkpoint 名 | 省略→自动 | 省略则不存 checkpoint (仅 probe) |
| LoRA rank | 16 | server 目前硬拒其他值 |
| base model | `openpi/pi05-libero-low-mem-finetune` | 唯一支持的 openpi_pi05 后端模型 |

## 步骤

### 1. 确认 server 在线 + 模型已广告

```bash
curl -s http://localhost:<port>/api/v1/healthz   # unhealthy 是 Ray-free 模式的预期标记, 不算失败
curl -s http://localhost:<port>/openapi.json | head   # 200 即就绪
```

### 2. 跑训练 (用启动器, 它配好 PYTHONPATH/env)

```bash
MINT_BASE_URL=http://localhost:<port> \
bash scripts/remote/run_client.sh scripts/train/train_http_multiprod.py \
  --base-url http://localhost:<port> \
  --lance-dataset <lance-path> \
  --steps 400 --batch-size 128 \
  --num-producers auto \
  --save-checkpoint-name <name>   # 省略则不存 checkpoint
  --output-json results/logs/<run>.json
```

启动器预检会先打 `/openapi.json`; 端口不对会立即报 `MINT server preflight failed`。

### 3. 读结果

脚本每步打印一行 JSON: `{"step": N, "loss": x, "queue_wait": y, "step_time": z}`。最后打印 `final loss ... throughput=N samples/s`。

- **稳态水位看 `step_time`**, 不是末行 throughput。
- 末行 `throughput = batch_size * steps / elapsed`, 把首步 ~80s XLA 编译摊进了每步 —— 短跑 (steps<50) 会显著低估。稳态 `step_time ≈ 2.14-2.16s` → 真实 ~52 samples/s。
- `queue_wait` 应 ≈0.001s (生产者喂得及); 若涨到几秒, 说明生产者数不够或磁盘 IO 跟不上。

### 4. 确认 GPU 真分 8 卡 (可选, 排查"没加速")

```bash
nvidia-smi dmon -s u -d 1 -c 10
```

8 卡应**同步涨跌** (GPU phase 全 100%, HTTP phase 全 0%), 各卡显存 ~62GB。若只有卡 0 满载、其余闲置 → batch 不是 8 的倍数, 没分片 (见踩坑 §2)。

## 踩坑 (避免重复)

### 1. stale-server → 吞吐腰斩, 别误诊为 client 问题
旧 server (跑十几小时) 上 client 测得 ~25 samples/s (单步 4.9s)。真相: server 老化导致**服务端处理变慢**, 不是 client HTTP 往返。代码证据: driver 与 client 共用同一份 `_post_json`(driver 直接 import client 的 helper), 发请求方式逐字节相同。**修复: 重启 server**, 不是改 client / 不是上 async。

### 2. batch 不是 8 的倍数 → 不分片, 拿不到多卡加速
`--batch-size 8` 在 8 卡 server 上, worker 仍 jit 仍能跑 (不报错), 但**每卡重复算同一份 batch** (JAX 对 `PartitionSpec()`-replicated 输入的默认行为), 只拿到 jit 加速、没有数据并行。要占满 8 卡必须 `--batch-size` 是 8 的倍数 (128 是推荐值)。

### 3. norm_stats 启动卡死 → 已用向量化版修复
client 早期 `_compute_norm_stats` 纯 Python `for` 循环遍历 686 万 frame, 冷启动几分钟卡死。脚本内 `_compute_norm_stats_fast` 用 PyArrow 向量化 (`.combine_chunks().flatten()`), ~12s。**不要**退回慢版。该函数自包含 (不依赖 mint)。

### 4. throughput 数字偏低 → 口径陷阱
`throughput = batch_size * steps / elapsed` 把首步 ~80s 编译摊进每步。400 步实测 ~52 (摊销影响小); 但短跑 (10 步) 会显示 ~12, **别被吓到** —— 看稳态 `step_time≈2.14s` 才是真实水位。要算"稳态吞吐"用 `batch_size / step_time`。

### 5. GPU 显存触顶点 = bs 1152~1280 之间; bs≤1024 显存几乎不涨
**LoRA 冻结 base 权重, 梯度只算 r=16 参数, batch 大小对 GPU 显存几乎无影响** —— bs 64→1152 (18×) 每卡显存仅 61901→61907 MiB (差 6 MiB), 62GB 几乎全是固定开销 (base 权重 bf16 ~12GB + JAX 编译缓存)。服务化时 GPU 显存不是约束, 真正约束是触顶时的**崩溃行为**:

| bs | 每卡样本 | 每卡显存 | step_time | server 行为 | 结论 |
|---|---|---|---|---|---|
| 128 | 16 | 61.9 GB | 2.14s | 存活 | ✅ 推荐甜点 (~52 samples/s) |
| 256 | 32 | 61.9 GB | 3.96s | 存活 | ✅ |
| 512 | 64 | 61.9 GB | 7.8s | 存活 | ✅ |
| 1024 | 128 | 61.9 GB | 15s | 存活 | ✅ **安全上限** |
| 1152 | 144 | 61.9 GB | 17s | 存活 | ✅ |
| 1280 | 160 | — | — | **崩溃 (server 进程被杀)** | ❌ 服务中断 |
| 1536 | 192 | — | — | 优雅返回 error | ❌ (alloc 56.7GB) |
| 2048 | 256 | — | — | 优雅返回 error | ❌ (alloc 75.2GB) |

**触顶点: bs 1152~1280 之间 (每卡 ~144-160 样本)**。每卡每样本线性激活 ~294MB (`compute_loss` 里 PaliGemma LLM 的 `b×seq×embed_dim` 激活), 每卡余 ~18GB / 294MB ≈ 61 样本 → 8 卡 bs ≈ 488~1024 区间安全, 实测 1152 仍活、1280 崩。

**服务化关键: 触顶行为分两档, 必须远离崩溃区**:
- **bs=1280: 直接杀死 server 进程** (client 报 `ConnectionError: Remote end closed connection`, healthz=000, 需重启 server)。这是最危险的档位 —— 服务中断。
- **bs≥1536: 反而优雅返回** `RESOURCE_EXHAUSTED` error (server 存活, client 收到 error 可重试)。
- 不知为何中间档 (1280) 比更大档 (1536) 更暴力 (疑似 OOM 时 XLA 编译阶段崩溃 vs 运行阶段优雅捕获)。**服务化务必把单请求 bs 控制在 ≤1024**, 留足余量, 永远别让 batch 落进 1152-1280 崩溃区。

**真正的吞吐瓶颈不是 GPU 显存, 而是 client 侧**: bs 翻倍 `queue_wait` 暴涨 (64→17s, 1024→70s), client 构建/传输大 batch 跟不上, GPU 空等; throughput 在 bs≥256 后封顶 ~66 samples/s 不再涨 (step_time 随 bs 线性涨, 单步延迟↑但吞吐不升)。**bs=128 是吞吐甜点 (52 samples/s、2.14s/step)。**

### 5b. 大 batch 拉高 GPU 占用率 (但吞吐不升) — 8 卡 + 4 卡实测 (100 步, 2026-07-31)

加大 batch 的**收益是 GPU 占用率, 不是吞吐**: 单卡算更多样本 → GPU 计算段拉长 → HTTP 空闲间隙占比下降 → busy_mean 从 ~71% 拉到 ~95% (忙窗几乎打满)。但 step_time 随 bs 近线性涨, throughput 封顶。

**8 卡** (8 生产者):

| bs | 每卡样本 | step_time | throughput | SM% busy_mean | 全样本 | 结果 |
|---|---|---|---|---|---|---|
| 128 (甜点) | 16 | 2.14s | 52 | 71.2% | 50.0% | ✅ 吞吐最优 |
| 1024 | 128 | 15.4s | 57.7 | **91.5%** | 51.2% | ✅ 安全上限, 占用拉满 |
| 1152 | 144 | 18.3s | 59.4 | **95.6%** | 51.8% | ✅ 触顶前, 占用最高 |

**4 卡** (4 生产者, 真 4 卡分片):

| bs | 每卡样本 | step_time | throughput | SM% busy_mean | 全样本 | 结果 |
|---|---|---|---|---|---|---|
| 128 (甜点) | 32 | 2.16s | 52 | 74.5% | 50.0% | ✅ 吞吐最优 |
| 512 | 128 | 12.1s | 36.7 | **94.5%** | 59.8% | ✅ 安全上限, 占用拉满 |
| 768 | 192 | — | — | — | — | ❌ OOM (优雅, alloc 56.7GB) |

**服务化选型**:
- 要**吞吐**: bs=128 (8 卡/4 卡都 52 samples/s, 最优)。
- 要**GPU 占用率** (如按 GPU 时计费、或想让单卡算满): 8 卡 bs=1024 (busy 91.5%, 57.7 samples/s, 仍安全); 4 卡 bs=512 (busy 94.5%, 但吞吐反降到 36.7 — 4 卡大 batch 不划算)。
- **触顶点与卡数无关, 由每卡样本数决定**: 每卡 ≤128 安全、每卡 192 OOM (alloc 56.7GB)、每卡 ~160 崩溃区。8 卡安全上限 bs=1024, 4 卡安全上限 bs=512。
- **4 卡 server 起法陷阱**: `_start_pi05_server_8gpu.sh:24` 硬编码 `CUDA_VISIBLE_DEVICES=0-7`, 会覆盖命令行传的 4 卡限制 → 假 4 卡 (8 卡都占, 每卡 79.6GB)。4 卡必须 inline 起 server (不经过该脚本), 见下文"跑 4 卡"。

### 6. async 流水线 → 已作废, 不要复活
曾试 httpx.AsyncClient 让 HTTP 往返与 GPU 重叠, 前台 3 步 ~60 samples/s 但 step4 静默退出 (无 traceback, py-spy 抓不到 host-venv standalone python)。**根因是 §1 的误诊**: 同步版在 fresh server 上已达水位, async 解的是不存在的问题。代码已回退, 同步版是唯一推荐实现。

### 7. polling 不是瓶颈 (别去优化)
`/api/v1/mint/vla/train_step` 返回 `request_id`, 但 server 端 `handle_train_step` 用 `_run_inline` **同步 await 完 forward_backward+optim_step 才返回**。client 第一次 `/retrieve_future` 就拿到结果, 没有 1s 轮询空等。别试图把 `poll_interval_s=1.0` 调小或改 long-poll。

### 8. SM% 短跑测不准 → 必须 ≥60 步 + fresh server
同一配置 (bs=128, p=4, 4 卡) 的 SM 利用率 (nvidia-smi `utilization.gpu`) 随步数读出**不同值**: 短跑 ~33% → 60 步 ~50% → 400 步才是真实 **74.5%** (busy_mean)。但 `step_time` 全程 2.16s 不变, 说明 GPU 实际计算量从未改变 —— 变的只是采样能否抓到稳态满载段。短跑时 server JIT 编译抖动 + 预热污染了"忙窗"判定。这与 §4 (短跑 throughput 偏低) 是同一类口径陷阱。

**测 SM 利用率的正确姿势**:
- 高频采样 `nvidia-smi --query-gpu=utilization.gpu -i <visible-gpus> -d 0.2` (1s 采样 vs ~2-3s step 会大量落在 HTTP 间隙, 均值失真)。
- 只采 server 实际可见的卡 (4 卡 server 采 0-3, 别采全部 8 张把空闲卡拉低均值)。
- `gpu_busy_mean` = 仅在"任一卡 >5% busy"的样本上求均值 (排除 HTTP 空闲间隙); 另报"忙窗均值" (样本 >5% busy 时) 看纯计算阶段 (~90%)。
- **跑 ≥60 步**, 且 server 必须 fresh。30 步的 33% 是错的。

## 跑 4 卡 (限 server 可见 GPU)

8 卡脚本能直接复用到 4 卡, **前提是起一个只看 4 张 GPU 的 fresh server** —— 数据并行分片数由 server 进程的 `CUDA_VISIBLE_DEVICES` 决定, client 无法单方面限制。bs 仍是"可见 GPU 数 (4) 的倍数"。

起 4 卡 fresh server — **必须 inline 起, 不要用 `_start_pi05_server_8gpu.sh`** (该脚本第 24 行硬编码 `CUDA_VISIBLE_DEVICES=0-7`, 会覆盖你传的 4 卡限制 → 假 4 卡, 8 张卡全占满 79.6GB/卡, bs=512 都 OOM):

```bash
# inline 起, 绕开硬编码脚本; 关键: CUDA_VISIBLE_DEVICES=0,1,2,3
REPO=/vePFS-Mindverse/user/intern/wenxi/mint
GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
EXTRA=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
export CUDA_VISIBLE_DEVICES=0,1,2,3 MINT_PORT=30560 MINT_HOST=0.0.0.0
export MINT_UVICORN_WORKERS=1 MINT_SKIP_SUPERVISOR=1 MINT_ALLOW_NO_RAY=1 MINT_USAGE_BACKEND=disabled
export MINT_RAY_NAMESPACE="vla_4gpu" MINT_SUPPORTED_MODELS="openpi/pi05-libero-low-mem-finetune"
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR=/vePFS-Mindverse/share/mint/dev/data/wenxi/openpi-pi05-checkpoints
export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1 MINT_RUNTIME_CHECKPOINT_DIR=/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${EXTRA}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
nohup "$GRB/host-venv/bin/python" -u "$REPO/scripts/wip/_run_local_openpi_server.py" > /tmp/pi05_server_4gpu.log 2>&1 &
# 验证真 4 卡: 模型加载后应只有 GPU 0-3 占用 (~62GB), 4-7 全空
curl -s http://127.0.0.1:30560/openapi.json >/dev/null && echo ready
nohup "$GRB/host-venv/bin/python" -u "$REPO/scripts/wip/_run_local_openpi_server.py" > /tmp/pi05_server_4gpu.log 2>&1 &
curl -s http://127.0.0.1:30540/openapi.json >/dev/null && echo ready   # 200 即就绪
```

4 卡推荐参数 (2026-07-31 实测, fresh server, bs 必须是 4 的倍数):

| 配置 | step_time | throughput | SM% busy_mean | 备注 |
|---|---|---|---|---|
| **bs=128 + 4 生产者** | 2.16s | 52.2 samples/s | **74.5%** (忙窗 ~90%) | 4 卡最优; loss 1.01→0.089 |
| bs=128 + 8 生产者 | ~2.13s | ~53 | — | ⚠️ step~221 静默退出 (生产者多于卡数), **4 卡别用 >4 生产者** |
| bs=256 + 8 生产者 | 6.3s | 40.6 | ~48 | **4 卡不 OOM 但 step_time 翻倍, 无收益, 别用** |

- **生产者数 = 卡数** (4 生产者 ↔ 4 卡); >4 生产者会撞静默退出, <4 会饿队列。
- 4 卡吞吐 ~52 samples/s (≈ 8 卡 52, 因 4 卡每步算 2× 样本但卡数减半), busy_mean 74.5% / 忙窗 ~90% —— 与 8 卡同形态 (HTTP-bound), 全样本均值 50% (HTTP 往返 + optim Python 遍历空转)。

## 关键脚本

- `scripts/train/train_http_multiprod.py` — 本 skill 的训练脚本, 纯 HTTP + 多生产者预取。
- `scripts/train/openpi_vla_smoke_lance_base.py` — HTTP/dataset/transform helper (L 模块); `train_http_multiprod` import 它。mint driver 也 import 同一份。
- `scripts/remote/run_client.sh` — 启动器, 配 PYTHONPATH (含 openpi) + 预检 server。

## 实测水位 (fresh server, 8×A800, mano lance, 同口径 400 步 + 0.2s 采样, 2026-07-31)

两个 SM 口径都要看:
- **busy_mean**: 仅在"任一卡 >5% busy"的样本上求均值 (排除 HTTP 空闲间隙) = GPU 真在算时的利用率。
- **全样本均值**: 含 HTTP 往返 + optim Python 遍历的空闲间隙 (GPU 空转) = 整体占空比。

| 配置 | step_time | throughput | SM% busy_mean | SM% 全样本 | 忙窗峰值 | 备注 |
|---|---|---|---|---|---|---|
| 8 卡 bs=128 + 8 生产者 (同步) | 2.14s | 51.6 samples/s | **71.2%** | 50.0% | ~90% | 8 卡; loss 1.05→0.097 |
| 4 卡 bs=128 + 4 生产者 | 2.16s | 52.2 samples/s | **74.5%** | 50.0% | ~90% | 4 卡; loss 1.01→0.089 |
| 2 卡 bs=128 + 2 生产者 | 5.43s | 19.5 samples/s | **92.4%** | 62.4% | ~90% | 2 卡; loss 1.01→0.14 |
| 1 卡 bs=128 + 1 生产者 | 10.03s | 11.3 samples/s | **97.9%** | 80.7% | ~90% | 1 卡; loss 0.98→0.18 |
| 8 卡 bs=128 + 8 生产者 (旧 server) | ~4.9s | ~25 | ~23% | — | — | stale-server 误诊, 非瓶颈 |

**卡数规律 (实测 1/2/4/8 卡, bs=128)**: bs=128 适用于所有卡数 (每卡样本 128/64/32/16, 都在安全上限内)。**GPU 越少, 单卡 busy_mean 越高但吞吐越低**: 1 卡 busy_mean 97.9% 但吞吐仅 11.3; 8 卡 busy_mean 71.2% 但吞吐 52 (4.6×)。原因: 卡少 → 每卡算的样本多 → GPU 计算段长、HTTP 空闲占比低 → busy_mean 高; 但总算力受卡数限, 吞吐仍随卡数涨。**4 卡与 8 卡吞吐相同 (都 52)**: 4 卡每卡算 32 (8 卡 2 倍) 但 step_time 没翻倍 (2.16s ≈ 2.14s), 故吞吐持平 —— 4→8 卡翻倍卡数吞吐几乎不变。**推荐**: 卡多选 8 卡 (吞吐最高); 卡少时 1 卡也能跑 (单卡算满 97.9%), 只是吞吐按卡数线性降。

对照 mint driver (同 fresh 8 卡 server): 2.1s/57 — 一致。

**GPU 已打满到实用区**: 忙窗峰值 ~90% (8 卡同步涨跌, 数据并行 `PartitionSpec(DATA_AXIS)` 真分片), busy_mean 71-75%。全样本均值 50% 是因为每步有约一半时间在 **HTTP 往返 + optim_step 的 Python pytree 遍历** (server 端 `_run_inline` 同步执行, GPU 空等) —— 这是同步 HTTP 训练的结构性上限, 不是数据并行没生效。要继续提升需把 optim Python 遍历挪进 jit (碰 `mint_server` 源码, 越界, 需授权)。

**4 卡 SM 略高于 8 卡 (74.5% vs 71.2%)**: 4 卡每卡算 bs/4=32、8 卡每卡算 bs/8=16, 4 卡单卡负载更满、空闲间隙更少。但 4 卡吞吐与 8 卡相当 (52 vs 52, 因 4 卡 step_time 2.16s ≈ 8 卡 2.14s) —— 4 卡每步算 2 倍样本但卡数减半, 总吞吐持平。8 卡吞吐上限更高 (能加大 batch); 4 卡单卡吃得更满。
