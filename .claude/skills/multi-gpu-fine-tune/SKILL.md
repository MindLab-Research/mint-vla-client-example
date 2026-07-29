---
name: multi-gpu-fine-tune
description: 纯 HTTP 多卡 LoRA 微调 OpenPI pi0.5 — 不 import mint_server、不调 mint driver, 仅用 requests 打 HTTP, 8 卡 bs=128+8 生产者达 ~55 samples/s (mint 水位). 用于在 client 侧(不碰 mint 代码层级)跑通端到端训练并避免常见性能/启动陷阱.
---

# multi-gpu-fine-tune (纯 client 多卡微调)

通过**纯 HTTP** 在 8 卡 mint-server 上对 OpenPI pi0.5 做 LoRA 微调, **不 import `mint_server`、不 subprocess 调 mint driver**。多生产者预取内置, bs=128 + 8 生产者即可占满 8 卡 GPU (稳态 ~55 samples/s, 与 mint driver 一致)。

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
- 末行 `throughput = batch_size * steps / elapsed`, 把首步 ~80s XLA 编译摊进了每步 —— 短跑 (steps<50) 会显著低估。稳态 `step_time ≈ 2.1-2.2s` → 真实 ~58 samples/s。
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
`throughput = batch_size * steps / elapsed` 把首步 ~80s 编译摊进每步。1000 步实测 55.4 (摊销影响小); 但短跑 (10 步) 会显示 ~12, **别被吓到** —— 看稳态 `step_time≈2.16s` 才是真实水位。要算"稳态吞吐"用 `batch_size / step_time`。

### 5. bs=256 静默退出 → 与"打满 GPU"无关, 别靠加大 batch
bs=256 在 mint driver 稳定版也静默退出 (model created + auto 选 8 后、首步前退出, 0 traceback; server 端有 80s train_step 200 OK 但 client 0 输出)。疑似 OOM 或响应过大。**bs=128 已达 ~55 samples/s, 不需要靠 256 打满**。若未来确需更大 batch, 前台 `--steps 1 --batch-size 256` 看完整 stderr 精准诊断。

### 6. async 流水线 → 已作废, 不要复活
曾试 httpx.AsyncClient 让 HTTP 往返与 GPU 重叠, 前台 3 步 ~60 samples/s 但 step4 静默退出 (无 traceback, py-spy 抓不到 host-venv standalone python)。**根因是 §1 的误诊**: 同步版在 fresh server 上已达水位, async 解的是不存在的问题。代码已回退, 同步版是唯一推荐实现。

### 7. polling 不是瓶颈 (别去优化)
`/api/v1/mint/vla/train_step` 返回 `request_id`, 但 server 端 `handle_train_step` 用 `_run_inline` **同步 await 完 forward_backward+optim_step 才返回**。client 第一次 `/retrieve_future` 就拿到结果, 没有 1s 轮询空等。别试图把 `poll_interval_s=1.0` 调小或改 long-poll。

## 关键脚本

- `scripts/train/train_http_multiprod.py` — 本 skill 的训练脚本, 纯 HTTP + 多生产者预取。
- `scripts/train/openpi_vla_smoke_lance_base.py` — HTTP/dataset/transform helper (L 模块); `train_http_multiprod` import 它。mint driver 也 import 同一份。
- `scripts/remote/run_client.sh` — 启动器, 配 PYTHONPATH (含 openpi) + 预检 server。

## 实测水位 (fresh server, 2026-07-29, 8×A800, mano lance)

| 配置 | step_time | throughput | 备注 |
|---|---|---|---|
| bs=128 + 8 生产者 (同步) | ~2.16s | 55.4 samples/s (1000步) | 8 卡同步满载, loss 0.96→0.087 |
| bs=128 + 8 生产者 (旧 server) | ~4.9s | ~25 | stale-server 误诊, 非瓶颈 |

对照 mint driver (同 fresh server): 2.1s/57 — 一致。
