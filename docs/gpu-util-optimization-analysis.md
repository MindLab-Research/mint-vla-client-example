# 4 卡 / 8 卡 GPU 利用率优化分析

**目标**: 在纯 client + 纯数据并行 (不碰 mint/openpi 源码) 的约束下, 把 8 卡 SM 利用率从 ~50% 拉高、4 卡从 70% 进一步压榨, 缩短 step_time。
**生成日期**: 2026-07-30
**数据来源**: 本会话实测 (fresh server, bs=128, 400 步, 0.2s 采样), 代码路径 `mint_server/backend/openpi/openpi_pi05_worker.py`

---

## 1. 实测现状 (同口径, fresh server, 400 步, 0.2s 采样, 2026-07-31)

| 配置 | step_time | throughput | SM% busy_mean | SM% 全样本 | 忙窗峰值 |
|---|---|---|---|---|---|
| 8 卡 bs=128 + 8 生产者 | 2.14s | 51.6 samples/s | **71.2%** | 50.0% | ~90% |
| 4 卡 bs=128 + 4 生产者 | 2.16s | 52.2 samples/s | **74.5%** | 50.0% | ~90% |

- **两个 SM 口径**: busy_mean = 仅"任一卡 >5% busy"样本上求均值 (GPU 真在算时); 全样本均值 = 含 HTTP 往返 + optim Python 遍历的空闲间隙 (整体占空比)。**之前文档误把"全样本均值 49.8%"标成 busy_mean, 真实 busy_mean 一直是 71-75%**。
- 4 卡 busy_mean 略高于 8 卡 (74.5% vs 71.2%): 4 卡每卡算 32 样本 (负载更满), 8 卡每卡算 16 (更快进入"等 HTTP"空窗)。
- **吞吐 4 卡 ≈ 8 卡** (52 vs 52): 4 卡每步算 2× 样本但卡数减半, 总吞吐持平; 8 卡能加大 batch 拉高上限。
- GPU 已打满到实用区: 忙窗峰值 ~90% (8 卡同步涨跌, 数据并行真分片), 全样本均值 50% 是同步 HTTP 训练的结构性上限 (HTTP 往返 + optim Python 遍历空转), 非数据并行未生效。

## 2. 瓶颈解剖: 每步 ~2.16s 花在哪

server 端单步 = `forward_backward()` → `optim_step()` 两段, 全程同步 (client `_post_json` 阻塞到算完, 实测 `post_time` 占满 step, `retrieve≈0`)。

### `forward_backward` → `_compute_flow_matching_grads_batched` (worker.py:588-623)

| 阶段 | 代码 | 位置 | GPU 状态 |
|---|---|---|---|
| ① 图解码+stack | `_stack_flow_matching_batch` (489-523), `np.stack([_decode_image(...) ...])` 串行解 128 张图 | CPU/Python | **空闲** |
| ② host→device | `jax.device_put(obs, input_sharding)` (610-611) | PCIe | **空闲** |
| ③ forward+backward+allreduce | `grad_fn(...)` jitted (616-618) | GPU | **满载** |

### `optim_step` (worker.py:1079-1109) — **最大确定性空窗**

| 阶段 | 代码 | GPU 状态 |
|---|---|---|
| ④ optimizer update | `tx.update` + `apply_updates` (1095-1096) | 部分 GPU |
| ⑤ **Python pytree 遍历** | `params.filter(...)` (1094) + `nnx.merge` (1098) + `nnx.state`/`nnx.update` (1099-1100) + `dataclasses.replace` (1101) | **空闲 (CPU 串行)** |

**关键**: ⑤ 是纯 Python/nnx pytree 遍历, **不在 jit 内**, 每步都跑, GPU 干等。这是空窗的主要来源之一。

### 空窗构成 (8 卡 ~1.1s 空窗 / 4 卡 ~1.0s 空窗, step_time ~2.16s)
```
GPU 满载:  ③ grad_fn (forward+backward+allreduce)          ~1.1s (8卡) / ~1.2s (4卡)
GPU 空闲:  ① 图解码 + ② device_put + ⑤ optim Python 遍历 + 同步HTTP  ~1.0s (8卡) / ~1.0s (4卡)
```
满载/空闲约 1:1, 对应全样本均值 ~50% (含空闲间隙); busy_mean (仅忙窗) ~71-75%, 忙窗峰值 ~90%。8 卡每卡算 16、4 卡每卡算 32, 但两者 step_time 相近 (4 卡算 2× 样本 × 半数卡), 故空窗占比接近。

---

## 3. 可优化空间 (按可行性 + 收益排序)

### ❌ A. 重开 XLA command_buffer — 已验证无效 (2026-07-31, 单变量对照)

**现状**: 启动脚本写死 `export XLA_FLAGS="--xla_gpu_enable_command_buffer="` (等于**关闭**)。
**假设**: command_buffer 合并 CUDA kernel launch, 减少 CPU dispatch 开销 → 压缩 launch 间隙。
**实测 (同 server、同 4 生产者、同 400 步、同 0.2s 采样, 单变量只切 CB)**:

| | 关 CB | 开 CB | 变化 |
|---|---|---|---|
| 4 卡 step_time | 2.16s | 2.17s | **0** |
| 4 卡 busy_mean | 74.8% | 74.5% | -0.3% (噪声) |
| 8 卡 step_time | 2.14s | 2.14s | **0** |
| 8 卡 busy_mean | 71.2% | 72.6% | +1.4% (噪声) |

**结论: 无效, 别开。** 瓶颈不在 kernel launch 间隙, 而在 HTTP 往返 + optim Python 遍历 (§2 的 ①②⑤), command_buffer 碰不到这些。GPU 计算段本就满载 (忙窗 ~90%), CB 优化的是已满载段之间的 launch 缝, 那段缝不是瓶颈。维持启动脚本关闭即可 (关闭也是历史 OOM 缓解, padding 修复后虽不再必要, 但开了也没收益, 保持现状)。
**前置已确认**: worker `_padded_prompt` (openpi_pi05_worker.py:435) padding 修复仍在 → shape 恒定、只编译一次 → 开 CB 不会触发历史 OOM (实测 400 步无显存爬升)。

### ✅ B. 确认/调高 JAX 预分配 (只改 server env)

**现状**: 脚本未设 `XLA_PYTHON_CLIENT_MEM_FRACTION` / `JAX_PREALLOC` → JAX 默认预分配 75% 显存。**所以预分配已是默认, 此条价值有限**, 仅当观察到显存碎片化导致的分配抖动时才需调:
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90   # 默认 0.75, 调高给 device_put 留余量
export JAX_PREALLOC=true                     # 显式确认 (默认就是 true)
```
**验证**: 同 A。**预期收益**: 低 (默认已预分配)。

### ⚠️ C. 把 optim_step 的 Python 遍历挪进 jit (需改 server 代码, 越界)

**问题**: ⑤ 的 `nnx.merge`/`nnx.state`/`filter`/`nnx.update` (1094-1100) 是 CPU 端 pytree 遍历, 每步串行, GPU 空等。
**改法思路** (需你决定是否破例碰 `mint_server` 代码): 把 optimizer 更新也 jit 进 grad_fn 或一个独立 jit, 让 ④⑤ 在 GPU 上一次完成, 消除 Python 遍历空窗。
**位置**: `openpi_pi05_worker.py:1079-1109` `optim_step`。
**风险**: 中。nnx 的 state 管理挪进 jit 要小心可变性; 改坏会影响训练正确性。**必须对比 loss 收敛曲线**确认数值不变。
**预期收益**: 高 (直接削掉每步一段 CPU 空窗)。
**边界**: 这条碰 `mint_server` 源码, 违背 skill "不伸手进 mint"。**需明确授权后**才做, 且应作为 A/B 验证完之后的下一步。

### ⚠️ D. 图解码并行化 (需改 server 代码, 越界)

**问题**: ① `_stack_flow_matching_batch` 的 `[ _decode_image(item...) for item in batch ]` (502 行) 串行解码 128 张 PNG, 纯 CPU。
**改法思路**: 线程池并行解码, 或让 client 直接传已解码的 numpy bytes (但 client→server 传输变大, 需权衡)。
**位置**: `openpi_pi05_worker.py:489-523`。
**预期收益**: 中 (仅 ① 段)。
**边界**: 同 C, 碰 server 代码。

### ❌ 已排除的方向 (别再试)

| 方向 | 为什么不行 |
|---|---|
| **FSDP (fsdp_devices>1)** | `fsdp_devices` 默认=1 写死在 openpi config; client `create_model` 只传 `base_model`, 传不了 `config_name`; `OpenPIPi05RuntimeInitOverrides.from_env()` 不覆盖 fsdp。要改 openpi 源码。 |
| **关 train_unembed** | 只减计算量不改分片, throughput 封顶不变, 且影响收敛。 |
| **PNG→JPEG** | 实测真实 payload 仅 4.6MB (transform 后图已 224×224), 编码仅 3.7ms —— 传输/编码根本非瓶颈。 |
| **async HTTP** | 踩坑 §6 已作废; server `_run_inline` 同步执行, HTTP 与 GPU 无法重叠。 |
| **调小 poll_interval** | 踩坑 §7; retrieve≈0 实测, 第一次 poll 即拿结果。 |

---

## 4. 执行顺序与结论

1. ~~**A (command_buffer)** — 已验证无效 (§3 A), 跳过。~~
2. ~~**B (JAX 预分配)** — 默认已开 (75% prealloc), 价值有限, 跳过。~~
3. **结论: 在"不碰 mint 源码"约束下已无可做。** GPU 已打满到实用区 (忙窗 ~90%, busy_mean 71-75%, 全样本 50% = 同步 HTTP 结构性上限)。
4. **C/D (optim Python 遍历挪进 jit / 图解码并行) 是唯一能再提升的方向, 但碰 `mint_server` 源码, 违背 skill "不伸手进 mint"。需明确授权后做, 改完必须 400 步对比 loss 收敛曲线确认数值不变。**

## 5. 验证口径 (避免重蹈短跑陷阱)

- **400 步** + fresh server (短跑 SM 不可信, 见 skill §8)。
- 0.2s 采样 `nvidia-smi --query-gpu=utilization.gpu -i <可见卡>`, 只采 server 可见卡。
- `gpu_busy_mean` = 仅在"任一卡 >5% busy"样本上求均值 (排除 HTTP 间隙)。
- 主指标: SM% busy_mean + step_time; 辅助: throughput = bs/step_time。
- 复现脚本: `results/sweep_4gpu/` 下 `sweep_one.sh` + 各 .csv/.log。

## 6. 关键代码位置速查

| 段 | 文件:行 | 内容 |
|---|---|---|
| forward+backward | `openpi_pi05_worker.py:588-623` | `_compute_flow_matching_grads_batched` |
| 图解码 (①) | `:489-523` | `_stack_flow_matching_batch` |
| device_put (②) | `:610-611` | host→device |
| grad_fn (③) | `:580-585` | jitted grad |
| optim_step (④⑤) | `:1079-1109` | optimizer + nnx Python 遍历 |
| mesh/sharding | `:310-325` | `make_mesh(fsdp_devices)`, fsdp=1 写死 |
| fsdp override | openpi `training/config.py:535` | `fsdp_devices: int = 1` |
