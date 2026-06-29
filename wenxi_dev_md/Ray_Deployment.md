# Ray_Deployment.md — 把 MinT 三机 dev 流程迁移到单台 8 卡机

> 目的：把当前「本地 client → driver(192.168.42.106) → Ray head(Volcano pod) →
> GPU worker pods」的三机拓扑，整体搬到**一台 8 卡机**上，直接在这台机器上完成一整套
> Ray 部署，并以 `RLcheck.sh`（Qwen3.6-27B text RL）跑通端到端验证。
>
> 配套文档：`Plan.md`(VLA WHAT/WHY)、`Excute.md`(VLA HOW)、`VLM.md`(Qwen3.6 VLM 扩展)。
> 本文件专门记录**单机 Ray 部署**的理解与施工蓝图。
>
> 决策（2026-06-29，已与用户对齐）：
> 1. Ray 布局 = **Head + 独立 worker 进程**（同机两进程，模拟 prod 的 head/worker 拆分）
> 2. 文件系统 = **PFS `/vePFS-Mindverse/share/` 已挂载**，复用现有 code mirror + gpu_rl runtime
> 3. 首个验证目标 = **RLcheck.sh**（Qwen3.6-27B text RL）端到端

---

## 0. MinT 是什么（先对齐事实）

MinT(`mint-server`) 是一个 **FastAPI 控制平面，不是计算引擎**。它只负责 HTTP、鉴权、
请求校验、异步 future 轮询，把所有 GPU 工作**经 Ray 派发给 worker 节点上的 detached
actor**。长任务返回 `{request_id}`，客户端轮询 `POST /api/v1/retrieve_future`
（408=pending，200=done）。

```
本地 client ──HTTP──> mint-server (FastAPI, CPU driver) ──Ray 直连 GCS──> GPU worker actors
                                                                          ├─ vLLM   (推理, 多 LoRA)
                                                                          ├─ Megatron/verl (文本训练)
                                                                          └─ OpenPI (VLA 训练+动作推理)
```

必须在 API 起来之前/周围存在的 detached 控制平面 actor（`bootstrap_control_plane.py` 顺序）：
`mint_config → mint_task_state_store → mint_model_work_scheduler →
mint_maintenance_cron → mint_model_actor_supervisor`。**server 重启只丢进程内缓存，
detached actor 存活**——所以清理时必须显式 kill 这些 actor（见 `clean2.sh`/`cleanup.sh`）。

---

## 1. 当前三机拓扑（迁移前）

| 角色 | 机器 | 跑什么 |
|------|------|--------|
| 本地 dev box | 本仓库 `/vePFS-Mindverse/user/intern/wenxi/mint` | 改代码；跑 client/tool 脚本(走 HTTP) |
| **Driver** | `mint-dev-driver` / `192.168.42.106:2222`(别名 `driver`/`mint-dev`) | 跑 mint-server API 进程；**直连** Ray head 的 GCS |
| **Ray head** | Volcano pod，IP 见 `ray_head_ip.txt`(当前 `192.168.42.183`) | Ray GCS + raylet + dashboard(`:8265`)；GPU **worker** 是另外的 Volcano pod 注册上来 |

铁律（来自 CLAUDE.md / mint-dev skill）：
- API server 跑在 **driver，绝不跑在 head pod 上**。
- 连接方式 = **直连 GCS**(`MINT_RAY_GCS_ADDRESS=<head_ip>:6379`)，**绝不用 Ray Client
  模式**(`ray://...:10001`)——后者在 Mint 用法下导致过 GCS 不稳定。
- 代码必须放在 `/vePFS-Mindverse/share/` 下，让**所有 Ray 节点**(head+worker)可见；
  worker 通过 `PFS_PYTHONPATH` import `mint_server`，**不走** Ray job 上传。

---

## 2. 当前 `step*.sh` 如何实现这套流程

1. **step1.sh** — `rsync` 本地 checkout → PFS mirror
   `/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi/`。**禁止 `--delete`**。
2. **step2.sh** — 读 head IP，跑 `gen_dev_placement.py` 查 Ray dashboard 上 alive 的
   GPU worker，写 `/tmp/mint_dev_run.env`，把 `Qwen/Qwen3.6-27B → {node_ip,gpu_count:4}`
   映射出来，再 `scp` 到 driver。这是「哪个模型放哪个 worker」的决策。
3. **step3.sh** — `ssh driver` 启动 `start_dev_server.sh`：传 `MINT_CODE_ROOT`、
   namespace `mint_wenxi_dev`、placement env、runtime root、supported-models。
   launcher 由 namespace hash 派生端口(→`30496`)，bootstrap 控制平面 actor，
   然后 `exec run_server.py`。
4. **step4.sh** — `/api/v1/healthz` 健康检查。
5. **step5.sh** — SSH 隧道 `localhost:8000 → driver:30496`，本地 client 才能访问。
6. **step6.sh** — 诊断：healthz / server_info / admission_stats / 日志 tail /
   dashboard cluster_status。

验证：`RLcheck.sh`=token RL(`rl_check.py`,Qwen3.6-27B)；`VLA_check.sh`=OpenPI 动作模态。
清理：`clean1.sh` 杀 server 进程；`clean2.sh`/`cleanup.sh` 杀 namespace 内 detached actor。

---

## 3. 单机迁移：哪些变，哪些不变

核心简化：**driver / head / worker 三个角色塌缩到一台机**，全部 `127.0.0.1` 可达。
PFS 仍挂载 → code mirror + `gpu_rl` runtime tier 路径不变。**去掉** `ssh driver`、
`scp` placement、Volcano pod、SSH 隧道。

| 当前（三机） | 单台 8 卡机 |
|---|---|
| step1: rsync 本地 → PFS mirror | **保持不变**。worker 仍经 `PFS_PYTHONPATH` 从 mirror import `mint_server`。 |
| Ray head = 远程 Volcano pod；head IP 读 `ray_head_ip.txt` | **本机** `ray start --head --num-gpus=0`。head IP = `127.0.0.1`。 |
| GPU worker = 另外的 Volcano pod 自动注册 | **本机**第二个进程 `ray start --address=127.0.0.1:6379 --num-gpus=8`。 |
| step2: 查远程 dashboard，scp placement 到 driver | 本机跑 `gen_dev_placement.py --head-ip 127.0.0.1`，读 `127.0.0.1:8265` dashboard，placement 指向本机 worker node IP。**无 scp**。 |
| step3: `ssh driver '... start_dev_server.sh'` | **本机直接**跑 `start_dev_server.sh`(无 ssh)，`MINT_RAY_GCS_ADDRESS=127.0.0.1:6379`。 |
| step4/5/6: `ssh driver curl` + 隧道 | 去掉 ssh wrapper 和隧道，直接 curl `localhost:30496`。 |
| clean*: `ssh driver` + 远程 head IP | 同逻辑，本机化：`127.0.0.1:6379`，无 ssh。 |

---

## 4. 必须留意的具体旋钮

1. **直连 GCS 到 localhost。** `start_dev_server.sh` 默认从 `ray_head_ip.txt` 读 head IP
   并设 `MINT_RAY_GCS_ADDRESS=<ip>:6379`(脚本 ~315 行：若已设则沿用)。单机显式设
   `MINT_RAY_GCS_ADDRESS=127.0.0.1:6379`。**仍是直连，不是 `ray://`**——这条铁律不变。

2. **Head 必须 0 GPU；worker 拿全部 8 卡。** 这正是 CLAUDE.md 里的 prod 不变量
   ("DLC head 必须 0 GPU")。若 head 进程也广播 GPU，placement 生成器的「跳过 head 节点」
   过滤(`gen_dev_placement.py:104`)会被搞混。所以 `--head --num-gpus=0`，
   worker `--num-gpus=8`。

3. **placement gpu_count。** step2 给 Qwen3.6-27B 请求 `--gpu-count 4`(known-models 表
   也是 4)。单台 8 卡 worker 放得下。生成器挑剩余 GPU 最多的 worker，单模型单节点直接成立。

4. **runtime 解释器。** `start_dev_server.sh:326` 要求
   `${PFS_RUNTIME_ENV_ROOT}/cpu/base-python/bin/python3.13` 存在。PFS 已挂载、复用
   `/vePFS-Mindverse/share/mint/dev/runtime`，该解释器存在。worker 实际 GPU 工作用
   `gpu_rl` tier(含 torch/vllm)，由 runtime_env 逐 actor 解析，与 API 进程解释器无关。

5. **同机共存。** `cpu` tier 解释器本意给 CPU-only API host；单 GPU 机上 API 进程与 GPU
   worker 共享机器，没问题——真正决定 GPU 工作的是 worker 的 `gpu_rl` runtime_env。

---

## 5. 上机前必须先做的只读 pre-flight（不要跳过）

> 来自 Hall of Shame 的教训：版本不匹配、stale PG、错误 PYTHONPATH 浪费过整夜。
> **先验证、再动手**，任何 `ray start` 之前先把下面这些查清楚并写下来。

```bash
# 5.1 PFS 是否可见 + runtime tier 在不在
ls -ld /vePFS-Mindverse/share/mint/dev/runtime/cpu /vePFS-Mindverse/share/mint/dev/runtime/gpu_rl
ls /vePFS-Mindverse/share/mint/dev/runtime/cpu/base-python/bin/python3.13

# 5.2 GPU 数量与型号
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 5.3 是否已有 Ray 在跑（避免端口/版本冲突）
pgrep -af raylet || echo "no raylet"
ps aux | grep -E "ray (start|--head)" | grep -v grep || echo "no ray procs"

# 5.4 本机将用于 ray start 的解释器 + Ray 版本（必须与 gpu_rl tier 的 Ray 版本一致）
PY=/vePFS-Mindverse/share/mint/dev/runtime/cpu/base-python/bin/python3.13
$PY -c "import ray, sys; print('ray', ray.__version__, 'py', sys.version)"
GPU_PY=/vePFS-Mindverse/share/mint/dev/runtime/gpu_rl/host-venv/bin/python   # 路径需现场确认
$GPU_PY -c "import ray, sys; print('gpu_rl ray', ray.__version__, 'py', sys.version)" 2>/dev/null || echo "confirm gpu_rl python path"

# 5.5 本机 IP / hostname（确认 raylet 自报的 node_ip 是不是 127.0.0.1）
hostname -I
```

**必须确认的两件事：**
- (a) Ray 版本：用来 `ray start` 的解释器的 Ray 版本，**必须**与 worker `gpu_rl` tier
      的 Ray 版本一致，否则 actor 创建/调度报版本不匹配（Hall of Shame 2026-03-13）。
- (b) raylet 自报 node_ip：Ray 有时把节点绑到可路由 IP 而非 `127.0.0.1`，这个 IP 会
      作为 placement 里的 "worker node_ip"。先用 `hostname -I`/dashboard 看清楚，
      placement 的 `--head-ip` 和 worker 注册 IP 要一致。

---

## 6. 单机部署步骤（蓝图，pre-flight 通过后逐条落地）

> 命名沿用现有：namespace `mint_wenxi_dev`，端口由 namespace hash 派生 = `30496`。
> 占位变量上机后用 5.x 的实测值替换。

### step0_ray_up.sh（新增，替代远程 head/worker pod）
```bash
# 用与 gpu_rl tier 一致的 Ray 解释器（pre-flight 5.4 确认）
PY=/vePFS-Mindverse/share/mint/dev/runtime/cpu/base-python/bin/python3.13

# head：0 GPU，启 GCS + dashboard
$PY -m ray start --head --num-gpus=0 \
  --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# worker：注册到本机 GCS，广播 8 卡
$PY -m ray start --address=127.0.0.1:6379 --num-gpus=8

# 验证：head 0 GPU、worker 8 GPU、共 2 节点
$PY -c "import ray; ray.init(address='127.0.0.1:6379'); print(ray.cluster_resources()); ray.shutdown()"
```
（若 head 与 worker 需不同解释器，worker 用 gpu_rl 解释器 `ray start`——上机现场定。）

### step1_sync.sh（保持不变）
```bash
rsync -a --exclude '.git' --exclude '__pycache__' \
  ./ /vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi/
```

### step2_placement.sh（本机化：head-ip=127.0.0.1，无 scp）
```bash
rm -f /tmp/mint_dev_run.env
python scripts/tools/gen_dev_placement.py --head-ip 127.0.0.1 \
  --model Qwen/Qwen3.6-27B --gpu-count 4 \
  --output /tmp/mint_dev_run.env
# 若 raylet 自报 IP 非 127.0.0.1，则 --head-ip 用 hostname -I 的实测 IP
cat /tmp/mint_dev_run.env   # 确认 node_ip 是本机 worker IP
```

### step3_start.sh（本机化：无 ssh，显式 GCS=127.0.0.1）
```bash
MINT_CODE_ROOT=/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi \
MINT_DEV_USER=wenxi \
MINT_RAY_NAMESPACE=mint_wenxi_dev \
MINT_RAY_GCS_ADDRESS=127.0.0.1:6379 \
MINT_TASK_STATE_STORE_DB_PATH=/vePFS-Mindverse/share/mint/dev/data/wenxi/task-state/task_state.sqlite3 \
MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log \
MINT_DISABLE_MINT_ROUTE=0 \
MINT_UVICORN_WORKERS=1 \
MINT_SUPERVISOR_STATE_BACKEND=memory \
MINT_SUPPORTED_MODELS="Qwen/Qwen3.6-27B,Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507" \
MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/mint/dev/runtime \
nohup /vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi/scripts/start_dev_server.sh \
  >> /tmp/mint_dev_launch_wenxi.log 2>&1 &
```

### step4_health.sh（本机化：直接 curl，无 ssh/隧道）
```bash
curl -s http://localhost:30496/api/v1/healthz; echo   # 期望 {"status":"ready"}
```

### RLcheck（首个验证目标；本机无需 SSH 隧道）
```bash
MINT_BASE_URL=http://localhost:30496 \
TINKER_API_KEY=tml-dummy MINT_API_KEY=tml-dummy \
python scripts/tools/rl_check.py \
  --model Qwen/Qwen3.6-27B --steps 10 --group-size 4 --timeout-s 600
```

### 清理（本机化：head IP=127.0.0.1，无 ssh）
```bash
# 1. 杀 server 进程
kill $(pgrep -f "scripts/run_server.py" | head -1) 2>/dev/null; sleep 2
# 2. 杀 namespace 内 detached 控制平面 actor（复用 cleanup.sh 的 python 片段，
#    HEAD_IP 改 127.0.0.1，去掉 ssh driver 包装）
# 3. 若要彻底停 Ray：ray stop（先确认本机没有别人在用这套 Ray）
```

---

## 7. 上机执行顺序（checklist）

- [ ] §5 pre-flight 全部跑完，记录 GPU 数 / Ray 版本 / 本机 IP / runtime 路径
- [ ] 确认 (a) Ray 版本一致、(b) raylet node_ip
- [ ] step0：起 head(0 GPU)+worker(8 GPU)，`cluster_resources()` 见 8 GPU
- [ ] step1：rsync 代码到 PFS mirror
- [ ] step2：生成 placement，确认 node_ip 正确
- [ ] step3：本机起 server，看 launcher 打印的 MINT_PORT(=30496)
- [ ] step4：healthz = ready；§6 诊断 server_info/admission_stats 正常
- [ ] RLcheck：跑通 10 步，**关注 reward**（不是 loss，Hall of Shame）
- [ ] 收尾：清理 server 进程 + detached actor

---

## 8. 已知坑 / 铁律提醒

- **绝不 `ray://` Ray Client 模式**，单机也直连 GCS。
- **head 0 GPU、worker 8 GPU**，否则 placement 过滤 head 的逻辑失效。
- **Ray 版本必须 head/worker/gpu_rl 三方一致**（Hall of Shame 2026-03-13）。
- **stale PG/actor 是硬门禁**：新 placement 前先 list 全局 PG，清掉自己 namespace 的
  stale PG，否则 `NO_RESOURCES` 假象（Hall of Shame 2026-03-13）。
- **代码改了必须重启 server**：Python 不热重载（Hall of Shame 2026-01-05）。
- **失败不要缩配置**：按内存模型算，不要 trial-and-error 缩小（CLAUDE.md 铁律）。
- **绝不 `rsync --delete`**。
- 本文档先验证、记录实测值，再断言为事实——上机前所有 `<占位>` 路径/IP 都要现场确认。

---

## 9. 进度日志

- 2026-06-29：与用户对齐三项决策(head+worker 进程 / PFS 已挂 / RLcheck 首验)。
  通读现有三机流程(`step1..6.sh` + `start_dev_server.sh` + `gen_dev_placement.py`
  + `bootstrap_control_plane.py` + mint-dev SKILL)，写出本单机迁移蓝图。
  **下一步：上 8 卡机跑 §5 pre-flight。** 在 pre-flight 实测前不执行任何 `ray start`。

