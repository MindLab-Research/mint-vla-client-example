# VLA 算法集成指南

> 以 OpenPI pi0.5 为参考实现，说明 Mint-Server 的 Ray 集群工作逻辑，以及如何将一个新的 VLA 算法集成进来。

---

## 目录

1. [整体架构](#1-整体架构)
2. [Ray 集群工作逻辑详解](#2-ray-集群工作逻辑详解)
3. [OpenPI 作为参考：端到端数据流](#3-openpi-作为参考端到端数据流)
4. [集成新 VLA 算法的步骤](#4-集成新-vla-算法的步骤)
5. [关键文件速查表](#5-关键文件速查表)
6. [常见坑与注意事项](#6-常见坑与注意事项)

---

## 1. 整体架构

```
                  本地工作站                            Mint-Server (mint-dev / mint-prod)
                  ──────────                           ──────────────────────────────────
 tinker-cookbook ──HTTP──▶  FastAPI Server
                               │
                          ActionSessionManager          Ray 集群 (GPU Workers)
                               │                       ────────────────────────
                          ┌────▼────────────────┐      [Ray Actor: ModelActorSupervisor]
                          │  ModelWorkScheduler  │             │
                          │  (Ray detached actor)│     ┌───────┴────────┐
                          └────────┬────────────┘     │                │
                                   │               OpenPI Actor     vLLM Actor
                                   │               (num_gpus=1)   (num_gpus=N, TP)
                                   │                   │
                                   │              subprocess
                                   │           openpi_pi05_worker.py
                                   │                   │
                                   └──────────────  JAX Model
                                                  (checkpoint)
```

**关键分层：**

| 层次 | 组件 | 作用 |
|------|------|------|
| HTTP 接入层 | FastAPI (`action_sampling.py`) | 接受请求、异步等待结果 |
| 会话管理层 | `ActionSessionManager` | 路由到对应后端、生命周期管理 |
| 调度层 | `ModelWorkScheduler` (Ray detached actor) | 工作队列、副本分配、租约管理 |
| Actor 层 | `OpenPIActionRayRuntimeActor` | Ray actor，持有 GPU，管理 subprocess |
| 执行层 | `openpi_pi05_worker.py` | 真正运行 JAX/PyTorch，做推理/训练 |

---

## 2. Ray 集群工作逻辑详解

### 2.1 三类核心 Ray Actor

```
[mint-server 进程（CPU host）]
    ├─ ModelActorSupervisor    ← detached Ray actor，全局 actor 注册表
    ├─ ModelWorkScheduler      ← detached Ray actor，调度队列 + 租约
    └─ (FastAPI event loop)

[Ray Worker Node（GPU host）]
    ├─ OpenPIActionRayRuntimeActor   ← 每个 action session 一个，num_gpus=1
    ├─ OpenPISharedRayRuntimeActor   ← FAST 模型用，多 session 共享一个 actor
    └─ VllmRuntimeActor              ← 推理引擎，TP 并行
```

### 2.2 调度器工作原理

`ModelWorkScheduler` 是一个 **Ray detached actor**，服务器重启后仍然存活。

核心数据结构：
```
Domain Queue:  "openpi_pi05:pi0.5-vla-v1"  →  [WorkItem, WorkItem, ...]
                                                      ↓
Replica Pool:  replica-0 (actor_handle)    ←  Assignment Loop (每秒)
               replica-1 (actor_handle)
                    ↓
Lease:  { lease_id, consumer_id, expires_at, ... }
```

**工作流程：**
1. 请求进入 → 放入 Domain Queue
2. Assignment Loop（每 1s）：把 backlog 分配给空闲 replica
3. Worker 通过 `claim_from_replica_queue` 领取任务，拿到 Lease
4. Worker 完成后调用 `finish(lease_id)` 或 `renew(lease_id)` 续约
5. Reaper Loop（每 10s）：清理超时 Lease，重新排队

### 2.3 Actor 生命周期

```
创建：
  ActionSessionManager.create_session()
      → ray.remote(num_gpus=1).options(node_affinity=...).remote(...)
      → 等待 actor.ready_metadata.remote() 返回
      → ModelActorSupervisor.register() 注册

使用：
  client.act(...)
      → ray actor 收到调用
      → _ensure_runtime()  (懒加载 subprocess)
      → subprocess.request("act", payload)
      → 返回结果

销毁：
  ActionSessionManager.shutdown_session()
      → client.shutdown_session(action_session_id)
      → subprocess 收到 "shutdown" op，清理内存
      → ray.kill(actor)
      → ModelActorSupervisor.unregister()
```

### 2.4 Subprocess 通信协议

Ray actor 与真正的 JAX/PyTorch 进程之间用 **stdin/stdout JSON** 通信：

```
Ray Actor (Python)              Subprocess Worker (JAX)
─────────────────               ───────────────────────
write to stdin  ─────────────▶  读取 stdin
                                执行操作 (forward pass 等)
read from stdout ◀─────────────  写入 stdout
```

消息格式（每行一个 JSON）：
```json
// 请求
{"op": "act", "payload": {"action_session_id": "xxx", "observation": {...}}}

// 响应
{"status": "ok", "result": {"action": [...], "logprobs": [...]}}
```

**支持的 op（pi0.5 为例）：**
- `create_session` — 加载 checkpoint，初始化模型状态
- `act` — 推理，返回 action
- `train_step` — 训练一步，返回 loss/metrics
- `save_state` — 保存训练状态到磁盘
- `load_state` — 从磁盘恢复训练状态
- `shutdown_session` — 清理单个 session

---

## 3. OpenPI 作为参考：端到端数据流

### 3.1 推理流程（act）

```
POST /api/v1/action/act/
    ↓
action_sampling.py::_do_act()
    ↓
ActionSessionManager.act(session_id, observation, ...)
    ↓
openpi_pi05_action_worker.py::OpenPIPi05ActionSessionManager.act()
    ↓
OpenPIActionRayRuntimeClient.act()
    ↓  [Ray remote call]
OpenPIActionRayRuntimeActor.act()
    ↓  [subprocess stdin]
openpi_pi05_worker.py::handle_act(payload)
    ├─ preprocess observation (images, language)
    ├─ model.predict_action(obs, state)
    └─ return {"action": [...], "logprobs": [...]}
    ↑  [subprocess stdout]
    ↑  [Ray result]
响应返回给调用方
```

### 3.2 训练流程（train_step）

```
POST /api/v1/train_step  (or via tinker SDK)
    ↓
ActionSessionManager.train_step(session_id, batch, ...)
    ↓  [Ray remote call]
OpenPIActionRayRuntimeActor.train_step()
    ↓  [subprocess stdin]
openpi_pi05_worker.py::handle_train_step(payload)
    ├─ batch = deserialize(payload["batch"])
    ├─ loss, grads = model.compute_loss_and_grads(batch)
    ├─ optimizer.apply_updates(grads)
    └─ return {"loss": 0.123, "grad_norm": 0.05, ...}
    ↑  [subprocess stdout]
响应返回
```

### 3.3 会话状态持久化

`openpi_session_state.py` 管理训练状态（不是推理）：
- 状态存到磁盘：`/vePFS-Mindverse/share/mint/dev/session_states/<session_id>/`
- 内容：模型权重 checkpoint + optimizer 状态 + RNG state + metadata
- 用于：服务器重启后恢复训练

### 3.4 两种 Actor 模式对比

| 模式 | 适用场景 | 文件 | 特点 |
|------|----------|------|------|
| **Shared / Pooled** | FAST 模型，轻量推理 | `openpi_shared_ray_runtime.py` | 多 session 共用一个 Ray actor，内存效率高 |
| **Per-Session** | pi0.5，有状态训练 | `openpi_action_ray_runtime.py` | 1 session = 1 Ray actor = 1 GPU，完全隔离 |

新 VLA 算法通常需要 **Per-Session 模式**（有训练状态）。

---

## 4. 集成新 VLA 算法的步骤

假设你的新算法叫 **MyVLA**，框架是 PyTorch（JAX 同理，协议不变）。

### 步骤总览

```
Step 1: 写 Worker 子进程              → myVLA_worker.py
Step 2: 写 Ray Runtime（Actor + Client）→ myVLA_ray_runtime.py
Step 3: 写 Session Manager           → myVLA_session_manager.py
Step 4: 注册到 ActionSessionManager  → action_session_manager.py（4 处改动）
Step 5: 注册模型到 model_registry    → model_registry.py（或环境变量覆盖）
Step 6: 配置 Placement JSON          → 告知 Ray 用哪张/哪些 GPU
Step 7: 冒烟测试                      → 参考 PI05check.sh 风格
```

---

### Step 1：写 Worker 子进程 `myVLA_worker.py`

这是**真正运行模型的进程**，通过 stdin/stdout JSON 协议与 Ray Actor 通信。

参考：`mint_server/backend/openpi/openpi_pi05_worker.py`

```python
# mint_server/backend/openpi/myVLA_worker.py
"""
MyVLA worker subprocess.
协议：每行一个 JSON；stdin 读请求，stdout 写响应。
"""
from __future__ import annotations
import json, sys, traceback
from dataclasses import dataclass
from typing import Any

import torch
# from myvla import MyVLAModel   ← 替换为你的实际包

_sessions: dict[str, "MyVLASession"] = {}

@dataclass
class MyVLASession:
    model: Any
    optimizer: Any
    config: dict

def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def _recv() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return json.loads(line)

# ── op 处理 ──────────────────────────────────────────────────────────────────

def handle_create_session(p: dict) -> dict:
    sid = p["action_session_id"]
    model = torch.load(p["checkpoint_path"])   # ← 替换
    model.eval()
    _sessions[sid] = MyVLASession(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        config=p,
    )
    return {"status": "ok", "action_session_id": sid}

def handle_act(p: dict) -> dict:
    session = _sessions[p["action_session_id"]]
    with torch.no_grad():
        action = session.model.predict(p["observation"])   # ← 替换
    return {"status": "ok", "action": action.tolist()}

def handle_train_step(p: dict) -> dict:
    session = _sessions[p["action_session_id"]]
    session.optimizer.zero_grad()
    loss = session.model.compute_loss(p["batch"])          # ← 替换
    loss.backward()
    session.optimizer.step()
    return {"status": "ok", "loss": loss.item()}

def handle_shutdown_session(p: dict) -> dict:
    sid = p.get("action_session_id")
    if sid and sid in _sessions:
        del _sessions[sid]
        torch.cuda.empty_cache()
    return {"status": "ok"}

OP_HANDLERS = {
    "create_session":    handle_create_session,
    "act":               handle_act,
    "train_step":        handle_train_step,
    "shutdown_session":  handle_shutdown_session,
}

# ── 主循环 ────────────────────────────────────────────────────────────────────

def main() -> None:
    _send({"status": "ready"})   # ← 必须第一行，Ray actor 以此判断进程就绪

    while True:
        try:
            msg = _recv()
        except EOFError:
            break
        op = msg.get("op")
        handler = OP_HANDLERS.get(op)
        if handler is None:
            _send({"status": "error", "error": f"unknown op: {op}"})
            continue
        try:
            _send(handler(msg.get("payload", {})))
        except Exception as e:
            _send({"status": "error", "error": str(e),
                   "traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
```

**关键约定：**
- 启动后第一行输出 `{"status": "ready"}`，否则 Ray actor 会超时报错。
- 每个请求对应**恰好一行**响应（不能换行）。
- `status: "error"` 时要带 `traceback`，方便从 Ray actor 侧看到原因。

---

### Step 2：写 Ray Runtime `myVLA_ray_runtime.py`

Ray actor 负责：启动/持有 subprocess worker、转发请求、管理生命周期。

参考：`mint_server/backend/openpi/openpi_action_ray_runtime.py`
以及：`mint_server/backend/openpi/openpi_ray_runtime.py`（通用 subprocess client 基类）

```python
# mint_server/backend/openpi/myVLA_ray_runtime.py
from __future__ import annotations
import asyncio, json, os, sys
from typing import Any
import ray

# ── Subprocess 通信基类（可复用 openpi_ray_runtime 里的实现）────────────────
# 这里给出一个简化版，实际建议直接继承 OpenPIDirectWorkerClient
class _SubprocessClient:
    """管理单个 worker 子进程的 stdin/stdout 通信。"""

    def __init__(self, process: asyncio.subprocess.Process):
        self._proc = process

    @classmethod
    async def start(cls, python: str, module: str, env: dict) -> "_SubprocessClient":
        merged_env = {**os.environ, **env}
        proc = await asyncio.create_subprocess_exec(
            python, "-m", module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        # 等待 "ready" 信号
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=120.0)
        msg = json.loads(line)
        assert msg.get("status") == "ready", f"unexpected: {msg}"
        return cls(proc)

    async def request(self, op: str, payload: dict, timeout_s: float = 60.0) -> dict:
        msg = json.dumps({"op": op, "payload": payload}) + "\n"
        self._proc.stdin.write(msg.encode())
        await self._proc.stdin.drain()
        line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout_s)
        result = json.loads(line)
        if result.get("status") == "error":
            raise RuntimeError(f"worker error: {result.get('error')}\n{result.get('traceback', '')}")
        return result

    async def shutdown(self) -> None:
        try:
            self._proc.stdin.close()
            await asyncio.wait_for(self._proc.wait(), timeout=10.0)
        except Exception:
            self._proc.kill()


# ── Ray Actor ─────────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1, max_concurrency=1)
class MyVLARayActor:
    """每个 action_session 对应一个 Ray actor，独占 1 GPU。"""

    def __init__(
        self,
        action_session_id: str,
        checkpoint_path: str,
        python_executable: str,
        worker_module: str = "mint_server.backend.openpi.myVLA_worker",
        extra_env: dict | None = None,
    ):
        self._sid = action_session_id
        self._checkpoint_path = checkpoint_path
        self._python = python_executable
        self._module = worker_module
        self._extra_env = extra_env or {}
        self._worker: _SubprocessClient | None = None
        self._ready_metadata: dict | None = None

    async def _ensure_worker(self) -> _SubprocessClient:
        if self._worker is None:
            self._worker = await _SubprocessClient.start(
                python=self._python,
                module=self._module,
                env=self._extra_env,
            )
            # 在 worker 内初始化 session
            await self._worker.request("create_session", {
                "action_session_id": self._sid,
                "checkpoint_path": self._checkpoint_path,
            })
            self._ready_metadata = {"action_session_id": self._sid, "status": "ready"}
        return self._worker

    async def ready_metadata(self) -> dict:
        """ActionSessionManager 调用此方法等待 actor 就绪。"""
        await self._ensure_worker()
        return self._ready_metadata

    async def act(self, action_session_id: str, observation: dict,
                  extra_inputs: dict | None = None) -> dict:
        worker = await self._ensure_worker()
        return await worker.request("act", {
            "action_session_id": action_session_id,
            "observation": observation,
            "extra_inputs": extra_inputs or {},
        })

    async def train_step(self, action_session_id: str, batch: dict) -> dict:
        worker = await self._ensure_worker()
        return await worker.request("train_step", {
            "action_session_id": action_session_id,
            "batch": batch,
        }, timeout_s=300.0)

    async def shutdown_session(self, action_session_id: str) -> None:
        if self._worker:
            await self._worker.request("shutdown_session",
                                       {"action_session_id": action_session_id})
            await self._worker.shutdown()
            self._worker = None


# ── Ray Runtime Client（供 Session Manager 调用）─────────────────────────────

class MyVLARayRuntimeClient:
    """包装 Ray actor，提供异步接口。"""

    def __init__(self, actor: Any, action_session_id: str):
        self._actor = actor
        self._sid = action_session_id

    async def ready(self) -> dict:
        return await self._actor.ready_metadata.remote()

    async def act(self, action_session_id: str, observation: dict,
                  extra_inputs: dict | None = None) -> dict:
        return await self._actor.act.remote(action_session_id, observation, extra_inputs)

    async def train_step(self, action_session_id: str, batch: dict) -> dict:
        return await self._actor.train_step.remote(action_session_id, batch)

    async def shutdown_session(self, action_session_id: str) -> None:
        await self._actor.shutdown_session.remote(action_session_id)
        ray.kill(self._actor)
```

---

### Step 3：写 Session Manager `myVLA_session_manager.py`

Session Manager 是 ActionSessionManager 和 Ray Runtime 之间的**粘合层**，负责：
- 创建/销毁 Ray actor
- 向 ModelActorSupervisor 注册
- 缓存 `action_session_id → client` 映射
- 服务重启后从 detached actor 恢复

参考：`mint_server/backend/openpi/openpi_pi05_action_worker.py`

```python
# mint_server/backend/openpi/myVLA_session_manager.py
from __future__ import annotations
import ray
from mint_server.backend.actors.model_actor_supervisor import (
    model_actor_supervisor, ActorType,
)
from .myVLA_ray_runtime import MyVLARayActor, MyVLARayRuntimeClient

MYVLA_ACTOR_TYPE = ActorType.OPENPI   # 暂时复用 OPENPI；或自行扩展枚举
MYVLA_NUM_GPUS = 1


class MyVLASessionManager:
    def __init__(self):
        # action_session_id → MyVLARayRuntimeClient
        self._clients: dict[str, MyVLARayRuntimeClient] = {}

    # ── 创建 session ──────────────────────────────────────────────────────────

    async def create_session(
        self,
        action_session_id: str,
        checkpoint_path: str,
        python_executable: str,
        node_ip: str | None = None,           # 指定 GPU 节点；None=让 Ray 自动选
        extra_env: dict | None = None,
    ) -> str:
        actor_name = f"mint_myvla_{action_session_id}"

        # 构建 Ray actor options
        options = dict(
            name=actor_name,
            namespace="mint",
            lifetime="detached",              # 服务重启后 actor 仍然存活
            num_gpus=MYVLA_NUM_GPUS,
            max_concurrency=1,
        )
        if node_ip:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            node_id = _node_id_for_ip(node_ip)
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            )

        # 创建 Ray actor
        actor = MyVLARayActor.options(**options).remote(
            action_session_id=action_session_id,
            checkpoint_path=checkpoint_path,
            python_executable=python_executable,
            extra_env=extra_env or {},
        )

        # 等待 actor 就绪（subprocess 加载完模型）
        client = MyVLARayRuntimeClient(actor=actor, action_session_id=action_session_id)
        await client.ready()

        # 向 ModelActorSupervisor 注册（使用工作队列调度）
        supervisor = model_actor_supervisor()
        await supervisor.register.remote(
            actor_name=actor_name,
            actor_type=MYVLA_ACTOR_TYPE,
            num_gpus=MYVLA_NUM_GPUS,
            actor_handle=actor,
            base_model="myvla",
            session_id=action_session_id,
            metadata={"checkpoint_path": checkpoint_path},
        )

        self._clients[action_session_id] = client
        return action_session_id

    # ── 推理 ──────────────────────────────────────────────────────────────────

    async def act(self, action_session_id: str, observation: dict,
                  extra_inputs: dict | None = None) -> dict:
        client = self._get_client(action_session_id)
        return await client.act(action_session_id, observation, extra_inputs)

    # ── 训练 ──────────────────────────────────────────────────────────────────

    async def train_step(self, action_session_id: str, batch: dict) -> dict:
        client = self._get_client(action_session_id)
        return await client.train_step(action_session_id, batch)

    # ── 销毁 session ──────────────────────────────────────────────────────────

    async def shutdown_session(self, action_session_id: str) -> None:
        client = self._clients.pop(action_session_id, None)
        if client is None:
            client = self._try_recover_client(action_session_id)
        if client:
            await client.shutdown_session(action_session_id)

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _get_client(self, sid: str) -> MyVLARayRuntimeClient:
        client = self._clients.get(sid)
        if client is None:
            # 尝试从 detached actor 恢复（服务重启场景）
            client = self._try_recover_client(sid)
        if client is None:
            raise KeyError(f"MyVLA session not found: {sid}")
        return client

    def _try_recover_client(self, sid: str) -> MyVLARayRuntimeClient | None:
        actor_name = f"mint_myvla_{sid}"
        try:
            actor = ray.get_actor(actor_name, namespace="mint")
            client = MyVLARayRuntimeClient(actor=actor, action_session_id=sid)
            self._clients[sid] = client
            return client
        except ValueError:
            return None


def _node_id_for_ip(ip: str) -> str:
    """从 Ray 集群中查找 IP 对应的 node_id。"""
    for node in ray.nodes():
        if node.get("NodeManagerAddress") == ip and node.get("Alive"):
            return node["NodeID"]
    raise ValueError(f"No alive Ray node found for IP: {ip}")
```

---

### Step 4：注册到 `action_session_manager.py`

需要改动 4 处，改动量小，不影响已有后端。

参考文件：`mint_server/backend/openpi/action_session_manager.py`

**4-a. 导入你的 Session Manager**

```python
# action_session_manager.py 顶部 import 区域，仿照 pi05 的导入方式
from mint_server.backend.openpi.myVLA_session_manager import MyVLASessionManager
```

**4-b. 在 `OpenPISessionManager.__init__` 中初始化**

```python
class OpenPISessionManager:
    def __init__(self, ...):
        # ... 已有的 FAST / pi05 初始化 ...
        self._myvla = MyVLASessionManager()   # ← 新增
```

**4-c. 在 `create_session` 中路由**

```python
async def create_session(self, base_model: str, ...):
    if _is_openpi_fast_model(base_model):
        return await self._openpi_fast.create_session(...)
    elif _is_openpi_pi05_model(base_model):
        return await self._openpi_pi05.create_session(...)
    elif _is_myvla_model(base_model):          # ← 新增
        return await self._myvla.create_session(
            action_session_id=action_session_id,
            checkpoint_path=checkpoint_path,
            python_executable=_resolve_python(),
            node_ip=_pick_node_ip(base_model),
        )
    else:
        raise ValueError(f"Unknown model: {base_model}")
```

**4-d. 在 `act` / `train_step` / `shutdown_session` 中透传**

```python
async def act(self, action_session_id: str, ...):
    # 通过 manager lookup 找到正确的后端
    mgr = self._manager_for_session.get(action_session_id)
    # ... 已有逻辑已经处理透传，只要 create_session 时把 mgr 存进去即可
```

> **最简单的做法**：在 `create_session` 里，把 `self._myvla` 存到
> `self._manager_for_session[action_session_id]`，然后 `act/train_step/shutdown`
> 直接从这个字典取，无需额外分支。

**4-e. 添加模型识别函数**

```python
_MYVLA_PREFIXES = ("myvla/", "myorg/myvla")

def _is_myvla_model(base_model: str) -> bool:
    return any(base_model.startswith(p) for p in _MYVLA_PREFIXES)
```

---

### Step 5：注册模型到 `model_registry.py`

`model_registry.py` 控制模型的并行度（TP/EP/CP）、显存上限、别名等。
不注册也能跑，但 Mint 服务不知道这个模型的 GPU 需求，调度可能出错。

参考：`mint_server/backend/core/model_registry.py`

**方式 A：直接在文件里加（推荐，改动持久）**

```python
# model_registry.py，在 _REGISTRY 字典里添加：
_REGISTRY["myvla/myvla-7b"] = ModelConfig(
    model_name="myvla/myvla-7b",
    num_gpus=1,
    description="MyVLA 7B VLA model",
    actor_type=ActorType.OPENPI,   # 复用 OPENPI 类型
)
```

**方式 B：环境变量覆盖（适合快速实验，不修改代码）**

```bash
export MINT_MODEL_CONFIG_OVERRIDES_JSON='{
  "myvla/myvla-7b": {
    "num_gpus": 1,
    "actor_type": "openpi"
  }
}'
```

**方式 C：`MINT_SUPPORTED_MODELS`（控制服务对外暴露的模型列表）**

```bash
export MINT_SUPPORTED_MODELS="myvla/myvla-7b,openpi/pi0.5-vla-v1"
```

---

### Step 6：配置 Placement JSON

Placement JSON 告诉 Mint 该把哪个模型的 actor 放到哪张 GPU 上。
不配置的话 Ray 会自动选节点，但在多机环境下可能选错。

生成工具：`scripts/tools/gen_dev_placement.py`（开发环境）

**手写格式（单机 8 卡，MyVLA 放 GPU 2）：**

```json
{
  "myvla/myvla-7b": {
    "slices": [
      {
        "node_ip": "10.0.1.5",
        "gpus": [2],
        "count": 1
      }
    ]
  }
}
```

**传入方式：**

```bash
# 开发环境：通过环境变量
export MINT_MYVLA_MODEL_PLACEMENT_JSON='{"myvla/myvla-7b": {"slices": [...]}}'

# 或直接传给 run_server.py 的 --model-placement-json 参数
python scripts/run_server.py \
  --model-placement-json '{"myvla/myvla-7b": ...}'
```

> 约定：每个模型的 placement 环境变量名为
> `MINT_<MODEL_NAME_UPPER_SNAKE>_MODEL_PLACEMENT_JSON`，
> 启动脚本 (`step3_start.sh`) 里把它们都 export 一遍。

---

### Step 7：冒烟测试

参考：`PI05check.sh`（项目根目录）

**最小验证脚本 `scripts/wip/myvla_smoke.py`：**

```python
#!/usr/bin/env python3
"""
MyVLA 端到端冒烟测试。
用法：MINT_BASE_URL=http://localhost:${MINT_PORT} python scripts/wip/myvla_smoke.py
"""
import os, json, time, httpx

BASE_URL = os.environ["MINT_BASE_URL"].rstrip("/")
API_KEY  = os.environ.get("MINT_API_KEY", "dummy")
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
MODEL    = "myvla/myvla-7b"
CKPT     = "/path/to/myvla/checkpoint"   # ← 替换为实际路径

def post(path, body):
    r = httpx.post(f"{BASE_URL}{path}", headers=HEADERS, json=body, timeout=180)
    r.raise_for_status()
    return r.json()

def poll(request_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.post(f"{BASE_URL}/api/v1/retrieve_future",
                       headers=HEADERS,
                       json={"request_id": request_id},
                       timeout=10)
        if r.status_code == 200:
            return r.json()
        if r.status_code != 408:
            raise RuntimeError(f"retrieve_future error: {r.status_code} {r.text}")
        time.sleep(1)
    raise TimeoutError(f"request {request_id} timed out")

# ── 1. health check ───────────────────────────────────────────────────────────
print("[1/4] health check ...", end=" ")
r = httpx.get(f"{BASE_URL}/api/v1/healthz", headers=HEADERS, timeout=10)
r.raise_for_status()
print("OK")

# ── 2. create session ─────────────────────────────────────────────────────────
print("[2/4] create session ...", end=" ")
resp = post("/api/v1/create_session", {
    "base_model": MODEL,
    "checkpoint_path": CKPT,
})
sid = resp["action_session_id"]
print(f"OK  session_id={sid}")

# ── 3. act ────────────────────────────────────────────────────────────────────
print("[3/4] act ...", end=" ")
fake_obs = {
    "image": [[[[0]*3]*64]*64],   # dummy 64x64 RGB image
    "language": "pick up the block",
    "state": [0.0]*7,
}
resp = post("/api/v1/asample", {
    "action_session_id": sid,
    "observation": fake_obs,
})
result = poll(resp["request_id"])
print(f"OK  action={result['action'][:3]}...")

# ── 4. shutdown session ───────────────────────────────────────────────────────
print("[4/4] shutdown session ...", end=" ")
post("/api/v1/shutdown_session", {"action_session_id": sid})
print("OK")

print("\n✅ MyVLA smoke test PASSED")
```

**运行方式：**

```bash
# 确保 SSH tunnel 已建立
ssh -f -N -L ${MINT_PORT}:localhost:${MINT_PORT} mint-dev

MINT_BASE_URL=http://localhost:${MINT_PORT} \
MINT_API_KEY=dummy \
python scripts/wip/myvla_smoke.py
```

**逐步验证清单（先手动，再跑脚本）：**

```bash
# a) Ray 集群里看到 MyVLA actor
ssh mint-dev 'python -c "import ray; ray.init(address=\"auto\"); \
  print([a for a in ray.util.list_named_actors() if \"myvla\" in a.lower()])"'

# b) subprocess worker 进程确实在跑（在 GPU 节点上）
ssh <gpu_node> 'ps aux | grep myVLA_worker'

# c) GPU 显存被占用
ssh <gpu_node> 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

---

## 5. 关键文件速查表

### 5.1 OpenPI 参考实现（你需要模仿的文件）

| 文件 | 作用 | 你的对应文件 |
|------|------|-------------|
| `openpi_pi05_worker.py` | subprocess worker，实际跑 JAX 模型 | `myVLA_worker.py` |
| `openpi_action_ray_runtime.py` | Ray actor + client，封装 subprocess | `myVLA_ray_runtime.py` |
| `openpi_pi05_action_worker.py` | Session Manager，路由+生命周期 | `myVLA_session_manager.py` |
| `openpi_ray_runtime.py` | subprocess 通信基类（可复用） | 直接继承/复用 |
| `openpi_session_state.py` | 训练状态持久化（可选，有状态训练需要） | 按需复用或新写 |

### 5.2 需要改动的现有文件

| 文件 | 改动内容 | 改动量 |
|------|----------|--------|
| `action_session_manager.py` | import + 路由分支 + `_is_myvla_model()` | ~20 行 |
| `model_registry.py` | 新增模型配置条目 | ~5 行（或用环境变量绕过）|
| `step3_start.sh` | export placement 环境变量 | ~3 行 |

### 5.3 调度与 Actor 管理（只读参考，一般不需改动）

| 文件 | 作用 |
|------|------|
| `scheduling/model_work_scheduler.py` | Ray detached actor，管理工作队列和租约 |
| `actors/model_actor_supervisor.py` | 全局 actor 注册表，`/internal/actors` 端点的数据来源 |
| `actors/model_actor_launchers.py` | actor 启动函数注册表（如需自定义启动逻辑可在此注册）|
| `actors/model_actor_inventory.py` | 按 domain/replica 索引的 actor 状态追踪 |

---

## 6. 常见坑与注意事项

### 6.1 Worker 子进程启动失败

**现象：** `create_session` 卡住或超时（默认 120s）

**排查步骤：**
```bash
# 直接手动跑 worker，看 stderr
ssh <gpu_node> 'python -m mint_server.backend.openpi.myVLA_worker'
# 正常应输出: {"status": "ready"}
# 然后 ctrl+c 退出

# 如果没有 ready 输出，说明模型 import 或 checkpoint 加载失败
# 检查 PYTHONPATH 是否包含你的算法包
```

**常见原因：**
- PYTHONPATH 没有设置，找不到你的 `myvla` 包
- checkpoint 路径不存在或权限不对
- CUDA 版本不匹配（JAX/PyTorch 与驱动不兼容）
- Worker 没有输出 `{"status": "ready"}` 作为第一行

---

### 6.2 Ray Actor 找不到可用 GPU

**现象：** `create_session` 报 `No resources available` 或长时间 pending

**排查：**
```python
# 在 Ray 头节点或 driver 上运行
import ray
ray.init(address="auto")

# 查看 GPU 资源
print(ray.cluster_resources())   # 应有 "GPU": N
print(ray.available_resources()) # 应有剩余 GPU

# 查看有没有僵尸 actor 占着 GPU
actors = ray.util.list_named_actors(all_namespaces=True)
print([a for a in actors if "myvla" in a.lower()])
```

**如果有僵尸 actor：**
```python
# kill 掉（替换 actor_name）
actor = ray.get_actor("mint_myvla_<session_id>", namespace="mint")
ray.kill(actor)
```

---

### 6.3 服务重启后 Session 恢复

Actor 使用 `lifetime="detached"`，服务重启后 actor 仍然存在于 Ray 集群。
`_try_recover_client()` 会通过 `ray.get_actor(actor_name, namespace="mint")` 恢复。

**注意**：如果你的模型有 optimizer state，重启服务不会丢失（Ray actor 还在），
但如果整个 Ray 集群重启了，就需要从磁盘 checkpoint 重建。
建议在 `handle_train_step` 后定期调用 `handle_save_state` 写磁盘。

---

### 6.4 代码改完一定要重启服务

Python 服务不会热重载。改了任何 `.py` 文件后，必须：

```bash
# 使用 mint-dev skill 的标准重启流程（参考 mint_server/backend/openpi/ 的 skill 文档）
# 1. kill 所有 detached actor（否则旧 epoch 会报错）
# 2. 停止 API 服务进程
# 3. 重启

# 验证新代码已生效
ssh mint-dev 'grep "my_new_function" /path/to/mint_server/backend/openpi/myVLA_worker.py'
```

---

### 6.5 多 Session 并发时的 GPU 内存

每个 Per-Session actor 占用 1 GPU（`num_gpus=1`）。
如果集群只有 4 张 GPU，同时跑 5 个 session 会有 1 个排队等待。

**确认当前 GPU 占用：**
```bash
# 查看所有 actor 和它们占用的资源
ssh mint-dev 'curl -s http://localhost:${MINT_PORT}/internal/actors | python -m json.tool'
```

---

### 6.6 subprocess 通信 Deadlock

**现象：** `act` 调用永久挂住，没有 timeout

**原因：** subprocess 写了太多 stderr/stdout，pipe buffer 满了导致死锁

**解决：** subprocess 启动时把 stderr 重定向到文件（不要 `stderr=PIPE`）
```python
# myVLA_ray_runtime.py 中
proc = await asyncio.create_subprocess_exec(
    python, "-m", module,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=None,    # ← 继承父进程 stderr（会出现在 Ray actor 日志里），不用 PIPE
    env=merged_env,
)
```

---

## 快速参考：新算法集成 Checklist

```
□ myVLA_worker.py    — subprocess worker，能手动跑出 {"status": "ready"}
□ myVLA_ray_runtime.py — Ray actor + client，create_session/act/train_step/shutdown
□ myVLA_session_manager.py — Session Manager，含 detached actor 恢复逻辑
□ action_session_manager.py — 添加 import、_is_myvla_model()、路由分支
□ model_registry.py — 添加模型配置（或用 MINT_MODEL_CONFIG_OVERRIDES_JSON）
□ step3_start.sh — export MINT_MYVLA_MODEL_PLACEMENT_JSON
□ scripts/wip/myvla_smoke.py — 冒烟测试，从 health → create → act → shutdown 全跑通
□ 验证：ssh <gpu_node> 'nvidia-smi' 看到显存被占用
□ 验证：curl /internal/actors 看到 myvla actor 在线
```

