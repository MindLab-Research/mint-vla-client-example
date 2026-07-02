# OpenPI π0.5 训练数据流全景 (client → server → Ray actor → 返回)

本文档以 `scripts/wip/openpi_vla_smoke.py`（最小冒烟）与
`scripts/wip/openpi_libero_sft.py`（真实 LeRobot 微调）为例，逐层讲清楚
**客户端怎么发、服务端怎么组合分配、GPU actor 怎么把数据拿去训练、结果怎么回来**。
所有关键环节都附代码位置（文件:行号）与片段。

> 术语：本文的 "PI / π0 / π0.5" 指 Physical Intelligence 的 openpi 模型。
> 模型本体是 **JAX + Flax(nnx) + Orbax**；被组里的胶水层 `mint_server/backend/openpi/`
> 包成 mint-server 的训练/推理后端。

---

## 0. 三进程拓扑与两道边界

```
本地脚本 (numpy/pandas, 无 JAX)         mint-server (FastAPI, 控制面)          Ray GPU actor (JAX)
─────────────────────────────         ────────────────────────────         ────────────────────
openpi_libero_sft.py / smoke.py        routes/ → dispatch → engine          OpenPIPi05WorkerSession
  读 LeRobot parquet          ──HTTP──▶  路由/校验/计费/入队/降格   ──Ray──▶   flax/jax compute_loss
  openpi.transforms 预处理               引擎路由 + payload 构造                梯度累积 / optim
        ▲                                        │                                   │
        └──────────── retrieve_future 轮询 ◀──────┴─────── task_futures 回填 ◀─────────┘
```

- **边界 1 = HTTP/JSON**：client 与 server 之间。图像走 base64 PNG，张量走 `{data, shape, dtype}`。
- **边界 2 = Ray**：server 主进程与 GPU actor 之间。`actor.request.remote(op, payload)`。
- **关键点**：JAX/openpi 这些重依赖**只在 actor 进程内 import**，server 主进程保持干净，
  因此能与 PyTorch 系（Qwen36/verl）模型在同一 server 共存。

---

## 1. Server 的分层组成（5 层流水线）

一步 `train_step` 从 HTTP 进来到 GPU 上算梯度，穿过 5 层，每层只做一件事：

| 层 | 文件 | 职责 |
|----|------|------|
| ① 路由层 | `routes/mint.py`, `routes/training.py` | 收 HTTP、校验、计费、入队、返回 future |
| ② 队列/派发层 | `backend/scheduling/model_work_dispatch.py` | 后台异步取出、按 `op` 分发、请求降格 |
| ③ 引擎路由层 | `backend/training/training_engine_router.py` | 按模型 `training_backend` 选后端引擎 |
| ④ 后端引擎层 | `backend/openpi/openpi_pi05_training.py` | datum → runtime payload，编排 fb+optim |
| ⑤ Ray 运行时/Actor | `backend/openpi/openpi_ray_runtime.py` + `openpi_pi05_worker.py` | GPU 上跑 JAX |

---

## 2. 客户端：怎么发起一次完整训练

### 2.1 冒烟版 (`openpi_vla_smoke.py`)：用假数据把链路跑通

冒烟脚本不关心 loss 好坏，只验证「链路是不是通的」。它用一张 1×1 的 PNG 占位：

```python
# scripts/wip/openpi_vla_smoke.py:13
PNG_1X1_BASE64 = "iVBORw0KGgoAAAANSUhEUgAA...ORK5CYII="

def _image_chunk() -> dict[str, Any]:                              # :62
    return {"type": "image", "data": PNG_1X1_BASE64, "format": "png", "expected_tokens": 256}

def _observation(prompt_tokens, *, state_dim):                     # :66
    return {
        "state": {"data": [0.0] * state_dim, "shape": [state_dim], "dtype": "float32"},
        "model_input": {
            "chunks": [_image_chunk(), _image_chunk(), _image_chunk(),   # 3 张图 = camera_layout
                       {"type": "encoded_text", "tokens": prompt_tokens}],
        },
    }
```

两种模型的**监督信号不同**（这决定了 loss_fn）：

```python
def _fast_datum():                                                 # :75  pi0-fast → 离散动作 token
    return {"observation": _observation([11,12,13], state_dim=8),
            "supervision": {"target_tokens": {"data":[21,22],"shape":[2],"dtype":"int64"},
                            "weights": {...}, "token_ar_mask": {...}}}

def _pi05_datum():                                                 # :86  pi05 → 连续动作
    return {"observation": _observation([11,12,13], state_dim=8),
            "supervision": {"actions": {"data":[0.0]*(10*7), "shape":[10,7], "dtype":"float32"}}}
```

客户端主流程（`main`, :117）串起整条链路，每一步都用 `_await_result` 轮询 future：

```python
model_id, create_result = _create_model(base_url, headers, base_model=args.model)   # :132 建模型(拉起 JAX actor)
datum = _fast_datum() if "pi0-fast" in args.model else _pi05_datum()                # :133 选数据
train_result = _await_result(base_url, headers, _post_json(                         # :134 训练一步
    base_url, "/api/v1/mint/vla/train_step", headers,
    {"model_id": model_id, "loss_fn": "cross_entropy" if "pi0-fast" in args.model
                                       else "flow_matching", "data": [datum]}))
# 推理段(可 --skip-action 跳过)：导权重 → 建 action session → act 出动作
save_result   = _await_result(..., "/api/v1/save_weights_for_sampler", {...})        # :138
action_created= _post_json(..., "/api/v1/mint/action_sessions", {...})              # :142
action_result = _await_result(..., f".../action_sessions/{id}/act", {...})          # :144
```

`finally` 里**无条件清理**，避免残留 actor 占 GPU：

```python
finally:                                                            # :162
    if action_session_id: requests.delete(f".../action_sessions/{action_session_id}", ...)
    if model_id:          _delete_model(base_url, headers, model_id)
```

**future 轮询模式**（贯穿所有 mint 异步 API）：

```python
def _poll_future(base_url, headers, request_id, *, timeout_s=1800.0):   # :40
    while time.time() < deadline:
        resp = requests.post(f"{base_url}/api/v1/retrieve_future",
                             json={"request_id": request_id}, ...)
        if resp.status_code == 408:                                # 408 = 还没算完
            time.sleep(1.0); continue
        resp.raise_for_status()
        return resp.json()                                         # 200 = 拿到结果
```

### 2.2 真实微调版 (`openpi_libero_sft.py`)：LeRobot 数据 + openpi transform

真实训练与冒烟唯一的区别在**数据从哪来、怎么预处理**——发送协议完全一致。

```python
# 数据源是 LeRobot dataset 格式(parquet + meta jsonl)
DATASET_ROOT = Path('.../.hf-lerobot/physical-intelligence/libero')      # :63

# 用 openpi 官方 transform 流水线做预处理(与官方训练完全一致)
transform = T.compose([
    *data_cfg.repack_transforms.inputs,
    *data_cfg.data_transforms.inputs,
    T.Normalize(data_cfg.norm_stats, ...),   # 用 assets 里的 norm_stats 归一化
    *data_cfg.model_transforms.inputs,        # tokenize prompt 等
])

# 训练循环：每步组 batch → POST → 轮询 → 记 loss
for step in range(1, args.steps + 1):                                    # :340
    batch = [datum_builder(args.base_model, rng.choice(items)) for _ in range(args.batch_size)]
    resp = requests.post(f'{base_url}/api/v1/mint/vla/train_step',
                         json={'model_id': model_id, 'loss_fn': loss_fn, 'data': batch})
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=1800)
    loss = float(result['metrics']['loss:mean'])
```

**要点**：客户端全程是 numpy/pandas，**没有 JAX**；LeRobot 只是数据格式，
openpi transform 把原始帧翻译成 `{image, state, prompt tokens, actions}`，序列化成
JSON 发出去。server 完全不读 LeRobot 文件。

---

## 3. ① 路由层：接住请求，但不立刻算

`vla_train_step` 收到 POST 后做校验+计费，然后**入队并立刻返回 future**，不同步执行训练：

```python
# routes/mint.py:594
@router.post("/vla/train_step", response_model=UntypedAPIFuture)
async def vla_train_step(request: VLATrainStepRequest, http_request: Request):
    session = ... get_session(request.model_id) ...
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    _, max_seq_len = _vla_token_stats(request.data)          # :615 算序列长度
    max_model_len = training_routes._get_max_model_len(base_model)
    if max_seq_len > max_model_len:                          # :622 超限直接 400
        raise HTTPException(status_code=400, detail="Input sequence length ... exceeds max_model_len")

    request_id = uuid.uuid4().hex                            # :633
    # ... 计费 billing_input, 入队(op="mint.vla.train_step") ...
    return UntypedAPIFuture(request_id=request_id)           # 立刻返回, client 开始轮询
```

序列长度闸门 `_vla_token_stats`：只把 `encoded_text` 与 `target_tokens` 计入长度
（图像的 `expected_tokens` 用于计费，不计入这里的序列长度）：

```python
# routes/mint.py:395
def _vla_token_stats(data):
    for item in data:
        for chunk in item.observation.model_input.chunks:
            if chunk.type == "encoded_text":
                seq_len += len(chunk.tokens)
        target_tokens = item.supervision.get("target_tokens")
        if target_tokens is not None:
            seq_len += int(target_tokens.shape[0])
```

**设计要点**：请求/计算解耦 —— 路由层只入队+返 future，真正计算在后台 worker 异步跑，
所以支持并发与长任务（1800s 超时）。

---

## 4. ② 队列/派发层：异步取出 + 请求降格 (lowering)

后台 worker 从队列取出，按 `op` 名分发到对应 handler：

```python
# backend/scheduling/model_work_dispatch.py:478
if op == "mint.vla.train_step":
    async def _run():
        req = VLATrainStepRequest.model_validate_json(item.request_json)
        await mint._do_vla_train_step(item.request_id, req, item.user_id, ...)
```

`_do_vla_train_step` 做**关键翻译**：把 VLA 专用请求**降格成通用 `TrainStepRequest`**，
之后就复用文本训练的整套基础设施：

```python
# routes/mint.py:1101
async def _do_vla_train_step(request_id, request, user_id, ...):
    internal_request = _lower_vla_train_step_request(request)      # VLA → 通用
    await training_routes._do_train_step(request_id, internal_request, user_id, ...)
```

降格逻辑：`observation.model_input` → 通用 `model_input`；`supervision`(actions/target_tokens)
→ `loss_fn_inputs`；`observation.state` 也进 `loss_fn_inputs`：

```python
# routes/mint.py:417
def _lower_vla_datum(item: VLADatum) -> Datum:
    if "state" in item.supervision:
        raise ValueError("VLADatum.supervision must not contain 'state'; use observation.state")
    return Datum(
        model_input=item.observation.model_input,
        loss_fn_inputs={"state": item.observation.state, **item.supervision},
    )

# routes/mint.py:431
def _lower_vla_train_step_request(request) -> TrainStepRequest:
    return TrainStepRequest(
        model_id=request.model_id, seq_id=request.seq_id, adam_params=request.adam_params,
        forward_backward_input=ForwardBackwardInput(
            data=[_lower_vla_datum(item) for item in request.data],
            loss_fn=request.loss_fn, loss_fn_config=request.loss_fn_config))
```

**设计要点**：请求降格让 VLA 免费复用整套训练/调度/计费/future 基础设施，无需重写。

---

## 5. ③ 引擎路由层：按模型类型分叉到不同框架

通用 `_do_train_step` 调 `engine.train_step(session, request)`，这个 engine 是
`TrainingEngineRouter`。它看 `session.base_model` 的 `training_backend` 决定派给谁：

```python
# routes/training.py:3251
async def _do_train_step(request_id, request, user_id, ...):
    engine = _current_training_engine()
    manager = _current_training_manager()
    session = manager.get_session(request.model_id) or await _restore_training_session(request.model_id)
    session = await _materialize_training_session_for_stateful_use(session)
    result = await run_async_with_otel_span(
        "training.train_step.execute",
        lambda: engine.train_step(session, request),               # :3283 打到引擎
        ...)
    await task_futures.async_resolve(request_id, result, ...)      # :3301 回填 future
```

```python
# backend/training/training_engine_router.py:46
def _engine_for_base_model(self, base_model):
    training_backend = get_model_config(base_model).training_backend
    if training_backend == OPENPI_FAST_TRAINING_BACKEND:   return self._openpi_fast_engine
    if training_backend == OPENPI_PI05_TRAINING_BACKEND:   return self._openpi_pi05_engine
    return self._text_engine                                # verl(Qwen36 等文本模型)

async def train_step(self, session, request):              # :75
    return await self._engine_for_session(session).train_step(session, request)
```

**这就是 JAX 与 PyTorch 能共存的分叉点**：同一个 `train_step` 入口，按模型 `training_backend`
路由到 openpi(JAX) 或 verl(PyTorch) 引擎。你之前 `RLcheck` 崩的 Qwen36 走的是 `_text_engine`(verl)
那一支，与 openpi 完全独立。

---

## 6. ④ 后端引擎层：datum → runtime payload，编排 fb + optim

`OpenPIPi05TrainingEngine.train_step` 是 **forward_backward + optim_step 两步合一**：

```python
# backend/openpi/openpi_pi05_training.py:454
async def train_step(self, session, request):
    fb_result = await self.forward_backward(session, request)          # 前向+反向(累积梯度)
    optim_request = ... adam_params ...
    optim_result = await self.optim_step(session, optim_request)       # 优化器更新
    metrics = {**fb_result["metrics"], **optim_result["metrics"]}
    fb_result["metrics"] = metrics
    return fb_result
```

`forward_backward` 里对 batch 每个 datum 调 `payload_builder`，把通用请求**翻译成 openpi payload**，
再通过 `_request_runtime` 打到 actor：

```python
# backend/openpi/openpi_pi05_training.py:411
model_config = self._model_config(session.base_model)
runtime = self._runtime_for_session(session)
result = await self._request_runtime(runtime, "forward_backward", {
    "loss_fn": loss_fn,
    "loss_fn_config": dict(request.forward_backward_input.loss_fn_config or {}),
    "batch": [payload_builder(datum=datum, model_config=model_config)
              for datum in request.forward_backward_input.data],
})
session.accumulated_gradients += 1                                     # :424 只累积
```

### 6.1 模型输入是怎么"确定"的 —— `ModelConfig` 说了算

`payload_builder`（`build_openpi_pi05_sft_runtime_payload`）拿你的 chunks 去和
**模型固有规格 `ModelConfig` 对表**。这才是"模型输入怎么定"的权威来源：

```python
# backend/openpi/openpi_pi05_training.py:101
image_chunks = [c for c in model_input.chunks if c.type == "image"]
text_chunks  = [c for c in model_input.chunks if c.type == "encoded_text"]
if len(text_chunks) != 1:                                     # :110 prompt 必须恰好一段
    raise ValueError("OpenPI pi0.5 expects exactly one encoded_text prompt chunk")

camera_layout = tuple(model_config.camera_layout)             # :113
if len(image_chunks) != len(camera_layout):                   # 图像张数必须 == 相机数
    raise ValueError(f"expects {len(camera_layout)} image chunks, got {len(image_chunks)}")

action_dim = int(model_config.action_dim or 0)                # :119
state = _pad([float(v) for v in state_data], action_dim, key="state")   # :126 state pad 到 action_dim

image_bytes = {                                               # :128 图像→相机名: 按位置 zip
    name: {"data": base64.b64encode(chunk.data).decode(), "format": chunk.format}
    for name, chunk in zip(camera_layout, image_chunks, strict=True)
}
return {                                                      # :136 标准化 payload
    "image_bytes": image_bytes,
    "image_mask": {name: True for name in camera_layout},
    "state": state,
    "tokenized_prompt": [int(t) for t in text_chunks[0].tokens],
    "tokenized_prompt_mask": [True] * len(prompt_tokens),
}
```

**结论**：脚本发几张图、state 多少维不是脚本随便定的 —— `camera_layout` 决定图像张数与
相机顺序（图像按**位置 zip** 到相机名，发反了就串），`action_dim` 决定 state 维度。
权威定义在 `backend/core/model_registry.py` 里该 base_model 的配置。

### 6.2 optim_step

```python
# backend/openpi/openpi_pi05_training.py:439
async def optim_step(self, session, request):
    runtime = self._runtime_for_session(session)
    result = await self._request_runtime(runtime, "optim_step", self._optim_payload(request.adam_params))
    session.current_step += 1                                 # 真正推进训练步
    session.accumulated_gradients = 0
    result["metrics"]["step"] = session.current_step
    return result
```

**梯度累积语义**：`forward_backward` 只累积梯度，`optim_step` 才应用更新并 `current_step += 1`。
`train_step` 把两者打包，所以 client 一次调用 = 一个完整训练步。

---

## 6.5 Actor 生命周期：create_model 时 GPU actor 怎么被拉起来

前面第 7 节讲的是「actor 已存在时 request 怎么打进去」。这里补上更早的一步：
**第一次 `POST /api/v1/create_model` 时，那个 `@ray.remote` 的 JAX actor 是怎么被创建、
放到哪张 GPU、多个模型如何共享的。**

### 6.5.1 触发链：create_model → create_training_session

client 建模型时，路由层最终调到引擎的 `create_training_session`：

```
POST /api/v1/create_model
  → routes/training.py (create_model 路由)
  → engine_router.create_training_session(session)          # 按 base_model 选 openpi_pi05 引擎
  → OpenPIPi05TrainingEngine.create_training_session(session)
```

```python
# backend/openpi/openpi_pi05_training.py:363
async def create_training_session(self, session):
    model_config = self._model_config(session.base_model)
    config_name = get_openpi_pi05_config_name(session.base_model)
    client = await self._runtime_factory(                       # ① 拿到/创建 Ray actor 的客户端
        session=session, model_config=model_config, config_name=config_name)
    try:
        await self._request_runtime(client, "create_session",   # ② 在 actor 内加载模型
            self._create_session_payload(session=session, model_config=model_config))
    except Exception:
        if callable(getattr(client, "close", None)):
            await client.close()
        raise
    self._runtime_clients[session.model_id] = client            # ③ 缓存 client, 供后续 train_step 复用
    session.backend = OPENPI_PI05_TRAINING_BACKEND
    session.is_active = True
```

注意这里的**两阶段**：`_runtime_factory` 只负责「弄到一个能用的 GPU actor 句柄」，
真正把 π0.5 权重加载进显存是第 ② 步 `create_session` op（对应 worker 的
`OpenPIPi05WorkerSession.__init__`，见 7.3）。`create_session` 的 payload 正是
把模型规格从 `ModelConfig` 传进 actor 的地方：

```python
# backend/openpi/openpi_pi05_training.py:334
def _create_session_payload(self, *, session, model_config):
    return {
        "model_id": session.model_id, "session_id": session.session_id,
        "base_model": session.base_model,
        "config_name": get_openpi_pi05_config_name(session.base_model),
        "learning_rate": float(session.learning_rate),
        "action_dim": int(model_config.action_dim or 0),        # ← state 维度来源
        "action_horizon": int(model_config.action_horizon or 0),
        "max_token_len": int(model_config.max_model_len),
        "camera_layout": list(model_config.camera_layout),      # ← 图像张数/相机顺序来源
    }
```

### 6.5.2 共享 actor 池：多个模型可复用同一张 GPU actor

`_default_runtime_factory` 用的是 **shared runtime** —— 关键设计：actor 按
`pool_key` 复用，而不是一个模型一个 actor：

```python
# backend/openpi/openpi_pi05_training.py:294
async def _default_runtime_factory(*, session, model_config, config_name):
    spec = dataclasses.replace(OpenPIFastRuntimeSpec.from_env(),
                               worker_module=OPENPI_PI05_WORKER_MODULE)
    return await start_openpi_shared_ray_runtime(
        session=session, spec=spec, config_name=config_name, model_config=model_config)
```

```python
# backend/openpi/openpi_shared_ray_runtime.py:690
async def start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config, ...):
    ensure_openpi_ray_initialized()
    pool_key = _normalize_pool_key(spec=spec, session=session,             # 相同规格 → 相同 key
                                   config_name=config_name, model_config=model_config)
    actor_name = _shared_actor_name(pool_key)

    with _SHARED_POOL_LOCK:                                                 # 进程内锁, 防并发重复创建
        entry = _SHARED_ACTORS.get(actor_name)
        actor = entry.actor if entry is not None else None
        named_actor = _get_named_shared_actor(actor_name)                   # 复用 Ray 里已存在的 detached actor
        if named_actor is not None:
            actor = named_actor
        if actor is None:                                                   # 没有才新建
            actor = OpenPISharedRayRuntimeActor.options(
                name=actor_name, namespace=RAY_NAMESPACE,
                lifetime="detached",                                        # detached: 活过 server 重启
                runtime_env={"env_vars": runtime_env_vars},                 # JAX/openpi 环境只在这注入
                **_single_node_actor_options(base_model=..., actor_name=actor_name),   # 单节点/GPU 放置
            ).remote(actor_name=actor_name, pool_key=pool_key, spec=spec, ...)
```

三个要点：

- **detached lifetime**：actor 活过 server 进程重启（server 挂了重连即可，不用重载模型）。
- **named actor 复用**：先查 Ray 里有没有同名 actor，有就直接用 —— 这就是文档开头说的
  「首次 ~80s，后续 ~2s」的原因。
- **runtime_env 注入**：JAX/openpi 的环境变量（含 `LD_LIBRARY_PATH` 等）**只在这个 actor 的
  runtime_env 里设**，server 主进程不受影响。这也是框架隔离的落点。

### 6.5.3 GPU 放置：actor 落在哪张卡

`_single_node_actor_options` 通过 `parse_model_gpu_placement` 解析该模型的 GPU 放置策略，
π0.5 是单 GPU actor，要求恰好 1 张卡、1 个 slice：

```python
# backend/openpi/openpi_shared_ray_runtime.py:139
placement = parse_model_gpu_placement(...)
if len(placement.slices) != 1:
    raise ...("expected exactly 1 placement slice for single-GPU actor")
if placement.total_gpus != 1:
    raise ...("expected exactly 1 GPU")
# → required_gpus_by_node_ip={placement.slices[0].node_ip: 1}   固定到某节点某卡
```

这与 `@ray.remote(num_gpus=1, max_concurrency=1)` 配合 —— Ray 保证该 actor 独占 1 张 GPU、
且串行处理请求（不会并发跑两个 forward_backward 撞显存）。

### 6.5.4 ready + 注册到 actor 清单

actor 起来后 `client.ready()` 等它就绪（触发 `_ensure_runtime` 真正 import JAX、编译），
然后注册进全局 actor 清单，`/internal/actors` 就能看到它：

```python
# backend/openpi/openpi_shared_ray_runtime.py:788, 804
metadata = await client.ready()                                  # 等 actor 就绪, 拿 actor_id/node_ip/pid
await _pool_call("register", actor_name=actor_name, actor_type=ActorType.OPENPI,
                 num_gpus=1, actor_handle=entry.actor, base_model=..., node_id=..., ...)
```

冒烟脚本里 `actors_before` / `actors_after` 两张快照对比的就是这份清单，用来确认这次
测试起了哪些 actor、finally 有没有清干净。

### 6.5.5 完整生命周期时序

```
create_model                                                          delete_model
     │                                                                     │
     ▼                                                                     ▼
create_training_session                                              shutdown_session
     │                                                                     │
     ├─ _runtime_factory → start_openpi_shared_ray_runtime                 │
     │      ├─ pool_key 命中已有 detached actor? ──是──▶ 复用(~2s)          │
     │      └─ 否 ─▶ OpenPISharedRayRuntimeActor.options(detached,          │
     │                num_gpus=1).remote()  拉起新 actor(~80s)             │
     │                └─ 放置到 placement 指定的节点/GPU                     │
     ├─ client.ready()  等 actor import JAX + 就绪                          │
     ├─ "create_session" op ─▶ worker.__init__: 加载权重, 冻结非 LoRA        │
     └─ register → /internal/actors 可见                                   │
     │                                                                     │
     ├───────── 多次 train_step / act 复用同一 actor ────────────────────┤
     │                                                                     │
                                                          "shutdown" op ─▶ 释放 GPU
                                              (detached 可选保留供其他 model 复用)
```

---

## 7. ⑤ Ray 运行时 + Actor：GPU 上跑 JAX

### 7.1 Ray 边界：`_request_runtime` → `actor.request.remote()`

引擎侧的 `_request_runtime` 最终落到 Ray 远程调用：

```python
# backend/openpi/openpi_ray_runtime.py:232
async def request(self, op, payload=None, *, timeout_s=None):
    if self._closed:
        raise OpenPIFastWorkerProtocolError("OpenPI Ray runtime client is closed")
    result = await self._ray_get(
        self._actor.request.remote(op, payload or {}, timeout_s=timeout_s),   # 真正的 Ray 调用
        timeout_s=timeout_s)
    return result
```

`self._actor` 是一个独占 1 张 GPU、串行执行的 detached actor：

```python
# backend/openpi/openpi_ray_runtime.py:119
@ray.remote(num_gpus=1, max_concurrency=1)
class OpenPIRayRuntimeActor:
    async def _ensure_runtime(self):                          # :133 懒初始化
        if self._runtime is None:
            self._runtime = await OpenPIDirectWorkerClient.start(self._spec)
        return self._runtime

    async def request(self, op, payload=None, *, timeout_s=None):   # :157
        runtime = await self._ensure_runtime()
        return await runtime.request(op, payload or {}, timeout_s=timeout_s)
```

错误处理：Ray 超时转成 `OpenPIFastWorkerProtocolError`，`RayTaskError` 解包还原成
原始 worker 异常（`_ray_get`, :206），这样 client 看到的是可读的业务错误而非 Ray 内部栈。

### 7.2 Worker 侧 op 分发

actor 内部 `_dispatch` 把 op 字符串映射到 session 方法：

```python
# backend/openpi/openpi_pi05_worker.py:1140
def _dispatch(session, op, payload):
    if session is None:
        raise RuntimeError("OpenPI pi0.5 worker session is not initialized")
    if op == "forward_backward":      return session.forward_backward(payload), False
    if op == "optim_step":            return session.optim_step(payload), False
    if op == "save_weights":          return session.save_weights(payload), False
    if op == "save_sampler_weights":  return session.save_sampler_weights(payload), False
    if op == "load_weights":          return session.load_weights(payload), False
    if op == "shutdown":              return session.shutdown(), True
    raise ValueError(f"Unknown OpenPI pi0.5 worker op: {op!r}")
```

### 7.3 Actor 初始化：加载模型 + 冻结非 LoRA

首次 `create_model` 时构造 `OpenPIPi05WorkerSession`，这里才 import JAX 并加载预训练权重：

```python
# backend/openpi/openpi_pi05_worker.py:160
class OpenPIPi05WorkerSession:
    def __init__(self, payload):
        import flax.nnx as nnx; import jax; import jax.numpy as jnp; import optax
        import openpi.models.pi0_config as pi0_config
        import orbax.checkpoint as ocp
        # orbax 版本兼容补丁(openpi pin 0.11.13, runtime 0.11.40)
        from mint_server.backend.openpi.openpi_orbax_compat import install_restore_params_compat
        install_restore_params_compat(openpi_model)                # :206

        model_cfg = pi0_config.Pi0Config(                          # :226
            pi05=True, action_dim=self._action_dim, action_horizon=self._action_horizon,
            max_token_len=self._max_token_len, paligemma_variant="gemma_2b_lora")
        freeze_filter = nnx.Not(nnx_utils.PathRegex(".*lora.*"))   # :234 冻结除 LoRA 外全部
```

### 7.4 数据 → JAX 张量：`_observation_from_payload`

worker 把 payload 解码成 openpi 的 `Observation` 对象（真正喂给模型的输入）：

```python
# backend/openpi/openpi_pi05_worker.py:141
def _decode_image(encoded):
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)    # base64 PNG → [H,W,3] uint8

# :354
def _observation_from_payload(self, item):
    images = {key: jnp.asarray(np.expand_dims(_decode_image(value), axis=0), dtype=jnp.uint8)
              for key, value in dict(item["image_bytes"]).items()}       # 加 batch 维
    observation = self._openpi_model.Observation.from_dict({           # :365 组装 openpi Observation
        "image": images,
        "image_mask": image_mask,
        "state": jnp.asarray([item["state"]], dtype=jnp.float32),
        "tokenized_prompt": jnp.asarray([item["tokenized_prompt"]], dtype=jnp.int32),
        "tokenized_prompt_mask": jnp.asarray([item["tokenized_prompt_mask"]], dtype=jnp.bool_)})
    actions = jnp.asarray(item["actions"], dtype=jnp.float32)[None, ...]
    return observation, actions
```

### 7.5 前向反向：只对 LoRA 参数求梯度

```python
# backend/openpi/openpi_pi05_worker.py:687
def forward_backward(self, payload):
    loss_fn = str(payload.get("loss_fn") or "")
    if loss_fn not in {"flow_matching", "importance_sampling", "ppo"}:
        raise ValueError(...)
    for item in list(payload["batch"]):
        if loss_fn == "flow_matching":                            # SFT 路径
            observation, actions = self._observation_from_payload(item)
            grads, loss_value, grad_norm, param_norm = self._compute_grads(observation, actions)
        else:                                                     # RL 路径(importance_sampling/ppo)
            observation, chains, old_logprobs, advantages = self._rl_observation_from_payload(item)
            ...
```

```python
# backend/openpi/openpi_pi05_worker.py:405
def _compute_grads(self, observation, actions):
    model = nnx.merge(self._state.model_def, self._state.params)
    model.train()
    self._rng, step_rng = jax.random.split(self._rng)
    def loss_fn(model_obj, rng, obs, act):
        chunked_loss = model_obj.compute_loss(rng, obs, act, train=True)   # flow-matching loss
        return jnp.mean(chunked_loss)
    diff_state = nnx.DiffState(0, self._config.trainable_filter)           # 只对可训练(LoRA)求导
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, step_rng, observation, actions)
    grad_norm, param_norm = self._grad_and_param_norm(model, grads)
    return grads, _float_scalar(jax.device_get(loss)), grad_norm, param_norm
```

---

## 8. 结果回传路径

梯度算完后，结果沿原路**逐层冒泡**回到 client：

```
worker.forward_backward 返回 {metrics, loss_fn_outputs}
  → actor.request (Ray) 返回 ref
  → OpenPIRayRuntimeClient._ray_get 解包
  → engine.forward_backward / train_step 合并 metrics
  → _do_train_step: task_futures.async_resolve(request_id, result)      # routes/training.py:3301
  → client 下一次 retrieve_future 轮询拿到 200 + result['metrics']['loss:mean']
```

`async_resolve` 把结果写进 future 存储，之前一直返回 408 的 `retrieve_future` 这时返回 200。

---

## 9. 完整时序图（一次 train_step）

```
client                 routes/mint.py         dispatch          engine_router      pi05_training       ray_runtime        worker(JAX/GPU)
  │                          │                    │                   │                  │                 │                    │
  │ POST /vla/train_step     │                    │                   │                  │                 │                    │
  ├─────────────────────────▶│ 校验+计费           │                   │                  │                 │                    │
  │                          │ 入队(op=vla.train)  │                   │                  │                 │                    │
  │◀── 200 {request_id} ─────┤ (立即返回 future)   │                   │                  │                 │                    │
  │                          │                    │                   │                  │                 │                    │
  │ POST /retrieve_future    │           后台 worker 取出               │                  │                 │                    │
  ├─────────────────────────▶│──────────────────▶│ _do_vla_train_step │                  │                 │                    │
  │◀── 408 (pending) ────────┤                    │ lower→通用请求      │                  │                 │                    │
  │  (每 1s 轮询)             │                    │ _do_train_step ───▶│ train_step ─────▶│ fb: build       │                    │
  │                          │                    │                   │                  │ payload(camera/  │                    │
  │                          │                    │                   │                  │ action_dim)      │                    │
  │                          │                    │                   │                  │ _request_runtime ├─ actor.request ──▶│ _observation_from_payload
  │                          │                    │                   │                  │                 │ (Ray, num_gpus=1)  │ _compute_grads(LoRA)
  │                          │                    │                   │                  │                 │◀──── metrics ──────┤ optim_step
  │                          │                    │                   │◀──── result ─────┤◀── result ──────┤                    │
  │                          │        async_resolve(request_id, result)                  │                 │                    │
  │ POST /retrieve_future    │                    │                   │                  │                 │                    │
  ├─────────────────────────▶│                    │                   │                  │                 │                    │
  │◀── 200 {metrics:{loss}} ─┤                    │                   │                  │                 │                    │
```

---

## 10. 推理 (act) 路径简述

训练完导出权重后，可起 action session 让模型**输出机器人动作**（冒烟脚本 :137-144）：

```
POST /save_weights_for_sampler   → engine.save_weights_for_sampler → worker "save_sampler_weights" 导出 checkpoint
POST /mint/action_sessions       → action_session_manager.create_session (起 action actor, 加载 checkpoint)
POST /mint/action_sessions/{id}/act → act 路由 (routes/mint.py:500)
```

`act` 路由把 observation 拆成 `model_input` + `state` 打包成 `ActRequest`：

```python
# routes/mint.py:510
queued_request = ActRequest(
    action_session_id=action_session_id, seq_id=request.seq_id,
    observation=request.observation.model_input,
    extra_inputs={"state": request.observation.state},
    temperature=request.temperature, ...)
```

action worker (`openpi_pi05_action_worker.py`) 用 flow-matching **去噪采样**生成动作：
从高斯噪声出发，多步 `jax.random` 迭代去噪，输出 `[action_horizon, action_dim]` 的连续动作。

---

## 11. 三个核心设计要点（总结）

1. **请求/计算解耦**：路由层只入队 + 返 future，后台异步计算。支持并发与长任务。
   → `vla_train_step` 返回 `request_id`，client `_poll_future` 轮询（408 等 / 200 拿）。

2. **请求降格 (lowering)**：VLA 请求在派发层被翻译成通用 `TrainStepRequest`，
   免费复用整套训练/调度/计费基础设施。→ `_lower_vla_train_step_request`。

3. **引擎路由 + actor 隔离**：`TrainingEngineRouter` 按 `training_backend` 分叉到
   JAX(openpi) / PyTorch(verl) 引擎；重依赖关在各自 `@ray.remote(num_gpus=1)` actor 的
   runtime_env 里，server 主进程干净。→ 这也是 JAX 与 PyTorch 共存、互不干扰的根本。

---

## 12. 关键代码位置索引

| 环节 | 位置 |
|------|------|
| client 构造 observation/datum | `scripts/wip/openpi_vla_smoke.py:62-92` |
| client future 轮询 | `scripts/wip/openpi_vla_smoke.py:40` |
| client 真实训练循环 | `scripts/wip/openpi_libero_sft.py:340` |
| 路由入口 + 校验 + 入队 | `mint_server/routes/mint.py:594` |
| 序列长度闸门 | `mint_server/routes/mint.py:395` |
| 请求降格 | `mint_server/routes/mint.py:417, 431, 1101` |
| 队列派发 | `mint_server/backend/scheduling/model_work_dispatch.py:478` |
| 通用 train_step + future 回填 | `mint_server/routes/training.py:3251` |
| 引擎路由 | `mint_server/backend/training/training_engine_router.py:46, 75` |
| **create_model → 建训练会话** | `mint_server/backend/openpi/openpi_pi05_training.py:363` |
| **create_session payload(模型规格)** | `mint_server/backend/openpi/openpi_pi05_training.py:334` |
| **共享 actor 池 / 拉起 actor** | `mint_server/backend/openpi/openpi_shared_ray_runtime.py:690, 722` |
| **GPU 放置解析** | `mint_server/backend/openpi/openpi_shared_ray_runtime.py:139` |
| **actor ready + 注册清单** | `mint_server/backend/openpi/openpi_shared_ray_runtime.py:788, 804` |
| fb+optim 编排 | `mint_server/backend/openpi/openpi_pi05_training.py:454` |
| payload 构造 (输入规格) | `mint_server/backend/openpi/openpi_pi05_training.py:101-142` |
| Ray 边界 | `mint_server/backend/openpi/openpi_ray_runtime.py:119, 232` |
| worker op 分发 | `mint_server/backend/openpi/openpi_pi05_worker.py:1140` |
| worker 模型加载/冻结 | `mint_server/backend/openpi/openpi_pi05_worker.py:160, 234` |
| 数据→JAX 张量 | `mint_server/backend/openpi/openpi_pi05_worker.py:141, 354` |
| 前向反向 (LoRA 求导) | `mint_server/backend/openpi/openpi_pi05_worker.py:405, 687` |
| 推理 act | `mint_server/routes/mint.py:500` + `openpi_pi05_action_worker.py` |

