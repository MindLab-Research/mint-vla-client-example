# API 字段契约参考

本文档摘录本 skill 依赖的 mint-server API 精确字段定义，全部来自源码，不是意译。
用于避免每次都要重新翻源码确认 wire format。

所有写操作（`create_model`、`save_weights_for_sampler`、`vla/train_step`）都是
**future-based**：POST 返回 `{"request_id": "...", ...}`，真正结果需要轮询
`POST /api/v1/retrieve_future {"request_id": ...}` 获取（`408`/`503` 表示还没完成，继续轮询）。
`scripts/wip/openpi_vla_smoke_lance.py` 里的 `_await_result`/`_poll_future` 已经封装好这套逻辑，
本 skill 的 driver 脚本直接复用，不需要重新实现。

---

## 1. `POST /api/v1/create_model`

路由：`mint_server/routes/training.py:2045` 附近。

请求体 `CreateModelRequest`（`mint_server/models/types.py:335-346`）：

```python
class CreateModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoRAConfig | None = None
    rollout_correction_config: RolloutCorrectionConfig | None = None
    type: Literal["create_model"] = "create_model"
```

`lora_config` 字段类型 `LoRAConfig`（`mint_server/models/types.py:308-315`）：

```python
class LoRAConfig(BaseModel):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True
```

⚠️ **对 openpi_pi05 backend，`rank` 必须精确等于 16，`train_unembed`/`train_mlp`/`train_attn`
必须全部是 `True`（默认值恰好满足，但如果显式传 `False` 会被拒绝）。** 详见
`references/pipeline_reference.md` 第2.1节，服务器校验逻辑在
`mint_server/backend/openpi/openpi_pi05_training.py:40-62`。

`rollout_correction_config` 对 pi0.5 **必须是 `None`**（pi0.5 不是 MoE，传非 None 会被
`_validate_rollout_correction_config_or_400` 提前拦截为 400）。

**响应**（经 future 轮询后）：

```json
{
  "request_id": "...",
  "model_id": "<session_id>_0",
  "type": "create_model",
  "backend": "openpi_pi05"
}
```

⚠️ **`model_id` ≠ 你发送的 `session_id`**——服务器会追加后缀（本次验证观察到的是 `_0`）。
必须从响应体读取真正的 `model_id`，不能假设等于发送值。这是本次开发中真实踩过的 bug
（详见 `references/pipeline_reference.md` 第2.2节）。

---

## 2. `POST /api/v1/mint/vla/train_step`

路由：`mint_server/routes/mint.py:627` 附近。

请求体 `VLATrainStepRequest`（`mint_server/models/mint_types.py:70-77`）：

```python
class VLATrainStepRequest(MintBaseModel):
    model_id: str
    data: list[VLADatum]
    loss_fn: str
    loss_fn_config: dict[str, Any] | None = None
    adam_params: AdamParams | None = None
    seq_id: int | None = None
    type: Literal["mint_vla_train_step"] = "mint_vla_train_step"
```

`loss_fn` 对 pi0.5 固定传 `"flow_matching"`。

`data` 是 `VLADatum` 列表（`mint_server/models/mint_types.py:60-67`）：

```python
class VLAObservation(MintBaseModel):
    model_input: ModelInput
    state: TensorData

class VLADatum(MintBaseModel):
    observation: VLAObservation
    supervision: dict[str, TensorData]
```

`ModelInput`（`mint_server/models/types.py:71-`）是一个 `chunks` 列表，每个 chunk 是
`{"type": "image", "data": <base64 PNG>, "format": "png", "expected_tokens": 256}`
或 `{"type": "encoded_text", "tokens": [...]}`。camera chunk 数量按
`MODEL_CONFIGS[base_model].camera_layout` 逐个生成（顺序对应）。

`TensorData`（`mint_server/models/types.py:422-427`）：

```python
class TensorData(BaseModel):
    data: list[float] | float
    shape: list[int]
    dtype: str = "float32"
```

`observation.state` 是 rank-1 的 `TensorData`（shape `[action_dim]`）；
`supervision["actions"]` 是 rank-2 的 `TensorData`（shape `[action_horizon, action_dim]`）。

**完整构造流程**（Lance 原始帧 → 这个 wire format）由
`scripts/wip/openpi_vla_smoke_lance.py::_pi05_datum_from_transformed` 实现，本 skill 的 driver
脚本直接复用这个函数，不要重新实现转换逻辑。

**响应**（经 future 轮询后）：

```json
{
  "metrics": {"loss:mean": 0.1462, ...},
  ...
}
```

loss 在 `result["metrics"]["loss:mean"]`，不是 `result["loss"]`。

---

## 3. `POST /api/v1/save_weights_for_sampler`

路由：`mint_server/routes/training.py:3889` 附近。

请求体 `SaveWeightsForSamplerRequest`（`mint_server/models/types.py:779-790`）：

```python
class SaveWeightsForSamplerRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    path: str | None = None  # checkpoint name for named save (None for ephemeral)
    ttl_seconds: int | None = None
    retry: bool = False
    seq_id: int | None = None
    sampling_session_seq_id: int | None = None  # For ephemeral flow
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"
```

`path=None` 是临时（ephemeral）保存；传字符串是命名保存（推荐——命名保存可复现，
临时保存的 URI 依赖运行时状态，可能不适合长期引用）。

**响应**（经 future 轮询后，真实观测样例）：

```json
{
  "path": "mint://vla-lora-83f8100eb5af_0/sampler_weights/skill_verify3_run",
  "sampling_session_id": null,
  "owner_id": "000000000000000000000001",
  "type": "save_weights_for_sampler",
  "filesystem_path": "/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints/persistent_cache/000000000000000000000001/vla-lora-83f8100eb5af_0/skill_verify3_run/sampler",
  "storage_tier": "persistent_cache"
}
```

`path` 是 `mint://` URI，`create_action_session` 需要这个值作为 `model_path`。
`filesystem_path` 是实际磁盘路径，`owner_id` 需要透传给后续的 `create_action_session`
（否则会因为 owner_id 缺失返回 400，是历史上真实踩过的坑，见
`wenxi_dev_md/Question.md:126`）。

---

## 4. `POST /api/v1/mint/action_sessions`（可选的推理验证步骤）

请求体 `MintCreateActionSessionRequest`（`mint_server/models/mint_types.py:80-85`）：

```python
class MintCreateActionSessionRequest(MintBaseModel):
    session_id: str
    action_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None
    owner_id: str | None = None  # admin-only owner scope for checkpoint references
```

`model_path` 传上一步 `save_weights_for_sampler` 返回的 `path`（`mint://...` URI）。
`owner_id` 透传上一步返回的 `owner_id`。

**响应**（非 future，直接返回）：

```json
{"action_session_id": "<uuid>", ...}
```

## 5. `POST /api/v1/mint/action_sessions/{action_session_id}/act`

请求体：`{"observation": <VLAObservation dict>}`（构造方式与 `train_step` 的
`observation` 字段完全一致，复用 `_build_batch` 生成的 datum 里的 `observation` 部分）。

**响应**（经 future 轮询后，真实观测样例）：

```json
{
  "actions": {"data": [...], "shape": [10, 32], "dtype": "float32"},
  "policy_timing": {"infer_ms": 13517.9, "temperature": 0.0},
  "type": "act"
}
```

`actions.shape` 应该是 `[action_horizon, action_dim]`（本次验证观测到 `[10, 32]`，
与 `pi05-libero-low-mem-finetune` 的配置一致）。

---

## 6. 清理

`DELETE /api/v1/models/{model_id}`：删除服务器端的会话/训练状态。
**不会删除** `save_weights_for_sampler` 已经写到磁盘的 checkpoint 文件——那些是独立持久化的。
`scripts/wip/openpi_vla_smoke_lance.py::_delete_model` 已经封装好这个调用（吞掉异常，
清理失败不影响主流程），driver 脚本直接复用。
