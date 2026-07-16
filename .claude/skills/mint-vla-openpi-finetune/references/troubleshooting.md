# 故障排查

症状 → 原因 → 修复的快查表。所有条目均来自本 skill 开发过程中的**真实验证**
（起服务器、跑训练、复现错误），不是理论推测。详细背景见 `pipeline_reference.md`。

---

## `create_model` 返回 HTTP 400

### 症状A：错误信息里提到 "LoRA rank" 或 "does not support partial LoRA toggle"

**原因**：`--lora-rank` 传了非16的值，或者 `--lora-train-attn`/`--lora-train-mlp`/
`--lora-train-unembed` 传了 `--no-lora-train-*`。openpi_pi05 backend 硬性要求
`rank=16` 且三个 train 开关全部 `True`（`mint_server/backend/openpi/openpi_pi05_training.py:40-62`）。

**修复**：不要传这些 flag，用默认值（rank=16，三个开关默认True）。driver 脚本会在
发请求前用 `validate_lora_config()` 本地拦截，所以正常使用下不应该走到服务器返回400——
如果看到这个400，说明校验被绕过了（比如直接调用了 `_create_model_with_lora` 而没经过
`validate_lora_config`，检查调用顺序）。

### 症状B：错误信息里提到 "only supported on Megatron backend (MoE models)"

**原因**：传了非 None 的 `rollout_correction_config`。pi0.5 不是 MoE 模型
（`is_moe=False`），这个字段必须省略/None。

**修复**：不要传 `rollout_correction_config`。

### 症状C：错误信息里提到 "exceeds max_model_len"

**原因**：tokenized prompt 长度超过模型的 `max_model_len`（pi0.5 是200）。检查
Lance 数据集里的 `prompt` 字段是否异常长。

---

## `create_model` 成功（200），但紧接着的 `train_step` 返回 HTTP 503

**原因**：几乎肯定是 `model_id` 提取错误。`create_model` 响应里的 `model_id`
**不等于**你发送的 `session_id`——服务器会追加后缀（例如
`session_id="vla-lora-abc123"` → 响应 `model_id="vla-lora-abc123_0"`）。如果代码假设
两者相同，后续 `train_step` 用错误的 `model_id` 会导致服务器
`has_local_training_session(model_id)` 查找失败，返回503（不是400，容易误判成
GPU/服务不可用问题）。

**验证方法**：查服务器日志里 `train_step` 那条请求前后几行，确认 `elapsed_ms` 是否
异常小（本次真实案例是 `elapsed_ms=2.371`，说明是在进入实际训练逻辑前就被快速拒绝，
不是训练过程中真的挂了）。

**修复**：确保从 `create_model` 的响应体读取 `model_id`（`create_result.get("model_id")`），
不要用发送时生成的本地变量。参考
`scripts/wip/openpi_vla_smoke_lance.py::_create_model` 的正确写法。

---

## action_dim 不匹配（driver 在 dry-run 或早期就报错退出）

**症状**：driver 脚本打印类似
```
action_dim mismatch: dataset '...' has 64-dim state/actions, but base_model '...'
is configured with action_dim=32 in model_registry.py.
```
并以非零退出码结束。

**这是设计好的行为，不是bug。** 不要试图：
- 对数据集做 zero-padding 让维度凑够
- 加 mask 屏蔽多出来的维度让训练"跑起来"
- 改 `model_registry.py` 里的 `action_dim` 去匹配数据集（除非你真的要切换到一个不同维度的
  模型配置，且清楚这意味着从头训练一个新维度的 action head）

这些"绕过"方式全部在 `ActionHeadSummary.md` 的10个实验里验证过会失败或效果显著变差。

**正确修复**：
- 如果数据集维度比模型小：考虑换成维度匹配的模型配置，或收集补全维度的数据。
- 如果数据集维度比模型大：确认是否有多余的、不该训练的维度（比如未使用的传感器通道），
  考虑数据本身是否需要清洗，而不是让模型强行适配。

---

## Lance 数据集读取失败（`ArrowInvalid: ... Not found`）

**症状**：`probe_lance_dataset()` 报告最新版本读取失败，列出某个更早的可读版本号。

**原因**：数据集的 manifest 指向的数据分片文件缺失（可能是外部同步未完成，或写入中断）。
这不是本 skill 的driver脚本能修复的问题——它只能探测并报告。

**处理方式**：
1. 如果 driver 给出了一个可读的旧版本号，先确认这个旧版本的数据量/内容是否真的是你想要
   训练的数据（不要盲目假设"能读就是对的"），确认后用 `--lance-dataset-version <N>` 重跑。
2. 如果连最早的版本都读不出来，说明数据集本身有问题（可能还在同步中），需要找到写入这批
   数据的上游流程确认状态，不是等待或重试能解决的。

---

## 训练中途 OOM（`RESOURCE_EXHAUSTED`）

**症状**：训练进行到某一步（历史上观察到过第17步、第33步）后显存耗尽。

**不要立刻做的事**：不要第一反应就去调小 batch size 或调整 `XLA_FLAGS`——那些是历史上的
**缓解措施**，不是根因修复，可能只是把崩溃点往后推。

**先检查**：worker 端的 prompt padding 修复是否还在——变长 prompt 会导致每步都是不同的
JAX traced shape，每次都要重新编译且编译产物不释放，显存持续爬升。这个修复应该在
`mint_server/backend/openpi/openpi_pi05_worker.py` 的 `_observation_from_payload` 里，
把 token/mask pad 到固定的 `max_token_len`。如果这段代码被意外回退或改动过，OOM 会复发。

**已知的加固措施**（如果根因修复确认还在，仍建议保留）：
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=`（禁用CUDA command buffer堆积）
- `CUDA_VISIBLE_DEVICES` 绑定确认空闲的GPU卡（本机是共享GPU box，其他用户可能占用0/1/2号卡）

详见 `pipeline_reference.md` 第5节的完整根因分析。

---

## 训练"跑完"了但 loss 全是 null / 表面成功实则失败

**原因**：future 轮询逻辑没有正确处理失败的 future。服务器对失败的操作会返回
HTTP 200 但 body 里带 `{"error": ...}`（见 `routes/futures.py` 的 `_failed_payload`）——
如果轮询代码只检查 HTTP 状态码不检查 body 内容，会把失败误判为"空结果"而不是报错。

**修复**：确保用的是 `scripts/wip/openpi_vla_smoke_lance.py::_await_result` 的当前版本
（第111-122行），它会检查 `result.get("error")` 并主动 `raise RuntimeError`。不要自己
重新实现一个更简单的轮询逻辑而跳过这个检查。

---

## 服务器日志里出现 "Started a local Ray instance"，尽管是 no-Ray 模式

**这不是 bug，也不需要处理。** 详见 `pipeline_reference.md` 第3节——这是
`config_actor` 健康检查路径的已知副作用（调用 `ray.get_actor()` 触发 Ray SDK 自动
fallback 起本地单机集群），与 openpi pi0.5 的 Ray-free 训练路径无关，不影响训练是否成功。
不要因为看到这条日志就怀疑环境配置有问题，也不要手动去 kill 这个本地 Ray 进程当作"修复"。

---

## healthz 返回 503 / "unhealthy"

**这是 Ray-free 降级模式下的预期状态标记**，不代表服务不可用。判断服务器是否就绪应该看
`200` 或 `503` 都算就绪（`scripts/vla/PI05lance_local_norray.sh` 的健康检查逻辑正是这样
判断的），不要只接受 `200`。

---

## `ModuleNotFoundError: No module named 'lance'`

**原因**：`PYTHONPATH` 漏配了 `EXTRA_PYDEPS`（`/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps`）。
`lance` 模块只存在于这个目录，不在 `gpu_rl/site-packages` 下。

**修复**：确保 `PYTHONPATH` 按以下顺序拼接：
```
<repo_root>:<EXTRA_PYDEPS>:<GRB>/site-packages:<GRB>/src/openpi/src:<GRB>/src/openpi/packages/openpi-client/src
```
