# 计划：修复 Issue #343 — 恢复文档契约 `loss:sum`，保留兼容键 `loss:mean`

## 目标

修复训练 metrics 的 API 契约不一致问题：

- 官方参考明确要求返回 `loss:sum`
- 当前实现只返回 `loss:mean`
- 下游按文档读取 `loss:sum` 时，会把真实训练误判成 `0.0` / “没有 loss”

这次修复的目标是：

1. 在训练相关返回里恢复 `loss:sum`
2. 继续保留 `loss:mean`，避免打爆现有内部脚本和 merge-gate
3. 保证 `loss:sum` 的语义真的是“token-level loss raw sum”，不是把现有 mean 换个名字

---

## 先说结论：原计划里最需要修正的地方

### 1. `verl_training.py` 可以直接补 `loss:sum`

`tinker_server/backend/verl_training.py` 当前已经在本地循环里累计 raw loss：

- `forward_backward()` 里，`cross_entropy` 路径直接用 `-(target_logprobs * weights).sum()`，随后累计到 `total_loss`
- `importance_sampling` / `ppo` 路径也先得到每个 item 的 raw summed loss，再累计到 `total_loss`
- 最后才用 `avg_loss = total_loss / max(total_tokens, 1)` 生成 `loss:mean`

因此对 VERL 路径，`loss:sum = total_loss` 的思路是成立的。

### 2. Megatron 路径不能把现有 `loss_value` 直接当成 `loss:sum`

`tinker_server/backend/megatron_training.py` 里的 helper 当前返回的是**归一化后的 loss**，不是 raw sum：

- `create_sft_loss_fn()` 先算 `-weighted_log_probs.sum()`，但真正返回的是再除以 `batch_num_tokens` 并乘 `dp_size` 的 `nll`
- `create_logprob_extractor_fn()` 也是同样模式
- `create_ppo_loss_fn()` 当前只把 `verl_ppo_loss(...)` 的返回值作为 `loss`

上层 `forward_backward()` / `forward()` 聚合的 `loss_value`，本质上沿用的是这些 helper 给的归一化结果。

所以：

- `verl_training.py`：可以直接加 `loss:sum`
- `megatron_training.py` / `megatron_distributed.py`：必须先把 raw summed loss 从 helper 层显式带出来，再一路上传

### 3. 不建议在每次训练调用时打 deprecated warning

原计划提出每次返回 `loss:mean` 时都 `logger.warning(...)`。

我认为这不适合这次修复，原因：

- `forward_backward` 是高频调用，warning 会刷爆日志
- 这次 issue 的核心是“恢复文档契约”，不是“立刻移除旧键”
- 现有内部测试和脚本大量依赖 `loss:mean`

这次应先做兼容修复：

- 立即恢复 `loss:sum`
- 继续保留 `loss:mean`
- 不加逐次 warning

如果后续真要废弃 `loss:mean`，单开 follow-up issue，做 once-per-process / once-per-session 的限频提示。

---

## 修复范围

本次只修复当前代码里**已经实现**的训练路径：

- `cross_entropy`
- `importance_sampling`
- `ppo`

说明：

- 官方文档还提到 `cispo` / `dro`
- 但当前本仓库这几条实际后端路径里并没有对应实现分支
- 所以 #343 不应在计划里笼统写成“所有训练目标都一起修复”，应以当前真实实现为准

---

## 设计原则

### 原则 1：`loss:sum` 必须来自 pre-normalization 的 raw sum

不能：

- 把 `loss:mean` 重命名成 `loss:sum`
- 用 `loss_value * num_tokens` 之类未经证明的反推

必须：

- 在真正做归一化之前，保存 token-level raw summed loss
- 在聚合层把这些 raw sums 逐 micro-batch 相加
- 对外返回 `metrics["loss:sum"]`

### 原则 2：`loss:mean` 继续保持现有行为

本次修复不改现有客户端兼容行为：

- 旧脚本继续读 `loss:mean`
- merge-gate 现有用例不需要立刻改写
- 新增测试只负责保证 `loss:sum` 存在且数值正确

### 原则 3：验证必须直接打训练请求，不能只跑 service smoke

`scripts/tools/smoke.py service` 只测：

- `/healthz`
- `/get_server_capabilities`
- `/create_session`

它**不会触发 `forward_backward` 或 `forward`**，因此不能用来验证 #343。

---

## 具体修改方案

### A. `tinker_server/backend/verl_training.py`

#### `forward_backward()`

当前已有：

- `total_loss`：累计 raw summed item loss
- `total_tokens`：累计有效 token 数
- `avg_loss = total_loss / max(total_tokens, 1)`

改法：

- 在 `metrics` 中新增 `"loss:sum": float(total_loss)`
- 保留 `"loss:mean": avg_loss`

目标形式：

```python
metrics = {
    "loss:sum": float(total_loss),
    "loss:mean": avg_loss,
    "num_samples:sum": float(len(data_items)),
    "num_tokens:sum": float(total_tokens),
}
```

#### `forward()`

当前已有：

- `total_loss`：按 weighted token loss raw sum 累计
- `avg_loss = total_loss / max(total_tokens, 1)`

改法同上：

- 新增 `"loss:sum": float(total_loss)`
- 保留 `"loss:mean": avg_loss`

#### 备注

这一层**不需要**额外保存中间量，现有局部变量已经足够。

---

### B. `tinker_server/backend/megatron_training.py`

这是这次修复的核心。不能只在返回 dict 里补一行，必须从 helper 层把 raw sum 带出来。

#### B1. `create_sft_loss_fn()`

当前逻辑：

- 先算 `weighted_log_probs = log_probs_flat * loss_mask_float`
- raw token sum 实际是 `-weighted_log_probs.sum()`
- 之后构造归一化后的 `nll`
- 返回的 metrics 里只有 `"loss"` 和 `"num_tokens"`

改法：

- 在归一化之前保存：
  - `loss_sum = -weighted_log_probs.sum()`
- metrics 里新增：
  - `"loss_sum": loss_sum.detach()`

示意：

```python
loss_sum = -weighted_log_probs.sum()

if batch_num_tokens_value > 0:
    nll = loss_sum / batch_num_tokens_value * dp_size
else:
    nll = loss_sum

metrics = {
    "loss": nll.detach(),
    "loss_sum": loss_sum.detach(),
    "num_tokens": ...,
}
```

#### B2. `create_logprob_extractor_fn()`

当前逻辑：

- 有 `nll = -(log_probs * loss_mask_float).sum()`
- 但随后会按 `batch_num_tokens` / `dp_size` 再做归一化
- 返回 metrics 里没有 raw sum

改法：

- 同样在归一化前保存 `loss_sum`
- metrics 增加 `"loss_sum"`

目标是让 `forward()` 也能输出正确的 `loss:sum`。

#### B3. `create_ppo_loss_fn()`

这里不能复用现有 `loss` 直接当 `loss:sum`，必须显式保留 token-level raw objective sum。

当前代码已经在本地算出了这些 token-wise 量：

- `ratio`
- `advantages`
- `response_mask_float`
- `clip_low` / `clip_high`
- `pg_loss1`
- `pg_loss2`

改法：

- 对 `importance_sampling`（这里通过 `epsilon=inf` 进入）：
  - raw sum 定义为 `(-advantages * ratio * response_mask_float).sum()`
- 对 `ppo`：
  - raw sum 定义为 `(torch.maximum(pg_loss1, pg_loss2) * response_mask_float).sum()`

然后：

- 保留 `verl_ppo_loss(...)` 当前的 `loss` 作为现有 mean/normalized 行为来源
- 额外在 metrics 中返回 `"loss_sum"`

示意：

```python
loss, _ = verl_ppo_loss(...)

pg_loss1 = -advantages * ratio
clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
pg_loss2 = -advantages * clipped_ratio
loss_sum = (torch.maximum(pg_loss1, pg_loss2) * response_mask_float).sum()

metrics = {
    "loss": ...,
    "loss_sum": loss_sum.detach().item(),
    ...
}
```

备注：

- 这里最重要的是让 `importance_sampling` 路径拿到**确定的 raw sum**
- 如果后续发现 `rollout_correction_config` 对 raw objective 定义还有额外细节，再单独精修
- 但绝不能把当前 `loss` 直接冒充成 `loss:sum`

#### B4. `MegatronTrainingWorker.forward_backward()`

当前只聚合：

- `loss_value`
- `num_tokens`
- PPO 辅助指标

改法：

- 新增 `loss_sum_value = 0.0`
- 从 `result_metrics.get("loss_sum", [])` 里逐 micro-batch 累加
- 最终 metrics 改成：

```python
metrics = {
    "loss:sum": float(loss_sum_value),
    "loss:mean": float(loss_value),
    "num_samples:sum": float(valid_count),
    "num_tokens:sum": float(num_tokens),
}
```

#### B5. `MegatronTrainingWorker.forward()`

改法同上：

- 聚合 `result_metrics["loss_sum"]`
- 返回 `loss:sum` 和 `loss:mean`

#### B6. 空批次快速返回路径

两处空批次返回都补：

```python
"metrics": {
    "loss:sum": 0.0,
    "loss:mean": 0.0,
    "num_samples:sum": 0.0,
    "num_tokens:sum": 0.0,
}
```

---

### C. `tinker_server/backend/megatron_distributed.py`

这一层要做的不是“自己重新推导 loss”，而是把 worker 已经算好的 raw sum 往上传。

#### C1. worker rank0 result 构建处

在 rank0 的 `result_dict` 里新增：

- `"loss_sum_value": float(loss_sum_value)`

其中 `loss_sum_value` 来自对 `result["metrics"]["loss_sum"]` 的逐 micro-batch 累加。

#### C2. `MegatronWorkerGroup.forward_backward()`

当前 group-level metrics 用的是：

```python
"loss:mean": float(loss_value)
```

改成同时返回：

```python
"loss:sum": float(loss_sum_value),
"loss:mean": float(loss_value),
```

其中 `loss_sum_value = rank0_result.get("loss_sum_value", 0.0)`。

#### C3. `MegatronWorkerGroup.forward()`

同样：

- 从 `rank0_result` 读 `loss_sum_value`
- 在 group-level metrics 中暴露 `"loss:sum"`

---

## 测试与验证计划

### 1. 新增复现/回归脚本：`scripts/tools/reproduce_issue_343.py`

必须有一个 issue 专用脚本，符合 bugfix skill 的要求。

用途：

- 本地运行，目标服务通过 `TINKER_BASE_URL` / `TINKER_API_KEY` 指定
- 明确复现“训练已发生，但 `loss:sum` 之前缺失”的问题
- 修复后明确断言 `loss:sum` 存在且数值合理

脚本至少覆盖两类 case：

#### Case A: `cross_entropy`

做一个最小 SFT batch，调用 `forward_backward`，断言：

- `metrics["loss:sum"]` 存在
- `metrics["loss:mean"]` 仍存在
- `metrics["num_tokens:sum"] > 0`
- `loss:sum / num_tokens:sum` 与 `loss:mean` 数值一致或在容差内一致

#### Case B: `importance_sampling`

做一个最小 RL batch，调用 `forward_backward(loss_fn="importance_sampling")`，断言：

- `metrics["loss:sum"]` 存在
- `metrics["loss:mean"]` 仍存在
- `grad_norm:last` / `step` 所在的训练流程仍可继续
- 不再出现“训练发生了但 loss key 缺失”的情况

### 2. 新增一个 focused test，专门检查 metrics contract

建议新增一个小的 API contract 测试，而不是指望现有 merge-gate 用例顺带覆盖。

可以放在：

- `.claude/skills/merge-gate/tests/test_loss_metrics_contract.py`

最小断言：

- `forward_backward(..., loss_fn="cross_entropy")` 返回 `loss:sum`
- `forward_backward(..., loss_fn="importance_sampling")` 返回 `loss:sum`
- `loss:mean` 仍然存在

注意：

- 现有 `test_moe_rl` / `test_gradient_isolation` 主要只是继续证明兼容键没被打坏
- 它们本身**不能**证明 #343 被修好

### 3. 开发验证顺序

推荐顺序：

1. 先修 `verl_training.py`
2. 再修 `megatron_training.py` helper，把 `loss_sum` 从底层打通
3. 再修 `megatron_distributed.py` 上传链路
4. 跑 `reproduce_issue_343.py`
5. 跑 focused contract test
6. 最后再跑受影响的 merge-gate 子集，确认 `loss:mean` 兼容性没坏

---

## 受影响文件清单（修正版）

必改：

- `tinker_server/backend/verl_training.py`
- `tinker_server/backend/megatron_training.py`
- `tinker_server/backend/megatron_distributed.py`
- `scripts/tools/reproduce_issue_343.py`

建议新增：

- `.claude/skills/merge-gate/tests/test_loss_metrics_contract.py`

可能会顺手改到：

- 相关测试辅助文件（如果需要复用现有 `create_session` / `forward_backward` helper）

---

## 不做的事

本次**不做**：

- 不把 `loss:mean` 立即废弃
- 不在每个训练 step 打 deprecated warning
- 不修改官方 reference 文档（因为文档本身是对的）
- 不修改所有客户端为只读 `loss:sum`

---

## 完成标准

只有满足下面条件，#343 才算真的修好：

1. `forward_backward` 和 `forward` 的训练 metrics 同时返回：
   - `loss:sum`
   - `loss:mean`
   - `num_tokens:sum`
2. `loss:sum` 的数值来自 raw summed loss，不是 mean 伪装
3. `importance_sampling` 路径被直接验证过
4. 现有依赖 `loss:mean` 的测试和脚本仍然可运行

---

## 实施备注

如果开始动手实现，建议分成两个 commit 思路（即使最后不一定真的拆 commit）：

1. `backend`: 打通 `loss_sum` 计算和上传链路
2. `verification`: 新增 `reproduce_issue_343.py` + focused contract test

这样 review 时最清楚，也最容易证明“不是只改了文案，而是真的修了契约”。
