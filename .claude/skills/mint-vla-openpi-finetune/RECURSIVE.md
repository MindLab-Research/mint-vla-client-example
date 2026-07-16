# RECURSIVE.md — 持续更新的脚本/逻辑清单

这份文档和 `references/pipeline_reference.md` 分工不同：

- `references/pipeline_reference.md` 是**一次性调研**的快照，记录已验证过的事实、坑点、根因分析，
  内容偏"发生过什么、为什么"，不应该频繁改动。
- **这份 `RECURSIVE.md` 是活文档**，每次给这个 skill 加新脚本、新功能，或者发现服务器端约束发生
  变化时，都应该**追加更新**这里。目的是让下一次改动这个 skill 的人，先扫一眼这份清单就知道
  "现在有哪些脚本、各自干什么、哪些地方是已知的临时限制、以后可能要改"，不用重新翻遍整个仓库。

**更新规则**：
1. 新增脚本 -> 在对应分类下加一行，格式：`脚本路径` — 一句话用途 — 状态（生产/复用/待验证）。
2. 发现服务器端约束变化（比如 LoRA rank 限制被放宽了）-> 更新"已知限制 / 改进目标"章节的状态，
   不要删除历史记录，改成"已解决（日期，验证方式）"。
3. 不要把这份文档变成第二份 `pipeline_reference.md`——遇到需要深入解释"为什么"的内容，
   写到 `pipeline_reference.md` 里，这里只放"是什么、在哪、状态如何"的清单式条目。

---

## 1. 本 skill 拥有的脚本（生产代码，随 skill 迭代）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `scripts/tools/openpi_vla_lora_finetune.py` | 主 driver。`create_model → train_step×N → save_weights_for_sampler → [推理验证] → [MSE评估] → [推理写回lance] → cleanup` | 生产，已端到端验证 |
| `scripts/tools/openpi_vla_eval_mse.py` | MSE/L1 量化评估（归一化空间对比pred vs gt，含零基线对比）。可独立运行，也被 driver 的 `--eval-mse` 调用 | 生产，已端到端验证 |
| `scripts/tools/openpi_vla_infer_to_lance.py` | 对数据集逐帧推理，合并预测结果回原 Lance 结构（保留原11列+追加`pred_actions`/`pred_actions_physical`/`pred_action_mse`/`pred_meta`4列）。可独立运行，也被 driver 的 `--infer-to-lance` 调用 | 生产，已端到端验证（用`--max-samples 3`验证过schema正确性） |

### 1.1 与 `scripts/wip/` 原始脚本的对应关系

上面三个脚本都是"生产化"版本，通过 `importlib` 动态加载对应的 `scripts/wip/` 脚本文件，
复用其中的可复用函数/类，不复制粘贴、不修改原文件：

| 生产脚本 | 复用自（`scripts/wip/`） | 复用了什么 |
|---|---|---|
| `openpi_vla_lora_finetune.py` | `openpi_vla_smoke_lance.py` | `LanceViewpi05Dataset`, `_headers`, `_post_json`, `_get_json`, `_await_result`, `_build_model_config`, `_make_data_config`, `_compute_norm_stats`, `_build_batch`, `_delete_model` |
| `openpi_vla_eval_mse.py` | `openpi_vla_smoke_lance.py` | 同上（子集） |
| `openpi_vla_infer_to_lance.py` | `openpi_vla_smoke_lance.py` + `openpi_vla_infer_obs.py` | 前者同上；后者复用 `_unnormalize_actions` |

如果 `scripts/wip/openpi_vla_smoke_lance.py` 或 `openpi_vla_infer_obs.py` 的函数签名发生变化
（比如有人重命名了 `_build_batch` 或改了 `LanceViewpi05Dataset.__init__` 的参数），
上面三个生产脚本会直接因为 import 失败或调用失败而报错——**发现这类错误时先检查 `scripts/wip/`
里的源函数是否被改动过**，不要假设是生产脚本自己的 bug。

---

## 2. 已知限制 / 改进目标

这些是**当前存在、但可能随服务器代码演进而改变**的约束。每条都标注"当前状态"和"如何验证是否已变化"。

### 2.1 LoRA rank 服务器端强制要求 = 16

**当前状态（2026-07-15验证）**：`mint_server/backend/openpi/openpi_pi05_training.py` 的
`validate_openpi_pi05_create_request` 硬编码要求 `lora_config.rank == OPENPI_PI05_LORA_RANK`
（当前值16），且 `train_attn`/`train_mlp`/`train_unembed` 必须全部 `True`。这不是配置项，
是代码里的常量比较。

**本 skill 的应对方式**：driver 脚本（`openpi_vla_lora_finetune.py::validate_lora_config`）
**不再本地拦截**非16的值——只打印警告，然后仍然把用户指定的值发给服务器。如果服务器拒绝，
driver 会把服务器返回的 `detail` 原文透传给用户（见 `_create_model_with_lora` 的
`requests.exceptions.HTTPError` 处理），不会伪造或省略错误信息。

**改进目标**：如果未来这个约束在服务器端被放宽（比如 `OPENPI_PI05_LORA_RANK` 变成可配置，
或者对 rank 的要求变成一个范围而不是精确值），需要做的事：
1. 更新这条记录的"当前状态"，写清楚新的约束是什么（哪次验证、验证方式）。
2. 更新 `SKILL.md` 的用户问答表格里"LoRA rank"那一行的默认值/说明文字。
3. 更新 `references/pipeline_reference.md` 第2.1节和 `references/troubleshooting.md`
   里对应的症状描述（目前写的是"精确等于16"，如果放宽了这个描述就不准确了）。
4. **不需要改 driver 脚本本身**——它已经是"发送用户指定的值，让服务器判断"的设计，
   本身就是为了兼容未来约束变化而不需要改代码。

**如何验证是否已变化**：跑
```bash
python scripts/tools/openpi_vla_lora_finetune.py --lance-dataset <path> --lora-rank 8 --steps 1
```
如果 `create_model` 仍然报 "OpenPI pi0.5 training only supports the upstream LoRA rank 16"，
约束还在生效。如果这次成功了，说明约束已经放宽，按上面4步更新文档。

### 2.2 MANO 原始数据集不能直接使用

**当前状态**：`new_all_generated_mano.lance` 这类原始运动学捕捉数据（`hands`/`objects`/`contact`
schema，无 `image`/`wrist_image` 字段）不能直接喂给本 skill 的任何脚本——`LanceViewpi05Dataset`
硬性要求 `image`/`wrist_image`/`state`/`actions`/`prompt` 字段。需要先经过 MuJoCo 渲染管道
把运动学轨迹转换成带图像的 Lance 数据集，这个渲染转换脚本在本仓库内未找到（可能在别的代码库）。

**改进目标**：如果之后找到或写出了这个渲染转换管道，应该：
1. 在本 skill 里新增一个前置步骤（或独立脚本），让用户可以直接给 MANO 原始数据，
   skill 自动完成"渲染转换 → 得到 image/wrist_image schema → 后续微调流程"的全链路。
2. 更新 `SKILL.md` 的 Scope 章节，去掉"MANO conversion is out of scope"这句话。
3. 在这份 `RECURSIVE.md` 里新增对应脚本的记录（第1节表格）。

**如何验证**：尝试对目标数据集跑 `--dry-run`，如果 `probe_lance_dataset`/`validate_action_dim`
之前就报 `KeyError` 或字段缺失类错误，说明还是不兼容的 schema。

### 2.3 `--infer-to-lance` 对大数据集可能很慢

**当前状态**：`run_inference_and_merge_to_lance` 对数据集的**每一帧**都发一次 `act()` 请求
（本次验证观测到单次推理约14秒，包含首次编译/网络开销后单帧耗时会降低，但仍是逐帧串行）。
对 887 帧的 full 数据集，全量跑一次预计需要较长时间（未做过全量计时，验证时只测过
`--max-samples 3`）。

**改进目标**：如果这个耗时成为实际瓶颈，可以考虑：
- 支持批量 `act()` 调用（如果服务器 API 支持批量推理，当前 `openpi_vla_smoke_lance.py` 的
  `_build_batch` 已经支持多样本打包，但 `act` 端点是否接受批量输入未验证）。
- 增加一个 `--infer-to-lance-max-frames` 限制选项，避免用户无意中对超大数据集触发超长任务。

**如何验证当前耗时**：跑一次 `--infer-to-lance` 全量（不加 `--max-samples`），记录起止时间，
把结果补充到这条记录里。

---

## 3. 验证历史（每次给 skill 加新功能后，记一行）

| 日期 | 验证内容 | 结果 | 备注 |
|---|---|---|---|
| 2026-07-14 | driver 脚本首次端到端验证（32维数据集，4步训练） | ✅ 成功 | 发现并修复 LoRA rank 拦截过严 + model_id提取bug |
| 2026-07-15 | 用户真实调用：50步训练 + 推理验证 | ✅ 成功 | loss从0.13降到0.06左右，checkpoint正常落盘 |
| 2026-07-15 | LoRA rank改为"警告不拦截"后，验证 rank=8 透传服务器400原文 | ✅ 成功 | 确认服务器detail文本被正确显示，不是本地伪造错误 |
| 2026-07-15 | `--eval-mse` + `--infer-to-lance` 同时启用，`--max-samples 3` 快速验证 | ✅ 成功 | MSE aggregate输出正确；merged lance schema确认15列（原11+新4），pred_actions shape[10,32]正确 |

---

## 4. 如何给这份文档添加新条目

给 skill 加新脚本时：
1. 在第1节表格加一行（脚本路径、用途、状态）。
2. 如果新脚本复用了 `scripts/wip/` 的现有代码，在1.1节的表格加一行。
3. 如果引入了新的、可能随时间变化的约束，在第2节新增一个"2.X"小节，按现有格式写
   "当前状态 / 本skill的应对方式 / 改进目标 / 如何验证是否已变化"四段。
4. 做完真实端到端验证后，在第3节加一行记录（日期、验证内容、结果、备注）。
