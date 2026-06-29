# Qwen3.6-27B: text-only → text+image (VLM) 微调扩展方案

> 目标：把当前 Mint 里被"剥壳成纯文本 LLM"的 `Qwen/Qwen3.6-27B`，扩展成支持
> `text+image` 输入的 VLM 微调（LoRA）。本文档给出**贴合现有工程逻辑**的修改/新增
> 方案，作为后续逐步攻破的施工蓝图。
>
> 约定：路径都相对仓库根 `/vePFS-Mindverse/user/intern/wenxi/mint`。

---

## 0. 现状结论（先对齐事实）

上游 `Qwen/Qwen3.6-27B` 在 HuggingFace 上**本身就是 VLM**（`image-text-to-text`，
外层 `model_type=qwen3_5`，带 vision encoder，正常用 `AutoModelForMultimodalLM` 加载，
支持 text/image/video）。Mint **主动只接了它的语言模型骨架**，这是工程取舍，不是模型限制。

剥壳发生在三处，全部围绕"丢掉 vision、只留 text_config"：

| 链路 | 文件 | 剥壳动作 |
|------|------|----------|
| 模型声明 | `mint_server/backend/core/model_registry.py:250` | `supported_modalities=("text",)` |
| 推理(vLLM) | `mint_server/backend/inference/qwen35_text_vllm_adapter.py` | 只取 `text_config`，架构改写为 `Qwen3NextForCausalLM`，**删 M-RoPE**（`mrope_section`/`mrope_interleaved`，第 314-315 行） |
| 训练(LoRA) | `mint_server/backend/training/qwen36/qwen36_trainer.py:124` | 用 `text_config` 让 `AutoModelForCausalLM` 选中 `Qwen3_5ForCausalLM`（无 vision），并剥 `language_model.` 前缀 |

数据层关键事实：`mint_server/models/types.py` **已经定义了 `ImageChunk` 和
`ImageAssetPointerChunk`**（第 24-62 行），`ModelInput.chunks` 也已是
`EncodedTextChunk | ImageChunk | ImageAssetPointerChunk` 的判别联合。但
`ModelInput.to_ints()`（第 102 行）**显式拒绝**非文本 chunk，trainer 的
`forward_backward`（`qwen36_trainer.py:281`）也只读 `type=="encoded_text"` 的 chunk。

> 含义：**数据协议层已经为图像预留了位置，是被下游"主动忽略"的。** 这让 VLM 扩展
> 的数据层改动远比从零设计小——主线是"让下游别再忽略 image chunk"。

---

## 1. 设计原则（沿用项目既有风格）

1. **并存而非替换**：新增 VL 路径，**不破坏现有 text-only 路径**。沿用项目
   "text adapter / mm adapter 对称共存"的风格（参考 text adapter 已经预留的
   `mint_supported_modality` 字段）。
2. **由 `supported_modalities` 做路由开关**：`model_registry` 是唯一事实源，
   推理/训练两侧都读它来决定走 text 分支还是 mm 分支。
3. **隔离依赖栈复用**：Qwen3.6 已有独立 vLLM/transformers 栈
   （`qwen36-vllm-deps` / `qwen36-deps`，见 `multinode_inference.py:2763`）。
   VLM 用**同一套隔离栈**，只是不再剥 vision，省去再造依赖环境。
4. **vision encoder 默认冻结**：RL/LoRA 微调先**只训语言层 LoRA，冻结 vision tower
   与 projector**。这是工程量最小、最稳的起点；是否解冻作为后续旋钮。
5. **失败要响亮**（项目铁律）：mm 路径若拿到的 checkpoint 没有 vision_config，
   或 vLLM 隔离栈里找不到多模态架构类，**直接 raise**，绝不 fallback 到 text 路径
   产出"看似能跑但忽略了图像"的结果。

---

## 2. 分层改动清单

### 2.1 模型声明层 —— `model_registry.py`

新增一个 VL 变体条目，**保留原 text 条目不动**，避免影响现有 RLcheck：

```python
"Qwen/Qwen3.6-27B-VL": ModelConfig(
    num_parameters=27.0,
    is_moe=False,
    inference_tp=4, inference_dp=1,
    train_tp=1, train_ep=1,
    max_model_len=32768,
    max_num_seqs=128,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.90,
    max_loras=4, max_cpu_loras=16, max_lora_rank=64,
    gradient_checkpointing=True,
    vllm_engine="async",
    vllm_distributed_executor_backend="mp",
    supported_modalities=("text", "image"),   # ← 路由开关
    training_backend="verl_fsdp2_lora",
    # 新增字段（见下）
    # mm_image_token_budget=4096,
    # mm_processor_min_pixels=..., mm_processor_max_pixels=...,
),
```

`ModelConfig` dataclass（`model_registry.py:40` 起）需新增多模态字段，沿用现有
"`None` = 用默认" 的风格：

```python
mm_image_token_budget: int | None = None      # 单图最大视觉 token 数（影响 max_model_len 预算）
mm_processor_min_pixels: int | None = None     # 透传给 AutoProcessor 的图像分辨率下界
mm_processor_max_pixels: int | None = None     # 上界
mm_limit_images_per_prompt: int | None = None  # vLLM limit_mm_per_prompt
```

> 设计点：是否复用 `Qwen/Qwen3.6-27B` 同名条目（加一个 `enable_vision` flag）还是
> 用 `-VL` 新 ID？**推荐新 ID**，因为推理/训练/权重命名、LoRA 缓存 key 都按
> model id 区分，新 ID 隔离最干净，且能让两条路径在同集群并存做对比。

### 2.2 数据协议层 —— `models/types.py` + `models/mint_types.py`

数据类型**基本不用新建**（`ImageChunk`/`ImageAssetPointerChunk` 已存在）。要做的是：

1. **`ModelInput` 增加一个安全的多模态展开方法**，区别于 text-only 的 `to_ints()`：
   ```python
   def to_multimodal(self) -> tuple[list[int], list[ImageRef]]:
       """返回 (input_ids含<image>占位token, 有序图像引用列表)。
       text-only 调用方继续用 to_ints()；mm 调用方用这个。"""
   ```
   关键不变量：**image chunk 在 chunks 序列中的位置**决定了它在 token 序列里
   对应哪段 `<image>` 占位 token——位置即对齐，不能丢序。
2. **`mint_types.py` 训练 datum**：现有 `Datum.model_input: ModelInput` 已能携带
   image chunk，无需改类型；只需保证序列化/反序列化保留 image chunk（pydantic
   判别联合已支持）。若图像走 base64 inline（`ImageChunk.data`）注意 payload 大小，
   大图建议走 `ImageAssetPointerChunk`（PFS 路径/对象存储指针）。

> 设计点：图像预处理放客户端还是 server？
> - **客户端预处理**（传 `pixel_values`）：server 改动小，但要新增携带张量的 chunk 类型。
> - **server 端 `AutoProcessor`**（传原图 bytes/指针 + `<image>` 占位）：更贴合 Qwen
>   官方用法，复用模型自带 processor，**推荐**。本方案按此设计：image chunk 只携带
>   原图，预处理在 trainer/inferencer 内用 `AutoProcessor` 完成。

### 2.3 推理层 —— 新增 `qwen35_mm_vllm_adapter.py`

与 `qwen35_text_vllm_adapter.py` **对称**的兄弟文件，差异是"保留 vision"：

```python
"""Qwen3.5/3.6 multimodal vLLM adapter.

Mirror of qwen35_text_vllm_adapter, but KEEPS vision_config + M-RoPE so vLLM
routes into the multimodal Qwen3.5 architecture instead of the text-only shim.
"""
QWEN35_MM_VLLM_ARCHITECTURE = "<在 qwen36-vllm-deps 的 vLLM 里查实>"  # ★ 头号未知数
QWEN35_MM_SHIM_MARKER = "mint_qwen35_mm_shim"

def qwen35_as_multimodal_config(raw_config: dict) -> dict:
    if raw_config.get("model_type") not in _QWEN35_MODEL_TYPES:
        raise ValueError("not a qwen3_5 family config")
    config = dict(raw_config)                       # 整份透传，含 vision_config
    config["architectures"] = [QWEN35_MM_VLLM_ARCHITECTURE]
    config[QWEN35_MM_SHIM_MARKER] = True
    config["mint_supported_modality"] = "text_image"
    # 与 text adapter 相反：保留 rope_parameters 里的 mrope_section/mrope_interleaved
    return config

def materialize_qwen35_mm_vllm_config(model_path, *, root_dir=None) -> str | None:
    # 与 materialize_qwen35_text_vllm_config 同构，调用上面的转换并落盘
    ...
```

`multinode_inference.py`（vLLM 启动处，约 2763 行起）增加按模态分流：

```python
cfg = get_model_config(self.model_path)
if "image" in cfg.supported_modalities:
    vllm_config_path = materialize_qwen35_mm_vllm_config(self.model_path, root_dir=...)
    engine_kwargs["limit_mm_per_prompt"] = {"image": cfg.mm_limit_images_per_prompt or 1}
    # 不要剥 M-RoPE；engine 允许 mm_data
else:
    vllm_config_path = materialize_qwen35_text_vllm_config(self.model_path, root_dir=...)
```

采样请求路径需把 image chunk 转成 vLLM 的 `multi_modal_data={"image": [...]}` 一并提交。

### 2.4 训练层 —— 新增 `qwen36_vl_trainer.py`（或给 `qwen36_trainer.py` 加 mm 分支）

推荐**新增 `Qwen36VLTrainingWorker`**，与 `Qwen36TrainingWorker` 同接口（保证
`VerlTrainingEngine` 能透明路由），差异点：

1. **模型加载**：不再取 `text_config`、不再用 `Qwen3_5ForCausalLM`。改用多模态类
   （`AutoModelForImageTextToText` / `Qwen3_5ForConditionalGeneration`，需查实），
   **不剥 `language_model.` 前缀**（多模态类本就期望该前缀）。
2. **processor**：`AutoProcessor.from_pretrained(...)` 与 tokenizer 并存，用于把
   image chunk 的原图 → `pixel_values` + `image_grid_thw`。
3. **LoRA target**：起步沿用 `QWEN36_LORA_TARGET_MODULES`（语言层），
   **vision tower / multi-modal projector 标记 `requires_grad=False`**（冻结）。
   后续旋钮：把 projector 或 vision proj 加进 target_modules。
4. **`forward_backward`**：在现有按 `chunks` 拼 `input_ids` 的逻辑（`qwen36_trainer.py:281`）
   旁，增加处理 image chunk 的分支：
   ```python
   input_ids, image_refs = ModelInput(**model_input).to_multimodal()
   if image_refs:
       proc = self.processor(images=load_images(image_refs), text=None, return_tensors="pt")
       outputs = self.model(input_ids=input_ids_t,
                            pixel_values=proc["pixel_values"].to(self.device, torch.bfloat16),
                            image_grid_thw=proc["image_grid_thw"].to(self.device))
   else:
       outputs = self.model(input_ids=input_ids_t)   # 纯文本样本，与现状一致
   ```
   损失计算（cross_entropy / importance_sampling / ppo）**完全不动**——它只作用在
   语言 token 的 logits 上，image token 通过 `weights/loss_mask=0` 排除在损失外。
5. **`num_gpus`**：现 trainer 是 `@ray.remote(num_gpus=2)`。加上 vision tower 后
   显存上升，先**按内存模型计算**（不要试错），27B bf16≈54GB + vision encoder
   + 图像 activation，评估是否需要 `num_gpus` 提升或更激进的 gradient checkpointing。

### 2.5 路由层 —— `qwen36_verl_fsdp2_lora.py` / trainer manager

`is_qwen36_model()`（`qwen36_verl_fsdp2_lora.py:42`）需识别 `-VL` 变体；
trainer manager（`qwen36_trainer_manager.py`）按 `supported_modalities` 决定
实例化 `Qwen36TrainingWorker` 还是 `Qwen36VLTrainingWorker`。

---

## 3. 端到端数据流（目标态）

```
client                         server (route)              trainer / inferencer
──────                         ──────────────              ────────────────────
ModelInput.chunks =            mint.py 解析请求             Qwen36VLTrainingWorker:
  [EncodedTextChunk(...),  ──▶  get_model_config       ──▶   to_multimodal() →
   ImageChunk(png bytes),       "image" in modalities?        input_ids(含<image>占位)
   EncodedTextChunk(...)]       是 → mm 分支                  + AutoProcessor(原图)
                                                              → pixel_values/grid_thw
                                                              → model(input_ids, pixel_values,...)
                                                              → logits[语言token] → loss
```

不变量：
- image chunk 的**序列位置**↔ `<image>` 占位 token 段，一一对齐，禁止重排。
- 损失只在语言 token 上（image token 的 `loss_mask=0`）。
- 纯文本样本走 mm trainer 时，等价于现状（不传 pixel_values）。

---

## 4. 攻破顺序（建议的里程碑）

按"先证伪最大未知数，再搭骨架，最后端到端"的顺序：

1. **M0 查实 vLLM 多模态架构类名 + 输入格式**（头号未知数）
   在 `qwen36-vllm-deps` 的隔离 vLLM 栈里确认：
   - `Qwen3.6-27B` 完整 config 对应的多模态架构注册类名是什么；
   - 它接受的 `multi_modal_data` / `pixel_values` / `image_grid_thw` 格式。
   验证命令草案（在 GPU worker 上、用 qwen36 隔离栈 Python）：
   ```python
   from vllm import LLM
   # 用完整 config（不剥 vision）加载，打印 model_config.architectures
   # 跑一张图 + 一句话，确认能产出 token
   ```
   **此步不通，后面全是空中楼阁。**

2. **M1 transformers 多模态加载 + 一次 forward**
   在 qwen36 训练隔离栈里，用多模态类加载 27B、喂 `(input_ids, pixel_values,
   image_grid_thw)`，确认 forward 出 logits、显存可接受、grad 能回传到 LoRA。
   先**离线脚本**（`scripts/wip/`），不接 server。

3. **M2 数据层打通**
   实现 `ModelInput.to_multimodal()`，端到端把一条带 `ImageChunk` 的 datum 从
   client 序列化 → server 反序列化 → trainer 取出原图，断言序/对齐正确。

4. **M3 trainer 落地**
   `Qwen36VLTrainingWorker` 接 `VerlTrainingEngine`，跑通 forward_backward +
   optim_step + save_lora_weights，loss 下降。

5. **M4 inferencer 落地**
   `qwen35_mm_vllm_adapter` + `multinode_inference` 分流，采样带图请求出文本。

6. **M5 闭环**
   text+image 的 RL/SFT 小规模冒烟（仿 `RLcheck.sh`，模型换 `-VL`，数据带图）。

---

## 5. 风险与已知坑

- **M-RoPE**：text adapter 主动删了它（`qwen35_text_vllm_adapter.py:314`）。mm 路径
  必须保留，且 vLLM 与 transformers 两侧的位置编码要一致，否则图文对齐错乱。
- **隔离栈 ABI**：Qwen3.6 用独立 torch/vllm/transformers/flash_attn
  （`multinode_inference.py:2777` 起对 flash_attn 的过滤）。mm 多模态算子（vision
  attention、图像 patch）可能触发不同的 flash_attn/算子路径，需在 M0/M1 验证。
- **显存**：vision tower + 图像 activation 叠加，`num_gpus=2` 可能不够。
  **按内存模型计算，不要靠缩小配置试错**（项目铁律：见 CLAUDE.md "Configuration
  Debugging Principle"）。
- **权重命名前缀**：text trainer 剥了 `language_model.`；mm trainer 不能剥，
  save/load LoRA 与 vLLM 加载侧的 key 命名要对齐一致。
- **不要 fallback**：mm 路径任何前置条件不满足（无 vision_config、找不到 mm 架构类）
  一律 raise，禁止静默退回 text 路径。

---

## 6. 不改动的东西（明确边界）

- 现有 `Qwen/Qwen3.6-27B`（text-only）条目、`qwen35_text_vllm_adapter`、
  `Qwen36TrainingWorker`：**全部保留不动**，RLcheck 现状不受影响。
- 损失函数 / RL 算法（cross_entropy / importance_sampling / ppo）：不动。
- runtime_env / tier 组装逻辑：不动（VLM 复用 gpu_rl + qwen36 隔离栈，
  暂不需要新 tier）。
- video 模态：本期不做，只做 image；类型层 `video` 预留但不实现。
