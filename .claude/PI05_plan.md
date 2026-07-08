# 启动 openpi pi0.5 server + 写 PI05check.sh 训练起来

## 背景（已全部实测确认）

目标：在本机 server 上跑通 `openpi/pi05-libero-low-mem-finetune`（pi0.5，flow-matching VLA），
并写一个 `PI05check.sh` 端到端把它训练起来（参考 `RLcheck_local.sh` 的模式）。

代码侧**已经完备**，不需要写训练逻辑：
- model registry 已注册 `openpi/pi05-libero-low-mem-finetune`（`training_backend="openpi_pi05"`,
  flow_action, action_dim=32, action_horizon=10, camera 3 路）。
- 训练链路已存在：`create_model` → `/api/v1/mint/vla/train_step`(loss_fn=flow_matching)
  → `save_weights_for_sampler` → action_session → act。
- driver 脚本已存在：`scripts/wip/openpi_vla_smoke.py` 已支持 `--model openpi/pi05-libero-low-mem-finetune`
  （`_pi05_datum()` 用 actions[10,7]，SFT payload 会自动 pad 到 action_dim=32）。
- `VLA_check.sh` 已存在，但它是**三机 ssh 隧道版**（连 `driver`），本机跑不了。

pi0.5 worker 架构（已确认）：
- 以 `@ray.remote(num_gpus=1)` Ray actor **进程内 import** jax/flax/openpi 运行
  （`OpenPIDirectWorkerClient` 走 `importlib.import_module`，不是 subprocess）。
- actor 的 runtime_env 由 `_openpi_runtime_env_vars()` → `actor_runtime_env_vars(pythonpath=PFS_PYTHONPATH)` 构建。

## 唯一缺口（实测）

py3.13 gpu_rl runtime 里 **openpi 源码在、ml_dtypes 在，但 jax/jaxlib/flax/jaxtyping 缺失**：
```
jax -> <缺失>   jaxlib -> <缺失>   flax -> <缺失>   openpi -> OK
import openpi.models.pi0 → ModuleNotFoundError: No module named 'flax'
```
openpi pyproject 要求：`jax[cuda12]==0.5.3`, `flax==0.10.2`, `jaxtyping==0.2.36`。
- 阿里云镜像里 `jaxlib==0.5.3` 可装（已确认版本列表含 0.5.3）。
- jax 0.5.3 是 cu12 系，配 `/usr/local/cuda/compat`（580 libcuda）在 535 驱动机上能看到 8 卡
  （已用 py3.12 版实测 `jax.devices()` 返回 8× CudaDevice）。

另外两个已知坑（与 qwen36 同源，server 端配置）：
1. compat 软链 `/usr/local/cuda/compat/lib -> /usr/local/cuda/compat`（上次已建，容器本地易失）。
2. openpi actor 的 runtime_env 需带 compat 的 LD_LIBRARY_PATH（jax 也要 580 libcuda）。

## 实施步骤

### 1. 给 py3.13 gpu_rl tier 装 JAX 栈（核心）
用本地 uv 把 openpi 要求的 JAX 依赖装进 gpu_rl tier 的 site-packages：
```
uv pip install --python <gpu_rl py3.13> \
  --target /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages \
  "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36"
```
- 用阿里云镜像（与 Pin_Runtime 一致）。
- 装完用 gpu_rl host python 实测：`LD_LIBRARY_PATH=/usr/local/cuda/compat/lib python -c
  "import jax,flax,openpi.models.pi0; print(jax.devices())"` → 期望 8 卡 + pi0 import OK。
- 风险：jax_cuda12_plugin 自带 cu12 nvidia 库，可能与 gpu_rl 现有 torch cu13 的 nvidia 库
  在 LD 顺序上冲突。先验证 import；若冲突，靠 LD_LIBRARY_PATH 顺序隔离。

### 2. step3_start.sh 增加 openpi/pi05 所需 env
在 `MINT_SUPPORTED_MODELS` 加入 `openpi/pi05-libero-low-mem-finetune`，并新增：
```
MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR=/vePFS-Mindverse/share/models/openpi   # 含 pi05_base/params
MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets   # 含 pi05_libero
MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params  # 显式权重
```
（具体 checkpoint/weights/assets 路径在实施时再核对 worker 解析逻辑，确保 config_name
`pi05_libero` 能定位到 params + assets。）

### 3. openpi actor runtime_env 补 LD_LIBRARY_PATH（compat）
`_openpi_runtime_env_vars()`（`openpi_ray_runtime.py:26`）当前不设 LD_LIBRARY_PATH。
与 qwen36 修复一致，在 extra 里加 `"LD_LIBRARY_PATH": actor_ld_library_path()`，
让 pi05 Ray actor 加载 compat 的 580 libcuda（jax 需要）。改本地 checkout 后 rsync。

### 4. 写 PI05check.sh（仿 RLcheck_local.sh）
本机版（无 ssh 隧道），直指本机 `:30496`，调用已有的 smoke driver：
```bash
MINT_BASE_URL=http://localhost:30496 MINT_API_KEY=tml-dummy \
  <client_py> <CODE_ROOT>/scripts/wip/openpi_vla_smoke.py \
    --model openpi/pi05-libero-low-mem-finetune \
    --output-json /tmp/pi05_check_result.json
```
- client python 用 server runtime（smoke 只用 requests，无需 mindlab venv）。
- 跑完检查 train_result/save_result/action_result 非空、无报错。

### 5. compat 软链固化（防复发）
把 `ln -sfn /usr/local/cuda/compat /usr/local/cuda/compat/lib` 加进 step0（容器重启自动重建）。

### 6. 端到端验证
`bash step3_start.sh` → `bash step4_health.sh` → `./PI05check.sh`，确认 pi05 训练 step 跑通。

## 涉及文件
- 装包：`mint_env/runtime/gpu_rl/site-packages`（jax/flax/jaxtyping，不改代码）
- 改：`mint_server/backend/openpi/openpi_ray_runtime.py`（actor LD_LIBRARY_PATH）+ rsync
- 改：`step3_start.sh`（supported models + pi05 env）
- 改：`step0_ray_up.sh`（compat 软链固化）
- 新增：`PI05check.sh`

## 待确认/风险
- jax cu12 与 gpu_rl torch cu13 的 nvidia 库共存（步骤1验证）。
- pi05 checkpoint/assets 精确路径与 `pi05_libero` config 的匹配（步骤2实施时核对）。
- 是否需要装包前先确认磁盘（gpu_rl 已 57G，jax cu12 约 +3-5G）。
