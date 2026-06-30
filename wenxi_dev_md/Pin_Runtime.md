# Pin_Runtime.md — 把本分支用到的 Python 解释器/runtime 固定到 /vePFS-Mindverse/user/intern/wenxi/mint_env

> **迁移记录（2026-06-29 第二次）**：runtime 最初固定在 `/root/mint`（容器本地
> overlay 盘，易失，且只剩 ~28G）。为持久化，已整体迁移到持久 PFS
> `/vePFS-Mindverse/user/intern/wenxi/mint_env`（180T）。本文所有 `/root/mint`
> 路径已统一改为新位置；迁移做法见文末 §8。

> 目的：当前 `dev-vla-wenxi` 用到的解释器全部指向共享目录
> `/vePFS-Mindverse/share/mint/dev/{uv-home,runtime-builds}`，而这些会在另一条
> develop 分支上大改。为开发期稳定，把**整套 runtime（含解释器 + site-packages）
> copy 一份到 `/vePFS-Mindverse/user/intern/wenxi/mint_env`**，并用 **uv** 管理解释器，之后本分支只用 `/vePFS-Mindverse/user/intern/wenxi/mint_env`
> 下的副本，尽量不再受上游变动影响。
>
> 决策（已与用户对齐）：(1) 整个 runtime 都 copy（cpu+gpu_rl 两 tier）；
> (2) 先装 uv，用 uv 管理解释器。

---

## 0. 现状（全部实测，2026-06-29）

本机 = `di-20260629153014-bgkb4` / `192.168.42.227`，8× A800-80GB（就是单机部署目标机）。

```
runtime/{cpu,gpu_rl}                      ── symlink ──► runtime-builds/{cpu-v7-py313,gpu_rl-v7-py313}  (实体)
runtime-builds/*/base-python/bin/python3.13 ─┐
runtime-builds/*/host-venv/bin/python3.13    ├─ symlink ─► uv-home/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13
                                            ─┘                    （uv standalone CPython，104M，自包含）
```

- **真·解释器只有一个**：uv-home 里的 standalone CPython 3.13.14（uv 托管，目录自包含，内部 symlink 全相对）。
- **site-packages 各 tier 独立**：ray/torch/vllm 等在 `runtime-builds/<tier>/site-packages`，靠 `PYTHONPATH` / `.pth` 注入，不在解释器里。
- 体量：cpu tier **6.0G**，gpu_rl tier **57G**，合计 **63G**。`/root` overlay 96G（已用 2G）→ 装得下。
- 版本一致性（命门）：cpu 与 gpu_rl 两 tier 均 `ray 2.51.1 / torch 2.12.0+cu130 / py 3.13.14`。
- uv 二进制机器上原本没有；已用现有 standalone python 的 pip + 阿里云镜像装到
  `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-bin-pkg/bin/uv`（**uv 0.11.25**，已验证可运行）。
- uv 能精确复现 `cpython-3.13.14-linux-x86_64-gnu`（`uv python list --managed-python` 已列出同 build）。

### 需要重写的 symlink — 全清点（只有 13 个）

全部在两 tier 的 `base-python/bin/` 和 `host-venv/bin/` 下：

| tier | 路径 | 当前指向 | 类别 |
|---|---|---|---|
| cpu | base-python/bin/python3.13 | share/uv-home/.../python3.13 | uv-home |
| cpu | base-python/bin/{python,python3,python3.12} | /root/miniconda3/envs/mint-py313/... | conda（脏） |
| cpu | host-venv/bin/{python,python3,python3.13} | share/uv-home/.../python3.13 | uv-home |
| gpu_rl | base-python/bin/{python,python3,python3.13} | share/uv-home/.../python3.13 | uv-home |
| gpu_rl | host-venv/bin/python3.13 | share/uv-home/.../python3.13 | uv-home |
| gpu_rl | host-venv/bin/{python,python3} | /root/miniconda3/envs/mint-py313/... | conda（脏） |

---

## 1. 关键陷阱（为什么不能只 `--copy-from`）

`build_runtime_env.py` 的 `copy_runtime_env()` 用 `shutil.copytree(symlinks=True)`，
**绝对路径 symlink 原样保留**；`_rewrite_copied_runtime_metadata()` 只改
`manifest.json` / `.pth` / `activate_runtime_env.sh`，**不碰 base-python/host-venv
里的解释器 symlink**。

→ 若只 `--copy-from`，copy 出来的 `base-python/bin/python3.13` 仍指回
`share/uv-home`。**解释器没固定住**，上游一改就坏。必须额外做 symlink 重写。

---

## 2. 目标布局（/vePFS-Mindverse/user/intern/wenxi/mint_env）

```
/vePFS-Mindverse/user/intern/wenxi/mint_env/
├── uv-bin-pkg/bin/uv                         # uv 0.11.25（已装）
├── uv-home/                                   # 本分支私有 uv home（新）
│   ├── python/cpython-3.13.14-linux-x86_64-gnu/   # uv 重装的解释器（104M）
│   ├── cache/
│   └── env.sh                                 # 仿 share 版，指向 /vePFS-Mindverse/user/intern/wenxi/mint_env
└── runtime/
    ├── cpu/      → 实体 copy（重写 symlink 后指向 /vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home）
    └── gpu_rl/   → 实体 copy（同上）
```

---

## 3. 执行步骤

> 全程不动 share 原件（只读源）；所有写操作落在 `/vePFS-Mindverse/user/intern/wenxi/mint_env`。绝不 `rm -rf` share。

### Step P0 — uv + 私有 uv-home 解释器（uv 管理）
```bash
export UV=/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-bin-pkg/bin/uv
export UV_PYTHON_INSTALL_DIR=/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python
export UV_CACHE_DIR=/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/cache
# 用 uv 重装与上游同 build 的解释器到私有 uv-home（uv 管理，符合需求）
$UV python install cpython-3.13.14-linux-x86_64-gnu
# 验证
ls /vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13
/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13 -c \
  "import sys;print(sys.version)"
```
写 `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/env.sh`（仿 share 版，UV_HOME/CACHE/INSTALL_DIR/PYTHON_FOR_RUNTIME 全指 /vePFS-Mindverse/user/intern/wenxi/mint_env）。

### Step P1 — copy 两 tier 到 /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime（63G，慢，后台跑）
两条路二选一（执行时定）：
- (a) 复用工具：`build_runtime_env.py --copy-from <src> --mint-env ...`（会顺带 rewrite manifest/.pth）。
- (b) 直接 `cp -a`（保留 symlink），随后统一在 Step P2 重写。
> 倾向 (b)：更可控、可断点续传校验；P2 一次性把 manifest/.pth/symlink 全部本地化。

```bash
mkdir -p /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime
cp -a /vePFS-Mindverse/share/mint/dev/runtime-builds/cpu-v7-py313    /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/cpu
cp -a /vePFS-Mindverse/share/mint/dev/runtime-builds/gpu_rl-v7-py313 /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
```

### Step P2 — 重写副本里的外部引用（新脚本 `scripts/wip/pin_runtime_relink.py`）
1. **13 个解释器 symlink** → 全部重指到
   `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13`
   （conda 的脏链一并修正为同一目标，消除 conda 依赖）。
2. **manifest.json** 里 `env_root` / `host_python` → `/vePFS-Mindverse/user/intern/wenxi/mint_env/...`。
3. **`mint_runtime_env.pth`**（host purelib 里）→ site_packages / pythonpath 指 `/vePFS-Mindverse/user/intern/wenxi/mint_env/...`。
4. **`activate_runtime_env.sh`** → `PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/<tier>`。
5. grep 残留：`grep -rl "share/mint/dev/\(uv-home\|runtime-builds\)\|miniconda3" /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/*/{base-python,host-venv,*.json,*.sh}` 应为空。

### Step P3 — 验证副本自包含
```bash
PY=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/cpu/base-python/bin/python3.13
readlink -f $PY | grep -q /vePFS-Mindverse/user/intern/wenxi/mint_env && echo "OK: 解释器指向本地"
PYTHONPATH=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/cpu/site-packages $PY -c \
  "import ray;print('cpu ray',ray.__version__)"
GPY=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python3.13
PYTHONPATH=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages $GPY -c \
  "import torch;print('gpu_rl torch',torch.__version__)"
# 断网/不依赖 share：确认没有任何路径回指 share
find /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime -type l -lname '*share/mint/dev*' | grep . && echo "BAD: 仍有回指" || echo "OK: 无回指 share"
```

---

## 4. 完成判据
- `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/{cpu,gpu_rl}` 两 tier 实体存在，解释器 symlink 全指
  `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home`，无一回指 share 或 conda。
- 两 tier 分别能 `import ray`(2.51.1) / `import torch`(2.12.0+cu130)，版本与源一致。
- 后续单机部署（Ray_Deployment.md）改用
  `PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime`，与 share 解耦。

## 5. 不做 / 边界
- 不删除、不改写任何 share 下的原件（只读源）。
- 不重装 torch/vllm（铁律：NEVER reinstall torch）——site-packages 整体 copy，不动内容。
- video / 其他 tier 不处理；只 cpu + gpu_rl。

## 6. 进度日志
- 2026-06-29：完成只读探查，定位真·解释器=uv-home standalone CPython 3.13.14，
  清点出 13 个需重写 symlink，装好 uv 0.11.25。写出本固定方案待执行。
- 2026-06-29：**固定全部完成**。详见下方「最终状态」。

---

## 7. 最终状态（2026-06-29 完成，已验证）

### 实际固定的范围（比初版方案更大——探查中发现隐藏依赖）
四个栈，全部 copy 到 `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime` 并重指本地解释器，**全副本零回指 share/conda**：

| 栈 | 路径 | 体量 | 解释器 | 验证 import |
|---|---|---|---|---|
| cpu tier | `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/cpu` | 6.0G | 本地 3.13.14 | ray 2.51.1 |
| gpu_rl tier | `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl` | 57G | 本地 3.13.14 | torch 2.12.0+cu130, ray 2.51.1 |
| qwen36 隔离栈 | `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/qwen36-stack/{qwen36-deps,qwen36-vllm-deps}` | 5.1G | 见下 | torch 2.11.0+cu130, transformers 5.12.1, vllm 0.23.0 |
| sglang overlay | `gpu_rl/sglang-overlays/*-venv`（tier 内） | — | 本地 3.12.13 | （venv python 已重指） |

### 私有 uv-home（两个解释器）
- `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python/cpython-3.13.14-linux-x86_64-gnu`（uv 装，主解释器）
- `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/python/cpython-3.12.13-linux-x86_64-gnu`（uv 装，qwen36/sglang 用）
- `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-bin-pkg/bin/uv`（uv 0.11.25）
- `/vePFS-Mindverse/user/intern/wenxi/mint_env/uv-home/env.sh`（source 即激活私有环境）

### 关键发现 / 陷阱（务必记住）
1. **qwen36-vllm-deps 的 torch 是 cp313**——必须用 gpu_rl 的 **3.13** 解释器跑，
   不是 qwen36-deps 的 3.12。代码（`_import_vllm_async_engine_components`）正是在
   gpu_rl 3.13 进程里把 vllm-deps prepend 到 `sys.path[0]`。qwen36-deps 的 3.12
   主要提供 transformers v5（纯 python）。
2. **代码里有 share 硬编码路径，但都有 env 覆盖**——`MINT_QWEN36_DEPS_PATH` /
   `MINT_QWEN36_VLLM_DEPS_PATH`（默认指 `share/.../runtime/gpu_rl/qwen36-*`）。
   ⚠️ **gpu_rl tier 内的 `qwen36-deps`/`qwen36-vllm-deps` 顶层 symlink 已重指到
   本地 `qwen36-stack` 副本**，所以只要 server 用
   `PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime` 经由 `runtime/gpu_rl/qwen36-*` 访问，
   即走本地副本——但**保险起见，server 启动应显式设**：
   ```
   MINT_QWEN36_DEPS_PATH=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/qwen36-deps
   MINT_QWEN36_VLLM_DEPS_PATH=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/qwen36-vllm-deps
   ```
3. **qwen36-deps/qwen36-vllm-deps 不是标准 venv**（无 pyvenv.cfg），靠代码
   显式 prepend `sys.path`。import torch/vllm 必须额外带 gpu_rl 的
   `site-packages`（packaging 等基础库在那）。
4. torch import 必须从**非含 `torch/` 子目录的 cwd** 运行，否则报
   "loaded torch/_C folder" 误导性错误。

### 重写脚本
`scripts/wip/pin_runtime_relink.py`——幂等，可对任何 `/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime` 副本重跑。
处理：13 个 tier 解释器 symlink + manifest.json + activate_runtime_env.sh +
qwen36 栈内部/顶层 symlink + 通用兜底（任何指向 `mint-runtime-py31213` 的链 → 本地 3.12.13）。

### 后续单机部署（Ray_Deployment.md）接入方式
启动 server / Ray 时用 `PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime`（替代
`share/mint/dev/runtime`），并设上面两个 `MINT_QWEN36_*` env。其余流程不变。

---

## 8. 迁移 /root/mint → mint_env（2026-06-29 第二次）

原因：`/root/mint` 在容器 overlay 盘（易失、~28G 紧张）；迁到持久 PFS
`/vePFS-Mindverse/user/intern/wenxi/mint_env`（180T）。做法：

1. `rsync -a /root/mint/ <mint_env>/`（保留 symlink 原样，~68G）。copy 后副本里
   所有解释器 symlink 仍 dangle 回旧 `/root/mint/uv-home`。
2. `pin_runtime_relink.py` 已参数化：`MINT_RUNTIME_BASE=<mint_env> python
   scripts/wip/pin_runtime_relink.py`。脚本的 `EXTERNAL_MARKERS` /
   `QWEN36_SHARED_PY_MARKERS` / catch-all 都加入了 `/root/mint/uv-home`，
   会把副本里指向旧 base 的链重指到新 BASE 的本地解释器。幂等。
3. 手动收尾（脚本未覆盖的）：`<mint_env>/uv-home/env.sh` 里的 `/root/mint`；
   仓库 `.venv-mindlab` 的 `bin/python*` + `pyvenv.cfg` 重指新 base 的 3.13.14。
4. 部署脚本 step0/2/3 + RLcheck_local 的默认 runtime 路径已改为
   `<mint_env>/runtime`。
5. 验证自包含后，旧 `/root/mint` 可删（释放 overlay）。
