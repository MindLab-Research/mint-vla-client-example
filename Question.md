# PI05 端到端冒烟 —— act 阶段 500 问题排查记录

> 临时文档。用于记录当前卡点、已排除的方向、可能的根因和历史操作。
> 问题解决后删除。

---

## ✅ 已解决(2026-07-01 07:50)

**根因**:act 走 `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1` direct 分支时,
`async_ensure_pending` 建了一个裸 future(`domain_key="future:default"`,
无 scheduler 元数据)。future-namespace 与 model-work 共用**同一个
TaskStateStore**(dev 未设 `MINT_FUTURE_STATE_STORE_DB_PATH`),于是
`model_work_scheduler` 的 reaper loop(每 10s,`reap_lost_pending_tasks`)把这个
裸 pending future 当成"丢失的 model work"**收编**,assign→claim→finalize,
在 16s 推理窗口内把 future 提前终态化 → act 协程 `async_resolve` 撞
`CANNOT_STAGE_FOR_TERMINAL` → except 里 `async_fail` 撞
`TERMINAL_COMMIT_PAYLOAD_MISMATCH` → 500。

**这是竞争,不是"必现"**:live 日志时间线证明,reaper tick 落在 act 窗口
早段(4–6.5s)→ 500;落在晚段(8–10s)→ 200。每个 act 500 前都紧跟一条
`[model_work_scheduler] recovered lost pending tasks ... reason=scheduler_reaper_requeue`。

**下面第四节排除方向 #8 是错的**:grep `expire_active_tasks` 得出"无调用点",
但真正收编发生在 `reap_lost_pending_tasks`(经 `async_expire_active_tasks`
包装名),调用点在 `maintenance_cron_actor.py:82` + scheduler reaper loop。
查调用点必须连 async wrapper 名一起 grep。

**修复**:`model_work_scheduler.py` 加 `_is_scheduler_owned_record()`——只认带
`metadata["model_work_scheduler_append_attempt_id"]`(scheduler `create_task`
时盖的戳)的记录为自有 model work,在 hydrate 和 reaper 两处 scan 各加一行
`if not self._is_scheduler_owned_record(record): continue`。跨重启的真 model
work 总有此戳,仍正常 hydrate/reap。已同步 server 树 + clean_restart 生效。

**验证(07:48 重启后,按时间戳过滤)**:0 个 act 500 / 0 次 reaper 收编 /
15 个 act 200;单次干净 run `train OK loss=0.0024 + act OK n=70`。
(叠跑时偶见的 `train FAIL: actor died / ray.kill` 是共享 actor 生命周期竞争,
见第七节之外的 memory,与本 act 500 无关。)

以下为解决前的历史排查记录,保留备查。

---

## 一、当前状态(TL;DR)

PI05(`openpi/pi05-libero-low-mem-finetune`,pi0.5 flow-matching VLA)端到端冒烟,
经过多轮修复后,**前 4 步稳定通过,卡在最后的 `act` 推理返回 500**。

| 阶段 | 状态 |
|------|------|
| create_model | ✅ 200 |
| train_step(flow_matching) | ✅ 200(有真实 metrics:loss≈0.0024, grad_norm≈0.35, lr=1e-4) |
| save_weights_for_sampler | ✅ 200(checkpoint 完整) |
| action_sessions(建会话) | ✅ 200 |
| **act(推理)** | ❌ **500,稳定必现** |
| delete(清理) | ✅ 200 |

**重要修正**:一度以为"干净重启后单跑必过",是误判。实测**稳定必现 500**,
之前偶尔"通过"未经严格核对,不足为凭。

---

## 二、act 500 的真实根因(已锁死到这一层)

### 真实异常位置
关键:**真实堆栈只在 `/tmp/mint_dev_launch_wenxi.log`**(launcher 捕获的 server stderr),
结构化 server 日志和 Ray worker 日志里都**没有**。这是前期一直找不到根因的原因。

### 完整因果链
```
act 推理本身成功(跑满 ~16s,checkpoint 完整)
  ↓
async_resolve 写回结果 → stage_payload 阶段
  ↓
TaskStateConflictError: "cannot stage payload for terminal task"
  (reason = CANNOT_STAGE_FOR_TERMINAL)
  ← 说明该 act 任务在写回之前就已经是「终态(terminal)」了
  ↓
落入 except 块 (mint.py:560) → async_fail(request_id, ...)
  ↓
TaskStateConflictError: "terminal task commit payload mismatch"
  (reason = TERMINAL_COMMIT_PAYLOAD_MISMATCH)
  ↓
HTTP 500
```

一句话:**act 的 task future 在推理结果写回之前,就被某条路径提前「终态化」了**,
导致 resolve 时状态冲突。这是**控制面(task state machine)问题,不是模型/推理/环境问题**。

### 相关代码位置
- `mint_server/routes/mint.py:528` —— act 的 DIRECT_RUNTIME 分支入口
- `mint_server/routes/mint.py:533` —— `async_ensure_pending(request_id)`(建 pending)
- `mint_server/routes/mint.py:540` —— `action_session_manager.act(...)`(16s 推理)
- `mint_server/routes/mint.py:550` —— `async_resolve(...)`(此处抛第一个冲突)
- `mint_server/routes/mint.py:560` —— except 里 `async_fail(...)`(此处抛第二个冲突)
- `mint_server/routes/mint.py:520` —— `request_id = f"act_{uuid.uuid4().hex}"`(全新 uuid)
- `mint_server/backend/stores/task_state_store.py:2160` —— TERMINAL_COMMIT_PAYLOAD_MISMATCH
- `task_state_store.py:stage_payload` —— CANNOT_STAGE_FOR_TERMINAL
- `task_state_store.py:5084` —— `async_resolve` 定义
- `task_state_store.py:5052` —— `async_ensure_pending` 定义

---

## 三、DIRECT_RUNTIME act 路径的可疑设计

`MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1` 当前是**开着的**(server 进程环境确认)。
这条 direct 分支流程:

```
async_ensure_pending(request_id)   # 建 pending 记录
  → act()                          # 16s 推理,期间任务【一直停在 pending,从不 mark_running】
  → async_resolve(request_id)      # 此时任务已 terminal → 冲突
```

**可疑点**:任务在整个 16s 推理期间停留在 `pending`,直到推理结束才 resolve。
某处在这中间把它终态化了。

---

## 四、已经排除的方向(证伪的假设)

按时间顺序,这些假设都查过并被证伪:

1. ❌ **orbax 兼容 / worker module 问题** —— 早期修复过,现在不影响 act。
2. ❌ **gcsfs / tokenizer 下载失败** —— 由 OPENPI_DATA_HOME 兜底解决(见修复清单)。
3. ❌ **owner_id 缺失导致 400** —— driver 透传 owner_id 已解决 action_sessions 创建。
4. ❌ **多轮叠跑竞争(delete_model kill 共享 actor)** —— 单次干净 run 也必现,证伪。
5. ❌ **placement reaper 误杀 OpenPI actor** —— act 500 不是 actor died,是 task state 冲突。
6. ❌ **act() 自身双重完成 task** —— act() 不碰 request_id / task_futures,只返回推理 dict。
7. ❌ **ensure_task 复用旧 terminal 记录** —— request_id 是全新 uuid4,走新建 pending 分支。
8. ❌ **后台 reaper `expire_active_tasks` 把 pending 任务过期** ——
   全代码库搜索:`expire_active_tasks` **只有定义、没有任何调用点**,不会触发。
   且本次 run 日志里无 expire/reaper 事件。
9. ❌ **scheduler split-brain(两个 scheduler 抢 owner)** ——
   当前只有 1 个 `mint_model_work_scheduler` + 1 个 `mint_task_state_store` 存活,
   本次 run 无 scheduler 冲突刷新。
10. ❌ **JSON 结果文件误导** —— `/tmp/pi05_check_result.json` 只在成功时写,
    失败时是旧数据。曾被此坑误导过一次(把旧 run 的 "train actor died" 当成本次)。

---

## 五、当前卡点(未锁死)

**到底是谁在那 16 秒里,把这个全新 pending 任务变成了 terminal?**

三个最合理的嫌疑(act 自身、ensure_task 复用、后台 reaper)都被证伪。
任务确实在 stage 前变成 terminal,但终态化路径尚未找到。

**尚未排查完的方向**:
- `async_resolve` 里的 `resolve_with_payload` 分支(task_state_store.py:5100-5111)——
  可能是「第一条完成路径」,完成后代码却继续 fall-through 又 stage 一次。
- `_buffer_model_work_finalize` 缓冲机制 —— direct 模式下是否有 buffer 未 flush。
- DIRECT_RUNTIME 路径与正常 queued(scheduler)路径是否存在隐藏的双写。

---

## 六、下一步建议(二分实验)

**关掉 `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME`,让 act 走正常 queued(scheduler)路径跑一次。**

一次实验砍一半范围:
- 若 queued 路径能通 → 坐实是 **direct 路径的 bug**,且立刻有可用跑法;
- 若 queued 路径也 500 → 问题在更底层的 task_state_store,继续往下挖。

---

## 七、已确认的修复(应保留 / 待提交)

这些是让链路从最初的崩溃推进到现在的实质修复:

1. **`OPENPI_DATA_HOME` 兜底**
   - 文件:`mint_server/backend/openpi/openpi_ray_runtime.py` 的 `_openpi_runtime_env_vars()`
   - 作用:actor 拿不到 `OPENPI_DATA_HOME` 时,从 weights 路径推导默认值
     (`<weights>/../.. = /vePFS-Mindverse/share/models/openpi`),tokenizer 命中缓存。
   - 根因:Ray runtime_env 注入的 env 依赖构建进程的 `os.environ`,
     而构建 worker runtime_env 的控制面 actor 跨重启存活,`os.environ` 停留在旧集合
     (没有后加的 `OPENPI_DATA_HOME`)→ actor fallback 到默认 cache → 触发 gs:// 下载 → gcsfs 缺失。
   - 已同步到 share/code + git repo 两份。

2. **driver owner_id 透传**
   - 文件:`scripts/wip/openpi_vla_smoke.py:142`
   - 作用:action_sessions 请求带上 `save_result.get("owner_id")`,
     解决 admin checkpoint 引用缺 owner_id 的 400。
   - `MintCreateActionSessionRequest` 有合法的 `owner_id` 字段。
   - 已同步 share/code + repo。

3. **`step2_clean_restart.sh`**
   - 作用:清 `~/.cache/openpi` 残留软链 → 停 server → `ray.kill` 控制面 detached actor
     + OpenPI worker actor → 重启 → healthz。
   - 根因:mint dev server 依赖一批 detached Ray actor(控制面 + OpenPI worker),
     不随 server 进程 TERM 退出;不清理直接重启会复用旧 actor(过期 scheduler epoch /
     缺 env)。

---

## 八、可靠跑法(已记入 memory)

```bash
cd /vePFS-Mindverse/user/intern/wenxi/mint

# 1) 干净重启(每次跑前必做)
bash step2_clean_restart.sh

# 2) 等 scheduler lease 收敛(约 20 秒)
sleep 20

# 3) 删旧结果,避免读到陈旧 JSON
rm -f /tmp/pi05_check_result.json

# 4) 跑冒烟
bash PI05check.sh
```

### 严格核对真通过(不要只信 "OK" 那行)
PI05check.sh 的 "OK" 判定不严谨(只看 save_result.path,不检查 train_result),
结果文件又只在成功时写。核对:

```bash
python3 -c "
import json; d=json.load(open('/tmp/pi05_check_result.json'))
tr=d.get('train_result') or {}
print('train:', 'FAIL' if set(tr)<={'error','category'} else ('OK loss='+str((tr.get('metrics') or {}).get('loss:mean'))))
ar=d.get('action_result') or {}; a=ar.get('actions'); a=a.get('data') if isinstance(a,dict) else a
print('act  :', ('OK n='+str(len(a))) if a else 'FAIL')
"
# 真通过应看到 train: OK loss=0.00xx + act: OK n=70
```

同时用 `stat -c %y /tmp/pi05_check_result.json` 核对写入时间,别被旧数据误导。

---

## 九、关键排查技巧(踩坑总结)

1. **act 500 的真异常在 launcher 日志,不在结构化日志**:
   `/tmp/mint_dev_launch_wenxi.log`,不是 server 结构化 log,也不是 Ray worker log。
2. **结果 JSON 只在成功时写**:失败后读到的是上一次成功的旧数据。务必核对写入时间。
3. **多轮叠跑污染日志**:控制面日志跨 run 追加,worker 日志跨 run 交织。
   排查前记录日志行数作为「分界线」,跑完只切分界线之后的部分。
4. **PI05check.sh 判定不可靠**:只看 save,不看 train。train 崩了也可能报 "OK"。
5. **runtime_env env 注入依赖构建进程的 os.environ**:
   控制面 actor 跨重启存活,新加的 env 变量不会自动进入它构建的 worker runtime_env。

---

## 十、环境要点

- server 端口:30496(namespace hash 派生,namespace = `mint_wenxi_dev`)
- client python:`/vePFS-Mindverse/user/intern/wenxi/mint/.venv-mindlab/bin/python`(提供 requests)
- code root:`/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi`
- driver:`{code_root}/scripts/wip/openpi_vla_smoke.py`
- 关键 env(server 进程已确认):
  - `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1`
  - `OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi`
  - `MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params`
  - `MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR=/vePFS-Mindverse/share/mint/dev/data/wenxi/openpi-pi05-checkpoints`
  - `MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets`
- Ray:单机 8 卡,GCS 127.0.0.1:6379 / 192.168.42.227:6379
