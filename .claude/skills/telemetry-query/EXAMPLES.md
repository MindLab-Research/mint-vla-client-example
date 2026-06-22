# telemetry-query 调试示例

所有命令均通过 Victoria MCP 端点（JSON-RPC 2.0），脚本自动从 `.env` 读取
`MCP_API_KEY` 和 `MCP_*_URL`。示例命令均在真实生产端点上验证过。

> 提示：实际排查时把这些查询放在子代理中执行，保持主会话上下文清洁。

## 场景 1：查最近的错误日志

```bash
python .claude/skills/telemetry-query/debug_helper.py recent-errors --limit 20
```

输出已提取 `severity` / `service` / `_msg` / `exception` / `error` / `trace_id` 等关键
字段，时间格式化为 UTC。默认回看 7 天（`--lookback 10080`），排查更早的 incident 用
`--start/--end` 锁定精确时间窗（见场景 4.5）。0 命中时会自动提示数据边界（见场景 8）。

## 场景 2：按关键词搜日志

```bash
# 搜 OOM
python .claude/skills/telemetry-query/debug_helper.py find-logs '_msg:"CUDA out of memory"'

# 限定 ERROR 级别（--severity 接受 WARNING，会归一化为真实值 WARN）
python .claude/skills/telemetry-query/debug_helper.py find-logs 'gateway_upstream_error' --severity ERROR
```

LogsQL 用冒号语法（`severity:ERROR`、`_msg:"text"`），不是标签选择器。

## 场景 3：用 request_id 追踪问题

issue 里常给一组 request_id（如一次失败 run 报出的多个请求）。`find-request`
支持一次传多个 ID，内部 OR 查全：

```bash
# 一次追全部相关 request_id
python .claude/skills/telemetry-query/debug_helper.py find-request \
  <rid1> <rid2> <rid3> <rid4> --full
```

`--full` 不截断 `_msg`，避免长 stacktrace 被切掉。命中后从输出里的 `trace_id`
转去 trace（场景 4），从 `host.name`/`process.pid`/`code.file.path` 定位崩溃的
worker 进程和代码位置。

> 注意：若 incident 早于数据保留窗口（logs/traces 仅保留最近若干天），会 0 命中
> 并提示数据边界——这不是工具坏，是数据已滚出。此时需配合 Ray/actor 侧证据排查。

单个 trace_id 追日志：

```bash
python .claude/skills/telemetry-query/debug_helper.py find-by-trace <trace_id>
```

## 场景 4：日志 → trace 关联（最常用工作流）

1. 先在错误日志里拿到 `trace_id`：

   ```bash
   python .claude/skills/telemetry-query/debug_helper.py recent-errors --limit 5
   # → 例如 trace_id: ea1531733cda6ef75bb0dcea0713d956
   ```


2. 用该 id 一次拉齐 logs + trace：

   ```bash
   python .claude/skills/telemetry-query/debug_helper.py investigate --trace-id ea1531733cda6ef75bb0dcea0713d956 --lookback 10080
   ```

## 场景 4.5：精确历史时间窗 + 症状文本

无 request_id/trace_id、只有"症状文本 + 大致事故时刻"时，用 `--start/--end`
锁定精确窗口，不要用 `--lookback` 把窗口撑大（否则事故那一刻会被大量更新的
日志淹没，且默认按 recency 排序）：

```bash
python .claude/skills/telemetry-query/debug_helper.py find-logs '<症状关键词>' \
  --start '2026-06-21T04:10:00Z' --end '2026-06-21T04:20:00Z'
```

`--start/--end` 接受 RFC3339 / `YYYY-MM-DD HH:MM` / Unix 秒。关键：0 命中时工具会
**下钻到小时**探测数据真实起点，区分两种情况：
- "该精确窗口无数据，但保留库最早数据在更晚的某小时" → 事故早于实际留存，
  telemetry 查不到属正常（注意按天分桶会标 00:00，但当天数据可能几小时后才开始）。
- "该精确窗口有 N 条日志" → 数据存在，是 query 写窄了，放宽重试。

control-plane 类阻塞还要看 metrics（见场景 7 的 `mint_*` 健康/调度/延迟指标）和
Ray/actor 侧，telemetry 只是其中一条线。

## 场景 5：某服务的日志

```bash
python .claude/skills/telemetry-query/debug_helper.py service-logs --service mint --severity ERROR
```

已知服务名：`mint`（主体）、`mint-platform`（网关层，字段集不同：无
`_ray_timestamp_ns`，错误是 `gateway_upstream_error`）。用 `trace-services` 确认当前
列表。

## 场景 6：traces — 服务、慢请求、单条 trace

traces 现有真实 span：服务 `mint` / `mint-dev-test`，operation 是真实 endpoint
（`POST /api/v1/create_session`、`save_weights_for_sampler` 等），span 直接带
`exception.message`/`exception.type` 和 `logs` 事件。

```bash
# 列出当前被追踪的服务
python .claude/skills/telemetry-query/debug_helper.py trace-services

# 查 >1s 的慢请求（traces 时间窗用分钟，脚本内部转 Unix 毫秒）
python .claude/skills/telemetry-query/debug_helper.py slow-requests --service mint --min-duration 1000 --lookback 60

# 取单条完整 trace
python .claude/skills/telemetry-query/debug_helper.py get-trace <trace_id>
```

## 场景 7：metrics 确认影响范围

`mint_*` control-plane 指标已就位（200+），先用 `metric-inventory` 看有哪些族：

```bash
# 按族前缀分组列出指标名（一眼看出 control-plane 覆盖）
python .claude/skills/telemetry-query/debug_helper.py metric-inventory
python .claude/skills/telemetry-query/debug_helper.py metric-inventory --full   # 列全

# 实例存活
python .claude/skills/telemetry-query/debug_helper.py metric 'up'

# control-plane 健康/调度/延迟（按真实 label 名调整）
python .claude/skills/telemetry-query/debug_helper.py metric 'mint_public_healthz_cache_age_seconds'
python .claude/skills/telemetry-query/debug_helper.py metric 'mint_model_work_scheduler_backlog_depth'
python .claude/skills/telemetry-query/debug_helper.py metric 'rate(mint_http_server_requests_total[5m])'
```

常用 `mint_*` 族：健康 `mint_public_healthz_*`/`mint_model_actor_supervisor_domain_*`、
调度队列 `mint_model_work_scheduler_*`、请求延迟 `mint_http_server_request_duration_ms_*`。

## 场景 8：数据边界探测（incident 是否在保留期内）

查不到时，先确认数据本身存不存在，再决定是放宽查询还是放弃 telemetry 路线：

```bash
# 数据落在哪几天（默认 query=service.name:"mint"、回看 30 天）
python .claude/skills/telemetry-query/mcp_query.py hits --step 1d --compact

# 发现真实字段名 / 某字段的合法取值（如 severity 真实值是 WARN 不是 WARNING）
python .claude/skills/telemetry-query/mcp_query.py field-names --query 'service.name:"mint"'
python .claude/skills/telemetry-query/mcp_query.py field-values severity --query 'service.name:"mint"'
```

`debug_helper` 在 0 命中时已自动跑一次 `hits` 给提示，但底层探测可自定义 query/窗口。

## 场景 9：底层任意工具调用

便捷命令不够用时直接调 MCP 工具：

```bash
# 先看某端点有哪些工具
python .claude/skills/telemetry-query/mcp_query.py tools logs
python .claude/skills/telemetry-query/mcp_query.py tools logs --raw   # 看完整 schema

# logs：精确时间窗取日志
python .claude/skills/telemetry-query/mcp_query.py call logs query \
  --arg query='severity:ERROR' \
  --arg start='2026-06-16T00:00:00Z' \
  --arg end='2026-06-17T00:00:00Z' \
  --arg limit=50

# metrics：区间查询
python .claude/skills/telemetry-query/mcp_query.py call metrics query_range \
  --arg query='up' --arg start='2026-06-16T00:00:00Z' --arg step=1h

# traces：某服务的 operation 列表
python .claude/skills/telemetry-query/mcp_query.py call traces service_operations \
  --arg service_name=mint
```

## LogsQL 语法速查

| 目的 | 写法 |
|------|------|
| 字段精确匹配 | `severity:ERROR` |
| 服务过滤 | `service.name:"mint"` |
| 文本搜索 | `_msg:"keyword"` 或裸 `"keyword"` |
| 字段存在 | `request_id:*` |
| 多值 OR | `request_id:("a" OR "b" OR "c")` |
| 组合 | `severity:ERROR AND service.name:"mint"` |
| 排除 | `severity:ERROR AND NOT _msg:"timeout"` |

## 常见问题

- **查询返回空**：先看工具自动给的数据边界提示。若提示「该精确窗口无数据，最早
  数据在更晚的某小时」，说明 incident 早于数据保留窗口（logs/traces 仅保留最近
  若干天），telemetry 查不到属正常；若提示「该窗内有 N 条日志」，则是查询过窄/
  字段值不匹配，放宽 query 重试。
- **severity 查不到**：真实值是 `ERROR`/`WARN`/`INFO`/`DEBUG`（不是 `WARNING`）。
  `find-logs`/`service-logs` 的 `--severity` 会自动归一化，底层 `call` 需自己写对。
- **stacktrace 被截断**：默认 `_msg` 截到 300 字符，加 `--full` 看完整内容。
- **`isError: true`**：缺必填参数（如 logs `query` 必须带 `start`）。用
  `mcp_query.py tools <endpoint>` 看必填项。
- **时间格式报错**：traces 用 **Unix 毫秒整数**，logs/metrics 用 RFC3339 或
  Unix 秒。便捷命令已自动处理，底层 `call` 需自己注意。
