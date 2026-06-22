---
name: telemetry-query
description: |
  Query MinT production telemetry from VictoriaLogs, VictoriaTraces, and
  VictoriaMetrics via the Victoria MCP endpoints (JSON-RPC 2.0).

  Use for: incident triage from exact error text, request_id, trace_id, endpoint,
  metric name, Grafana/Victoria symptoms, or a narrow time window.

  Triggers: "telemetry", "telemetry query", "victoria", "victoriametrics",
  "victorialogs", "victoriatraces", "grafana", "promql", "logsql", "trace_id",
  "request_id", "metrics", "logs", "mcp"
---

# telemetry-query

通过 Victoria MCP 端点查询 MinT 生产 telemetry（logs / metrics / traces）。

**IMPORTANT: 所有 telemetry 查询操作应在子代理中完成，以保持主会话上下文清洁。**

## 工具选择

- **便捷调试** → `debug_helper.py`：高层封装，输出格式化，覆盖常见排查场景。
- **底层/自定义查询** → `mcp_query.py`：直接调用任意 MCP 工具，输出原始 JSON。

两者都通过同一个 JSON-RPC 2.0 MCP 客户端，自动从同目录 `.env` 读取
`MCP_API_KEY` 和 `MCP_*_URL`。

## MCP 端点

来源：`/vePFS-Mindverse/user/intern/yihang/infra-cluster-iaac/USAGE.zh-CN.md`

- Logs:    `https://otelmcp.macaron.xin/logs/mcp`    (VictoriaLogs, LogsQL)
- Metrics: `https://otelmcp.macaron.xin/metrics/mcp` (VictoriaMetrics, PromQL/MetricsQL)
- Traces:  `https://otelmcp.macaron.xin/traces/mcp`  (VictoriaTraces, Jaeger 风格)
- 认证: Header `x-api-key: <MCP_API_KEY>`

密钥：`MCP_API_KEY` 来自 K8s Secret `monitoring/victorialogs-mcp-api-keys`
（client id `ai-client-1`）。密钥值见 `/vePFS-Mindverse/share/mint/SECRET.md`。

## 协议要点（脚本已封装，调试时备查）

- JSON-RPC 2.0 over Streamable HTTP，单次 POST 请求-响应，**非 SSE 流**。
- 每个端点独立握手、独立 session：`initialize` → 响应 header 返回
  `mcp-session-id` → `notifications/initialized`（HTTP 202）→ `tools/call`。
  session 不可跨端点复用。
- 工具结果在 `result.content[0].text`，通常是**转义的 JSON 字符串**（需二次
  解析），命中 0 条时为空串。错误时 HTTP 仍 200，但 `result.isError: true`。
- **时间格式按端点不同**：logs/metrics 用 RFC3339 或 Unix 秒，traces 用
  **Unix 毫秒整数**。脚本已自动处理。

## Setup

参考同目录 `.env.example` 配置 `.env`（已被 gitignore，勿提交密钥）。

```bash
# .env 内容（密钥值见 /vePFS-Mindverse/share/mint/SECRET.md）
MCP_API_KEY=<从 SECRET.md 获取>
MCP_LOGS_URL=https://otelmcp.macaron.xin/logs/mcp
MCP_METRICS_URL=https://otelmcp.macaron.xin/metrics/mcp
MCP_TRACES_URL=https://otelmcp.macaron.xin/traces/mcp
```

脚本启动时自动 `load_dotenv()`，无需手动 source。

## Hard rules

- 从最窄的锚点开始：`request_id`、`trace_id`、精确错误文本、endpoint、
  service、metric 名、或一个小时间窗。
- 模糊故障先查 logs；已有 `trace_id` 才先查 trace。
- metrics 用于确认影响范围、速率、延迟、队列压力、饱和度，不替代 logs/traces
  定位根因。control-plane 排查优先看 `mint_*` 指标（见下）。
- 证据保持简短：命令、时间窗、稳定 id、命中数、1-3 行证明。
- 绝不打印 credentials、签名 URL、进程环境或密钥配置。
- **数据保留窗口滚动**：logs/traces 只保留最近若干天（实测约 5-7 天，更早的会被
  滚出），数据接近实时。查不到 ≠ 没发生过——0 命中时 `debug_helper` 会自动探测
  数据边界（下钻到小时报告真实起点），区分「事故早于留存」与「查询过窄」。
- 排查历史事故用 `--start/--end` 锁定精确时间窗，不要用大 `--lookback` 把窗口撑大
  （否则事故时刻会被海量新日志淹没，且默认按 recency 排序）。

## 数据形态（排查前必读）

- **两个服务字段集不同**：
  - `service.name="mint"`（主体，数百万条）：有 `_ray_timestamp_ns`、`request_id`、
    `trace_id`、`host.name`、`process.pid`、`service.instance.id`、`mint.cluster_id`、
    `code.file.path`/`code.function.name`、`severity`。
  - `service.name="mint-platform"`（网关层，约数万条）：**没有** `_ray_timestamp_ns`
    （时间只在 RFC3339 的 `_time`），错误形态是 `gateway_upstream_error`，关键字段
    是 `error`、`caller`、`span_id`、`stacktrace`。
- **空值占位是字符串 `"-"`**，不是缺字段；脚本已把 `"-"` 当空处理。
- **severity 真实取值**：`ERROR` / `WARN` / `INFO` / `DEBUG`（注意是 `WARN` 不是
  `WARNING`；`debug_helper --severity WARNING` 会自动归一化为 `WARN`）。
- **traces 有真实 span**：服务为 `mint` / `mint-dev-test`，operation 是真实
  endpoint（`POST /api/v1/create_session`、`save_weights_for_sampler`、
  `POST /oai/api/v1/chat/completions` 等）。span 直接带 `exception.message` /
  `exception.type` 和 `logs` 事件，`processes` 里有 `host.name`/`process.pid`/
  `service.instance.id`。慢请求/异常可直接从 trace 定位。
- **metrics 已有完整 `mint_*` 指标**（200+ 个，control-plane 可观测性已就位）：
  - 健康：`mint_public_healthz_cache_age_seconds`、`mint_public_healthz_refresh_total`、
    `mint_model_actor_supervisor_domain_healthy` / `_domain_unhealthy`
  - 调度/队列：`mint_model_work_scheduler_backlog_depth`、`_replica_queue_depth`、
    `_assigned_total`、`_requeued_total`、`mint_task_futures_queue_timeout_s`
  - 请求延迟：`mint_http_server_request_duration_ms_{bucket,count,sum}`、
    `mint_http_server_requests_total`
  - 另有 `sglang_*`（推理引擎）。用 `mcp_query.py call metrics metrics` 列全部。

## 便捷调试（debug_helper.py）

```bash
# 最近的 ERROR 日志（默认回看 7d）
python .claude/skills/telemetry-query/debug_helper.py recent-errors --lookback 10080 --limit 20

# 按 LogsQL 文本搜索（可加 --severity，WARNING 自动归一化为 WARN）
python .claude/skills/telemetry-query/debug_helper.py find-logs '_msg:"CUDA out of memory"'

# 排查历史事故：用 --start/--end 锁定精确时间窗（RFC3339 / 'YYYY-MM-DD HH:MM' / Unix秒）
python .claude/skills/telemetry-query/debug_helper.py find-logs '<症状关键词>' \
  --start '2026-06-21T04:10:00Z' --end '2026-06-21T04:20:00Z'

# 追 request_id / trace_id 的所有日志（find-request 支持多个 ID，一次 OR 查全）
python .claude/skills/telemetry-query/debug_helper.py find-request <request_id> [<request_id2> ...]
python .claude/skills/telemetry-query/debug_helper.py find-by-trace <trace_id>

# 某服务的日志
python .claude/skills/telemetry-query/debug_helper.py service-logs --service mint --severity ERROR

# traces：列出服务 / 查慢请求 / 取单条 trace
python .claude/skills/telemetry-query/debug_helper.py trace-services
python .claude/skills/telemetry-query/debug_helper.py slow-requests --service mint --min-duration 1000 --lookback 60
python .claude/skills/telemetry-query/debug_helper.py get-trace <trace_id>

# metrics：列出指标族 / 瞬时 PromQL
python .claude/skills/telemetry-query/debug_helper.py metric-inventory
python .claude/skills/telemetry-query/debug_helper.py metric 'up'

# 完整调查：logs(+trace)，--request-id 可重复
python .claude/skills/telemetry-query/debug_helper.py investigate --request-id <id1> --request-id <id2>
python .claude/skills/telemetry-query/debug_helper.py investigate --trace-id <trace_id>
```

输出特点：
- 自动提取关键字段：severity、service、msg、exception.*、error、request_id、
  trace_id、`host.name`、`process.pid`、`service.instance.id`、`mint.cluster_id`、
  `code.file.path`/`code.function.name`。空值（`"-"`）自动跳过。
- 时间统一格式化为可读 UTC（mint 与 mint-platform 两类时间戳都处理）。
- 同 (时间, msg, request_id, severity) 的重复行自动去重。
- **0 命中时给出数据边界提示**，区分「数据已过保留期」与「查询过窄」。
- 通用展示开关：`--fields a,b,c` 自定义展示字段，`--full` 不截断 `_msg`/字段值
  （看完整 stacktrace），`--json` 原始 payload，`--compact` 紧凑 JSON，
  `--verbose` 打印请求 method 到 stderr。

## 底层 MCP 调用（mcp_query.py）

便捷子命令（高频排查动作，已封装时间糖，默认查 logs 端点）：

```bash
# 探测数据落在哪几天（确认 incident 是否在保留期内）
python .claude/skills/telemetry-query/mcp_query.py hits --step 1d
# 默认 query=service.name:"mint"、since=30d，可加 --query/--since/--start/--end

# 发现真实字段名 / 某字段的合法取值
python .claude/skills/telemetry-query/mcp_query.py field-names --query 'service.name:"mint"'
python .claude/skills/telemetry-query/mcp_query.py field-values severity --query 'service.name:"mint"'
```

通用接口：

```bash
# 列出某端点的全部工具（名称 + 必填/可选参数）
python .claude/skills/telemetry-query/mcp_query.py tools logs
python .claude/skills/telemetry-query/mcp_query.py tools metrics
python .claude/skills/telemetry-query/mcp_query.py tools traces
# 加 --raw 看完整 inputSchema

# 调用任意工具，--arg key=value（value 尽量按 JSON 解析）
python .claude/skills/telemetry-query/mcp_query.py call logs query \
  --arg query='severity:ERROR' \
  --arg start='2026-06-16T00:00:00Z' \
  --arg limit=50

python .claude/skills/telemetry-query/mcp_query.py call metrics query \
  --arg query='up'

python .claude/skills/telemetry-query/mcp_query.py call traces traces \
  --arg service=mint --arg limit=20
```

注意 `--arg` 的值会先尝试 `json.loads`：数字/布尔写成 `limit=50`、
`nocache=false` 即可；含特殊字符的字符串用引号包住，如
`--arg query='severity:ERROR'`。

时间类参数（`start`/`end`/`time`/`step` 等）和标识类参数始终按**字符串**发送
——后端拒绝数字时间戳（报 `... has wrong type: float64`）。所以 `--arg start=1749600000`
会被当作字符串 Unix 秒（合法），写 RFC3339 `--arg start='2026-06-11T00:00:00Z'` 也行。
traces 端点的 `start`/`end` 用 **Unix 毫秒**（同样以字符串发送）。

## 各端点常用工具速查（探测自真实端点）

LogsQL（时间 RFC3339/Unix秒，`tenant` 默认 `0:0`）：
- `query`(query, start, [end, limit])：取日志行，结果是 NDJSON。
- `hits`(query, start, [end, step])：按时间桶计数，探测数据分布。
- `field_names` / `field_values` / `streams` / `stats_query`：字段与流分析。

PromQL/MetricsQL（无 tenant 参数）：
- `query`(query, [time])：瞬时。
- `query_range`(query, start, [end, step])：区间。
- `metrics`([match])：列出所有 metric 名。`labels` / `label_values` / `series`。

Traces（**时间 Unix 毫秒整数**，`tenant` 默认 `0:0`）：
- `services`()：列出服务。`service_operations`(service_name)。
- `traces`(service, [operation, start, end, minDuration, maxDuration, limit])。
- `trace`(trace_id)：取单条完整 trace。

完整工具清单随时用 `mcp_query.py tools <endpoint> --raw` 查询。

## Workflow

1. 用精确症状、`request_id` 或 `trace_id` 搜 logs。
2. 提取稳定 id 后转向（pivot），不要反复做宽泛文本搜索。
3. 有 trace id 或需要阶段时序时取 trace。
4. 用 metrics 判断事件是孤立还是系统性。
5. 只记录会改变诊断结论的证据。

## Notes

- LogsQL 用冒号语法（`severity:ERROR`、`service.name:"mint"`、`_msg:"text"`），
  **不是**标签选择器 `{service="mint"}`。
- 编辑 dashboard 前先看实时 label 名；OTel 属性（如 `http.status_code`）经
  Victoria/Grafana 归一化后名称可能不同。
- VictoriaTraces 数据稀疏时，用同一 `trace_id` 回到 logs 交叉验证。

## 参考文件

- `EXAMPLES.md`：真实调试场景示例。
- `.env.example`：环境变量模板。
