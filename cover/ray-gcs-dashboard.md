# Ray GCS 指标看板与排障说明

## 目标

这份文档说明如何利用 Ray GCS 指标观察 MinT 里“偶发 GCS 卡死 / 变慢”的问题。

当前服务侧已经补了两类接口：

- `GET /internal/metrics`
  - Prometheus 文本接口。
  - 会把一批 Ray head 上的 GCS 相关指标桥接出来，现有的 MinT 监控抓取链路可以直接复用。
- `GET /internal/ray_gcs_metrics`
  - JSON 快照接口。
  - 适合人工排查，用来看这次桥接到底抓到了哪个 Ray exporter、抓到了多少条样本、有没有抓取失败。

桥接时保留了 Ray 原始指标名和原始标签。也就是说，Grafana / PromQL 可以直接用下面列出来的名字。

## 当前桥接的指标

已经桥接的 Ray GCS / 控制面指标：

- `gcs_task_manager_task_events_reported`
- `gcs_task_manager_task_events_stored`
- `gcs_task_manager_task_events_dropped`
- `gcs_storage_operation_count`
- `gcs_storage_operation_latency_ms_*`
- `gcs_placement_group_count`
- `gcs_placement_group_creation_latency_ms_*`
- `gcs_placement_group_scheduling_latency_ms_*`
- `gcs_actors_count`
- `grpc_server_req_new`
- `grpc_server_req_handling`
- `grpc_server_req_succeeded`
- `grpc_server_req_failed`
- `grpc_server_req_process_time_ms_*`
- `health_check_rpc_latency_ms_*`

桥接自身的健康指标：

- `tinker_ray_gcs_metrics_bridge_up`
- `tinker_ray_gcs_metrics_bridge_scrape_error_count`
- `tinker_ray_gcs_metrics_bridge_sample_count`
- `tinker_ray_gcs_metrics_bridge_scrape_latency_ms`
- `tinker_ray_gcs_metrics_bridge_cache_age_s`

桥接出来的派生指标：

- `tinker_ray_gcs_gcs_task_manager_task_events_drop_ratio`
- `tinker_ray_gcs_gcs_task_manager_task_events_store_ratio`
- `tinker_ray_gcs_gcs_storage_operation_latency_ms_mean`
- `tinker_ray_gcs_gcs_placement_group_creation_latency_ms_mean`
- `tinker_ray_gcs_gcs_placement_group_scheduling_latency_ms_mean`
- `tinker_ray_gcs_grpc_server_req_process_time_ms_mean`
- `tinker_ray_gcs_health_check_rpc_latency_ms_mean`

## Dashboard 应该怎么搭

建议做成 5 行。

### 1. Task Event 压力

面板：

- `reported` 速率
```promql
rate(gcs_task_manager_task_events_reported{Component="gcs_server"}[5m])
```

- `stored` 速率
```promql
rate(gcs_task_manager_task_events_stored{Component="gcs_server"}[5m])
```

- `dropped` 速率
```promql
rate(gcs_task_manager_task_events_dropped{Component="gcs_server"}[5m])
```

- `drop ratio`
```promql
tinker_ray_gcs_gcs_task_manager_task_events_drop_ratio
```

怎么看：

- `reported` 很高但 `stored` 跟不上，说明 GCS 已经开始吃力。
- `dropped` 持续增长，说明不是“有点慢”，而是已经在丢 task events。
- 这类情况如果和高频 `/retrieve_future` 轮询同时出现，基本可以怀疑控制面被 FutureStore 状态查询打爆。

### 2. GCS Storage 延迟

面板：

- 平均 storage op 延迟
```promql
rate(gcs_storage_operation_latency_ms_sum{Component="gcs_server"}[5m])
/
rate(gcs_storage_operation_latency_ms_count{Component="gcs_server"}[5m])
```

- storage op 速率
```promql
rate(gcs_storage_operation_count{Component="gcs_server"}[5m])
```

怎么看：

- `count` 升高但 latency 稳定，通常只是负载高。
- `count` 升高同时 latency 上升，说明 GCS 元数据路径已经开始拥塞。
- 这是“用户还没大量报错，但 GCS 已经开始变慢”的最好信号之一。

### 3. Placement Group 压力

面板：

- PG 总数
```promql
gcs_placement_group_count{Component="gcs_server"}
```

- PG create 平均延迟
```promql
rate(gcs_placement_group_creation_latency_ms_sum{Component="gcs_server"}[5m])
/
rate(gcs_placement_group_creation_latency_ms_count{Component="gcs_server"}[5m])
```

- PG scheduling 平均延迟
```promql
rate(gcs_placement_group_scheduling_latency_ms_sum{Component="gcs_server"}[5m])
/
rate(gcs_placement_group_scheduling_latency_ms_count{Component="gcs_server"}[5m])
```

- PG scheduling p95
```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(gcs_placement_group_scheduling_latency_ms_bucket{Component="gcs_server"}[5m])
  )
)
```

怎么看：

- PG 数量高，不一定有问题。
- PG 数量高并且 scheduling latency 高，才说明控制面已经被 PG 元数据拖慢。
- 如果这个时间点 `/api/v1/healthz` 也跟着慢，往往是因为 healthz 正在扫 PG，而 GCS 正好拥塞。

### 4. GCS gRPC 请求负载

面板：

- 每个方法当前正在处理的请求数
```promql
sum by (grpc_server_method) (
  grpc_server_req_handling{Component="gcs_server"}
)
```

- 每个方法 / 状态的失败数
```promql
sum by (grpc_server_method, grpc_server_status) (
  increase(grpc_server_req_failed{Component="gcs_server"}[5m])
)
```

- 每个方法的平均处理延迟
```promql
sum by (grpc_server_method) (
  rate(grpc_server_req_process_time_ms_sum{Component="gcs_server"}[5m])
)
/
sum by (grpc_server_method) (
  rate(grpc_server_req_process_time_ms_count{Component="gcs_server"}[5m])
)
```

- 每个方法的 p95 处理延迟
```promql
histogram_quantile(
  0.95,
  sum by (le, grpc_server_method) (
    rate(grpc_server_req_process_time_ms_bucket{Component="gcs_server"}[5m])
  )
)
```

怎么看：

- `grpc_server_req_handling` 上升，说明请求在 GCS 里开始堆积。
- 如果只有某个方法特别高，说明热点就在那个方法上。
- 在这个仓库里，最值得怀疑的热点一般是 placement-group 查询、task-event 路径、以及高频状态查询相关 RPC。

### 5. 症状关联

面板：

- health check RPC 平均延迟
```promql
rate(health_check_rpc_latency_ms_sum{Component="gcs_server"}[5m])
/
rate(health_check_rpc_latency_ms_count{Component="gcs_server"}[5m])
```

- bridge 本身是否正常
```promql
tinker_ray_gcs_metrics_bridge_up
```

- MinT 内部的 Ray 集群探针延迟
```promql
tinker_ray_cluster_probe_latency_ms{probe="placement_groups"}
```

- 因 heartbeat 丢失而死掉的节点数
```promql
tinker_ray_cluster_dead_nodes_missing_heartbeats
```

怎么看：

- 如果 `health_check_rpc_latency_ms` 先抬头，再出现 `healthz` 慢，说明健康检查变慢只是表象，根因在 GCS。
- 如果 bridge 本身掉了，不要因为 dashboard 没线就直接断言 “GCS 恢复了”。
- 如果 GCS 延迟抬头，同时 heartbeat-loss 也开始出现，说明问题已经从“控制面慢”扩散到节点存活性。

## 发生 incident 时先看什么

建议按这个顺序看：

1. `gcs_task_manager_task_events_dropped`
2. `grpc_server_req_handling`
3. `gcs_storage_operation_latency_ms`
4. `gcs_placement_group_scheduling_latency_ms`
5. `health_check_rpc_latency_ms`

这个顺序有意义：

- `dropped` 是最硬的早期信号之一
- `handling` 表示 GCS 当前是否在排队
- storage latency 说明元数据路径是否已经慢了
- PG latency 说明调度路径是否开始受影响
- health check latency 说明用户侧症状是否已经出现

## 告警建议

先用简单阈值，后面再根据真实数据收紧。

- task events 出现 drop
```promql
increase(gcs_task_manager_task_events_dropped{Component="gcs_server"}[5m]) > 0
```

- GCS storage 平均延迟抬高
```promql
(
  rate(gcs_storage_operation_latency_ms_sum{Component="gcs_server"}[5m])
  /
  rate(gcs_storage_operation_latency_ms_count{Component="gcs_server"}[5m])
) > 100
```

- GCS 正在处理中的请求积压
```promql
sum(grpc_server_req_handling{Component="gcs_server"}) > 50
```

- PG scheduling 平均延迟抬高
```promql
(
  rate(gcs_placement_group_scheduling_latency_ms_sum{Component="gcs_server"}[5m])
  /
  rate(gcs_placement_group_scheduling_latency_ms_count{Component="gcs_server"}[5m])
) > 1000
```

- bridge 自己坏了
```promql
tinker_ray_gcs_metrics_bridge_up == 0
```

## 人工排查流程

当系统表现为“卡住了”或者“偶发很慢”时，建议这样查：

1. 先看 `GET /internal/ray_gcs_metrics`
   - `up` 是否为 `true`
   - `sources_with_metrics` 是否为空
   - `scrape_error_count` 是否大于 0

2. 再看 task-event 压力
   - `gcs_task_manager_task_events_reported`
   - `gcs_task_manager_task_events_dropped`

3. 再看当前 backlog
   - `grpc_server_req_handling`
   - 按 `grpc_server_method` 拆分

4. 再看 PG 是否是热点
   - `gcs_placement_group_scheduling_latency_ms`
   - `gcs_placement_group_count`

5. 最后和 MinT 侧症状对齐
   - `tinker_ray_cluster_probe_latency_ms{probe="placement_groups"}`
   - `tinker_ray_cluster_dead_nodes_missing_heartbeats`
   - 客户端的 `/api/v1/healthz`、`/retrieve_future`、长尾 timeout

## 一个重要前提

这些指标是从 Ray head exporter 桥接到 MinT 的 `/internal/metrics` 里的。

这很适合“当前监控系统只会抓 MinT，不会直接抓 Ray 节点”的情况。但也要记住：

- bridge 是观测层，不是指标源头本身
- dashboard 和告警可以基于 bridge 做
- 如果 bridge 没数据，最终还是要回到 Ray head exporter 本身确认

## Grafana 导入文件

已经生成了一份可导入的 Grafana dashboard JSON：

- `cover/ray-gcs-dashboard.json`

导入后需要把 Prometheus datasource 绑定到 `DS_PROMETHEUS`。
