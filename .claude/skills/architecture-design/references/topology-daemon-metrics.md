# Topology-Aware Supervision and Node Metrics DaemonSet

## Goal

Mint needs a topology-aware control plane for two related concerns:

1. Static model-runtime placement should refer to stable worker aliases instead
   of raw node IPs.
2. Node-local observability should run as a per-node daemon actor and push
   metrics through OpenTelemetry.

`ModelActorSupervisor` owns this reconciliation as a detached Ray actor. It is
the single supervisor for long-lived Ray actors that Mint intentionally keeps
alive, but it must keep model-runtime actors and daemon actors as separate
scheduling classes.

`MaintenanceCronActor` is not a reconciler. It owns cron-style jobs such as
future reaping, checkpoint cleanup, and stale-session cleanup. It must not call
`ModelActorSupervisor`, trigger reconcile, or own model/daemon reconciliation
state. `ModelActorSupervisor` may manage the cron actor lifecycle; cron does not
know about the supervisor.

## Non-goals

- Do not add a separate metrics registry actor or metrics supervisor actor.
- Do not make `/internal/metrics` part of the default observability path.
- Do not persist metrics snapshots.
- Do not let metrics collection block request handling.
- Do not make the API server create or destroy cloud worker jobs in V1.

## Topology Layers

### 1. Node Topology

Node topology describes the cluster workers that Mint may target. V1 topology
is an independent YAML file owned by deployment configuration, not a subsection
of model config. The file path is provided by `MINT_TOPOLOGY_CONFIG_PATH`.

- `alias`: stable reusable name in the form `mint-worker-{idx}`, for example
  `mint-worker-0`. The index is deployment-local and must not point at two live
  provider tasks at the same time.
- `node_ip`: Ray node IP.
- `ray_node_id`: optional Ray node ID observed at runtime.
- `provider`: explicit provider discriminator. V1 implementation may only
  operate against the static file, but the schema must support both `volc` and
  `pai`/DLC identities so the topology object does not bake in one platform's
  naming model.
- `provider_identity`: provider-specific stable identity such as Volcano job,
  resource queue, and worker index, or PAI/DLC job and pod identity.
- `role`: usually `gpu`.
- `gpu_count`: expected GPU count.
- `labels`: small bounded labels for selection.
- `enabled`: whether Mint should schedule actors onto the node.
- `mount_ok`: whether required shared paths are expected to exist.
- `runtime_env_ok`: whether the Mint runtime env is expected to import.

Example:

```yaml
version: 1
deployment_env: prod
cluster_id: volcano
state_path: /vePFS-Mindverse/share/mint/prod/runtime/topology_state.yaml
ray:
  head_ip_path: /vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt
providers:
  volcano:
    region: cn-beijing
    credentials:
      mode: default_chain
    templates:
      a800-8gpu-c1:
        template_path: /vePFS-Mindverse/share/mint/prod/mint-server/.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml
        resource_queue_id: q-20251126180002-26lwz
        gpu_count: 8
nodes:
  desired:
    - alias: mint-worker-0
      provider: volcano
      template: a800-8gpu-c1
      role: gpu
      enabled: true
      labels:
        pool: qwen
```

V1 reads desired state from the static config file, then rebuilds runtime state
from the provider and Ray. For Volcano, node management uses the Volcano Engine
ML Platform Python SDK (`volcengine-python-sdk`) from the detached
`mint_model_actor_supervisor` process. The supervisor must run on the trusted
driver/control-plane node; model/runtime actors, daemon actors, API workers, and
ConfigActor must not hold cloud provider credentials.

Volcano SDK credentials use the SDK default credential chain. This can reuse
credentials created by the Volcano CLI, but only when those credentials are
available to the driver process. Current SDKs read `VOLCENGINE_ACCESS_KEY` /
`VOLCENGINE_SECRET_KEY`, `VOLCENGINE_SESSION_TOKEN`, `VOLCENGINE_CLI_CONFIG_FILE`
or `~/.volcengine/config.json`, OIDC, and ECS role metadata. Older ML Platform
SDK docs also document `volc configure` writing `~/.volc/config` and
`~/.volc/credentials` plus `VOLC_ACCESSKEY` / `VOLC_SECRETKEY`; do not rely on
that legacy location unless the installed SDK version is verified to read it.
AK/SK, session tokens, signed requests, and credential file contents must not be
written to topology YAML, ConfigActor snapshots, Ray runtime_env, logs, metrics,
or `topology_state.yaml`.

The SDK provider lists jobs with `list_jobs`, reads worker IPs from
`list_job_instances` (`Ips.PrimaryIp` / `Ips.HostIp`), and submits new workers
with `create_job`. It must not scrape job logs for node IPs. Worker templates
remain YAML for operator readability, but the provider renders the template and
then converts the supported fields into a `CreateJobRequest`; unsupported
template features must fail loudly instead of silently falling back to CLI
submission. Ray liveness is observed from initialized `ray.nodes()` when
available and from the Ray dashboard `/api/v0/nodes` using `ray.dashboard_url`
or `ray.head_ip_path`.

`topology_state.yaml` is output-only debug/runtime state. The supervisor writes
it atomically after each reconcile; it must not be used as startup recovery
input. If a live provider task and alive Ray node already satisfy an alias, V1
must reuse it and must not submit a duplicate worker. V1 is scale-up only and
does not cancel or tear down disabled/removed cloud nodes automatically.

The Ray head node is not a desired worker and must not use the
`mint-worker-{idx}` namespace. During reconcile, the topology manager observes
the current Ray head from `ray.nodes()` and the Ray dashboard. If Ray does not
mark the head explicitly, the node whose IP matches `ray.head_ip_path` is
treated as the head. The runtime state then exposes it as `mint-head` with
`provider: ray` and `role: head`. `mint-head` is output-only runtime state: it
is eligible for the head `NodeMetricsCollectorActor`, but invalid for model
placement and never causes provider worker submission.

Worker creation is keyed by the numeric suffix in `mint-worker-{idx}` and by the
stable provider task name for that alias. A reconcile pass may submit all
missing desired workers in one bounded parallel batch
(`MINT_TOPOLOGY_SUBMIT_CONCURRENCY`, default 8), because the desired set is
static and alias-to-task-name mapping is deterministic. Provider/Ray readiness
is still evaluated per alias: model runtime placement can only resolve aliases
whose provider task has a node IP and whose Ray node is alive. Submit failures
are recorded on the affected alias in `topology_state.yaml`; successful live
tasks are always reused and must not be submitted again.

### 2. Static Actor Placement

Model runtime placement is topology-aware. Preferred configuration references
node aliases, not raw IPs:

```yaml
models:
  Qwen/Qwen3-30B-A3B-Instruct-2507:
    megatron:
      placement:
        - worker: mint-worker-0
          gpu_count: 4
          replica: 0
        - worker: mint-worker-1
          gpu_count: 4
          replica: 0
```

Equivalent normalized placement item:

```yaml
worker: mint-worker-0
gpu_count: 4
replica: 0
```

Accepted topology-aware placement keys:

- `worker` or `worker_alias`: required stable alias.
- `gpu_count`: required for GPU actors.
- `replica`: optional replica number, normalized into `replica_id`.
- `labels`: optional selector metadata for future placement policies.

Placement entries with the same `replica` are merged into one multi-node
runtime replica. Placement entries with distinct `replica` values produce
distinct `ModelActorSpec` entries and distinct scheduler replicas. Topology
shape names such as dense, multinode, or model shape should remain supervisor
metadata rather than actor-name components.

GPU placement entries fail fast if `gpu_count` is missing. A topology placement
item that names a `worker`, `worker_alias`, `node_ip`, or `node_pin` without a
GPU count is invalid because the launchers need the resolved placement slices to
build the per-actor execution contract.

At runtime, the supervisor resolves the alias into `node_ip` and passes only the
resolved placement to Ray launchers. Raw-IP placement is allowed only as an
explicit compatibility input at the config boundary. If the alias already exists
and is an IP address, the supervisor treats that node as pre-existing, does not
create a cloud worker for it, and records the resolved state for debugging.

The normalized internal form is:

```python
ResolvedPlacement(
    worker="mint-worker-0",
    node_ip="192.168.39.110",
    gpu_count=4,
)
```

Current implementation requirements:

- `ModelActorSpec` and placement parsing accept alias fields and normalize raw-IP
  compatibility input at the config boundary.
- `node_placement`, dense launchers, vLLM launchers, and Megatron launchers
  receive resolved node placements and do not parse topology themselves.
- Raw `node_ip` placement remains a compatibility/pre-existing-node input; new
  configs should use `worker` or `worker_alias`.

Static placement still applies to model-runtime actors only. Those actors are
registered with `ModelWorkScheduler` as replicas and may consume scheduler
leases.

### 3. Dynamic Actor Placement

Dynamic placement V1 supports only a DaemonSet policy:

- one actor per eligible node
- eligibility from topology labels and Ray liveness
- required eligibility checks: `enabled`, `role=gpu`, `gpu_count > 0`, alive Ray
  node, shared filesystem path availability, runtime env import probe, and NVML
  access probe
- head/API-only CPU nodes are excluded from model placement, but the observed
  `mint-head` runtime alias is eligible for a head-only node metrics daemon so
  Ray/GCS global gauges can be pushed without adding a separate actor class
- actor pinning with `resources={f"node:{node_ip}": 0.001}`
- `num_gpus=0`

The first daemon actor is `NodeMetricsCollectorActor`.

Node metrics daemon actors are enabled by default. Set
`MINT_NODE_METRICS_DAEMON_ENABLED=0` only for emergency rollback or isolated
tests. When disabled, topology and model placement still reconcile but no
daemon actors are created. Actor names are stable:
`mint_daemon_node_metrics_{worker_alias}`.

Daemon actors are not model replicas. They must not be registered with
`ModelWorkScheduler`, must not have queue IDs, and must not participate in task
claiming.

## ModelActorSupervisor Shape

`ModelActorSupervisor` is a detached actor. External operations bootstrap only
`mint_config`, then `mint_model_actor_supervisor`, then API workers. The
supervisor ensures the remaining CPU control-plane actors and all desired
runtime/daemon actors. Starting the supervisor also starts its owned periodic
reconcile loop; `MINT_ACTOR_RECONCILE_INTERVAL_S` controls the loop interval
and defaults to 5s. External operations may call one bootstrap command that
performs the same ordered steps, but API workers must not be the component that
creates the detached actors.

API processes, route handlers, and backend launchers must access it through a
client facade instead of importing a process-local singleton as an authority.
API clients may read/check supervisor state and may send fire-and-forget nudges,
but they must not ensure, create, or wait for supervisor reconciliation. A nudge
asks the supervisor to run a fast ensure pass for a desired domain. If the
supervisor is already reconciling, the nudge should return immediately without
starting duplicate work. Cron jobs must not call supervisor APIs.

It owns both scheduling classes:

```python
ModelActorSupervisor
  - model_specs: dict[(domain_key, replica_id), ModelActorSpec]
  - daemon_specs: dict[name, DaemonActorSpec]
  - topology: ClusterTopology
  - inventory: ModelActorInventoryState
```

Hard implementation invariants:

- Daemon specs are stored in separate supervisor maps, never in
  `ModelActorSpec` or the model `_desired` map.
- Daemon actors never pass through `_replica_registration_for_state()`,
  scheduler domain hydration, or scheduler sync.
- Daemon state never has `domain_key`, `replica_id`, `queue_id`,
  `consumer_id`, or scheduler capacity.
- Daemon actors do not enter `ModelActorInventory`. That inventory remains
  model-actor-only for GPU accounting, session idle semantics, and
  `/internal/actors` admin behavior.
- Daemon snapshot/admin state lives under the supervisor's separate
  `daemons` snapshot.
- `ModelActorInventory` state is owned by the detached supervisor. API
  processes may cache actor handles, but cached process-local inventory must not
  be treated as authoritative.
- OpenPI runtime actors are model/runtime actors for supervisor purposes. OpenPI
  shared runtime actors are durable model/runtime actors and must publish
  lifecycle, session binding, node placement, and GPU metadata through the
  supervisor contract instead of maintaining an independent runtime registry.
  Per-action-session named actors are not the target durable shape; migrate them
  into shared runtime sessions or keep them short-lived and outside scheduler
  leases.
- Supervisor owns operational/live state such as runtime health, actor handles,
  node placement, session binding, and inflight/protection metadata. User and
  business metadata such as task results, session indexes, heartbeats, and
  gateway routing belongs in `TaskStateStore`.
- Supervisor state storage must support memory-only and SQLite modes. The
  SQLite DB path defaults to
  `/vePFS-Mindverse/share/mint/<env>/runtime/supervisor_state.sqlite3` and may
  be overridden by `MINT_SUPERVISOR_STATE_DB_PATH`. The SQLite store is internal
  to `ModelActorSupervisor` and stores operational state needed for reconcile
  continuity; it is not a business metadata store and it must not make
  `ModelWorkScheduler` depend on `TaskStateStore`.
- `ModelWorkScheduler` may accept work for desired domains before a healthy
  replica exists. Such work remains pending until supervisor registers a healthy
  replica, bounded by request/task TTL. Preserve Tinker async semantics: clients
  call `retrieve_future`; local futures may wait on the server for the bounded
  long-poll timeout and then receive HTTP 408 while the task is still pending.
  Durable pending/result/tombstone TTLs are enforced by the async future reaper;
  scheduler lease TTLs remain separate and only protect ownership recovery.

`reconcile_once()` should run in this order:

1. ensure Supervisor-owned CPU control-plane dependencies
2. hydrate desired domains from active scheduler state
3. refresh/resolve topology
4. reconcile daemon actors
5. reconcile model runtime actors
6. sync model runtime replicas to `ModelWorkScheduler`

Only model runtime actors are synced to the scheduler.

The supervisor does not watch config files. External operations own reload or
restart. After reload/restart, the supervisor still performs reconciliation from
the current config, topology, provider state, Ray live state, and runtime actor
health.

The first supervisor storage implementation should include memory-only and
SQLite modes. On vePFS, prefer conservative SQLite journaling such as DELETE or
TRUNCATE with `synchronous=NORMAL`; do not enable WAL unless explicitly
configured and validated for the deployment filesystem. The SQLite schema should
stay small: `kv`, `owner`, and bounded `events`. `owner` records must include
`owner_id`, `epoch`, `started_at`, `last_heartbeat_at`, `lease_until`, and
`schema_version` so future HA work has a fencing boundary from the first
version.
`ModelWorkScheduler` owns hot scheduling state and must not use
`TaskStateStore` as scheduling authority or lifecycle owner. It may depend on
`TaskStateStore` for durable task admission, lease persistence, indexes, and
recovery.

API health contract:

- `/api/v1/healthz`: external business health. It is cached per API worker for
  30s and checks only `mint_model_work_scheduler` and `mint_task_state_store`.
  Dirty cache refresh is single-flight; concurrent calls wait up to 5s total.
  Underlying scheduler/task-store pings must use shorter timeouts inside that
  budget. Initial no-cache requests use the same limit. A dirty previous value
  is invalid if refresh fails. Responses are only
  `{"status":"ready"}` with HTTP 200 or `{"status":"unhealthy"}` with HTTP 503.
  Do not expose supervisor, cron, actor, Ray, or degraded details here.
- `/api/v1/internal/healthz`: internal operations health. It reads the current
  `mint_model_actor_supervisor` summary snapshot plus process-local
  maintenance-cron/startup degraded markers, returning ready/degraded/unhealthy
  with a deliberately small component summary. It must not fan out to every
  runtime actor on request and must not synthesize scheduler/task/topology/reaper
  inventories. Missing `mint_model_actor_supervisor` is unhealthy. A supervisor
  snapshot whose `snapshot_generated_at`, `observed_at`, or topology
  `observed_at` timestamp is older than 60s is degraded; do not use
  `last_reconcile_at` as the snapshot-staleness clock. Missing or degraded
  `mint_maintenance_cron` is degraded.

Health metrics:

- `mint_public_healthz_cache_age_seconds`
- `mint_public_healthz_refresh_total{result="ready|unhealthy|timeout|error"}`

The supervisor snapshot should expose daemon state separately:

```json
{
  "replicas": {"vllm:...::replica-0": {"state": "healthy"}},
  "daemons": {
    "node_metrics": {
      "mint-worker-0": {
        "node_ip": "192.168.39.110",
        "state": "healthy",
        "actor_name": "mint_daemon_node_metrics_mint-worker-0",
        "last_error": null
      }
    }
  }
}
```

## Deployment Labels and OTel Resource Attributes

All production API processes, model runtime actors, and daemon actors should
receive these deployment labels from environment or runtime config:

- `MINT_DEPLOYMENT_ENV`: `dev` or `prod`.
- `MINT_CLUSTER_ID`: `volcano` or `aliyun`. This is the stable Mint cluster
  family label; provider-specific values such as `volc`, `pai`, or DLC job
  names stay in topology metadata and must not replace this label.

When present, they are forwarded into Ray actor runtime environments and attached
to OTel resource attributes/metric attributes:

- `deployment.env` from `MINT_DEPLOYMENT_ENV`
- `mint.cluster_id` from `MINT_CLUSTER_ID`
- `service.name`

The API server service name is `mint-server`. The node metrics daemon service
name is `mint-node-metrics`. New service names must use the `mint-*` namespace.

`MINT_DEPLOYMENT_ENV` and `MINT_CLUSTER_ID` are part of metric identity for
production deployments. Missing labels are omitted rather than filled with
untrusted defaults; deployment wiring should set them so dev/prod and
Volcano/Aliyun data never join accidentally.

## Actor Naming

Detached actors owned by Mint should have short, stable names. Shape, placement,
generation, and session details belong in supervisor metadata or
`health_snapshot()`, not in the actor name.

Canonical GPU actor names:

- vLLM: `mint_vllm_{model_slug}`
- Megatron: `mint_megatron_{model_slug}`
- Dense trainer: `mint_dense_{model_slug}`
- OpenPI shared runtime: `mint_openpi_shared_{pool_slug_or_hash}`
- OpenPI action/session runtime: prefer shared-runtime logical sessions. If
  `mint_openpi_action_{hash}` remains as a compatibility path, it is
  session-scoped and must not be treated as durable low-cardinality model
  capacity.

Canonical daemon actor names:

- Node metrics: `mint_daemon_node_metrics_{worker_alias}`

Canonical supervisor wrapper names, if the wrapper actor remains necessary:

- Model runtime wrapper: `mint_model_runtime_{backend}_{model_slug}_{replica_id}`

The topology/daemon design introduces only the node metrics daemon actor. Other
detached control-plane actors such as `mint_task_state_store`,
`mint_model_work_scheduler`, `mint_config`, `mint_model_actor_supervisor`, and
`mint_maintenance_cron` are durable control-plane state managed outside the
topology daemon set. External operations bootstrap `mint_config` and
`mint_model_actor_supervisor`; the supervisor ensures the remaining CPU
control-plane actors. None of those actors should be counted as topology daemon
actors. Session/index/heartbeat/gateway metadata lives under
`mint_task_state_store`; do not reintroduce separate session metadata-store
actors, cleanup-executor actors, or startup-lease actors.

Do not include these in detached actor names:

- request IDs
- session IDs
- generation
- PID
- node IP
- hostname
- TP/PP/DP/world size
- max LoRA rank

Actor metadata should carry:

- `backend`: `vllm`, `megatron`, `dense`, or `openpi`
- `runtime_mode`: for example `single_node` or `multinode`
- `base_model`
- `replica_id`
- `generation`
- `tp`, `pp`, `dp`, `world_size` when relevant
- `max_lora_rank` when relevant
- `worker_aliases`
- `gpu_uuids`

The actor name is not the place for shape or lifecycle metadata. Keep those
fields in supervisor metadata and expose them through health snapshots and
metrics.

## NodeMetricsCollectorActor

`NodeMetricsCollectorActor` is a detached, node-pinned daemon actor. It collects
local node and GPU metrics on a fixed interval and exports them through OTel
metrics. The actor does not serve public API traffic and does not persist state.

Required behavior:

- exporter failure must not kill the actor
- missing NVML/nvidia-smi marks collector degraded but keeps the loop alive
- OTel push is the default metrics path when `OTEL_EXPORTER_OTLP_ENDPOINT` is
  set; exporter setup failure is recorded in `health_snapshot()` and does not
  block startup
- `health_snapshot()` reports sample time, last error, sample count, node
  identity, and exporter status
- `sample_cached()` returns the recent cached sample when fresh and may perform
  an on-demand refresh when the cache is stale; the background loop remains the
  normal producer
- `shutdown()` stops the sampling loop and flushes OTel providers best-effort;
  flush failure must not make shutdown fail

Exporter failure is not a supervisor restart condition. It marks the collector
as degraded because an unavailable OTel backend also prevents export-error
metrics from arriving. The supervisor should restart only on actor RPC failure,
a dead sampling loop, repeated local sampling crashes, or actor version mismatch.

### GPU Metrics

GPU identity is based on `gpu_uuid`. GPU name is optional diagnostic metadata and
must not be required for actor correlation because some MLP environments do not
expose stable GPU names.

Emit per `deployment.env`, `mint.cluster_id`, `worker_alias`, and `gpu_uuid`.
`node_ip`, `hostname`, `ray_node_id`, `ray_gpu_id`, and `gpu_index` are debug
metadata, not primary metric labels. If `gpu_uuid` is unavailable, GPU-level
actor correlation is unavailable; degrade to node/worker-level observability
instead of guessing from GPU index or PID.

- `mint_node_gpu_present`
- `mint_node_gpu_utilization_percent`
- `mint_node_gpu_memory_used_bytes`
- `mint_node_gpu_memory_total_bytes`
- `mint_node_gpu_power_draw_watts`
- `mint_node_gpu_power_limit_watts`
- `mint_node_gpu_temperature_celsius`
- `mint_node_gpu_sm_clock_mhz`
- `mint_node_gpu_memory_clock_mhz`
- `mint_node_gpu_pcie_link_gen`
- `mint_node_gpu_pcie_link_width`

### GPU Process Metrics

Aggregate by `gpu_uuid` and bounded `process_class`:

- `mint_node_gpu_processes`
- `mint_node_gpu_process_memory_used_bytes`

Allowed `process_class` values:

- `vllm`
- `megatron`
- `ray_python`
- `python`
- `other`

Do not use `pid` as a metric label.

These metrics are node pressure telemetry, not actor-level accounting. They do
not prove which Mint actor owns memory when multiple processes share a GPU.
Actor ownership remains a separate binding fact.

### Node Metrics

Emit per `deployment.env`, `mint.cluster_id`, and `worker_alias`. `node_ip` and
`hostname` may be attached as debug metadata but should not be the primary
dashboard grouping because they can churn across cluster recreates.

- `mint_node_cpu_utilization_ratio`
- `mint_node_load_1m`
- `mint_node_load5`
- `mint_node_load15`
- `mint_node_memory_used_bytes`
- `mint_node_memory_total_bytes`
- `mint_node_disk_used_bytes`
- `mint_node_disk_total_bytes`
- `mint_node_metrics_collector_up`
- `mint_node_metrics_collector_sample_age_s`
- `mint_node_metrics_collector_sample_duration_ms`
- `mint_node_metrics_collector_errors_total`

The disk usage sampling path defaults to `/vePFS-Mindverse/share/mint/<env>`
when `MINT_DEPLOYMENT_ENV` is set, otherwise `/vePFS-Mindverse/share/mint`.
Override it with `MINT_NODE_METRICS_DISK_PATH` for local development or
specialized mounts.

### Head Ray / GCS Metrics

Exactly one head `NodeMetricsCollectorActor` pushes global Ray/GCS gauges from
its cached background samples. Worker node collectors must not register these
families.

Ray live-state gauges:

- `mint_ray_cluster_up`
- `mint_ray_cluster_warning_count`
- `mint_ray_cluster_probe_error_count`
- `mint_ray_cluster_slow_probe_count`
- `mint_ray_cluster_total_probe_latency_ms`
- `mint_ray_cluster_cache_age_s`
- `mint_ray_cluster_last_success_unixtime`
- `mint_ray_cluster_last_success_age_s`
- `mint_ray_cluster_nodes{state=alive|dead}`
- `mint_ray_cluster_dead_nodes_missing_heartbeats`
- `mint_ray_cluster_{cpu,gpu,memory,object_store_memory}_{total|available}`
- `mint_ray_cluster_placement_groups_{total|created|removed|pending|pending_gpu}`
- `mint_ray_cluster_named_actors_total`
- `mint_ray_cluster_named_actors_namespace`
- `mint_ray_cluster_probe_success{probe}`
- `mint_ray_cluster_probe_latency_ms{probe}`

GCS bridge gauges:

- `mint_ray_gcs_metrics_bridge_up`
- `mint_ray_gcs_metrics_bridge_scrape_error_count`
- `mint_ray_gcs_metrics_bridge_sample_count`
- `mint_ray_gcs_metrics_bridge_scrape_latency_ms`
- `mint_ray_gcs_metrics_bridge_cache_age_s`
- `mint_ray_gcs_metrics_bridge_last_success_unixtime`
- `mint_ray_gcs_metrics_bridge_last_success_age_s`

The head collector also pushes selected raw GCS/grpc aggregates as
`mint_ray_gcs_raw_<upstream_name>` gauges, plus MinT-derived `mint_ray_gcs_*`
gauges such as task-event drop/store ratios and histogram means. It must not
attach source addresses, raw errors, node IDs, actor names, or placement group
names as metric labels. Do not push new MinT-owned metrics without the `mint_`
prefix.

## Actor Correlation

Actor-to-GPU correlation stays in supervisor-owned inventory:

- model actors publish `gpu_bindings`
- each binding should include `gpu_uuid` when possible
- `actor_name` is allowed as an internal metric label because metrics are pushed
  to the internal observability backend and are not a public API surface, but
  only on low-frequency actor binding facts

The node collector should not infer Mint actor ownership from PIDs. It only
publishes node-local GPU and process facts. Actor ownership is joined in the
observability backend or debug tooling via `gpu_uuid`.

Actor bindings and node metrics should share this composite identity:

1. `deployment.env`
2. `mint.cluster_id`
3. `worker_alias`
4. `gpu_uuid`

`ray_gpu_id` is Ray's logical allocation token and is useful only for debugging
Ray/CUDA remapping. It is not part of the correlation key. `ray_node_id` is also
debug metadata because it can change when Ray restarts. `gpu_index` is not a
reliable fallback in the target environments and should not be required.

If the join is ambiguous or missing, attribution must be reported as `unknown`;
do not guess from process names.

## Metric Label Contract

Keep labels bounded even for internal OTel metrics.

Per-sample node/GPU/process metrics may use:

- `deployment.env`
- `mint.cluster_id`
- `worker_alias`
- `gpu_uuid`
- `process_class`

They must not use:

- `actor_name`
- `request_id`
- `session_id`
- `pid`
- raw error strings

Actor binding metrics may additionally use:

- `actor_name`
- `actor_type`
- `backend`
- `base_model`
- `replica_id`
- `workload`

Alerting should prefer `worker_alias`, `actor_type`, `base_model`,
`replica_id`, and `gpu_uuid`. Volatile names such as `actor_name` are for
debugging and dashboards only.

## `/internal/metrics`

The Prometheus text endpoint is not the default metrics path and should remain
disabled by default. The required feature flag is
`MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED=1`. When disabled, the route should
return `404 Not Found` so Prometheus scrapes fail closed and operators do not
mistake it for the supported metrics path.

After OTel migration, `/internal/metrics` is sentinel-only. If enabled, it must
remain authenticated and must not trigger node-local sampling, Ray/GCS probing,
Ray actor snapshot collection, scheduler inspection, supervisor inspection, or
TaskStateStore inspection. Ray/GCS families are owned by the head
`NodeMetricsCollectorActor`; runtime, scheduler, supervisor, node, and
TaskStateStore families are pushed by their owning process or actor.

## Failure Semantics

- Node disappears from Ray: supervisor marks daemon state missing/stale and
  best-effort kills the old actor if reachable.
- Node joins Ray and matches selector: supervisor starts a collector.
- Collector unhealthy: supervisor restarts it with a new generation.
- OTel backend unavailable: collector increments error metrics and keeps
  sampling.
- NVML unavailable: collector reports degraded/up=0 for GPU sampling and keeps
  node metrics alive.

## Open Decisions Before Implementation

- Metric label cardinality budgets per backend.
