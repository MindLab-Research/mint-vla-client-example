# Topology-Aware Supervision and Node Metrics DaemonSet

## Goal

Mint needs a topology-aware control plane for two related concerns:

1. Static model-runtime placement should refer to stable worker aliases instead
   of raw node IPs.
2. Node-local observability should run as a per-node daemon actor and push
   metrics through OpenTelemetry.

`ModelActorSupervisor` owns this reconciliation. It remains the single
supervisor for long-lived Ray actors that Mint intentionally keeps alive, but it
must keep model-runtime actors and daemon actors as separate scheduling classes.

## Non-goals

- Do not add a separate metrics registry actor or metrics supervisor actor.
- Do not make `/internal/metrics` part of the default observability path.
- Do not persist metrics snapshots.
- Do not let metrics collection block request handling.
- Do not make the API server create or destroy cloud worker jobs in V1.

## Topology Layers

### 1. Node Topology

Node topology describes the cluster workers that Mint may target:

- `alias`: stable name such as `worker2` or `train-a`.
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

V1 reads this from a static config file and verifies it against `ray.nodes()`.
The topology reader may mark nodes stale or unavailable, but it should not
provision or tear down cloud nodes.

The future provider interface may expose operations like "request node from
template" and "wait for node to join Ray", but those operations are not part of
the first implementation. V1 topology is read-only from the server's
perspective.

### 2. Static Actor Placement

Model runtime placement should become topology-aware. The target configuration
references node aliases, not raw IPs:

```toml
[[models."Qwen/Qwen3-30B-A3B-Instruct-2507".megatron.placement]]
worker = "worker2"
gpu_count = 4
replica = 0
```

Accepted topology-aware placement keys:

- `worker` or `worker_alias`: required stable alias.
- `gpu_count`: required for GPU actors.
- `replica`: optional replica number, normalized into `replica_id`.
- `labels`: optional selector metadata for future placement policies.

At runtime, the supervisor resolves the alias into `node_ip` and passes only the
resolved placement to Ray launchers. The refactor target is to remove raw
`node_ip` from internal placement. During migration, any old raw-IP placement
parsing must be isolated at the config boundary and normalized immediately.
After configs have moved to aliases, `node_ip` should be rejected in model
placement config with a clear "use worker/worker_alias" error.

The normalized internal form is:

```python
ResolvedPlacement(
    worker="worker2",
    node_ip="192.168.39.110",
    gpu_count=4,
)
```

Implementation migration points:

- `ModelActorSpec` and placement parsing must accept alias fields and stop
  treating `worker`, `worker_idx`, or `worker_index` as raw runtime pins.
- `volc_placement`, dense launchers, vLLM launchers, and Megatron launchers
  should receive resolved node placements, not parse topology themselves.
- Shared dev/prod config must be updated before raw `node_ip` rejection is
  enforced.

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
- head/API-only CPU nodes are excluded unless explicitly labeled as eligible
- actor pinning with `resources={f"node:{node_ip}": 0.001}`
- `num_gpus=0`

The first daemon actor is `NodeMetricsCollectorActor`.

Daemon actors are not model replicas. They must not be registered with
`ModelWorkScheduler`, must not have queue IDs, and must not participate in task
claiming.

## ModelActorSupervisor Shape

`ModelActorSupervisor` should own both scheduling classes:

```python
ModelActorSupervisor
  - model_specs: dict[(domain_key, replica_id), ModelActorSpec]
  - daemon_specs: dict[name, DaemonActorSpec]
  - topology: ClusterTopology
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

`reconcile_once()` should run in this order:

1. refresh/resolve topology
2. reconcile daemon actors
3. reconcile model runtime actors
4. sync model runtime replicas to `ModelWorkScheduler`

Only model runtime actors are synced to the scheduler.

The supervisor snapshot should expose daemon state separately:

```json
{
  "replicas": {"vllm:...::replica-0": {"state": "healthy"}},
  "daemons": {
    "node_metrics": {
      "worker2": {
        "node_ip": "192.168.39.110",
        "state": "healthy",
        "actor_name": "mint_node_metrics_collector_worker2",
        "last_error": null
      }
    }
  }
}
```

## NodeMetricsCollectorActor

`NodeMetricsCollectorActor` is a detached, node-pinned daemon actor. It collects
local node and GPU metrics on a fixed interval and exports them through OTel
metrics. The actor does not serve public API traffic and does not persist state.

Required behavior:

- exporter failure must not kill the actor
- missing NVML/nvidia-smi marks collector degraded but keeps the loop alive
- collection interval defaults to 5 seconds
- `health_snapshot()` reports last successful sample time, last error, sample
  count, node identity, and exporter status
- `health_snapshot()` includes `last_sample_success_at`,
  `last_export_success_at`, `last_export_error`,
  `export_consecutive_failures`, `sampling_loop_alive`, and `collector_state`
- `shutdown()` stops the sampling loop and flushes OTel best-effort

Exporter failure is not a supervisor restart condition. It marks the collector
as degraded because an unavailable OTel backend also prevents export-error
metrics from arriving. The supervisor should restart only on actor RPC failure,
a dead sampling loop, repeated local sampling crashes, or actor version mismatch.

### GPU Metrics

GPU identity is based on `gpu_uuid`. GPU name is optional diagnostic metadata and
must not be required for actor correlation because some MLP environments do not
expose stable GPU names.

Emit per `gpu_uuid`, `node_ip`, `hostname`, `worker_alias`, and optional
`gpu_index`:

- `mint_node_gpu_present`
- `mint_node_gpu_utilization_ratio`
- `mint_node_gpu_memory_utilization_ratio`
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

Emit per `node_ip`, `hostname`, and worker alias when known:

- `mint_node_cpu_utilization_ratio`
- `mint_node_load1`
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

The disk path defaults to `/share/mint` and should be configurable.

## Actor Correlation

Actor-to-GPU correlation stays in `ModelActorInventory`:

- model actors publish `gpu_bindings`
- each binding should include `gpu_uuid` when possible
- `actor_name` is allowed as an internal metric label because metrics are pushed
  to the internal observability backend and are not a public API surface, but
  only on low-frequency actor binding facts

The node collector should not infer Mint actor ownership from PIDs. It only
publishes node-local GPU and process facts. Actor ownership is joined in the
observability backend or debug tooling via `gpu_uuid`.

Actor bindings and node metrics should share this composite identity when
available:

- `worker_alias`
- `ray_node_id`
- `node_ip`
- `hostname`
- `gpu_uuid`
- `gpu_index`
- `ray_gpu_id`

Join order:

1. `worker_alias + gpu_uuid`
2. `ray_node_id + gpu_uuid`
3. `node_ip + gpu_uuid`
4. `hostname + gpu_uuid`
5. `node identity + gpu_index`

If the join is ambiguous or missing, attribution must be reported as `unknown`;
do not guess from process names.

## Metric Label Contract

Keep labels bounded even for internal OTel metrics.

Per-sample node/GPU/process metrics may use:

- `worker_alias`
- `node_ip`
- `hostname`
- `gpu_uuid`
- `gpu_index`
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

Existing Prometheus metrics may be removed or reduced only after their signals
are covered by OTel push metrics and any dependent dashboard/alert has migrated.

If `/internal/metrics` is enabled for debugging, it must remain authenticated and
must not trigger node-local sampling. It may render cached process-local state
only.

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

- Exact static topology file path and schema ownership.
- Whether topology-aware placement rejects raw `node_ip` immediately or after a
  short config migration commit.
- Whether worker alias should be emitted as `worker` or `worker_alias` label;
  prefer `worker_alias` for clarity.
- Daemon feature flag and rollout default.
- Daemon actor naming and generation policy.
- Exact daemon state/admin surface under `ModelActorSupervisor`.
- OTel no-op behavior when exporter env or Python dependencies are missing.
- Provider fields required in V1 for Volc and PAI static topology.
- Alias uniqueness and conflict behavior when Ray reports duplicate/stale nodes.
- Topology hot reload behavior and whether config changes require supervisor
  restart.
- Metric label cardinality budgets per backend.
