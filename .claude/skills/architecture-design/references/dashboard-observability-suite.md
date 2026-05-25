# MinT Dashboard Observability Suite

This document defines the target Grafana dashboard split and metric ownership
for MinT after the OTel-push migration. It is a planning reference, not an
implementation log.

## Constraints

- Do not add new Ray actors for observability without an explicit design review.
- OTel export, metric callback, scrape bridge, and dashboard support code must
  not sit on the user request critical path or add strong backpressure to
  business traffic.
- `/internal/metrics` remains disabled by default and sentinel-only when
  enabled. Production dashboards must not depend on it for service, runtime,
  Ray, GCS, or node telemetry.
- User-facing dashboards may use already-redacted user/account grouping labels,
  but high-cardinality user identifiers must not be default trend dimensions.
- vLLM deep runtime dashboards consume owner-process OTel push from MinT vLLM
  actors. Current MinT dashboards use `mint_vllm_*` metric families emitted
  from the actor-local vLLM stats observer, not a dashboard scrape of native
  `vllm:*`.
- Megatron runtime metrics use MinT-owned `mint_megatron_*` families because
  Megatron does not provide a comparable native Prometheus surface in this
  deployment.
- Ray metrics should be bridged by allowlist. Do not mirror the entire Ray
  metrics surface into MinT dashboards.
- Node/GPU metrics are best-effort. Dashboard health must expose collector
  freshness and coverage before relying on GPU panels.

## Current NodeMetrics Reality Check

Current implementation:

- `NodeMetricsCollectorActor` already exists.
- It is a detached, node-pinned daemon managed by `ModelActorSupervisor`.
- It samples host metrics with `psutil`, GPU metrics with `pynvml`, and head
  Ray/GCS snapshots only on the head collector.
- It registers OTel observable gauges inside the daemon process and samples on
  a background loop.
- It does not serve API traffic and does not require adding a new actor class
  for the dashboard suite.

Observed locally in the project venv:

- Host metrics are available: CPU, load, memory, and disk for
  `/vePFS-Mindverse/share/mint`.
- GPU metrics are not available in the project venv because `pynvml` is not
  installed there.
- `nvidia-smi` is not present on the local workstation.

Observed on production Volcano workers:

- The driver host runtime does not have `pynvml`.
- GPU worker runtime uses `/opt/venv/bin/python3` and already has `pynvml`.
- A one-off Ray GPU probe on a production worker successfully ran
  `pynvml.nvmlInit()`, saw 8 GPUs, and read a GPU UUID.
- Existing node metrics daemons on `mint-worker-1` through `mint-worker-5`
  reported `gpu_count=8`, no GPU error, OTel enabled, and UUID-bearing GPU
  samples.
- `mint-worker-0`'s node metrics daemon was observed restarting during the
  probe. Treat that as daemon health evidence to investigate separately, not as
  proof that NVML is unavailable on workers.
- The topology manager represents the observed Ray head as the runtime-only
  alias `mint-head` with `provider: ray` and `role: head`. If Ray does not mark
  the head explicitly, the IP in `ray.head_ip_path` is used as the fallback
  head identifier. This alias is eligible only for the head
  `NodeMetricsCollectorActor`; it is invalid for model placement and never
  triggers provider worker creation.

Implication:

- Node host panels can be planned as first-class signals.
- GPU panels can use production worker NodeMetrics where the daemon is healthy,
  but must remain coverage-aware: if `mint_node_gpu_present` is absent or
  `mint_node_metrics_collector_errors_total` rises with NVML/import errors, the
  dashboard should show "GPU metrics unavailable" rather than imply idle GPUs.
- Ray/GCS panels depend on `mint-head` daemon coverage. If
  `mint_ray_cluster_*` or `mint_ray_gcs_*` is absent, first check whether
  topology state includes `mint-head` and whether
  `mint_daemon_node_metrics_mint-head` is healthy.
- Installing `nvidia-ml-py` in the project/driver venv is not required for
  worker-side GPU NodeMetrics as long as the worker image keeps `pynvml`
  available. If a future runtime image drops it, restoring `nvidia-ml-py` is a
  runtime dependency change and does not require a new actor.

## Target Dashboard Layers

### 1. MinT User Experience

Audience: users, product operations, and service owners who care about
externally visible behavior.

Questions:

- Are users succeeding?
- Which model, route, or redacted account group is degraded?
- Are requests timing out, waiting, or being rejected?
- Is pending behavior expected async waiting or service degradation?

Metric sources:

- `mint_http_server_*`
- `mint_retrieve_future_wait_total`
- `mint_sampling_admission_total`
- Task/future terminal outcome families from `TaskStateStore`
- Redacted account/key-group dimensions when available

Default dimensions:

- route
- model/base model
- status/status class
- outcome
- redacted account or key group as top-N drilldown, not default global split

Excluded:

- actor names
- Ray/GCS internals
- placement groups
- GPU UUIDs

### 2. MinT Service Health

Audience: oncall and operators doing first-level triage.

Questions:

- Can the service accept and track work?
- Which control-plane dependency is unhealthy or degraded?
- Is backlog growing before runtime capacity is reached?

Metric sources:

- public health and internal health outputs converted to low-cardinality
  metrics where available
- `mint_model_work_scheduler_*`
- `mint_scheduler_*`
- `mint_task_futures_*`
- `mint_task_future_reaper_*`
- `mint_model_actor_supervisor_*`
- `mint_node_metrics_daemon_*`
- API process gauges
- billing outbox metrics when available

Panels:

- public health cache age and refresh outcome
- HTTP request/error pressure
- scheduler decision rate, queue wait, ready-session, and chosen-depth
  histograms
- future pending/done/error/timeout pressure when TaskStateStore OTel gauges
  are active
- supervisor managed/desired/healthy domains when Supervisor OTel gauges are
  active
- daemon coverage and stale collectors

### 3. MinT vLLM Deep Dive

Audience: MinT/vLLM developers and runtime operators.

Questions:

- Is vLLM saturated by queueing, prefill, decode, cache, or request shape?
- Are we losing throughput because of KV pressure, preemption, cache misses, or
  long requests?
- Which model/deployment/instance is the bottleneck?

Metric sources:

- MinT-owned actor-local vLLM gauges from `VllmStatsObserver`, exported as
  `mint_vllm_*`.
- MinT actor-boundary metrics where they answer a MinT-specific question:
  `mint_vllm_actor_requests_total` and
  `mint_vllm_actor_request_duration_s_bucket`.

Reference dashboard:

- `/root/docs/ops/dashboards/deployment_vllm.json`

Panel groups:

- traffic: prompt tok/s, output tok/s, requests/s
- scheduler: running, waiting, KV cache usage
- latency: E2E, TTFT, ITL, queue, prefill, decode, inference time, time per
  output token
- cache: prefix cache query/hit, cached and recomputed prompt tokens
- request shape: prompt length, generation length, max tokens, `n`
- outcomes: success, finish reason
- preemption
- spec decode if present
- runtime config, including cache config

Implementation note:

- Prefer owner-process OTel push from the vLLM actor context. The actor already
  knows `actor_name`, base model, replica metadata, and deployment labels.
- Do not make API routes scrape vLLM metrics.
- Dashboard variables must discover `actor_name` and `base_model` from live
  VictoriaMetrics labels. Do not hard-code `job="deployment-vllm"` in MinT vLLM
  dashboards; that job remains valid only for the external deployvLLM reference
  dashboard.

### 4. MinT Megatron Deep Dive

Audience: MinT/Megatron developers and runtime operators.

Questions:

- Is training runtime making progress?
- Which stage is slow or stalled?
- Are rank memory, session switching, checkpoint/export, or worker health
  causing degradation?

Metric sources:

- `mint_megatron_*`
- `mint_training_operations_total`
- `mint_training_operation_duration_s`

Panel groups:

- operation latency by op and stage
- operation throughput by op
- session switch count
- active sessions, unknown-session count, step progress, learning rate, and
  per-rank GPU memory only when Megatron actor-local OTel gauges are active
- checkpoint/export/save latency
- rank heartbeat or worker health if implemented later

Excluded:

- reward, loss-quality, KL, and experiment-quality panels. Those belong to a
  separate experiment-quality dashboard, not runtime health.

### 5. MinT Ray Cluster Deep Dive

Audience: oncall and platform/runtime engineers.

Questions:

- Is Ray scheduling or control plane the bottleneck?
- Are placement groups pending because logical Ray resources are insufficient
  or because state is stale?
- Are GCS, raylet, object store, or actor churn causing symptoms?

Current bridged sources:

- `mint_ray_cluster_*`
- `mint_ray_gcs_metrics_bridge_*`
- selected `mint_ray_gcs_raw_*`
- selected `mint_ray_gcs_*` derived gauges

Current coverage caveat:

- These gauges are owned by the head `NodeMetricsCollectorActor`. If topology
  state does not include a ready `mint-head` entry, dashboards will not see live
  Ray/GCS OTel push series. The correct fix is to make head observation work,
  not to add a new metrics actor class.

Initial allowlist:

- cluster up, warning count, probe error count, slow probe count
- probe success and latency by probe
- alive/dead nodes and missing heartbeat dead nodes
- CPU/GPU/memory/object-store total and available
- placement group total/created/removed/pending/pending_gpu
- named actor total and namespace count
- GCS bridge up, scrape error count, sample count, scrape latency, cache age
- GCS task-event drop/store ratios and other derived control-plane bottleneck
  signals

Exploration candidates:

- actor state counts by state, if available without high-cardinality actor
  labels
- task backlog/scheduling delay aggregates, if Ray exports bounded summaries
- object store spill/restore/OOM counters
- raylet scheduling queue or resource pressure counters
- GCS pubsub queue/drop/latency aggregates
- worker start failure counts and restart/churn counters

Do not bridge:

- raw actor names
- placement group names
- node IDs as primary labels
- task IDs
- error strings

### 6. MinT GPU / Node Infra

Audience: runtime operators and infrastructure owners.

Questions:

- Are physical nodes and GPUs healthy?
- Does Ray's logical resource view match actual node/GPU pressure?
- Which actor/model is bound to a hot GPU when binding data is available?

Metric sources:

- `mint_node_metrics_collector_*`
- `mint_node_cpu_utilization_ratio`
- `mint_node_load*`
- `mint_node_memory_*`
- `mint_node_disk_*`
- `mint_node_gpu_*`
- `mint_model_actor_inventory_actor_gpu_binding`

Panel groups:

- collector coverage and sample freshness
- node CPU/load/memory/disk
- GPU present/utilization/memory/power/temperature/clocks
- GPU process count and process memory by bounded `process_class`
- actor-to-GPU binding coverage by `gpu_uuid`
- Ray logical GPU available vs physical GPU utilization

Best-effort behavior:

- If GPU metrics are unavailable, panels must surface collector errors and
  missing `mint_node_gpu_present` coverage.
- Do not infer actor ownership from process names or PIDs.
- Use `gpu_uuid` for joins; `gpu_index` is only diagnostic when present.

### 7. MinT Overview

Audience: oncall entry point.

Questions:

- Which layer is unhealthy?
- Which drill-down should be opened next?

Panels:

- user experience health from `mint_http_server_*` and retrieve/admission
  counters
- service health from public-health, scheduler, and training operation metrics
- vLLM health summary from `mint_vllm_*` and MinT actor-boundary metrics
- Megatron health summary from `mint_training_*` and
  `mint_megatron_session_switch_total`
- Node/GPU coverage summary from `mint_node_*`
- Ray/GCS summary only when head daemon coverage exists
- direct links to the layer dashboards

Do not put deep analysis panels here.

### 8. MinT OTel Push Debug

Audience: observability operators.

Questions:

- Is OTel export working?
- Are metrics fresh?
- Which service/process stopped pushing?

This dashboard is about the telemetry path, not service health. It may include
series freshness, exporter errors if available, and service/process identities.
It should not be the primary service triage dashboard.

## Implementation Phasing

1. Inventory current live metric families and label names in VictoriaMetrics.
2. Build the dashboard spec tables before editing JSON:
   - dashboard
   - panel group
   - query family
   - labels used
   - source process
   - missing metric gap
3. Update dashboard JSON in small batches:
   - user/service overview first
   - vLLM deep dive next
   - Ray and Node/GPU drill-downs next
   - Megatron deep dive last if new metrics are needed
4. Add missing runtime metrics only when the dashboard spec proves they are
   necessary.
5. Keep each metric addition owner-local:
   - vLLM metrics: vLLM actor process
   - Megatron metrics: Megatron worker/group process
   - Ray/GCS metrics: head node metrics daemon
   - Node/GPU metrics: per-node metrics daemon
   - API/user metrics: API worker or TaskStateStore/Scheduler owner

## Acceptance Criteria

- No new actor class is introduced without explicit approval.
- API request handlers do not perform dashboard-driven metric sampling.
- `/internal/metrics` is not used by dashboard queries except the debug sentinel
  if explicitly needed.
- Every dashboard has an audience and a first diagnostic question.
- vLLM panels use actor-local `mint_vllm_*` metrics. Native `vllm:*` queries
  may remain only in the external deployvLLM reference dashboard, not in MinT
  dashboards that are expected to work from OTel push alone.
- Megatron panels use `mint_megatron_*` and `mint_training_*` metrics only for
  performance and health, not training quality.
- Ray panels use an explicit allowlist and do not explode label cardinality.
- Node/GPU panels show collector coverage before GPU utilization conclusions.
