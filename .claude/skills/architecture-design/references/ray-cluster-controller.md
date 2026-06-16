# Ray Cluster Controller

## Node roles

The Mint deployment has three Ray node types:

```
ray driver   — runs mint-server (FastAPI + ModelActorSupervisor). CPU-heavy
               control plane. Lives on MLP (Volcano/Aliyun), not in k8s.
ray head     — GCS, scheduler, dashboard. Lives on MLP.
ray worker   — GPU compute (vLLM, Megatron, dense trainers). Lives on MLP.
```

k8s runs only `RayClusterController`. All Ray nodes (driver, head, worker)
live on the MLP provider. The controller reaches the Ray cluster and provider
APIs over the network; it does not share a filesystem with any Ray node.

## Role

`RayClusterController` is a standalone k8s Deployment that owns two concerns
separated from `ModelActorSupervisor`:

1. **Node lifecycle** — submit, health-poll, and retire provider worker jobs
   (Volcano, Aliyun DLC) based on desired state written to Redis.
2. **Ray connectivity health** — probe Ray GCS liveness and publish the live
   head address to Redis so mint-server can reconnect without operator
   intervention.

`ModelActorSupervisor` continues to own actor reconciliation, daemon DaemonSet,
and scheduler sync. It reads resolved topology from Redis instead of managing
provider jobs itself.

No authentication is required on the controller HTTP API in V1. A future ops
plane will add auth at that layer.

## Why k8s, not a detached Ray actor

`TopologyManager` currently runs inside `ModelActorSupervisor` (a detached Ray
actor on the driver node). This has two structural problems:

- If the Ray cluster restarts, the supervisor dies with it. While dead, no one
  monitors provider job health or updates the head address.
- Provider credentials live on whichever MLP node the supervisor lands on;
  node restarts silently lose them until operator re-injects.

A k8s Deployment is independent of MLP and Ray cluster health. It continues
observing provider state and updating Redis while Ray is fully down, and it
receives credentials from k8s Secrets in a controlled, auditable way.

## ConfigActor → Redis

`ConfigActor` (`mint_config`) is removed. Redis replaces it as the source of
deployment configuration for both the API server and Ray actors.

### What changes

| Before | After |
|---|---|
| `ConfigActor` detached Ray actor (`mint_config`) | Deleted |
| `get_snapshot()` call on import | `mint_server.config` reads from Redis on import |
| `MINT_CONFIG_ACTOR_HYDRATE=1` in runtime_env | Removed |
| Actor env hydration from ConfigActor at import time | Actors read config from Redis directly |
| External bootstrap must start `mint_config` before supervisor | Bootstrap only needs to ensure Redis has the config key |

### Redis config key

```
mint:cluster:{cluster_id}:config
  Hash. Written by external bootstrap / ops tooling.
  Contains the same fields previously published by ConfigActor get_snapshot():
    actor_env.*   (all actor-readable MINT_* and OTEL_* keys)
    fingerprint   (sha256 of the serialized snapshot, for staleness detection)
    written_at    ISO-8601
```

Bootstrap writes this key before starting mint-server. On mint-server startup,
`mint_server.config` reads `mint:cluster:{cluster_id}:config` and overlays
`actor_env` into `os.environ`. Ray actors receive `MINT_REDIS_URL` and
`MINT_CLUSTER_ID` in bootstrap `runtime_env` (the only two keys that cannot
come from Redis) and then read the config key on import.

### What does NOT change

Bootstrap `runtime_env` keys that Ray needs before any import still come from
env vars, not Redis:

```
PYTHONPATH, PFS_RUNTIME_ENV_ROOT, MINT_CODE_ROOT, PFS_HF_MODULES_PATH,
MINT_RAY_GCS_ADDRESS, MINT_RAY_NAMESPACE, MINT_REDIS_URL, MINT_CLUSTER_ID
```

These remain small and stable. Everything else that was in `actor_env` moves
to Redis.

If Redis is unreachable at mint-server startup, startup fails fast with a
clear error. There is no fallback to env-only config — a missing Redis
connection is a misconfiguration, not a degraded mode.

## Redis as shared state

Redis is the interface between the controller, mint-server, and
ModelActorSupervisor. **The controller must not touch the Ray cluster or any
provider API if Redis is unreachable** — read failure is a hard stop, not a
fallback to cached state.

All keys are namespaced under `mint:cluster:{cluster_id}:`.

### Full key map

```
Written by RayClusterController:
  topology                   resolved node runtime state (alias → ip/state/…)
  ray:head                   live GCS address + health flag
  controller:status          controller heartbeat for ops visibility

Written by external bootstrap / ops:
  config                     deployment config snapshot (replaces ConfigActor)
  desired                    desired node set (triggers scale-up / scale-down)

Written by ModelActorSupervisor:
  supervisor:node_actors:{alias}   actor count per node (drain gate, TTL 60s)

Read by mint-server (ray_utils.py):
  ray:head                   GCS address for reconnect

Read by ModelActorSupervisor (reconcile step 3):
  topology                   resolved alias → node_ip for actor placement
```

### Key schemas

```
mint:cluster:{cluster_id}:topology
  Type: Hash, field per alias
  {alias}: JSON {
    node_ip, ray_node_id, provider, provider_job_id,
    state: alive|pending|dead|unknown,
    enabled: bool, gpu_count: int, observed_at: ISO-8601
  }

mint:cluster:{cluster_id}:ray:head
  Type: Hash
  gcs_address: "192.168.x.x:6379"
  head_ip:     "192.168.x.x"
  healthy:     "1"|"0"
  observed_at: ISO-8601

mint:cluster:{cluster_id}:desired
  Type: Hash, field per alias
  {alias}: JSON { enabled: bool, gpu_count: int, template: str, provider: str }

mint:cluster:{cluster_id}:supervisor:node_actors:{alias}
  Type: Hash, TTL 60s
  actor_count: int    (alive + inflight actors on this node)
  last_updated_at: ISO-8601
```

`topology_state.yaml` is no longer written. `MINT_TOPOLOGY_STATE_PATH` is
deprecated.

## Scale-down safety protocol

Scale-down is initiated by an external caller writing `enabled: false` for an
alias into `mint:cluster:{cluster_id}:desired`. The controller does NOT act
immediately. It runs a drain check before retiring the provider job:

```
1. Read topology key for the alias.
   If state != alive → skip drain, node already gone.

2. Read supervisor:node_actors:{alias} from Redis.
   If key is missing or TTL-expired → treat as unknown → block drain.
   If actor_count > 0 → record drain_blocked_at → retry next cycle.
   Never force-kill actors.

3. When actor_count == 0 and key is fresh:
   - Call provider SDK to stop the job.
   - Write state=dead to topology key.
```

**If Redis read fails at any step, abort the entire reconcile cycle.**
Do not guess. Do not use cached drain results. Do not touch any provider job.

### Supervisor side of drain

`ModelActorSupervisor` writes `supervisor:node_actors:{alias}` with TTL 60s
each reconcile cycle. If the supervisor is down and the key expires, the
controller sees unknown and blocks drain. Safe default.

## Reconcile loop

Two independent goroutines:

### Node lifecycle (30s interval)

```
1. Read desired from Redis. On error → abort cycle.
2. Read topology (own last state) from Redis.
3. For each alias:
   a. Query provider SDK for job liveness.
   b. Probe Ray node liveness (ray.nodes() or dashboard /api/v0/nodes).
   c. Resolve state: alive | pending | dead | unknown.
   d. desired enabled=true, state=dead/unknown → submit worker job.
      Guard: if provider already has a live job for this alias, reuse it.
   e. desired enabled=false, state=alive → run drain check (see above).
4. Write resolved topology to Redis.
5. Write controller:status heartbeat.
```

### Ray health probe (10s interval)

```
1. Read gcs_address from mint:cluster:{cluster_id}:ray:head (or env fallback).
2. Probe: ray.nodes() with 5s timeout.
3. Success → write healthy=1, observed_at.
4. 3 consecutive failures → write healthy=0.
   Do not change gcs_address on failure.
   Only update gcs_address when a live head is confirmed at a new address.
```

mint-server `ray_utils.py` polls `ray:head` for address changes. Existing
`MINT_RAY_HEAD_ADDRESS_PATH` file fallback remains for deployments without
the controller.

## Provider abstraction

```python
class ClusterProvider(Protocol):
    def list_jobs(self) -> list[ProviderJob]: ...
    def submit_job(self, alias: str, template: str) -> str: ...  # returns job_id
    def stop_job(self, job_id: str) -> None: ...
```

Implementations: `VolcanoProvider`, `AliyunDLCProvider`.
Provider selection is per-alias from the `provider` field in desired state.
Multi-provider deployments (e.g. Volcano head + Aliyun workers) are supported
because each alias carries its own provider discriminator.

Credentials are injected via k8s Secrets into the controller Pod env:
`VOLCENGINE_ACCESS_KEY` / `VOLCENGINE_SECRET_KEY` for Volcano,
Aliyun equivalents for DLC. Credentials must never appear in Redis keys, logs,
or metrics.

## Impact on ModelActorSupervisor reconcile

Current reconcile order (from `topology-daemon-metrics.md`):

```
1. ensure CPU control-plane deps
2. hydrate desired domains from scheduler
3. refresh/resolve topology          ← changed
4. reconcile daemon actors
5. reconcile model runtime actors
6. sync to ModelWorkScheduler
```

Step 3 after this change:

```
3. read mint:cluster:{cluster_id}:topology from Redis
   On Redis error → skip steps 4-6, emit degraded metric, do not crash
   On stale observed_at (> 120s) → emit warning, continue with stale data
```

The supervisor never calls a provider SDK and holds no provider credentials.
`TopologyManager`'s provider submission path is deleted. Its alias→IP
resolution becomes a thin Redis hash lookup.

Step 1 additionally ensures Redis connectivity instead of ensuring `mint_config`.

## Architecture diagram

```
                    ┌─────────────────────────────────────────────┐
                    │              Kubernetes                      │
                    │                                              │
                    │   ┌──────────────────────────────────────┐  │
                    │   │       RayClusterController           │  │
                    │   │                                      │  │
                    │   │  node lifecycle loop (30s)           │  │
                    │   │  ray health probe (10s)              │  │
                    │   │  HTTP API (scale-up/down, no auth)   │  │
                    │   └──────────────┬───────────────────────┘  │
                    └──────────────────┼───────────────────────────┘
                                       │ Redis R/W
                    ┌──────────────────▼───────────────────────────┐
                    │                 Redis                         │
                    │  config  topology  ray:head  desired          │
                    │  controller:status  supervisor:node_actors:*  │
                    └──────┬───────────────────────┬───────────────┘
                           │ Redis R/W              │ Redis R/W
          ┌────────────────▼──────────────────────────────────────┐
          │                 MLP (Volcano / Aliyun)                │
          │                                                        │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  ray driver node  (CPU-heavy, runs mint-server) │  │
          │  │                                                  │  │
          │  │  mint-server (FastAPI)                           │  │
          │  │    ray_utils.py: reads ray:head from Redis       │  │
          │  │    config: reads mint:cluster:*:config on import │  │
          │  │                                                  │  │
          │  │  ModelActorSupervisor (detached Ray actor)       │  │
          │  │    reconcile step 3: reads topology from Redis   │  │
          │  │    writes supervisor:node_actors:* to Redis      │  │
          │  └─────────────────────────────────────────────────┘  │
          │                                                        │
          │  ┌──────────────────┐   ┌──────────────────────────┐  │
          │  │  ray head node   │   │  ray worker nodes        │  │
          │  │  GCS + dashboard │   │  GPU: vLLM, Megatron,    │  │
          │  └──────────────────┘   │  dense trainers          │  │
          │                         └──────────────────────────┘  │
          └────────────────────────────────────────────────────────┘
                           │ provider SDK
          ┌────────────────▼───────────────┐
          │  Volcano Engine / Aliyun DLC   │
          │  (provider job management)     │
          └────────────────────────────────┘
```

## Failure modes and safety properties

| Failure | Controller behavior |
|---|---|
| Redis unreachable | Abort reconcile cycle. No provider calls. Emit metric. |
| Provider SDK error on submit | Record error in topology key. Retry next cycle. |
| Provider SDK error on stop | Block retire. Do not mark dead. Retry next cycle. |
| `supervisor:node_actors` TTL expired | Treat as unknown. Block drain. |
| Ray GCS unreachable (3 consecutive) | Set healthy=0. Do not change gcs_address. |
| Controller pod crash/restart | No in-memory state. Rebuilds entirely from Redis. |
| Redis unreachable at mint-server startup | Startup fails fast. No degraded fallback. |
| Redis unreachable during supervisor reconcile | Skip steps 4-6. Emit degraded metric. |

## Deployment (k8s sketch)

```yaml
env:
  - name: MINT_CLUSTER_ID
    value: "volcano"
  - name: MINT_REDIS_URL
    valueFrom:
      secretKeyRef: { name: mint-redis, key: url }
  - name: VOLCENGINE_ACCESS_KEY
    valueFrom:
      secretKeyRef: { name: mint-provider-creds, key: access_key }
  - name: VOLCENGINE_SECRET_KEY
    valueFrom:
      secretKeyRef: { name: mint-provider-creds, key: secret_key }
  - name: MINT_TOPOLOGY_CONFIG_PATH   # seeds desired state on first run
    value: "/config/topology.yaml"
```

The static topology YAML seeds `desired` into Redis on controller startup if
the key is absent. After that, external callers own desired state via the
controller HTTP API or direct Redis writes.
