# Runtime configuration and ConfigActor

## Boundary

Runtime configuration is split by when the value must be available:

1. **Bootstrap runtime_env**: required before a Ray actor can import code or connect to Ray.
2. **Actor creation inputs**: required by Ray at `.options(...).remote(...)` time.
3. **ConfigActor snapshot**: low-frequency deployment/runtime config that actors can fetch after startup.
4. **Observability config**: OTEL/APM/logging values. These can be recorded in the snapshot, but libraries that initialize from env still require explicit env forwarding until their initialization path is migrated.
5. **Task state**: task/lease/result state is not configuration. It belongs in TaskStateStore, Scheduler, or FutureStore.

`ConfigActor` is a namespace-local detached actor. Namespace is the deployment isolation boundary, so the production actor name is stable inside the namespace and does not contain a run id.

## V1 semantics

- Actor name: `mint_config`, overridable by `MINT_CONFIG_ACTOR_NAME` for tests or controlled migrations.
- API: `get_snapshot()` only.
- Mutation: none. V1 has no `put`, no `set_many`, and no watch mechanism.
- Startup behavior: the startup owner creates or attaches to the actor. If an existing actor has a different snapshot fingerprint, startup fails fast instead of silently using stale configuration.
- Persistence: none inside ConfigActor. The API process rebuilds the read-only snapshot from env/config file on startup.
- Secret handling: the V1 snapshot is for configuration discovery and future non-secret migration. Secret-like keys are redacted; do not use ConfigActor as a secret distribution service.

## Bootstrap runtime_env

These values remain explicit actor bootstrap inputs because the actor cannot query ConfigActor before it starts:

- `PYTHONPATH`
- `PFS_RUNTIME_ENV_ROOT`
- `PFS_TINKER_PATH`
- `PFS_HF_MODULES_PATH`
- `RAY_ADDRESS`
- `TINKER_RAY_NAMESPACE` / `MINT_RAY_NAMESPACE`
- `TINKER_CONFIG_PATH`
- Ray Client packaging inputs such as `MINT_RAY_JOB_WORKING_DIR`, `MINT_RAY_WORKING_DIR`, and `MINT_RAY_PY_MODULES_CSV`
- library loader bootstrap such as `TINKER_ACTOR_LD_LIBRARY_PATH`

These should be small and relatively stable. If a frequently changed business setting needs bootstrap propagation, treat that as a design smell and migrate the consumer to the ConfigActor snapshot when possible.

## Actor creation inputs

These cannot be fetched after actor startup because Ray needs them when creating the actor:

- actor name, lifetime, restart policy, max concurrency;
- `num_gpus`, custom resources, node pins, placement group strategy;
- control-plane pinning such as `MINT_CONTROL_PLANE_PINNED_NODE_IP`;
- model placement maps that determine actor creation placement.

The snapshot can include these for audit/debugging, but not as the source of truth for already-created actors.

## Snapshot candidates

Good ConfigActor candidates are low-frequency runtime settings currently passed through env only because there was no namespace-local config service:

- vLLM tuning knobs such as max sequences, batched tokens, LoRA slot counts, and request timing toggles;
- model registry override JSON;
- future replay paths and retrieve throttling policy;
- scheduler fairness/coalesce/debug settings;
- checkpoint index and usage backend settings;
- Megatron runtime feature flags and diagnostics;
- local payload/checkpoint roots.

V1 only publishes them. Consumers should migrate incrementally in small PRs.

## Observability

OTEL and APM keys are included in the classification and may appear in the snapshot for introspection. They still need explicit env forwarding where OpenTelemetry or logging libraries read env during import or provider initialization.
