# Runtime configuration and ConfigActor

## Boundary

Runtime configuration is split by when the value must be available:

1. **Bootstrap runtime_env**: required before a Ray actor can import code or connect to Ray.
2. **Actor creation inputs**: required by Ray at `.options(...).remote(...)` time.
3. **ConfigActor actor_env**: namespace-local deployment/runtime config that actor processes hydrate from ConfigActor during import.
4. **Observability config**: OTEL/APM/logging values. These are distributed through ConfigActor actor_env for actor hydration.
5. **Task state**: task/lease/result state is not configuration. It belongs in TaskStateStore, Scheduler, or FutureStore.

`ConfigActor` is a namespace-local detached actor. Namespace is the deployment isolation boundary, so the production actor name is stable inside the namespace and does not contain a run id.

## V1 semantics

- Actor name: `mint_config`, overridable by `MINT_CONFIG_ACTOR_NAME` for tests or controlled migrations.
- API: `get_snapshot()` only.
- Mutation: none. V1 has no `put`, no `set_many`, and no watch mechanism.
- Startup behavior: the startup owner creates or attaches to the actor. If an existing actor has a different snapshot fingerprint, startup fails fast instead of silently using stale configuration.
- Persistence: none inside ConfigActor. The API process rebuilds the read-only snapshot from env/config file on startup.
- Actor hydration: normal Ray actors receive only bootstrap runtime_env plus `MINT_CONFIG_ACTOR_HYDRATE=1`. `tinker_server.config` fetches ConfigActor once on import and overlays `actor_env` into `os.environ` before module-level config constants are computed.
- Secret handling: `env` and `server_config` remain redacted for introspection, but `actor_env` contains real values because it is the actor configuration distribution payload. Namespace access to ConfigActor is therefore a trust boundary.

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

## Actor Env

ConfigActor owns actor-readable deployment/runtime configuration. The API process builds `actor_env` from:

- known actor creation, snapshot, observability, and task-state configuration keys;
- unclassified `MINT_`, `TINKER_`, and `OTEL_` keys, so newly added deployment knobs do not silently fall back to direct runtime_env forwarding;
- canonical `MINT_*` actor name aliases when only legacy `TINKER_*` names are set.

`actor_env` deliberately excludes bootstrap runtime_env keys and ConfigActor hydration control flags. Per-actor identity or execution contract values may still be passed through `extra` at actor creation, because they are not deployment configuration.

## Observability

OTEL and APM keys are included in `actor_env`. Actor processes hydrate them before `tinker_server.config` module globals are evaluated, then actor initialization calls the normal observability setup path.
