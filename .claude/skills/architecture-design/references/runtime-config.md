# Runtime configuration and ConfigActor

## Boundary

Runtime configuration is split by when the value must be available:

1. **Bootstrap runtime_env**: required before a Ray actor can import code or connect to Ray.
2. **Actor creation inputs**: required by Ray at `.options(...).remote(...)` time.
3. **ConfigActor actor_env**: namespace-local deployment/runtime config that actor processes hydrate from ConfigActor during import.
4. **Observability config**: OTEL/APM/logging values. These are distributed through ConfigActor actor_env for actor hydration.
5. **Task state**: task/lease/result state is not configuration. Durable task/result state belongs in TaskStateStore; hot scheduling state belongs in ModelWorkScheduler. TaskFutureService is only the route-facing facade.

`ConfigActor` is a namespace-local detached actor. Namespace is the deployment isolation boundary, so the production actor name is stable inside the namespace and does not contain a run id.

`MINT_*` is the canonical environment/config namespace for server-owned deployment settings. The runtime-env compatibility helper accepts the SDK-facing environment aliases and canonicalizes them to `MINT_*`: reading a canonical key checks the canonical value first, then its compatibility alias. Code outside that helper should use canonical `MINT_*` names, and ConfigActor snapshots should publish canonical `MINT_*` keys.

## V1 semantics

- Actor name: `mint_config`, overridable by `MINT_CONFIG_ACTOR_NAME` for tests or controlled migrations.
- API: `get_snapshot()` only.
- Mutation: none. V1 has no `put`, no `set_many`, and no watch mechanism.
- Bootstrap behavior: external operations create or attach to `mint_config` before `mint_model_actor_supervisor` and API workers start. If an existing actor has a different snapshot fingerprint, bootstrap fails fast instead of silently using stale configuration. API workers only read/check the existing actor and must not create it.
- Persistence: none inside ConfigActor. The read-only snapshot is rebuilt from env/config file during external bootstrap.
- Actor hydration: normal Ray actors receive only bootstrap runtime_env plus `MINT_CONFIG_ACTOR_HYDRATE=1`. `mint_server.config` fetches ConfigActor once on import and overlays `actor_env` into `os.environ` before module-level config constants are computed.
- Secret handling: `env` and `server_config` remain redacted for introspection, but `actor_env` contains real values because it is the actor configuration distribution payload. Namespace access to ConfigActor is therefore a trust boundary.

## Bootstrap runtime_env

These values remain explicit actor bootstrap inputs because the actor cannot query ConfigActor before it starts:

- `PYTHONPATH`
- `PFS_RUNTIME_ENV_ROOT`
- `MINT_CODE_ROOT`
- `PFS_HF_MODULES_PATH`
- `RAY_ADDRESS`
- `MINT_RAY_NAMESPACE`
- `MINT_CONFIG_PATH`
- Ray Client packaging inputs such as `MINT_RAY_JOB_WORKING_DIR`, `MINT_RAY_WORKING_DIR`, and `MINT_RAY_PY_MODULES_CSV`
- library loader bootstrap such as `MINT_ACTOR_LD_LIBRARY_PATH`

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
- unclassified `MINT_`, compatibility alias, and `OTEL_` keys, so newly added deployment knobs do not silently fall back to direct runtime_env forwarding;
- canonicalized `MINT_*` names when only a compatibility alias is set.

`actor_env` deliberately excludes bootstrap runtime_env keys and ConfigActor hydration control flags. Per-actor identity or execution contract values may still be passed through `extra` at actor creation, because they are not deployment configuration.

## Observability

OTEL and APM keys are included in `actor_env`. Actor processes hydrate them before `mint_server.config` module globals are evaluated, then actor initialization calls the normal observability setup path.
