# Issue: Stale Ray Actors Block Server Operations

## Summary

Detached Ray actors persist across server restarts and consume GPU resources indefinitely, causing new requests to fail or hang with "Insufficient GPUs" errors.

## Observed Behavior

1. **Server appears stuck**: Health check timeouts, requests hang
2. **Training fails**: `"Insufficient GPUs: need 1, available 0 after eviction"`
3. **Actor name conflicts**: `"The name tinker_vllm_qwen3-0.6b is already taken"`

## Root Cause

Ray actors created with `lifetime="detached"` survive:
- Server process restarts
- Server crashes
- New server deployments

These actors hold GPU allocations even when no longer needed.

## Evidence

Cluster state during failure:
```
GPUs: 0 / 2
Actors: [
  {'name': 'dense_trainer_pool_...', 'namespace': 'tinker'},
  {'name': 'tinker_vllm_qwen2.5-7b-instruct', 'namespace': 'tinker'},
  {'name': 'tinker_vllm_qwen3-0.6b', 'namespace': 'tinker'}
]
```

All 2 GPUs consumed by stale actors from previous sessions.

## Current Workaround

Manual cleanup before operations:
```python
import ray
ray.init(address="auto")
for name in ray.util.list_named_actors(all_namespaces=True):
    if "vllm" in name["name"] or "trainer" in name["name"]:
        try:
            actor = ray.get_actor(name["name"], namespace=name["namespace"])
            ray.kill(actor)
        except: pass
```

## Proposed Solutions

### 1. Server Startup Cleanup (Recommended)
On server start, kill all actors in the `tinker` namespace:
```python
# In app lifespan startup
actors = ray.util.list_named_actors(all_namespaces=True)
for actor_info in actors:
    if actor_info["namespace"] == "tinker":
        try:
            actor = ray.get_actor(actor_info["name"], namespace="tinker")
            ray.kill(actor)
            logger.info(f"Cleaned up stale actor: {actor_info['name']}")
        except Exception:
            pass
```

### 2. Reconnect to Existing Actors
Instead of failing on "name already taken", reconnect:
```python
try:
    self.server = ExtendedVLLMHttpServer.options(...).remote(...)
except ValueError as e:
    if "already taken" in str(e):
        self.server = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
```

### 3. Session-Based Actor Lifecycle
Tie actor lifetime to session heartbeats. Kill actors when session expires.

### 4. Admin Endpoints
Add `/api/v1/admin/cleanup_actors` endpoint to manually trigger cleanup.

## Impact

- **Severity**: High - blocks all training and new sampling sessions
- **Frequency**: Every server restart when previous actors exist
- **User Impact**: Users see failures, must contact admin for cleanup

## Files Affected

- `tinker_server/backend/multi_lora_engine.py` - vLLM actor creation
- `tinker_server/backend/verl_training.py` - Training actor creation
- `tinker_server/backend/session_manager.py` - Session/engine management
- `tinker_server/app.py` - Lifespan startup hook

## Related Logs

```
ValueError: The name tinker_vllm_qwen3-0.6b (namespace=tinker) is already taken
```

```
Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}.
This is likely due to all cluster resources being claimed by actors.
```
