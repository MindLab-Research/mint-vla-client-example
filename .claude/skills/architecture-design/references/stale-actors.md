# Stale Ray Actors Block Server Operations

## Summary

Detached Ray actors persist across server restarts and consume GPU resources indefinitely, causing new requests to fail or hang with \"Insufficient GPUs\" errors.

## Observed Behavior

1. Server appears stuck: health check timeouts, requests hang
2. Training fails: \"Insufficient GPUs: need 1, available 0 after eviction\"
3. Actor name conflicts: \"The name mint_vllm_qwen3-0.6b is already taken\"

## Root Cause

Ray actors created with `lifetime=\"detached\"` survive server process restarts and hold GPU allocations.

## Workaround

Before starting a new server process in a namespace, kill named actors in that namespace.

## Related Logs

```
ValueError: The name mint_vllm_qwen3-0.6b (namespace=mint) is already taken
```

```
Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}.
This is likely due to all cluster resources being claimed by actors.
```
