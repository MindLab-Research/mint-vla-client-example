from __future__ import annotations

import asyncio
import concurrent.futures
from functools import lru_cache
from typing import Any


async def _await_ray_ref(ref: Any) -> Any:
    if hasattr(ref, "__await__"):
        return await ref

    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return await fut
        if isinstance(fut, concurrent.futures.Future):
            return await asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return await fut

    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


def _ensure_ray_initialized() -> None:
    import ray

    if ray.is_initialized():
        return

    from ..config import RAY_NAMESPACE
    from ..ray_utils import init_ray

    init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)


@lru_cache(maxsize=1)
def _pending_gpu_pg_observation_remote():
    import ray

    @ray.remote(num_cpus=0)
    def _task() -> dict[str, Any] | None:
        tbl = ray.util.placement_group_table()
        candidates: set[str] = set()
        for info in tbl.values():
            if not isinstance(info, dict):
                continue
            name = info.get("name")
            if not isinstance(name, str) or not name:
                continue
            state = info.get("state")
            if state in ("CREATED", "REMOVED"):
                continue
            candidates.add(name)

        pending: list[str] = []
        for name in sorted(candidates):
            try:
                pg = ray.util.get_placement_group(name)
            except Exception:
                continue
            try:
                info = ray.util.placement_group_table(pg)
            except Exception:
                continue
            state = info.get("state")
            if state in ("CREATED", "REMOVED"):
                continue
            bundles = info.get("bundles") or {}
            total_gpu = 0.0
            for bundle in bundles.values():
                if isinstance(bundle, dict):
                    total_gpu += float(bundle.get("GPU", 0) or 0)
            if total_gpu <= 0:
                continue
            pending.append(name)

        if not pending:
            return None

        ar = ray.available_resources()
        cr = ray.cluster_resources()
        return {
            "reason": "pending_placement_groups",
            "pending_pg_count": len(pending),
            "pending_pg_names": pending[:20],
            "ray_gpu_available": float(ar.get("GPU", 0) or 0),
            "ray_gpu_total": float(cr.get("GPU", 0) or 0),
        }

    return _task


@lru_cache(maxsize=1)
def _lookup_actor_handle_remote():
    import ray

    @ray.remote(num_cpus=0)
    def _task(actor_name: str, namespace: str):
        return ray.get_actor(actor_name, namespace=namespace)

    return _task


@lru_cache(maxsize=1)
def _kill_named_actor_remote():
    import ray

    @ray.remote(num_cpus=0)
    def _task(actor_name: str, namespace: str, base_model: str | None) -> bool:
        from ..backend import ray_kill

        actor = ray.get_actor(actor_name, namespace=namespace)
        ray_kill.kill(
            actor,
            reason="dense_kill_by_api",
            actor_name=actor_name,
            namespace=namespace,
            base_model=base_model,
            no_restart=True,
        )
        try:
            pg = ray.util.get_placement_group(f"{actor_name}_pg")
            ray.util.remove_placement_group(pg)
        except Exception:
            pass
        return True

    return _task


async def async_pending_gpu_pg_observation(*, timeout_s: float) -> dict[str, Any] | None:
    ref = _pending_gpu_pg_observation_remote().remote()
    return await asyncio.wait_for(_await_ray_ref(ref), timeout=float(timeout_s))


async def async_lookup_actor_handle(actor_name: str, namespace: str, *, timeout_s: float = 5.0):
    _ensure_ray_initialized()
    ref = _lookup_actor_handle_remote().remote(str(actor_name), str(namespace))
    return await asyncio.wait_for(_await_ray_ref(ref), timeout=float(timeout_s))


async def async_kill_named_actor(
    actor_name: str,
    namespace: str,
    *,
    base_model: str | None,
    timeout_s: float = 10.0,
) -> bool:
    _ensure_ray_initialized()
    ref = _kill_named_actor_remote().remote(str(actor_name), str(namespace), base_model)
    return bool(await asyncio.wait_for(_await_ray_ref(ref), timeout=float(timeout_s)))
