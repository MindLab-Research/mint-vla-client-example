from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from functools import lru_cache
from typing import Any, Awaitable


def control_plane_task_runtime_env() -> dict[str, object]:
    from mint_server.config import PFS_PYTHONPATH, actor_runtime_env

    return actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        include_config_snapshot=False,
        include_ray_attach_hints=False,
    )


def _discard_late_result(fut: asyncio.Future) -> None:
    try:
        fut.result()
    except BaseException:
        pass


def _silence_late_result(fut: asyncio.Future) -> None:
    if fut.done():
        _discard_late_result(fut)
        return
    fut.add_done_callback(_discard_late_result)


def _ray_ref_to_future(ref: Any) -> asyncio.Future:
    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return fut
        if isinstance(fut, concurrent.futures.Future):
            return asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return asyncio.ensure_future(fut)

    if hasattr(ref, "__await__"):
        return asyncio.ensure_future(ref)

    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


async def _await_ray_ref(ref: Any) -> Any:
    return await _ray_ref_to_future(ref)


async def _await_any(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _run_awaitable_sync(awaitable: Awaitable[Any], *, timeout_s: float | None = None) -> Any:
    if isinstance(awaitable, asyncio.Future):
        if awaitable.done():
            return awaitable.result()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "sync_get_ray_ref cannot wait on a pending asyncio.Future attached "
                "to the current event loop; use async_get_ray_ref instead"
            )

    async def _await() -> Any:
        if timeout_s is None:
            return await awaitable
        return await _await_with_ray_get_timeout(awaitable, timeout_s=float(timeout_s))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await())

    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(_await())
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_target, name="mint-sync-ray-ref-await", daemon=True)
    thread.start()
    join_timeout = None if timeout_s is None else max(float(timeout_s) + 1.0, 1.0)
    thread.join(join_timeout)
    if thread.is_alive():
        import ray

        raise ray.exceptions.GetTimeoutError(f"timed out after {float(timeout_s):.3f}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _await_shielded_with_timeout(awaitable: Awaitable[Any], *, timeout_s: float) -> Any:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=float(timeout_s))
    except asyncio.TimeoutError:
        _silence_late_result(task)
        raise
    except asyncio.CancelledError:
        _silence_late_result(task)
        raise


async def _await_with_ray_get_timeout(awaitable: Awaitable[Any], *, timeout_s: float) -> Any:
    try:
        return await _await_shielded_with_timeout(awaitable, timeout_s=float(timeout_s))
    except asyncio.TimeoutError as exc:
        import ray

        raise ray.exceptions.GetTimeoutError(f"timed out after {float(timeout_s):.3f}s") from exc


async def _await_ray_ref_with_timeout(ref: Any, *, timeout_s: float) -> Any:
    # Match ray.get(ref, timeout=...): timing out the wait must not cancel
    # the local Ray future or imply cancellation of the remote work.
    return await _await_shielded_with_timeout(_await_ray_ref(ref), timeout_s=float(timeout_s))


async def async_get_ray_ref(ref: Any, *, timeout_s: float | None = None) -> Any:
    if timeout_s is None:
        return await _await_ray_ref(ref)
    try:
        return await _await_ray_ref_with_timeout(ref, timeout_s=float(timeout_s))
    except asyncio.TimeoutError as exc:
        import ray

        raise ray.exceptions.GetTimeoutError(f"timed out after {float(timeout_s):.3f}s") from exc


def sync_get_ray_ref(ref: Any, *, timeout_s: float | None = None) -> Any:
    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, concurrent.futures.Future):
            try:
                return fut.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError as exc:
                import ray

                raise ray.exceptions.GetTimeoutError(f"timed out after {float(timeout_s):.3f}s") from exc
        if isinstance(fut, asyncio.Future) or hasattr(fut, "__await__"):
            return _run_awaitable_sync(fut, timeout_s=timeout_s)

    if hasattr(ref, "__await__"):
        return _run_awaitable_sync(ref, timeout_s=timeout_s)

    return ref


def is_actor_lookup_not_found(exc: Exception) -> bool:
    candidate: Exception | None = exc
    as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            candidate = as_instanceof_cause()
        except Exception:
            candidate = exc
    return isinstance(candidate, ValueError)


def _ensure_ray_initialized() -> None:
    import ray

    if ray.is_initialized():
        return

    # Do not attempt to init/reconnect Ray on HTTP request paths. Startup owns
    # the Ray driver connection; request paths only surface invariant breakage
    # or runtime disconnection.
    raise RuntimeError("Ray is not initialized")


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
    def _task(
        actor: Any,
        actor_name: str,
        namespace: str,
        base_model: str | None,
        reason: str,
        verify_absent: bool,
    ) -> bool:
        from mint_server.backend.ray_cluster import ray_kill

        if actor is None:
            try:
                actor = ray.get_actor(actor_name, namespace=namespace)
            except ValueError:
                return False
        ray_kill.kill(
            actor,
            reason=reason,
            actor_name=actor_name,
            namespace=namespace,
            base_model=base_model,
            no_restart=True,
            verify_absent=verify_absent,
        )
        try:
            pg = ray.util.get_placement_group(f"{actor_name}_pg")
            ray.util.remove_placement_group(pg)
        except Exception:
            pass
        return True

    return _task


@lru_cache(maxsize=1)
def _placement_group_table_remote():
    import ray

    @ray.remote(num_cpus=0)
    def _task() -> dict[str, Any]:
        # Keep this in a Ray task so the API-server event loop never blocks on
        # Ray control-plane calls like placement_group_table().
        try:
            tbl = ray.util.placement_group_table()
        except Exception:
            return {}
        return {} if not isinstance(tbl, dict) else tbl

    return _task


async def async_pending_gpu_pg_observation(*, timeout_s: float) -> dict[str, Any] | None:
    _ensure_ray_initialized()
    ref = _pending_gpu_pg_observation_remote().options(runtime_env=control_plane_task_runtime_env()).remote()
    return await _await_ray_ref_with_timeout(ref, timeout_s=float(timeout_s))


async def async_placement_group_table(*, timeout_s: float = 5.0) -> dict[str, Any]:
    _ensure_ray_initialized()
    ref = _placement_group_table_remote().options(runtime_env=control_plane_task_runtime_env()).remote()
    out = await _await_ray_ref_with_timeout(ref, timeout_s=float(timeout_s))
    if not isinstance(out, dict):
        raise TypeError(f"placement_group_table returned non-dict: {type(out)}")
    return out


async def async_lookup_actor_handle(actor_name: str, namespace: str, *, timeout_s: float = 15.0):
    _ensure_ray_initialized()
    ref = _lookup_actor_handle_remote().options(runtime_env=control_plane_task_runtime_env()).remote(
        str(actor_name),
        str(namespace),
    )
    return await _await_ray_ref_with_timeout(ref, timeout_s=float(timeout_s))


async def async_kill_named_actor(
    actor_name: str,
    namespace: str,
    *,
    actor_handle: Any | None = None,
    base_model: str | None,
    reason: str = "kill_named_actor_by_api",
    verify_absent: bool = False,
    timeout_s: float = 10.0,
) -> bool:
    _ensure_ray_initialized()
    ref = _kill_named_actor_remote().options(runtime_env=control_plane_task_runtime_env()).remote(
        actor_handle,
        str(actor_name),
        str(namespace),
        base_model,
        str(reason),
        bool(verify_absent),
    )
    return bool(await _await_ray_ref_with_timeout(ref, timeout_s=float(timeout_s)))
