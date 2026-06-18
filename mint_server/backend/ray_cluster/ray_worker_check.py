"""Helper to detect if the current process is a Ray worker (not a driver).

Used to guard Ray cluster-state APIs (ray.nodes, placement_group_table, etc.)
that trigger auto_init in Ray Client mode, spawning zombie GCS processes.
"""

from __future__ import annotations


def is_ray_worker_process() -> bool:
    """Check if this process is a Ray worker (not a driver).

    Returns True inside detached actor worker processes where calling
    ray.nodes() / placement_group_table() / ray._private.state APIs
    would trigger auto_init and spawn local GCS.
    """
    try:
        import ray

        ctx = ray.get_runtime_context()
        worker = getattr(ctx, "worker", None)
        if worker is None:
            return False
        worker_mode = getattr(worker, "mode", None)
        driver_mode = getattr(ray, "DRIVER_MODE", None)
        return worker_mode is not None and worker_mode != driver_mode
    except Exception:
        return False
