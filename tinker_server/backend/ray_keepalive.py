from __future__ import annotations

import asyncio
import time
from typing import Any

import ray

from ..config import config as server_config
from .resource_pool import get_resource_pool


async def ray_get_with_resource_pool_keepalive(
    ref: Any,
    *,
    actor_name: str,
    interval_s: float = 30.0,
    timeout_s: float | None = None,
) -> Any:
    """ray.get(ref) while periodically touching ResourcePool for actor_name.

    vLLM inference requests can run longer than ResourcePool's session idle
    timeout; without periodic touches, a busy vLLM actor can be considered idle
    and evicted mid-request.

    Implementation notes:
    - Uses ray.get(..., timeout=...) in a thread to avoid blocking the asyncio loop.
    - Touches before each timed wait; stops touching once the ref resolves.
    """
    if interval_s <= 0:
        interval_s = 30.0

    # Keepalive must be more frequent than ResourcePool's idle cutoff, otherwise
    # a busy actor can still appear idle and be evicted mid-request.
    idle_timeout_s = float(getattr(server_config, "resource_pool_session_idle_timeout_s", 300) or 300)
    interval_s = min(interval_s, max(0.5, idle_timeout_s / 4.0))

    pool = get_resource_pool()
    start = time.time()

    pool.mark_inflight(actor_name, +1)
    try:
        while True:
            pool.touch(actor_name)

            wait_s = interval_s
            if timeout_s is not None and timeout_s > 0:
                remaining = timeout_s - (time.time() - start)
                if remaining <= 0:
                    raise asyncio.TimeoutError(f"ray_get_timeout_s={timeout_s} actor_name={actor_name}")
                wait_s = min(wait_s, remaining)

            try:
                return await asyncio.to_thread(ray.get, ref, timeout=wait_s)
            except ray.exceptions.GetTimeoutError:
                continue
    finally:
        pool.mark_inflight(actor_name, -1)
