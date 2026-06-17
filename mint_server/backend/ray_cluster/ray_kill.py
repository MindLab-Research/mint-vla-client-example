from __future__ import annotations

import structlog
import os
import time
import traceback
from typing import Any

import ray

from mint_server.observability.logging_context import record_ray_cluster_op_otel
from mint_server.backend.ray_cluster.model_actor_pg_names import actor_placement_group_names
from mint_server.backend.ray_cluster.ray_placement_groups import remove_named_placement_group

logger = structlog.get_logger(__name__)

DEFAULT_VERIFY_TIMEOUT_S = 10.0
DEFAULT_VERIFY_POLL_INTERVAL_S = 0.25


class ActorStillAliveError(RuntimeError):
    """Raised when a kill path cannot prove the named actor disappeared."""


def _verify_named_actor_absent(
    *,
    actor_name: str,
    namespace: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify actor disappearance for actor_name={actor_name!r} "
                f"namespace={namespace!r}: {type(e).__name__}: {e}"
            ) from e

        if time.monotonic() >= deadline:
            raise ActorStillAliveError(
                f"ray.kill returned but actor still exists: actor_name={actor_name!r} "
                f"namespace={namespace!r} timeout_s={timeout_s}"
            )
        time.sleep(poll_interval_s)

def _remove_placement_group_for_actor_name(actor_name: str | None, namespace: str | None) -> None:
    if not actor_name:
        return

    for pg_name in actor_placement_group_names(actor_name, namespace):
        try:
            removed = remove_named_placement_group(pg_name, namespace=namespace)
        except Exception:
            logger.warning("failed_remove____s___s", placement_group=pg_name)
            continue
        if removed:
            logger.warning("removed", placement_group=pg_name)


def kill(
    actor: Any,
    *,
    reason: str,
    actor_name: str | None = None,
    namespace: str | None = None,
    no_restart: bool | None = None,
    verify_absent: bool = False,
    verify_timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
    verify_poll_interval_s: float = DEFAULT_VERIFY_POLL_INTERVAL_S,
    **context: Any,
) -> None:
    parts = [f"[ray.kill] reason={reason}"]
    if actor_name:
        parts.append(f"actor_name={actor_name}")
    if namespace:
        parts.append(f"namespace={namespace}")
    for k in sorted(context):
        v = context[k]
        parts.append(f"{k}={v}")

    msg = " ".join(parts)
    if os.environ.get("MINT_LOG_KILL_STACK", "0") == "1":
        msg += "\n" + "".join(traceback.format_stack(limit=30))
    logger.warning(msg)

    _kill_t0 = time.perf_counter()
    _kill_ok = True
    try:
        if no_restart is None:
            ray.kill(actor)
        else:
            ray.kill(actor, no_restart=no_restart)
    except Exception:
        _kill_ok = False
        raise
    finally:
        record_ray_cluster_op_otel(
            op="actor_kill",
            status="ok" if _kill_ok else "error",
            duration_s=time.perf_counter() - _kill_t0,
            actor_name=str(actor_name or "unknown"),
            reason=str(reason),
        )

    _remove_placement_group_for_actor_name(actor_name, namespace)
    if verify_absent:
        if not actor_name or not namespace:
            raise ValueError("verify_absent=True requires actor_name and namespace")
        _verify_named_actor_absent(
            actor_name=actor_name,
            namespace=namespace,
            timeout_s=verify_timeout_s,
            poll_interval_s=verify_poll_interval_s,
        )
