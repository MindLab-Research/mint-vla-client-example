from __future__ import annotations

import logging
import os
import traceback
from typing import Any

import ray

logger = logging.getLogger(__name__)

def _remove_placement_group_for_actor_name(actor_name: str | None) -> None:
    if not actor_name:
        return

    pg_name = f"{actor_name}_pg"
    try:
        from ray.util import get_placement_group, remove_placement_group
    except Exception:
        return

    try:
        pg = get_placement_group(pg_name)
    except Exception:
        return

    try:
        remove_placement_group(pg)
        logger.warning(f"[ray.kill] removed placement_group={pg_name}")
    except Exception as e:
        logger.warning(f"[ray.kill] failed remove placement_group={pg_name}: {type(e).__name__}: {e}")


def kill(
    actor: Any,
    *,
    reason: str,
    actor_name: str | None = None,
    namespace: str | None = None,
    no_restart: bool | None = None,
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

    if no_restart is None:
        ray.kill(actor)
    else:
        ray.kill(actor, no_restart=no_restart)

    _remove_placement_group_for_actor_name(actor_name)
