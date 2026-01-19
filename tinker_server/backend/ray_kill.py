from __future__ import annotations

import logging
import os
import traceback
from typing import Any

import ray

logger = logging.getLogger(__name__)


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

