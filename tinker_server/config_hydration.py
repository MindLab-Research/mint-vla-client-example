"""Hydrate actor process env from the namespace-local ConfigActor."""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping

from .runtime_env import env_nonempty

logger = logging.getLogger(__name__)

CONFIG_ACTOR_DEFAULT_NAME = "mint_config"
HYDRATE_ENV = "MINT_CONFIG_ACTOR_HYDRATE"
CONFIG_ACTOR_SELF_ENV = "MINT_CONFIG_ACTOR_SELF"

_HYDRATED = False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _config_actor_name(environ: MutableMapping[str, str]) -> str:
    value = env_nonempty(environ, "MINT_CONFIG_ACTOR_NAME")
    return value or CONFIG_ACTOR_DEFAULT_NAME


def _ray_namespace(environ: MutableMapping[str, str]) -> str:
    value = env_nonempty(environ, "MINT_RAY_NAMESPACE")
    return value or "tinker"


def hydrate_from_config_actor(
    environ: MutableMapping[str, str] | None = None,
    *,
    timeout_s: float = 10.0,
) -> bool:
    """Populate env from ConfigActor once in Ray actor processes.

    This runs before `tinker_server.config` loads its module-level settings. It
    intentionally avoids importing `tinker_server.config` or `runtime_config` to
    keep ConfigActor's own import path acyclic.
    """
    global _HYDRATED
    environ = os.environ if environ is None else environ
    if _HYDRATED:
        return True
    if not _truthy(environ.get(HYDRATE_ENV)):
        return False
    if _truthy(environ.get(CONFIG_ACTOR_SELF_ENV)):
        return False

    actor_name = _config_actor_name(environ)
    namespace = _ray_namespace(environ)
    try:
        import ray

        actor = ray.get_actor(actor_name, namespace=namespace)
        snapshot = ray.get(actor.get_snapshot.remote(), timeout=timeout_s)
    except Exception as e:
        raise RuntimeError(
            "Failed to hydrate runtime config from ConfigActor: "
            f"actor_name={actor_name!r} namespace={namespace!r}"
        ) from e

    actor_env = snapshot.get("actor_env") if isinstance(snapshot, dict) else None
    if not isinstance(actor_env, dict):
        raise RuntimeError(
            "ConfigActor snapshot is missing actor_env: "
            f"actor_name={actor_name!r} namespace={namespace!r}"
        )

    applied = 0
    for key, value in actor_env.items():
        if not isinstance(key, str) or value is None:
            continue
        environ[key] = str(value)
        applied += 1
    _HYDRATED = True
    logger.info(
        "Hydrated runtime config from ConfigActor actor=%s namespace=%s keys=%s",
        actor_name,
        namespace,
        applied,
    )
    return True
