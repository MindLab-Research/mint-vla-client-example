"""Namespace-local read-only ConfigActor."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from ..config import PFS_PYTHONPATH, RAY_NAMESPACE, actor_runtime_env, apply_detached_actor_resources, otel_env_vars
from ..config_hydration import CONFIG_ACTOR_SELF_ENV
from ..runtime_config import ConfigSnapshot, build_config_snapshot, config_actor_name

logger = logging.getLogger(__name__)

_ACTOR_HANDLE: Any | None = None


class ConfigActorUnavailableError(RuntimeError):
    """Raised when the namespace-local ConfigActor is unavailable."""


class ConfigActorSnapshotMismatchError(RuntimeError):
    """Raised when an existing ConfigActor has a different snapshot fingerprint."""


def _ray_namespace() -> str:
    return RAY_NAMESPACE


def _actor_options(*, actor_name: str) -> dict[str, object]:
    options: dict[str, object] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(
            pythonpath=PFS_PYTHONPATH,
            extra={**otel_env_vars(), CONFIG_ACTOR_SELF_ENV: "1"},
            include_config_snapshot=False,
        ),
    }
    apply_detached_actor_resources(options, ray)
    return options


def _make_remote_class():
    @ray.remote(num_cpus=0, max_concurrency=64)
    class ConfigActor:
        def __init__(self, snapshot: dict[str, object]):
            self._snapshot = dict(snapshot)

        def get_snapshot(self) -> dict[str, object]:
            return dict(self._snapshot)

        def ping(self) -> dict[str, object]:
            return {
                "ok": True,
                "actor_name": self._snapshot.get("actor_name"),
                "ray_namespace": self._snapshot.get("ray_namespace"),
                "fingerprint": self._snapshot.get("fingerprint"),
            }

    return ConfigActor


def _ensure_fingerprint_matches(actor: Any, expected: ConfigSnapshot, *, timeout_s: float) -> dict[str, object]:
    snapshot = ray.get(actor.get_snapshot.remote(), timeout=timeout_s)
    actual = str(snapshot.get("fingerprint") or "")
    if actual != expected.fingerprint:
        raise ConfigActorSnapshotMismatchError(
            "Existing ConfigActor snapshot fingerprint mismatch: "
            f"actor_name={expected.actor_name!r} namespace={expected.ray_namespace!r} "
            f"expected={expected.fingerprint!r} actual={actual!r}"
        )
    return snapshot


def ensure_started(*, timeout_s: float = 30.0) -> dict[str, object]:
    """Ensure the namespace-local read-only ConfigActor exists and matches this process config."""
    global _ACTOR_HANDLE
    snapshot = build_config_snapshot(
        ray_namespace=_ray_namespace(),
        actor_name=config_actor_name(),
    )
    remote_cls = _make_remote_class()
    options = _actor_options(actor_name=snapshot.actor_name)
    actor = remote_cls.options(**options).remote(snapshot.to_dict())
    _ensure_fingerprint_matches(actor, snapshot, timeout_s=timeout_s)
    _ACTOR_HANDLE = actor
    logger.info(
        "ConfigActor ready actor=%s namespace=%s fingerprint=%s",
        snapshot.actor_name,
        snapshot.ray_namespace,
        snapshot.fingerprint,
    )
    return snapshot.to_dict()


async def async_ensure_started(*, timeout_s: float = 30.0) -> dict[str, object]:
    return await asyncio.to_thread(ensure_started, timeout_s=timeout_s)


def get_snapshot(*, timeout_s: float = 10.0) -> dict[str, object]:
    global _ACTOR_HANDLE
    actor_name = config_actor_name()
    if _ACTOR_HANDLE is None:
        try:
            _ACTOR_HANDLE = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception as e:
            raise ConfigActorUnavailableError(
                f"ConfigActor unavailable actor_name={actor_name!r} namespace={_ray_namespace()!r}"
            ) from e
    return ray.get(_ACTOR_HANDLE.get_snapshot.remote(), timeout=timeout_s)


async def async_get_snapshot(*, timeout_s: float = 10.0) -> dict[str, object]:
    return await asyncio.to_thread(get_snapshot, timeout_s=timeout_s)


def ping(*, timeout_s: float = 5.0) -> dict[str, object]:
    global _ACTOR_HANDLE
    actor_name = config_actor_name()
    if _ACTOR_HANDLE is None:
        try:
            _ACTOR_HANDLE = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception as e:
            raise ConfigActorUnavailableError(
                f"ConfigActor unavailable actor_name={actor_name!r} namespace={_ray_namespace()!r}"
            ) from e
    out = ray.get(_ACTOR_HANDLE.ping.remote(), timeout=timeout_s)
    if not isinstance(out, dict):
        raise TypeError(f"ConfigActor.ping returned non-dict: {type(out)}")
    if not bool(out.get("ok")):
        raise ConfigActorUnavailableError(f"ConfigActor ping failed: {out!r}")
    return out


async def async_ping(*, timeout_s: float = 5.0) -> dict[str, object]:
    return await asyncio.to_thread(ping, timeout_s=timeout_s)
