"""Detached Ray store for gateway routing metadata.

This supports recovery after API server restarts when acting as a gateway/router:
- Sampling sessions created on upstreams need local routing state for /asample.
- Training model_ids created on upstreams need local routing state for training ops.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import otel_env_vars


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _actor_name() -> str:
    return os.environ.get("MINT_GATEWAY_SESSION_STORE_ACTOR_NAME", "tinker_gateway_session_store")


def _get_or_create_actor():
    import ray

    name = _actor_name()
    namespace = _ray_namespace()
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    @ray.remote
    class _GatewaySessionStoreActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._sampling_sessions: dict[str, dict[str, str]] = {}
            self._training_models: dict[str, dict[str, str | None]] = {}

        def upsert_sampling_session(self, sampling_session_id: str, info: dict[str, str]) -> None:
            self._sampling_sessions[sampling_session_id] = dict(info)

        def get_sampling_session(self, sampling_session_id: str) -> dict[str, str] | None:
            return self._sampling_sessions.get(sampling_session_id)

        def delete_sampling_session(self, sampling_session_id: str) -> None:
            self._sampling_sessions.pop(sampling_session_id, None)

        def upsert_training_model(self, model_id: str, info: dict[str, str | None]) -> None:
            self._training_models[model_id] = dict(info)

        def get_training_model(self, model_id: str) -> dict[str, str | None] | None:
            return self._training_models.get(model_id)

        def delete_training_model(self, model_id: str) -> None:
            self._training_models.pop(model_id, None)

        def list(self) -> dict[str, Any]:
            return {
                "sampling_sessions": dict(self._sampling_sessions),
                "training_models": dict(self._training_models),
            }

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )

    try:
        return _GatewaySessionStoreActor.options(
            **options
        ).remote()
    except Exception as e:
        # Concurrency: another process may have created the detached actor after our initial
        # ray.get_actor(name) check but before this .remote() call.
        try:
            return ray.get_actor(name, namespace=namespace)
        except Exception:
            raise RuntimeError(
                f"Failed to create detached gateway session store actor name={name!r} namespace={namespace!r}: "
                f"{type(e).__name__}: {e}"
            ) from e


def _ensure_ray_initialized() -> None:
    import ray

    if ray.is_initialized():
        return
    try:
        from tinker_server.ray_utils import init_ray

        init_ray(namespace=_ray_namespace(), ignore_reinit_error=True)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ray for gateway session store: {type(e).__name__}: {e}") from e
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized after init_ray() for gateway session store")


def upsert_sampling_session(*, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(
        actor.upsert_sampling_session.remote(
            sampling_session_id,
            {"upstream_alias": upstream_alias, "base_model": base_model},
        )
    )


def get_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    info = ray.get(actor.get_sampling_session.remote(sampling_session_id))
    if not isinstance(info, dict):
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return upstream_alias, base_model


def delete_sampling_session(sampling_session_id: str) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(actor.delete_sampling_session.remote(sampling_session_id))


def upsert_training_model(
    *,
    model_id: str,
    upstream_alias: str,
    base_model: str,
    owner_id: str | None = None,
) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(
        actor.upsert_training_model.remote(
            model_id,
            {
                "upstream_alias": upstream_alias,
                "base_model": base_model,
                "owner_id": owner_id,
            },
        )
    )


def get_training_model(model_id: str) -> tuple[str, str] | None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    info = ray.get(actor.get_training_model.remote(model_id))
    if not isinstance(info, dict):
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return upstream_alias, base_model


def get_training_model_info(model_id: str) -> dict[str, str | None] | None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    info = ray.get(actor.get_training_model.remote(model_id))
    if not isinstance(info, dict):
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    owner_id = info.get("owner_id")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if owner_id is not None and not isinstance(owner_id, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return {
        "upstream_alias": upstream_alias,
        "base_model": base_model,
        "owner_id": owner_id,
    }


def delete_training_model(model_id: str) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(actor.delete_training_model.remote(model_id))
