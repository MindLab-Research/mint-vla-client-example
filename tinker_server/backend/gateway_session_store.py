"""Detached Ray store for gateway routing metadata.

This supports recovery after API server restarts when acting as a gateway/router:
- Sampling sessions created on upstreams need local routing state for /asample.
- Training model_ids created on upstreams need local routing state for training ops.
"""

from __future__ import annotations

import os
from typing import Any


def _ray_namespace() -> str:
    return (
        os.environ.get("TINKER_RAY_NAMESPACE")
        or os.environ.get("MINT_RAY_NAMESPACE")
        or "tinker"
    )


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
            self._sampling_sessions: dict[str, dict[str, str]] = {}
            self._training_models: dict[str, dict[str, str]] = {}

        def upsert_sampling_session(self, sampling_session_id: str, info: dict[str, str]) -> None:
            self._sampling_sessions[sampling_session_id] = dict(info)

        def get_sampling_session(self, sampling_session_id: str) -> dict[str, str] | None:
            return self._sampling_sessions.get(sampling_session_id)

        def delete_sampling_session(self, sampling_session_id: str) -> None:
            self._sampling_sessions.pop(sampling_session_id, None)

        def upsert_training_model(self, model_id: str, info: dict[str, str]) -> None:
            self._training_models[model_id] = dict(info)

        def get_training_model(self, model_id: str) -> dict[str, str] | None:
            return self._training_models.get(model_id)

        def delete_training_model(self, model_id: str) -> None:
            self._training_models.pop(model_id, None)

        def list(self) -> dict[str, Any]:
            return {
                "sampling_sessions": dict(self._sampling_sessions),
                "training_models": dict(self._training_models),
            }

    try:
        return _GatewaySessionStoreActor.options(
            name=name,
            namespace=namespace,
            lifetime="detached",
        ).remote()
    except Exception as e:
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

        addr = (os.environ.get("RAY_ADDRESS") or "").strip()
        if not addr:
            # Volcano head writes the canonical GCS IP to PFS.
            candidates: list[str] = []
            pfs_tinker_path = (os.environ.get("PFS_TINKER_PATH") or "").strip()
            if pfs_tinker_path:
                candidates.append(os.path.join(pfs_tinker_path, "ray_head_ip.txt"))
            candidates.extend(
                [
                    "/vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt",
                    "/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt",
                ]
            )
            for p in candidates:
                try:
                    ip = open(p, "r", encoding="utf-8").read().strip()
                except OSError:
                    continue
                if ip:
                    addr = f"{ip}:6379"
                    break

        init_ray(address=addr or "auto")
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


def upsert_training_model(*, model_id: str, upstream_alias: str, base_model: str) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(
        actor.upsert_training_model.remote(
            model_id,
            {"upstream_alias": upstream_alias, "base_model": base_model},
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


def delete_training_model(model_id: str) -> None:
    import ray

    _ensure_ray_initialized()
    actor = _get_or_create_actor()
    ray.get(actor.delete_training_model.remote(model_id))
