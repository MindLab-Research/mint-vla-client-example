"""Gateway routing metadata facade backed by TaskStateStore."""

from __future__ import annotations

from typing import Any

from mint_server.backend.stores.task_state_store import task_state_store


def _gateway_store_enabled() -> bool:
    try:
        from mint_server.gateway.gateway import get_gateway_config

        cfg = get_gateway_config()
    except Exception:
        return False
    return bool(cfg is not None and cfg.model_to_upstream)


def _sampling_tuple(info: dict[str, str] | None) -> tuple[str, str] | None:
    if not isinstance(info, dict):
        return None
    upstream_alias = info.get("upstream_alias")
    base_model = info.get("base_model")
    if not isinstance(upstream_alias, str) or not isinstance(base_model, str):
        return None
    if not upstream_alias or not base_model:
        return None
    return upstream_alias, base_model


def _training_info(info: dict[str, str | None] | None) -> dict[str, str | None] | None:
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


def upsert_sampling_session(*, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
    task_state_store.upsert_gateway_sampling_session(
        sampling_session_id=str(sampling_session_id),
        upstream_alias=str(upstream_alias),
        base_model=str(base_model),
    )


async def async_upsert_sampling_session(*, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
    await task_state_store.async_upsert_gateway_sampling_session(
        sampling_session_id=str(sampling_session_id),
        upstream_alias=str(upstream_alias),
        base_model=str(base_model),
    )


def get_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    return _sampling_tuple(task_state_store.get_gateway_sampling_session(sampling_session_id=str(sampling_session_id)))


async def async_get_sampling_session(sampling_session_id: str) -> tuple[str, str] | None:
    return _sampling_tuple(
        await task_state_store.async_get_gateway_sampling_session(sampling_session_id=str(sampling_session_id))
    )


def delete_sampling_session(sampling_session_id: str) -> None:
    task_state_store.delete_gateway_sampling_session(sampling_session_id=str(sampling_session_id))


async def async_delete_sampling_session(sampling_session_id: str) -> None:
    await task_state_store.async_delete_gateway_sampling_session(sampling_session_id=str(sampling_session_id))


def upsert_training_model(
    *,
    model_id: str,
    upstream_alias: str,
    base_model: str,
    owner_id: str | None = None,
) -> None:
    task_state_store.upsert_gateway_training_model(
        model_id=str(model_id),
        upstream_alias=str(upstream_alias),
        base_model=str(base_model),
        owner_id=owner_id,
    )


async def async_upsert_training_model(
    *,
    model_id: str,
    upstream_alias: str,
    base_model: str,
    owner_id: str | None = None,
) -> None:
    await task_state_store.async_upsert_gateway_training_model(
        model_id=str(model_id),
        upstream_alias=str(upstream_alias),
        base_model=str(base_model),
        owner_id=owner_id,
    )


def get_training_model(model_id: str) -> tuple[str, str] | None:
    info = _training_info(task_state_store.get_gateway_training_model(model_id=str(model_id)))
    if info is None:
        return None
    return str(info["upstream_alias"]), str(info["base_model"])


async def async_get_training_model(model_id: str) -> tuple[str, str] | None:
    info = _training_info(await task_state_store.async_get_gateway_training_model(model_id=str(model_id)))
    if info is None:
        return None
    return str(info["upstream_alias"]), str(info["base_model"])


def get_training_model_info(model_id: str) -> dict[str, str | None] | None:
    return _training_info(task_state_store.get_gateway_training_model(model_id=str(model_id)))


async def async_get_training_model_info(model_id: str) -> dict[str, str | None] | None:
    return _training_info(await task_state_store.async_get_gateway_training_model(model_id=str(model_id)))


def delete_training_model(model_id: str) -> None:
    task_state_store.delete_gateway_training_model(model_id=str(model_id))


async def async_delete_training_model(model_id: str) -> None:
    await task_state_store.async_delete_gateway_training_model(model_id=str(model_id))


def list_gateway_routes() -> dict[str, Any]:
    return task_state_store.list_gateway_routes()
