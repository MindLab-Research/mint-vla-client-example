from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel

from ..auth_identity import can_manage_system_user_data
from ..auth_identity import get_user_data as _request_user_data
from .async_ray_control import (
    async_get_ray_ref,
    async_kill_named_actor,
    async_lookup_actor_handle,
    async_placement_group_table,
    is_actor_lookup_not_found,
)
from .model_actor_pg_names import actor_placement_group_names

logger = logging.getLogger(__name__)


class KillActorsRequest(BaseModel):
    actor_type: str
    model_name: str | None = None
    actor_name: str | None = None
    force: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class ActorListRequest:
    actor_type: str | None = None
    model_name: str | None = None
    refresh_metadata: bool = True


def require_admin(request: Request) -> None:
    from ..config import config as server_config

    if not server_config.auth_enabled:
        return
    user_data = getattr(request.state, "user_data", None)
    if not can_manage_system_user_data(user_data):
        raise HTTPException(status_code=403, detail="System management access required")


async def _augment_with_placement_groups(actors: list[dict]) -> None:
    try:
        timeout_s = float(os.environ.get("MINT_ACTORS_PG_TABLE_TIMEOUT_S", "2.0"))
        try:
            tbl = await async_placement_group_table(timeout_s=timeout_s)
        except asyncio.TimeoutError:
            return
        except Exception:
            return

        by_name: dict[str, dict] = {}
        for info in tbl.values():
            if isinstance(info, dict):
                name = info.get("name")
                if isinstance(name, str) and name:
                    by_name[name] = info

        for actor in actors:
            name = actor.get("actor_name")
            if not isinstance(name, str) or not name:
                continue
            namespace = actor.get("namespace")
            pg_name = next(
                (
                    candidate
                    for candidate in actor_placement_group_names(
                        name,
                        str(namespace) if namespace is not None else None,
                    )
                    if candidate in by_name
                ),
                None,
            )
            if pg_name is None:
                continue
            try:
                info = by_name.get(pg_name)
                if not isinstance(info, dict):
                    continue
                bundles = info.get("bundles") or {}
                if not isinstance(bundles, dict):
                    continue
                total_gpu = 0
                for bundle in bundles.values():
                    if isinstance(bundle, dict):
                        total_gpu += int(bundle.get("GPU", 0) or 0)
                actor["pg_name"] = pg_name
                actor["pg_bundle_count"] = len(bundles)
                actor["pg_total_gpus"] = int(total_gpu)
            except Exception:
                continue
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ray unavailable for actor inventory: {e}") from e


async def list_actor_inventory(req: ActorListRequest) -> dict[str, Any]:
    from .model_actor_supervisor import ActorType, get_model_actor_supervisor

    parsed_actor_type: ActorType | None = None
    if req.actor_type is not None:
        t = req.actor_type.strip().lower()
        allowed = {x.value for x in ActorType}
        if t not in allowed:
            raise HTTPException(status_code=422, detail=f"Invalid type {req.actor_type!r}; expected one of {sorted(allowed)}")
        parsed_actor_type = ActorType(t)

    try:
        pool = get_model_actor_supervisor()
        actors = await pool.async_list_actors(
            refresh_metadata=req.refresh_metadata,
            actor_type=parsed_actor_type,
            model_name=req.model_name,
        )
        total_gpus_used = await pool.async_total_gpus_used()
        await _augment_with_placement_groups(actors)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ray unavailable for actor inventory: {e}") from e
    return {"actors": actors, "total_gpus_used": total_gpus_used}


def _entry_actor_type_name(entry: object) -> str:
    raw = getattr(entry, "actor_type", None)
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def _entry_matches_kill_request(
    entry: object,
    *,
    actor_type: str,
    model_name: str | None,
    actor_name: str | None,
) -> bool:
    entry_name = str(getattr(entry, "actor_name", "") or "")
    if actor_name is not None and entry_name != actor_name:
        return False

    entry_type = _entry_actor_type_name(entry)
    if actor_type != "all" and entry_type != actor_type:
        return False

    if model_name is not None and str(getattr(entry, "base_model", "") or "") != model_name:
        return False

    return True


def _collect_kill_target_entries(
    *,
    actor_type: str,
    model_name: str | None,
    actor_name: str | None,
) -> list[object]:
    from .model_actor_supervisor import get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    return [
        entry
        for entry in pool.iter_entries()
        if _entry_matches_kill_request(
            entry,
            actor_type=actor_type,
            model_name=model_name,
            actor_name=actor_name,
        )
    ]


def _kill_target_snapshot(entries: list[object]) -> list[dict[str, object]]:
    return [
        {
            "actor_name": str(getattr(entry, "actor_name", "") or ""),
            "actor_type": _entry_actor_type_name(entry),
            "base_model": str(getattr(entry, "base_model", "") or ""),
            "current_session": getattr(entry, "current_session", None),
            "inflight_count": int(getattr(entry, "inflight_count", 0) or 0),
            "creating": bool(getattr(entry, "creating", False)),
            "protected": bool(getattr(entry, "protected", False)),
        }
        for entry in entries
    ]


def _request_audit_fields(request: Request) -> dict[str, object]:
    user_data = _request_user_data(request)
    client = getattr(request, "client", None)
    return {
        "client_host": getattr(client, "host", None),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "user_agent": request.headers.get("user-agent"),
        "origin": request.headers.get("origin"),
        "referer": request.headers.get("referer"),
        "user_id": user_data.get("user_id") if isinstance(user_data, dict) else None,
        "is_admin": bool(can_manage_system_user_data(user_data)),
    }


def _log_kill_request(
    request: Request,
    body: KillActorsRequest,
    *,
    stage: str,
    targets: list[dict[str, object]],
    detail: str | None = None,
    result: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "stage": stage,
        "actor_type": body.actor_type,
        "model_name": body.model_name,
        "actor_name": body.actor_name,
        "force": body.force,
        "reason": body.reason,
        "targets": targets,
    }
    payload.update(_request_audit_fields(request))
    if detail is not None:
        payload["detail"] = detail
    if result is not None:
        payload["result"] = result
    logger.info("[actors.kill] %s", payload)


def _raise_if_busy_kill_targets(
    *,
    request: Request,
    body: KillActorsRequest,
    targets: list[dict[str, object]],
) -> None:
    if body.force:
        return
    busy = [target for target in targets if int(target.get("inflight_count", 0) or 0) > 0]
    if not busy:
        return
    actor_list = ", ".join(str(target.get("actor_name") or "<unknown>") for target in busy)
    detail = (
        f"Refusing to kill busy actor(s): {actor_list}. "
        "Pass force=true to override."
    )
    _log_kill_request(request, body, stage="blocked_busy", targets=targets, detail=detail)
    raise HTTPException(status_code=409, detail=detail)


def _remove_actor_pg(actor_name: str, *, namespace: str | None = None) -> None:
    from .ray_placement_groups import PlacementGroupNotFoundError, get_named_placement_group
    import ray

    first_error: Exception | None = None
    removed_any = False
    for pg_name in actor_placement_group_names(actor_name, namespace):
        try:
            pg = get_named_placement_group(pg_name, namespace=namespace)
        except PlacementGroupNotFoundError:
            continue
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        ray.util.remove_placement_group(pg)
        removed_any = True
    if first_error is not None and not removed_any:
        raise first_error


async def _kill_exact_vllm_actor(*, actor_name: str) -> int:
    from .multi_lora_engine import PERSISTENT_NAMESPACE
    from .model_actor_supervisor import ActorType, ModelActorSupervisorStaleError, get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.VLLM:
        return 0

    namespace = entry.namespace if entry is not None else PERSISTENT_NAMESPACE
    try:
        actor = await async_lookup_actor_handle(actor_name, namespace)
    except Exception as exc:
        if not is_actor_lookup_not_found(exc):
            raise
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
        return 0

    try:
        await async_kill_named_actor(
            actor_name,
            namespace,
            actor_handle=actor,
            base_model=entry.base_model if entry is not None else None,
            reason="vllm_kill_by_actor_name",
        )
    except ModelActorSupervisorStaleError:
        raise
    pool.unregister(actor_name)
    _remove_actor_pg(actor_name)
    return 1


async def _kill_exact_megatron_actor(*, actor_name: str) -> int:
    from .megatron_distributed import PERSISTENT_NAMESPACE
    from .model_actor_supervisor import ActorType, get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.MEGATRON:
        return 0

    namespace = entry.namespace if entry is not None else PERSISTENT_NAMESPACE
    try:
        actor = await async_lookup_actor_handle(actor_name, namespace)
    except Exception as exc:
        if not is_actor_lookup_not_found(exc):
            raise
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
        return 0

    try:
        try:
            await async_get_ray_ref(actor.shutdown.remote(), timeout_s=10.0)
        except Exception:
            pass
        await async_kill_named_actor(
            actor_name,
            namespace,
            actor_handle=actor,
            base_model=entry.base_model if entry is not None else None,
            reason="kill_megatron_actor_by_name",
            verify_absent=True,
        )
    finally:
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
    return 1


async def _kill_exact_dense_actor(*, actor_name: str) -> int:
    from .model_actor_supervisor import ActorType, get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.DENSE:
        return 0
    if entry is None:
        return 0

    try:
        actor = await async_lookup_actor_handle(entry.actor_name, entry.namespace)
    except Exception as exc:
        if not is_actor_lookup_not_found(exc):
            raise
        _remove_actor_pg(entry.actor_name, namespace=entry.namespace)
        pool.unregister(entry.actor_name)
        return 0

    await async_kill_named_actor(
        entry.actor_name,
        entry.namespace,
        actor_handle=entry.actor_handle if entry.actor_handle is not None else actor,
        base_model=entry.base_model,
        reason="dense_kill_by_actor_name",
    )
    _remove_actor_pg(entry.actor_name, namespace=entry.namespace)
    pool.unregister(entry.actor_name)
    return 1


async def _kill_dense_actors(base_model: str | None) -> int:
    from .model_actor_supervisor import ActorType, get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    targets = [
        e
        for e in pool.iter_entries()
        if e.actor_type == ActorType.DENSE and (base_model is None or e.base_model == base_model)
    ]

    killed = 0
    for entry in targets:
        await async_kill_named_actor(
            entry.actor_name,
            entry.namespace,
            actor_handle=getattr(entry, "actor_handle", None),
            base_model=entry.base_model,
            reason="dense_kill_by_api",
        )
        _remove_actor_pg(entry.actor_name, namespace=entry.namespace)
        pool.unregister(entry.actor_name)
        killed += 1
    return killed


async def kill_actors(request: Request, body: KillActorsRequest) -> dict[str, Any]:
    t = body.actor_type.strip().lower()
    model_name = body.model_name
    actor_name = body.actor_name.strip() if body.actor_name else None

    targets = _kill_target_snapshot(
        _collect_kill_target_entries(
            actor_type=t,
            model_name=model_name,
            actor_name=actor_name,
        )
    )
    _log_kill_request(request, body, stage="received", targets=targets)
    _raise_if_busy_kill_targets(request=request, body=body, targets=targets)

    killed_by_type: dict[str, int] = {"vllm": 0, "megatron": 0, "dense": 0}

    if actor_name:
        if t == "all":
            raise HTTPException(status_code=422, detail="actor_name cannot be combined with actor_type=all")
        if t == "vllm":
            from .model_actor_supervisor import ModelActorSupervisorStaleError

            try:
                killed_by_type["vllm"] = await _kill_exact_vllm_actor(actor_name=actor_name)
            except ModelActorSupervisorStaleError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e
        elif t == "megatron":
            killed_by_type["megatron"] = await _kill_exact_megatron_actor(actor_name=actor_name)
        elif t == "dense":
            try:
                killed_by_type["dense"] = await _kill_exact_dense_actor(actor_name=actor_name)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Ray unavailable for dense actor kill: {e}") from e
        else:
            raise HTTPException(status_code=422, detail="actor_type must be one of: vllm, megatron, dense, all")
        result = {
            "killed": int(sum(killed_by_type.values())),
            "killed_by_type": killed_by_type,
        }
        _log_kill_request(request, body, stage="completed", targets=targets, result=result)
        return result

    if t in ("vllm", "all"):
        from .multi_lora_engine import kill_persistent_vllm_actor
        from .model_actor_supervisor import ModelActorSupervisorStaleError

        try:
            if t == "vllm":
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(model_name) else 0
            else:
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(None) else 0
        except ModelActorSupervisorStaleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    if t in ("megatron", "all"):
        from .megatron_distributed import kill_megatron_actor

        if t == "megatron":
            killed_by_type["megatron"] = 1 if kill_megatron_actor(model_name) else 0
        else:
            killed_by_type["megatron"] = 1 if kill_megatron_actor(None) else 0

    if t in ("dense", "all"):
        try:
            killed_by_type["dense"] = await _kill_dense_actors(model_name if t == "dense" else None)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ray unavailable for dense actor kill: {e}") from e

    if t not in ("vllm", "megatron", "dense", "all"):
        raise HTTPException(status_code=422, detail="actor_type must be one of: vllm, megatron, dense, all")

    result = {
        "killed": int(sum(killed_by_type.values())),
        "killed_by_type": killed_by_type,
    }
    _log_kill_request(request, body, stage="completed", targets=targets, result=result)
    return result
