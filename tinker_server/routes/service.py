"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- POST /session_heartbeat: Keep session alive
- POST /telemetry: Accept telemetry data (discarded)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..backend.session_heartbeat_store import session_heartbeat_store
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.types import (
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    TelemetryRequest,
    TelemetryResponse,
)
from ..server_info import get_server_info

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory session storage
sessions: dict[str, dict] = {}
sampling_sessions: dict[str, str] = {}  # sampling_session_id -> base_model

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return getattr(request.state, "user_data", None)


def _get_user_id(request: Request) -> str | None:
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("user_id")
    return None


@router.get("/healthz", response_model=None)
async def healthz() -> dict:
    """Health check endpoint.

    Returns HTTP 503 when the server can connect to Ray but Ray has pending GPU
    placement-group demand in the configured namespace. This indicates the API
    surface may be healthy while Ray-backed workloads are capacity-degraded.

    Also returns HTTP 503 when startup reconciliation recorded a degraded state
    (e.g., actor cleanup/reconciliation failed).
    """
    from ..health_state import get_startup_degraded_state

    degraded = get_startup_degraded_state()
    if degraded is not None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": degraded.get("reason", "startup_degraded"),
                "error": degraded.get("error", ""),
                "details": degraded.get("details", {}),
            },
        )
    try:
        import ray

        from ..config import RAY_NAMESPACE
        from ..ray_utils import init_ray

        if not ray.is_initialized():
            init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

        def _pending_gpu_pg_names_in_namespace() -> list[str]:
            tbl = ray.util.placement_group_table()
            candidates: set[str] = set()
            for info in tbl.values():
                if not isinstance(info, dict):
                    continue
                name = info.get("name")
                if not isinstance(name, str) or not name:
                    continue
                state = info.get("state")
                if state in ("CREATED", "REMOVED"):
                    continue
                candidates.add(name)

            pending: list[str] = []
            for name in sorted(candidates):
                try:
                    pg = ray.util.get_placement_group(name)
                except Exception:
                    continue
                try:
                    info = ray.util.placement_group_table(pg)
                except Exception:
                    continue
                state = info.get("state")
                if state in ("CREATED", "REMOVED"):
                    continue
                bundles = info.get("bundles") or {}
                total_gpu = 0.0
                for b in bundles.values():
                    if isinstance(b, dict):
                        total_gpu += float(b.get("GPU", 0) or 0)
                if total_gpu <= 0:
                    continue
                pending.append(name)
            return pending

        pending_pg_names = await asyncio.to_thread(_pending_gpu_pg_names_in_namespace)
        if pending_pg_names:
            ar = ray.available_resources()
            cr = ray.cluster_resources()
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "reason": "pending_placement_groups",
                    "pending_pg_count": len(pending_pg_names),
                    "pending_pg_names": pending_pg_names[:20],
                    "ray_gpu_available": float(ar.get("GPU", 0) or 0),
                    "ray_gpu_total": float(cr.get("GPU", 0) or 0),
                },
            )

        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "ray_unavailable",
                "error": f"{type(e).__name__}: {e}",
            },
        )


@router.get("/get_server_capabilities")
async def get_server_capabilities(http_request: Request) -> dict:
    """Return server capabilities for tinker client."""
    from ..backend.model_registry import get_model_config, list_supported_models
    from ..gateway import get_gateway_config, get_upstream_capabilities, upstream_for_model

    supported_local = list_supported_models()
    cfg = get_gateway_config()

    if cfg is None or not cfg.model_to_upstream:
        supported = supported_local
        models = [
            {
                "model_name": m,
                "max_context_length": get_model_config(m).max_model_len,
                "num_parameters": get_model_config(m).num_parameters,
            }
            for m in supported
        ]
        models.sort(key=lambda x: x["num_parameters"])
        return {
            "supported_models": models,
        }

    incoming_headers = dict(http_request.headers)
    remote_models = list(cfg.model_to_upstream.keys())

    # Fetch capabilities once per upstream alias that has at least one routed model.
    alias_to_caps: dict[str, dict[str, int]] = {}
    for alias in set(cfg.model_to_upstream.values()):
        upstream = cfg.upstreams.get(alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {alias!r}")
        try:
            alias_to_caps[alias] = await get_upstream_capabilities(
                upstream=upstream, incoming_headers=incoming_headers
            )
        except Exception as e:
            logger.exception("Upstream capabilities unavailable: %s", alias)
            raise HTTPException(
                status_code=503,
                detail=f"Upstream {alias!r} capabilities unavailable: {type(e).__name__}: {e}",
            )

    merged: list[dict] = []
    seen: set[str] = set()
    for m in supported_local + remote_models:
        if m in seen:
            continue
        seen.add(m)

        if m in cfg.model_to_upstream:
            upstream = upstream_for_model(m)
            if upstream is None:
                raise HTTPException(status_code=500, detail=f"Gateway misconfig: no upstream resolved for model {m!r}")
            caps = alias_to_caps.get(upstream.alias, {})
            config = None
            try:
                config = get_model_config(m)
            except (ValueError, KeyError):
                config = None

            if m in caps:
                max_len = int(caps[m])
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: model {m!r} not present in upstream {upstream.alias!r} capabilities",
                )
            num_params = config.num_parameters if config is not None else None
        else:
            config = get_model_config(m)
            max_len = int(config.max_model_len)
            num_params = config.num_parameters

        entry = {"model_name": m, "max_context_length": max_len}
        if num_params is not None:
            entry["num_parameters"] = num_params
        merged.append(entry)

    # Sort by num_parameters (models without num_parameters go last)
    merged.sort(key=lambda x: (x.get("num_parameters") is None, x.get("num_parameters", float("inf"))))

    return {
        "supported_models": merged,
    }


@router.get("/server_info")
async def server_info() -> dict:
    return get_server_info()


@router.post("/create_session")
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new session.

    Sessions are used to group related operations together.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "tags": request.tags,
        "metadata": request.user_metadata,
    }
    return CreateSessionResponse(session_id=session_id)


@router.post("/create_sampling_session")
async def create_sampling_session(
    request: CreateSamplingSessionRequest,
    http_request: Request,
) -> CreateSamplingSessionResponse:
    """Create a sampling session using the shared multi-LoRA engine.

    Uses the shared multi-LoRA engine for efficient session management:
    - Without model_path: Uses base model (no LoRA)
    - With model_path: Loads LoRA adapter into shared engine

    First call lazily initializes the multi-LoRA engine (~60s).
    Subsequent calls register sessions instantly (<1s).
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    sampling_session_id = str(uuid.uuid4())
    user_id = _get_user_id(http_request)

    # Determine base_model from request or infer from model_path
    base_model = request.base_model
    if not base_model and request.model_path:
        # Try to infer base_model from adapter_config.json
        adapter_path = _resolve_model_path(request.model_path, user_id=user_id)
        base_model = _infer_base_model_from_adapter(adapter_path)

    if not base_model:
        raise HTTPException(
            status_code=422,
            detail="base_model is required. Provide base_model or model_path with adapter_config.json containing base_model_name_or_path.",
        )

    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=base_model, http_request=http_request)

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(base_model)
        )

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return upstream sampling_session_id (tracking it for subsequent asample routing).
    from ..gateway import forward_json, register_remote_sampling_session, upstream_for_model

    upstream = upstream_for_model(base_model)
    if upstream is not None:
        payload = request.model_dump()
        payload["base_model"] = base_model
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_sampling_session",
                incoming_headers=dict(http_request.headers),
                json_body=payload,
                timeout_s=90.0,
            )
        except Exception:
            raise HTTPException(status_code=503, detail=f"Upstream {upstream.alias!r} create_sampling_session failed")

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        sampling_session_id_remote = data.get("sampling_session_id")
        if not isinstance(sampling_session_id_remote, str) or not sampling_session_id_remote:
            raise HTTPException(
                status_code=502, detail="Upstream create_sampling_session returned invalid sampling_session_id"
            )

        register_remote_sampling_session(
            sampling_session_id=sampling_session_id_remote,
            upstream_alias=upstream.alias,
            base_model=base_model,
        )
        return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id_remote)

    # Get or create engine for this model (dynamically creates vLLM actor if needed)
    # Do not block on vLLM cold-start here (can exceed reverse-proxy timeouts).
    # Warm vLLM in the background; /asample work will await readiness.
    async def _warm_engine() -> None:
        try:
            await session_manager.get_engine_for_model(base_model)
        except Exception as e:
            logger.warning(f"[create_sampling_session] warm engine failed: model={base_model} err={e}")

    asyncio.create_task(_warm_engine())

    if request.model_path:
        # Resolve adapter directory (file://, mint://, absolute path).
        adapter_path = _resolve_model_path(request.model_path, user_id=user_id)

        # Fast validation: ensure weights exist; loading happens on first /asample.
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=400, detail=f"Adapter weights not found: {weights_path}")

        # The MinT SDK `create_sampling_client(model_path=...)` does not expose lora_rank.
        # Use adapter_config.json (if present) as the source of truth to avoid
        # registering a mismatched rank (e.g., default 32 vs adapter r=64).
        lora_rank = request.lora_rank
        config_path = os.path.join(adapter_path, "adapter_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    adapter_config = json.load(f)
                inferred_rank = adapter_config.get("r")
                if isinstance(inferred_rank, int) and inferred_rank > 0:
                    if int(lora_rank) != inferred_rank:
                        logger.info(
                            f"[create_sampling_session] overriding lora_rank={lora_rank} "
                            f"with adapter_config.r={inferred_rank} for {adapter_path}"
                        )
                    lora_rank = inferred_rank
        except Exception as e:
            logger.warning(
                f"[create_sampling_session] failed to infer lora_rank from {config_path}: "
                f"{type(e).__name__}: {e}"
            )

        # Register session now; LoRA will be loaded lazily on first /asample.
        session_manager.register_multi_lora_session(
            session_id=sampling_session_id,
            base_model=base_model,
            lora_rank=lora_rank,
            adapter_path=adapter_path,
            lora_loaded=False,
        )
    else:
        # Base model (no LoRA): register session directly
        session_manager.register_base_model_session(sampling_session_id, base_model=base_model)

    # Store metadata
    sampling_sessions[sampling_session_id] = base_model

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


def _resolve_model_path(model_path: str, *, user_id: str | None) -> str:
    """Resolve model_path URI to filesystem path.

    Args:
        model_path: URI like file:///path, mint://{uuid}/..., or absolute path.

    Returns:
        Absolute filesystem path to adapter directory.
    """
    from ..checkpoints import get_checkpoints_dir, resolve_checkpoint_uri

    checkpoint_dir = get_checkpoints_dir()
    resolved = resolve_checkpoint_uri(model_path, checkpoint_dir, user_id=user_id)
    if user_id and user_id != "admin":
        resolved_real = os.path.realpath(resolved)
        checkpoints_real = os.path.realpath(checkpoint_dir)
        allowed_real = os.path.realpath(os.path.join(checkpoint_dir, user_id))
        if resolved_real.startswith(checkpoints_real + os.sep) and not resolved_real.startswith(
            allowed_real + os.sep
        ):
            raise HTTPException(status_code=403, detail="Access denied")
    return resolved


def _infer_base_model_from_adapter(adapter_path: str) -> str | None:
    """Infer base_model from adapter_config.json if present.

    Args:
        adapter_path: Filesystem path to adapter directory.

    Returns:
        base_model name if found, None otherwise.
    """
    import json
    import os

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path") or config.get("base_model")
    return None


def _load_adapter_from_path(adapter_path: str, lora_rank: int) -> tuple[dict, dict]:
    """Load LoRA adapter weights and config from disk.

    Args:
        adapter_path: Filesystem path to adapter directory.
        lora_rank: LoRA rank for config.

    Returns:
        (state_dict, peft_config) tuple.
    """
    import json
    import os

    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cannot load adapter_model.safetensors on this API host (missing torch-backed safetensors). "
                "Load adapters via a GPU worker environment."
            ),
        ) from e

    # Load weights
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=400,
            detail=f"Adapter weights not found: {weights_path}",
        )
    state_dict = load_file(weights_path)

    # Load config
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            peft_config = json.load(f)
    else:
        # Construct minimal config if missing
        # Include MLP modules for dense models (vLLM supports them).
        # Note: MoE expert LoRA is NOT supported by vLLM's FusedMoE kernel.
        peft_config = {
            "r": lora_rank,
            "lora_alpha": lora_rank,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }

    return state_dict, peft_config


@router.post("/session_heartbeat")
async def session_heartbeat(
    request: SessionHeartbeatRequest,
) -> SessionHeartbeatResponse:
    """Keep session alive.

    Accepts heartbeat and returns acknowledgment. Session validation not implemented.
    """
    session_heartbeat_store.update(request.session_id)
    if session_manager is not None:
        # Keep sampling sessions alive during long training phases between sample calls.
        session_manager.mark_session_inflight(request.session_id, 0)
    return SessionHeartbeatResponse()


@router.post("/telemetry")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Accept telemetry data from tinker client.

    Silently accepts and discards telemetry data.
    """
    return TelemetryResponse(status="accepted")


# =============================================================================
# Admin endpoints for actor management
# =============================================================================
def _require_admin(request: Request) -> None:
    """Raise 403 if not admin user."""
    from ..config import config as server_config
    if not server_config.auth_enabled:
        return
    user_data = getattr(request.state, "user_data", None)
    if not user_data or user_data.get("user_id") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def _augment_with_placement_groups(actors: list[dict]) -> None:
    try:
        import ray
        from ..config import RAY_NAMESPACE
        from ..ray_utils import init_ray

        if not ray.is_initialized():
            init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

        for a in actors:
            name = a.get("actor_name")
            if not isinstance(name, str) or not name:
                continue
            pg_name = f"{name}_pg"
            try:
                pg = ray.util.get_placement_group(pg_name)
                bundles = getattr(pg, "bundle_specs", None)
                if isinstance(bundles, list):
                    a["pg_name"] = pg_name
                    a["pg_bundle_count"] = len(bundles)
                    a["pg_total_gpus"] = sum(int(b.get("GPU", 0) or 0) for b in bundles if isinstance(b, dict))
            except Exception:
                continue
    except Exception:
        return


@router.get("/actors")
async def list_actors(
    request: Request,
    actor_type: str | None = Query(None, alias="type"),
    model_name: str | None = None,
) -> dict:
    """List actors in the unified ResourcePool. Admin only when auth enabled.

    Query params:
        type: Optional filter ("vllm" | "megatron" | "dense")
        model_name: Optional filter on ActorEntry.base_model
    """
    _require_admin(request)
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    actors = pool.list_actors()

    if actor_type is not None:
        t = actor_type.strip().lower()
        allowed = {x.value for x in ActorType}
        if t not in allowed:
            raise HTTPException(status_code=422, detail=f"Invalid type {actor_type!r}; expected one of {sorted(allowed)}")
        actors = [a for a in actors if a.get("actor_type") == t]

    if model_name is not None:
        actors = [a for a in actors if a.get("base_model") == model_name]

    _augment_with_placement_groups(actors)
    return {"actors": actors, "total_gpus_used": pool.total_gpus_used()}


class KillActorsRequest(BaseModel):
    """Request to kill actor(s)."""

    actor_type: str  # "vllm" | "megatron" | "dense" | "all"
    model_name: str | None = None  # optional per-type model filter


def _kill_dense_actors(base_model: str | None) -> int:
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    targets = [
        e
        for e in pool.iter_entries()
        if e.actor_type == ActorType.DENSE and (base_model is None or e.base_model == base_model)
    ]

    killed = 0
    try:
        import ray
        from ..backend import ray_kill
        from ..config import RAY_NAMESPACE
        from ..ray_utils import init_ray

        if not ray.is_initialized():
            init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

        for e in targets:
            try:
                actor = ray.get_actor(e.actor_name, namespace=e.namespace)
                ray_kill.kill(
                    actor,
                    reason="dense_kill_by_api",
                    actor_name=e.actor_name,
                    namespace=e.namespace,
                    base_model=e.base_model,
                    no_restart=True,
                )
            except Exception:
                pass
            pool.unregister(e.actor_name)
            try:
                pg = ray.util.get_placement_group(f"{e.actor_name}_pg")
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
            killed += 1
    except Exception:
        for e in targets:
            pool.unregister(e.actor_name)
            killed += 1
    return killed


@router.post("/actors/kill")
async def kill_actors(request: Request, body: KillActorsRequest) -> dict:
    """Kill actor(s) by type. Admin only when auth enabled."""
    _require_admin(request)

    t = body.actor_type.strip().lower()
    model_name = body.model_name

    killed_by_type: dict[str, int] = {"vllm": 0, "megatron": 0, "dense": 0}

    if t in ("vllm", "all"):
        from ..backend.multi_lora_engine import kill_persistent_vllm_actor
        from ..backend.resource_pool import ResourcePoolStaleError

        try:
            if t == "vllm":
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(model_name) else 0
            else:
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(None) else 0
        except ResourcePoolStaleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    if t in ("megatron", "all"):
        from ..backend.megatron_distributed import kill_megatron_actor

        if t == "megatron":
            killed_by_type["megatron"] = 1 if kill_megatron_actor(model_name) else 0
        else:
            killed_by_type["megatron"] = 1 if kill_megatron_actor(None) else 0

    if t in ("dense", "all"):
        killed_by_type["dense"] = _kill_dense_actors(model_name if t == "dense" else None)

    if t not in ("vllm", "megatron", "dense", "all"):
        raise HTTPException(status_code=422, detail="actor_type must be one of: vllm, megatron, dense, all")

    return {
        "killed": int(sum(killed_by_type.values())),
        "killed_by_type": killed_by_type,
    }
