from __future__ import annotations

from fastapi import HTTPException, Request

from .config import ALLOW_UNSUPPORTED_MODELS


def _normalize_local_model_name(base_model: str) -> str | None:
    from .backend.model_registry import MODEL_CONFIGS, normalize_model_name

    if base_model in MODEL_CONFIGS:
        return base_model
    try:
        return normalize_model_name(base_model)
    except ValueError:
        return None


async def enforce_base_model_allowed(*, base_model: str, http_request: Request) -> str:
    """Reject `base_model` based on MINT_SUPPORTED_MODELS + ALLOW_UNSUPPORTED_MODELS policy.

    Returns:
        Canonical local model name (for MODEL_CONFIGS-backed models), or the original
        `base_model` for gateway-routed models.
    """
    from .backend.model_registry import MODEL_CONFIGS, list_supported_models
    from .gateway import get_upstream_capabilities, upstream_for_model

    local_name = _normalize_local_model_name(base_model)
    supported = set(list_supported_models())

    if local_name is not None:
        if not ALLOW_UNSUPPORTED_MODELS and local_name not in supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported base_model: {local_name!r}. "
                    "Not present in MINT_SUPPORTED_MODELS; set ALLOW_UNSUPPORTED_MODELS=1 to override."
                ),
            )
        return local_name

    if base_model in MODEL_CONFIGS:
        return base_model

    upstream = upstream_for_model(base_model)
    if upstream is not None:
        if not ALLOW_UNSUPPORTED_MODELS and base_model not in supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported base_model: {base_model!r}. "
                    "Not present in MINT_SUPPORTED_MODELS; set ALLOW_UNSUPPORTED_MODELS=1 to override."
                ),
            )
        caps = await get_upstream_capabilities(upstream=upstream, incoming_headers=dict(http_request.headers))
        if base_model not in caps:
            raise HTTPException(
                status_code=500,
                detail=f"Gateway misconfig: model {base_model!r} not present in upstream {upstream.alias!r} capabilities",
            )
        return base_model

    if not ALLOW_UNSUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported base_model: {base_model!r}. Not present in MODEL_CONFIGS and not gateway-routed.",
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported base_model: {base_model!r}. Not present in MODEL_CONFIGS and not gateway-routed.",
    )
