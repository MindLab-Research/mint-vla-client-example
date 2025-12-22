"""FastAPI application for tinker-server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .backend.multi_lora_engine import MultiModelInferenceManager
from .backend.session_manager import SessionManager
from .backend.training_session_manager import TrainingSessionManager
from .backend.verl_training import VerlTrainingEngine
from .config import config
from .routes import futures, sampling, service, training, weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def _cleanup_stale_actors() -> None:
    """Clean up stale Ray actors and register alive ones with ResourcePool.

    Detached actors survive server restarts and can block resources.
    This function:
    1. Kills dead/unresponsive actors in the 'tinker' namespace
    2. Registers alive actors with ResourcePool for proper GPU tracking
    """
    import os

    # Skip cleanup if disabled (useful for debugging)
    if os.environ.get("MINT_SKIP_ACTOR_CLEANUP", "").lower() in ("1", "true"):
        logger.info("Skipping actor cleanup (MINT_SKIP_ACTOR_CLEANUP=1)")
        return

    try:
        import ray
        from .backend.multi_lora_engine import PERSISTENT_NAMESPACE
        from .backend.resource_pool import get_resource_pool, ActorType

        if not ray.is_initialized():
            ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

        # Get all named actors in the tinker namespace
        actors = ray.util.list_named_actors(all_namespaces=True)
        tinker_actors = [a for a in actors if a.get("namespace") == PERSISTENT_NAMESPACE]

        if not tinker_actors:
            logger.info("No actors found in tinker namespace")
            return

        logger.info(f"Found {len(tinker_actors)} actors in tinker namespace, checking status...")

        resource_pool = get_resource_pool()
        cleaned = 0
        registered = 0

        for actor_info in tinker_actors:
            name = actor_info["name"]
            try:
                actor = ray.get_actor(name, namespace=PERSISTENT_NAMESPACE)

                # Check if actor is alive
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=2)

                    # Actor is alive - register it with ResourcePool
                    # Determine actor type and GPU count from name
                    if name.startswith("tinker_vllm_"):
                        actor_type = ActorType.VLLM
                        # vLLM GPU count depends on model - default 1, could query actor
                        num_gpus = 1  # Most models use 1 GPU
                    elif name.startswith("dense_trainer_pool_"):
                        actor_type = ActorType.DENSE
                        num_gpus = 1
                    elif name.startswith("persistent_megatron"):
                        actor_type = ActorType.MEGATRON
                        num_gpus = 8  # MoE uses 8 GPUs
                    else:
                        logger.debug(f"Unknown actor type for {name}, skipping registration")
                        continue

                    resource_pool.register(
                        actor_name=name,
                        actor_type=actor_type,
                        num_gpus=num_gpus,
                        actor_handle=actor,
                        namespace=PERSISTENT_NAMESPACE,
                    )
                    # Mark as ready since the actor passed health check
                    resource_pool.mark_ready(name)
                    registered += 1
                    logger.info(f"Registered existing actor: {name} ({actor_type.value}, {num_gpus} GPUs)")

                except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    # Actor is dead or unresponsive
                    logger.info(f"Cleaning up dead/unresponsive actor: {name}")
                    try:
                        ray.kill(actor, no_restart=True)
                        cleaned += 1
                    except Exception as kill_err:
                        logger.warning(f"Failed to kill actor {name}: {kill_err}")

            except ValueError:
                # Actor name registered but no actor exists
                logger.debug(f"Actor {name} not found (name registered but no actor)")

        logger.info(f"Actor cleanup complete: {cleaned} cleaned, {registered} registered")

    except Exception as e:
        # Don't fail startup if cleanup fails
        logger.warning(f"Actor cleanup failed (continuing anyway): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes both inference SessionManager and training components
    on startup, shuts down all sessions on application exit.
    """
    # ==========================================================================
    # Cleanup: Kill stale actors from previous server runs
    # ==========================================================================
    await _cleanup_stale_actors()

    # ==========================================================================
    # Inference: Initialize SessionManager
    # ==========================================================================
    logger.info(f"Initializing inference session manager with model: {config.model_path}")

    inference_manager = SessionManager(
        model_path=config.model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )

    # Make session manager available to routes
    service.session_manager = inference_manager
    sampling.session_manager = inference_manager

    # Start background cleanup task
    await inference_manager.start_cleanup_task()

    logger.info("Inference session manager initialized")

    # ==========================================================================
    # Multi-Model Inference: Initialize manager for dynamic engine creation
    # ==========================================================================
    multi_model_manager: MultiModelInferenceManager | None = None

    if config.enable_multi_lora:
        logger.info(
            f"Initializing Multi-Model Inference Manager: max_loras={config.max_loras}, "
            f"max_cpu_loras={config.max_cpu_loras}, max_lora_rank={config.max_lora_rank}"
        )

        # Create manager - engines are created lazily per model
        multi_model_manager = MultiModelInferenceManager(
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_loras=config.max_loras,
            max_cpu_loras=config.max_cpu_loras,
            max_lora_rank=config.max_lora_rank,
        )

        # Register with session manager
        inference_manager.set_multi_model_manager(multi_model_manager)
        logger.info("Multi-model inference manager initialized (engines created on-demand)")
    else:
        logger.info("Multi-LoRA disabled, using per-session engines")

    # ==========================================================================
    # Training: Initialize TrainingSessionManager and VerlTrainingEngine
    # ==========================================================================
    logger.info("Initializing training components")

    train_manager = TrainingSessionManager()
    train_engine = VerlTrainingEngine()
    await train_engine.initialize()

    # Make training components available to routes
    training.training_manager = train_manager
    training.training_engine = train_engine
    training.inference_manager = inference_manager  # For ephemeral save flow

    # Weights router also needs training components and inference manager
    weights.training_manager = train_manager
    weights.training_engine = train_engine
    weights.inference_manager = inference_manager  # For multi-LoRA sampling registration

    logger.info("Training components initialized")

    yield

    # ==========================================================================
    # Shutdown
    # ==========================================================================
    logger.info("Shutting down all sessions")

    # Shutdown training sessions
    await train_manager.shutdown_all(train_engine)

    # Shutdown inference sessions
    await inference_manager.shutdown_all()

    # Shutdown multi-model inference manager
    if multi_model_manager is not None:
        await multi_model_manager.shutdown_all()
        logger.info("Multi-model inference manager shutdown")


app = FastAPI(
    lifespan=lifespan,
    title="MinT",
    description="Mind Lab Toolkit - Training API for LLMs",
    version="0.1.0",
    docs_url=None,  # Disable built-in Swagger UI, /docs redirects to /doc/
)

# Paths that don't require authentication
UNAUTHENTICATED_PATHS = {"/api/v1/healthz", "/", "/doc", "/doc/", "/docs", "/docs/"}
# Path prefixes that don't require authentication
UNAUTHENTICATED_PREFIXES = ("/doc/", "/doc")


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Validate API key from X-API-Key or Authorization: Bearer header."""
    path = request.url.path

    # Skip auth if no API key configured (dev mode)
    if not config.api_key:
        return await call_next(request)

    # Skip auth for specific paths
    if path in UNAUTHENTICATED_PATHS:
        return await call_next(request)

    # Skip auth for paths with unauthenticated prefixes (e.g., /doc)
    if path.startswith(UNAUTHENTICATED_PREFIXES):
        return await call_next(request)

    # Try X-API-Key header first, then Authorization: Bearer
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

    if not config.validate_api_key(api_key):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key"},
        )

    return await call_next(request)


# Register routes with API prefix
app.include_router(service.router, prefix="/api/v1", tags=["service"])
app.include_router(sampling.router, prefix="/api/v1", tags=["sampling"])
app.include_router(futures.router, prefix="/api/v1", tags=["futures"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(weights.router, prefix="/api/v1", tags=["weights"])

# Redirects to documentation (must be defined BEFORE mount)
@app.get("/")
async def root():
    """Redirect root to documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/doc/", status_code=302)


@app.get("/doc")
async def doc_redirect():
    """Redirect /doc to /doc/ for consistent behavior."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/doc/", status_code=301)


@app.get("/docs")
async def docs_redirect():
    """Alias /docs to /doc."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/doc/", status_code=302)


@app.get("/docs/")
async def docs_slash_redirect():
    """Alias /docs/ to /doc/."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/doc/", status_code=302)


# Mount documentation static files
# Use MINT_DOC_PATH env var to override the default path
_doc_path = os.environ.get("MINT_DOC_PATH", str(Path(__file__).parent.parent / "mint-doc" / "out"))
if Path(_doc_path).exists():
    app.mount("/doc", StaticFiles(directory=_doc_path, html=True), name="documentation")
    logger.info(f"Documentation mounted at /doc from {_doc_path}")
else:
    logger.warning(f"Documentation directory not found at {_doc_path}, /doc will not be available")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
