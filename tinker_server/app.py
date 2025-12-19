"""FastAPI application for tinker-server."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
    """Clean up stale Ray actors from previous server runs.

    Detached actors survive server restarts and can block resources.
    This function kills all actors in the 'tinker' namespace on startup.

    Only actors in DEAD or PENDING_CREATION state are killed to preserve
    actively running actors (e.g., if multiple servers share the cluster).
    """
    import os

    # Skip cleanup if disabled (useful for debugging)
    if os.environ.get("MINT_SKIP_ACTOR_CLEANUP", "").lower() in ("1", "true"):
        logger.info("Skipping actor cleanup (MINT_SKIP_ACTOR_CLEANUP=1)")
        return

    try:
        import ray
        from .backend.multi_lora_engine import PERSISTENT_NAMESPACE

        if not ray.is_initialized():
            ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

        # Get all named actors in the tinker namespace
        actors = ray.util.list_named_actors(all_namespaces=True)
        tinker_actors = [a for a in actors if a.get("namespace") == PERSISTENT_NAMESPACE]

        if not tinker_actors:
            logger.info("No stale actors found in tinker namespace")
            return

        logger.info(f"Found {len(tinker_actors)} actors in tinker namespace, checking for stale ones...")

        cleaned = 0
        for actor_info in tinker_actors:
            name = actor_info["name"]
            try:
                actor = ray.get_actor(name, namespace=PERSISTENT_NAMESPACE)

                # Check if actor is alive by calling a lightweight method
                # Use very short timeout - we just want to know if it responds
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=2)
                    logger.debug(f"Actor {name} is alive, keeping")
                except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    # Actor is dead or unresponsive
                    logger.info(f"Cleaning up dead/unresponsive actor: {name}")
                    try:
                        ray.kill(actor, no_restart=True)
                        cleaned += 1
                    except Exception as kill_err:
                        logger.warning(f"Failed to kill actor {name}: {kill_err}")

            except ValueError:
                # Actor name registered but no actor exists (shouldn't happen)
                logger.debug(f"Actor {name} not found (name registered but no actor)")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale actors")
        else:
            logger.info("All actors are alive, no cleanup needed")

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
    title="Tinker Server",
    description="Tinker-compatible inference server wrapping verl",
    version="0.1.0",
)

# Paths that don't require authentication
UNAUTHENTICATED_PATHS = {"/api/v1/healthz", "/"}

print("=== REGISTERING AUTH MIDDLEWARE ===", flush=True)


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Validate API key from X-API-Key or Authorization: Bearer header."""
    # Skip auth if no API key configured (dev mode)
    if not config.api_key:
        return await call_next(request)

    if request.url.path in UNAUTHENTICATED_PATHS:
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


# Root redirect to docs
@app.get("/")
async def root():
    """Redirect to API docs."""
    return {"message": "Tinker Server", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
