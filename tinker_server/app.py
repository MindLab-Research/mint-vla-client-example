"""FastAPI application for tinker-server."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backend.multi_lora_engine import MultiLoRAInferenceEngine
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes both inference SessionManager and training components
    on startup, shuts down all sessions on application exit.
    """
    # ==========================================================================
    # Inference: Initialize SessionManager
    # ==========================================================================
    logger.info(f"Initializing inference session manager with model: {config.model_path}")

    inference_manager = SessionManager(
        model_path=config.model_path,
        tensor_parallel_size=config.tensor_parallel_size,
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
    # Multi-LoRA: Initialize shared inference engine (optional)
    # ==========================================================================
    multi_lora_engine: MultiLoRAInferenceEngine | None = None

    if config.enable_multi_lora:
        logger.info(
            f"Initializing Multi-LoRA engine: max_loras={config.max_loras}, "
            f"max_cpu_loras={config.max_cpu_loras}, max_lora_rank={config.max_lora_rank}"
        )

        multi_lora_engine = MultiLoRAInferenceEngine(
            model_path=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_loras=config.max_loras,
            max_cpu_loras=config.max_cpu_loras,
            max_lora_rank=config.max_lora_rank,
        )
        await multi_lora_engine.initialize()

        # Register with session manager
        inference_manager.set_multi_lora_engine(multi_lora_engine)
        logger.info("Multi-LoRA engine initialized and registered")
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

    # Weights router also needs training components
    weights.training_manager = train_manager
    weights.training_engine = train_engine

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

    # Shutdown multi-LoRA engine
    if multi_lora_engine is not None:
        await multi_lora_engine.shutdown()
        logger.info("Multi-LoRA engine shutdown")


app = FastAPI(
    lifespan=lifespan,
    title="Tinker Server",
    description="Tinker-compatible inference server wrapping verl",
    version="0.1.0",
)

# Paths that don't require authentication
UNAUTHENTICATED_PATHS = {"/api/v1/healthz", "/"}


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Validate X-API-Key header for all protected endpoints."""
    if request.url.path in UNAUTHENTICATED_PATHS:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key", "")
    if not config.validate_api_key(api_key):
        logger.warning(f"Invalid API key from {request.client.host if request.client else 'unknown'}")
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
