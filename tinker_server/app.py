"""FastAPI application for tinker-server."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .backend import verl_inference
from .config import config
from .routes import futures, sampling, service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes the verl inference engine on startup and
    shuts it down on application exit.
    """
    # Startup: initialize verl engine
    logger.info(f"Initializing inference engine with model: {config.model_path}")

    engine = verl_inference.VerlInferenceEngine(
        model_path=config.model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )
    await engine.initialize()

    # Make engine available to routes
    verl_inference.verl_engine = engine
    sampling.verl_engine = engine

    logger.info("Inference engine initialized")

    yield

    # Shutdown
    logger.info("Shutting down inference engine")
    if verl_inference.verl_engine:
        await verl_inference.verl_engine.shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="Tinker Server",
    description="Tinker-compatible inference server wrapping verl",
    version="0.1.0",
)

# Register routes with API prefix
app.include_router(service.router, prefix="/api/v1", tags=["service"])
app.include_router(sampling.router, prefix="/api/v1", tags=["sampling"])
app.include_router(futures.router, prefix="/api/v1", tags=["futures"])


# Root redirect to docs
@app.get("/")
async def root():
    """Redirect to API docs."""
    return {"message": "Tinker Server", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
