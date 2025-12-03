"""FastAPI application for tinker-server."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .backend.session_manager import SessionManager
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

    Initializes the SessionManager on startup and
    shuts down all sessions on application exit.
    """
    # Startup: initialize session manager
    logger.info(f"Initializing session manager with model: {config.model_path}")

    manager = SessionManager(
        model_path=config.model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )

    # Make session manager available to routes
    service.session_manager = manager
    sampling.session_manager = manager

    # Start background cleanup task
    await manager.start_cleanup_task()

    logger.info("Session manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down all sessions")
    await manager.shutdown_all()


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
