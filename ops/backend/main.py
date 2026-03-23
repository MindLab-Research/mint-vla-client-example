from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import OpsBackendConfig
from .service import DeployService, DeployServiceError, DirectMintOpsService


class RecycleActorRequest(BaseModel):
    actor_type: Literal["vllm", "megatron", "dense", "all"]
    model_name: str | None = None
    actor_name: str | None = None


class RebuildActorRequest(BaseModel):
    kind: Literal["vllm", "training"] = "vllm"
    models: list[str] = Field(default_factory=list)
    sample_ping: bool = False
    lora_rank: int = 16
    poll_timeout_s: float = 900.0
    poll_interval_s: float = 2.0


class AppState:
    def __init__(self, service: DeployService, config: OpsBackendConfig):
        self.service = service
        self.config = config


def get_service(app: FastAPI) -> DeployService:
    return app.state.container.service


def create_app(
    config: OpsBackendConfig | None = None,
    service: DeployService | None = None,
) -> FastAPI:
    repo_root = Path(__file__).resolve().parents[2]
    config = config or OpsBackendConfig.from_repo_root(repo_root)
    service = service or DirectMintOpsService(config)

    app = FastAPI(title="Mint Ops Console API", version="0.2.0")
    app.state.container = AppState(service=service, config=config)

    allow_any_origin = config.cors_origins == ("*",)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_any_origin else list(config.cors_origins),
        allow_credentials=not allow_any_origin,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _service() -> DeployService:
        return get_service(app)

    @app.get("/api/health")
    def health() -> dict:
        cfg = app.state.container.config
        return {
            "ok": True,
            "config_ready": cfg.is_ready,
            "mint_base_url": cfg.mint_base_url,
            "ray_address": cfg.ray_address,
            "api_key_configured": bool(cfg.api_key),
            "frontend_dist_present": bool(cfg.frontend_dist and cfg.frontend_dist.exists()),
            "notes": [
                "deploy state reads mint /api/v1/actors and ray state directly",
                "deploy state does not call mint /api/healthz",
                "deploy state does not scan local processes",
            ],
        }

    @app.get("/api/deploy/state")
    def deploy_state(
        actor_type: str | None = Query(default=None),
        model_query: str | None = Query(default=None),
        svc: DeployService = Depends(_service),
    ) -> dict:
        try:
            return svc.get_deploy_state(actor_type=actor_type, model_query=model_query)
        except DeployServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/deploy/actors/recycle")
    def recycle_actor(
        payload: RecycleActorRequest,
        svc: DeployService = Depends(_service),
    ) -> dict:
        try:
            return svc.recycle_actor(actor_type=payload.actor_type, model_name=payload.model_name, actor_name=payload.actor_name)
        except DeployServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/deploy/actors/rebuild")
    def rebuild_actor(
        payload: RebuildActorRequest,
        svc: DeployService = Depends(_service),
    ) -> dict:
        try:
            return svc.rebuild_actor(
                kind=payload.kind,
                models=payload.models,
                sample_ping=payload.sample_ping,
                lora_rank=payload.lora_rank,
                poll_timeout_s=payload.poll_timeout_s,
                poll_interval_s=payload.poll_interval_s,
            )
        except DeployServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    frontend_dist = config.frontend_dist
    assets_dir = frontend_dist / "assets" if frontend_dist else None
    if frontend_dist and frontend_dist.exists() and assets_dir and assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="ops-assets")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = frontend_dist / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
