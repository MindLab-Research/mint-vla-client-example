from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_tinker_client_config_route_returns_sdk_bootstrap_defaults() -> None:
    from mint_server.routes import service

    app = FastAPI()
    app.include_router(service.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post("/api/v1/client/config", json={"sdk_version": "0.18.0"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pjwt_auth_enabled": False,
        "credential_default_source": "api_key",
        "sample_dispatch_bytes_semaphore_size": 10 * 1024 * 1024,
        "inflight_response_bytes_semaphore_size": 50 * 1024 * 1024,
    }
