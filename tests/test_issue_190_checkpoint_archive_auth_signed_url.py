import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mk_checkpoint(
    root: Path,
    *,
    owner: str,
    run_id: str,
    name: str,
    checkpoint_type: str,
) -> None:
    ckpt_dir = root / owner / run_id / name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"dummy-lora")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": name,
                "owner_id": None,
                "model_id": run_id,
                "model_name": "dummy",
                "created_at": "2026-02-26T00:00:00Z",
                "step": 0,
                "checkpoint_type": checkpoint_type,
                "optimizer_present": checkpoint_type == "training",
                "backend": "dense",
                "type": checkpoint_type,
            }
        ),
        encoding="utf-8",
    )


def test_issue_190_redirect_location_is_fetchable_without_headers_under_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tinker_server import config as config_module
    from tinker_server.app import api_key_auth_middleware
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)

    # Enable auth via hardcoded API key.
    monkeypatch.setattr(config_module.config, "api_key", "sekret", raising=False)
    monkeypatch.setattr(config_module.config, "token_secret_key", None, raising=False)

    run_id = "run-190-auth"
    _mk_checkpoint(
        tmp_path,
        owner="anonymous",
        run_id=run_id,
        name="0001",
        checkpoint_type="training",
    )

    app = FastAPI()
    app.middleware("http")(api_key_auth_middleware)
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(
        f"/api/v1/training_runs/{run_id}/checkpoints/weights/0001/archive?owner_id=anonymous",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1", "X-API-Key": "sekret"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    location = resp.headers["Location"]
    assert "direct=1" in location
    assert "download_token=" in location

    parsed = urlparse(location)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    direct = client.get(path)
    assert direct.status_code == 200
    assert direct.headers.get("content-type") == "application/gzip"
    assert direct.content[:2] == b"\x1f\x8b"
