import json
from pathlib import Path
from urllib.parse import urlparse

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


def test_issue_190_archive_endpoint_redirect_for_tinker_sdk(tmp_path: Path) -> None:
    from tinker_server.routes import weights as weights_routes

    # Patch checkpoints root for this test module.
    weights_routes.CHECKPOINTS_DIR = str(tmp_path)

    run_id = "run-190"
    _mk_checkpoint(
        tmp_path,
        owner="anonymous",
        run_id=run_id,
        name="0001",
        checkpoint_type="training",
    )
    _mk_checkpoint(
        tmp_path,
        owner="anonymous",
        run_id=run_id,
        name="0002",
        checkpoint_type="sampler",
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    tinker_ua = {"User-Agent": "AsyncTinker/Python 0.13.1"}

    # Training checkpoint (canonical checkpoint_id includes "weights/").
    resp = client.get(
        f"/api/v1/training_runs/{run_id}/checkpoints/weights/0001/archive",
        headers=tinker_ua,
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "Location" in resp.headers
    assert "Expires" in resp.headers

    location = resp.headers["Location"]
    assert "direct=1" in location
    path = urlparse(location).path + "?" + urlparse(location).query
    direct = client.get(path)
    assert direct.status_code == 200
    assert direct.headers.get("content-type") == "application/gzip"
    assert int(direct.headers.get("content-length", "1")) >= 0
    assert direct.content[:2] == b"\x1f\x8b"  # gzip magic

    # Sampler checkpoint.
    resp2 = client.get(
        f"/api/v1/training_runs/{run_id}/checkpoints/sampler_weights/0002/archive",
        headers=tinker_ua,
        follow_redirects=False,
    )
    assert resp2.status_code == 302
    location2 = resp2.headers["Location"]
    assert "direct=1" in location2

    path2 = urlparse(location2).path + "?" + urlparse(location2).query
    direct2 = client.get(path2)
    assert direct2.status_code == 200
    assert direct2.content[:2] == b"\x1f\x8b"


def test_issue_190_invalid_checkpoint_is_404_even_for_tinker(tmp_path: Path) -> None:
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    run_id = "run-190-missing"

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(
        f"/api/v1/training_runs/{run_id}/checkpoints/weights/does-not-exist/archive",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1"},
        follow_redirects=False,
    )
    assert resp.status_code == 404
