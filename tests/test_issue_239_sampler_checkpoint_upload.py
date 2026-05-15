import io
import json
import tarfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_tar_gz_bytes(root: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel_path, data in files.items():
            name = f"{root}/{rel_path}"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_issue_239_sampler_only_upload_is_supported(tmp_path: Path) -> None:
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)

    payload = _make_tar_gz_bytes(
        "ckpt_sampler",
        {
            "adapter_model.safetensors": b"dummy-lora",
            "adapter_config.json": json.dumps(
                {"base_model_name_or_path": "Qwen/Qwen3-0.6B"}
            ).encode("utf-8"),
        },
    )

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.user_data = {"user_id": "anonymous"}
        return await call_next(request)

    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/checkpoints/upload",
        files={"file": ("ckpt.tar.gz", payload, "application/gzip")},
    )
    assert resp.status_code == 200, resp.text
    ckpt_id = resp.json()["checkpoint_id"]

    meta = json.loads((tmp_path / "anonymous" / ckpt_id / "metadata.json").read_text("utf-8"))
    assert meta["checkpoint_type"] == "sampler"
    assert meta["optimizer_present"] is False


def test_issue_239_openpi_sampler_upload_is_supported(tmp_path: Path) -> None:
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)

    payload = _make_tar_gz_bytes(
        "ckpt_openpi_sampler",
        {
            "params/_METADATA": b"orbax",
            "assets/physical-intelligence/libero/norm_stats.json": b"{}",
        },
    )

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.user_data = {"user_id": "anonymous"}
        return await call_next(request)

    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/checkpoints/upload",
        files={"file": ("ckpt_openpi.tar.gz", payload, "application/gzip")},
    )
    assert resp.status_code == 200, resp.text
    ckpt_id = resp.json()["checkpoint_id"]

    meta = json.loads((tmp_path / "anonymous" / ckpt_id / "metadata.json").read_text("utf-8"))
    assert meta["checkpoint_type"] == "sampler"
    assert meta["optimizer_present"] is False


def test_issue_239_training_declared_without_optimizer_is_rejected(tmp_path: Path) -> None:
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)

    payload = _make_tar_gz_bytes(
        "ckpt_bad",
        {
            "adapter_model.safetensors": b"dummy-lora",
            "metadata.json": json.dumps({"checkpoint_type": "training"}).encode("utf-8"),
        },
    )

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.user_data = {"user_id": "anonymous"}
        return await call_next(request)

    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/checkpoints/upload",
        files={"file": ("ckpt.tar.gz", payload, "application/gzip")},
    )
    assert resp.status_code == 400
    assert "declares checkpoint_type" in resp.text
