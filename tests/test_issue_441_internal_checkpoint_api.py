import io
import tarfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tinker_server.routes import internal as internal_routes


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_app(user_data: dict) -> TestClient:
    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request: Request, call_next):
        request.state.user_data = user_data
        return await call_next(request)

    app.include_router(internal_routes.router, prefix="/internal")
    return TestClient(app)


def test_issue_441_internal_checkpoint_list_requires_catalog(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: False)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 503, resp.text
    assert resp.json() == {"detail": "Checkpoint catalog unavailable"}


def test_issue_441_internal_checkpoint_list_uses_catalog_when_available(monkeypatch, tmp_path: Path) -> None:
    ckpt_id = "5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11"

    async def _list_catalog_checkpoints(*, owner_id: str | None, is_admin: bool):
        assert owner_id == "owner-a"
        assert is_admin is False
        return [
            {
                "ckpt_id": ckpt_id,
                "owner_id": "owner-a",
                "model_id": "model-catalog",
                "raw_checkpoint_id": "ckpt-catalog",
                "checkpoint_type": "sampler",
                "model_name": "Qwen/Qwen3-0.6B",
                "checkpoint_created_at": "2026-01-03T00:00:00Z",
                "published_at": "2026-01-03T00:00:01Z",
                "size_bytes": 2048,
                "storage_root": str(tmp_path),
            }
        ]

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "list_catalog_checkpoints", _list_catalog_checkpoints)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert [item["checkpoint_id"] for item in resp.json()["checkpoints"]] == [ckpt_id]


def test_issue_441_internal_checkpoint_list_accepts_uuid_catalog_ids(monkeypatch, tmp_path: Path) -> None:
    ckpt_id = uuid.UUID("5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11")

    async def _list_catalog_checkpoints(*, owner_id: str | None, is_admin: bool):
        assert owner_id == "owner-a"
        assert is_admin is False
        return [
            {
                "ckpt_id": ckpt_id,
                "owner_id": "owner-a",
                "model_id": "model-catalog",
                "raw_checkpoint_id": "ckpt-catalog",
                "checkpoint_type": "sampler",
                "model_name": "Qwen/Qwen3-0.6B",
                "checkpoint_created_at": "2026-01-03T00:00:00Z",
                "published_at": "2026-01-03T00:00:01Z",
                "size_bytes": 2048,
                "storage_root": str(tmp_path),
            }
        ]

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "list_catalog_checkpoints", _list_catalog_checkpoints)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert [item["checkpoint_id"] for item in resp.json()["checkpoints"]] == [str(ckpt_id)]


def test_issue_441_internal_checkpoint_list_returns_503_when_catalog_query_fails(monkeypatch) -> None:
    async def _scan_checkpoints_from_catalog(user_id: str | None, *, is_admin: bool = False):
        assert user_id == "owner-a"
        assert is_admin is False
        raise RuntimeError("catalog offline")

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "_scan_checkpoints_from_catalog", _scan_checkpoints_from_catalog)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 503, resp.text
    assert resp.json() == {"detail": "Checkpoint catalog unavailable"}


def test_issue_441_internal_checkpoint_archive_uses_catalog_entry(monkeypatch, tmp_path: Path) -> None:
    storage_root = tmp_path / "tos"
    ckpt_dir = storage_root / "owner-a" / "model-catalog" / "ckpt-catalog" / "sampler"
    _touch(ckpt_dir / "adapter_model.safetensors", b"weights")

    async def _get_catalog_checkpoint(checkpoint_id: str, *, owner_id: str | None, is_admin: bool):
        assert checkpoint_id == "5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11"
        assert owner_id == "owner-a"
        assert is_admin is False
        return {
            "ckpt_id": checkpoint_id,
            "owner_id": "owner-a",
            "model_id": "model-catalog",
            "raw_checkpoint_id": "ckpt-catalog",
            "checkpoint_type": "sampler",
            "model_name": "Qwen/Qwen3-0.6B",
            "checkpoint_created_at": "2026-01-03T00:00:00Z",
            "published_at": "2026-01-03T00:00:01Z",
            "size_bytes": 2048,
            "storage_root": str(storage_root),
        }

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "get_catalog_checkpoint", _get_catalog_checkpoint)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="ckpt-catalog.tar.gz"'

    archive = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    assert "sampler/adapter_model.safetensors" in archive.getnames()


def test_issue_441_internal_checkpoint_archive_returns_404_when_catalog_path_missing(monkeypatch, tmp_path: Path) -> None:
    async def _get_catalog_checkpoint(checkpoint_id: str, *, owner_id: str | None, is_admin: bool):
        assert checkpoint_id == "catalog-id"
        assert owner_id == "owner-a"
        assert is_admin is False
        return {
            "ckpt_id": checkpoint_id,
            "owner_id": "owner-a",
            "model_id": "model-catalog",
            "raw_checkpoint_id": "missing-ckpt",
            "checkpoint_type": "sampler",
            "model_name": "Qwen/Qwen3-0.6B",
            "checkpoint_created_at": "2026-01-03T00:00:00Z",
            "published_at": "2026-01-03T00:00:01Z",
            "size_bytes": 2048,
            "storage_root": str(tmp_path / "missing-root"),
        }

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "get_catalog_checkpoint", _get_catalog_checkpoint)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/catalog-id/archive")
    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": "Checkpoint not found"}


def test_issue_441_internal_checkpoint_archive_requires_catalog(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: False)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/catalog-id/archive")
    assert resp.status_code == 503, resp.text
    assert resp.json() == {"detail": "Checkpoint catalog unavailable"}
