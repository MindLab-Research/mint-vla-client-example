import io
import json
import tarfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tinker_server import checkpoints as checkpoint_helpers
from tinker_server.checkpoints import write_checkpoint_metadata
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


def _configure_checkpoint_roots(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    internal_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoint_helpers.CHECKPOINTS_DIR = str(tmp_path)
    checkpoint_helpers.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoint_helpers.RUNTIME_CHECKPOINTS_DIR = str(runtime_root)


def test_issue_441_internal_checkpoint_list_uses_three_level_layout(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "model-1" / "ckpt-training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    write_checkpoint_metadata(
        str(ckpt_dir),
        {
            "checkpoint_id": "ckpt-training",
            "owner_id": "owner-a",
            "model_id": "model-1",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "training",
            "type": "training",
            "storage_tier": "persistent_tos",
        },
    )

    hidden_dir = tmp_path / "owner-b" / "model-2" / "ckpt-hidden"
    _touch(hidden_dir / "adapter_model.safetensors")
    write_checkpoint_metadata(
        str(hidden_dir),
        {
            "checkpoint_id": "ckpt-hidden",
            "owner_id": "owner-b",
            "model_id": "model-2",
            "model_name": "Other/Model",
            "created_at": "2026-01-02T00:00:00Z",
            "checkpoint_type": "sampler",
            "type": "sampler",
        },
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload == {
        "checkpoints": [
            {
                "checkpoint_id": "model-1_ckpt-training",
                "model_name": "Qwen/Qwen3-0.6B",
                "created_at": "2026-01-01T00:00:00Z",
                "type": "training",
                "size_bytes": payload["checkpoints"][0]["size_bytes"],
            }
        ]
    }
    assert payload["checkpoints"][0]["size_bytes"] > 0


def test_issue_441_internal_checkpoint_archive_resolves_metadata_backed_id(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "model-archive" / "ckpt-archive"
    _touch(ckpt_dir / "adapter_model.safetensors", b"weights")
    _touch(ckpt_dir / "optimizer.pt", b"optimizer")
    write_checkpoint_metadata(
        str(ckpt_dir),
        {
            "checkpoint_id": "ckpt-archive",
            "owner_id": "owner-a",
            "model_id": "model-archive",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "training",
            "type": "training",
        },
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/model-archive_ckpt-archive/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="ckpt-archive.tar.gz"'

    archive = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    names = archive.getnames()
    assert "ckpt-archive/metadata.json" in names
    assert "ckpt-archive/adapter_model.safetensors" in names

    metadata = json.loads(archive.extractfile("ckpt-archive/metadata.json").read().decode("utf-8"))
    assert metadata["model_id"] == "model-archive"


def test_issue_441_internal_checkpoint_archive_keeps_owner_isolation(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "model-1" / "ckpt-private"
    _touch(ckpt_dir / "adapter_model.safetensors")
    write_checkpoint_metadata(
        str(ckpt_dir),
        {
            "checkpoint_id": "ckpt-private",
            "owner_id": "owner-a",
            "model_id": "model-1",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "sampler",
            "type": "sampler",
        },
    )

    client = _make_app({"user_id": "owner-b", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/model-1_ckpt-private/archive")
    assert resp.status_code == 403, resp.text


def test_issue_441_internal_checkpoint_list_keeps_same_checkpoint_name_across_models(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    for model_id in ("model-a", "model-b"):
        ckpt_dir = tmp_path / "owner-a" / model_id / "checkpoint-100"
        _touch(ckpt_dir / "adapter_model.safetensors")
        write_checkpoint_metadata(
            str(ckpt_dir),
            {
                "checkpoint_id": "checkpoint-100",
                "owner_id": "owner-a",
                "model_id": model_id,
                "model_name": f"Model/{model_id}",
                "created_at": "2026-01-01T00:00:00Z",
                "checkpoint_type": "sampler",
                "type": "sampler",
            },
        )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    checkpoint_ids = {entry["checkpoint_id"] for entry in resp.json()["checkpoints"]}
    assert checkpoint_ids == {"model-a_checkpoint-100", "model-b_checkpoint-100"}


def test_issue_441_internal_checkpoint_archive_catalog_fallback_respects_checkpoint_type(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_checkpoint_roots(tmp_path)

    training_dir = tmp_path / "owner-a" / "model-shared" / "same-id" / "training"
    sampler_dir = tmp_path / "owner-a" / "model-shared" / "same-id" / "sampler"

    _touch(training_dir / "adapter_model.safetensors", b"train-weights")
    _touch(training_dir / "optimizer.pt", b"optimizer")
    write_checkpoint_metadata(
        str(training_dir),
        {
            "checkpoint_id": "same-id",
            "owner_id": "owner-a",
            "model_id": "model-shared",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "training",
            "type": "training",
        },
    )

    _touch(sampler_dir / "adapter_model.safetensors", b"sampler-weights")
    write_checkpoint_metadata(
        str(sampler_dir),
        {
            "checkpoint_id": "same-id",
            "owner_id": "owner-a",
            "model_id": "model-shared",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "sampler",
            "type": "sampler",
        },
    )

    async def _get_catalog_checkpoint(checkpoint_id: str, *, owner_id: str | None, is_admin: bool):
        assert checkpoint_id == "catalog-id"
        assert owner_id == "owner-a"
        assert is_admin is False
        return {
            "ckpt_id": checkpoint_id,
            "owner_id": "owner-a",
            "model_id": "model-shared",
            "raw_checkpoint_id": "same-id",
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
    assert resp.status_code == 200, resp.text

    archive = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    names = archive.getnames()
    assert "sampler/adapter_model.safetensors" in names
    assert "training/optimizer.pt" not in names


def test_issue_441_internal_checkpoint_archive_catalog_fallback_does_not_cross_model(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "model-b" / "same-id" / "sampler"
    _touch(ckpt_dir / "adapter_model.safetensors", b"weights")
    write_checkpoint_metadata(
        str(ckpt_dir),
        {
            "checkpoint_id": "same-id",
            "owner_id": "owner-a",
            "model_id": "model-b",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "sampler",
            "type": "sampler",
        },
    )

    async def _get_catalog_checkpoint(checkpoint_id: str, *, owner_id: str | None, is_admin: bool):
        assert checkpoint_id == "catalog-id"
        assert owner_id == "owner-a"
        assert is_admin is False
        return {
            "ckpt_id": checkpoint_id,
            "owner_id": "owner-a",
            "model_id": "model-a",
            "raw_checkpoint_id": "same-id",
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


def test_issue_441_internal_checkpoint_archive_scopes_raw_id_to_request_owner(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    for owner_id in ("owner-a", "owner-b"):
        ckpt_dir = tmp_path / owner_id / "model-shared" / "same-id"
        _touch(ckpt_dir / "adapter_model.safetensors")
        write_checkpoint_metadata(
            str(ckpt_dir),
            {
                "checkpoint_id": "same-id",
                "owner_id": owner_id,
                "model_id": "model-shared",
                "model_name": "Qwen/Qwen3-0.6B",
                "created_at": "2026-01-01T00:00:00Z",
                "checkpoint_type": "sampler",
                "type": "sampler",
            },
        )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/same-id/archive")
    assert resp.status_code == 200, resp.text



def test_issue_441_internal_checkpoint_admin_ids_include_owner_prefix(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    for owner_id, model_id, checkpoint_id in (
        ("owner-a", "model-1", "ckpt-a"),
        ("owner-b", "model-2", "ckpt-b"),
    ):
        ckpt_dir = tmp_path / owner_id / model_id / checkpoint_id
        _touch(ckpt_dir / "adapter_model.safetensors", owner_id.encode("utf-8"))
        write_checkpoint_metadata(
            str(ckpt_dir),
            {
                "checkpoint_id": checkpoint_id,
                "owner_id": owner_id,
                "model_id": model_id,
                "model_name": f"Model/{model_id}",
                "created_at": "2026-01-01T00:00:00Z",
                "checkpoint_type": "sampler",
                "type": "sampler",
            },
        )

    client = _make_app({"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    checkpoint_ids = {entry["checkpoint_id"] for entry in resp.json()["checkpoints"]}
    assert checkpoint_ids == {"owner-a:model-1_ckpt-a", "owner-b:model-2_ckpt-b"}

    resp = client.get("/internal/v1/checkpoints/owner-b:model-2_ckpt-b/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="ckpt-b.tar.gz"'



def test_issue_441_internal_checkpoint_list_supports_legacy_two_level_metadata_layout(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "legacy-ckpt"
    _touch(ckpt_dir / "adapter_model.safetensors")
    write_checkpoint_metadata(
        str(ckpt_dir),
        {
            "checkpoint_id": "legacy-ckpt",
            "owner_id": "owner-a",
            "model_name": "Legacy/Model",
            "created_at": "2026-01-01T00:00:00Z",
            "checkpoint_type": "sampler",
            "type": "sampler",
        },
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert resp.json()["checkpoints"] == [
        {
            "checkpoint_id": "legacy-ckpt",
            "model_name": "Legacy/Model",
            "created_at": "2026-01-01T00:00:00Z",
            "type": "sampler",
            "size_bytes": resp.json()["checkpoints"][0]["size_bytes"],
        }
    ]

    resp = client.get("/internal/v1/checkpoints/legacy-ckpt/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="legacy-ckpt.tar.gz"'



def test_issue_441_internal_checkpoint_ambiguous_raw_id_across_models_returns_not_found(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    for model_id in ("model-a", "model-b"):
        ckpt_dir = tmp_path / "owner-a" / model_id / "same-id"
        _touch(ckpt_dir / "adapter_model.safetensors")
        write_checkpoint_metadata(
            str(ckpt_dir),
            {
                "checkpoint_id": "same-id",
                "owner_id": "owner-a",
                "model_id": model_id,
                "model_name": f"Model/{model_id}",
                "created_at": "2026-01-01T00:00:00Z",
                "checkpoint_type": "sampler",
                "type": "sampler",
            },
        )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/same-id/archive")
    assert resp.status_code == 404, resp.text



def test_issue_441_internal_checkpoint_admin_can_access_metadata_less_legacy_checkpoint(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    ckpt_dir = tmp_path / "owner-a" / "legacy-plain"
    _touch(ckpt_dir / "adapter_model.safetensors", b"legacy")

    client = _make_app({"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert resp.json()["checkpoints"] == [
        {
            "checkpoint_id": "owner-a_legacy-plain",
            "model_name": "unknown",
            "created_at": resp.json()["checkpoints"][0]["created_at"],
            "type": "training",
            "size_bytes": resp.json()["checkpoints"][0]["size_bytes"],
        }
    ]

    resp = client.get("/internal/v1/checkpoints/owner-a_legacy-plain/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="owner-a_legacy-plain.tar.gz"'



def test_issue_441_internal_checkpoint_user_list_includes_own_metadata_less_legacy_checkpoint(tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    _touch(tmp_path / "owner-a" / "legacy-plain" / "adapter_model.safetensors")
    _touch(tmp_path / "owner-b" / "other-plain" / "adapter_model.safetensors")

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    checkpoint_ids = {entry["checkpoint_id"] for entry in resp.json()["checkpoints"]}
    assert checkpoint_ids == {"owner-a_legacy-plain"}


def test_issue_441_internal_checkpoint_list_uses_catalog_when_available(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    async def _list_catalog_checkpoints(*, owner_id: str | None, is_admin: bool):
        assert owner_id == "owner-a"
        assert is_admin is False
        return [
            {
                "ckpt_id": "5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11",
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
    monkeypatch.setattr(
        internal_routes,
        "_scan_checkpoints",
        lambda *_args, **_kwargs: [
            internal_routes.CheckpointInfo(
                checkpoint_id="legacy-fs-id",
                model_name="Legacy/Model",
                created_at="2026-01-01T00:00:00Z",
                type="training",
                size_bytes=123,
            )
        ],
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    ids = {item["checkpoint_id"] for item in payload["checkpoints"]}
    assert ids == {"5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11", "legacy-fs-id"}


def test_issue_441_internal_checkpoint_list_catalog_hides_same_fs_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    async def _list_catalog_checkpoints(*, owner_id: str | None, is_admin: bool):
        assert owner_id == "owner-a"
        assert is_admin is False
        return [
            {
                "ckpt_id": "5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11",
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
    monkeypatch.setattr(
        internal_routes,
        "_scan_checkpoints",
        lambda *_args, **_kwargs: [
            internal_routes.CheckpointInfo(
                checkpoint_id="model-catalog_ckpt-catalog",
                model_name="Qwen/Qwen3-0.6B",
                created_at="2026-01-03T00:00:00Z",
                type="sampler",
                size_bytes=2048,
            )
        ],
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    checkpoint_ids = [item["checkpoint_id"] for item in resp.json()["checkpoints"]]
    assert checkpoint_ids == ["5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11"]



def test_issue_441_internal_checkpoint_list_accepts_uuid_catalog_ids(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

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
    monkeypatch.setattr(
        internal_routes,
        "_scan_checkpoints",
        lambda *_args, **_kwargs: [
            internal_routes.CheckpointInfo(
                checkpoint_id="model-catalog_ckpt-catalog",
                model_name="Qwen/Qwen3-0.6B",
                created_at="2026-01-03T00:00:00Z",
                type="sampler",
                size_bytes=2048,
            )
        ],
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert [item["checkpoint_id"] for item in resp.json()["checkpoints"]] == [str(ckpt_id)]



def test_issue_441_internal_checkpoint_archive_uses_catalog_entry(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

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
    monkeypatch.setattr(
        internal_routes,
        "_resolve_checkpoint_entry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("filesystem fallback not expected")),
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="ckpt-catalog.tar.gz"'

    archive = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    names = archive.getnames()
    assert "sampler/adapter_model.safetensors" in names


def test_issue_441_internal_checkpoint_archive_rejects_catalog_path_traversal(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    async def _get_catalog_checkpoint(checkpoint_id: str, *, owner_id: str | None, is_admin: bool):
        assert checkpoint_id == "catalog-id"
        assert owner_id == "owner-a"
        assert is_admin is False
        return {
            "ckpt_id": checkpoint_id,
            "owner_id": "owner-a",
            "model_id": "model-catalog",
            "raw_checkpoint_id": "../escape",
            "checkpoint_type": "sampler",
            "model_name": "Qwen/Qwen3-0.6B",
            "checkpoint_created_at": "2026-01-03T00:00:00Z",
            "published_at": "2026-01-03T00:00:01Z",
            "size_bytes": 2048,
            "storage_root": str(tmp_path / "tos"),
        }

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "get_catalog_checkpoint", _get_catalog_checkpoint)
    monkeypatch.setattr(internal_routes, "_resolve_checkpoint_entry", lambda *_args, **_kwargs: None)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/catalog-id/archive")
    assert resp.status_code == 404, resp.text


def test_issue_441_internal_checkpoint_archive_catalog_missing_path_falls_back_fs(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    fs_ckpt_dir = tmp_path / "owner-a" / "model-fs" / "ckpt-fs"
    _touch(fs_ckpt_dir / "adapter_model.safetensors", b"weights-fs")

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
    monkeypatch.setattr(
        internal_routes,
        "_resolve_checkpoint_entry",
        lambda _checkpoint_id, **_kwargs: (
            str(fs_ckpt_dir),
            {
                "checkpoint_id": "ckpt-fs",
                "owner_id": "owner-a",
                "model_id": "model-fs",
                "checkpoint_type": "sampler",
                "type": "sampler",
            },
        ),
    )

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints/catalog-id/archive")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="ckpt-fs.tar.gz"'


def test_issue_441_internal_checkpoint_list_catalog_passes_shadow_ids_to_fs_scan(monkeypatch, tmp_path: Path) -> None:
    _configure_checkpoint_roots(tmp_path)

    captured: dict[str, object] = {}

    async def _scan_checkpoints_from_catalog(user_id: str | None, *, is_admin: bool = False):
        assert user_id == "owner-a"
        assert is_admin is False
        return (
            [
                internal_routes.CheckpointInfo(
                    checkpoint_id="5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11",
                    model_name="Qwen/Qwen3-0.6B",
                    created_at="2026-01-03T00:00:00Z",
                    type="sampler",
                    size_bytes=2048,
                )
            ],
            {"model-catalog_ckpt-catalog"},
        )

    def _scan_checkpoints(user_id: str | None, *, is_admin: bool = False, exclude_public_ids=None):
        captured["user_id"] = user_id
        captured["is_admin"] = is_admin
        captured["exclude_public_ids"] = set(exclude_public_ids or set())
        return []

    monkeypatch.setattr(internal_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(internal_routes, "_scan_checkpoints_from_catalog", _scan_checkpoints_from_catalog)
    monkeypatch.setattr(internal_routes, "_scan_checkpoints", _scan_checkpoints)

    client = _make_app({"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/internal/v1/checkpoints")
    assert resp.status_code == 200, resp.text
    assert captured == {
        "user_id": "owner-a",
        "is_admin": False,
        "exclude_public_ids": {"model-catalog_ckpt-catalog"},
    }
    assert [item["checkpoint_id"] for item in resp.json()["checkpoints"]] == [
        "5d6fbbf8-6c5b-4e91-8e9f-f7e51f0d7d11"
    ]
