import json
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _app_with_user(router, user_data: dict) -> TestClient:
    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request: Request, call_next):
        request.state.user_data = user_data
        return await call_next(request)

    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_issue_557_resolve_checkpoint_path_admin_requires_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    with pytest.raises(ValueError, match="owner_id is required"):
        checkpoints.resolve_checkpoint_path(
            "tinker://run-557/weights/ckpt-a",
            user_id=None,
            is_admin=True,
        )


def test_issue_557_resolve_checkpoint_path_admin_uses_owner_scope_without_glob(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(checkpoints.glob, "glob", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("glob scan not expected")))

    resolved = checkpoints.resolve_checkpoint_path(
        "tinker://run-557/weights/ckpt-a",
        user_id="owner-a",
        is_admin=True,
    )
    assert resolved == str(ckpt_dir)


def test_issue_557_weights_list_requires_catalog(monkeypatch) -> None:
    from tinker_server.routes import weights as wt

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: False)
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/api/v1/training_runs/run-557/checkpoints")
    assert resp.status_code == 503, resp.text
    assert resp.json() == {"detail": "Checkpoint catalog unavailable"}


def test_issue_557_weights_archive_admin_requires_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get("/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?direct=1")
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "owner_id is required for admin checkpoint access"}


def test_issue_557_weights_delete_admin_requires_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.delete("/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a")
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "owner_id is required for admin checkpoint access"}


def test_issue_557_weights_archive_admin_owner_scope_blocks_wrong_owner(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    wrong_owner_dir = tmp_path / "run-557" / "ckpt-a" / "training"
    _touch(wrong_owner_dir / "adapter_model.safetensors")
    _touch(wrong_owner_dir / "optimizer.pt")
    (wrong_owner_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-b",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get(
        "/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?direct=1&owner_id=owner-a"
    )
    assert resp.status_code == 403, resp.text
    assert resp.json() == {"detail": "Access denied"}


def test_issue_557_weights_archive_sdk_redirect_preserves_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get(
        "/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?owner_id=owner-a",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1"},
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text
    assert "direct=1" in resp.headers["location"]
    assert "owner_id=owner-a" in resp.headers["location"]


def test_issue_557_training_helper_requires_owner_id_for_admin() -> None:
    from fastapi import HTTPException
    from tinker_server.routes import training as tr

    with pytest.raises(HTTPException, match="owner_id is required"):
        tr._resolve_state_path(
            "tinker://run-557/weights/ckpt-a",
            user_id="admin",
            is_admin=True,
            owner_id=None,
        )


def test_issue_557_weights_helper_requires_owner_id_for_admin() -> None:
    from fastapi import HTTPException
    from tinker_server.routes import weights as wt

    with pytest.raises(HTTPException, match="owner_id is required"):
        wt._resolve_mint_path(
            "tinker://run-557/weights/ckpt-a",
            user_id="admin",
            is_admin=True,
            owner_id=None,
        )


def test_issue_557_mint_helper_requires_owner_id_for_admin() -> None:
    from fastapi import HTTPException
    from tinker_server.routes import mint as mint_routes

    with pytest.raises(HTTPException, match="owner_id is required"):
        mint_routes._resolve_checkpoint_for_user(
            "tinker://run-557/weights/ckpt-a",
            user_id="admin",
            is_admin=True,
            owner_id=None,
        )


def test_issue_557_mint_action_session_admin_missing_owner_id_returns_400(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import mint as mint_routes
    import tinker_server.supported_models_gate as model_gate

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "model_name": "Qwen/Qwen3-0.6B",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubActionSessionManager:
        async def create_session(self, **kwargs):
            raise AssertionError(f"unexpected create_session: {kwargs}")

    async def _enforce_base_model_allowed(*, base_model: str, http_request):
        _ = http_request
        return base_model

    monkeypatch.setattr(mint_routes, "action_session_manager", StubActionSessionManager())
    monkeypatch.setattr(mint_routes, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_gate, "enforce_base_model_allowed", _enforce_base_model_allowed)
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    client = _app_with_user(mint_routes.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.post(
        "/api/v1/action_sessions",
        json={
            "session_id": "sess-557",
            "model_path": "tinker://run-557/weights/ckpt-a",
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "owner_id is required for admin checkpoint references"}


def test_issue_557_mint_action_session_admin_accepts_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import mint as mint_routes
    import tinker_server.supported_models_gate as model_gate

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "model_name": "Qwen/Qwen3-0.6B",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubActionSessionManager:
        async def create_session(self, **kwargs):
            assert kwargs["model_path"].endswith("/owner-a/run-557/ckpt-a/training")
            return "action-session-557"

    async def _enforce_base_model_allowed(*, base_model: str, http_request):
        _ = http_request
        return base_model

    monkeypatch.setattr(mint_routes, "action_session_manager", StubActionSessionManager())
    monkeypatch.setattr(mint_routes, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_gate, "enforce_base_model_allowed", _enforce_base_model_allowed)
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    client = _app_with_user(mint_routes.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.post(
        "/api/v1/action_sessions",
        json={
            "session_id": "sess-557",
            "base_model": "Qwen/Qwen3-0.6B",
            "model_path": "tinker://run-557/weights/ckpt-a",
            "owner_id": "owner-a",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"action_session_id": "action-session-557"}


def test_issue_557_weights_archive_sdk_redirect_token_uses_owner_scope(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    import tinker_server.config as config_module
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)
    monkeypatch.setattr(config_module.config, "api_key", "secret", raising=False)

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    first = client.get(
        "/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?owner_id=owner-a",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1"},
        follow_redirects=False,
    )
    assert first.status_code == 302, first.text

    second = client.get(first.headers["location"])
    assert second.status_code == 200, second.text
    assert second.headers["content-disposition"] == 'attachment; filename="run-557_weights_ckpt-a.tar.gz"'


class _ServiceRequest:
    def __init__(self, user_data: dict | None) -> None:
        self.state = type("State", (), {"user_data": user_data})()
        self.headers = {}


def test_issue_557_service_resolve_model_path_admin_uses_owner_scope(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import service as service_routes

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    resolved = service_routes._resolve_model_path(
        "tinker://run-557/weights/ckpt-a",
        user_id="admin",
        owner_id="owner-a",
        http_request=_ServiceRequest({"user_id": "admin", "user_role": "admin", "is_admin": True}),
    )
    assert resolved.endswith("/owner-a/run-557/ckpt-a/training")


def test_issue_557_mint_helper_maps_permission_error_to_403(monkeypatch) -> None:
    from fastapi import HTTPException
    from tinker_server.routes import mint as mint_routes

    monkeypatch.setattr(mint_routes, "resolve_checkpoint_path", lambda *_args, **_kwargs: "/tmp/blocked")
    monkeypatch.setattr(
        mint_routes,
        "ensure_checkpoint_path_allowed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("blocked")),
    )

    with pytest.raises(HTTPException) as exc:
        mint_routes._resolve_checkpoint_for_user(
            "tinker://run-557/weights/ckpt-a",
            user_id="owner-a",
            is_admin=False,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "blocked"


def test_issue_557_weights_helper_maps_permission_error_to_403(monkeypatch) -> None:
    from fastapi import HTTPException
    from tinker_server.routes import weights as wt

    monkeypatch.setattr(wt, "resolve_checkpoint_path", lambda *_args, **_kwargs: "/tmp/blocked")
    monkeypatch.setattr(
        wt,
        "ensure_checkpoint_path_allowed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("blocked")),
    )

    with pytest.raises(HTTPException) as exc:
        wt._resolve_mint_path(
            "tinker://run-557/weights/ckpt-a",
            user_id="owner-a",
            is_admin=False,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "blocked"


def test_issue_557_weights_archive_rejects_catalog_row_when_metadata_owner_mismatches(tmp_path, monkeypatch) -> None:
    from tinker_server.routes import weights as wt

    wrong_owner_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(wrong_owner_dir / "adapter_model.safetensors")
    _touch(wrong_owner_dir / "optimizer.pt")
    (wrong_owner_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-b",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _get_catalog_checkpoint_by_key(**kwargs):
        return {
            "ckpt_id": "55700000-0000-0000-0000-000000000001",
            "owner_id": "owner-a",
            "model_id": "run-557",
            "raw_checkpoint_id": "ckpt-a",
            "checkpoint_type": "training",
            "storage_root": str(tmp_path),
        }

    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(wt, "get_catalog_checkpoint_by_key", _get_catalog_checkpoint_by_key)

    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            wt._resolve_weight_checkpoint(
                model_id="run-557",
                checkpoint_id="weights/ckpt-a",
                request_user_id="admin",
                owner_id="owner-a",
                is_admin=True,
            )
        )
    assert exc.value.status_code == 404


def test_issue_557_parse_checkpoint_time_normalizes_naive_timestamps(tmp_path) -> None:
    from tinker_server.routes import weights as wt

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    parsed = wt._parse_checkpoint_time("2026-03-14T00:00:00", fallback_path=str(fallback))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_issue_557_weights_archive_admin_rejects_invalid_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "admin", "user_role": "admin", "is_admin": True})
    resp = client.get(
        "/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?direct=1&owner_id=.."
    )
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "Invalid owner_id"}


def test_issue_557_weights_catalog_checkpoint_path_rejects_escape_segments() -> None:
    from tinker_server.routes import weights as wt

    row = {
        "storage_root": "/tmp/checkpoints",
        "owner_id": "..",
        "model_id": "run-557",
        "raw_checkpoint_id": "ckpt-a",
        "checkpoint_type": "training",
    }
    assert wt._catalog_checkpoint_path(row) is None


def test_issue_557_resolve_checkpoint_path_admin_rejects_invalid_owner_id(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))

    with pytest.raises(ValueError, match="Invalid owner_id"):
        checkpoints.resolve_checkpoint_path(
            "tinker://run-557/weights/ckpt-a",
            user_id="../owner-a",
            is_admin=True,
        )


def test_issue_557_weights_delete_succeeds_when_catalog_tombstone_fails(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "training"
    _touch(ckpt_dir / "adapter_model.safetensors")
    _touch(ckpt_dir / "optimizer.pt")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "training",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    async def _mark_catalog_checkpoint_deleted(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: True)
    async def _get_catalog_checkpoint_by_key(**_kwargs):
        return {
            "ckpt_id": "55700000-0000-0000-0000-000000000002",
            "owner_id": "owner-a",
            "model_id": "run-557",
            "raw_checkpoint_id": "ckpt-a",
            "checkpoint_type": "training",
            "storage_root": str(tmp_path),
        }

    monkeypatch.setattr(wt, "get_catalog_checkpoint_by_key", _get_catalog_checkpoint_by_key)
    monkeypatch.setattr(wt, "mark_catalog_checkpoint_deleted", _mark_catalog_checkpoint_deleted)

    client = _app_with_user(wt.router, {"user_id": "owner-a", "user_role": "user"})
    resp = client.delete("/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "deleted", "checkpoint_id": "weights/ckpt-a"}
    assert not ckpt_dir.exists()


def test_issue_557_weights_archive_type_mismatch_returns_404(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    sampler_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "sampler"
    _touch(sampler_dir / "adapter_model.safetensors")
    (sampler_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "sampler",
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: False)
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/api/v1/training_runs/run-557/checkpoints/weights/ckpt-a/archive?direct=1")
    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": "Checkpoint 'weights/ckpt-a' not found"}


def test_issue_557_weights_archive_untyped_checkpoint_id_uses_only_typed_view(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    sampler_dir = tmp_path / "owner-a" / "run-557" / "ckpt-a" / "sampler"
    _touch(sampler_dir / "adapter_model.safetensors")
    (sampler_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-a",
                "owner_id": "owner-a",
                "model_id": "run-557",
                "checkpoint_type": "sampler",
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: False)
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_archive", _no_remote)

    client = _app_with_user(wt.router, {"user_id": "owner-a", "user_role": "user"})
    resp = client.get("/api/v1/training_runs/run-557/checkpoints/ckpt-a/archive?direct=1")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == 'attachment; filename="run-557_ckpt-a.tar.gz"'
