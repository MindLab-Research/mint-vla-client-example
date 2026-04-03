import json
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _DummyClient:
    async def aclose(self) -> None:
        return None


def _patch_async_remote_training_model_info(monkeypatch, gw, payload: dict) -> None:
    async def _fake_async_remote_training_model_info(_model_id: str):
        return dict(payload)

    monkeypatch.setattr(gw, "async_remote_training_model_info", _fake_async_remote_training_model_info)


def _client() -> TestClient:
    from tinker_server.routes import weights as weights_routes

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    return TestClient(app)


def _client_with_user(user_data: dict | None) -> TestClient:
    from tinker_server.routes import weights as weights_routes

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.user_data = user_data
        return await call_next(request)

    app.include_router(weights_routes.router, prefix="/api/v1")
    return TestClient(app)


def test_issue_276_gateway_list_checkpoints_proxies_remote(monkeypatch) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": None},
    )
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: Upstream("up", "http://upstream.example", "none"))
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _fake_forward_json(*, upstream, method, path, incoming_headers, json_body, timeout_s):
        assert upstream.alias == "up"
        assert method == "GET"
        assert path == "/api/v1/training_runs/run-276/checkpoints"
        assert json_body is None
        return httpx.Response(
            200,
            json={
                "model_id": "run-276",
                "checkpoints": [
                    {
                        "checkpoint_id": "weights/ckpt_123",
                        "checkpoint_type": "training",
                        "tinker_path": "tinker://run-276/weights/ckpt_123",
                        "path": "mint://run-276/weights/ckpt_123",
                        "step": 0,
                        "created_at": "2026-03-08T00:00:00Z",
                        "time": "2026-03-08T00:00:00Z",
                    }
                ],
            },
        )

    monkeypatch.setattr(gw, "forward_json", _fake_forward_json)

    resp = _client().get("/api/v1/training_runs/run-276/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["model_id"] == "run-276"
    assert payload["checkpoints"][0]["checkpoint_id"] == "weights/ckpt_123"


def test_issue_276_gateway_list_checkpoints_remote_error_passthrough(monkeypatch) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": None},
    )
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: Upstream("up", "http://upstream.example", "none"))
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _fake_forward_json(*, upstream, method, path, incoming_headers, json_body, timeout_s):
        return httpx.Response(404, json={"detail": "Checkpoint not found upstream"})

    monkeypatch.setattr(gw, "forward_json", _fake_forward_json)

    resp = _client().get("/api/v1/training_runs/run-276/checkpoints")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Checkpoint not found upstream"}


def test_issue_276_gateway_archive_redirect_proxies_remote(monkeypatch) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": None},
    )
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: Upstream("up", "http://upstream.example", "none"))
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _fake_forward_request(*, upstream, method, path, incoming_headers, params, timeout_s, stream):
        assert upstream.alias == "up"
        assert method == "GET"
        assert path == "/api/v1/training_runs/run-276/checkpoints/weights/ckpt_123/archive"
        assert params == {}
        assert stream is True
        return _DummyClient(), httpx.Response(
            302,
            headers={"Location": "https://upstream.example/archive.tar.gz", "Expires": "Sun, 08 Mar 2026 00:15:00 GMT"},
        )

    monkeypatch.setattr(gw, "forward_request", _fake_forward_request)

    resp = _client().get(
        "/api/v1/training_runs/run-276/checkpoints/weights/ckpt_123/archive",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("http://testserver/api/v1/training_runs/run-276/checkpoints/weights/ckpt_123/archive?direct=1")
    assert "upstream.example" not in resp.headers["location"]
    assert resp.headers["expires"] == "Sun, 08 Mar 2026 00:15:00 GMT"


def test_issue_276_gateway_archive_direct_download_proxies_remote(monkeypatch) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": None},
    )
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: Upstream("up", "http://upstream.example", "none"))
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _fake_forward_request(*, upstream, method, path, incoming_headers, params, timeout_s, stream):
        assert params == {"direct": "1"}
        return _DummyClient(), httpx.Response(
            200,
            content=b"\x1f\x8bremote-archive",
            headers={
                "content-type": "application/gzip",
                "content-disposition": 'attachment; filename="run-276.tar.gz"',
                "content-length": "16",
            },
        )

    monkeypatch.setattr(gw, "forward_request", _fake_forward_request)

    resp = _client().get("/api/v1/training_runs/run-276/checkpoints/weights/ckpt_123/archive?direct=1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/gzip"
    assert resp.headers["content-disposition"] == 'attachment; filename="run-276.tar.gz"'
    assert resp.content.startswith(b"\x1f\x8b")


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (403, "Access denied upstream"),
        (404, "Checkpoint not found upstream"),
    ],
)
def test_issue_276_gateway_archive_remote_error_passthrough(monkeypatch, status_code: int, detail: str) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": None},
    )
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: Upstream("up", "http://upstream.example", "none"))
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _fake_forward_request(*, upstream, method, path, incoming_headers, params, timeout_s, stream):
        return _DummyClient(), httpx.Response(status_code, text=json.dumps({"detail": detail}))

    monkeypatch.setattr(gw, "forward_request", _fake_forward_request)

    resp = _client().get("/api/v1/training_runs/run-276/checkpoints/weights/ckpt_123/archive?direct=1")
    assert resp.status_code == status_code
    assert resp.json() == {"detail": detail}


def test_issue_276_gateway_remote_checkpoint_owner_mismatch_denied_before_forward(monkeypatch) -> None:
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.routes import weights as wt

    _patch_async_remote_training_model_info(
        monkeypatch,
        gw,
        {"upstream_alias": "up", "base_model": "Qwen/Qwen3-0.6B", "owner_id": "owner-a"},
    )
    monkeypatch.setattr(
        gw,
        "upstream_for_alias",
        lambda _alias: Upstream("up", "http://upstream.example", "static_api_key", api_key="secret"),
    )
    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)

    async def _unexpected_forward_json(**_kwargs):
        raise AssertionError("checkpoint list should fail closed before upstream forwarding")

    monkeypatch.setattr(gw, "forward_json", _unexpected_forward_json)

    resp = _client_with_user({"user_id": "owner-b"}).get("/api/v1/training_runs/run-276/checkpoints")
    assert resp.status_code == 403
    assert resp.json() == {"detail": "Access denied"}
