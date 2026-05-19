from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def test_issue_261_response_uses_final_trace_id(monkeypatch):
    from mint_server import config as config_module
    from mint_server.app import api_key_auth_middleware

    monkeypatch.setattr(config_module.config, "api_key", None, raising=False)
    monkeypatch.setattr(config_module.config, "internal_api_token", "", raising=False)

    app = FastAPI()
    app.middleware("http")(api_key_auth_middleware)
    final_trace_id = "b" * 32

    @app.get("/trace")
    async def _trace(request: Request):
        request.state.trace_id = final_trace_id
        return JSONResponse({"ok": True})

    client = TestClient(app)
    response = client.get("/trace", headers={"X-Trace-Id": "a" * 32})
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == final_trace_id
