from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tinker_server.app import external_compatibility_middleware
from tinker_server.compatibility import rewrite_legacy_tinker_uris


def test_rewrite_legacy_tinker_uris_recurses() -> None:
    payload = {
        "model_path": "tinker://run/sampler_weights/ckpt",
        "items": [{"state_path": "tinker://run/weights/ckpt"}],
        "base_model": "Qwen/Qwen3-0.6B",
    }

    rewritten, changed = rewrite_legacy_tinker_uris(payload)

    assert changed is True
    assert rewritten == {
        "model_path": "mint://run/sampler_weights/ckpt",
        "items": [{"state_path": "mint://run/weights/ckpt"}],
        "base_model": "Qwen/Qwen3-0.6B",
    }


def test_external_compatibility_middleware_rewrites_json_body() -> None:
    app = FastAPI()
    app.middleware("http")(external_compatibility_middleware)

    @app.post("/echo")
    async def _echo(request: Request):
        return JSONResponse(await request.json())

    client = TestClient(app)
    response = client.post(
        "/echo",
        json={
            "model_path": "tinker://run/sampler_weights/ckpt",
            "nested": {"state_path": "tinker://run/weights/ckpt"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_path": "mint://run/sampler_weights/ckpt",
        "nested": {"state_path": "mint://run/weights/ckpt"},
    }
