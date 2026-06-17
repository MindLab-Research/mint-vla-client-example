import json
from types import SimpleNamespace

import pytest


def _gateway_cfg() -> dict:
    return {
        "model_to_upstream": {"zai-org/GLM-5": "glm5"},
        "upstreams": {
            "glm5": {
                "base_url": "http://example.com:18000",
                "auth_mode": "static_api_key",
                "api_key": "secret",
            }
        },
    }


@pytest.mark.anyio
async def test_enforce_base_model_allowed_accepts_supported_gateway_model(monkeypatch):
    import mint_server.gateway.gateway as gw
    import mint_server.auth.supported_models_gate as gate

    gw._gateway_config = None
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", "Qwen/Qwen3-0.6B,zai-org/GLM-5")
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(_gateway_cfg()))
    monkeypatch.setattr(gate, "ALLOW_UNSUPPORTED_MODELS", False)

    async def _caps(*, upstream, incoming_headers, cache_ttl_s=5.0):
        assert upstream.alias == "glm5"
        return {"zai-org/GLM-5": 32768}

    monkeypatch.setattr(gw, "get_upstream_capabilities", _caps)

    got = await gate.enforce_base_model_allowed(
        base_model="zai-org/GLM-5",
        http_request=SimpleNamespace(headers={}),
    )

    assert got == "zai-org/GLM-5"


@pytest.mark.anyio
async def test_enforce_base_model_allowed_rejects_unadvertised_gateway_model(monkeypatch):
    import mint_server.gateway.gateway as gw
    import mint_server.auth.supported_models_gate as gate

    gw._gateway_config = None
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", "Qwen/Qwen3-0.6B")
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(_gateway_cfg()))
    monkeypatch.setattr(gate, "ALLOW_UNSUPPORTED_MODELS", False)

    with pytest.raises(gate.HTTPException, match="Not present in MINT_SUPPORTED_MODELS"):
        await gate.enforce_base_model_allowed(
            base_model="zai-org/GLM-5",
            http_request=SimpleNamespace(headers={}),
        )
