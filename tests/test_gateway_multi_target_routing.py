import json


def _reset_gateway_module_state():
    import mint_server.gateway.gateway as gw

    gw._gateway_config = None
    gw._remote_sampling_sessions.clear()
    gw._remote_training_models.clear()


def test_gateway_config_synonyms(monkeypatch):
    _reset_gateway_module_state()
    cfg = {
        "model_to_deployment_target": {"Qwen/Qwen3-235B-A22B-Instruct-2507": "mint-prod-aliyun"},
        "deployment_targets": {
            "mint-prod-aliyun": {"base_url": "http://example.com:18000/", "auth_mode": "none"}
        },
    }
    monkeypatch.setenv("MINT_GATEWAY_CONFIG_JSON", json.dumps(cfg))

    import mint_server.gateway.gateway as gw

    up = gw.upstream_for_model("Qwen/Qwen3-235B-A22B-Instruct-2507")
    assert up is not None
    assert up.alias == "mint-prod-aliyun"
    assert up.base_url == "http://example.com:18000"
    assert up.auth_mode == "none"


def test_gateway_request_id_roundtrip():
    import mint_server.gateway.gateway as gw

    rid = gw.encode_request_id(upstream_alias="mint-prod-aliyun", upstream_request_id="abc123")
    assert gw.decode_request_id(rid) == ("mint-prod-aliyun", "abc123")
    assert gw.decode_request_id("not-a-gw-id") is None


def test_gateway_remote_sampling_session_in_memory(monkeypatch):
    _reset_gateway_module_state()
    monkeypatch.delenv("MINT_GATEWAY_CONFIG_JSON", raising=False)

    import mint_server.gateway.gateway as gw

    gw.register_remote_sampling_session(
        sampling_session_id="sess1",
        upstream_alias="mint-prod-aliyun",
        base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    assert gw.remote_sampling_session("sess1") == ("mint-prod-aliyun", "Qwen/Qwen3-235B-A22B-Instruct-2507")

