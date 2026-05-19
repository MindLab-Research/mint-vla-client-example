from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

from mint_server.backend import model_work_dispatch
from mint_server.models.types import ModelInput, SampleRequest, SamplingParams


def test_sampling_work_executor_forwards_gateway_auth(monkeypatch):
    captured: dict[str, object] = {}

    async def _capture_do_sample(request_id, req, user_id, gateway_auth=None) -> None:
        captured["request_id"] = request_id
        captured["sampling_session_id"] = req.sampling_session_id
        captured["user_id"] = user_id
        captured["gateway_auth"] = gateway_auth

    ray_module = types.ModuleType("ray")
    ray_module.is_initialized = lambda: True

    request = SampleRequest(
        sampling_session_id="sess-test",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )
    item = SimpleNamespace(
        request_id="req-test",
        op="sampling.asample",
        request_json=request.model_dump_json().encode("utf-8"),
        user_id="user-test",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        throttle_principal="apikey:bbbbbbbbbbbbbbbbbbbbbbbb",
        webhook_url=None,
        extra={
            "gateway_auth": {
                "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "user_role": "user",
                "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
                "request_id": "req-billing-test",
            }
        },
    )

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setattr("mint_server.routes.sampling._do_sample", _capture_do_sample)

    asyncio.run(model_work_dispatch.execute_model_work_item(item))

    assert captured == {
        "request_id": "req-test",
        "sampling_session_id": "sess-test",
        "user_id": "user-test",
        "gateway_auth": {
            "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "user_role": "user",
            "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "request_id": "req-billing-test",
        },
    }
