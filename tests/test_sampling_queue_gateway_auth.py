from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

from mint_server.backend.scheduling import model_work_dispatch
from mint_server.models.types import (
    ModelInput,
    SampleRequest,
    SamplingParams,
    SaveStateRequest,
)


def test_sampling_work_executor_forwards_gateway_auth(monkeypatch):
    captured: dict[str, object] = {}

    async def _capture_do_sample(
        request_id, req, user_id, gateway_auth=None, suppress_billing=False
    ) -> None:
        captured["request_id"] = request_id
        captured["sampling_session_id"] = req.sampling_session_id
        captured["user_id"] = user_id
        captured["gateway_auth"] = gateway_auth
        captured["suppress_billing"] = suppress_billing

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
            },
            "suppress_billing": True,
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
        "suppress_billing": True,
    }


def test_weights_save_weights_dispatches_sampler_helper(monkeypatch):
    captured: dict[str, object] = {}

    async def _capture_do_save_weights(
        request_id,
        req,
        user_id=None,
        webhook_url=None,
        prefer_tinker=False,
    ) -> None:
        captured["helper"] = "_do_save_weights"
        captured["request_id"] = request_id
        captured["model_id"] = req.model_id
        captured["user_id"] = user_id
        captured["webhook_url"] = webhook_url
        captured["prefer_tinker"] = prefer_tinker

    async def _unexpected_do_save_state(*_args, **_kwargs) -> None:
        raise AssertionError("weights.save_weights must not dispatch to _do_save_state")

    ray_module = types.ModuleType("ray")
    ray_module.is_initialized = lambda: True

    request = SaveStateRequest(model_id="run-save-weights", path="sampler-ckpt")
    item = SimpleNamespace(
        request_id="req-save-weights",
        op="weights.save_weights",
        request_json=request.model_dump_json().encode("utf-8"),
        user_id="user-test",
        apikey_id=None,
        throttle_principal=None,
        webhook_url="https://example.invalid/hook",
        extra={"prefer_tinker": True},
    )

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setattr(
        "mint_server.routes.weights._do_save_weights", _capture_do_save_weights
    )
    monkeypatch.setattr(
        "mint_server.routes.weights._do_save_state", _unexpected_do_save_state
    )

    asyncio.run(model_work_dispatch.execute_model_work_item(item))

    assert captured == {
        "helper": "_do_save_weights",
        "request_id": "req-save-weights",
        "model_id": "run-save-weights",
        "user_id": "user-test",
        "webhook_url": "https://example.invalid/hook",
        "prefer_tinker": True,
    }
