from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_ID = "0123456789abcdef01234567"
_ACCOUNT_ID = "111111111111111111111111"
_APIKEY_ID = "222222222222222222222222"


def _install_billing_user(monkeypatch, mint_routes) -> None:
    from mint_server.gateway.gateway_auth import GatewayAuthContext

    billing_auth = GatewayAuthContext(
        user_id=_USER_ID,
        user_role="user",
        account_id=_ACCOUNT_ID,
        apikey_id=_APIKEY_ID,
        request_id="gateway-request-1",
        cap_write=True,
    )
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: _USER_ID)
    monkeypatch.setattr(
        mint_routes,
        "_get_user_data",
        lambda _request: {
            "user_id": _USER_ID,
            "account_id": _ACCOUNT_ID,
            "apikey_id": _APIKEY_ID,
            "request_id": "gateway-request-1",
        },
    )
    monkeypatch.setattr(
        mint_routes,
        "build_billing_auth_context",
        lambda _request, *, fallback_request_id=None: billing_auth,
    )


class _StubTaskFutureService:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []
        self.resolved: list[tuple[str, dict]] = []
        self.billing_observations: dict[str, list[dict]] = {}
        self.failed: list[tuple[str, str]] = []

    async def async_create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    async def async_create_model_work_with_id(self, request_id: str, **kwargs) -> None:
        self.created.append(request_id)
        self.queued.append((request_id, kwargs.get("meta")))

    async def async_update_meta(self, _request_id: str, _meta: dict) -> None:
        return None

    async def async_mark_queued(
        self, request_id: str, meta: dict | None = None
    ) -> None:
        self.queued.append((request_id, meta))

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    async def async_resolve(
        self, request_id: str, payload: dict, *, billing_observations=None
    ) -> None:
        self.resolved.append((request_id, payload))
        self.billing_observations[request_id] = list(billing_observations or [])

    async def async_fail(self, request_id: str, message: str) -> None:
        self.failed.append((request_id, message))


class _StubModelWorkScheduler:
    def __init__(
        self, *, fail: bool = False, task_futures: _StubTaskFutureService | None = None
    ) -> None:
        self.calls: list[dict] = []
        self.cancelled: list[dict] = []
        self.fail = bool(fail)
        self.task_futures = task_futures

    async def append(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        if self.task_futures is not None:
            await self.task_futures.async_create_model_work_with_id(
                kwargs["request_id"],
                op=kwargs.get("op"),
                domain_key=kwargs.get("domain_key"),
                request_json=kwargs.get("request_json"),
                meta=kwargs.get("extra"),
            )
        if self.fail:
            raise RuntimeError("scheduler unavailable")
        return {"ok": True, "scheduler_instance_id": "scheduler-mint"}

    async def cancel_request(self, **kwargs) -> dict:
        self.cancelled.append(dict(kwargs))
        return {"ok": True}


def test_mint_action_route_does_not_cleanup_gateway_owned_future_when_enqueue_fails(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(fail=True, task_futures=task_futures)

    monkeypatch.setattr(mint_routes, "task_futures", task_futures, raising=False)
    monkeypatch.setattr(mint_routes, "action_session_manager", object(), raising=False)

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions/action-session-1/act",
        json={
            "observation": {
                "state": {
                    "data": [0.0] * 8,
                    "shape": [8],
                    "dtype": "float32",
                },
                "model_input": {
                    "chunks": [
                        {
                            "type": "image",
                            "data": "aW1n",
                            "format": "png",
                            "expected_tokens": 256,
                        }
                    ]
                },
            },
        },
    )

    assert resp.status_code == 503, resp.text
    assert len(task_futures.created) == 1
    assert task_futures.cleaned == []
    assert len(scheduler.calls) == 1


def test_mint_action_route_enqueues_billing_observation(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)

    class _StubActionSessionManager:
        def get_billing_metadata(self, action_session_id: str) -> dict:
            assert action_session_id == "action-session-1"
            return {
                "base_model": "openpi/pi0-fast-libero-low-mem-finetune",
                "policy_family": "ar_action_tokens",
                "action_dim": 7,
                "action_horizon": 10,
                "action_token_budget": 64,
                "action_output_tokens": 64,
            }

    monkeypatch.setattr(mint_routes, "task_futures", task_futures, raising=False)
    monkeypatch.setattr(
        mint_routes,
        "action_session_manager",
        _StubActionSessionManager(),
        raising=False,
    )
    _install_billing_user(monkeypatch, mint_routes)

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions/action-session-1/act",
        json={
            "observation": {
                "state": {"data": [0.0] * 8, "shape": [8], "dtype": "float32"},
                "model_input": {
                    "chunks": [
                        {
                            "type": "image",
                            "data": "aW1n",
                            "format": "png",
                            "expected_tokens": 256,
                        },
                        {"type": "encoded_text", "tokens": [1, 2, 3]},
                    ]
                },
            },
            "temperature": 0.05,
        },
    )

    assert resp.status_code == 200, resp.text
    queued = scheduler.calls[0]
    assert "billing_observations" not in queued["extra"]
    billing_input = queued["extra"]["billing_observation_input"]
    assert billing_input["charge_item"] == "inference"
    assert billing_input["quantity"] == 323
    assert billing_input["unit"] == "estimated_tokens"
    assert billing_input["route"] == "mint.action.act"
    assert billing_input["dimension"] == "action"
    assert billing_input["model"] == "openpi/pi0-fast-libero-low-mem-finetune"
    assert billing_input["metadata"]["input_tokens"] == 259
    assert billing_input["metadata"]["action_output_tokens"] == 64
    assert billing_input["metadata"]["action_token_budget"] == 64
    assert queued["extra"]["gateway_auth"]["apikey_id"] == _APIKEY_ID


def test_mint_create_action_session_maps_capacity_runtime_error_to_503(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes
    import mint_server.auth.supported_models_gate as supported_models_gate

    class _StubActionSessionManager:
        async def create_session(self, **kwargs):
            _ = kwargs
            raise RuntimeError(
                "[OpenPIActionRuntime] node pinning model='openpi/pi0-fast-libero-low-mem-finetune' actor='mint_openpi_action_test': "
                "pinned node capacity check failed: required_by_node={'192.168.38.176': 1}"
            )

    async def _allow_model(*, base_model: str, http_request):
        _ = http_request
        return base_model

    monkeypatch.setattr(
        mint_routes,
        "action_session_manager",
        _StubActionSessionManager(),
        raising=False,
    )
    monkeypatch.setattr(
        mint_routes, "can_access_model", lambda base_model, user_data: True
    )
    monkeypatch.setattr(mint_routes, "_get_user_data", lambda request: None)
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, **_: "/resolved/checkpoint",
    )
    monkeypatch.setattr(
        supported_models_gate, "enforce_base_model_allowed", _allow_model
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions",
        json={
            "base_model": "openpi/pi0-fast-libero-low-mem-finetune",
            "session_id": "act-test",
            "action_session_seq_id": 0,
            "model_path": "mint://model/checkpoint",
        },
    )

    assert resp.status_code == 503, resp.text
    assert "pinned node capacity check failed" in resp.text


def test_mint_create_action_session_uses_bypass_cap_for_checkpoint_paths(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes
    import mint_server.auth.supported_models_gate as supported_models_gate

    captured: dict[str, object] = {}

    class _StubActionSessionManager:
        async def create_session(self, **kwargs):
            captured["create_session_kwargs"] = kwargs
            return "action-session-123"

    async def _allow_model(*, base_model: str, http_request):
        _ = http_request
        return base_model

    def _infer(model_path: str, *, user_id: str | None, is_admin: bool) -> str:
        captured["infer"] = {
            "model_path": model_path,
            "user_id": user_id,
            "is_admin": is_admin,
        }
        return "openpi/pi0-fast-libero-low-mem-finetune"

    def _resolve(
        path: str, *, user_id: str | None, is_admin: bool, owner_id: str | None = None
    ) -> str:
        captured["resolve"] = {
            "path": path,
            "user_id": user_id,
            "is_admin": is_admin,
            "owner_id": owner_id,
        }
        return "/resolved/user-a/checkpoint"

    monkeypatch.setattr(
        mint_routes,
        "action_session_manager",
        _StubActionSessionManager(),
        raising=False,
    )
    monkeypatch.setattr(
        mint_routes, "can_access_model", lambda base_model, user_data: True
    )
    monkeypatch.setattr(mint_routes, "_get_user_data", lambda request: None)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda request: "user-a")
    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda request: True)
    monkeypatch.setattr(mint_routes, "_infer_base_model_from_checkpoint", _infer)
    monkeypatch.setattr(mint_routes, "_resolve_checkpoint_for_user", _resolve)
    monkeypatch.setattr(
        supported_models_gate, "enforce_base_model_allowed", _allow_model
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions",
        json={
            "session_id": "act-test",
            "action_session_seq_id": 0,
            "model_path": "mint://model/checkpoint",
            "owner_id": "user-a",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["action_session_id"] == "action-session-123"
    assert captured["infer"] == {
        "model_path": "mint://model/checkpoint",
        "user_id": "user-a",
        "is_admin": True,
    }
    assert captured["resolve"] == {
        "path": "mint://model/checkpoint",
        "user_id": "user-a",
        "is_admin": True,
        "owner_id": "user-a",
    }
    assert (
        captured["create_session_kwargs"]["model_path"] == "/resolved/user-a/checkpoint"
    )
    assert (
        captured["create_session_kwargs"]["base_model"]
        == "openpi/pi0-fast-libero-low-mem-finetune"
    )


def test_mint_vla_train_step_route_enqueues_expected_request(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)

    session = SimpleNamespace(
        model_id="model-123",
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
        backend="openpi_fast",
    )

    class _StubTrainingManager:
        def get_session(self, model_id: str):
            return session if model_id == "model-123" else None

        def mark_inflight(self, model_id: str, delta: int) -> None:
            raise AssertionError(
                f"VLA HTTP route must not use local manager inflight: {model_id} {delta}"
            )

    inflight_calls: list[tuple[str, int]] = []

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", object())
    monkeypatch.setattr(mint_routes, "training_manager", _StubTrainingManager())
    _install_billing_user(monkeypatch, mint_routes)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    from mint_server.routes import training as training_routes

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(
        training_routes,
        "_build_training_scheduler_extra",
        lambda *, session, model_id, training_op, seq_id=None: {
            "scheduler_domain": f"training:{session.base_model}",
            "scheduler_session_key": model_id,
            "training_op": training_op,
            "seq_id": seq_id,
            "backend": session.backend,
        },
    )

    async def _fake_enqueue_training_request_with_trace(**kwargs):
        await kwargs["enqueue_coro"]

    monkeypatch.setattr(
        training_routes,
        "_enqueue_training_request_with_trace",
        _fake_enqueue_training_request_with_trace,
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/vla/train_step",
        json={
            "model_id": "model-123",
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "observation": {
                        "state": {
                            "data": [0.0] * 8,
                            "shape": [8],
                            "dtype": "float32",
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "type": "image",
                                    "data": "aW1n",
                                    "format": "png",
                                    "expected_tokens": 256,
                                },
                                {"type": "encoded_text", "tokens": [1, 2, 3]},
                            ]
                        },
                    },
                    "supervision": {
                        "target_tokens": {
                            "data": [11, 12],
                            "shape": [2],
                            "dtype": "int64",
                        },
                        "weights": {
                            "data": [1.0, 1.0],
                            "shape": [2],
                            "dtype": "float32",
                        },
                        "token_ar_mask": {
                            "data": [1, 1],
                            "shape": [2],
                            "dtype": "int64",
                        },
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert inflight_calls == [("model-123", +1)]
    assert task_futures.created == [request_id]
    assert len(scheduler.calls) == 1
    queued = scheduler.calls[0]
    assert queued["op"] == "mint.vla.train_step"
    assert queued["request_id"] == request_id
    assert queued["domain_key"] == "training:openpi/pi0-fast-libero-low-mem-finetune"
    assert queued["affinity_group"] == "training_session:model-123"
    queued_meta = queued["extra"]
    assert queued_meta["op"] == "mint.vla.train_step"
    assert queued_meta["model_id"] == "model-123"
    assert queued_meta["queue_state"] == "queued"
    assert "billing_observations" not in queued["extra"]
    billing_input = queued["extra"]["billing_observation_input"]
    assert billing_input["charge_item"] == "training"
    assert billing_input["quantity"] == 261
    assert billing_input["unit"] == "estimated_tokens"
    assert billing_input["route"] == "mint.vla.train_step"
    assert billing_input["dimension"] == "train"
    assert billing_input["model"] == "openpi/pi0-fast-libero-low-mem-finetune"
    assert billing_input["metadata"] == {
        "model_id": "model-123",
        "loss_fn": "cross_entropy",
        "datum_count": 1,
    }
    request_json = json.loads(queued["request_json"].decode("utf-8"))
    assert request_json["data"][0]["observation"]["state"]["shape"] == [8]
    assert request_json["data"][0]["supervision"]["target_tokens"]["shape"] == [2]


def test_mint_vla_train_step_route_releases_durable_inflight_on_enqueue_failure(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(fail=True, task_futures=task_futures)
    session = SimpleNamespace(
        model_id="model-123",
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
        backend="openpi_fast",
    )

    class _StubTrainingManager:
        def get_session(self, model_id: str):
            return session if model_id == "model-123" else None

        def mark_inflight(self, model_id: str, delta: int) -> None:
            raise AssertionError(
                f"VLA HTTP route must not use local manager inflight: {model_id} {delta}"
            )

    inflight_calls: list[tuple[str, int]] = []

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    async def _fake_enqueue_training_request_with_trace(**kwargs):
        await kwargs["enqueue_coro"]

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", object())
    monkeypatch.setattr(mint_routes, "training_manager", _StubTrainingManager())
    _install_billing_user(monkeypatch, mint_routes)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    from mint_server.routes import training as training_routes

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(
        training_routes,
        "_build_training_scheduler_extra",
        lambda *, session, model_id, training_op, seq_id=None: {
            "scheduler_domain": f"training:{session.base_model}",
            "scheduler_session_key": model_id,
            "training_op": training_op,
            "seq_id": seq_id,
            "backend": session.backend,
        },
    )
    monkeypatch.setattr(
        training_routes,
        "_enqueue_training_request_with_trace",
        _fake_enqueue_training_request_with_trace,
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/vla/train_step",
        json={
            "model_id": "model-123",
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "observation": {
                        "state": {
                            "data": [0.0] * 8,
                            "shape": [8],
                            "dtype": "float32",
                        },
                        "model_input": {
                            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                        },
                    },
                    "supervision": {
                        "target_tokens": {
                            "data": [11, 12],
                            "shape": [2],
                            "dtype": "int64",
                        },
                    },
                }
            ],
        },
    )

    assert resp.status_code == 503, resp.text
    assert inflight_calls == [("model-123", +1), ("model-123", -1)]


def test_mint_vla_train_step_background_lowers_observation_and_supervision(
    monkeypatch,
) -> None:
    from mint_server.models.mint_types import VLATrainStepRequest
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    captured: dict[str, object] = {}

    async def _fake_do_train_step(
        request_id: str, request, user_id: str | None, gateway_auth=None
    ) -> None:
        _ = request_id, user_id, gateway_auth
        captured["request"] = request

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(training_routes, "_do_train_step", _fake_do_train_step)

    request = VLATrainStepRequest.model_validate(
        {
            "model_id": "model-123",
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "observation": {
                        "state": {
                            "data": [0.0] * 8,
                            "shape": [8],
                            "dtype": "float32",
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "type": "image",
                                    "data": "aW1n",
                                    "format": "png",
                                    "expected_tokens": 256,
                                },
                                {"type": "encoded_text", "tokens": [1, 2, 3]},
                            ]
                        },
                    },
                    "supervision": {
                        "target_tokens": {
                            "data": [11, 12],
                            "shape": [2],
                            "dtype": "int64",
                        },
                        "weights": {
                            "data": [1.0, 1.0],
                            "shape": [2],
                            "dtype": "float32",
                        },
                        "token_ar_mask": {
                            "data": [1, 1],
                            "shape": [2],
                            "dtype": "int64",
                        },
                    },
                }
            ],
        }
    )

    asyncio.run(mint_routes._do_vla_train_step("req-1", request, "user-a"))

    internal = captured["request"]
    lowered = internal.forward_backward_input.data[0]
    assert lowered.model_input.chunks[1].tokens == [1, 2, 3]
    assert lowered.loss_fn_inputs["state"].shape == [8]
    assert lowered.loss_fn_inputs["target_tokens"].shape == [2]
    assert lowered.loss_fn_inputs["token_ar_mask"].shape == [2]


def test_mint_vla_train_step_background_releases_durable_inflight_on_lowering_failure(
    monkeypatch,
) -> None:
    from mint_server.models.mint_types import VLATrainStepRequest
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    inflight_calls: list[tuple[str, int]] = []

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    async def _unexpected_do_train_step(*_args, **_kwargs) -> None:
        raise AssertionError("_do_train_step should not run when VLA lowering fails")

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(training_routes, "_do_train_step", _unexpected_do_train_step)

    request = VLATrainStepRequest.model_validate(
        {
            "model_id": "model-123",
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "observation": {
                        "state": {
                            "data": [0.0] * 8,
                            "shape": [8],
                            "dtype": "float32",
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "type": "encoded_text",
                                    "tokens": [1, 2, 3],
                                }
                            ]
                        },
                    },
                    "supervision": {
                        "target_tokens": {
                            "data": [11],
                            "shape": [1],
                            "dtype": "int64",
                        },
                        "weights": {
                            "data": [1.0],
                            "shape": [1],
                            "dtype": "float32",
                        },
                        "token_ar_mask": {
                            "data": [1, 1],
                            "shape": [2],
                            "dtype": "int64",
                        },
                        "state": {
                            "data": [0.0],
                            "shape": [1],
                            "dtype": "float32",
                        },
                    },
                }
            ],
        }
    )

    asyncio.run(mint_routes._do_vla_train_step("req-lower-fail", request, "user-a"))

    assert task_futures.failed
    assert inflight_calls == [("model-123", -1)]


def test_mint_vla_train_step_route_uses_detached_session_info(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)

    async def _fake_route_session_info(model_id: str):
        assert model_id == "model-123"
        return {
            "model_id": "model-123",
            "base_model": "openpi/pi0-fast-libero-low-mem-finetune",
            "backend": "openpi_fast",
        }

    async def _fake_enqueue_training_request_with_trace(**kwargs):
        await kwargs["enqueue_coro"]

    async def _mark_training_inflight(_model_id: str, _delta: int) -> None:
        return None

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _fake_route_session_info
    )

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(
        training_routes,
        "_build_training_scheduler_extra",
        lambda *, session, model_id, training_op, seq_id=None: {
            "scheduler_domain": f"training:{session['base_model']}",
            "scheduler_session_key": model_id,
            "training_op": training_op,
            "seq_id": seq_id,
            "backend": session["backend"],
        },
    )
    monkeypatch.setattr(
        training_routes,
        "_enqueue_training_request_with_trace",
        _fake_enqueue_training_request_with_trace,
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/vla/train_step",
        json={
            "model_id": "model-123",
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "observation": {
                        "state": {"data": [0.0] * 8, "shape": [8], "dtype": "float32"},
                        "model_input": {
                            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                        },
                    },
                    "supervision": {
                        "target_tokens": {
                            "data": [11, 12],
                            "shape": [2],
                            "dtype": "int64",
                        },
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(scheduler.calls) == 1
    assert scheduler.calls[0]["op"] == "mint.vla.train_step"
    assert (
        scheduler.calls[0]["domain_key"]
        == "training:openpi/pi0-fast-libero-low-mem-finetune"
    )


def test_model_work_dispatch_executes_mint_vla_train_step(monkeypatch) -> None:
    from mint_server.backend.scheduling import model_work_dispatch as dispatch
    import ray

    captured: dict[str, object] = {}

    async def _fake_do_vla_train_step(
        request_id: str,
        request,
        user_id: str | None,
        gateway_auth=None,
        billing_observation_input=None,
    ) -> None:
        captured["request_id"] = request_id
        captured["request"] = request
        captured["user_id"] = user_id
        captured["gateway_auth"] = gateway_auth
        captured["billing_observation_input"] = billing_observation_input

    from mint_server.routes import mint as mint_routes

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(mint_routes, "_do_vla_train_step", _fake_do_vla_train_step)

    item = SimpleNamespace(
        op="mint.vla.train_step",
        request_id="req-1",
        request_json=json.dumps(
            {
                "model_id": "model-123",
                "loss_fn": "cross_entropy",
                "data": [
                    {
                        "observation": {
                            "state": {
                                "data": [0.0] * 8,
                                "shape": [8],
                                "dtype": "float32",
                            },
                            "model_input": {
                                "chunks": [
                                    {"type": "encoded_text", "tokens": [1, 2, 3]}
                                ]
                            },
                        },
                        "supervision": {
                            "target_tokens": {
                                "data": [11, 12],
                                "shape": [2],
                                "dtype": "int64",
                            },
                        },
                    }
                ],
            }
        ).encode("utf-8"),
        user_id="user-a",
        extra={
            "gateway_auth": {
                "account_id": "acct",
                "apikey_id": "key",
                "request_id": "gw-req",
            },
            "billing_observation_input": {"charge_item": "training", "quantity": 5},
        },
    )

    asyncio.run(dispatch.execute_model_work_item(item))

    assert captured["request_id"] == "req-1"
    assert captured["user_id"] == "user-a"
    assert captured["request"].model_id == "model-123"
    assert captured["gateway_auth"]["apikey_id"] == "key"
    assert captured["billing_observation_input"] == {
        "charge_item": "training",
        "quantity": 5,
    }


def test_model_work_dispatch_executes_mint_action_with_billing(monkeypatch) -> None:
    from mint_server.backend.scheduling import model_work_dispatch as dispatch
    import ray

    captured: dict[str, object] = {}

    async def _fake_do_act(
        request_id: str,
        request,
        billing_observations=None,
        gateway_auth=None,
        billing_observation_input=None,
    ) -> None:
        captured["request_id"] = request_id
        captured["request"] = request
        captured["billing_observations"] = billing_observations
        captured["gateway_auth"] = gateway_auth
        captured["billing_observation_input"] = billing_observation_input

    from mint_server.routes import action_sampling

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(action_sampling, "_do_act", _fake_do_act)

    item = SimpleNamespace(
        op="mint.action.act",
        request_id="act-1",
        request_json=json.dumps(
            {
                "action_session_id": "action-session-1",
                "observation": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "extra_inputs": {},
            }
        ).encode("utf-8"),
        user_id="user-a",
        extra={
            "gateway_auth": {
                "account_id": "acct",
                "apikey_id": "key",
                "request_id": "gw-act",
            },
            "billing_observation_input": {"charge_item": "inference", "quantity": 67},
        },
    )

    asyncio.run(dispatch.execute_model_work_item(item))

    assert captured["request_id"] == "act-1"
    assert captured["request"].action_session_id == "action-session-1"
    assert captured["billing_observations"] is None
    assert captured["gateway_auth"]["apikey_id"] == "key"
    assert captured["billing_observation_input"] == {
        "charge_item": "inference",
        "quantity": 67,
    }


def test_mint_interpolate_route_enqueues_expected_request(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", object())
    monkeypatch.setattr(mint_routes, "training_manager", object())
    _install_billing_user(monkeypatch, mint_routes)

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)

    resolved_flags: list[bool] = []

    def _resolve(path: str, *, user_id, is_admin, owner_id=None):
        resolved_flags.append(bool(is_admin))
        return f"/resolved/{user_id}/{path.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda _request: True)
    monkeypatch.setattr(mint_routes, "_resolve_checkpoint_for_user", _resolve)
    monkeypatch.setattr(
        mint_routes, "_require_peft_adapter_checkpoint", lambda _path: None
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/checkpoints/interpolate",
        json={
            "source_paths": [
                "mint://teacher/sampler_weights/ckpt-a",
                "mint://student/sampler_weights/ckpt-b",
            ],
            "coefficients": [0.9, 0.1],
            "output_path": "ema-0010",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "request_id" in body
    assert task_futures.created == [body["request_id"]]
    assert len(scheduler.calls) == 1
    queued = scheduler.calls[0]
    assert queued["op"] == "mint.interpolate_checkpoints"
    assert queued["request_id"] == body["request_id"]
    assert queued["user_id"] == _USER_ID
    assert queued["domain_key"] == "internal:runtime"
    queued_meta = queued["extra"]
    assert queued_meta["op"] == "mint.interpolate_checkpoints"
    assert queued_meta["queue_state"] == "queued"
    assert queued_meta["stage"] == "queued"
    assert isinstance(queued_meta["queued_at"], float)
    assert queued_meta["checkpoint_count"] == 2
    assert queued_meta["output_path"] == "ema-0010"
    assert "billing_observations" not in queued["extra"]
    billing_input = queued["extra"]["billing_observation_input"]
    assert billing_input["charge_item"] == "checkpoint_storage"
    assert billing_input["quantity"] == 2
    assert billing_input["unit"] == "checkpoint_inputs"
    assert billing_input["route"] == "mint.interpolate_checkpoints"
    assert billing_input["dimension"] == "checkpoint"
    assert billing_input["metadata"] == {
        "checkpoint_count": 2,
        "output_path": "ema-0010",
        "output_checkpoint_type": "sampler",
    }
    request_json = json.loads(queued["request_json"].decode("utf-8"))
    assert request_json["source_paths"] == [
        f"/resolved/{_USER_ID}/ckpt-a",
        f"/resolved/{_USER_ID}/ckpt-b",
    ]
    assert resolved_flags == [True, True]


def test_mint_interpolate_do_path_claims_checkpoint_and_writes_ckpt_id(
    monkeypatch, tmp_path
) -> None:
    from mint_server.models.mint_types import InterpolateCheckpointsRequest
    from mint_server.routes import mint as mint_routes
    import mint_server.backend.core.mintx_ops as mintx_ops

    task_futures = _StubTaskFutureService()
    written: dict[str, object] = {}
    claimed: dict[str, object] = {}

    async def _claim(**kwargs):
        claimed.update(kwargs)
        return "ckpt-rec-1"

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "_claim_sampler_checkpoint_or_raise", _claim)
    monkeypatch.setattr(
        mintx_ops,
        "_validate_source_metadata",
        lambda _paths: ("model-123", "Qwen/Qwen3-0.6B", "dense", {}),
    )
    monkeypatch.setattr(
        mint_routes,
        "build_persistent_cache_dir",
        lambda **_kwargs: str(
            tmp_path / "persistent_cache" / "owner-a" / "model-123" / "ema-0010"
        ),
    )
    monkeypatch.setattr(
        mint_routes,
        "interpolate_checkpoints_to_dir",
        lambda **_kwargs: SimpleNamespace(
            output_checkpoint_type="sampler", has_rank_shards=False
        ),
    )
    monkeypatch.setattr(
        mint_routes,
        "read_checkpoint_metadata",
        lambda _path: {"checkpoint_id": "ema-0010", "checkpoint_type": "sampler"},
    )
    monkeypatch.setattr(
        mint_routes,
        "write_checkpoint_metadata",
        lambda path, metadata: written.update({"path": path, "metadata": metadata}),
    )
    monkeypatch.setattr(
        mint_routes,
        "begin_async_checkpoint_mirror",
        lambda *_args, **_kwargs: (
            "/tos-mindverse/mint_checkpoints/owner-a/model-123/ema-0010/sampler"
        ),
    )
    monkeypatch.setattr(
        mint_routes,
        "checkpoint_uri",
        lambda model_id, checkpoint_name, prefer_tinker, checkpoint_type: (
            f"mint://{model_id}/sampler_weights/{checkpoint_name}"
        ),
    )

    request = InterpolateCheckpointsRequest(
        source_paths=["/resolved/ckpt-a", "/resolved/ckpt-b"],
        coefficients=[0.9, 0.1],
        output_path="ema-0010",
        retry=True,
    )

    gateway_auth = {
        "user_id": _USER_ID,
        "user_role": "user",
        "account_id": _ACCOUNT_ID,
        "apikey_id": _APIKEY_ID,
        "request_id": "gateway-request-1",
        "cap_write": True,
    }
    billing_input = {
        "charge_item": "checkpoint_storage",
        "quantity": 2,
        "unit": "checkpoint_inputs",
        "route": "mint.interpolate_checkpoints",
        "dimension": "checkpoint",
        "model": None,
        "metadata": {"checkpoint_count": 2},
    }
    asyncio.run(
        mint_routes._do_interpolate_checkpoints(
            "req-interp-1",
            request,
            "owner-a",
            gateway_auth,
            billing_input,
        )
    )

    assert claimed["owner_id"] == "owner-a"
    assert claimed["model_id"] == "model-123"
    assert claimed["raw_checkpoint_id"] == "ema-0010"
    assert claimed["retry"] is True
    assert written["path"].endswith("ema-0010")
    assert written["metadata"]["ckpt_id"] == "ckpt-rec-1"
    assert "created_at" in written["metadata"]
    assert task_futures.failed == []
    observations = task_futures.billing_observations["req-interp-1"]
    assert len(observations) == 1
    assert observations[0]["account_id"] == _ACCOUNT_ID
    assert observations[0]["apikey_id"] == _APIKEY_ID
    assert observations[0]["charge_item"] == "checkpoint_storage"
    assert observations[0]["quantity"] == 2
    assert observations[0]["route"] == "mint.interpolate_checkpoints"
    assert observations[0]["metadata"] == {"checkpoint_count": 2}
    assert task_futures.resolved == [
        (
            "req-interp-1",
            {
                "checkpoint_id": "ema-0010",
                "checkpoint_record_id": "ckpt-rec-1",
                "path": "mint://model-123/sampler_weights/ema-0010",
                "checkpoint_type": "sampler",
                "source_paths": ["/resolved/ckpt-a", "/resolved/ckpt-b"],
                "coefficients": [0.9, 0.1],
                "has_rank_shards": False,
                "filesystem_path": str(
                    tmp_path / "persistent_cache" / "owner-a" / "model-123" / "ema-0010"
                ),
                "persistent_filesystem_path": "/tos-mindverse/mint_checkpoints/owner-a/model-123/ema-0010/sampler",
                "mirror_status": "pending",
                "type": "mint_interpolate_checkpoints",
            },
        )
    ]


def test_mint_interpolate_do_path_marks_failed_checkpoint(
    monkeypatch, tmp_path
) -> None:
    from mint_server.models.mint_types import InterpolateCheckpointsRequest
    from mint_server.routes import mint as mint_routes
    import mint_server.backend.core.mintx_ops as mintx_ops

    task_futures = _StubTaskFutureService()
    failed_marks: list[tuple[str | None, str]] = []

    async def _claim(**_kwargs):
        return "ckpt-rec-failed"

    async def _mark_failed(ckpt_id: str | None, *, fail_reason: str) -> None:
        failed_marks.append((ckpt_id, fail_reason))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "_claim_sampler_checkpoint_or_raise", _claim)
    monkeypatch.setattr(mint_routes, "mark_checkpoint_failed", _mark_failed)
    monkeypatch.setattr(
        mintx_ops,
        "_validate_source_metadata",
        lambda _paths: ("model-123", "Qwen/Qwen3-0.6B", "dense", {}),
    )
    monkeypatch.setattr(
        mint_routes,
        "build_persistent_cache_dir",
        lambda **_kwargs: str(
            tmp_path / "persistent_cache" / "owner-a" / "model-123" / "ema-0011"
        ),
    )

    def _raise_interpolate(**_kwargs):
        raise RuntimeError("interpolate_failed")

    monkeypatch.setattr(
        mint_routes, "interpolate_checkpoints_to_dir", _raise_interpolate
    )

    request = InterpolateCheckpointsRequest(
        source_paths=["/resolved/ckpt-a", "/resolved/ckpt-b"],
        coefficients=[0.9, 0.1],
        output_path="ema-0011",
        retry=False,
    )

    asyncio.run(
        mint_routes._do_interpolate_checkpoints("req-interp-err", request, "owner-a")
    )

    assert failed_marks == [("ckpt-rec-failed", "upload_error")]
    assert task_futures.resolved == []
    assert task_futures.failed == [("req-interp-err", "interpolate_failed")]


def test_mint_interpolate_do_path_mark_failed_error_does_not_mask_root_failure(
    monkeypatch, tmp_path
) -> None:
    from mint_server.models.mint_types import InterpolateCheckpointsRequest
    from mint_server.routes import mint as mint_routes
    import mint_server.backend.core.mintx_ops as mintx_ops

    task_futures = _StubTaskFutureService()

    async def _claim(**_kwargs):
        return "ckpt-rec-failed"

    async def _mark_failed(_ckpt_id: str | None, *, fail_reason: str) -> None:
        assert fail_reason == "upload_error"
        raise RuntimeError("mark_failed_broken")

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "_claim_sampler_checkpoint_or_raise", _claim)
    monkeypatch.setattr(mint_routes, "mark_checkpoint_failed", _mark_failed)
    monkeypatch.setattr(
        mintx_ops,
        "_validate_source_metadata",
        lambda _paths: ("model-123", "Qwen/Qwen3-0.6B", "dense", {}),
    )
    monkeypatch.setattr(
        mint_routes,
        "build_persistent_cache_dir",
        lambda **_kwargs: str(
            tmp_path / "persistent_cache" / "owner-a" / "model-123" / "ema-0012"
        ),
    )

    def _raise_interpolate(**_kwargs):
        raise RuntimeError("interpolate_failed")

    monkeypatch.setattr(
        mint_routes, "interpolate_checkpoints_to_dir", _raise_interpolate
    )

    request = InterpolateCheckpointsRequest(
        source_paths=["/resolved/ckpt-a", "/resolved/ckpt-b"],
        coefficients=[0.9, 0.1],
        output_path="ema-0012",
        retry=False,
    )

    asyncio.run(
        mint_routes._do_interpolate_checkpoints("req-interp-err-2", request, "owner-a")
    )

    assert task_futures.resolved == []
    assert task_futures.failed == [("req-interp-err-2", "interpolate_failed")]


def test_mint_reverse_kl_route_and_background_path(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes
    from mint_server.models.mint_types import ForwardBackwardReverseKLRequest

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)
    inflight_calls: list[tuple[str, int]] = []

    class _StubSession:
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        model_id = "model-123"
        backend = "megatron"

    class _StubTrainingManager:
        def get_session(self, model_id: str):
            return _StubSession() if model_id == "model-123" else None

    class _StubTrainingEngine:
        async def forward_backward_reverse_kl(self, session, request):
            assert session.model_id == "model-123"
            assert request.reference_model_path == "/resolved/ref-step-0010"
            return {
                "outputs": [
                    {
                        "loss": {
                            "data": [0.25],
                            "shape": [1],
                            "dtype": "float32",
                        }
                    }
                ],
                "metrics": {
                    "loss:mean": 0.25,
                    "reverse_kl:mean": 0.25,
                    "num_samples:sum": 1.0,
                    "num_tokens:sum": 2.0,
                },
                "type": "mint_forward_backward_reverse_kl",
            }

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-123":
            return {
                "model_id": model_id,
                "session_id": "sess-123",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
            }
        return None

    async def _noop_protect(_info: dict) -> None:
        return None

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    async def _materialize_training_session_for_stateful_use(session):
        return session

    resolved_flags: list[bool] = []

    def _resolve(path, **kwargs):
        resolved_flags.append(bool(kwargs.get("is_admin")))
        return "/resolved/ref-step-0010"

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", _StubTrainingEngine())
    monkeypatch.setattr(mint_routes, "training_manager", _StubTrainingManager())
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda _request: True)
    monkeypatch.setattr(mint_routes, "_resolve_checkpoint_for_user", _resolve)
    monkeypatch.setattr(
        mint_routes, "_require_peft_adapter_checkpoint", lambda _path: None
    )
    monkeypatch.setattr(
        mint_routes, "_protect_training_session_enqueue_window", _noop_protect
    )
    monkeypatch.setattr(
        mint_routes, "_get_max_model_len", lambda _base_model: 2048, raising=False
    )

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(
        training_routes,
        "_materialize_training_session_for_stateful_use",
        _materialize_training_session_for_stateful_use,
    )
    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _get_training_route_session_info
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/sampler_weights/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "reference_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                },
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }
    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert task_futures.created == [request_id]
    assert len(scheduler.calls) == 1
    queued = scheduler.calls[0]
    assert queued["op"] == "mint.forward_backward_reverse_kl"
    assert queued["request_id"] == request_id
    queued_meta = queued["extra"]
    assert queued_meta["op"] == "mint.forward_backward_reverse_kl"
    assert queued_meta["model_id"] == "model-123"
    assert queued_meta["session_id"] == "sess-123"
    assert queued_meta["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert queued_meta["backend"] == "megatron"
    assert queued_meta["queue_state"] == "queued"
    assert queued_meta["stage"] == "queued"
    assert isinstance(queued_meta["queued_at"], float)
    queued_request_json = json.loads(queued["request_json"].decode("utf-8"))
    assert queued_request_json["reference_model_path"] == "/resolved/ref-step-0010"
    assert resolved_flags == [True]
    assert inflight_calls == [("model-123", +1)]

    request = ForwardBackwardReverseKLRequest.model_validate(queued_request_json)
    import asyncio

    asyncio.run(
        mint_routes._do_forward_backward_reverse_kl(request_id, request, "user-a")
    )
    assert inflight_calls == [("model-123", +1), ("model-123", -1)]

    assert task_futures.resolved == [
        (
            request_id,
            {
                "outputs": [
                    {
                        "loss": {
                            "data": [0.25],
                            "shape": [1],
                            "dtype": "float32",
                        }
                    }
                ],
                "metrics": {
                    "loss:mean": 0.25,
                    "reverse_kl:mean": 0.25,
                    "num_samples:sum": 1.0,
                    "num_tokens:sum": 2.0,
                },
                "type": "mint_forward_backward_reverse_kl",
            },
        )
    ]


def test_mint_reverse_kl_route_uses_detached_training_info_without_route_runtime(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)
    inflight_calls: list[tuple[str, int]] = []

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-123":
            return {
                "model_id": model_id,
                "session_id": "sess-123",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
            }
        return None

    async def _noop_protect(_info: dict) -> None:
        return None

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", None)
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda _request: False)
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, **_: "/resolved/ref-step-0010",
    )
    monkeypatch.setattr(
        mint_routes, "_require_peft_adapter_checkpoint", lambda _path: None
    )
    monkeypatch.setattr(
        mint_routes, "_protect_training_session_enqueue_window", _noop_protect
    )
    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _get_training_route_session_info
    )

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/sampler_weights/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "reference_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                },
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }
    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert task_futures.created == [request_id]
    assert len(scheduler.calls) == 1
    queued = scheduler.calls[0]
    assert queued["op"] == "mint.forward_backward_reverse_kl"
    assert queued["request_id"] == request_id
    queued_meta = queued["extra"]
    assert queued_meta["op"] == "mint.forward_backward_reverse_kl"
    assert queued_meta["model_id"] == "model-123"
    assert queued_meta["session_id"] == "sess-123"
    assert queued_meta["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert queued_meta["backend"] == "megatron"
    assert queued_meta["queue_state"] == "queued"
    assert queued_meta["stage"] == "queued"
    assert isinstance(queued_meta["queued_at"], float)
    queued_request_json = json.loads(queued["request_json"].decode("utf-8"))
    assert queued_request_json["reference_model_path"] == "/resolved/ref-step-0010"
    assert inflight_calls == [("model-123", +1)]


def test_mint_reverse_kl_route_releases_durable_inflight_on_enqueue_failure(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(fail=True, task_futures=task_futures)
    inflight_calls: list[tuple[str, int]] = []

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-123":
            return {
                "model_id": model_id,
                "session_id": "sess-123",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
            }
        return None

    async def _noop_protect(_info: dict) -> None:
        return None

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", None)
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda _request: False)
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, **_: "/resolved/ref-step-0010",
    )
    monkeypatch.setattr(
        mint_routes, "_require_peft_adapter_checkpoint", lambda _path: None
    )
    monkeypatch.setattr(
        mint_routes, "_protect_training_session_enqueue_window", _noop_protect
    )
    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _get_training_route_session_info
    )

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/sampler_weights/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "reference_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                },
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }

    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 503, resp.text
    assert inflight_calls == [("model-123", +1), ("model-123", -1)]
    assert task_futures.cleaned == []


def test_mint_reverse_kl_route_propagates_detached_store_503(monkeypatch) -> None:
    from fastapi import HTTPException
    from mint_server.routes import mint as mint_routes

    async def _raise_store_error(_model_id: str):
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        )

    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _raise_store_error
    )
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(mint_routes, "training_engine", None)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/sampler_weights/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "reference_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                },
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }

    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "Training session store unavailable"


def test_mint_reverse_kl_route_refreshes_detached_enqueue_protection(
    monkeypatch,
) -> None:
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    scheduler = _StubModelWorkScheduler(task_futures=task_futures)
    protected: list[dict] = []
    inflight_calls: list[tuple[str, int]] = []

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-123":
            return {
                "model_id": model_id,
                "session_id": "sess-123",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
            }
        return None

    async def _protect_training_session_enqueue_window(session_info: dict) -> None:
        protected.append(dict(session_info))

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", None)
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(mint_routes, "can_bypass_ownership", lambda _request: False)
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, **_: "/resolved/ref-step-0010",
    )
    monkeypatch.setattr(
        mint_routes, "_require_peft_adapter_checkpoint", lambda _path: None
    )
    monkeypatch.setattr(
        mint_routes,
        "_protect_training_session_enqueue_window",
        _protect_training_session_enqueue_window,
    )
    monkeypatch.setattr(
        mint_routes, "_get_route_training_store_info", _get_training_route_session_info
    )

    import mint_server.backend.scheduling.model_work_scheduler as mws

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/sampler_weights/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                },
                "reference_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                },
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }

    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 200, resp.text
    assert [entry["session_id"] for entry in protected] == ["sess-123"]
    assert inflight_calls == [("model-123", +1)]


def test_mint_reverse_kl_background_restores_detached_training_session(
    monkeypatch,
) -> None:
    from mint_server.models.mint_types import ForwardBackwardReverseKLRequest
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    inflight_calls: list[tuple[str, int]] = []
    restored: list[str] = []
    materialized: list[str] = []

    class _StubSession:
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        model_id = "model-123"
        backend = "megatron"

    class _StubTrainingManager:
        def get_session(self, model_id: str):
            assert model_id == "model-123"
            return None

    class _StubTrainingEngine:
        async def forward_backward_reverse_kl(self, session, request):
            assert session.model_id == "model-123"
            assert getattr(session, "materialized", False) is True
            assert request.reference_model_path == "/resolved/ref-step-0010"
            return {"ok": True, "type": "mint_forward_backward_reverse_kl"}

    async def _restore_training_session(model_id: str):
        restored.append(model_id)
        return _StubSession() if model_id == "model-123" else None

    async def _materialize_training_session_for_stateful_use(session):
        materialized.append(session.model_id)
        session.materialized = True
        return session

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", _StubTrainingEngine())
    monkeypatch.setattr(mint_routes, "training_manager", _StubTrainingManager())
    monkeypatch.setattr(
        training_routes, "_restore_training_session", _restore_training_session
    )
    monkeypatch.setattr(
        training_routes,
        "_materialize_training_session_for_stateful_use",
        _materialize_training_session_for_stateful_use,
    )
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)

    request = ForwardBackwardReverseKLRequest.model_validate(
        {
            "model_id": "model-123",
            "reference_model_path": "/resolved/ref-step-0010",
            "temperature": 1.0,
            "data": [
                {
                    "student_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "reference_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                    },
                    "target_tokens": {
                        "data": [7, 8],
                        "shape": [2],
                        "dtype": "int64",
                    },
                    "weights": {
                        "data": [1.0, 1.0],
                        "shape": [2],
                        "dtype": "float32",
                    },
                }
            ],
        }
    )

    asyncio.run(
        mint_routes._do_forward_backward_reverse_kl("req-restore", request, "user-a")
    )

    assert restored == ["model-123"]
    assert materialized == ["model-123"]
    assert inflight_calls == [("model-123", -1)]
    assert task_futures.resolved == [
        ("req-restore", {"ok": True, "type": "mint_forward_backward_reverse_kl"})
    ]


def test_mint_reverse_kl_background_releases_durable_inflight_on_early_failure(
    monkeypatch,
) -> None:
    from mint_server.models.mint_types import ForwardBackwardReverseKLRequest
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import training as training_routes

    task_futures = _StubTaskFutureService()
    inflight_calls: list[tuple[str, int]] = []

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(mint_routes, "task_futures", task_futures)
    monkeypatch.setattr(mint_routes, "training_engine", None)
    monkeypatch.setattr(mint_routes, "training_manager", None)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )

    request = ForwardBackwardReverseKLRequest.model_validate(
        {
            "model_id": "model-123",
            "reference_model_path": "/resolved/ref-step-0010",
            "temperature": 1.0,
            "data": [
                {
                    "student_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "reference_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]
                    },
                    "target_tokens": {
                        "data": [7, 8],
                        "shape": [2],
                        "dtype": "int64",
                    },
                    "weights": {
                        "data": [1.0, 1.0],
                        "shape": [2],
                        "dtype": "float32",
                    },
                }
            ],
        }
    )

    asyncio.run(
        mint_routes._do_forward_backward_reverse_kl("req-early-fail", request, "user-a")
    )

    assert task_futures.failed == [
        ("req-early-fail", "Training engine not initialized")
    ]
    assert inflight_calls == [("model-123", -1)]
