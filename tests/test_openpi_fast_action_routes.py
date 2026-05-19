from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


class _FakeActionSessionManager:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.act_calls: list[dict[str, object]] = []
        self.shutdown_calls: list[str] = []

    async def create_session(
        self,
        *,
        session_id: str,
        action_session_seq_id: int | None,
        base_model: str,
        model_path: str | None,
        user_id: str | None,
    ) -> str:
        self.create_calls.append(
            {
                "session_id": session_id,
                "action_session_seq_id": action_session_seq_id,
                "base_model": base_model,
                "model_path": model_path,
                "user_id": user_id,
            }
        )
        return "action-session-1"

    async def act(
        self,
        *,
        action_session_id: str,
        observation,
        extra_inputs,
        temperature=None,
    ) -> dict[str, object]:
        self.act_calls.append(
            {
                "action_session_id": action_session_id,
                "observation": observation,
                "extra_inputs": extra_inputs,
                "temperature": temperature,
            }
        )
        return {
            "actions": {
                "data": [0.0] * 28,
                "shape": [4, 7],
                "dtype": "float32",
            },
            "policy_timing": {"infer_ms": 8.0},
        }

    async def shutdown_session(self, action_session_id: str) -> None:
        self.shutdown_calls.append(action_session_id)


class _FakeTaskFutureService:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.resolved: list[tuple[str, dict[str, object]]] = []
        self.failed: list[tuple[str, str]] = []

    def create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    def mark_queued(self, request_id: str, meta=None) -> None:
        _ = request_id, meta

    def resolve(self, request_id: str, payload: dict[str, object]) -> None:
        self.resolved.append((request_id, payload))

    def fail(self, request_id: str, error: str) -> None:
        self.failed.append((request_id, error))


class _AsyncFakeTaskFutureService:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []

    async def async_create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    async def async_create_model_work_with_id(self, request_id: str, **_kwargs) -> None:
        self.created.append(request_id)

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.queued.append((request_id, meta))

    async def async_update_meta(self, _request_id: str, _meta: dict) -> None:
        return None

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)


class _AsyncResolvingTaskFutureService(_AsyncFakeTaskFutureService):
    def __init__(self) -> None:
        super().__init__()
        self.resolved: list[tuple[str, dict[str, object]]] = []
        self.failed: list[tuple[str, str]] = []

    async def async_resolve(self, request_id: str, payload: dict[str, object]) -> None:
        self.resolved.append((request_id, payload))

    async def async_fail(self, request_id: str, error: str) -> None:
        self.failed.append((request_id, error))


class _StubModelWorkScheduler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def append(self, **kwargs) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {"ok": True, "scheduler_instance_id": "scheduler-action"}

    async def cancel_request(self, **kwargs) -> dict[str, object]:
        return {"ok": True, **dict(kwargs)}


def test_create_action_session_route_resolves_checkpoint_path_before_manager(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    manager = _FakeActionSessionManager()
    resolve_calls: list[dict[str, object]] = []
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", OPENPI_FAST_MODEL)
    monkeypatch.setattr(mint_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-1")
    monkeypatch.setattr(mint_routes, "_can_bypass_checkpoint_ownership", lambda _request: False)
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, *, user_id, is_admin, owner_id=None: (
            resolve_calls.append({"path": path, "user_id": user_id, "is_admin": is_admin, "owner_id": owner_id}),
            "/runtime/persistent_cache/user-1/model-1/export-1",
        )[1],
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions",
        json={
            "session_id": "session-1",
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "mint://model-1/sampler_weights/export-1",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"action_session_id": "action-session-1"}
    assert resolve_calls == [
        {
            "path": "mint://model-1/sampler_weights/export-1",
            "user_id": "user-1",
            "is_admin": False,
            "owner_id": None,
        }
    ]
    assert manager.create_calls == [
        {
            "session_id": "session-1",
            "action_session_seq_id": None,
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "/runtime/persistent_cache/user-1/model-1/export-1",
            "user_id": "user-1",
        }
    ]


def test_create_action_session_route_infers_base_model_with_admin_scope(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    manager = _FakeActionSessionManager()
    infer_calls: list[dict[str, object]] = []
    resolve_calls: list[dict[str, object]] = []
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", OPENPI_FAST_MODEL)
    monkeypatch.setattr(mint_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "admin")
    monkeypatch.setattr(mint_routes, "_can_bypass_checkpoint_ownership", lambda _request: True)
    monkeypatch.setattr(
        mint_routes,
        "_infer_base_model_from_checkpoint",
        lambda model_path, *, user_id, is_admin: (
            infer_calls.append({"model_path": model_path, "user_id": user_id, "is_admin": is_admin}),
            OPENPI_FAST_MODEL,
        )[1],
    )
    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, *, user_id, is_admin, owner_id=None: (
            resolve_calls.append({"path": path, "user_id": user_id, "is_admin": is_admin, "owner_id": owner_id}),
            "/runtime/persistent_cache/anonymous/model-1/export-1",
        )[1],
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions",
        json={
            "session_id": "session-1",
            "model_path": "mint://model-1/sampler_weights/export-1",
            "owner_id": "owner-a",
        },
    )

    assert resp.status_code == 200, resp.text
    assert infer_calls == [
        {
            "model_path": "mint://model-1/sampler_weights/export-1",
            "user_id": "owner-a",
            "is_admin": True,
        }
    ]
    assert resolve_calls == [
        {
            "path": "mint://model-1/sampler_weights/export-1",
            "user_id": "admin",
            "is_admin": True,
            "owner_id": "owner-a",
        }
    ]
    assert manager.create_calls == [
        {
            "session_id": "session-1",
            "action_session_seq_id": None,
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "/runtime/persistent_cache/anonymous/model-1/export-1",
            "user_id": "admin",
        }
    ]


def test_act_route_rejects_missing_state(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    monkeypatch.setattr(mint_routes, "action_session_manager", _FakeActionSessionManager(), raising=False)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions/action-session-1/act",
        json={
            "observation": {
                "model_input": {
                    "chunks": [
                        {
                            "type": "image",
                            "data": "aW1n",
                            "format": "png",
                            "expected_tokens": 256,
                        },
                        {
                            "type": "encoded_text",
                            "tokens": [1, 2, 3],
                        },
                    ]
                }
            },
        },
    )

    assert resp.status_code == 422, resp.text
    assert "state" in resp.text


def test_delete_action_session_route_calls_shutdown(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(mint_routes, "action_session_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.delete("/api/v1/mint/action_sessions/action-session-1")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "action_session_id": "action-session-1",
        "status": "deleted",
    }
    assert manager.shutdown_calls == ["action-session-1"]


def test_do_act_resolves_future_with_actions(monkeypatch) -> None:
    from mint_server.routes import action_sampling as action_routes
    from mint_server.models.types import ActRequest, EncodedTextChunk, ImageChunk, ModelInput, TensorData

    manager = _FakeActionSessionManager()
    task_futures = _FakeTaskFutureService()
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(action_routes, "task_futures", task_futures, raising=False)

    request = ActRequest(
        action_session_id="action-session-1",
        observation=ModelInput(
            chunks=[
                ImageChunk(data=b"img", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[1, 2, 3]),
            ]
        ),
        extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
        temperature=3.0,
    )

    asyncio.run(action_routes._do_act("req-1", request))

    assert manager.act_calls == [
        {
            "action_session_id": "action-session-1",
            "observation": request.observation,
            "extra_inputs": request.extra_inputs,
            "temperature": 3.0,
        }
    ]
    assert task_futures.resolved == [
        (
            "req-1",
            {
                "actions": {
                    "data": [0.0] * 28,
                    "shape": [4, 7],
                    "dtype": "float32",
                },
                "policy_timing": {"infer_ms": 8.0},
                "type": "act",
            },
        )
    ]
    assert task_futures.failed == []


def test_do_act_prefers_async_task_futures_api(monkeypatch) -> None:
    from mint_server.routes import action_sampling as action_routes
    from mint_server.models.types import ActRequest, EncodedTextChunk, ImageChunk, ModelInput, TensorData

    manager = _FakeActionSessionManager()
    task_futures = _AsyncResolvingTaskFutureService()
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(action_routes, "task_futures", task_futures, raising=False)

    request = ActRequest(
        action_session_id="action-session-1",
        observation=ModelInput(
            chunks=[
                ImageChunk(data=b"img", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[1, 2, 3]),
            ]
        ),
        extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
        temperature=1.5,
    )

    asyncio.run(action_routes._do_act("req-2", request))

    assert manager.act_calls == [
        {
            "action_session_id": "action-session-1",
            "observation": request.observation,
            "extra_inputs": request.extra_inputs,
            "temperature": 1.5,
        }
    ]
    assert task_futures.resolved == [
        (
            "req-2",
            {
                "actions": {
                    "data": [0.0] * 28,
                    "shape": [4, 7],
                    "dtype": "float32",
                },
                "policy_timing": {"infer_ms": 8.0},
                "type": "act",
            },
        )
    ]
    assert task_futures.failed == []


def test_do_act_logs_when_future_fail_marking_fails(monkeypatch) -> None:
    from mint_server.routes import action_sampling as action_routes
    from mint_server.models.types import ActRequest, EncodedTextChunk, ImageChunk, ModelInput, TensorData

    class _ExplodingActionSessionManager:
        async def act(self, *, action_session_id: str, observation, extra_inputs):
            _ = action_session_id, observation, extra_inputs
            raise RuntimeError("boom")

    class _ExplodingTaskFutureService(_FakeTaskFutureService):
        def fail(self, request_id: str, error: str) -> None:
            _ = request_id, error
            raise RuntimeError("fail-store-boom")

    log_calls: list[str] = []

    def _record_log(message: str, request_id: str) -> None:
        log_calls.append(message % request_id)

    monkeypatch.setattr(
        action_routes,
        "action_session_manager",
        _ExplodingActionSessionManager(),
        raising=False,
    )
    monkeypatch.setattr(action_routes, "task_futures", _ExplodingTaskFutureService(), raising=False)
    monkeypatch.setattr(action_routes.logger, "exception", _record_log)

    request = ActRequest(
        action_session_id="action-session-1",
        observation=ModelInput(
            chunks=[
                ImageChunk(data=b"img", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[1, 2, 3]),
            ]
        ),
        extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
    )

    asyncio.run(action_routes._do_act("req-1", request))

    assert log_calls == [
        "[act] failed to mark future failed: request_id=req-1",
        "[act] background failed: request_id=req-1",
    ]


def test_mint_action_route_enqueues_expected_request(monkeypatch) -> None:
    from mint_server.routes import mint as mint_routes

    task_futures = _AsyncFakeTaskFutureService()
    scheduler = _StubModelWorkScheduler()

    monkeypatch.setattr(mint_routes, "task_futures", task_futures, raising=False)
    monkeypatch.setattr(mint_routes, "action_session_manager", object(), raising=False)

    import mint_server.backend.model_work_scheduler as mws

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
                        },
                        {
                            "type": "encoded_text",
                            "tokens": [1, 2, 3],
                        },
                    ]
                },
            },
        },
    )

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert task_futures.created == []
    queued_request_id, queued_meta = task_futures.queued[0]
    assert queued_request_id == request_id
    assert queued_meta["op"] == "mint.action.act"
    assert queued_meta["action_session_id"] == "action-session-1"
    assert queued_meta["queue_state"] == "queued"
    assert len(scheduler.calls) == 1
    queued = scheduler.calls[0]
    assert queued["op"] == "mint.action.act"
    assert queued["domain_key"] == "internal:control"
    request_json = json.loads(queued["request_json"].decode("utf-8"))
    assert request_json["action_session_id"] == "action-session-1"
    assert request_json["extra_inputs"]["state"]["shape"] == [8]


def test_legacy_action_public_routes_are_not_exposed(monkeypatch) -> None:
    from mint_server.routes import action_sampling as action_routes
    from mint_server.routes import mint as mint_routes
    from mint_server.routes import service as service_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(service_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(service_routes, "_get_user_id", lambda _request: "user-1")
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(action_routes, "task_futures", _FakeTaskFutureService(), raising=False)
    monkeypatch.setattr(mint_routes, "action_session_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(service_routes.router, prefix="/api/v1")
    app.include_router(action_routes.router, prefix="/api/v1")
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    create_resp = client.post(
        "/api/v1/create_action_session",
        json={
            "session_id": "session-1",
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "mint://model-1/sampler_weights/export-1",
        },
    )
    act_resp = client.post(
        "/api/v1/act",
        json={
            "action_session_id": "action-session-1",
            "observation": {
                "chunks": [
                    {
                        "type": "image",
                        "data": "aW1n",
                        "format": "png",
                        "expected_tokens": 256,
                    }
                ]
            },
            "extra_inputs": {
                "state": {
                    "data": [0.0] * 8,
                    "shape": [8],
                    "dtype": "float32",
                }
            },
        },
    )
    delete_resp = client.delete("/api/v1/action_sessions/action-session-1")

    assert create_resp.status_code == 404, create_resp.text
    assert act_resp.status_code == 404, act_resp.text
    assert delete_resp.status_code == 404, delete_resp.text
