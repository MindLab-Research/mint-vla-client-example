from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

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
    ) -> dict[str, object]:
        self.act_calls.append(
            {
                "action_session_id": action_session_id,
                "observation": observation,
                "extra_inputs": extra_inputs,
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


class _FakeFutureStore:
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


class _AsyncFakeFutureStore:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []

    async def async_create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.queued.append((request_id, meta))

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)


class _StubCapacityManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.released: list[str] = []

    async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
        self.calls.append(
            {
                "request_id": request_id,
                "queue_bytes": queue_bytes,
                "object_store_bytes": object_store_bytes,
            }
        )
        return {"ok": True}

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)


class _StubQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def enqueue(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        user_id: str | None,
        webhook_url: str | None,
        extra: dict | None = None,
    ) -> None:
        self.calls.append(
            {
                "request_id": request_id,
                "op": op,
                "request_json": json.loads(request_json.decode("utf-8")),
                "user_id": user_id,
                "webhook_url": webhook_url,
                "extra": extra,
            }
        )


def test_create_action_session_route_returns_action_session_id(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(mint_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-1")

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions",
        json={
            "session_id": "session-1",
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "tinker://model-1/sampler_weights/export-1",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"action_session_id": "action-session-1"}
    assert manager.create_calls == [
        {
            "session_id": "session-1",
            "action_session_seq_id": None,
            "base_model": OPENPI_FAST_MODEL,
            "model_path": "tinker://model-1/sampler_weights/export-1",
            "user_id": "user-1",
        }
    ]


def test_act_route_rejects_missing_state(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes

    monkeypatch.setattr(mint_routes, "action_session_manager", _FakeActionSessionManager(), raising=False)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions/action-session-1/act",
        json={
            "observation": {
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
            "extra_inputs": {},
        },
    )

    assert resp.status_code == 400, resp.text
    assert "state" in resp.text


def test_delete_action_session_route_calls_shutdown(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes

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
    from tinker_server.routes import action_sampling as action_routes
    from tinker_server.models.types import ActRequest, EncodedTextChunk, ImageChunk, ModelInput, TensorData

    manager = _FakeActionSessionManager()
    future_store = _FakeFutureStore()
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(action_routes, "future_store", future_store, raising=False)

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

    assert manager.act_calls
    assert future_store.resolved == [
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
    assert future_store.failed == []


def test_do_act_logs_when_future_fail_marking_fails(monkeypatch) -> None:
    from tinker_server.routes import action_sampling as action_routes
    from tinker_server.models.types import ActRequest, EncodedTextChunk, ImageChunk, ModelInput, TensorData

    class _ExplodingActionSessionManager:
        async def act(self, *, action_session_id: str, observation, extra_inputs):
            _ = action_session_id, observation, extra_inputs
            raise RuntimeError("boom")

    class _ExplodingFutureStore(_FakeFutureStore):
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
    monkeypatch.setattr(action_routes, "future_store", _ExplodingFutureStore(), raising=False)
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
    from tinker_server.routes import mint as mint_routes

    future_store = _AsyncFakeFutureStore()
    capacity = _StubCapacityManager()
    queue = _StubQueue()

    monkeypatch.setattr(mint_routes, "future_store", future_store, raising=False)
    monkeypatch.setattr(mint_routes, "action_session_manager", object(), raising=False)

    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as queue_module

    monkeypatch.setattr(capacity_module, "capacity_manager", capacity)
    monkeypatch.setattr(queue_module, "api_work_queue", queue)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/action_sessions/action-session-1/act",
        json={
            "observation": {
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
            "extra_inputs": {
                "state": {
                    "data": [0.0] * 8,
                    "shape": [8],
                    "dtype": "float32",
                }
            },
        },
    )

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert future_store.created == [request_id]
    assert future_store.queued == [
        (
            request_id,
            {"op": "mint.action.act", "action_session_id": "action-session-1"},
        )
    ]
    assert len(capacity.calls) == 1
    assert len(queue.calls) == 1
    queued = queue.calls[0]
    assert queued["op"] == "mint.action.act"
    assert queued["request_json"]["action_session_id"] == "action-session-1"
    assert queued["request_json"]["extra_inputs"]["state"]["shape"] == [8]


def test_legacy_action_public_routes_are_not_exposed(monkeypatch) -> None:
    from tinker_server.routes import action_sampling as action_routes
    from tinker_server.routes import mint as mint_routes
    from tinker_server.routes import service as service_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(service_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(service_routes, "_get_user_id", lambda _request: "user-1")
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(action_routes, "future_store", _FakeFutureStore(), raising=False)
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
            "model_path": "tinker://model-1/sampler_weights/export-1",
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
