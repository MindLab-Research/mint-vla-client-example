from __future__ import annotations

import asyncio
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


def test_create_action_session_route_returns_action_session_id(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(service_routes, "action_session_manager", manager, raising=False)
    monkeypatch.setattr(service_routes, "_get_user_id", lambda _request: "user-1")

    app = FastAPI()
    app.include_router(service_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_action_session",
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
    from tinker_server.routes import action_sampling as action_routes

    monkeypatch.setattr(action_routes, "action_session_manager", _FakeActionSessionManager(), raising=False)
    monkeypatch.setattr(action_routes, "future_store", _FakeFutureStore(), raising=False)

    app = FastAPI()
    app.include_router(action_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
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
    from tinker_server.routes import action_sampling as action_routes

    manager = _FakeActionSessionManager()
    monkeypatch.setattr(action_routes, "action_session_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(action_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.delete("/api/v1/action_sessions/action-session-1")

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


def test_act_route_returns_future_id_and_schedules_background_work(monkeypatch) -> None:
    from tinker_server.routes import action_sampling as action_routes

    future_store = _FakeFutureStore()
    scheduled: list[object] = []

    monkeypatch.setattr(action_routes, "future_store", future_store, raising=False)
    monkeypatch.setattr(action_routes, "action_session_manager", _FakeActionSessionManager(), raising=False)
    monkeypatch.setattr(
        action_routes.asyncio,
        "create_task",
        lambda coro: scheduled.append(coro) or SimpleNamespace(),
    )

    app = FastAPI()
    app.include_router(action_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
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
    assert "request_id" in resp.json()
    assert future_store.created == [resp.json()["request_id"]]
    assert len(scheduled) == 1

    scheduled[0].close()
