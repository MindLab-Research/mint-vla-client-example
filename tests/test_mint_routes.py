from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient


class _StubFutureStore:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []
        self.resolved: list[tuple[str, dict]] = []
        self.failed: list[tuple[str, str]] = []

    async def async_create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.queued.append((request_id, meta))

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    def resolve(self, request_id: str, payload: dict) -> None:
        self.resolved.append((request_id, payload))

    async def async_fail(self, request_id: str, message: str) -> None:
        self.failed.append((request_id, message))


class _StubCapacityManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []
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
        self.calls: list[dict] = []

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


def test_mint_action_route_cleans_up_future_when_enqueue_fails(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes

    future_store = _StubFutureStore()
    capacity = _StubCapacityManager()

    class _ExplodingQueue:
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
            _ = request_id, op, request_json, user_id, webhook_url, extra
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr(mint_routes, "future_store", future_store, raising=False)
    monkeypatch.setattr(mint_routes, "action_session_manager", object(), raising=False)

    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as queue_module

    monkeypatch.setattr(capacity_module, "capacity_manager", capacity)
    monkeypatch.setattr(queue_module, "api_work_queue", _ExplodingQueue())

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

    assert resp.status_code == 503, resp.text
    assert len(future_store.created) == 1
    assert future_store.cleaned == future_store.created
    assert capacity.released == future_store.created


def test_mint_interpolate_route_enqueues_expected_request(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes

    future_store = _StubFutureStore()
    capacity = _StubCapacityManager()
    queue = _StubQueue()

    monkeypatch.setattr(mint_routes, "future_store", future_store)
    monkeypatch.setattr(mint_routes, "training_engine", object())
    monkeypatch.setattr(mint_routes, "training_manager", object())
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")

    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as queue_module

    monkeypatch.setattr(capacity_module, "capacity_manager", capacity)
    monkeypatch.setattr(queue_module, "api_work_queue", queue)

    monkeypatch.setattr(
        mint_routes,
        "_resolve_checkpoint_for_user",
        lambda path, *, user_id, is_admin: f"/resolved/{user_id}/{path.rsplit('/', 1)[-1]}",
    )

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/mint/checkpoints/interpolate",
        json={
            "source_paths": ["mint://teacher/ckpt-a", "mint://student/ckpt-b"],
            "coefficients": [0.9, 0.1],
            "output_path": "ema-0010",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "request_id" in body
    assert future_store.created == [body["request_id"]]
    assert future_store.queued == [
        (body["request_id"], {"op": "mint.interpolate_checkpoints"})
    ]
    assert len(queue.calls) == 1
    queued = queue.calls[0]
    assert queued["op"] == "mint.interpolate_checkpoints"
    assert queued["user_id"] == "user-a"
    assert queued["request_json"]["source_paths"] == [
        "mint://teacher/ckpt-a",
        "mint://student/ckpt-b",
    ]


def test_mint_reverse_kl_route_and_background_path(monkeypatch) -> None:
    from tinker_server.routes import mint as mint_routes
    from tinker_server.models.mint_types import ForwardBackwardReverseKLRequest

    future_store = _StubFutureStore()
    capacity = _StubCapacityManager()
    queue = _StubQueue()

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

    monkeypatch.setattr(mint_routes, "future_store", future_store)
    monkeypatch.setattr(mint_routes, "training_engine", _StubTrainingEngine())
    monkeypatch.setattr(mint_routes, "training_manager", _StubTrainingManager())
    monkeypatch.setattr(mint_routes, "_get_user_id", lambda _request: "user-a")
    monkeypatch.setattr(mint_routes, "is_admin_request", lambda _request: False)
    monkeypatch.setattr(mint_routes, "_resolve_checkpoint_for_user", lambda path, **_: "/resolved/ref-step-0010")
    monkeypatch.setattr(mint_routes, "_get_max_model_len", lambda _base_model: 2048, raising=False)

    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as queue_module
    from tinker_server.routes import training as training_routes

    monkeypatch.setattr(capacity_module, "capacity_manager", capacity)
    monkeypatch.setattr(queue_module, "api_work_queue", queue)
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)

    app = FastAPI()
    app.include_router(mint_routes.router, prefix="/api/v1/mint")
    client = TestClient(app)

    payload = {
        "model_id": "model-123",
        "reference_model_path": "mint://teacher/ref-step-0010",
        "temperature": 1.0,
        "data": [
            {
                "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "reference_input": {"chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]},
                "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }
    resp = client.post("/api/v1/mint/forward_backward_reverse_kl", json=payload)

    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    assert future_store.created == [request_id]
    assert future_store.queued == [
        (request_id, {"op": "mint.forward_backward_reverse_kl", "model_id": "model-123"})
    ]
    assert len(queue.calls) == 1
    queued = queue.calls[0]
    assert queued["op"] == "mint.forward_backward_reverse_kl"
    assert queued["request_json"]["reference_model_path"] == "/resolved/ref-step-0010"
    assert capacity.calls[0]["object_store_bytes"] == 256 * 1024

    request = ForwardBackwardReverseKLRequest.model_validate(queued["request_json"])
    import asyncio

    asyncio.run(mint_routes._do_forward_backward_reverse_kl(request_id, request, "user-a"))

    assert future_store.resolved == [
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
