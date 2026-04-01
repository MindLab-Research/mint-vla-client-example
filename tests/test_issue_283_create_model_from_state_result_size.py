import json
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_issue_283_create_model_from_state_uses_small_result_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module
    import tinker_server.gateway as gateway_module

    training_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "run-283"
    ckpt_name = "weights-0001"
    ckpt_dir = (tmp_path / "runtime" / "persistent_cache" / "anonymous" / run_id / ckpt_name).resolve()
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": run_id,
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 8,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubCapacityManager:
        def __init__(self) -> None:
            self.calls: list[dict] = []

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
            return None

    class StubWorkQueue:
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
                    "request_json": request_json,
                    "user_id": user_id,
                    "webhook_url": webhook_url,
                    "extra": extra,
                }
            )

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            return None

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            return None

        async def async_cleanup(self, request_id: str) -> None:
            return None

    stub_capacity = StubCapacityManager()
    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", stub_capacity)
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(gateway_module, "get_gateway_config", lambda: None)
    monkeypatch.setattr(gateway_module, "remote_training_model", lambda model_id: None)
    monkeypatch.setattr(gateway_module, "upstream_for_model", lambda base_model: None)
    monkeypatch.setattr(training_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(training_routes, "training_engine", object())
    monkeypatch.setattr(training_routes, "training_manager", object())
    monkeypatch.setattr(training_routes, "can_access_model", lambda base_model, user_data: True)

    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "s283",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "state_path": f"tinker://{run_id}/weights/{ckpt_name}",
            "lora_config": {"rank": 8},
            "load_optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(stub_capacity.calls) == 1
    assert stub_capacity.calls[0]["object_store_bytes"] == 256 * 1024
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "training.create_model_from_state"
    assert stub_queue.calls[0]["extra"]["execution_serial_key"] == "training_session:s283_0"
    assert stub_queue.calls[0]["extra"]["scheduler_session_key"] == "s283_0"
    assert stub_queue.calls[0]["extra"]["training_op"] == "create_model_from_state"
    queued_payload = json.loads(stub_queue.calls[0]["request_json"].decode("utf-8"))
    assert queued_payload["state_path"] != f"tinker://{run_id}/weights/{ckpt_name}"
    assert queued_payload["state_path"].startswith(str(tmp_path))


def test_issue_283_create_model_from_state_missing_checkpoint_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.gateway as gateway_module

    training_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    monkeypatch.setattr(gateway_module, "get_gateway_config", lambda: None)
    monkeypatch.setattr(gateway_module, "remote_training_model", lambda model_id: None)
    monkeypatch.setattr(gateway_module, "upstream_for_model", lambda base_model: None)
    monkeypatch.setattr(training_routes, "training_engine", object())
    monkeypatch.setattr(training_routes, "training_manager", object())
    monkeypatch.setattr(training_routes, "can_access_model", lambda base_model, user_data: True)

    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "s283-missing",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "state_path": "tinker://missing-run/weights/missing-ckpt",
            "lora_config": {"rank": 8},
            "load_optimizer": False,
        },
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Checkpoint not found: tinker://missing-run/weights/missing-ckpt"


def test_issue_283_create_model_from_state_background_uses_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.models.types import CreateModelFromStateRequest, LoRAConfig

    checkpoint_dir = tmp_path / "resolved-checkpoint"
    checkpoint_dir.mkdir()

    class StubSession:
        def __init__(self) -> None:
            self.model_id = "s283-bg_0"
            self.current_step = 11
            self.learning_rate = 1e-4
            self.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
            self.backend = "megatron"
            self.created_at = "2026-03-13T00:00:00Z"
            self.last_activity = 0.0

    class StubTrainingManager:
        def get_session(self, model_id: str):
            return None

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

        def delete_session(self, model_id: str) -> None:
            return None

        def create_session(self, **kwargs):
            return StubSession()

    class StubTrainingEngine:
        def __init__(self) -> None:
            self.load_calls: list[dict] = []
            self._resource_pool_actor_names = {}

        async def unbind_session(self, session) -> None:
            return None

        async def create_training_session(self, session) -> None:
            return None

        async def load_weights(self, *, session, load_path: str, load_optimizer: bool) -> None:
            self.load_calls.append({"load_path": load_path, "load_optimizer": load_optimizer})

    class StubFutureStore:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        def resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    stub_engine = StubTrainingEngine()
    stub_future_store = StubFutureStore()

    monkeypatch.setattr(training_routes, "training_engine", stub_engine)
    monkeypatch.setattr(training_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(training_routes, "future_store", stub_future_store)

    req = CreateModelFromStateRequest(
        session_id="s283-bg",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        state_path=str(checkpoint_dir),
        lora_config=LoRAConfig(rank=8),
        load_optimizer=True,
    )

    asyncio.run(training_routes._do_create_model_from_state("req-283-bg", req, user_id=None))

    assert stub_engine.load_calls == [
        {"load_path": str(checkpoint_dir), "load_optimizer": True}
    ]
    assert stub_future_store.resolved == [
        ("req-283-bg", {"request_id": "req-283-bg", "model_id": "s283-bg_0", "type": "create_model_from_state"})
    ]


def test_issue_283_load_state_route_queues_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "load-route-283"
    ckpt_name = "weights-0001"
    ckpt_dir = tmp_path / "anonymous" / run_id / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": "model-283",
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 5,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
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
                    "request_json": request_json,
                    "user_id": user_id,
                    "webhook_url": webhook_url,
                    "extra": extra,
                }
            )

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    class StubSession:
        model_id = "model-283"
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        backend = "megatron"

    class StubTrainingManager:
        def get_session(self, model_id: str):
            _ = model_id
            return StubSession()

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", object())
    monkeypatch.setattr(weights_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"tinker://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    queued_payload = json.loads(stub_queue.calls[0]["request_json"].decode("utf-8"))
    expected_path = weights_routes._resolve_mint_path(
        f"tinker://{run_id}/weights/{ckpt_name}",
        user_id=None,
        is_admin=False,
    )
    assert queued_payload["path"] == expected_path


def test_issue_283_load_state_background_uses_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import weights as weights_routes
    from tinker_server.models.types import LoadStateRequest

    checkpoint_dir = tmp_path / "resolved-load-state"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "load-283",
                "owner_id": None,
                "model_id": "model-283",
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 5,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubSession:
        model_id = "model-283"
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        backend = "megatron"

    class StubTrainingManager:
        def get_session(self, model_id: str):
            return StubSession()

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

    class StubTrainingEngine:
        def __init__(self) -> None:
            self.load_calls: list[dict] = []

        async def load_weights(self, session, load_path: str, load_optimizer: bool) -> None:
            self.load_calls.append({"load_path": load_path, "load_optimizer": load_optimizer})

    class StubFutureStore:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        def resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    stub_engine = StubTrainingEngine()
    stub_future_store = StubFutureStore()

    monkeypatch.setattr(weights_routes, "training_engine", stub_engine)
    monkeypatch.setattr(weights_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(weights_routes, "future_store", stub_future_store)

    req = LoadStateRequest(model_id="model-283", path=str(checkpoint_dir), optimizer=True)

    asyncio.run(weights_routes._do_load_state("req-283-load", req, user_id=None))

    assert stub_engine.load_calls == [
        {"load_path": str(checkpoint_dir), "load_optimizer": True}
    ]
    assert stub_future_store.resolved == [
        ("req-283-load", {"path": str(checkpoint_dir), "type": "load_weights"})
    ]


def test_issue_283_save_routes_use_detached_training_info_without_route_runtime(monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
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

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    for route_path, op in (("save_weights", "weights.save_weights"), ("save_state", "weights.save_state")):
        resp = client.post(
            f"/api/v1/{route_path}",
            json={"model_id": "model-283", "path": f"{route_path}-283"},
        )
        assert resp.status_code == 200, resp.text

    assert [call["op"] for call in stub_queue.calls] == ["weights.save_weights", "weights.save_state"]
    assert [call["request_json"]["model_id"] for call in stub_queue.calls] == ["model-283", "model-283"]
    assert all(call["extra"]["execution_serial_key"] == "training_session:model-283" for call in stub_queue.calls)


def test_issue_283_load_state_route_uses_detached_training_info_without_route_runtime(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "load-route-283-detached"
    ckpt_name = "weights-0001"
    ckpt_dir = tmp_path / "anonymous" / run_id / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": "model-283",
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 5,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
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
                    "request_json": request_json,
                    "user_id": user_id,
                    "webhook_url": webhook_url,
                    "extra": extra,
                }
            )

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"tinker://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    queued_payload = json.loads(stub_queue.calls[0]["request_json"].decode("utf-8"))
    expected_path = weights_routes._resolve_mint_path(
        f"tinker://{run_id}/weights/{ckpt_name}",
        user_id=None,
        is_admin=False,
    )
    assert queued_payload["path"] == expected_path


def test_issue_283_save_routes_restore_inflight_protection(monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def enqueue(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    class StubTrainingManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def mark_inflight(self, model_id: str, delta: int) -> None:
            self.calls.append((model_id, delta))

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    manager = StubTrainingManager()
    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    for route_path in ("save_weights", "save_state"):
        resp = client.post(
            f"/api/v1/{route_path}",
            json={"model_id": "model-283", "path": f"{route_path}-283"},
        )
        assert resp.status_code == 200, resp.text

    assert manager.calls == [("model-283", 1), ("model-283", 1)]
    assert [call["op"] for call in stub_queue.calls] == ["weights.save_weights", "weights.save_state"]


def test_issue_283_load_state_route_restores_inflight_protection(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "load-route-283-inflight"
    ckpt_name = "weights-0001"
    ckpt_dir = tmp_path / "anonymous" / run_id / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": "model-283",
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 5,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def enqueue(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    class StubTrainingManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def mark_inflight(self, model_id: str, delta: int) -> None:
            self.calls.append((model_id, delta))

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    manager = StubTrainingManager()
    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"tinker://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert manager.calls == [("model-283", 1)]
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "weights.load_state"


@pytest.mark.parametrize("route_path", ["save_weights", "save_state", "load_state"])
def test_issue_283_weights_routes_propagate_detached_store_503(monkeypatch, route_path: str) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    import tinker_server.backend.training_session_store as training_store_module
    import tinker_server.gateway as gateway_module

    async def _get_training_route_session_info(_model_id: str):
        return None

    async def _async_get_training_session_info(_model_id: str):
        raise RuntimeError("store down")

    async def _unexpected_async_remote_training_model(*_args, **_kwargs):
        raise AssertionError("remote fallback should not run after detached-store failure")

    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)
    monkeypatch.setattr(training_store_module, "async_get_training_session_info", _async_get_training_session_info)
    monkeypatch.setattr(gateway_module, "async_remote_training_model", _unexpected_async_remote_training_model)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(weights_routes, "training_engine", None)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    payload = {"model_id": "model-283", "path": "ckpt-283"}
    if route_path == "load_state":
        payload["optimizer"] = False

    resp = client.post(f"/api/v1/{route_path}", json=payload)

    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "Training session store unavailable"
def test_issue_283_save_routes_refresh_detached_enqueue_protection(monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def enqueue(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    protected: list[dict] = []

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "session_id": "sess-283",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    async def _protect_training_session_enqueue_window(session_info: dict) -> None:
        protected.append(dict(session_info))

    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(weights_routes, "_protect_training_session_enqueue_window", _protect_training_session_enqueue_window)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    for route_path in ("save_weights", "save_state"):
        resp = client.post(
            f"/api/v1/{route_path}",
            json={"model_id": "model-283", "path": f"{route_path}-283"},
        )
        assert resp.status_code == 200, resp.text

    assert [call["op"] for call in stub_queue.calls] == ["weights.save_weights", "weights.save_state"]
    assert [entry["session_id"] for entry in protected] == ["sess-283", "sess-283"]

def test_issue_283_load_state_route_refreshes_detached_enqueue_protection(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes
    from tinker_server import checkpoints as checkpoints_module
    import tinker_server.backend.capacity_manager as capacity_module
    import tinker_server.backend.api_work_queue as work_queue_module

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "load-route-283-protect"
    ckpt_name = "weights-0001"
    ckpt_dir = tmp_path / "anonymous" / run_id / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": "model-283",
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "created_at": "2026-03-13T00:00:00Z",
                "step": 5,
                "checkpoint_type": "training",
                "optimizer_present": True,
                "backend": "megatron",
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    class StubCapacityManager:
        async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": True}

        async def async_release_all(self, request_id: str) -> None:
            _ = request_id

    class StubWorkQueue:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def enqueue(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    class StubFutureStore:
        async def async_create_with_id(self, request_id: str) -> None:
            _ = request_id

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            _ = (request_id, meta)

        async def async_cleanup(self, request_id: str) -> None:
            _ = request_id

    protected: list[dict] = []

    async def _get_training_route_session_info(model_id: str):
        if model_id == "model-283":
            return {
                "model_id": model_id,
                "session_id": "sess-283-load",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "user_id": None,
            }
        return None

    async def _protect_training_session_enqueue_window(session_info: dict) -> None:
        protected.append(dict(session_info))

    stub_queue = StubWorkQueue()

    monkeypatch.setattr(capacity_module, "capacity_manager", StubCapacityManager())
    monkeypatch.setattr(work_queue_module, "api_work_queue", stub_queue)
    monkeypatch.setattr(weights_routes, "future_store", StubFutureStore())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(weights_routes, "_protect_training_session_enqueue_window", _protect_training_session_enqueue_window)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"tinker://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "weights.load_state"
    assert [entry["session_id"] for entry in protected] == ["sess-283-load"]
