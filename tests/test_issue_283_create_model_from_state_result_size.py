import json
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _make_write_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def inject_write_user(request: Request, call_next):
        request.state.user_data = {"user_role": "internal", "caps_from_headers": False}
        return await call_next(request)

    return app


class StubModelWorkScheduler:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def append(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        return {"ok": True, "scheduler_instance_id": "scheduler-283"}

    async def cancel_request(self, **kwargs) -> dict:
        return {"ok": True, **dict(kwargs)}


def _queued_payload(call: dict) -> dict:
    raw = call["request_json"]
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    return dict(raw)


def test_issue_283_create_model_from_state_queues_resolved_checkpoint_path_without_admission_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    from mint_server.routes import training as training_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.backend.model_work_scheduler as scheduler_module
    import mint_server.gateway as gateway_module

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

    class StubTaskFutureService:
        async def async_create_with_id(self, request_id: str) -> None:
            return None

        async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
            return None

        async def async_cleanup(self, request_id: str) -> None:
            return None

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(gateway_module, "get_gateway_config", lambda: None)
    monkeypatch.setattr(gateway_module, "remote_training_model", lambda model_id: None)
    monkeypatch.setattr(gateway_module, "upstream_for_model", lambda base_model: None)
    monkeypatch.setattr(training_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(training_routes, "training_engine", object())
    monkeypatch.setattr(training_routes, "training_manager", object())
    monkeypatch.setattr(training_routes, "can_access_model", lambda base_model, user_data: True)

    app = _make_write_app()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "s283",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "state_path": f"mint://{run_id}/weights/{ckpt_name}",
            "lora_config": {"rank": 8},
            "load_optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "training.create_model_from_state"
    assert stub_queue.calls[0]["extra"]["execution_serial_key"] == "training_session:s283_0"
    assert stub_queue.calls[0]["extra"]["scheduler_session_key"] == "s283_0"
    assert stub_queue.calls[0]["extra"]["training_op"] == "create_model_from_state"
    queued_payload = _queued_payload(stub_queue.calls[0])
    assert queued_payload["state_path"] != f"mint://{run_id}/weights/{ckpt_name}"
    assert queued_payload["state_path"].startswith(str(tmp_path))


def test_issue_283_create_model_from_state_missing_checkpoint_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    from mint_server.routes import training as training_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.gateway as gateway_module

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

    app = _make_write_app()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "s283-missing",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "state_path": "mint://missing-run/weights/missing-ckpt",
            "lora_config": {"rank": 8},
            "load_optimizer": False,
        },
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Checkpoint not found: mint://missing-run/weights/missing-ckpt"


def test_issue_283_create_model_from_state_background_uses_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.models.types import CreateModelFromStateRequest, LoRAConfig

    checkpoint_dir = tmp_path / "resolved-checkpoint"
    checkpoint_dir.mkdir()

    class StubSession:
        def __init__(self) -> None:
            self.model_id = "s283-bg_0"
            self.current_step = 11
            self.learning_rate = 1e-4
            self.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
            self.backend = "megatron"
            self.lora_config = LoRAConfig(rank=8)
            self.created_at = "2026-03-13T00:00:00Z"
            self.last_activity = 0.0

    class StubTrainingManager:
        def __init__(self) -> None:
            self.persisted: list[str] = []

        def get_session(self, model_id: str):
            return None

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

        def mark_persisted(self, model_id: str) -> None:
            self.persisted.append(model_id)

        def delete_session(self, model_id: str) -> None:
            return None

        def create_session(self, **kwargs):
            return StubSession()

    class StubTrainingEngine:
        def __init__(self) -> None:
            self.load_calls: list[dict] = []
            self._model_actor_supervisor_actor_names = {}

        async def unbind_session(self, session) -> None:
            return None

        async def create_training_session(self, session) -> None:
            return None

        async def load_weights(self, *, session, load_path: str, load_optimizer: bool) -> dict:
            self.load_calls.append({"load_path": load_path, "load_optimizer": load_optimizer})
            return {
                "backend": "bumblebee",
                "checkpoint_path": str(checkpoint_dir),
                "adapter_model_path": str(checkpoint_dir / "adapter_model.safetensors"),
                "loaded_tensors": 123,
                "migration_source_backend": "megatron",
                "migration_target_backend": "bumblebee",
                "migration_mode": "weights_only",
                "optimizer_restored": False,
                "requested_optimizer_restore": bool(load_optimizer),
            }

        async def get_tokenizer_info(self, session) -> dict:
            _ = session
            return {}

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        def resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    stub_engine = StubTrainingEngine()
    stub_task_futures = StubTaskFutureService()

    import mint_server.backend.session_index_store as session_index_store_module
    import mint_server.backend.training_session_store as training_store_module

    training_store_updates: list[dict] = []
    session_index_updates: list[tuple[str, str, str | None, str]] = []

    async def _async_upsert_training_session(info: dict) -> None:
        training_store_updates.append(dict(info))

    monkeypatch.setattr(training_routes, "training_engine", stub_engine)
    monkeypatch.setattr(training_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(training_routes, "task_futures", stub_task_futures)
    monkeypatch.setattr(training_store_module, "async_upsert_training_session", _async_upsert_training_session)
    monkeypatch.setattr(
        session_index_store_module,
        "add_training_run_to_session",
        lambda session_id, training_run_id, user_id=None, created_at=None: session_index_updates.append(
            (session_id, training_run_id, user_id, created_at)
        ),
    )

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
    assert stub_task_futures.resolved == [
        (
            "req-283-bg",
            {
                "request_id": "req-283-bg",
                "model_id": "s283-bg_0",
                "type": "create_model_from_state",
                "load_metadata": {
                    "migration_source_backend": "megatron",
                    "migration_target_backend": "bumblebee",
                    "migration_mode": "weights_only",
                    "optimizer_restored": False,
                    "requested_optimizer_restore": True,
                },
            },
        )
    ]
    assert training_store_updates[0]["model_id"] == "s283-bg_0"
    assert training_store_updates[0]["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert training_store_updates[0]["lora_config"]["rank"] == 8
    assert session_index_updates == [("s283-bg", "s283-bg_0", None, "2026-03-13T00:00:00Z")]


def test_issue_417_create_model_from_state_persists_loaded_lora_config(
    tmp_path: Path, monkeypatch
) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.backend.training_session_manager import TrainingSessionManager
    from mint_server.models.types import CreateModelFromStateRequest, LoRAConfig
    import mint_server.backend.session_index_store as session_index_store_module
    import mint_server.backend.training_session_store as training_store_module

    checkpoint_dir = tmp_path / "loaded-config-checkpoint"
    checkpoint_dir.mkdir()
    model_id = "s417-cmfs_0"
    manager = TrainingSessionManager()

    class StubTrainingEngine:
        def __init__(self) -> None:
            self._model_actor_supervisor_actor_names = {model_id: "megatron-actor-417"}

        async def create_training_session(self, session) -> None:
            return None

        async def load_weights(self, *, session, load_path: str, load_optimizer: bool) -> None:
            assert load_path == str(checkpoint_dir)
            assert load_optimizer is True
            session.current_step = 17
            session.learning_rate = 3e-4
            session.lora_config = LoRAConfig(
                rank=16,
                train_attn=False,
                train_mlp=True,
                train_unembed=False,
            )

        async def get_tokenizer_info(self, session) -> dict:
            _ = session
            return {}

        async def shutdown_session(self, session) -> None:
            _ = session

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        def resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    training_store_updates: list[dict] = []

    async def _async_upsert_training_session(info: dict) -> None:
        training_store_updates.append(dict(info))

    monkeypatch.setattr(training_routes, "training_engine", StubTrainingEngine())
    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(training_store_module, "async_upsert_training_session", _async_upsert_training_session)
    monkeypatch.setattr(
        session_index_store_module,
        "add_training_run_to_session",
        lambda *args, **kwargs: None,
    )

    req = CreateModelFromStateRequest(
        session_id="s417-cmfs",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        state_path=str(checkpoint_dir),
        lora_config=LoRAConfig(rank=4, train_attn=True, train_mlp=True, train_unembed=True),
        load_optimizer=True,
    )
    asyncio.run(training_routes._do_create_model_from_state("req-417-cmfs", req, user_id="user-417"))

    assert len(training_store_updates) == 1
    payload = training_store_updates[0]
    assert payload["model_id"] == model_id
    assert payload["current_step"] == 17
    assert payload["learning_rate"] == 3e-4
    assert payload["actor_name"] == "megatron-actor-417"
    assert payload["lora_config"] == {
        "rank": 16,
        "seed": None,
        "train_unembed": False,
        "train_mlp": True,
        "train_attn": False,
    }


def test_issue_283_create_model_from_state_background_restores_openpi_training_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.backend.training_engine_router import TrainingEngineRouter
    from mint_server.models.types import CreateModelFromStateRequest
    import mint_server.backend.session_index_store as session_index_store_module
    import mint_server.backend.training_session_store as training_store_module

    checkpoint_dir = tmp_path / "openpi-training"
    (checkpoint_dir / "1" / "params").mkdir(parents=True)
    (checkpoint_dir / "1" / "train_state").mkdir(parents=True)
    (checkpoint_dir / "1" / "assets" / "physical-intelligence" / "libero").mkdir(parents=True)
    (checkpoint_dir / "1" / "params" / "_METADATA").write_text("", encoding="utf-8")
    (checkpoint_dir / "1" / "train_state" / "_METADATA").write_text("", encoding="utf-8")
    (checkpoint_dir / "1" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json").write_text(
        "{}",
        encoding="utf-8",
    )

    class StubSession:
        def __init__(self) -> None:
            self.model_id = "s283-openpi_0"
            self.current_step = 3
            self.learning_rate = 1e-4
            self.base_model = "openpi/pi0-fast-libero-low-mem-finetune"
            self.backend = "openpi_fast"
            self.lora_config = None
            self.created_at = "2026-03-28T00:00:00Z"
            self.last_activity = 0.0

    class StubTrainingManager:
        def get_session(self, model_id: str):
            return None

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

        def delete_session(self, model_id: str) -> None:
            return None

        def mark_persisted(self, model_id: str) -> None:
            _ = model_id

        def create_session(self, **kwargs):
            return StubSession()

    class _RecordingEngine:
        def __init__(self, label: str) -> None:
            self.label = label
            self.calls: list[tuple[str, dict]] = []

        async def create_training_session(self, session) -> None:
            self.calls.append(("create_training_session", {"base_model": session.base_model}))

        async def load_weights(self, session, load_path: str, load_optimizer: bool = True) -> None:
            self.calls.append(
                (
                    "load_weights",
                    {
                        "base_model": session.base_model,
                        "load_path": load_path,
                        "load_optimizer": load_optimizer,
                    },
                )
            )

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        def resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    text_engine = _RecordingEngine("text")
    openpi_fast_engine = _RecordingEngine("openpi-fast")
    router = TrainingEngineRouter(text_engine=text_engine, openpi_fast_engine=openpi_fast_engine)
    stub_task_futures = StubTaskFutureService()
    training_store_updates: list[dict] = []

    async def _async_upsert_training_session(info: dict) -> None:
        training_store_updates.append(dict(info))

    monkeypatch.setattr(training_routes, "training_engine", router)
    monkeypatch.setattr(training_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(training_routes, "task_futures", stub_task_futures)
    monkeypatch.setattr(training_store_module, "async_upsert_training_session", _async_upsert_training_session)
    monkeypatch.setattr(
        session_index_store_module,
        "add_training_run_to_session",
        lambda *args, **kwargs: None,
    )

    req = CreateModelFromStateRequest(
        session_id="s283-openpi",
        model_seq_id=0,
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
        state_path=str(checkpoint_dir),
        load_optimizer=True,
    )

    asyncio.run(training_routes._do_create_model_from_state("req-283-openpi", req, user_id=None))

    assert openpi_fast_engine.calls == [
        ("create_training_session", {"base_model": "openpi/pi0-fast-libero-low-mem-finetune"}),
        (
            "load_weights",
            {
                "base_model": "openpi/pi0-fast-libero-low-mem-finetune",
                "load_path": str(checkpoint_dir),
                "load_optimizer": True,
            },
        ),
    ]
    assert text_engine.calls == []
    assert training_store_updates[0]["lora_config"] is None
    assert stub_task_futures.resolved == [
        (
            "req-283-openpi",
            {
                "request_id": "req-283-openpi",
                "model_id": "s283-openpi_0",
                "type": "create_model_from_state",
            },
        )
    ]


def test_issue_283_load_state_route_queues_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.backend.model_work_scheduler as scheduler_module

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

    class StubTaskFutureService:
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

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", object())
    monkeypatch.setattr(weights_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"mint://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    queued_payload = _queued_payload(stub_queue.calls[0])
    expected_path = weights_routes._resolve_mint_path(
        f"mint://{run_id}/weights/{ckpt_name}",
        user_id=None,
        is_admin=False,
    )
    assert queued_payload["path"] == expected_path


def test_issue_283_load_state_background_uses_resolved_path(tmp_path: Path, monkeypatch) -> None:
    from mint_server.routes import weights as weights_routes
    from mint_server.models.types import LoadStateRequest
    import mint_server.backend.training_session_store as training_store_module

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
        session_id = "session-283"
        model_seq_id = 0
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        backend = "megatron"
        lora_config = None
        rollout_correction_config = None
        user_metadata = {}
        user_id = None
        learning_rate = 1e-4
        current_step = 0
        metadata_version = 2
        materialization_state = "ready"
        created_at = "2026-03-13T00:00:00Z"
        last_activity = 1.0
        tokenizer_info = None
        tokenizer_identity = None
        tokenizer_source_path = None
        actor_name = None
        namespace = None

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

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    stub_engine = StubTrainingEngine()
    stub_task_futures = StubTaskFutureService()
    training_store_updates: list[dict] = []

    async def _async_upsert_training_session(info: dict) -> None:
        training_store_updates.append(dict(info))

    monkeypatch.setattr(weights_routes, "training_engine", stub_engine)
    monkeypatch.setattr(weights_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(weights_routes, "task_futures", stub_task_futures)
    monkeypatch.setattr(
        training_store_module,
        "async_upsert_training_session",
        _async_upsert_training_session,
    )

    req = LoadStateRequest(model_id="model-283", path=str(checkpoint_dir), optimizer=True)

    asyncio.run(weights_routes._do_load_state("req-283-load", req, user_id=None))

    assert stub_engine.load_calls == [
        {"load_path": str(checkpoint_dir), "load_optimizer": True}
    ]
    assert stub_task_futures.resolved == [
        ("req-283-load", {"path": str(checkpoint_dir), "type": "load_weights"})
    ]
    assert training_store_updates[0]["model_id"] == "model-283"
    assert training_store_updates[0]["session_id"] == "session-283"
    assert training_store_updates[0]["metadata_version"] == 3


def test_issue_417_load_state_persists_loaded_lora_config(tmp_path: Path, monkeypatch) -> None:
    from mint_server.routes import weights as weights_routes
    from mint_server.models.types import LoadStateRequest, LoRAConfig
    import mint_server.backend.training_session_store as training_store_module

    checkpoint_dir = tmp_path / "issue-417-load"
    checkpoint_dir.mkdir()

    class StubSession:
        model_id = "model-417"
        session_id = "session-417"
        model_seq_id = 0
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        backend = "megatron"
        lora_config = LoRAConfig(rank=4, train_attn=True, train_mlp=True, train_unembed=True)
        rollout_correction_config = None
        user_metadata = {"created": "before-load"}
        user_id = "original-user"
        learning_rate = 1e-4
        current_step = 0
        metadata_version = 2
        materialization_state = "ready"
        created_at = "2026-03-13T00:00:00Z"
        last_activity = 1.0
        tokenizer_info = {"source": "create"}
        tokenizer_identity = "tok-identity"
        tokenizer_source_path = "/models/tokenizer"
        actor_name = None
        namespace = None

    session = StubSession()

    class StubTrainingManager:
        def __init__(self) -> None:
            self.persisted: list[str] = []

        def get_session(self, model_id: str):
            assert model_id == "model-417"
            return session

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

        def mark_persisted(self, model_id: str) -> None:
            self.persisted.append(model_id)

    class StubTrainingEngine:
        def __init__(self) -> None:
            self._model_actor_supervisor_actor_names = {"model-417": "megatron-actor-417"}

        async def load_weights(self, session, load_path: str, load_optimizer: bool) -> None:
            assert load_path == str(checkpoint_dir)
            assert load_optimizer is False
            session.current_step = 12
            session.learning_rate = 2e-4
            session.lora_config = LoRAConfig(
                rank=16,
                train_attn=False,
                train_mlp=True,
                train_unembed=False,
            )

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            raise AssertionError(f"unexpected fail({request_id}): {error}")

    training_manager = StubTrainingManager()
    task_futures = StubTaskFutureService()
    training_store_updates: list[dict] = []

    async def _async_upsert_training_session(info: dict) -> None:
        training_store_updates.append(dict(info))

    monkeypatch.setattr(weights_routes, "training_engine", StubTrainingEngine())
    monkeypatch.setattr(weights_routes, "training_manager", training_manager)
    monkeypatch.setattr(weights_routes, "task_futures", task_futures)
    monkeypatch.setattr(training_store_module, "async_upsert_training_session", _async_upsert_training_session)

    req = LoadStateRequest(model_id="model-417", path=str(checkpoint_dir), optimizer=False)
    asyncio.run(weights_routes._do_load_state("req-417-load", req, user_id="admin-user"))

    assert task_futures.resolved == [
        ("req-417-load", {"path": str(checkpoint_dir), "type": "load_weights"})
    ]
    assert training_manager.persisted == ["model-417"]
    assert training_store_updates == [
        {
            "model_id": "model-417",
            "session_id": "session-417",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "lora_config": {
                "rank": 16,
                "seed": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
            "rollout_correction_config": None,
            "user_metadata": {"created": "before-load"},
            "learning_rate": 2e-4,
            "current_step": 12,
            "backend": "megatron",
            "actor_name": "megatron-actor-417",
            "namespace": "mint",
            "user_id": "original-user",
            "created_at": "2026-03-13T00:00:00Z",
            "last_activity": 1.0,
            "metadata_version": 3,
            "materialization_state": "ready",
            "tokenizer_info": {"source": "create"},
            "tokenizer_identity": "tok-identity",
            "tokenizer_source_path": "/models/tokenizer",
        }
    ]


def test_issue_417_load_state_reports_success_when_metadata_persist_fails_after_actor_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from mint_server.routes import weights as weights_routes
    from mint_server.models.types import LoadStateRequest, LoRAConfig
    import mint_server.backend.training_session_store as training_store_module

    checkpoint_dir = tmp_path / "issue-417-load-persist-fail"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_type": "training",
                "optimizer_present": True,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    class StubSession:
        model_id = "model-417-persist-fail"
        session_id = "session-417-persist-fail"
        model_seq_id = 0
        base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        backend = "megatron"
        lora_config = LoRAConfig(rank=4, train_attn=True, train_mlp=True, train_unembed=True)
        rollout_correction_config = None
        user_metadata = {}
        user_id = "original-user"
        learning_rate = 1e-4
        current_step = 0
        metadata_version = 2
        materialization_state = "ready"
        created_at = "2026-03-13T00:00:00Z"
        last_activity = 1.0
        tokenizer_info = None
        tokenizer_identity = None
        tokenizer_source_path = None
        actor_name = None
        namespace = None

    session = StubSession()

    class StubTrainingManager:
        def get_session(self, model_id: str):
            assert model_id == "model-417-persist-fail"
            return session

        def mark_inflight(self, model_id: str, delta: int) -> None:
            _ = (model_id, delta)

        def mark_persisted(self, model_id: str) -> None:
            raise AssertionError(f"mark_persisted must not run after failed upsert: {model_id}")

    class StubTrainingEngine:
        _model_actor_supervisor_actor_names = {"model-417-persist-fail": "megatron-actor-417"}

        async def load_weights(self, session, load_path: str, load_optimizer: bool) -> None:
            assert load_path == str(checkpoint_dir)
            assert load_optimizer is True
            session.current_step = 77
            session.learning_rate = 9e-5
            session.lora_config = LoRAConfig(rank=16, train_attn=False, train_mlp=True, train_unembed=False)

    class StubTaskFutureService:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []
            self.failed: list[tuple[str, str]] = []

        async def async_resolve(self, request_id: str, payload: dict) -> None:
            self.resolved.append((request_id, payload))

        async def async_fail(self, request_id: str, error: str) -> None:
            self.failed.append((request_id, error))

    async def _async_upsert_training_session(info: dict) -> None:
        _ = info
        raise RuntimeError("detached store unavailable")

    task_futures = StubTaskFutureService()
    monkeypatch.setattr(weights_routes, "training_engine", StubTrainingEngine())
    monkeypatch.setattr(weights_routes, "training_manager", StubTrainingManager())
    monkeypatch.setattr(weights_routes, "task_futures", task_futures)
    monkeypatch.setattr(training_store_module, "async_upsert_training_session", _async_upsert_training_session)

    req = LoadStateRequest(model_id="model-417-persist-fail", path=str(checkpoint_dir), optimizer=True)
    asyncio.run(weights_routes._do_load_state("req-417-persist-fail", req, user_id="user-417"))

    assert task_futures.failed == []
    assert task_futures.resolved == [
        (
            "req-417-persist-fail",
            {
                "path": str(checkpoint_dir),
                "type": "load_weights",
                "metadata_persisted": False,
                "metadata_persist_error": "RuntimeError: detached store unavailable",
            },
        )
    ]
    assert session.current_step == 77
    assert session.learning_rate == pytest.approx(9e-5)
    assert session.lora_config.rank == 16
    assert session.lora_config.train_attn is False


def test_issue_283_save_routes_use_detached_training_info_without_route_runtime(monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    import mint_server.backend.model_work_scheduler as scheduler_module

    class StubTaskFutureService:
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

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    for route_path, op in (("save_weights", "weights.save_weights"), ("save_state", "weights.save_state")):
        resp = client.post(
            f"/api/v1/{route_path}",
            json={"model_id": "model-283", "path": f"{route_path}-283"},
        )
        assert resp.status_code == 200, resp.text

    assert [call["op"] for call in stub_queue.calls] == ["weights.save_weights", "weights.save_state"]
    assert [_queued_payload(call)["model_id"] for call in stub_queue.calls] == ["model-283", "model-283"]
    assert all(call["extra"]["execution_serial_key"] == "training_session:model-283" for call in stub_queue.calls)


def test_issue_283_load_state_route_uses_detached_training_info_without_route_runtime(tmp_path: Path, monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.backend.model_work_scheduler as scheduler_module

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

    class StubTaskFutureService:
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

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"mint://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    queued_payload = _queued_payload(stub_queue.calls[0])
    expected_path = weights_routes._resolve_mint_path(
        f"mint://{run_id}/weights/{ckpt_name}",
        user_id=None,
        is_admin=False,
    )
    assert queued_payload["path"] == expected_path


def test_issue_283_save_routes_restore_inflight_protection(monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    import mint_server.backend.model_work_scheduler as scheduler_module

    class StubTaskFutureService:
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
    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
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
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.backend.model_work_scheduler as scheduler_module

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

    class StubTaskFutureService:
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
    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"mint://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert manager.calls == [("model-283", 1)]
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "weights.load_state"


@pytest.mark.parametrize("route_path", ["save_weights", "save_state", "load_state"])
def test_issue_283_weights_routes_propagate_detached_store_503(monkeypatch, route_path: str) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    import mint_server.backend.training_session_store as training_store_module
    import mint_server.gateway as gateway_module

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

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    payload = {"model_id": "model-283", "path": "ckpt-283"}
    if route_path == "load_state":
        payload["optimizer"] = False

    resp = client.post(f"/api/v1/{route_path}", json=payload)

    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "Training session store unavailable"
def test_issue_283_save_routes_refresh_detached_enqueue_protection(monkeypatch) -> None:
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    import mint_server.backend.model_work_scheduler as scheduler_module

    class StubTaskFutureService:
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

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(weights_routes, "_protect_training_session_enqueue_window", _protect_training_session_enqueue_window)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
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
    from mint_server.routes import training as training_routes
    from mint_server.routes import weights as weights_routes
    from mint_server import checkpoints as checkpoints_module
    import mint_server.backend.model_work_scheduler as scheduler_module

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

    class StubTaskFutureService:
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

    stub_queue = StubModelWorkScheduler()

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", stub_queue)
    monkeypatch.setattr(weights_routes, "task_futures", StubTaskFutureService())
    monkeypatch.setattr(weights_routes, "training_engine", None)
    monkeypatch.setattr(weights_routes, "training_manager", None)
    monkeypatch.setattr(weights_routes, "_protect_training_session_enqueue_window", _protect_training_session_enqueue_window)
    monkeypatch.setattr(training_routes, "_get_training_route_session_info", _get_training_route_session_info)

    app = _make_write_app()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/load_state",
        json={
            "model_id": "model-283",
            "path": f"mint://{run_id}/weights/{ckpt_name}",
            "optimizer": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(stub_queue.calls) == 1
    assert stub_queue.calls[0]["op"] == "weights.load_state"
    assert [entry["session_id"] for entry in protected] == ["sess-283-load"]
