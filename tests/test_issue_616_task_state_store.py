from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from mint_server.backend.contracts.control_plane_contracts import ConflictReason
from mint_server.backend.stores.task_payload_store import TaskPayloadStore
from mint_server.backend.stores.task_state_store import (
    TaskStateConflictError,
    TaskFutureService,
    TaskStateStore,
    TaskStateStoreClient,
    _TaskStateStoreActor,
    _task_state_actor_identity_mismatches,
    billing_observations_from_input,
    build_billing_observation,
)


def _create_task(store: TaskStateStore, request_id: str = "req-1") -> None:
    created = store.create_task(
        request_id=request_id,
        op="sampling.asample",
        domain_key="vllm:Qwen/Qwen3-4B-Instruct-2507",
        request_json=b'{"prompt": "hi"}',
        payload_hash="hash-1",
        metadata={"queue_kind": "model_work_scheduler"},
        now=100.0,
    )
    assert created.ok is True
    assert created.created is True


def test_duplicate_create_task_preserves_model_work_append_owner_marker() -> None:
    store = TaskStateStore.in_memory()
    try:
        first = store.create_task(
            request_id="append-owner",
            op="sampling.asample",
            domain_key="vllm:model-a",
            request_json=b'{"prompt":"first"}',
            metadata={
                "model_work_scheduler_append_attempt_id": "attempt-a",
                "stage": "first",
            },
        )
        second = store.create_task(
            request_id="append-owner",
            op="sampling.asample",
            domain_key="vllm:model-a",
            request_json=b'{"prompt":"second"}',
            metadata={
                "model_work_scheduler_append_attempt_id": "attempt-b",
                "stage": "duplicate",
            },
        )

        assert first.created is True
        assert second.created is False
        assert second.record["request_json"] == b'{"prompt":"first"}'
        assert second.record["metadata"]["model_work_scheduler_append_attempt_id"] == "attempt-a"
        assert second.record["metadata"]["stage"] == "first"
    finally:
        store.close()


def test_task_state_store_client_async_ensure_ready_can_create_actor(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as module
    import ray

    calls: dict[str, object] = {}
    ping = {
        "ok": True,
        "actor_name": "mint_task_state_store",
        "namespace": "mint",
        "db_path": "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3",
        "hot_kv_db_path": "/vePFS-Mindverse/share/mint/dev/data/task-hot-kv/task_hot.rocksdb",
        "future_db_path": "/vePFS-Mindverse/share/mint/dev/data/future-state/futures.rocksdb",
        "payload_root_dir": "/vePFS-Mindverse/share/mint/dev/data/task-state/payloads",
    }

    class _PingRemote:
        def remote(self) -> dict[str, object]:
            return ping

    class _Actor:
        ping = _PingRemote()

    async def _fake_async_get_ray_ref(ref, *, timeout_s=10.0):
        calls["timeout_s"] = timeout_s
        return ref

    def _fake_create_ray_actor_handle():
        calls["created_handle"] = True
        return _Actor()

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_actor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(module, "_create_ray_actor_handle", _fake_create_ray_actor_handle)
    monkeypatch.setattr(module, "async_get_ray_ref", _fake_async_get_ray_ref)

    client = TaskStateStoreClient()
    out = asyncio.run(client.async_ensure_ready(timeout_s=7.0, create_if_missing=True))

    assert out == ping
    assert calls == {"created_handle": True, "timeout_s": 7.0}


def test_task_state_store_ray_actor_ping_uses_health_concurrency_group(monkeypatch) -> None:
    import mint_server.backend.stores.task_state_store as module
    import ray

    captured: dict[str, object] = {}

    class _OptionsProxy:
        def __init__(self, cls):
            self._cls = cls

        def remote(self, db_path: str):
            captured["db_path"] = db_path
            actor = self._cls(db_path)
            ping = actor.ping

            class _RemoteMethod:
                def remote(self):
                    return ping()

            actor.ping = _RemoteMethod()
            return actor

    class _RemoteClass:
        def __init__(self, cls):
            self._cls = cls

        def options(self, **options):
            captured["options"] = options
            return _OptionsProxy(self._cls)

    def _fake_remote(**remote_kwargs):
        captured["remote_kwargs"] = remote_kwargs

        def _decorator(cls):
            captured["actor_cls"] = cls
            return _RemoteClass(cls)

        return _decorator

    def _fake_method(**method_kwargs):
        captured["method_kwargs"] = method_kwargs

        def _decorator(fn):
            captured["method_name"] = fn.__name__
            return fn

        return _decorator

    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(ray, "method", _fake_method)
    def _fake_actor_runtime_env(**kwargs):
        captured["runtime_env_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(module, "actor_runtime_env", _fake_actor_runtime_env)
    monkeypatch.setattr(module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "sync_get_ray_ref", lambda ref, *, timeout_s=None: ref)
    monkeypatch.setattr(module, "_task_state_store_db_path", lambda: ":memory:")

    module._create_ray_actor(require_ready=True)

    assert captured["remote_kwargs"]["concurrency_groups"] == {"health": 8}
    assert captured["runtime_env_kwargs"]["include_ray_attach_hints"] is False
    assert captured["method_kwargs"] == {"concurrency_group": "health"}
    assert captured["method_name"] == "ping"


def test_training_session_inflight_is_durable_metadata(tmp_path: Path) -> None:
    store = TaskStateStore(str(tmp_path / "task-state" / "task-state-training-inflight.sqlite3"))
    try:
        store.upsert_training_session(
            model_id="model-inflight",
            info={
                "model_id": "model-inflight",
                "session_id": "session-inflight",
                "base_model": "Qwen/Qwen3-0.6B",
                "metadata_version": 1,
                "last_activity": 10.0,
            },
        )

        assert store.mark_training_session_inflight(model_id="model-inflight", delta=1) == 1
        info = store.get_training_session(model_id="model-inflight")
        assert info is not None
        assert info["inflight_ops"] == 1
        assert info["last_activity"] >= 10.0

        assert store.mark_training_session_inflight(model_id="model-inflight", delta=-3) == 0
        info = store.get_training_session(model_id="model-inflight")
        assert info is not None
        assert info["inflight_ops"] == 0
        assert store.mark_training_session_inflight(model_id="missing", delta=1) is None
    finally:
        store.close()
