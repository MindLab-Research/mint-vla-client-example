import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tinker_server.backend import sampling_cleanup_executor as cleanup_executor_module
from tinker_server.backend import sampling_session_store as sampling_store_module
from tinker_server.backend import session_manager as session_manager_module
from tinker_server.routes import service as service_route

future_store_module = importlib.import_module("tinker_server.backend.future_store")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_369_detached_sampling_cleanup_removes_stale_session(monkeypatch) -> None:
    deleted = []
    cleaned_indices = []
    failed_sampling = []
    unloaded = []

    async def _async_list_sampling_sessions():
        return [
            {
                "session_id": "sess-stale",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "adapter_path": "/tmp/_ephemeral_adapter_stale",
                "uses_base_model": False,
                "lora_loaded": True,
                "lora_int_id": 7,
                "last_activity": 0.0,
                "inflight_requests": 0,
            },
            {
                "session_id": "sess-live",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "adapter_path": "/tmp/_ephemeral_adapter_live",
                "uses_base_model": False,
                "lora_loaded": True,
                "lora_int_id": 8,
                "last_activity": 1e12,
                "inflight_requests": 0,
            },
        ]

    monkeypatch.setattr(sampling_store_module, "async_list_sampling_sessions", _async_list_sampling_sessions)
    monkeypatch.setattr(sampling_store_module, "delete_sampling_session", lambda session_id: deleted.append(session_id))
    async def _async_fail_sampling_requests_for_session(session_id: str, error: str) -> list[str]:
        failed_sampling.append((session_id, error))
        return ["req-sample"]

    monkeypatch.setattr(
        future_store_module.future_store,
        "async_fail_sampling_requests_for_session",
        _async_fail_sampling_requests_for_session,
    )
    monkeypatch.setattr(cleanup_executor_module, "_cleanup_sampler_indices", lambda sampler_id: cleaned_indices.append(sampler_id))

    async def _remove_loaded_lora_if_last_reference(**kwargs):
        unloaded.append(kwargs)

    monkeypatch.setattr(cleanup_executor_module, "_remove_loaded_lora_if_last_reference", _remove_loaded_lora_if_last_reference)
    monkeypatch.setattr(cleanup_executor_module.os.path, "isdir", lambda path: path == "/tmp/_ephemeral_adapter_stale")
    monkeypatch.setattr(cleanup_executor_module.os.path, "basename", lambda path: path.split("/")[-1])
    monkeypatch.setattr(cleanup_executor_module.shutil, "rmtree", lambda path: cleaned_indices.append(f"rmtree:{path}"))
    monkeypatch.setattr(cleanup_executor_module.time, "time", lambda: 5000.0)

    cleaned = await cleanup_executor_module.cleanup_stale_sampling_sessions_once_impl(stale_after_s=60.0)

    assert cleaned == ["sess-stale"]
    assert deleted == ["sess-stale"]
    assert cleaned_indices == ["sess-stale", "rmtree:/tmp/_ephemeral_adapter_stale"]
    assert failed_sampling == [
        ("sess-stale", "Sampling session terminated due to sampling inactivity (> 60.0s)")
    ]
    assert unloaded == [{"base_model": "Qwen/Qwen3-4B-Instruct-2507", "lora_int_id": 7}]


@pytest.mark.anyio
async def test_issue_369_detached_sampling_cleanup_keeps_shared_adapter_loaded(monkeypatch) -> None:
    deleted = []
    unloaded = []

    async def _async_list_sampling_sessions():
        return [
            {
                "session_id": "sess-a",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "adapter_path": "/tmp/shared_adapter",
                "uses_base_model": False,
                "lora_loaded": True,
                "lora_int_id": 7,
                "last_activity": 0.0,
                "inflight_requests": 0,
            },
            {
                "session_id": "sess-b",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "adapter_path": "/tmp/shared_adapter",
                "uses_base_model": False,
                "lora_loaded": True,
                "lora_int_id": 7,
                "last_activity": 1e12,
                "inflight_requests": 0,
            },
        ]

    monkeypatch.setattr(sampling_store_module, "async_list_sampling_sessions", _async_list_sampling_sessions)
    monkeypatch.setattr(sampling_store_module, "delete_sampling_session", lambda session_id: deleted.append(session_id))
    async def _async_fail_sampling_requests_for_session(session_id: str, error: str) -> list[str]:
        return []

    monkeypatch.setattr(
        future_store_module.future_store,
        "async_fail_sampling_requests_for_session",
        _async_fail_sampling_requests_for_session,
    )
    monkeypatch.setattr(cleanup_executor_module, "_cleanup_sampler_indices", lambda sampler_id: None)

    async def _remove_loaded_lora_if_last_reference(**kwargs):
        unloaded.append(kwargs)

    monkeypatch.setattr(cleanup_executor_module, "_remove_loaded_lora_if_last_reference", _remove_loaded_lora_if_last_reference)
    monkeypatch.setattr(cleanup_executor_module.time, "time", lambda: 5000.0)

    cleaned = await cleanup_executor_module.cleanup_stale_sampling_sessions_once_impl(stale_after_s=60.0)

    assert cleaned == ["sess-a"]
    assert deleted == ["sess-a"]
    assert unloaded == []


@pytest.mark.anyio
async def test_issue_369_sampling_heartbeat_sampling_store_failure_is_503(monkeypatch) -> None:
    touched = []

    class _StubSessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            touched.append((session_id, delta))

    async def _async_update(session_id: str):
        touched.append(("heartbeat", session_id))

    async def _async_set_last_activity(_session_id: str, _last_activity: float):
        raise RuntimeError("store down")

    service_route.session_manager = _StubSessionManager()
    monkeypatch.setattr("tinker_server.routes.service.session_heartbeat_store.async_update", _async_update)
    monkeypatch.setattr(sampling_store_module, "async_set_sampling_session_last_activity", _async_set_last_activity)

    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None))
    with pytest.raises(HTTPException, match="Sampling session store unavailable"):
        await service_route.session_heartbeat(SimpleNamespace(session_id="sess-heartbeat"), http_request)

    assert touched == [("heartbeat", "sess-heartbeat")]


@pytest.mark.anyio
async def test_issue_369_sampling_heartbeat_updates_detached_last_activity(monkeypatch) -> None:
    touched = []
    local_refresh = []

    class _StubSessionManager:
        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            local_refresh.append((session_id, delta))

    service_route.session_manager = _StubSessionManager()
    async def _async_update(session_id: str):
        touched.append(("heartbeat", session_id))

    async def _async_set_last_activity(session_id: str, last_activity: float):
        touched.append(("sampling_last_activity", session_id, isinstance(last_activity, float)))
        return last_activity

    monkeypatch.setattr("tinker_server.routes.service.session_heartbeat_store.async_update", _async_update)
    monkeypatch.setattr(sampling_store_module, "async_set_sampling_session_last_activity", _async_set_last_activity)

    async def _noop_touch_child_sampler_sessions(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service_route, "_touch_child_sampler_sessions", _noop_touch_child_sampler_sessions)

    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None))
    await service_route.session_heartbeat(SimpleNamespace(session_id="sess-heartbeat"), http_request)

    assert touched[0] == ("heartbeat", "sess-heartbeat")
    assert touched[1][0] == "sampling_last_activity"
    assert touched[1][1] == "sess-heartbeat"
    assert touched[1][2] is True
    assert local_refresh == [("sess-heartbeat", 0)]


def test_issue_369_session_manager_base_model_getter_restores_from_detached_store(monkeypatch) -> None:
    manager = session_manager_module.SessionManager()
    monkeypatch.setattr(
        "tinker_server.backend.sampling_session_store.get_sampling_session_info",
        lambda session_id: {
            "session_id": session_id,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "uses_base_model": True,
            "metadata_version": 3,
            "last_activity": 1.0,
            "inflight_requests": 0,
        },
    )

    assert manager.get_session_base_model("sess-restore") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    snapshot = manager.get_sampling_session_snapshot("sess-restore")
    assert snapshot is not None
    assert snapshot.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
