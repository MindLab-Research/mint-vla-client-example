from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from concurrent.futures import Future

import anyio
import pytest

from tinker_server.backend.future_store import FutureStatus
from tinker_server.models.types import (
    AdamParams,
    ForwardBackwardInput,
    ForwardBackwardRequest,
    ForwardRequest,
    FutureRetrieveRequest,
    GetInfoRequest,
    ModelInput,
    OptimStepRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SamplingParams,
    TrainStepRequest,
)
from tinker_server.routes import futures as futures_route
from tinker_server.routes import internal as internal_route
from tinker_server.routes import sampling as sampling_route
from tinker_server.routes import service as service_route
from tinker_server.routes import training as training_route


@pytest.fixture(autouse=True)
def _reset_retrieve_future_caches(monkeypatch):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())


def _request_stub(user_id: str = "admin"):
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": user_id}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def _install_minimal_ray_module(monkeypatch):
    ray_mod = types.ModuleType("ray")
    ray_mod.actor = SimpleNamespace(ActorHandle=object)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    return ray_mod


def _install_namespace_module(monkeypatch, module_name: str, namespace: str):
    module = types.ModuleType(module_name)
    module.PERSISTENT_NAMESPACE = namespace
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _install_resource_pool_module(monkeypatch, *, pool, actor_types=None, stale_error=RuntimeError):
    module_name = "tinker_server.backend.resource_pool"
    module = types.ModuleType(module_name)
    module.ActorType = actor_types or SimpleNamespace(VLLM="vllm", MEGATRON="megatron", DENSE="dense")
    module.ResourcePoolStaleError = stale_error
    module.get_resource_pool = lambda: pool
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


class _GatewayResponse:
    def __init__(self, payload: dict, *, status_code: int = 200, text: str = ""):
        self._payload = dict(payload)
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return dict(self._payload)


def _patch_training_route_remote_fallback(monkeypatch) -> None:
    async def _restore_training_session(_model_id: str):
        return None

    async def _get_training_route_session_info(_model_id: str):
        return None

    monkeypatch.setattr(training_route, "training_manager", SimpleNamespace(get_session=lambda _model_id: None))
    monkeypatch.setattr(training_route, "training_engine", object())
    monkeypatch.setattr(training_route, "_restore_training_session", _restore_training_session)
    monkeypatch.setattr(training_route, "_get_training_route_session_info", _get_training_route_session_info)
    monkeypatch.setattr(training_route, "can_access_model", lambda _base_model, _user_data: True)


def _install_gateway_forward_stubs(
    monkeypatch,
    *,
    response_payload: dict,
):
    import tinker_server.gateway as gw

    calls: dict[str, object] = {}

    async def _fake_async_remote_training_model(model_id: str):
        calls["async_remote_training_model"] = model_id
        return ("upstream-a", "Qwen/Qwen3-0.6B")

    def _unexpected_sync_remote_training_model(*_args, **_kwargs):
        raise AssertionError("sync remote_training_model should not be used on async request path")

    def _fake_upstream_for_alias(alias: str):
        calls["upstream_alias"] = alias
        return SimpleNamespace(alias=alias)

    async def _fake_forward_json(*, upstream, method, path, incoming_headers, json_body, timeout_s, **_kwargs):
        calls["forward_json"] = {
            "upstream_alias": upstream.alias,
            "method": method,
            "path": path,
            "incoming_headers": dict(incoming_headers),
            "json_body": dict(json_body),
            "timeout_s": timeout_s,
        }
        return _GatewayResponse(response_payload)

    def _fake_encode_request_id(*, upstream_alias: str, upstream_request_id: str) -> str:
        calls["encode_request_id"] = (upstream_alias, upstream_request_id)
        return f"{upstream_alias}:{upstream_request_id}"

    def _fake_register_pending_save_weights_for_sampler_future(
        *,
        upstream_alias: str,
        upstream_request_id: str,
        base_model: str,
    ) -> None:
        calls["register_pending_save_weights_for_sampler_future"] = (
            upstream_alias,
            upstream_request_id,
            base_model,
        )

    monkeypatch.setattr(gw, "async_remote_training_model", _fake_async_remote_training_model)
    monkeypatch.setattr(gw, "remote_training_model", _unexpected_sync_remote_training_model, raising=False)
    monkeypatch.setattr(gw, "upstream_for_alias", _fake_upstream_for_alias)
    monkeypatch.setattr(gw, "forward_json", _fake_forward_json)
    monkeypatch.setattr(gw, "encode_request_id", _fake_encode_request_id)
    monkeypatch.setattr(
        gw,
        "register_pending_save_weights_for_sampler_future",
        _fake_register_pending_save_weights_for_sampler_future,
        raising=False,
    )
    return calls


class _AsyncOnlyPendingFutureStore:
    def __init__(self, meta: dict | None = None):
        self.calls: list[tuple[str, str]] = []
        self._meta = dict(meta or {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"})

    async def async_get_status(self, request_id: str) -> FutureStatus:
        self.calls.append(("async_get_status", request_id))
        return FutureStatus.PENDING

    async def async_get_meta(self, request_id: str):
        self.calls.append(("async_get_meta", request_id))
        return dict(self._meta)

    def get_status(self, request_id: str) -> FutureStatus:
        raise AssertionError("sync get_status should not be used on request path")

    def get_meta(self, request_id: str):
        raise AssertionError("sync get_meta should not be used on request path")


class _AsyncOnlyTerminalFutureStore:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.cleanup_calls: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        self.calls.append(("async_get_status", request_id))
        return FutureStatus.DONE

    async def async_get_meta(self, request_id: str):
        self.calls.append(("async_get_meta", request_id))
        return {"op": "sampling.asample"}

    async def async_get_result(self, request_id: str):
        self.calls.append(("async_get_result", request_id))
        return {"ok": request_id}

    async def async_get_error(self, request_id: str):
        self.calls.append(("async_get_error", request_id))
        return None

    async def async_cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)

    def get_status(self, request_id: str) -> FutureStatus:
        raise AssertionError("sync get_status should not be used on request path")

    def get_result(self, request_id: str):
        raise AssertionError("sync get_result should not be used on request path")

    def get_error(self, request_id: str):
        raise AssertionError("sync get_error should not be used on request path")


class _AsyncOnlyUnknownFutureStore:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        self.calls.append(("async_get_status", request_id))
        raise KeyError(request_id)

    async def async_debug_snapshot(self, *, timeout_s: float = 10.0):
        self.calls.append(("async_debug_snapshot", str(timeout_s)))
        return {"status": "debug"}

    def debug_snapshot(self):
        raise AssertionError("sync debug_snapshot should not be used on request path")


class _StubApiWorkQueue:
    async def describe_pending_request(self, request_id: str, op: str | None) -> dict:
        _ = request_id, op
        return {"found": True, "position": 0, "depth": 1, "ema_exec_s": 2.0}

    async def find_position(self, request_id: str) -> dict:
        return {"found": True, "position": 0, "depth": 1}

    async def get_eta_state(self, op: str | None) -> dict:
        return {"ema_exec_s": 2.0}

    async def stats(self, timeout_s: float = 10.0) -> dict:
        _ = timeout_s
        return {"depth": 0}

    async def rss_bytes(self, timeout_s: float = 10.0) -> int:
        _ = timeout_s
        return 123


class _AsyncOnlyResourcePool:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def async_touch(self, actor_name: str) -> bool:
        self.calls.append(("async_touch", actor_name))
        return True

    async def async_set_session(self, actor_name: str, session_id: str | None) -> None:
        self.calls.append(("async_set_session", session_id))

    def touch(self, actor_name: str) -> bool:
        raise AssertionError("sync touch should not be used on request path")

    def set_session(self, actor_name: str, session_id: str | None) -> None:
        raise AssertionError("sync set_session should not be used on request path")


class _AsyncOnlySamplingFutureStore:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.marked: list[str] = []

    async def async_ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        self.calls.append(("async_ensure_pending", request_id))
        return {"created": True, "meta": None}

    async def async_create_with_id(self, request_id: str) -> str:
        self.calls.append(("async_create_with_id", request_id))
        return request_id

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.calls.append(("async_mark_queued", request_id))
        self.marked.append(request_id)

    def mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)

    def cleanup(self, request_id: str) -> None:
        return None

    def forget(self, request_id: str) -> None:
        return None

    def ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        raise AssertionError("sync ensure_pending should not be used on request path")

    def create_with_id(self, request_id: str) -> str:
        raise AssertionError("sync create_with_id should not be used on request path")


class _AsyncOnlyTrainingFutureStore:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.marked: list[str] = []
        self.cleaned: list[str] = []

    async def async_create_with_id(self, request_id: str) -> str:
        self.calls.append(("async_create_with_id", request_id))
        return request_id

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.calls.append(("async_mark_queued", request_id))
        self.marked.append(request_id)

    def mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)

    def cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    def create_with_id(self, request_id: str) -> str:
        raise AssertionError("sync create_with_id should not be used on request path")


class _AsyncOnlyCapacityManager:
    def __init__(self):
        self.calls: list[tuple[str, int, int]] = []
        self.released: list[str] = []

    async def async_try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict:
        self.calls.append((request_id, int(queue_bytes), int(object_store_bytes)))
        return {"ok": True}

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)

    def release_all(self, request_id: str) -> None:
        self.released.append(request_id)

    def try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int) -> dict:
        raise AssertionError("sync try_reserve should not be used on request path")


class _RecordingQueue:
    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, **kwargs):
        self.calls.append(dict(kwargs))


class _RecordingTrainingManager:
    def __init__(self):
        self.sessions: dict[str, object] = {}
        self.create_calls: list[dict[str, object]] = []
        self.deleted: list[str] = []

    def get_session(self, model_id: str):
        return self.sessions.get(model_id)

    def create_session(self, **kwargs):
        session = SimpleNamespace(
            model_id=kwargs["model_id"],
            session_id=kwargs["session_id"],
            model_seq_id=kwargs["model_seq_id"],
            base_model=kwargs["base_model"],
            lora_config=kwargs["lora_config"],
            rollout_correction_config=kwargs["rollout_correction_config"],
            user_metadata=kwargs["user_metadata"],
            user_id=kwargs["user_id"],
            learning_rate=kwargs["learning_rate"],
            current_step=0,
            is_active=False,
            created_at="",
            backend="peft",
        )
        self.create_calls.append(dict(kwargs))
        self.sessions[kwargs["model_id"]] = session
        return session

    def delete_session(self, model_id: str) -> bool:
        self.deleted.append(model_id)
        return self.sessions.pop(model_id, None) is not None


class _SamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return object()


class _AsyncOnlyAdmissionCapacityManager:
    async def async_snapshot(self, timeout_s: float = 10.0):
        from tinker_server.backend.capacity_manager import CapacitySnapshot

        _ = timeout_s
        return CapacitySnapshot(
            queue_bytes_budget=1,
            queue_bytes_reserved=2,
            object_store_bytes_reserved=3,
            object_store_free_bytes=4,
            rejects_total=5,
            reserves_total=6,
        )

    async def async_rss_bytes(self, timeout_s: float = 10.0) -> int:
        _ = timeout_s
        return 111

    def snapshot(self, timeout_s: float = 10.0):
        raise AssertionError("sync snapshot should not be used on request path")

    def rss_bytes(self, timeout_s: float = 10.0):
        raise AssertionError("sync rss_bytes should not be used on request path")


class _AsyncOnlyAdmissionFutureStore:
    async def async_ensure_ready(self, timeout_s: float = 10.0):
        _ = timeout_s
        return {"pending": 0}

    async def async_rss_bytes(self, timeout_s: float = 10.0) -> int:
        _ = timeout_s
        return 222

    def ensure_ready(self, timeout_s: float = 10.0):
        raise AssertionError("sync ensure_ready should not be used on request path")

    def rss_bytes(self, timeout_s: float = 10.0):
        raise AssertionError("sync rss_bytes should not be used on request path")


def test_issue_360_retrieve_future_pending_uses_async_store_calls(monkeypatch):
    store = _AsyncOnlyPendingFutureStore(
        {"queue_state": "queued", "stage": "queued", "op": "sampling.asample", "actor_name": "actor-a", "model_id": "model-a"}
    )
    pool = _AsyncOnlyResourcePool()
    monkeypatch.setattr(futures_route, "future_store", store)
    import tinker_server.backend.api_work_queue as wq
    import tinker_server.config as config_module
    import tinker_server.backend.resource_pool as rp

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue())
    monkeypatch.setattr(rp, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_pending_async")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), response)

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert ("async_get_status", "rid_pending_async") in store.calls
    assert ("async_get_meta", "rid_pending_async") in store.calls
    assert pool.calls == [("async_touch", "actor-a"), ("async_set_session", "model-a")]


def test_issue_360_retrieve_future_terminal_uses_async_result(monkeypatch):
    store = _AsyncOnlyTerminalFutureStore()
    monkeypatch.setattr(futures_route, "future_store", store)

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)

    body = FutureRetrieveRequest(request_id="rid_done_async")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), response)

    assert payload == {"ok": "rid_done_async"}
    assert ("async_get_status", "rid_done_async") in store.calls
    assert ("async_get_result", "rid_done_async") in store.calls
    assert store.cleanup_calls == ["rid_done_async"]


def test_issue_360_retrieve_future_unknown_admin_uses_async_debug_snapshot(monkeypatch):
    from fastapi import HTTPException

    store = _AsyncOnlyUnknownFutureStore()
    monkeypatch.setattr(futures_route, "future_store", store)

    body = FutureRetrieveRequest(request_id="rid_unknown_async")
    response = _response_stub()

    with pytest.raises(HTTPException) as exc:
        anyio.run(futures_route.retrieve_future, body, _request_stub(), response)

    assert exc.value.status_code == 404
    assert exc.value.detail["future_store"] == {"status": "debug"}
    assert ("async_get_status", "rid_unknown_async") in store.calls
    assert any(name == "async_debug_snapshot" for name, _value in store.calls)


def test_issue_360_asample_admission_uses_async_capacity_and_future(monkeypatch):
    fs = _AsyncOnlySamplingFutureStore()
    cap = _AsyncOnlyCapacityManager()
    q = _RecordingQueue()

    monkeypatch.setattr(sampling_route, "session_manager", _SamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", fs)

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(cm, "capacity_manager", cap)
    monkeypatch.setattr(awq, "api_work_queue", q)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="sess_async",
        seq_id=1,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    out = anyio.run(sampling_route.asample, req, _request_stub("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(cap.calls) == 1
    assert any(name == "async_ensure_pending" for name, _rid in fs.calls)
    assert len(q.calls) == 1


def test_issue_360_internal_admission_stats_uses_async_store_calls(monkeypatch):
    import importlib

    wq = importlib.import_module("tinker_server.backend.api_work_queue")
    cm = importlib.import_module("tinker_server.backend.capacity_manager")
    fs = importlib.import_module("tinker_server.backend.future_store")
    _install_minimal_ray_module(monkeypatch)
    rp = importlib.import_module("tinker_server.backend.resource_pool")

    monkeypatch.setattr(cm, "capacity_manager", _AsyncOnlyAdmissionCapacityManager())
    monkeypatch.setattr(fs, "future_store", _AsyncOnlyAdmissionFutureStore())
    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue())
    monkeypatch.setattr(
        rp,
        "get_resource_pool",
        lambda: SimpleNamespace(rss_snapshot=lambda timeout_s=10.0: {"rss_bytes": 333}),
    )

    payload = anyio.run(internal_route.admission_stats)

    assert payload["capacity"]["queue_bytes_budget"] == 1
    assert payload["future_store"]["pending"] == 0
    assert payload["actors"]["capacity_manager"]["rss_bytes"] == 111
    assert payload["actors"]["api_work_queue"]["rss_bytes"] == 123
    assert payload["actors"]["future_store"]["rss_bytes"] == 222


@pytest.mark.anyio
async def test_issue_360_api_work_queue_request_helpers_use_async_actor(monkeypatch):
    import importlib

    _install_minimal_ray_module(monkeypatch)
    wq = importlib.import_module("tinker_server.backend.api_work_queue")
    client = wq.ApiWorkQueueClient()
    calls: list[tuple[str, object]] = []

    class _DescribePendingHandle:
        def remote(self, request_id: str, op: str | None):
            calls.append(("describe_pending_request.remote", (request_id, op)))
            return object()

    class _FindPositionHandle:
        def remote(self, *, request_id: str):
            calls.append(("find_position.remote", request_id))
            return object()

    class _EtaHandle:
        def remote(self, op: str | None):
            calls.append(("get_eta_state.remote", op))
            return object()

    class _Actor:
        describe_pending_request = _DescribePendingHandle()
        find_position = _FindPositionHandle()
        get_eta_state = _EtaHandle()

    async def _fake_get_async(*, require_ready: bool = True):
        calls.append(("_get_ray_actor_async", require_ready))
        return _Actor()

    def _unexpected_get_sync(*, require_ready: bool = True):
        raise AssertionError("sync _get_ray_actor should not be used on request path")

    async def _fake_await(_ref, *, timeout_s=None):
        calls.append(("_await_ray_ref", timeout_s))
        idx = len([item for item in calls if item[0] == "_await_ray_ref"])
        if idx == 1:
            return {"found": True, "position": 0, "depth": 1, "ema_exec_s": 2.0}
        if idx == 2:
            return {"found": True, "position": 0, "depth": 1}
        return {"ema_exec_s": 2.0}

    monkeypatch.setattr(client, "_get_ray_actor_async", _fake_get_async)
    monkeypatch.setattr(client, "_get_ray_actor", _unexpected_get_sync)
    monkeypatch.setattr(client, "_await_ray_ref", _fake_await)

    pending = await client.describe_pending_request("rid-360", "sampling.asample")
    pos = await client.find_position("rid-360")
    eta = await client.get_eta_state("sampling.asample")

    assert pending == {"found": True, "position": 0, "depth": 1, "ema_exec_s": 2.0}
    assert pos == {"found": True, "position": 0, "depth": 1}
    assert eta == {"ema_exec_s": 2.0}
    assert ("_get_ray_actor_async", False) in calls
    assert ("describe_pending_request.remote", ("rid-360", "sampling.asample")) in calls
    assert ("find_position.remote", "rid-360") in calls
    assert ("get_eta_state.remote", "sampling.asample") in calls


@pytest.mark.anyio
async def test_issue_360_api_work_queue_stats_omits_unready_scheduler_metrics(monkeypatch):
    import importlib

    _install_minimal_ray_module(monkeypatch)
    ray_mod = importlib.import_module("ray")
    monkeypatch.setattr(ray_mod, "is_initialized", lambda: True, raising=False)

    wq = importlib.import_module("tinker_server.backend.api_work_queue")
    client = wq.ApiWorkQueueClient()

    class _StatsHandle:
        def remote(self):
            return object()

    class _Actor:
        stats = _StatsHandle()

    client._ray_actor = _Actor()

    async def _fake_await(_ref, *, timeout_s=None):
        _ = timeout_s
        return {
            "depth": 5,
            "depth_legacy": 5,
            "depth_scheduled": 4,
            "scheduler_metrics_ready": False,
            "scheduler_enabled": False,
            "scheduler_picks_total": 0,
            "scheduler_switches_total": 0,
        }

    monkeypatch.setattr(client, "_await_ray_ref", _fake_await)

    payload = await client.stats(timeout_s=1.0)

    assert payload["depth"] == 5
    assert payload["depth_legacy"] == 5
    assert payload["scheduler_metrics_ready"] is False
    assert "depth_scheduled" not in payload
    assert "scheduler_enabled" not in payload
    assert "scheduler_picks_total" not in payload


@pytest.mark.anyio
async def test_issue_360_async_started_probes_skip_ready_snapshot(monkeypatch):
    import importlib

    fs_module = importlib.import_module("tinker_server.backend.future_store")
    wq_module = importlib.import_module("tinker_server.backend.api_work_queue")

    future_store = fs_module.FutureStore()
    api_work_queue = wq_module.ApiWorkQueueClient()
    calls: list[tuple[str, bool]] = []

    async def _fake_future_get(*, require_ready: bool = True):
        calls.append(("future_store", bool(require_ready)))
        return object()

    async def _fake_queue_get(*, require_ready: bool = True):
        calls.append(("api_work_queue", bool(require_ready)))
        return object()

    monkeypatch.setattr(future_store, "_get_ray_actor_async", _fake_future_get)
    monkeypatch.setattr(api_work_queue, "_get_ray_actor_async", _fake_queue_get)

    await future_store.async_ensure_started()
    await api_work_queue.async_ensure_started()

    assert calls == [
        ("future_store", False),
        ("api_work_queue", False),
    ]


def test_issue_360_training_optim_step_admission_uses_async_capacity_and_future(monkeypatch):
    fs = _AsyncOnlyTrainingFutureStore()
    cap = _AsyncOnlyCapacityManager()
    q = _RecordingQueue()

    session = SimpleNamespace(backend="peft", base_model="Qwen/Qwen3-0.6B")
    async def _restore_training_session(_mid):
        return None

    monkeypatch.setattr(training_route, "future_store", fs)
    monkeypatch.setattr(
        training_route,
        "training_manager",
        SimpleNamespace(get_session=lambda _mid: session, mark_inflight=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(training_route, "training_engine", object())
    monkeypatch.setattr(training_route, "_restore_training_session", _restore_training_session)

    async def _get_training_route_session_info(_model_id: str):
        return {
            "model_id": "run-360",
            "session_id": "sess-360",
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "user_id": "user-a",
        }

    async def _protect_training_session_enqueue_window(_session_info: dict):
        return None

    monkeypatch.setattr(training_route, "_get_training_route_session_info", _get_training_route_session_info)
    monkeypatch.setattr(training_route, "_protect_training_session_enqueue_window", _protect_training_session_enqueue_window)

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(cm, "capacity_manager", cap)
    monkeypatch.setattr(awq, "api_work_queue", q)
    monkeypatch.setattr(rse, "estimate_small_result_bytes", lambda: 0)

    req = OptimStepRequest(
        model_id="run-360",
        adam_params=AdamParams(learning_rate=1e-4),
    )
    out = anyio.run(training_route.optim_step, req, _request_stub("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(cap.calls) == 1
    assert any(name == "async_create_with_id" for name, _rid in fs.calls)
    assert len(q.calls) == 1


def _install_stateless_training_enqueue_stubs(monkeypatch, *, route_session_info: dict | None):
    fs = _AsyncOnlyTrainingFutureStore()
    cap = _AsyncOnlyCapacityManager()
    q = _RecordingQueue()

    async def _get_training_route_session_info(_model_id: str):
        return route_session_info

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.result_size_estimator as rse
    import tinker_server.client_compat as client_compat

    monkeypatch.setattr(training_route, "future_store", fs)
    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "_get_training_route_session_info", _get_training_route_session_info)
    monkeypatch.setattr(training_route, "build_billing_auth_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(training_route, "is_admin_request", lambda _request: False)
    monkeypatch.setattr(cm, "capacity_manager", cap)
    monkeypatch.setattr(awq, "api_work_queue", q)
    monkeypatch.setattr(rse, "estimate_small_result_bytes", lambda: 0)
    monkeypatch.setattr(rse, "estimate_forward_backward_result_bytes", lambda _request: 0)
    monkeypatch.setattr(client_compat, "prefer_tinker_uri", lambda _request: True)

    return fs, cap, q


@pytest.mark.parametrize(
    ("route_name", "request_factory", "expected_op", "expected_training_op", "expected_seq_id", "expect_prefer_tinker"),
    [
        (
            "forward_backward",
            lambda: ForwardBackwardRequest(
                model_id="run-detached",
                seq_id=21,
                forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
            ),
            "training.forward_backward",
            "forward_backward",
            21,
            False,
        ),
        (
            "train_step",
            lambda: TrainStepRequest(
                model_id="run-detached",
                seq_id=22,
                forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
                adam_params=AdamParams(learning_rate=1e-4),
            ),
            "training.train_step",
            "train_step",
            22,
            False,
        ),
        (
            "forward",
            lambda: ForwardRequest(
                model_id="run-detached",
                seq_id=23,
                forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
            ),
            "training.forward",
            "forward",
            23,
            False,
        ),
        (
            "optim_step",
            lambda: OptimStepRequest(
                model_id="run-detached",
                seq_id=24,
                adam_params=AdamParams(learning_rate=1e-4),
            ),
            "training.optim_step",
            "optim_step",
            24,
            False,
        ),
        (
            "save_weights_for_sampler",
            lambda: SaveWeightsForSamplerRequest(
                model_id="run-detached",
                seq_id=25,
                path=None,
            ),
            "training.save_weights_for_sampler",
            "save_weights_for_sampler",
            25,
            True,
        ),
    ],
)
def test_issue_360_training_enqueue_routes_use_detached_metadata_with_api_globals_none(
    monkeypatch,
    route_name: str,
    request_factory,
    expected_op: str,
    expected_training_op: str,
    expected_seq_id: int,
    expect_prefer_tinker: bool,
):
    fs, cap, q = _install_stateless_training_enqueue_stubs(
        monkeypatch,
        route_session_info={
            "model_id": "run-detached",
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "user_id": "user-a",
        },
    )

    route = getattr(training_route, route_name)
    out = anyio.run(route, request_factory(), _request_stub("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(cap.calls) == 1
    assert any(name == "async_create_with_id" for name, _rid in fs.calls)
    assert len(q.calls) == 1
    queued = q.calls[0]
    assert queued["op"] == expected_op
    assert queued["user_id"] == "user-a"
    assert queued["extra"]["training_op"] == expected_training_op
    assert queued["extra"]["scheduler_session_key"] == "run-detached"
    assert queued["extra"]["execution_serial_key"] == "training_session:run-detached"
    assert queued["extra"].get("seq_id") == expected_seq_id
    if expect_prefer_tinker:
        assert queued["extra"]["prefer_tinker"] is True
        assert queued["extra"]["is_admin"] is False
    else:
        assert "prefer_tinker" not in queued["extra"]


def test_issue_360_training_enqueue_propagates_detached_store_503(monkeypatch):
    async def _raise_store_unavailable(_model_id: str):
        raise training_route.HTTPException(status_code=503, detail="Training session store unavailable")

    async def _unexpected_async_remote_training_model(*_args, **_kwargs):
        raise AssertionError("remote fallback should not run after detached-store failure")

    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "_get_training_route_session_info", _raise_store_unavailable)

    import tinker_server.gateway as gw

    monkeypatch.setattr(gw, "async_remote_training_model", _unexpected_async_remote_training_model)

    req = ForwardRequest(
        model_id="run-detached-fail",
        seq_id=26,
        forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )

    with pytest.raises(training_route.HTTPException) as excinfo:
        anyio.run(training_route.forward, req, _request_stub("user-a"))

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Training session store unavailable"


def test_issue_360_training_route_session_helper_returns_503_on_store_failure(monkeypatch):
    import tinker_server.backend.training_session_store as tss

    async def _async_get_training_session_info(_model_id: str):
        raise RuntimeError("store down")

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)

    with pytest.raises(training_route.HTTPException) as excinfo:
        anyio.run(training_route._get_training_route_session_info, "run-detached-fail")

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Training session store unavailable"


def test_issue_360_training_route_session_helper_does_not_fallback_to_local_state(monkeypatch):
    import tinker_server.backend.training_session_store as tss

    async def _async_get_training_session_info(_model_id: str):
        return None

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)
    monkeypatch.setattr(
        training_route,
        "training_manager",
        SimpleNamespace(get_session=lambda _model_id: SimpleNamespace(model_id="local-only")),
    )

    info = anyio.run(training_route._get_training_route_session_info, "run-detached-miss")

    assert info is None


def test_issue_360_training_enqueue_refreshes_detached_heartbeat_before_queue(monkeypatch):
    fs, cap, q = _install_stateless_training_enqueue_stubs(
        monkeypatch,
        route_session_info={
            "model_id": "run-detached",
            "session_id": "sess-detached",
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "user_id": "user-a",
        },
    )

    calls: list[tuple[str, float | None]] = []

    class _HeartbeatStore:
        async def async_update(self, session_id: str, now: float | None = None) -> None:
            calls.append((session_id, now))

    import tinker_server.backend.session_heartbeat_store as shs

    monkeypatch.setattr(shs, "session_heartbeat_store", _HeartbeatStore())

    req = ForwardRequest(
        model_id="run-detached",
        seq_id=31,
        forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )

    out = anyio.run(training_route.forward, req, _request_stub("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(calls) == 1
    assert calls[0][0] == "sess-detached"
    assert calls[0][1] is not None
    assert len(cap.calls) == 1
    assert any(name == "async_create_with_id" for name, _rid in fs.calls)
    assert len(q.calls) == 1


def test_issue_360_training_enqueue_returns_503_when_heartbeat_protection_fails(monkeypatch):
    _fs, _cap, q = _install_stateless_training_enqueue_stubs(
        monkeypatch,
        route_session_info={
            "model_id": "run-detached",
            "session_id": "sess-detached",
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "user_id": "user-a",
        },
    )

    class _BrokenHeartbeatStore:
        async def async_update(self, session_id: str, now: float | None = None) -> None:
            _ = (session_id, now)
            raise RuntimeError("heartbeat store unavailable")

    import tinker_server.backend.session_heartbeat_store as shs

    monkeypatch.setattr(shs, "session_heartbeat_store", _BrokenHeartbeatStore())

    req = ForwardRequest(
        model_id="run-detached",
        seq_id=32,
        forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )

    with pytest.raises(training_route.HTTPException) as excinfo:
        anyio.run(training_route.forward, req, _request_stub("user-a"))

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Training heartbeat store unavailable"
    assert q.calls == []


@pytest.mark.parametrize(
    ("route_name", "request_factory", "expected_path", "response_payload", "expect_register"),
    [
        (
            "forward_backward",
            lambda: ForwardBackwardRequest(
                model_id="run-remote",
                seq_id=11,
                forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
            ),
            "/api/v1/forward_backward",
            {"request_id": "upstream-rid"},
            False,
        ),
        (
            "train_step",
            lambda: TrainStepRequest(
                model_id="run-remote",
                seq_id=12,
                forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
                adam_params=AdamParams(learning_rate=1e-4),
            ),
            "/api/v1/train_step",
            {"request_id": "upstream-rid"},
            False,
        ),
        (
            "forward",
            lambda: ForwardRequest(
                model_id="run-remote",
                seq_id=13,
                forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
            ),
            "/api/v1/forward",
            {"request_id": "upstream-rid"},
            False,
        ),
        (
            "optim_step",
            lambda: OptimStepRequest(
                model_id="run-remote",
                seq_id=14,
                adam_params=AdamParams(learning_rate=1e-4),
            ),
            "/api/v1/optim_step",
            {"request_id": "upstream-rid"},
            False,
        ),
        (
            "save_weights_for_sampler",
            lambda: SaveWeightsForSamplerRequest(
                model_id="run-remote",
                seq_id=15,
                path=None,
            ),
            "/api/v1/save_weights_for_sampler",
            {"request_id": "upstream-rid"},
            True,
        ),
        (
            "get_info",
            lambda: GetInfoRequest(model_id="run-remote"),
            "/api/v1/get_info",
            {
                "model_id": "run-remote",
                "model_data": {
                    "arch": "QwenForCausalLM",
                    "model_name": "Qwen/Qwen3-0.6B",
                    "tokenizer_id": "Qwen/Qwen3-0.6B",
                },
                "model_name": "Qwen/Qwen3-0.6B",
                "is_lora": True,
                "lora_rank": 8,
                "type": "get_info",
            },
            False,
        ),
    ],
)
def test_issue_360_training_remote_forwarding_uses_async_gateway_helpers(
    monkeypatch,
    route_name: str,
    request_factory,
    expected_path: str,
    response_payload: dict,
    expect_register: bool,
):
    _patch_training_route_remote_fallback(monkeypatch)
    calls = _install_gateway_forward_stubs(monkeypatch, response_payload=response_payload)

    route = getattr(training_route, route_name)
    result = anyio.run(route, request_factory(), _request_stub("admin"))

    assert calls["async_remote_training_model"] == "run-remote"
    assert calls["upstream_alias"] == "upstream-a"

    forward_call = calls["forward_json"]
    assert forward_call["method"] == "POST"
    assert forward_call["path"] == expected_path
    assert forward_call["upstream_alias"] == "upstream-a"
    assert forward_call["json_body"]["model_id"] == "run-remote"

    if route_name == "get_info":
        assert result.model_id == "run-remote"
        assert result.model_name == "Qwen/Qwen3-0.6B"
        assert result.lora_rank == 8
        assert "encode_request_id" not in calls
    else:
        assert calls["encode_request_id"] == ("upstream-a", "upstream-rid")
        assert result.request_id == "upstream-a:upstream-rid"

    if expect_register:
        assert calls["register_pending_save_weights_for_sampler_future"] == (
            "upstream-a",
            "upstream-rid",
            "Qwen/Qwen3-0.6B",
        )
    else:
        assert "register_pending_save_weights_for_sampler_future" not in calls


def test_issue_360_service_get_session_uses_async_index_store(monkeypatch):
    import tinker_server.backend.session_index_store as sis

    def _sync_get_session_index(_session_id: str):
        raise AssertionError("sync get_session_index should not be used on request path")

    async def _async_get_session_index(session_id: str):
        return {
            "session_id": session_id,
            "training_run_ids": ["run-360"],
            "sampler_ids": ["sampler-360"],
            "user_id": "admin",
        }

    monkeypatch.setattr(sis, "get_session_index", _sync_get_session_index)
    monkeypatch.setattr(sis, "async_get_session_index", _async_get_session_index)

    out = anyio.run(service_route.get_session, "sess-360", _request_stub("admin"))

    assert out.training_run_ids == ["run-360"]
    assert out.sampler_ids == ["sampler-360"]


def test_issue_360_service_list_sessions_uses_async_index_store(monkeypatch):
    import tinker_server.backend.session_index_store as sis

    def _sync_list_session_index():
        raise AssertionError("sync list_session_index should not be used on request path")

    async def _async_list_session_index():
        return [
            {
                "session_id": "sess-360-b",
                "created_at": "2026-03-21T00:00:02",
                "user_id": "admin",
            },
            {
                "session_id": "sess-360-a",
                "created_at": "2026-03-21T00:00:01",
                "user_id": "admin",
            },
        ]

    monkeypatch.setattr(sis, "list_session_index", _sync_list_session_index)
    monkeypatch.setattr(sis, "async_list_session_index", _async_list_session_index)

    out = anyio.run(service_route.list_sessions, 20, 0, _request_stub("admin"))

    assert out.sessions == ["sess-360-b", "sess-360-a"]


def test_issue_360_training_run_metadata_uses_async_store(monkeypatch):
    import tinker_server.backend.training_session_store as tss

    def _sync_get_training_session_info(_model_id: str):
        raise AssertionError("sync get_training_session_info should not be used on request path")

    async def _async_get_training_session_info(model_id: str):
        return {
            "model_id": model_id,
            "base_model": "Qwen/Qwen3-0.6B",
            "user_id": "admin",
            "created_at": "2026-03-21T00:00:00",
            "model_seq_id": 1,
        }

    monkeypatch.setattr(tss, "get_training_session_info", _sync_get_training_session_info)
    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)
    monkeypatch.setattr(training_route, "training_manager", None)

    out = anyio.run(training_route.get_training_run, "run-360", _request_stub("admin"))

    assert out.training_run_id == "run-360"
    assert out.base_model == "Qwen/Qwen3-0.6B"


class _TrainingManagerStub:
    def __init__(self):
        self.sessions: dict[str, SimpleNamespace] = {}
        self.deleted: list[str] = []

    def get_session(self, model_id: str):
        return self.sessions.get(model_id)

    def create_session(self, **kwargs):
        session = SimpleNamespace(
            model_id=kwargs["model_id"],
            session_id=kwargs["session_id"],
            model_seq_id=kwargs["model_seq_id"],
            base_model=kwargs["base_model"],
            lora_config=kwargs.get("lora_config"),
            rollout_correction_config=kwargs.get("rollout_correction_config"),
            user_metadata=kwargs.get("user_metadata") or {},
            user_id=kwargs.get("user_id"),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            backend="peft",
            created_at="",
            current_step=0,
            is_active=False,
        )
        self.sessions[kwargs["model_id"]] = session
        return session

    def delete_session(self, model_id: str) -> bool:
        self.deleted.append(model_id)
        return self.sessions.pop(model_id, None) is not None


def _patch_restore_training_info(monkeypatch, info: dict):
    import tinker_server.backend.training_session_store as tss

    async def _async_get_training_session_info(_model_id: str):
        return dict(info)

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)


def test_issue_360_restore_training_session_binds_async_lookup_actor(monkeypatch):
    manager = _TrainingManagerStub()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})
    worker = object()

    _patch_restore_training_info(
        monkeypatch,
        {
            "model_id": "run-restore-hit",
            "session_id": "sess-restore-hit",
            "model_seq_id": 3,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "actor_name": "trainer-a",
            "namespace": "ns-a",
            "current_step": 7,
        },
    )
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda *_args, **_kwargs: None)

    async def _async_lookup_actor_handle(actor_name: str, namespace: str, *, timeout_s: float = 5.0):
        _ = timeout_s
        assert actor_name == "trainer-a"
        assert namespace == "ns-a"
        return worker

    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)

    restored = anyio.run(training_route._restore_training_session, "run-restore-hit")

    assert restored is manager.sessions["run-restore-hit"]
    assert restored.current_step == 7
    assert engine._workers["run-restore-hit"] is worker
    assert engine._resource_pool_actor_names["run-restore-hit"] == "trainer-a"
    assert manager.deleted == []


def test_issue_360_restore_training_session_rolls_back_created_session_on_lookup_miss(monkeypatch):
    manager = _TrainingManagerStub()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})

    _patch_restore_training_info(
        monkeypatch,
        {
            "model_id": "run-restore-miss",
            "session_id": "sess-restore-miss",
            "model_seq_id": 4,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "actor_name": "trainer-missing",
            "namespace": "ns-missing",
        },
    )
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda *_args, **_kwargs: None)

    async def _async_lookup_actor_handle(*_args, **_kwargs):
        return None

    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)

    restored = anyio.run(training_route._restore_training_session, "run-restore-miss")

    assert restored is None
    assert "run-restore-miss" not in manager.sessions
    assert manager.deleted == ["run-restore-miss"]
    assert engine._workers == {}
    assert engine._resource_pool_actor_names == {}


def test_issue_360_restore_training_session_rolls_back_existing_session_on_lookup_miss(monkeypatch):
    manager = _TrainingManagerStub()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})

    # Pre-existing local session should not be mutated when restore misses.
    existing = manager.create_session(
        model_id="run-restore-existing-miss",
        session_id="sess-local",
        model_seq_id=9,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=None,
        rollout_correction_config=None,
        user_metadata={},
        user_id="admin",
        learning_rate=1e-4,
    )
    existing.backend = "peft"
    existing.created_at = "local-created"
    existing.current_step = 123
    existing.is_active = False

    _patch_restore_training_info(
        monkeypatch,
        {
            "model_id": "run-restore-existing-miss",
            "session_id": "sess-remote",
            "model_seq_id": 10,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "megatron",
            "actor_name": "trainer-missing",
            "namespace": "ns-missing",
            "current_step": 7,
            "created_at": "remote-created",
        },
    )
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda *_args, **_kwargs: None)

    async def _async_lookup_actor_handle(*_args, **_kwargs):
        return None

    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)

    restored = anyio.run(training_route._restore_training_session, "run-restore-existing-miss")

    assert restored is None
    assert manager.deleted == []
    assert manager.sessions["run-restore-existing-miss"] is existing
    assert existing.backend == "peft"
    assert existing.created_at == "local-created"
    assert existing.current_step == 123
    assert existing.is_active is False
    assert engine._workers == {}
    assert engine._resource_pool_actor_names == {}


def test_issue_360_restore_training_session_without_actor_name_keeps_session_semantics(monkeypatch):
    manager = _TrainingManagerStub()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})

    _patch_restore_training_info(
        monkeypatch,
        {
            "model_id": "run-restore-no-actor",
            "session_id": "sess-restore-no-actor",
            "model_seq_id": 5,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "current_step": 9,
        },
    )
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)

    async def _unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("actor lookup should not run when actor_name is absent")

    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _unexpected_lookup)

    restored = anyio.run(training_route._restore_training_session, "run-restore-no-actor")

    assert restored is manager.sessions["run-restore-no-actor"]
    assert restored.current_step == 9
    assert manager.deleted == []
    assert engine._workers == {}
    assert engine._resource_pool_actor_names == {}


def test_issue_360_restore_training_session_reuses_resource_pool_handle(monkeypatch):
    manager = _RecordingTrainingManager()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})
    worker = object()

    async def _async_get_training_session_info(model_id: str):
        return {
            "model_id": model_id,
            "session_id": "sess-restore",
            "model_seq_id": 7,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "user_metadata": {"owner": "u1"},
            "user_id": "u1",
            "learning_rate": 2e-4,
            "backend": "megatron",
            "created_at": "2026-03-21T00:00:00",
            "current_step": 9,
            "actor_name": "trainer-actor",
            "namespace": "ns-restore",
        }

    async def _async_lookup_actor_handle(_actor_name: str, _namespace: str, *, timeout_s: float = 5.0):
        _ = timeout_s
        raise AssertionError("async actor lookup should not run when a ResourcePool handle is available")

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda actor_name, namespace: worker)
    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    import tinker_server.backend.training_session_store as tss

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)

    session = anyio.run(training_route._restore_training_session, "run-restore")

    assert session is manager.sessions["run-restore"]
    assert session.backend == "megatron"
    assert session.current_step == 9
    assert session.is_active is True
    assert engine._workers["run-restore"] is worker
    assert engine._resource_pool_actor_names["run-restore"] == "trainer-actor"


def test_issue_360_restore_training_session_falls_back_to_async_actor_lookup(monkeypatch):
    manager = _RecordingTrainingManager()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})
    worker = object()
    lookup_calls: list[tuple[str, str, float]] = []

    async def _async_get_training_session_info(model_id: str):
        return {
            "model_id": model_id,
            "session_id": "sess-restore",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-0.6B",
            "user_metadata": {},
            "user_id": "u1",
            "learning_rate": 1e-4,
            "backend": "peft",
            "created_at": "2026-03-21T00:00:00",
            "current_step": 3,
            "actor_name": "trainer-actor",
            "namespace": "ns-restore",
        }

    async def _async_lookup_actor_handle(actor_name: str, namespace: str, *, timeout_s: float = 5.0):
        lookup_calls.append((actor_name, namespace, timeout_s))
        return worker

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda actor_name, namespace: None)
    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    import tinker_server.backend.training_session_store as tss

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)

    session = anyio.run(training_route._restore_training_session, "run-restore")

    assert session is manager.sessions["run-restore"]
    assert lookup_calls == [("trainer-actor", "ns-restore", 5.0)]
    assert engine._workers["run-restore"] is worker
    assert engine._resource_pool_actor_names["run-restore"] == "trainer-actor"


def test_issue_360_restore_training_session_missing_actor_returns_none(monkeypatch):
    manager = _RecordingTrainingManager()
    engine = SimpleNamespace(_workers={}, _resource_pool_actor_names={})

    async def _async_get_training_session_info(model_id: str):
        return {
            "model_id": model_id,
            "session_id": "sess-restore",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-0.6B",
            "user_metadata": {},
            "user_id": "u1",
            "learning_rate": 1e-4,
            "backend": "peft",
            "created_at": "2026-03-21T00:00:00",
            "current_step": 3,
            "actor_name": "trainer-actor",
            "namespace": "ns-restore",
        }

    async def _async_lookup_actor_handle(_actor_name: str, _namespace: str, *, timeout_s: float = 5.0):
        _ = timeout_s
        raise RuntimeError("actor lookup failed")

    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_find_actor_handle", lambda actor_name, namespace: None)
    monkeypatch.setattr(training_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    import tinker_server.backend.training_session_store as tss

    monkeypatch.setattr(tss, "async_get_training_session_info", _async_get_training_session_info)

    session = anyio.run(training_route._restore_training_session, "run-restore")

    assert session is None
    assert "run-restore" not in manager.sessions
    assert manager.deleted == ["run-restore"]
    assert engine._workers == {}
    assert engine._resource_pool_actor_names == {}


def test_issue_360_kill_dense_actors_uses_actor_name_without_cached_handle(monkeypatch):
    unregister_calls: list[str] = []
    kill_calls: list[tuple[str, str, str | None]] = []

    async def _async_kill_named_actor(
        actor_name: str,
        namespace: str,
        *,
        actor_handle=None,
        base_model: str | None,
        reason: str,
        timeout_s: float = 10.0,
        verify_absent: bool = False,
    ):
        _ = (actor_handle, reason, timeout_s, verify_absent)
        kill_calls.append((actor_name, namespace, base_model))
        return True

    pool = SimpleNamespace(
        iter_entries=lambda: [
            SimpleNamespace(
                actor_type="dense",
                actor_name="dense-a",
                namespace="ns-dense",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
        unregister=lambda actor_name: unregister_calls.append(actor_name),
    )

    _install_minimal_ray_module(monkeypatch)
    import tinker_server.backend.resource_pool as rp

    monkeypatch.setattr(rp, "ActorType", SimpleNamespace(DENSE="dense"))
    monkeypatch.setattr(rp, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(service_route, "async_kill_named_actor", _async_kill_named_actor)

    killed = anyio.run(service_route._kill_dense_actors, "Qwen/Qwen3-0.6B")

    assert killed == 1
    assert kill_calls == [("dense-a", "ns-dense", "Qwen/Qwen3-0.6B")]
    assert unregister_calls == ["dense-a"]


def test_issue_360_kill_exact_vllm_actor_propagates_lookup_failures(monkeypatch):
    unregister_calls: list[str] = []
    removed_pgs: list[str] = []

    pool = SimpleNamespace(
        get=lambda actor_name: SimpleNamespace(
            actor_type="vllm",
            namespace="ns-vllm",
            base_model="Qwen/Qwen3-0.6B",
        ),
        unregister=lambda actor_name: unregister_calls.append(actor_name),
    )

    async def _async_lookup_actor_handle(*_args, **_kwargs):
        raise RuntimeError("ray unavailable")

    _install_minimal_ray_module(monkeypatch)
    _install_namespace_module(monkeypatch, "tinker_server.backend.multi_lora_engine", "ns-vllm")
    import tinker_server.backend.resource_pool as rp

    monkeypatch.setattr(rp, "ActorType", SimpleNamespace(VLLM="vllm"))
    monkeypatch.setattr(rp, "ResourcePoolStaleError", RuntimeError)
    monkeypatch.setattr(rp, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(service_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    monkeypatch.setattr(service_route, "is_actor_lookup_not_found", lambda exc: False)
    monkeypatch.setattr(service_route, "_remove_actor_pg", lambda actor_name: removed_pgs.append(actor_name))

    with pytest.raises(RuntimeError, match="ray unavailable"):
        anyio.run(lambda: service_route._kill_exact_vllm_actor(actor_name="vllm-a"))

    assert unregister_calls == []
    assert removed_pgs == []


def test_issue_360_kill_exact_megatron_actor_verifies_absence(monkeypatch):
    unregister_calls: list[str] = []
    removed_pgs: list[str] = []
    kill_calls: list[dict] = []

    class _ShutdownRemote:
        def remote(self):
            return _AwaitableObjectRef(result=None)

    actor = SimpleNamespace(shutdown=_ShutdownRemote())
    pool = SimpleNamespace(
        get=lambda actor_name: SimpleNamespace(
            actor_type="megatron",
            namespace="ns-mega",
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
        unregister=lambda actor_name: unregister_calls.append(actor_name),
    )

    async def _async_lookup_actor_handle(*_args, **_kwargs):
        return actor

    async def _async_kill_named_actor(actor_name: str, namespace: str, *, actor_handle=None, base_model: str | None, reason: str, verify_absent: bool, timeout_s: float = 10.0):
        _ = timeout_s
        kill_calls.append(
            {
                "actor_name": actor_name,
                "namespace": namespace,
                "actor_handle": actor_handle,
                "base_model": base_model,
                "reason": reason,
                "verify_absent": verify_absent,
            }
        )
        return True

    _install_minimal_ray_module(monkeypatch)
    _install_namespace_module(monkeypatch, "tinker_server.backend.megatron_distributed", "ns-mega")
    import tinker_server.backend.resource_pool as rp

    monkeypatch.setattr(rp, "ActorType", SimpleNamespace(MEGATRON="megatron"))
    monkeypatch.setattr(rp, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(service_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    monkeypatch.setattr(service_route, "async_kill_named_actor", _async_kill_named_actor)
    monkeypatch.setattr(service_route, "_remove_actor_pg", lambda actor_name: removed_pgs.append(actor_name))

    killed = anyio.run(lambda: service_route._kill_exact_megatron_actor(actor_name="mega-a"))

    assert killed == 1
    assert kill_calls == [
        {
            "actor_name": "mega-a",
            "namespace": "ns-mega",
            "actor_handle": actor,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "reason": "kill_megatron_actor_by_name",
            "verify_absent": True,
        }
    ]
    assert unregister_calls == ["mega-a"]
    assert removed_pgs == ["mega-a"]


def test_issue_360_kill_dense_actors_cleans_pg_when_async_kill_fails(monkeypatch):
    unregister_calls: list[str] = []
    removed_pgs: list[str] = []

    async def _async_kill_named_actor(*_args, **_kwargs):
        raise RuntimeError("kill failed")

    pool = SimpleNamespace(
        iter_entries=lambda: [
            SimpleNamespace(
                actor_type="dense",
                actor_name="dense-fail",
                namespace="ns-dense",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
        unregister=lambda actor_name: unregister_calls.append(actor_name),
    )

    _install_minimal_ray_module(monkeypatch)
    import tinker_server.backend.resource_pool as rp

    monkeypatch.setattr(rp, "ActorType", SimpleNamespace(DENSE="dense"))
    monkeypatch.setattr(rp, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(service_route, "async_kill_named_actor", _async_kill_named_actor)
    monkeypatch.setattr(service_route, "_remove_actor_pg", lambda actor_name: removed_pgs.append(actor_name))

    killed = anyio.run(service_route._kill_dense_actors, "Qwen/Qwen3-0.6B")

    assert killed == 1
    assert unregister_calls == ["dense-fail"]
    assert removed_pgs == ["dense-fail"]


def test_issue_360_kill_exact_vllm_actor_passes_resolved_handle_to_async_kill(monkeypatch):
    actor = object()
    unregister_calls: list[str] = []
    removed_pgs: list[str] = []
    kill_calls: list[dict] = []

    pool = SimpleNamespace(
        get=lambda actor_name: SimpleNamespace(
            actor_type="vllm",
            namespace="ns-vllm",
            base_model="Qwen/Qwen3-0.6B",
        ),
        unregister=lambda actor_name: unregister_calls.append(actor_name),
    )

    async def _async_lookup_actor_handle(*_args, **_kwargs):
        return actor

    async def _async_kill_named_actor(actor_name: str, namespace: str, *, actor_handle=None, base_model: str | None, reason: str, timeout_s: float = 10.0, verify_absent: bool = False):
        _ = (timeout_s, verify_absent)
        kill_calls.append(
            {
                "actor_name": actor_name,
                "namespace": namespace,
                "actor_handle": actor_handle,
                "base_model": base_model,
                "reason": reason,
            }
        )
        return True

    _install_minimal_ray_module(monkeypatch)
    _install_namespace_module(monkeypatch, "tinker_server.backend.multi_lora_engine", "ns-vllm")
    _install_resource_pool_module(
        monkeypatch,
        pool=pool,
        actor_types=SimpleNamespace(VLLM="vllm"),
        stale_error=RuntimeError,
    )
    monkeypatch.setattr(service_route, "async_lookup_actor_handle", _async_lookup_actor_handle)
    monkeypatch.setattr(service_route, "async_kill_named_actor", _async_kill_named_actor)
    monkeypatch.setattr(service_route, "_remove_actor_pg", lambda actor_name: removed_pgs.append(actor_name))

    killed = anyio.run(lambda: service_route._kill_exact_vllm_actor(actor_name="vllm-a"))

    assert killed == 1
    assert kill_calls == [
        {
            "actor_name": "vllm-a",
            "namespace": "ns-vllm",
            "actor_handle": actor,
            "base_model": "Qwen/Qwen3-0.6B",
            "reason": "vllm_kill_by_actor_name",
        }
    ]
    assert unregister_calls == ["vllm-a"]
    assert removed_pgs == ["vllm-a"]


def test_issue_364_kill_busy_vllm_actor_rejected(monkeypatch):
    busy_entry = SimpleNamespace(
        actor_name="tinker_vllm_qwen3-0.6b",
        actor_type="vllm",
        namespace="ns-vllm",
        base_model="Qwen/Qwen3-0.6B",
        current_session=None,
        inflight_count=1,
        creating=False,
        protected=False,
    )
    pool = SimpleNamespace(iter_entries=lambda: [busy_entry])
    kill_calls: list[str | None] = []

    _install_minimal_ray_module(monkeypatch)

    monkeypatch.setattr(service_route, "_require_admin", lambda _request: None)
    _install_resource_pool_module(
        monkeypatch,
        pool=pool,
        actor_types=SimpleNamespace(VLLM="vllm", MEGATRON="megatron", DENSE="dense"),
    )
    mle_module = types.ModuleType("tinker_server.backend.multi_lora_engine")
    monkeypatch.setitem(sys.modules, "tinker_server.backend.multi_lora_engine", mle_module)
    monkeypatch.setattr(
        mle_module,
        "kill_persistent_vllm_actor",
        lambda model_name=None: kill_calls.append(model_name) or True,
        raising=False,
    )

    with pytest.raises(service_route.HTTPException, match="Refusing to kill busy actor"):
        anyio.run(
            service_route.kill_actors,
            _request_stub("admin"),
            service_route.KillActorsRequest(actor_type="vllm", model_name="Qwen/Qwen3-0.6B"),
        )

    assert kill_calls == []


def test_issue_364_kill_busy_vllm_actor_force_override(monkeypatch):
    busy_entry = SimpleNamespace(
        actor_name="tinker_vllm_qwen3-0.6b",
        actor_type="vllm",
        namespace="ns-vllm",
        base_model="Qwen/Qwen3-0.6B",
        current_session=None,
        inflight_count=2,
        creating=False,
        protected=False,
    )
    pool = SimpleNamespace(iter_entries=lambda: [busy_entry])
    kill_calls: list[str | None] = []

    _install_minimal_ray_module(monkeypatch)

    monkeypatch.setattr(service_route, "_require_admin", lambda _request: None)
    _install_resource_pool_module(
        monkeypatch,
        pool=pool,
        actor_types=SimpleNamespace(VLLM="vllm", MEGATRON="megatron", DENSE="dense"),
    )
    mle_module = types.ModuleType("tinker_server.backend.multi_lora_engine")
    monkeypatch.setitem(sys.modules, "tinker_server.backend.multi_lora_engine", mle_module)
    monkeypatch.setattr(
        mle_module,
        "kill_persistent_vllm_actor",
        lambda model_name=None: kill_calls.append(model_name) or True,
        raising=False,
    )

    result = anyio.run(
        service_route.kill_actors,
        _request_stub("admin"),
        service_route.KillActorsRequest(
            actor_type="vllm",
            model_name="Qwen/Qwen3-0.6B",
            force=True,
            reason="test-force",
        ),
    )

    assert result == {"killed": 1, "killed_by_type": {"vllm": 1, "megatron": 0, "dense": 0}}
    assert kill_calls == ["Qwen/Qwen3-0.6B"]


class _AwaitableObjectRef:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def __await__(self):
        async def _run():
            if self._error is not None:
                raise self._error
            return self._result

        return _run().__await__()

    def future(self):
        fut = Future()
        if self._error is not None:
            fut.set_exception(self._error)
        else:
            fut.set_result(self._result)
        return fut


class _RemoteCall:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def remote(self, *args, **kwargs):
        return _AwaitableObjectRef(result=self._result, error=self._error)


def test_issue_360_future_store_async_get_status_backend_api(monkeypatch):
    import importlib

    fs_module = importlib.import_module("tinker_server.backend.future_store")

    store = fs_module.FutureStore()
    assert hasattr(store, "async_get_status"), "FutureStore must expose async_get_status for request paths"

    actor = SimpleNamespace(get_status=_RemoteCall(result="done"))
    store._ray_actor = actor

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: True  # type: ignore[attr-defined]

    class _ActorDiedError(Exception):
        pass

    class _RayTaskError(Exception):
        def __init__(self, message: str, *, cause: Exception | None = None):
            super().__init__(message)
            self.cause = cause

    ray_mod.exceptions = SimpleNamespace(
        ActorDiedError=_ActorDiedError,
        RayTaskError=_RayTaskError,
    )
    monkeypatch.setitem(sys.modules, "ray", ray_mod)

    out = anyio.run(store.async_get_status, "rid_backend_async")
    assert out == FutureStatus.DONE


def test_issue_360_session_index_async_reacquires_dead_actor(monkeypatch):
    import importlib

    store_module = importlib.import_module("tinker_server.backend.session_index_store")

    class _ActorDiedError(Exception):
        pass

    class _RayActorError(Exception):
        pass

    stale_actor = SimpleNamespace(get_session=_RemoteCall(error=_ActorDiedError("dead actor")))
    recovered_actor = SimpleNamespace(get_session=_RemoteCall(result={"session_id": "sess-reacquired"}))
    reacquire_calls: list[tuple[str, str]] = []

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray_mod.exceptions = SimpleNamespace(
        ActorDiedError=_ActorDiedError,
        RayActorError=_RayActorError,
    )

    def _get_actor(name: str, *, namespace: str):
        reacquire_calls.append((name, namespace))
        return recovered_actor

    ray_mod.get_actor = _get_actor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setattr(store_module, "_ACTOR_HANDLE", stale_actor)

    out = anyio.run(store_module.async_get_session_index, "sess-reacquired")

    assert out == {"session_id": "sess-reacquired"}
    assert reacquire_calls == [(store_module._actor_name(), store_module._ray_namespace())]
    assert store_module._ACTOR_HANDLE is recovered_actor


def test_issue_360_gateway_session_store_async_reacquires_dead_actor(monkeypatch):
    import importlib

    store_module = importlib.import_module("tinker_server.backend.gateway_session_store")

    class _ActorDiedError(Exception):
        pass

    class _RayActorError(Exception):
        pass

    stale_actor = SimpleNamespace(get_sampling_session=_RemoteCall(error=_ActorDiedError("dead actor")))
    recovered_actor = SimpleNamespace(
        get_sampling_session=_RemoteCall(
            result={"upstream_alias": "up-a", "base_model": "Qwen/Qwen3-0.6B"}
        )
    )
    reacquire_calls: list[tuple[str, str]] = []

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray_mod.exceptions = SimpleNamespace(
        ActorDiedError=_ActorDiedError,
        RayActorError=_RayActorError,
    )

    def _get_actor(name: str, *, namespace: str):
        reacquire_calls.append((name, namespace))
        return recovered_actor

    ray_mod.get_actor = _get_actor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setattr(store_module, "_ACTOR_HANDLE", stale_actor)

    out = anyio.run(store_module.async_get_sampling_session, "sampling-1")

    assert out == ("up-a", "Qwen/Qwen3-0.6B")
    assert reacquire_calls == [(store_module._actor_name(), store_module._ray_namespace())]
    assert store_module._ACTOR_HANDLE is recovered_actor


def test_issue_360_training_session_store_async_reacquires_dead_actor(monkeypatch):
    import importlib

    store_module = importlib.import_module("tinker_server.backend.training_session_store")

    class _ActorDiedError(Exception):
        pass

    class _RayActorError(Exception):
        pass

    stale_actor = SimpleNamespace(get=_RemoteCall(error=_ActorDiedError("dead actor")))
    recovered_actor = SimpleNamespace(get=_RemoteCall(result={"model_id": "run-reacquired"}))
    reacquire_calls: list[tuple[str, str]] = []

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray_mod.exceptions = SimpleNamespace(
        ActorDiedError=_ActorDiedError,
        RayActorError=_RayActorError,
    )

    def _get_actor(name: str, *, namespace: str):
        reacquire_calls.append((name, namespace))
        return recovered_actor

    ray_mod.get_actor = _get_actor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setattr(store_module, "_ACTOR_HANDLE", stale_actor)

    out = anyio.run(store_module.async_get_training_session_info, "run-reacquired")

    assert out == {"model_id": "run-reacquired"}
    assert reacquire_calls == [(store_module._actor_name(), store_module._ray_namespace())]
    assert store_module._ACTOR_HANDLE is recovered_actor


def test_issue_360_sampling_session_store_async_reacquires_dead_actor(monkeypatch):
    import importlib

    store_module = importlib.import_module("tinker_server.backend.sampling_session_store")

    class _ActorDiedError(Exception):
        pass

    class _RayActorError(Exception):
        pass

    stale_actor = SimpleNamespace(get=_RemoteCall(error=_ActorDiedError("dead actor")))
    recovered_actor = SimpleNamespace(get=_RemoteCall(result={"session_id": "sampling-reacquired"}))
    reacquire_calls: list[tuple[str, str]] = []

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray_mod.exceptions = SimpleNamespace(
        ActorDiedError=_ActorDiedError,
        RayActorError=_RayActorError,
    )

    def _get_actor(name: str, *, namespace: str):
        reacquire_calls.append((name, namespace))
        return recovered_actor

    ray_mod.get_actor = _get_actor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setattr(store_module, "_ACTOR_HANDLE", stale_actor)

    out = anyio.run(store_module.async_get_sampling_session_info, "sampling-reacquired")

    assert out == {"session_id": "sampling-reacquired"}
    assert reacquire_calls == [(store_module._actor_name(), store_module._ray_namespace())]
    assert store_module._ACTOR_HANDLE is recovered_actor
