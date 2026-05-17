from types import SimpleNamespace

import anyio
import pytest

import tinker_server.config as config_module
from tinker_server.backend.task_state_store import FutureStatus, _meta_with_request_op
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


@pytest.fixture(autouse=True)
def _reset_retrieve_future_state(monkeypatch):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())
    monkeypatch.setattr(config_module.config, "retrieve_future_hot_ttl_s", 60.0, raising=False)


class _StubTaskStateFutures:
    _UNSET = object()

    def __init__(self, status: FutureStatus, *, result=_UNSET, error=_UNSET, meta=None):
        self._status = status
        self._result = {"ok": "default"} if result is self._UNSET else result
        self._error = "error:default" if error is self._UNSET else error
        self._meta = dict(meta or {})
        self.cleanup_calls: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        return self._status

    async def async_get_result(self, request_id: str):
        return self._result

    async def async_get_error(self, request_id: str):
        return self._error

    async def async_get_meta(self, request_id: str):
        return dict(self._meta)

    async def async_cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


class _UnknownTaskStateFutures:
    async def async_get_status(self, request_id: str) -> FutureStatus:
        raise KeyError(f"Unknown request_id: {request_id}")

    async def async_debug_snapshot(self):
        return {"stub": True}


class _StubTaskStateStore:
    def __init__(self, record: dict):
        self.record = dict(record)

    async def async_get_task(self, request_id: str) -> dict:
        if self.record.get("request_id") != request_id:
            raise KeyError(request_id)
        return dict(self.record)


def _request_stub(*, admin: bool = True):
    user_data = {"user_id": "admin"} if admin else {"user_id": "user"}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_issue_440_train_step_result_uses_task_state_store_as_replay_index(monkeypatch, tmp_path):
    import tinker_server.backend.task_state_store as task_state_store_module
    from tinker_server.backend.task_payload_store import TaskPayloadStore

    payload_root = tmp_path / "payloads"
    monkeypatch.setenv("MINT_TASK_PAYLOAD_ROOT_DIR", str(payload_root))
    monkeypatch.setattr(futures_route, "task_state_futures", _UnknownTaskStateFutures())
    payload_meta = TaskPayloadStore(payload_root).write_json_payload(
        request_id="rid-train-step",
        attempt_id="attempt-1",
        payload={"ok": "rid-train-step"},
    )
    monkeypatch.setattr(
        task_state_store_module,
        "task_state_store",
        _StubTaskStateStore(
            {
                "request_id": "rid-train-step",
                "op": "training.train_step",
                "status": "retrieved",
                "result_path": payload_meta["path"],
                "result_checksum": payload_meta["checksum"],
                "metadata": {
                    "op": "training.train_step",
                    "model_id": "m1",
                    "done_at": 990.0,
                    "terminal_status": "done",
                },
                "updated_at": 1000.0,
            }
        ),
    )

    body = FutureRetrieveRequest(request_id="rid-train-step")
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())

    assert payload == {"ok": "rid-train-step"}


def test_issue_440_random_unknown_without_task_state_record_stays_404(monkeypatch):
    monkeypatch.setattr(futures_route, "task_state_futures", _UnknownTaskStateFutures())

    body = FutureRetrieveRequest(request_id="rid-unknown")
    with pytest.raises(futures_route.HTTPException) as exc:
        anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert exc.value.status_code == 404


def test_issue_440_known_terminal_future_evicted_not_unknown(monkeypatch):
    import tinker_server.backend.task_state_store as task_state_store_module

    monkeypatch.setattr(futures_route, "task_state_futures", _UnknownTaskStateFutures())
    monkeypatch.setattr(
        task_state_store_module,
        "task_state_store",
        _StubTaskStateStore(
            {
                "request_id": "rid-evicted",
                "op": "training.forward_backward",
                "status": "retrieved",
                "result_path": "/tmp/does-not-exist-task-payload.json",
                "result_checksum": "sha256:missing",
                "metadata": {
                    "op": "training.forward_backward",
                    "model_id": "m2",
                    "done_at": 995.0,
                    "terminal_status": "done",
                },
                "updated_at": 1010.0,
            }
        ),
    )

    body = FutureRetrieveRequest(request_id="rid-evicted")
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())

    assert payload["error"] == "Known terminal future evicted"
    assert payload["request_id"] == "rid-evicted"
    assert payload["op"] == "training.forward_backward"


def test_issue_440_failure_replay_keeps_public_masking(monkeypatch):
    import tinker_server.backend.task_state_store as task_state_store_module

    monkeypatch.setattr(config_module.config, "api_key", "", raising=False)
    monkeypatch.setattr(config_module.config, "internal_api_token", "secret", raising=False)
    monkeypatch.setattr(futures_route, "task_state_futures", _UnknownTaskStateFutures())
    monkeypatch.setattr(
        task_state_store_module,
        "task_state_store",
        _StubTaskStateStore(
            {
                "request_id": "rid-failed",
                "op": "training.optim_step",
                "status": "retrieved",
                "error": "secret backend trace",
                "metadata": {
                    "op": "training.optim_step",
                    "model_id": "m3",
                    "done_at": 1990.0,
                    "terminal_status": "failed",
                },
                "updated_at": 2000.0,
            }
        ),
    )

    body = FutureRetrieveRequest(request_id="rid-failed")
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(admin=False), _response_stub())

    assert payload == {"error": futures_route.GENERIC_ERROR_MESSAGE, "category": "system"}


def test_issue_440_concurrent_task_state_replay_returns_equivalent_payloads(monkeypatch, tmp_path):
    import tinker_server.backend.task_state_store as task_state_store_module
    from tinker_server.backend.task_payload_store import TaskPayloadStore

    payload_root = tmp_path / "payloads"
    monkeypatch.setenv("MINT_TASK_PAYLOAD_ROOT_DIR", str(payload_root))
    monkeypatch.setattr(futures_route, "task_state_futures", _UnknownTaskStateFutures())
    payload_meta = TaskPayloadStore(payload_root).write_json_payload(
        request_id="rid-race",
        attempt_id="attempt-1",
        payload={"ok": "rid-race"},
    )
    monkeypatch.setattr(
        task_state_store_module,
        "task_state_store",
        _StubTaskStateStore(
            {
                "request_id": "rid-race",
                "op": "training.train_step",
                "status": "retrieved",
                "result_path": payload_meta["path"],
                "result_checksum": payload_meta["checksum"],
                "metadata": {"op": "training.train_step", "terminal_status": "done"},
                "updated_at": 3000.0,
            }
        ),
    )
    body = FutureRetrieveRequest(request_id="rid-race")

    async def _run():
        results = [None, None]

        async def _slot(i: int):
            results[i] = await futures_route.retrieve_future(body, _request_stub(), _response_stub())

        async with anyio.create_task_group() as tg:
            tg.start_soon(_slot, 0)
            tg.start_soon(_slot, 1)
        return results

    assert anyio.run(_run) == [{"ok": "rid-race"}, {"ok": "rid-race"}]


def test_issue_440_meta_with_request_op_restores_missing_op():
    meta = {"model_id": "m4", "done_at": 1.0, "final_status": FutureStatus.FAILED.value}

    out = _meta_with_request_op(meta, "training.train_step")

    assert out["op"] == "training.train_step"
    assert out["model_id"] == "m4"
    assert out["final_status"] == FutureStatus.FAILED.value
