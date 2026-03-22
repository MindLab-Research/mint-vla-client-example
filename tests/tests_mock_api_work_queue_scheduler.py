import asyncio
import importlib
import importlib.machinery
import sys
import types

import pytest


def _install_ray_stub(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    class _Ctx:
        def get_task_id(self) -> str:
            return "task-mock"

        def get_job_id(self) -> str:
            return "job-mock"

    def remote(*_args, **_kwargs):
        def _decorator(cls):
            class _RemoteWrapped(cls):
                @classmethod
                def options(cls_, **_opts):
                    class _OptionsHandle:
                        def remote(self, *args, **kwargs):
                            return cls_(*args, **kwargs)

                    return _OptionsHandle()

            return _RemoteWrapped

        return _decorator

    def get_actor(*_args, **_kwargs):
        raise ValueError("named actor not found")

    ray.remote = remote  # type: ignore[attr-defined]
    ray.get = lambda ref, timeout=None: ref  # type: ignore[attr-defined]
    ray.get_actor = get_actor  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: {}  # type: ignore[attr-defined]
    ray.get_runtime_context = lambda: _Ctx()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray)


def _load_api_work_queue_module(monkeypatch):
    monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-env")
    monkeypatch.setenv("PFS_TINKER_PATH", "/tmp/tinker")
    monkeypatch.setenv("PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    _install_ray_stub(monkeypatch)
    import tinker_server.config as config_module
    import tinker_server.backend.api_work_queue as api_work_queue

    importlib.reload(config_module)
    return importlib.reload(api_work_queue)


def _item(
    request_id: str,
    *,
    domain: str,
    session_key: str | None = None,
    legacy_session_id: str | None = None,
    created_at: float = 0.0,
) -> dict:
    serial_session_key = session_key or legacy_session_id or "unknown"
    extra = {
        "scheduler_enabled": True,
        "scheduler_domain": domain,
        "execution_serial_key": f"training_session:{serial_session_key}",
    }
    if session_key is not None:
        extra["scheduler_session_key"] = session_key
    if legacy_session_id is not None:
        extra["session_id"] = legacy_session_id
    return {
        "request_id": request_id,
        "op": "training.forward_backward",
        "request_json": b"{}",
        "user_id": None,
        "apikey_id": None,
        "throttle_principal": None,
        "webhook_url": None,
        "extra": extra,
        "created_at": created_at,
    }


async def _enqueue_many(actor, items: list[dict]) -> None:
    for it in items:
        await actor.enqueue(it)


async def _dequeue_many(actor, n: int) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        item = await actor.dequeue("consumer-job")
        out.append(item)
        await actor.finalize_request(item["request_id"])
    return out


def _session_key_from_item(item: dict) -> str:
    extra = item.get("extra", {}) or {}
    return str(extra.get("scheduler_session_key") or extra.get("session_id") or "")


def test_mock_scheduler_sticky_then_fairness_rr(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "rr")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "2")
    monkeypatch.setenv("MINT_SCHEDULER_STARVATION_S", "1000000000000")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "0")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    t0 = 10000000000.0
    items = [
        _item("r1", domain="d", session_key="A", created_at=t0 + 1.0),
        _item("r2", domain="d", session_key="B", created_at=t0 + 2.0),
        _item("r3", domain="d", session_key="A", created_at=t0 + 3.0),
        _item("r4", domain="d", session_key="B", created_at=t0 + 4.0),
    ]
    asyncio.run(_enqueue_many(actor, items))
    out = asyncio.run(_dequeue_many(actor, 4))
    sessions = [_session_key_from_item(x) for x in out]

    # First pick is oldest A, then sticky keeps A once (max_consecutive=2),
    # then fairness rotates to B and sticky keeps B.
    assert sessions == ["A", "A", "B", "B"]


def test_mock_scheduler_invariant_current_session_without_queue_raises(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "rr")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "20")
    monkeypatch.setenv("MINT_SCHEDULER_STARVATION_S", "1000000000000")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    asyncio.run(_enqueue_many(actor, [_item("r1", domain="d", session_key="B", created_at=1.0)]))

    # Corrupt scheduler state to simulate impossible stale pointer.
    state = actor._sched_domains["d"]
    state["current_session"] = "ghost-session"
    state["last_session"] = "ghost-session"

    with pytest.raises(RuntimeError, match="scheduler invariant violated"):
        asyncio.run(actor.dequeue("consumer-job"))


def test_mock_scheduler_accepts_new_and_legacy_session_key_fields(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "oldest")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "0")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    asyncio.run(
        _enqueue_many(
            actor,
            [
                _item("r1", domain="d", session_key="new-key-A", created_at=1.0),
                _item("r2", domain="d", legacy_session_id="legacy-key-B", created_at=2.0),
            ],
        )
    )
    out = asyncio.run(_dequeue_many(actor, 2))
    sessions = {_session_key_from_item(x) for x in out}

    assert sessions == {"new-key-A", "legacy-key-B"}


def test_mock_scheduler_does_not_idle_wait_for_missing_followup(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "oldest")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "20")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    asyncio.run(_enqueue_many(actor, [_item("r1", domain="d", session_key="A", created_at=1.0)]))
    first = asyncio.run(actor.dequeue("consumer-job"))
    assert _session_key_from_item(first) == "A"
    asyncio.run(actor.finalize_request("r1"))

    asyncio.run(_enqueue_many(actor, [_item("r2", domain="d", session_key="B", created_at=2.0)]))
    second = asyncio.run(asyncio.wait_for(actor.dequeue("consumer-job"), timeout=0.05))

    assert _session_key_from_item(second) == "B"


def test_issue_194_dequeue_assigns_monotonic_execution_serial_seq(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    asyncio.run(
        _enqueue_many(
            actor,
            [
                _item("r1", domain="d", session_key="A", created_at=1.0),
                _item("r2", domain="d", session_key="A", created_at=2.0),
            ],
        )
    )
    out = asyncio.run(_dequeue_many(actor, 2))
    seqs = [int((x.get("extra") or {}).get("execution_serial_seq")) for x in out]
    epochs = [str((x.get("extra") or {}).get("execution_serial_epoch") or "") for x in out]

    assert seqs == [1, 2]
    assert epochs[0]
    assert epochs == [epochs[0], epochs[0]]


def test_stale_dequeue_returns_stale_consumer_sentinel(monkeypatch):
    monkeypatch.setenv("MINT_API_WORK_QUEUE_DEQUEUE_POLL_S", "0.05")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()
    actor.set_active_job_id("consumer-job-new")

    async def _run() -> None:
        item = await actor.dequeue("consumer-job-old")
        assert item["op"] == "__stale_consumer__"
        extra = item.get("extra") or {}
        assert extra["consumer_job_id"] == "consumer-job-old"
        assert extra["active_job_id"] == "consumer-job-new"

    asyncio.run(_run())


def test_queue_stats_group_by_apikey_and_principal(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    item_a = _item("r1", domain="d", session_key="A", created_at=1.0)
    item_a["op"] = "sampling.asample"
    item_a["apikey_id"] = "bbbbbbbbbbbbbbbbbbbbbbbb"
    item_a["throttle_principal"] = "apikey:bbbbbbbbbbbbbbbbbbbbbbbb"
    item_b = _item("r2", domain="d", session_key="B", created_at=2.0)
    item_b["op"] = "sampling.asample"
    item_b["apikey_id"] = "bbbbbbbbbbbbbbbbbbbbbbbb"
    item_b["throttle_principal"] = "apikey:bbbbbbbbbbbbbbbbbbbbbbbb"

    asyncio.run(_enqueue_many(actor, [item_a, item_b]))

    stats = actor.stats()
    assert stats["by_apikey_id"] == {"bbbbbbbbbbbbbbbbbbbbbbbb": 2}
    assert stats["by_throttle_principal"] == {"apikey:bbbbbbbbbbbbbbbbbbbbbbbb": 2}


def test_issue_324_queue_throttles_per_apikey_not_per_user(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    monkeypatch.setattr(api_work_queue.server_config, "sampling_max_pending_asample_per_apikey", 1)
    actor = api_work_queue._get_or_create_ray_actor()

    base = _item("r1", domain="d", session_key="A", created_at=1.0)
    base["op"] = "sampling.asample"
    base["user_id"] = "aaaaaaaaaaaaaaaaaaaaaaaa"
    base["apikey_id"] = "bbbbbbbbbbbbbbbbbbbbbbbb"
    base["throttle_principal"] = "apikey:bbbbbbbbbbbbbbbbbbbbbbbb"

    other_key = dict(base)
    other_key["request_id"] = "r2"
    other_key["apikey_id"] = "cccccccccccccccccccccccc"
    other_key["throttle_principal"] = "apikey:cccccccccccccccccccccccc"

    asyncio.run(actor.enqueue(base))

    rejected = asyncio.run(actor.enqueue(dict(base, request_id="r1b")))
    assert rejected == {
        "ok": False,
        "detail": {
        "code": "sampling_principal_backpressure",
        "scope": "api_key",
        "limit": 1,
        "pending": 1,
        "message": "Sampling backpressure: principal budget exhausted",
        },
    }

    asyncio.run(actor.enqueue(other_key))
    stats = actor.stats()
    assert stats["by_apikey_id"] == {
        "bbbbbbbbbbbbbbbbbbbbbbbb": 1,
        "cccccccccccccccccccccccc": 1,
    }


def test_issue_324_finalize_request_releases_running_slot(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    monkeypatch.setattr(api_work_queue.server_config, "sampling_max_pending_asample_per_apikey", 1)
    actor = api_work_queue._get_or_create_ray_actor()

    item = _item("r1", domain="d", session_key="A", created_at=1.0)
    item["op"] = "sampling.asample"
    item["user_id"] = "aaaaaaaaaaaaaaaaaaaaaaaa"
    item["apikey_id"] = "bbbbbbbbbbbbbbbbbbbbbbbb"
    item["throttle_principal"] = "apikey:bbbbbbbbbbbbbbbbbbbbbbbb"

    asyncio.run(actor.enqueue(item))
    dequeued = asyncio.run(actor.dequeue("consumer-job"))
    assert dequeued["request_id"] == "r1"
    assert actor.stats()["by_apikey_id"] == {"bbbbbbbbbbbbbbbbbbbbbbbb": 1}
    asyncio.run(actor.finalize_request("r1"))
    assert actor.stats()["by_apikey_id"] == {}


def test_issue_324_consumer_handoff_releases_leased_slots(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    monkeypatch.setattr(api_work_queue.server_config, "sampling_max_pending_asample_per_apikey", 1)
    actor = api_work_queue._get_or_create_ray_actor()

    item = _item("r1", domain="d", session_key="A", created_at=1.0)
    item["op"] = "sampling.asample"
    item["user_id"] = "aaaaaaaaaaaaaaaaaaaaaaaa"
    item["apikey_id"] = "bbbbbbbbbbbbbbbbbbbbbbbb"
    item["throttle_principal"] = "apikey:bbbbbbbbbbbbbbbbbbbbbbbb"

    actor.set_active_job_id("consumer-old")
    asyncio.run(actor.enqueue(item))
    asyncio.run(actor.dequeue("consumer-old"))
    assert actor.stats()["by_apikey_id"] == {"bbbbbbbbbbbbbbbbbbbbbbbb": 1}

    actor.set_active_job_id("consumer-new")
    assert actor.stats()["by_apikey_id"] == {}

    accepted = asyncio.run(actor.enqueue(dict(item, request_id="r2")))
    assert accepted == {"ok": True}


def test_issue_324_scheduler_lease_survives_handoff_until_stale_request_reconciled(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "oldest")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "0")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    actor.set_active_job_id("consumer-old")
    asyncio.run(_enqueue_many(actor, [_item("r1", domain="d", session_key="A", created_at=1.0)]))
    first = asyncio.run(actor.dequeue("consumer-old"))
    assert first["request_id"] == "r1"

    asyncio.run(_enqueue_many(actor, [_item("r2", domain="d", session_key="B", created_at=2.0)]))
    actor.set_active_job_id("consumer-new")
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(actor.dequeue("consumer-new"), timeout=0.05))

    released_ids = actor.release_scheduler_leases_for_consumer("consumer-old")
    assert released_ids == ["r1"]
    second = asyncio.run(asyncio.wait_for(actor.dequeue("consumer-new"), timeout=0.05))

    assert second["request_id"] == "r2"


def test_issue_324_stale_finalize_after_handoff_cannot_restore_followup_bias(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "oldest")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "8")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "0")

    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._get_or_create_ray_actor()

    actor.set_active_job_id("consumer-old")
    asyncio.run(_enqueue_many(actor, [_item("r1", domain="d", session_key="A", created_at=1.0)]))
    first = asyncio.run(actor.dequeue("consumer-old"))
    assert first["request_id"] == "r1"

    asyncio.run(
        _enqueue_many(
            actor,
            [
                _item("r2", domain="d", session_key="B", created_at=2.0),
                _item("r3", domain="d", session_key="A", created_at=3.0),
            ],
        )
    )
    actor.set_active_job_id("consumer-new")
    assert actor.release_scheduler_leases_for_consumer("consumer-old") == ["r1"]
    asyncio.run(actor.finalize_request("r1"))

    second = asyncio.run(asyncio.wait_for(actor.dequeue("consumer-new"), timeout=0.05))
    assert second["request_id"] == "r2"


def test_issue_324_reconcile_stale_running_requests_fails_pending_leased_requests(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    client = api_work_queue.ApiWorkQueueClient()
    released: list[str] = []
    failed: list[tuple[str, str]] = []

    class _RemoteMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    actor = types.SimpleNamespace(
        release_stale_scheduler_leases=_RemoteMethod(
            lambda active_consumer_job_id: ["leased-r1"] if active_consumer_job_id == "consumer-new" else []
        )
    )

    monkeypatch.setattr(client, "_get_ray_actor", lambda: actor)
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")

    monkeypatch.setattr(
        future_store_module,
        "future_store",
        types.SimpleNamespace(
            fail_stale_running_requests=lambda active_consumer_job_id, error: [],
            fail=lambda request_id, error: failed.append((request_id, error)),
        ),
    )
    monkeypatch.setattr(
        capacity_manager_module,
        "capacity_manager",
        types.SimpleNamespace(release_all=lambda request_id: released.append(request_id)),
    )

    asyncio.run(client._reconcile_stale_running_requests("consumer-new"))

    assert failed == [("leased-r1", "api server restarted while request was dequeued before execution began")]
    assert released == ["leased-r1"]
