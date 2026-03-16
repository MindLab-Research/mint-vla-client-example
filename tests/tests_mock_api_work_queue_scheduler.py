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
    ray.get_actor = get_actor  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: {}  # type: ignore[attr-defined]
    ray.get_runtime_context = lambda: _Ctx()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray)


def _load_api_work_queue_module(monkeypatch):
    _install_ray_stub(monkeypatch)
    import tinker_server.backend.api_work_queue as api_work_queue

    return importlib.reload(api_work_queue)


def _item(
    request_id: str,
    *,
    domain: str,
    session_key: str | None = None,
    legacy_session_id: str | None = None,
    created_at: float = 0.0,
) -> dict:
    extra = {
        "scheduler_enabled": True,
        "scheduler_domain": domain,
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
        out.append(await actor.dequeue("consumer-job"))
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
    actor.finalize_request("r1")
    assert actor.stats()["by_apikey_id"] == {}
