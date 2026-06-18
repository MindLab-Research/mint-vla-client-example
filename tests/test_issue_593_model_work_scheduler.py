import asyncio
import time
from typing import Any

import pytest

from mint_server.backend.scheduling.model_work_scheduler import (
    CURRENT_CODE_IDENTITY,
    ModelWorkSchedulerConflictError,
    ModelWorkSchedulerCodeIdentityMismatchError,
    ModelWorkSchedulerClient,
    ModelWorkItem,
    _ModelWorkSchedulerActor,
    _model_work_scheduler_use_task_state_store_from_env,
    _ray_model_work_scheduler_actor_name,
)
from mint_server.backend.scheduling.scheduler_admission import AdmissionAccounting
from mint_server.backend.stores.task_state_store import TaskStateConflictError, TaskStateStore


@pytest.fixture(autouse=True)
def disable_scheduler_assignment_loop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0")


def _work(
    request_id: str,
    *,
    domain_key: str = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    affinity_group: str | None = "lora:session-a:generation:1",
    ordering_key: str | None = "session:session-a",
    token_cost: int = 1,
    throttle_principal: str | None = "apikey:key-a",
) -> dict:
    return {
        "request_id": request_id,
        "op": "sampling.asample",
        "request_json": b"{}",
        "user_id": "user-a",
        "apikey_id": "key-a",
        "throttle_principal": throttle_principal,
        "webhook_url": None,
        "extra": {},
        "created_at": 100.0,
        "domain_key": domain_key,
        "affinity_group": affinity_group,
        "ordering_key": ordering_key,
        "token_cost": token_cost,
    }


def _replica(
    replica_id: str,
    *,
    domain_key: str = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    consumer_id: str | None = None,
    generation: int = 10,
    status: str = "healthy",
) -> dict:
    return {
        "domain_key": domain_key,
        "replica_id": replica_id,
        "consumer_id": consumer_id or f"consumer-{replica_id}",
        "generation": generation,
        "status": status,
        "queue_id": f"{domain_key}::{replica_id}",
        "capacity": 4,
        "actor_name": f"actor-{replica_id}",
        "node_pins": ["10.0.0.1"],
        "updated_at": 101.0,
    }


class _MockTaskStateStoreClient:
    def __init__(self, store: TaskStateStore | None = None) -> None:
        self.store = store or TaskStateStore.in_memory()
        self.calls: list[tuple[str, dict]] = []
        self.blockers: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}
        self.overrides: dict[str, object] = {}

    def block_method(self, method: str) -> tuple[asyncio.Event, asyncio.Event]:
        entered = asyncio.Event()
        release = asyncio.Event()
        self.blockers[str(method)] = (entered, release)
        return entered, release

    def count(self, method: str) -> int:
        return sum(1 for name, _kwargs in self.calls if name == method)

    async def _call(self, method: str, **kwargs):
        method = str(method)
        store_method = {
            "acquire_owner": "acquire_scheduler_owner",
            "renew_owner": "renew_scheduler_owner",
        }.get(method, method)
        self.calls.append((method, dict(kwargs)))
        blocker = self.blockers.get(method) or self.blockers.get(store_method)
        if blocker is not None:
            entered, release = blocker
            entered.set()
            await release.wait()
        override = self.overrides.get(method, self.overrides.get(store_method))
        if override is not None:
            if isinstance(override, BaseException):
                raise override
            if callable(override):
                out = override(**kwargs)
                if asyncio.iscoroutine(out):
                    return await out
                return out
            return override
        return getattr(self.store, store_method)(**kwargs)

    async def async_ensure_ready(self, **_kwargs):
        return self.store.ping()

    async def async_ping(self, **_kwargs):
        return self.store.ping()

    async def async_acquire_owner(self, **kwargs):
        return await self._call("acquire_owner", **kwargs)

    async def async_renew_owner(self, **kwargs):
        return await self._call("renew_owner", **kwargs)

    async def async_create_task(self, **kwargs):
        return await self._call("create_task", **kwargs)

    async def async_assign_task(self, **kwargs):
        return await self._call("assign_task", **kwargs)

    async def async_claim_task(self, **kwargs):
        return await self._call("claim_task", **kwargs)

    async def async_renew_lease(self, **kwargs):
        return await self._call("renew_lease", **kwargs)

    async def async_begin_finalize(self, **kwargs):
        return await self._call("begin_finalize", **kwargs)

    async def async_commit_finalize_success(self, **kwargs):
        return await self._call("commit_finalize_success", **kwargs)

    async def async_commit_finalize_failure(self, **kwargs):
        return await self._call("commit_finalize_failure", **kwargs)

    async def async_complete_task_failure(self, **kwargs):
        return await self._call("complete_task_failure", **kwargs)

    async def async_requeue_task(self, **kwargs):
        return await self._call("requeue_task", **kwargs)

    async def async_forget_task(self, **kwargs):
        return await self._call("forget_task", **kwargs)

    async def async_get_task(self, request_id: str):
        return await self._call("get_task", request_id=str(request_id))

    async def async_list_active_tasks(self, **kwargs):
        return await self._call("list_active_tasks", **kwargs)

    async def async_wait_task_status_change(self, **kwargs):
        return await self._call("wait_task_status_change", **kwargs)

    async def async_update_task_metadata(self, **kwargs):
        return await self._call("update_task_metadata", **kwargs)

    def close(self) -> None:
        self.store.close()


class _SchedulerMockHarness:
    def __init__(self, *, owner_id: str = "scheduler-mock") -> None:
        self.task_state = _MockTaskStateStoreClient()
        self.actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id=owner_id,
        )

    async def claim_one(self, request_id: str = "req-mock") -> dict:
        await self.actor.sync_replicas([_replica("replica-0")])
        assert (await self.actor.append(_work(request_id), assign=True)).ok is True
        claimed = await self.actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        return claimed.leases[0]

    def close(self) -> None:
        self.task_state.close()


def test_scheduler_append_assign_uses_typed_assignment_under_wire_wrapper() -> None:
    class _WireAssignScheduler(_ModelWorkSchedulerActor):
        async def assign_pending(self, **kwargs: Any) -> Any:
            out = await super().assign_pending(**kwargs)
            return out.to_wire()

        async def _assign_pending_unlocked(self, **kwargs: Any) -> Any:
            out = await super()._assign_pending_unlocked(**kwargs)
            return out.to_wire()

    async def _run() -> None:
        actor = _WireAssignScheduler()
        try:
            await actor.sync_replicas([_replica("replica-0")])

            appended = await actor.append(_work("req-wire-assign"), assign=True)

            assert appended.ok is True
            assert appended.assigned is not None
            assert appended.assigned["ok"] is True
            assert appended.assigned["assigned"] == 1
            contains = await actor.contains_request(request_id="req-wire-assign")
            assert contains.location == "assigned"
            claimed = await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-0",
                consumer_generation=10,
                max_items=1,
                lease_ttl_s=30.0,
            )
            assert claimed.ok is True
            assert len(claimed.leases) == 1
        finally:
            await actor.shutdown_background_loops()

    asyncio.run(_run())


def test_scheduler_owner_heartbeat_runs_without_request_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_OWNER_HEARTBEAT_INTERVAL_S", "0.01")

    async def _run() -> None:
        store = _MockTaskStateStoreClient()
        actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=store,
            owner_id="scheduler-heartbeat",
        )
        try:
            assert actor.stats()["owner_heartbeat_running"] is True
            deadline = time.time() + 1.0
            while store.count("renew_owner") < 1 and time.time() < deadline:
                await asyncio.sleep(0.01)
            assert store.count("acquire_owner") >= 1
            assert store.count("renew_owner") >= 1
            assert actor.stats()["owner_heartbeat_running"] is True
        finally:
            await actor.shutdown_background_loops()
            store.close()

    asyncio.run(_run())


def test_scheduler_background_loops_are_taskgroup_managed_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0.01")
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_REAPER_INTERVAL_S", "0.01")
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_OWNER_HEARTBEAT_INTERVAL_S", "0.01")

    async def _run() -> None:
        store = _MockTaskStateStoreClient()
        actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=store,
            owner_id="scheduler-background-taskgroup",
        )
        try:
            stats = actor.stats()
            assert stats["background_loop_manager_running"] is True
            assert stats["assignment_loop_running"] is True
            assert stats["owner_heartbeat_running"] is True
            assert stats["reaper_loop_running"] is True
            assert stats["background_loop_names"] == [
                "assignment",
                "owner_heartbeat",
                "reaper",
            ]

            await actor.shutdown_background_loops()

            stopped = actor.stats()
            assert stopped["background_loop_manager_running"] is False
            assert stopped["assignment_loop_running"] is False
            assert stopped["owner_heartbeat_running"] is False
            assert stopped["reaper_loop_running"] is False
            assert stopped["background_loop_names"] == []
        finally:
            await actor.shutdown_background_loops()
            store.close()

    asyncio.run(_run())


def test_scheduler_background_loops_defer_without_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0.01")
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_REAPER_INTERVAL_S", "0.01")
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_OWNER_HEARTBEAT_INTERVAL_S", "0.01")

    store = _MockTaskStateStoreClient()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-background-deferred",
    )
    try:
        stats = actor.stats()
        assert stats["background_loop_manager_running"] is False
        assert set(stats["background_loop_start_deferred"]) == {
            "assignment",
            "owner_heartbeat",
            "reaper",
        }
    finally:
        store.close()


def test_scheduler_assigns_to_registered_replica_queue() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1"))).ok is True
        assert (await actor.sync_replicas([_replica("replica-0")])).replicas == 1

        stats = actor.stats()
        queue = stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"]
        assert queue["depth"] == 1
        assert stats["backlog_depth"] == 0

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-1"]
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 0

    asyncio.run(_run())


def test_scheduler_claims_first_item_when_it_exceeds_token_budget() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-expensive", token_cost=100))
        await actor.append(_work("req-next", token_cost=1))
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            token_budget=50,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-expensive"]
        assert claimed.remaining_queue_depth == 1

    asyncio.run(_run())


def test_scheduler_same_ordering_key_is_claimed_serially() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-serial-1", ordering_key="session:serial"))
        await actor.append(_work("req-serial-2", ordering_key="session:serial"))
        await actor.assign_pending()

        first = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )
        blocked = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in first.leases] == ["req-serial-1"]
        assert blocked.leases == []

    asyncio.run(_run())


def test_scheduler_multi_claim_for_training_domains_stays_on_same_affinity() -> None:
    domain_key = "bumblebee:Qwen/Qwen3-30B-A3B-Instruct-2507"
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(
            _work(
                "req-a1",
                domain_key=domain_key,
                affinity_group="training_session:a",
                ordering_key="training_session:a:step-1",
            )
        )
        await actor.append(
            _work(
                "req-b1",
                domain_key=domain_key,
                affinity_group="training_session:b",
                ordering_key="training_session:b:step-1",
            )
        )
        await actor.append(
            _work(
                "req-a2",
                domain_key=domain_key,
                affinity_group="training_session:a",
                ordering_key="training_session:a:step-2",
            )
        )
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-a1", "req-a2"]
        assert claimed.remaining_queue_depth == 1

    asyncio.run(_run())


def test_scheduler_same_affinity_domains_can_be_disabled_by_constructor() -> None:
    domain_key = "bumblebee:Qwen/Qwen3-30B-A3B-Instruct-2507"
    actor = _ModelWorkSchedulerActor(same_affinity_multi_claim_domains=())

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(
            _work(
                "req-a1",
                domain_key=domain_key,
                affinity_group="training_session:a",
                ordering_key="training_session:a",
            )
        )
        await actor.append(
            _work(
                "req-b1",
                domain_key=domain_key,
                affinity_group="training_session:b",
                ordering_key="training_session:b",
            )
        )
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=2,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-a1", "req-b1"]

    asyncio.run(_run())


def test_scheduler_same_affinity_domains_can_be_overridden_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_CLAIM_SAME_AFFINITY_DOMAINS", "custom:")
    domain_key = "custom:model-a"
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(
            _work("req-a1", domain_key=domain_key, affinity_group="group-a", ordering_key="group-a:1")
        )
        await actor.append(
            _work("req-b1", domain_key=domain_key, affinity_group="group-b", ordering_key="group-b:1")
        )
        await actor.append(
            _work("req-a2", domain_key=domain_key, affinity_group="group-a", ordering_key="group-a:2")
        )
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-a1", "req-a2"]
        assert claimed.remaining_queue_depth == 1

    asyncio.run(_run())


def test_scheduler_default_actor_name_uses_mint_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME", raising=False)

    assert _ray_model_work_scheduler_actor_name() == "mint_model_work_scheduler"


def test_scheduler_snapshots_include_code_identity() -> None:
    actor = _ModelWorkSchedulerActor()

    assert actor.ping()["code_identity"] == CURRENT_CODE_IDENTITY
    assert actor.stats()["code_identity"] == CURRENT_CODE_IDENTITY


def test_scheduler_actor_task_state_store_env_gate_defaults_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_MODEL_WORK_SCHEDULER_USE_TASK_STATE_STORE", raising=False)

    assert _model_work_scheduler_use_task_state_store_from_env() is True


def test_scheduler_actor_task_state_store_env_gate_can_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("0", "false", "False", "no", "off", "disabled"):
        monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_USE_TASK_STATE_STORE", value)
        assert _model_work_scheduler_use_task_state_store_from_env() is False


def test_scheduler_client_rejects_stale_code_identity() -> None:
    client = ModelWorkSchedulerClient()

    client._validate_code_identity({"code_identity": CURRENT_CODE_IDENTITY})
    with pytest.raises(ModelWorkSchedulerCodeIdentityMismatchError):
        client._validate_code_identity({"code_identity": "stale-scheduler-code"})


def test_scheduler_client_forwards_sync_replicas_hydration_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RemoteMethod:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict]] = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True}

    class _Actor:
        def __init__(self) -> None:
            self.sync_replicas = _RemoteMethod()

    actor = _Actor()
    client = ModelWorkSchedulerClient()

    async def _get_actor(*, create_if_missing: bool = False):
        assert create_if_missing is False
        return actor

    async def _await_ref(ref, *, timeout_s: float):
        assert timeout_s == 3.0
        return ref

    client._get_ray_actor_async = _get_actor  # type: ignore[method-assign]
    client._await_ray_ref = _await_ref  # type: ignore[method-assign]

    async def _run() -> None:
        out = await client.sync_replicas(
            [_replica("replica-0")],
            hydrate_task_state=False,
            timeout_s=3.0,
        )

        assert out.ok is True

    asyncio.run(_run())

    assert actor.sync_replicas.calls == [
        (
            ([_replica("replica-0")],),
            {"hydrate_task_state": False},
        )
    ]


def test_issue_638_scheduler_registers_actor_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.observability.logging_context as logging_context

    calls = {"count": 0}
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: calls.__setitem__("count", calls["count"] + 1))

    _ModelWorkSchedulerActor()

    assert calls["count"] == 1


def test_issue_638_scheduler_registers_otel_gauges(monkeypatch: pytest.MonkeyPatch) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.observability.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    _ModelWorkSchedulerActor()

    assert "mint_model_work_scheduler_depth" in gauges
    assert "mint_model_work_scheduler_appended_total" in gauges
    assert "mint_model_work_scheduler_domain_backlog_depth" in gauges
    assert "mint_model_work_scheduler_replica_queue_depth" in gauges
    assert "mint_model_work_scheduler_leases" in gauges
    assert "mint_model_load_pct" in gauges
    assert "mint_model_pending_requests" in gauges
    assert "mint_sampling_inflight_by_domain" in gauges
    assert "mint_sampling_inflight_principal_domain_max" in gauges
    assert "mint_sampling_admission_would_reject_total" in gauges
    assert "mint_sampling_admission_reject_total" in gauges


def test_issue_638_scheduler_otel_callbacks_emit_existing_dashboard_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.observability.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setenv("MINT_DEPLOYMENT_ENV", "prod")
    monkeypatch.setenv("MINT_CLUSTER_ID", "volcano")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    actor = _ModelWorkSchedulerActor()

    async def _setup() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-1"), assign=True)
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert len(claimed.leases) == 1

    asyncio.run(_setup())

    depth_obs = gauges["mint_model_work_scheduler_depth"][0](None)
    assert depth_obs[0].value == 1.0
    assert depth_obs[0].attributes["deployment.env"] == "prod"
    assert depth_obs[0].attributes["mint.cluster_id"] == "volcano"

    queue_obs = gauges["mint_model_work_scheduler_replica_queue_depth"][0](None)
    assert len(queue_obs) == 1
    assert queue_obs[0].value == 0.0
    assert queue_obs[0].attributes["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert queue_obs[0].attributes["replica_id"] == "replica-0"
    assert queue_obs[0].attributes["queue_id"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
    assert queue_obs[0].attributes["status"] == "healthy"

    lease_obs = gauges["mint_model_work_scheduler_leases"][0](None)
    assert lease_obs[0].value == 1.0

    inflight_obs = gauges["mint_model_inflight_workers"][0](None)
    assert inflight_obs[0].value == 1.0
    assert inflight_obs[0].attributes["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert inflight_obs[0].attributes["workload"] == "sample"

    sampling_inflight_obs = gauges["mint_sampling_inflight_by_domain"][0](None)
    assert sampling_inflight_obs[0].value == 1.0
    assert sampling_inflight_obs[0].attributes["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"

    principal_max_obs = gauges["mint_sampling_inflight_principal_domain_max"][0](None)
    assert principal_max_obs[0].value == 1.0
    assert principal_max_obs[0].attributes["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_638_scheduler_otel_callbacks_do_not_start_assignment_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.observability.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    actor = _ModelWorkSchedulerActor()

    def _unexpected_start():
        raise AssertionError("OTel callback must not start assignment loop")

    monkeypatch.setattr(actor, "_ensure_assignment_loop_started", _unexpected_start)

    assert gauges["mint_model_work_scheduler_depth"][0](None)[0].value == 0.0


def test_model_work_scheduler_contains_request_uses_lookup_concurrency_group(monkeypatch) -> None:
    import mint_server.backend.scheduling.model_work_scheduler as module
    import ray

    captured: dict[str, Any] = {"methods": {}}

    class _OptionsProxy:
        def __init__(self, cls):
            self._cls = cls

        def remote(self, **kwargs):
            captured["init_kwargs"] = kwargs
            actor = self._cls(**kwargs)

            class _RemoteMethod:
                def __init__(self, fn):
                    self._fn = fn

                def remote(self, **method_kwargs):
                    return self._fn(**method_kwargs)

            actor.ping = _RemoteMethod(actor.ping)
            actor.contains_request = _RemoteMethod(actor.contains_request)
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
        def _decorator(fn):
            captured["methods"][fn.__name__] = method_kwargs
            return fn

        return _decorator

    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(ray, "method", _fake_method)
    def _fake_actor_runtime_env(**kwargs):
        captured["runtime_env_kwargs"] = kwargs
        return {
            "env_vars": {
                "PYTHONPATH": kwargs.get("pythonpath", ""),
                **dict(kwargs.get("extra") or {}),
            }
        }

    monkeypatch.setattr(module, "actor_runtime_env", _fake_actor_runtime_env)
    monkeypatch.setattr(module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_model_work_scheduler_actor_resources", lambda: None)
    monkeypatch.setattr(module, "_await_ray_ref_sync", lambda ref, *, timeout_s=None: ref)

    module._create_ray_actor(require_ready=True)

    assert captured["remote_kwargs"]["concurrency_groups"] == {"health": 8, "lookup": 16}
    assert captured["runtime_env_kwargs"]["extra"]["MINT_GIT_SHA"] == CURRENT_CODE_IDENTITY
    assert captured["runtime_env_kwargs"]["include_ray_attach_hints"] is False
    env_vars = captured["options"]["runtime_env"]["env_vars"]
    assert "MINT_GIT_SHA" in env_vars
    assert env_vars["MINT_GIT_SHA"] == CURRENT_CODE_IDENTITY
    assert captured["methods"]["ping"] == {"concurrency_group": "health"}
    assert captured["methods"]["contains_request"] == {"concurrency_group": "lookup"}


def test_scheduler_append_can_assign_immediately() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert await actor.is_empty() is True
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(_work("req-1"), assign=True, assign_max_items=1)

        assert out.ok is True
        assert await actor.is_empty() is False
        assert out.scheduler_instance_id
        contains = await actor.contains_request(request_id="req-1")
        assert contains.present is True
        assert contains.location == "assigned"
        assert contains.scheduler_instance_id == out.scheduler_instance_id
        assert out.assigned is not None
        assert out.assigned["assigned"] == 1
        assert actor.stats()["backlog_depth"] == 0
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 1

    asyncio.run(_run())


def test_admission_accounting_snapshot_preserves_scheduler_stats_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN", "10")
    accounting = AdmissionAccounting()
    item = ModelWorkItem.from_dict(
        _work(
            "req-token-1",
            token_cost=7,
            throttle_principal="apikey:key-a",
        )
    )
    second = ModelWorkItem.from_dict(
        _work(
            "req-token-2",
            token_cost=4,
            throttle_principal="apikey:key-a",
        )
    )

    accounting.track_locked(item)
    decision = accounting.limit_decision_locked(second)

    assert decision["ok"] is True
    assert decision["would_reject"] is True
    assert decision["reason"] == "principal_domain_token_budget_exceeded"
    domain = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert accounting.inflight_snapshot() == {
        "mode": "observe",
        "per_principal_domain_limit": 1024,
        "per_domain_limit": 10240,
        "per_principal_domain_token_limit": 10,
        "per_domain_token_limit": 0,
        "by_domain": {domain: 1},
        "principal_domain_max_by_domain": {domain: 1},
        "active_principals_by_domain": {domain: 1},
        "tokens_by_domain": {domain: 7},
        "principal_domain_token_max_by_domain": {domain: 7},
    }
    assert accounting.admission_counters_snapshot() == {
        "would_reject": [
            {
                "domain_key": domain,
                "reason": "principal_domain_token_budget_exceeded",
                "count": 1,
            }
        ],
        "reject": [],
    }

    accounting.untrack_locked("req-token-1")

    released = accounting.inflight_snapshot()
    assert released["by_domain"] == {}
    assert released["tokens_by_domain"] == {}


def test_sampling_token_budget_admission_enforce_rejects_principal_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "100")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN", "10")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-token-1", token_cost=7))).ok is True
        rejected = await actor.append(_work("req-token-2", token_cost=4))

        assert rejected.ok is False
        assert rejected.reason == "principal_domain_token_budget_exceeded"
        assert rejected.extra["current"] == 11
        assert rejected.extra["limit"] == 10
        domain = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert actor.stats()["sampling_inflight"]["tokens_by_domain"][domain] == 7

    asyncio.run(_run())


def test_contains_request_does_not_hydrate_task_state_store() -> None:
    actor = _ModelWorkSchedulerActor(use_task_state_store=True)

    async def _unexpected_hydrate():
        raise AssertionError("contains_request must remain a lightweight memory lookup")

    actor._ensure_task_state_ready = _unexpected_hydrate  # type: ignore[method-assign]

    async def _run() -> None:
        contains = await actor.contains_request(
            request_id="req-missing",
            hydrate_task_state=False,
        )

        assert contains.ok is True
        assert contains.present is False
        assert contains.location is None

    asyncio.run(_run())


def test_append_does_not_hydrate_task_state_store_before_enqueue() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _unexpected_hydrate():
        raise AssertionError("append must not scan active task-state before enqueueing new work")

    actor._ensure_task_state_ready = _unexpected_hydrate  # type: ignore[method-assign]

    async def _run() -> None:
        out = await actor.append(_work("req-append-no-hydrate"))

        assert out.ok is True
        assert out.request_id == "req-append-no-hydrate"
        assert store.get_task("req-append-no-hydrate")["request_id"] == "req-append-no-hydrate"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_sync_replicas_can_skip_task_state_store_hydration() -> None:
    actor = _ModelWorkSchedulerActor(use_task_state_store=True)

    async def _noop_owner():
        return 1

    async def _unexpected_hydrate():
        raise AssertionError("sync_replicas must not scan active task-state before syncing replicas")

    actor._ensure_task_state_owner = _noop_owner  # type: ignore[method-assign]
    actor._ensure_task_state_ready = _unexpected_hydrate  # type: ignore[method-assign]

    async def _run() -> None:
        out = await actor.sync_replicas(
            [_replica("replica-0")],
            hydrate_task_state=False,
        )

        assert out.ok is True
        assert out.replicas == 1

    asyncio.run(_run())


def test_sampling_inflight_admission_observe_records_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1"))).ok is True
        second = await actor.append(_work("req-2"))

        assert second.ok is True
        assert second.extra["sampling_inflight_admission"]["would_reject"] is True
        assert second.extra["sampling_inflight_admission"]["reason"] == "principal_domain_inflight_limit_exceeded"
        stats = actor.stats()
        domain = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert stats["sampling_inflight"]["by_domain"][domain] == 2
        assert stats["sampling_inflight"]["principal_domain_max_by_domain"][domain] == 2
        assert stats["sampling_admission_counters"]["would_reject"] == [
            {
                "domain_key": domain,
                "reason": "principal_domain_inflight_limit_exceeded",
                "count": 1,
            }
        ]

    asyncio.run(_run())


def test_scheduler_private_probe_fields_delegate_to_collaborators() -> None:
    actor = _ModelWorkSchedulerActor(use_task_state_store=True)

    assert actor._background_loop_tasks is actor._loops.tasks
    assert actor._background_loop_start_deferred is actor._loops.start_deferred
    assert actor._assignment_loop_task is actor._loops.assignment_task
    assert actor._owner_heartbeat_task is actor._loops.owner_heartbeat_task
    assert actor._reaper_loop_task is actor._loops.reaper_task
    assert actor._background_loops_shutdown is actor._loops.shutdown

    assert actor._sampling_inflight_by_domain is actor._admission.inflight_by_domain
    assert (
        actor._sampling_inflight_by_principal_domain
        is actor._admission.inflight_by_principal_domain
    )
    assert actor._sampling_inflight_tokens_by_domain is actor._admission.inflight_tokens_by_domain
    assert (
        actor._sampling_inflight_tokens_by_principal_domain
        is actor._admission.inflight_tokens_by_principal_domain
    )
    assert (
        actor._sampling_principal_domain_by_request_id
        is actor._admission.principal_domain_by_request_id
    )
    assert actor._sampling_token_cost_by_request_id is actor._admission.token_cost_by_request_id
    assert actor._sampling_admission_would_reject is actor._admission.would_reject
    assert actor._sampling_admission_reject is actor._admission.reject

    actor._sampling_inflight_by_domain = {"domain-a": 2}
    assert actor._admission.inflight_by_domain == {"domain-a": 2}


def test_sampling_inflight_admission_enforce_rejects_principal_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1"))).ok is True
        rejected = await actor.append(_work("req-2"))

        assert rejected.ok is False
        assert rejected.reason == "principal_domain_inflight_limit_exceeded"
        assert rejected.extra["current"] == 1
        assert rejected.extra["limit"] == 1
        assert actor.stats()["backlog_depth"] == 1

    asyncio.run(_run())


def test_sampling_inflight_admission_enforce_rejects_domain_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "100")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1", throttle_principal="apikey:key-a"))).ok is True
        rejected = await actor.append(_work("req-2", throttle_principal="apikey:key-b"))

        assert rejected.ok is False
        assert rejected.reason == "domain_inflight_limit_exceeded"
        assert rejected.extra["current"] == 1
        assert rejected.extra["limit"] == 1
        domain = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert actor.stats()["sampling_inflight"]["by_domain"][domain] == 1

    asyncio.run(_run())


def test_sampling_inflight_admission_releases_count_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-1"), assign=True)).ok is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease_id = str(claimed.leases[0]["lease_id"])
        lease = claimed.leases[0]
        finalizing = await actor.begin_finalize_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/result-req-1.json",
        )
        assert finalizing.ok is True
        completed = await actor.finish_lease_success(
            request_id=str(lease["item"]["request_id"]),
            lease_id=lease_id,
            attempt_id=str(lease["attempt_id"]),
            scheduler_epoch=int(lease["scheduler_epoch"] or 0),
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            result_path="/tmp/result-req-1.json",
        )
        assert completed.ok is True
        assert (await actor.append(_work("req-2"), assign=True)).ok is True

    asyncio.run(_run())


def test_scheduler_assignment_loop_moves_backlog_to_replica_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0.01")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-loop"))).ok is True
        await actor.sync_replicas([_replica("replica-0")])
        for _ in range(20):
            stats = actor.stats()
            queue = stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"]
            if stats["backlog_depth"] == 0 and queue["depth"] == 1:
                return
            await asyncio.sleep(0.02)
        stats = actor.stats()
        raise AssertionError(f"assignment loop did not drain backlog: {stats!r}")

    asyncio.run(_run())


def test_scheduler_cancel_request_removes_assigned_work() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-1"), assign=True)

        out = await actor.cancel_request(request_id="req-1", reason="test_cancel")

        assert out.cancelled is True
        assert (await actor.contains_request(request_id="req-1")).present is False
        assert await actor.is_empty() is True

    asyncio.run(_run())


def test_replica_can_claim_only_own_queue_and_generation() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.append(_work("req-1"))
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.assign_pending()

        with pytest.raises(ModelWorkSchedulerConflictError, match="consumer_id mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-1",
                consumer_generation=10,
            )
        with pytest.raises(ModelWorkSchedulerConflictError, match="generation mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-0",
                consumer_generation=9,
            )

    asyncio.run(_run())


def test_scheduler_sync_reassigns_requeued_work_to_new_consumer_generation() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas(
            [_replica("replica-0", consumer_id="consumer-old", generation=1)]
        )
        assert (await actor.append(_work("req-recycle"), assign=True)).ok is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-old",
            consumer_generation=1,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-recycle"]

        await actor.sync_replicas(
            [_replica("replica-0", consumer_id="consumer-new", generation=2)]
        )
        stats = actor.stats()
        queue = stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"]
        assert queue["consumer_id"] == "consumer-new"
        assert queue["generation"] == 2
        assert queue["depth"] == 1

        with pytest.raises(ModelWorkSchedulerConflictError, match="consumer_id mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-old",
                consumer_generation=1,
            )

        claimed_after_recycle = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-new",
            consumer_generation=2,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed_after_recycle.leases] == [
            "req-recycle"
        ]
        assert claimed_after_recycle.leases[0]["consumer_id"] == "consumer-new"
        assert claimed_after_recycle.leases[0]["consumer_generation"] == 2

    asyncio.run(_run())


def test_affinity_sticks_to_same_healthy_replica() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.append(_work("req-1", affinity_group="lora:a"))
        await actor.append(_work("req-2", affinity_group="lora:a"))
        await actor.append(_work("req-3", affinity_group="lora:b"))
        await actor.assign_pending()

        stats = actor.stats()["replica_queues"]
        depths = {
            queue["replica_id"]: queue["depth"]
            for queue in stats.values()
        }
        assert depths["replica-0"] == 2
        assert depths["replica-1"] == 1

    asyncio.run(_run())


def test_assignment_accounts_for_active_leases() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0"), _replica("replica-1")])
        await actor.append(_work("req-active", affinity_group="lora:a"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
        )
        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-active"]

        await actor.append(_work("req-next", affinity_group="lora:b"))
        await actor.assign_pending()

        stats = actor.stats()["replica_queues"]
        depths = {queue["replica_id"]: queue["depth"] for queue in stats.values()}
        assert depths["replica-0"] == 0
        assert depths["replica-1"] == 1

    asyncio.run(_run())


def test_unhealthy_replica_requeues_assigned_and_leased_work() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-assigned"))
        await actor.append(_work("req-leased"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
        )
        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-assigned"]

        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])
        assert sync.requeued == 2
        stats = actor.stats()
        assert stats["backlog_depth"] == 2
        assert stats["leases"] == []
        assert stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"][
            "status"
        ] == "unhealthy"

    asyncio.run(_run())


def test_lease_finish_fail_and_expiry() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _claim_one(request_id: str) -> dict[str, Any]:
        await actor.append(_work(request_id))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=1.0,
        )
        return claimed.leases[0]

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])

        success_lease = await _claim_one("req-complete")
        lease_id = success_lease["lease_id"]
        finalizing = await actor.begin_finalize_lease(
            lease_id=str(lease_id),
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/result-req-complete.json",
        )
        assert finalizing.ok is True
        complete = await actor.finish_lease_success(
            request_id=str(success_lease["item"]["request_id"]),
            lease_id=str(lease_id),
            attempt_id=str(success_lease["attempt_id"]),
            scheduler_epoch=int(success_lease["scheduler_epoch"] or 0),
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            result_path="/tmp/result-req-complete.json",
        )
        assert complete.ok is True
        assert actor.stats()["counters"]["completed"] == 1

        fail_lease = await _claim_one("req-fail")
        failed = await actor.fail_lease(
            lease_id=str(fail_lease["lease_id"]),
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=False,
            reason="runtime_error",
        )
        assert failed.ok is True and failed.request_id == "req-fail" and failed.requeued is False
        assert actor.stats()["counters"]["failed"] == 1

        expire_lease = await _claim_one("req-expire")
        assert (await actor.expire_leases(now=time.time() + 999.0)).expired == 1
        assert actor.stats()["backlog_depth"] == 1
        assert str(expire_lease["lease_id"]) not in {
            str(lease["lease_id"]) for lease in actor.stats()["leases"]
        }

    asyncio.run(_run())


def test_validate_lease_rejects_requeued_or_stale_leases() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-validate"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )
        lease_id = str(claimed.leases[0]["lease_id"])

        validated = await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )
        assert validated.ok is True
        failed = await actor.fail_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=True,
            reason="test_requeue",
        )
        assert failed.requeued is True
        rejected = await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )
        assert rejected.ok is False and rejected.reason == "unknown_lease"

    asyncio.run(_run())


def test_finalizing_lease_survives_replica_sync_until_finalize_ttl() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        await actor.append(_work("req-finalizing"))
        await actor.assign_pending()
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            lease_ttl_s=1.0,
        )
        lease_id = str(claimed.leases[0]["lease_id"])

        finalizing = await actor.begin_finalize_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=60.0,
        )
        assert finalizing.ok is True
        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])

        assert sync.requeued == 0
        assert actor.stats()["leases"][0]["lease_id"] == lease_id

    asyncio.run(_run())


def test_scheduler_persists_append_assign_claim_and_begin_finalize_to_task_state_store() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(
            _work(
                "req-persisted",
                affinity_group="lora:persisted",
            ),
            assign=True,
        )
        assert out.ok is True

        record = store.get_task("req-persisted")
        assert record["status"] == "assigned"
        assert record["subqueue_id"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        assert record["scheduler_epoch"] == 1
        assert record["metadata"]["model_work_scheduler_instance_id"] == out.scheduler_instance_id

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed.leases[0]
        record = store.get_task("req-persisted")
        assert record["status"] == "leased"
        assert record["lease_id"] == lease["lease_id"]
        assert record["attempt_id"] == lease["attempt_id"]
        assert record["runtime_generation"] == 10
        claimed_expires_at = float(record["lease_expires_at"])

        renewed = await actor.renew_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            lease_ttl_s=120.0,
        )
        assert renewed.ok is True
        record = store.get_task("req-persisted")
        assert record["status"] == "leased"
        assert float(record["lease_expires_at"]) > claimed_expires_at
        assert renewed.lease is not None
        assert renewed.lease["lease_expires_at"] == record["lease_expires_at"]

        finalizing = await actor.begin_finalize_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/req-persisted.json",
        )
        assert finalizing.ok is True
        record = store.get_task("req-persisted")
        assert record["status"] == "finalizing"
        assert record["staged_payload_path"] == "/tmp/req-persisted.json"
        assert actor.stats()["task_state_store_enabled"] is True
        assert actor.stats()["scheduler_epoch"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_renew_lease_rejects_durable_terminal_task_state() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-terminal-renew"), assign=True)).ok is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed.leases[0]
        store.complete_task_failure(
            request_id="req-terminal-renew",
            error="external terminalization",
        )

        renewed = await actor.renew_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            lease_ttl_s=30.0,
        )

        assert renewed.ok is False and renewed.reason == "terminal"
        assert actor.stats()["leases"] == []

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_renew_lease_task_state_rpc_does_not_hold_scheduler_lock() -> None:
    harness = _SchedulerMockHarness()

    async def _run() -> None:
        lease = await harness.claim_one("req-renew-lock")
        entered, release = harness.task_state.block_method("renew_lease")
        renew_task = asyncio.create_task(
            harness.actor.renew_lease(
                lease_id=lease["lease_id"],
                consumer_id="consumer-replica-0",
                consumer_generation=10,
                lease_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        validated = await asyncio.wait_for(
            harness.actor.validate_lease(
                lease_id=lease["lease_id"],
                consumer_id="consumer-replica-0",
                consumer_generation=10,
            ),
            timeout=0.2,
        )
        assert validated.ok is True

        release.set()
        renewed = await asyncio.wait_for(renew_task, timeout=1.0)
        assert renewed.ok is True

    try:
        asyncio.run(_run())
    finally:
        harness.close()


def test_scheduler_accepts_pre_registered_pending_task_state_store_future() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )
    work = _work("req-pre-registered")

    async def _run() -> None:
        store.create_task(
            request_id=work["request_id"],
            op=work["op"],
            domain_key=work["domain_key"],
            request_json=work["request_json"],
            metadata={
                "affinity_group": work["affinity_group"],
                "ordering_key": work["ordering_key"],
            },
        )
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(work, assign=True)

        assert out.ok is True
        assert out.idempotent is True
        assert store.get_task("req-pre-registered")["status"] == "assigned"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_rolls_back_new_task_when_assign_fails_after_create() -> None:
    store = TaskStateStore.in_memory()
    task_state = _MockTaskStateStoreClient(store)
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=task_state,
        owner_id="scheduler-test",
    )
    task_state.overrides["assign_task"] = RuntimeError("assign failed after create")

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        with pytest.raises(RuntimeError, match="assign failed after create"):
            await actor.append(_work("req-assign-fails"), assign=True)

        with pytest.raises(KeyError):
            store.get_task("req-assign-fails")
        assert (await actor.contains_request(request_id="req-assign-fails")).present is False
        assert actor.stats()["backlog_depth"] == 0

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_hydrates_active_task_state_after_restart() -> None:
    store = TaskStateStore.in_memory()
    actor_a = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )
    actor_b = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor_a.sync_replicas([_replica("replica-0")])
        assert (await actor_a.append(_work("req-restart"), assign=True)).ok is True
        assert store.get_task("req-restart")["status"] == "assigned"

        await actor_b.sync_replicas([_replica("replica-0")])
        contains = await actor_b.contains_request(request_id="req-restart")
        assert contains.present is True
        assert contains.location == "assigned"
        claimed = await actor_b.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-restart"]
        assert store.get_task("req-restart")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_hydrates_sampling_inflight_counts_from_task_state_store() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )
    domain = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"

    async def _run() -> None:
        for request_id, principal in (
            ("req-hydrate-a", "apikey:key-a"),
            ("req-hydrate-b", "apikey:key-a"),
            ("req-hydrate-c", "apikey:key-b"),
        ):
            store.create_task(
                request_id=request_id,
                op="sampling.asample",
                domain_key=domain,
                request_json=b"{}",
                metadata={
                    "op": "sampling.asample",
                    "throttle_principal": principal,
                    "domain_key": domain,
                },
            )

        await actor._ensure_task_state_ready()
        contains = await actor.contains_request(
            request_id="req-hydrate-a",
            hydrate_task_state=False,
        )
        assert contains.present is True
        stats = actor.stats()
        assert stats["sampling_inflight"]["by_domain"][domain] == 3
        assert stats["sampling_inflight"]["principal_domain_max_by_domain"][domain] == 2
        assert stats["sampling_inflight"]["active_principals_by_domain"][domain] == 2

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_persists_requeue_before_reclaim() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-requeue"), assign=True)).ok is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed.leases[0]
        failed = await actor.fail_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            reason="executor_failed",
            requeue=True,
        )
        assert failed.requeued is True
        assert store.get_task("req-requeue")["status"] == "pending"

        await actor.assign_pending(max_items=1)
        reclaimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [item["item"]["request_id"] for item in reclaimed.leases] == ["req-requeue"]
        assert store.get_task("req-requeue")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_drops_terminal_stale_head_and_claims_next() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-stale"), assign=True)).ok is True
        assert (await actor.append(_work("req-valid"), assign=True)).ok is True
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 2

        store.complete_task_failure(
            request_id="req-stale",
            error="client_abandoned",
            result_path=None,
        )
        store.mark_task_retrieved(request_id="req-stale")

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == ["req-valid"]
        assert store.get_task("req-stale")["status"] == "retrieved"
        assert store.get_task("req-valid")["status"] == "leased"
        assert (await actor.contains_request(request_id="req-stale")).present is False
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 0
        assert actor.stats()["counters"]["stale_dropped"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_requeues_pending_stale_head() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-pending-stale"), assign=True)).ok is True
        store.requeue_task(
            request_id="req-pending-stale",
            scheduler_epoch=1,
            reason="test_external_requeue",
        )

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert claimed.leases == []
        assert store.get_task("req-pending-stale")["status"] == "pending"
        assert (await actor.contains_request(request_id="req-pending-stale")).location == "backlog"
        assert actor.stats()["counters"]["requeued"] == 1
        assert actor.stats()["counters"]["stale_dropped"] == 0

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_drops_terminal_stale_backlog_head_and_assigns_next() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=_MockTaskStateStoreClient(store),
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        assert (await actor.append(_work("req-stale"), assign=False)).ok is True
        assert (await actor.append(_work("req-valid"), assign=False)).ok is True
        assert actor.stats()["backlog_depth"] == 2

        store.complete_task_failure(
            request_id="req-stale",
            error="client_abandoned",
            result_path=None,
        )
        store.mark_task_retrieved(request_id="req-stale")

        synced = await actor.sync_replicas([_replica("replica-0")])

        assert synced.assigned is not None
        assert synced.assigned["assigned"] == 1
        assert store.get_task("req-stale")["status"] == "retrieved"
        assert store.get_task("req-valid")["status"] == "assigned"
        assert (await actor.contains_request(request_id="req-stale")).present is False
        assert actor.stats()["backlog_depth"] == 0
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 1
        assert actor.stats()["counters"]["stale_dropped"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_recognizes_wrapped_task_state_conflict() -> None:
    actor = _ModelWorkSchedulerActor()

    class _WrappedConflict(RuntimeError):
        def as_instanceof_cause(self):
            return TaskStateConflictError("cannot claim assigned task; current status='retrieved'")

    conflict = actor._claim_conflict_cause(_WrappedConflict("RayTaskError(TaskStateConflictError)"))

    assert isinstance(conflict, TaskStateConflictError)


def test_issue_645_scheduler_reconciles_wrapped_task_state_conflict() -> None:
    class _WrappedConflict(RuntimeError):
        def as_instanceof_cause(self):
            return TaskStateConflictError("cannot claim assigned task; current status='retrieved'")

    store = TaskStateStore.in_memory()
    task_state = _MockTaskStateStoreClient(store)
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=task_state,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-wrapped-conflict"), assign=True)).ok is True
        store.complete_task_failure(
            request_id="req-wrapped-conflict",
            error="client_abandoned",
            result_path=None,
        )
        store.mark_task_retrieved(request_id="req-wrapped-conflict")

        task_state.overrides["claim_task"] = _WrappedConflict("RayTaskError(TaskStateConflictError)")
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert claimed.leases == []
        assert store.get_task("req-wrapped-conflict")["status"] == "retrieved"
        assert (await actor.contains_request(request_id="req-wrapped-conflict")).present is False
        assert actor.stats()["counters"]["stale_dropped"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_does_not_reconcile_unrelated_assign_conflict() -> None:
    store = TaskStateStore.in_memory()
    task_state = _MockTaskStateStoreClient(store)
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=task_state,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        assert (await actor.append(_work("req-conflict"), assign=False)).ok is True

        task_state.overrides["assign_task"] = TaskStateConflictError(
            "terminal task commit payload mismatch"
        )

        with pytest.raises(TaskStateConflictError, match="terminal task commit payload mismatch"):
            await actor.sync_replicas([_replica("replica-0")])

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_does_not_reconcile_unrelated_task_state_conflict() -> None:
    store = TaskStateStore.in_memory()
    task_state = _MockTaskStateStoreClient(store)
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=task_state,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-conflict"), assign=True)).ok is True
        store.complete_task_failure(
            request_id="req-conflict",
            error="client_abandoned",
            result_path=None,
        )

        task_state.overrides["claim_task"] = TaskStateConflictError(
            "terminal task commit payload mismatch"
        )

        with pytest.raises(TaskStateConflictError, match="terminal task commit payload mismatch"):
            await actor.claim_from_replica_queue(
                domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                replica_id="replica-0",
                consumer_id="consumer-replica-0",
                consumer_generation=10,
                max_items=1,
                lease_ttl_s=30.0,
            )

    try:
        asyncio.run(_run())
    finally:
        store.close()
