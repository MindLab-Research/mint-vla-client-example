import asyncio
import time

import pytest

from mint_server.backend.model_work_scheduler import (
    CURRENT_CODE_IDENTITY,
    ModelWorkSchedulerConflictError,
    ModelWorkSchedulerCodeIdentityMismatchError,
    ModelWorkSchedulerClient,
    _ModelWorkSchedulerActor,
    _ray_model_work_scheduler_actor_name,
)
from mint_server.backend.task_state_store import TaskStateConflictError, TaskStateStore


@pytest.fixture(autouse=True)
def disable_scheduler_assignment_loop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0")


def _work(
    request_id: str,
    *,
    domain_key: str = "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    affinity_group: str | None = "lora:session-a:generation:1",
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
        "ordering_key": "session:session-a",
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


def test_scheduler_assigns_to_registered_replica_queue() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1")))["ok"] is True
        assert (await actor.sync_replicas([_replica("replica-0")]))["replicas"] == 1

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
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-1"]
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

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-expensive"]
        assert claimed["remaining_queue_depth"] == 1

    asyncio.run(_run())


def test_scheduler_multi_claim_for_training_domains_stays_on_same_affinity() -> None:
    domain_key = "bumblebee:Qwen/Qwen3-30B-A3B-Instruct-2507"
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(_work("req-a1", domain_key=domain_key, affinity_group="training_session:a"))
        await actor.append(_work("req-b1", domain_key=domain_key, affinity_group="training_session:b"))
        await actor.append(_work("req-a2", domain_key=domain_key, affinity_group="training_session:a"))
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-a1", "req-a2"]
        assert claimed["remaining_queue_depth"] == 1

    asyncio.run(_run())


def test_scheduler_same_affinity_domains_can_be_disabled_by_constructor() -> None:
    domain_key = "bumblebee:Qwen/Qwen3-30B-A3B-Instruct-2507"
    actor = _ModelWorkSchedulerActor(same_affinity_multi_claim_domains=())

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(_work("req-a1", domain_key=domain_key, affinity_group="training_session:a"))
        await actor.append(_work("req-b1", domain_key=domain_key, affinity_group="training_session:b"))
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=2,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-a1", "req-b1"]

    asyncio.run(_run())


def test_scheduler_same_affinity_domains_can_be_overridden_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_CLAIM_SAME_AFFINITY_DOMAINS", "custom:")
    domain_key = "custom:model-a"
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0", domain_key=domain_key)])
        await actor.append(_work("req-a1", domain_key=domain_key, affinity_group="group-a"))
        await actor.append(_work("req-b1", domain_key=domain_key, affinity_group="group-b"))
        await actor.append(_work("req-a2", domain_key=domain_key, affinity_group="group-a"))
        await actor.assign_pending()

        claimed = await actor.claim_from_replica_queue(
            domain_key=domain_key,
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-a1", "req-a2"]
        assert claimed["remaining_queue_depth"] == 1

    asyncio.run(_run())


def test_scheduler_default_actor_name_uses_mint_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME", raising=False)

    assert _ray_model_work_scheduler_actor_name() == "mint_model_work_scheduler"


def test_scheduler_snapshots_include_code_identity() -> None:
    actor = _ModelWorkSchedulerActor()

    assert actor.ping()["code_identity"] == CURRENT_CODE_IDENTITY
    assert actor.stats()["code_identity"] == CURRENT_CODE_IDENTITY


def test_scheduler_client_rejects_stale_code_identity() -> None:
    client = ModelWorkSchedulerClient()

    client._validate_code_identity({"code_identity": CURRENT_CODE_IDENTITY})
    with pytest.raises(ModelWorkSchedulerCodeIdentityMismatchError):
        client._validate_code_identity({"code_identity": "stale-scheduler-code"})


def test_issue_638_scheduler_registers_actor_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.logging_context as logging_context

    calls = {"count": 0}
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: calls.__setitem__("count", calls["count"] + 1))

    _ModelWorkSchedulerActor()

    assert calls["count"] == 1


def test_issue_638_scheduler_registers_otel_gauges(monkeypatch: pytest.MonkeyPatch) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.logging_context as logging_context

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

    import mint_server.logging_context as logging_context

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
        assert len(claimed["leases"]) == 1

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

    import mint_server.logging_context as logging_context

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


def test_scheduler_append_can_assign_immediately() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert await actor.is_empty() is True
        await actor.sync_replicas([_replica("replica-0")])
        out = await actor.append(_work("req-1"), assign=True, assign_max_items=1)

        assert out["ok"] is True
        assert await actor.is_empty() is False
        assert out["scheduler_instance_id"]
        contains = await actor.contains_request(request_id="req-1")
        assert contains["present"] is True
        assert contains["location"] == "assigned"
        assert contains["scheduler_instance_id"] == out["scheduler_instance_id"]
        assert out["assigned"]["assigned"] == 1
        assert actor.stats()["backlog_depth"] == 0
        assert actor.stats()["replica_queues"][
            "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        ]["depth"] == 1

    asyncio.run(_run())


def test_sampling_inflight_admission_observe_records_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1")))["ok"] is True
        second = await actor.append(_work("req-2"))

        assert second["ok"] is True
        assert second["sampling_inflight_admission"]["would_reject"] is True
        assert second["sampling_inflight_admission"]["reason"] == "principal_domain_inflight_limit_exceeded"
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


def test_sampling_inflight_admission_enforce_rejects_principal_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-1")))["ok"] is True
        rejected = await actor.append(_work("req-2"))

        assert rejected["ok"] is False
        assert rejected["reason"] == "principal_domain_inflight_limit_exceeded"
        assert rejected["current"] == 1
        assert rejected["limit"] == 1
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
        assert (await actor.append(_work("req-1", throttle_principal="apikey:key-a")))["ok"] is True
        rejected = await actor.append(_work("req-2", throttle_principal="apikey:key-b"))

        assert rejected["ok"] is False
        assert rejected["reason"] == "domain_inflight_limit_exceeded"
        assert rejected["current"] == 1
        assert rejected["limit"] == 1
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
        assert (await actor.append(_work("req-1"), assign=True))["ok"] is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease_id = str(claimed["leases"][0]["lease_id"])
        assert (await actor.complete_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        ))["ok"] is True
        assert (await actor.append(_work("req-2"), assign=True))["ok"] is True

    asyncio.run(_run())


def test_scheduler_assignment_loop_moves_backlog_to_replica_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0.01")
    actor = _ModelWorkSchedulerActor()

    async def _run() -> None:
        assert (await actor.append(_work("req-loop")))["ok"] is True
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

        assert out["cancelled"] is True
        assert (await actor.contains_request(request_id="req-1"))["present"] is False
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
        assert (await actor.append(_work("req-recycle"), assign=True))["ok"] is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-old",
            consumer_generation=1,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-recycle"]

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
        assert [lease["item"]["request_id"] for lease in claimed_after_recycle["leases"]] == [
            "req-recycle"
        ]
        assert claimed_after_recycle["leases"][0]["consumer_id"] == "consumer-new"
        assert claimed_after_recycle["leases"][0]["consumer_generation"] == 2

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
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-active"]

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
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-assigned"]

        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])
        assert sync["requeued"] == 2
        stats = actor.stats()
        assert stats["backlog_depth"] == 2
        assert stats["leases"] == []
        assert stats["replica_queues"]["vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"][
            "status"
        ] == "unhealthy"

    asyncio.run(_run())


def test_lease_complete_fail_and_expiry() -> None:
    actor = _ModelWorkSchedulerActor()

    async def _claim_one(request_id: str) -> str:
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
        return str(claimed["leases"][0]["lease_id"])

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])

        complete_lease = await _claim_one("req-complete")
        assert (await actor.complete_lease(
            lease_id=complete_lease,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        ))["ok"] is True
        assert actor.stats()["counters"]["completed"] == 1

        fail_lease = await _claim_one("req-fail")
        failed = await actor.fail_lease(
            lease_id=fail_lease,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=False,
            reason="runtime_error",
        )
        assert failed == {"ok": True, "request_id": "req-fail", "requeued": False}
        assert actor.stats()["counters"]["failed"] == 1

        expire_lease = await _claim_one("req-expire")
        assert (await actor.expire_leases(now=time.time() + 999.0))["expired"] == 1
        assert actor.stats()["backlog_depth"] == 1
        assert expire_lease not in {lease["lease_id"] for lease in actor.stats()["leases"]}

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
        lease_id = str(claimed["leases"][0]["lease_id"])

        assert (await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        ))["ok"] is True
        assert (await actor.fail_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            requeue=True,
            reason="test_requeue",
        ))["requeued"] is True
        assert (await actor.validate_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
        )) == {"ok": False, "reason": "unknown_lease"}

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
        lease_id = str(claimed["leases"][0]["lease_id"])

        assert (await actor.begin_finalize_lease(
            lease_id=lease_id,
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=60.0,
        ))["ok"] is True
        sync = await actor.sync_replicas([_replica("replica-0", status="unhealthy")])

        assert sync["requeued"] == 0
        assert actor.stats()["leases"][0]["lease_id"] == lease_id

    asyncio.run(_run())


def test_scheduler_persists_append_assign_claim_and_begin_finalize_to_task_state_store() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
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
        assert out["ok"] is True

        record = store.get_task("req-persisted")
        assert record["status"] == "assigned"
        assert record["subqueue_id"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0"
        assert record["scheduler_epoch"] == 1
        assert record["metadata"]["model_work_scheduler_instance_id"] == out["scheduler_instance_id"]

        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed["leases"][0]
        record = store.get_task("req-persisted")
        assert record["status"] == "leased"
        assert record["lease_id"] == lease["lease_id"]
        assert record["attempt_id"] == lease["attempt_id"]
        assert record["runtime_generation"] == 10

        finalizing = await actor.begin_finalize_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            finalize_ttl_s=30.0,
            staged_payload_path="/tmp/req-persisted.json",
        )
        assert finalizing["ok"] is True
        record = store.get_task("req-persisted")
        assert record["status"] == "finalizing"
        assert record["staged_payload_path"] == "/tmp/req-persisted.json"
        assert actor.stats()["task_state_store_enabled"] is True
        assert actor.stats()["scheduler_epoch"] == 1

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_accepts_pre_registered_pending_task_state_store_future() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
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

        assert out["ok"] is True
        assert out["idempotent"] is True
        assert store.get_task("req-pre-registered")["status"] == "assigned"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_rolls_back_new_task_when_assign_fails_after_create() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )
    original_task_state_call = actor._task_state_call

    async def _task_state_call(method: str, **kwargs):
        if method == "assign_task":
            raise RuntimeError("assign failed after create")
        return await original_task_state_call(method, **kwargs)

    actor._task_state_call = _task_state_call  # type: ignore[method-assign]

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        with pytest.raises(RuntimeError, match="assign failed after create"):
            await actor.append(_work("req-assign-fails"), assign=True)

        with pytest.raises(KeyError):
            store.get_task("req-assign-fails")
        assert (await actor.contains_request(request_id="req-assign-fails"))["present"] is False
        assert actor.stats()["backlog_depth"] == 0

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_hydrates_active_task_state_after_restart() -> None:
    store = TaskStateStore.in_memory()
    actor_a = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )
    actor_b = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor_a.sync_replicas([_replica("replica-0")])
        assert (await actor_a.append(_work("req-restart"), assign=True))["ok"] is True
        assert store.get_task("req-restart")["status"] == "assigned"

        await actor_b.sync_replicas([_replica("replica-0")])
        contains = await actor_b.contains_request(request_id="req-restart")
        assert contains["present"] is True
        assert contains["location"] == "assigned"
        claimed = await actor_b.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-restart"]
        assert store.get_task("req-restart")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_scheduler_hydrates_sampling_inflight_counts_from_task_state_store() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
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

        contains = await actor.contains_request(request_id="req-hydrate-a")
        assert contains["present"] is True
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
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-requeue"), assign=True))["ok"] is True
        claimed = await actor.claim_from_replica_queue(
            domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            max_items=1,
            lease_ttl_s=30.0,
        )
        lease = claimed["leases"][0]
        failed = await actor.fail_lease(
            lease_id=lease["lease_id"],
            consumer_id="consumer-replica-0",
            consumer_generation=10,
            reason="executor_failed",
            requeue=True,
        )
        assert failed["requeued"] is True
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
        assert [item["item"]["request_id"] for item in reclaimed["leases"]] == ["req-requeue"]
        assert store.get_task("req-requeue")["status"] == "leased"

    try:
        asyncio.run(_run())
    finally:
        store.close()


def test_issue_645_scheduler_drops_terminal_stale_head_and_claims_next() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-stale"), assign=True))["ok"] is True
        assert (await actor.append(_work("req-valid"), assign=True))["ok"] is True
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

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == ["req-valid"]
        assert store.get_task("req-stale")["status"] == "retrieved"
        assert store.get_task("req-valid")["status"] == "leased"
        assert (await actor.contains_request(request_id="req-stale"))["present"] is False
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
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-pending-stale"), assign=True))["ok"] is True
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

        assert claimed["leases"] == []
        assert store.get_task("req-pending-stale")["status"] == "pending"
        assert (await actor.contains_request(request_id="req-pending-stale"))["location"] == "backlog"
        assert actor.stats()["counters"]["requeued"] == 1
        assert actor.stats()["counters"]["stale_dropped"] == 0

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


def test_issue_645_scheduler_does_not_reconcile_unrelated_task_state_conflict() -> None:
    store = TaskStateStore.in_memory()
    actor = _ModelWorkSchedulerActor(
        use_task_state_store=True,
        task_state_store=store,
        owner_id="scheduler-test",
    )

    async def _run() -> None:
        await actor.sync_replicas([_replica("replica-0")])
        assert (await actor.append(_work("req-conflict"), assign=True))["ok"] is True
        store.complete_task_failure(
            request_id="req-conflict",
            error="client_abandoned",
            result_path=None,
        )

        async def _raise_unrelated(_method: str, **_kwargs):
            raise TaskStateConflictError("terminal task commit payload mismatch")

        actor._task_state_call = _raise_unrelated  # type: ignore[method-assign]

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
