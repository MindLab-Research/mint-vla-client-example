from __future__ import annotations

import os

import pytest

from tinker_server.backend.model_actor_inventory import ActorType
from tinker_server.backend.model_actor_launchers import (
    ModelActorLauncherRegistry,
    placement_env_for_spec,
)
from tinker_server.backend.model_actor_supervisor import (
    ModelActorSpec,
    ModelActorSupervisor,
    desired_specs_from_env,
    domain_key_for_training_base_model,
    domain_key_for_vllm_base_model,
    queue_id_for_replica,
)
from tinker_server.backend.model_actor_placement import ModelActorPlacementReconciler


class _FakeRuntimeActor:
    def __init__(self, *, actor_name: str, domain_key: str, replica_id: str, generation: int) -> None:
        self.actor_name = actor_name
        self.domain_key = domain_key
        self.replica_id = replica_id
        self.generation = int(generation)
        self.running = True
        self.active_request_id = None
        self.start_calls = 0
        self.shutdown_calls = 0
        self.health_errors: list[BaseException] = []

    def health_snapshot(self) -> dict:
        if self.health_errors:
            raise self.health_errors.pop(0)
        return {
            "actor_name": self.actor_name,
            "domain_key": self.domain_key,
            "replica_id": self.replica_id,
            "actor_generation": self.generation,
            "running": self.running,
            "active_request_id": self.active_request_id,
        }

    def shutdown(self) -> dict:
        self.shutdown_calls += 1
        self.running = False
        return {"ok": True}

    def start(self) -> dict:
        self.start_calls += 1
        self.running = True
        return self.health_snapshot()


def test_issue_593_supervisor_exposes_explicit_inventory_contract() -> None:
    supervisor = ModelActorSupervisor()

    entry = supervisor.register(
        actor_name="vllm-contract-actor",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        base_model="model-a",
        metadata={"launcher_key": "legacy_vllm"},
    )
    supervisor.mark_ready("vllm-contract-actor")
    supervisor.mark_inflight("vllm-contract-actor", +1)
    supervisor.set_session("vllm-contract-actor", "session-a")

    assert entry.metadata["launcher_contract"] == "model_actor_supervisor"
    current = supervisor.get("vllm-contract-actor")
    assert current is not None
    assert current.current_session == "session-a"
    assert current.inflight_count == 1
    listed = supervisor.cached_snapshot()
    assert any(row["actor_name"] == "vllm-contract-actor" for row in listed)
    assert supervisor.total_gpus_used() >= 1

    assert supervisor.clear_session("session-a", actor_type=ActorType.VLLM) == 1
    assert supervisor.unregister("vllm-contract-actor") is True


@pytest.mark.anyio
async def test_issue_593_supervisor_uses_launcher_registry_when_no_factory() -> None:
    launched: list[tuple[ModelActorSpec, int]] = []

    async def _launch(spec: ModelActorSpec, generation: int):
        launched.append((spec, generation))
        return _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

    async def _sync(_registrations):
        return None

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:model-a",
                replica_id="replica-0",
                base_model="model-a",
                launcher_key="test_launcher",
                gpu_count=1,
            )
        ],
        scheduler_sync=_sync,
        placement_reconciler=lambda _desired: {"reclaimed": 0},
        launcher_registry=ModelActorLauncherRegistry({"test_launcher": _launch}),
    )

    await supervisor.reconcile_once()

    assert len(launched) == 1
    assert launched[0][0].domain_key == "vllm:model-a"
    snapshot = supervisor.snapshot()["replicas"]
    assert snapshot["vllm:model-a::replica-0"]["launcher_key"] == "test_launcher"


@pytest.mark.anyio
async def test_issue_593_supervisor_creates_replica_and_syncs_scheduler() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _sync(registrations):
        synced.append([registration.to_dict() for registration in registrations])

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:model-a",
                replica_id="replica-0",
                base_model="model-a",
                gpu_count=4,
            )
        ],
        runtime_factory=_factory,
        scheduler_sync=_sync,
    )

    out = await supervisor.reconcile_once()

    assert out["ok"] is True
    assert len(created) == 1
    assert created[0].start_calls == 1
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["state"] == "healthy"
    assert replica["generation"] >= 1
    assert replica["queue_id"] == "vllm:model-a::replica-0"
    assert replica["base_model"] == "model-a"
    assert out["snapshot"]["domains"]["vllm:model-a"] == {
        "replicas": 1,
        "healthy": 1,
        "unhealthy": 0,
    }
    assert synced[-1][0]["domain_key"] == "vllm:model-a"
    assert synced[-1][0]["replica_id"] == "replica-0"
    assert synced[-1][0]["queue_id"] == "vllm:model-a::replica-0"
    assert synced[-1][0]["status"] == "healthy"
    assert synced[-1][0]["capacity"] == 4
    assert await supervisor.async_snapshot() == supervisor.snapshot()


@pytest.mark.anyio
async def test_issue_593_supervisor_projects_scheduler_backlog_to_desired_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_model = "Qwen/Qwen3-0.6B"
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", base_model)
    monkeypatch.setenv(
        "MINT_DENSE_MODEL_PLACEMENT_JSON",
        f'{{"{base_model}":{{"replica":0,"node_ip":"10.0.0.8","gpu_count":1}}}}',
    )
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _scheduler_stats():
        return {
            "backlog_depth_by_domain": {domain_key_for_training_base_model(base_model): 1},
            "replica_queues": {},
            "leases": [],
        }

    async def _placement_reconciler(_desired):
        return {"ok": True, "blocked": {}}

    supervisor = ModelActorSupervisor(
        runtime_factory=_factory,
        scheduler_stats=_scheduler_stats,
        placement_reconciler=_placement_reconciler,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )

    out = await supervisor.reconcile_once()

    domain_key = domain_key_for_training_base_model(base_model)
    label = f"{domain_key}::replica-0"
    assert len(created) == 1
    assert created[0].domain_key == domain_key
    assert out["snapshot"]["replicas"][label]["base_model"] == base_model
    assert out["snapshot"]["replicas"][label]["node_pins"] == ["10.0.0.8"]
    assert synced[-1][0]["domain_key"] == domain_key
    assert synced[-1][0]["status"] == "healthy"


@pytest.mark.anyio
async def test_issue_593_supervisor_restarts_dead_runtime_with_monotonic_generation() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    supervisor = ModelActorSupervisor(
        specs=[ModelActorSpec(domain_key="vllm:model-a", replica_id="replica-0")],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )
    await supervisor.reconcile_once()
    first_generation = created[0].generation
    created[0].health_errors.append(RuntimeError("actor died"))

    out = await supervisor.reconcile_once()

    assert len(created) == 2
    assert created[1].generation > first_generation
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["state"] == "healthy"
    assert replica["generation"] == created[1].generation
    assert replica["crash_count"] == 1
    assert out["snapshot"]["restarted_total"] == 1
    assert synced[-1][0]["generation"] == created[1].generation


@pytest.mark.anyio
async def test_issue_593_supervisor_restarts_runtime_that_reports_not_running() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    supervisor = ModelActorSupervisor(
        specs=[ModelActorSpec(domain_key="vllm:model-a", replica_id="replica-0")],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )
    await supervisor.reconcile_once()
    first_generation = created[0].generation
    created[0].running = False

    out = await supervisor.reconcile_once()

    assert len(created) == 2
    assert created[1].generation > first_generation
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["state"] == "healthy"
    assert replica["generation"] == created[1].generation
    assert replica["crash_count"] == 1
    assert out["snapshot"]["restarted_total"] == 1
    assert synced[-1][0]["status"] == "healthy"


@pytest.mark.anyio
async def test_issue_593_supervisor_blocks_unavailable_node_pin_and_syncs_unclaimable() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []
    cleaned: list[list[str]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _cleaner(specs: dict[tuple[str, str], ModelActorSpec]):
        cleaned.append([f"{domain}::{replica}" for domain, replica in sorted(specs)])

    async def _sync(registrations):
        synced.append([registration.to_dict() for registration in registrations])

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:model-a",
                replica_id="replica-0",
                node_pins=("10.0.0.9",),
            )
        ],
        runtime_factory=_factory,
        node_inventory=lambda: {"10.0.0.1"},
        scheduler_sync=_sync,
        orphan_pg_cleaner=_cleaner,
    )

    out = await supervisor.reconcile_once()

    assert created == []
    assert cleaned == [["vllm:model-a::replica-0"]]
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["state"] == "blocked"
    assert replica["last_error"] == "node pin unavailable: 10.0.0.9"
    assert out["snapshot"]["blocked_total"] == 1
    assert synced[-1][0]["status"] == "blocked"


@pytest.mark.anyio
async def test_issue_593_supervisor_does_not_recycle_busy_runtime_without_force() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    supervisor = ModelActorSupervisor(
        specs=[ModelActorSpec(domain_key="vllm:model-a", replica_id="replica-0")],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )
    await supervisor.reconcile_once()
    created[0].active_request_id = "active-req"

    busy = await supervisor.recycle(domain_key="vllm:model-a", replica_id="replica-0")
    forced = await supervisor.recycle(domain_key="vllm:model-a", replica_id="replica-0", force=True)

    assert busy == {
        "ok": False,
        "domain_key": "vllm:model-a",
        "replica_id": "replica-0",
        "reason": "busy",
        "active_request_id": "active-req",
    }
    assert forced == {
        "ok": True,
        "domain_key": "vllm:model-a",
        "replica_id": "replica-0",
        "recycled": True,
    }
    assert created[0].shutdown_calls == 1
    assert supervisor.snapshot()["busy_recycle_skipped_total"] == 1
    assert synced[-1][0]["status"] == "dead"


def test_issue_593_supervisor_parses_desired_specs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MINT_MODEL_ACTOR_DESIRED_JSON",
        '[{"base_model":"Qwen/Test","replica":1,"node_pins":["10.0.0.1"],"gpu_count":4}]',
    )

    specs = desired_specs_from_env()

    assert specs == [
        ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-1",
            base_model="Qwen/Test",
            node_pins=("10.0.0.1",),
            gpu_count=4,
        ),
        ModelActorSpec(
            domain_key="internal:control",
            launcher_key="internal_control",
            gpu_count=0,
        ),
    ]


def test_issue_593_supervisor_builds_runtime_placement_env_from_node_pin() -> None:
    env = placement_env_for_spec(
        ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-2",
            base_model="Qwen/Test",
            node_pins=("10.0.0.17",),
            gpu_count=4,
        )
    )

    placement = '{"Qwen/Test":{"gpu_count":4,"node_ip":"10.0.0.17","replica":2}}'
    assert env == {
        "MINT_MODEL_PLACEMENT_JSON": placement,
        "MINT_VLLM_MODEL_PLACEMENT_JSON": placement,
        "MINT_DENSE_MODEL_PLACEMENT_JSON": placement,
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement,
        "MINT_MODEL_ACTOR_REPLICA_ID": "replica-2",
        "MINT_VLLM_PINNED_NODE_IP_JSON": '{"Qwen/Test":"10.0.0.17"}',
        "MINT_VLLM_MODEL_NODE_IPS_JSON": '{"Qwen/Test":["10.0.0.17"]}',
    }


def test_issue_593_supervisor_builds_runtime_placement_env_from_multi_node_slices() -> None:
    env = placement_env_for_spec(
        ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-0",
            base_model="Qwen/Test",
            placement_slices=(("replica-0", "10.0.0.17", 4), ("replica-0", "10.0.0.18", 4)),
            gpu_count=4,
        )
    )

    placement = (
        '{"Qwen/Test":[{"gpu_count":4,"node_ip":"10.0.0.17","replica":0},'
        '{"gpu_count":4,"node_ip":"10.0.0.18","replica":0}]}'
    )
    assert env == {
        "MINT_MODEL_PLACEMENT_JSON": placement,
        "MINT_VLLM_MODEL_PLACEMENT_JSON": placement,
        "MINT_DENSE_MODEL_PLACEMENT_JSON": placement,
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement,
        "MINT_MODEL_ACTOR_REPLICA_ID": "replica-0",
        "MINT_VLLM_MODEL_NODE_IPS_JSON": '{"Qwen/Test":["10.0.0.17","10.0.0.18"]}',
    }


def test_issue_593_supervisor_builds_runtime_placement_env_from_multi_node_pins() -> None:
    env = placement_env_for_spec(
        ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-0",
            base_model="Qwen/Test",
            node_pins=("10.0.0.17", "10.0.0.18"),
            gpu_count=4,
        )
    )

    placement = (
        '{"Qwen/Test":[{"gpu_count":4,"node_ip":"10.0.0.17","replica":0},'
        '{"gpu_count":4,"node_ip":"10.0.0.18","replica":0}]}'
    )
    assert env == {
        "MINT_MODEL_PLACEMENT_JSON": placement,
        "MINT_VLLM_MODEL_PLACEMENT_JSON": placement,
        "MINT_DENSE_MODEL_PLACEMENT_JSON": placement,
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON": placement,
        "MINT_MODEL_ACTOR_REPLICA_ID": "replica-0",
        "MINT_VLLM_MODEL_NODE_IPS_JSON": '{"Qwen/Test":["10.0.0.17","10.0.0.18"]}',
    }


def test_issue_593_supervisor_falls_back_to_persistent_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_PLACEMENT_JSON", raising=False)
    monkeypatch.delenv("MINT_VLLM_MODEL_PLACEMENT_JSON", raising=False)
    monkeypatch.delenv("MINT_DENSE_MODEL_PLACEMENT_JSON", raising=False)
    monkeypatch.delenv("MINT_MEGATRON_MODEL_PLACEMENT_JSON", raising=False)
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/A, Qwen/B")

    specs = desired_specs_from_env()

    assert [spec.domain_key for spec in specs] == [
        "vllm:Qwen/A",
        "training:Qwen/A",
        "vllm:Qwen/B",
        "training:Qwen/B",
        "internal:control",
    ]
    assert domain_key_for_vllm_base_model("Qwen/A") == "vllm:Qwen/A"
    assert queue_id_for_replica("vllm:Qwen/A", "replica-2") == "vllm:Qwen/A::replica-2"


def test_issue_593_persistent_specs_inherit_runtime_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_DESIRED_JSON", raising=False)
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/A")
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        '{"Qwen/A":{"replica":1,"node_ip":"10.0.0.7","gpu_count":2}}',
    )
    monkeypatch.setenv(
        "MINT_DENSE_MODEL_PLACEMENT_JSON",
        '{"Qwen/A":{"replica":0,"node_ip":"10.0.0.8","gpu_count":1}}',
    )

    specs = desired_specs_from_env()

    assert specs[:2] == [
        ModelActorSpec(
            domain_key="vllm:Qwen/A",
            replica_id="replica-1",
            base_model="Qwen/A",
            launcher_key="legacy_vllm",
            node_pins=("10.0.0.7",),
            placement_slices=(("replica-1", "10.0.0.7", 2),),
            gpu_count=2,
        ),
        ModelActorSpec(
            domain_key="training:Qwen/A",
            base_model="Qwen/A",
            launcher_key="training",
            node_pins=("10.0.0.8",),
            placement_slices=(("replica-0", "10.0.0.8", 1),),
            gpu_count=1,
        ),
    ]


def test_issue_593_persistent_specs_preserve_multi_node_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_DESIRED_JSON", raising=False)
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/A")
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        (
            '{"Qwen/A":['
            '{"replica":0,"node_ip":"10.0.0.7","gpu_count":4},'
            '{"replica":0,"node_ip":"10.0.0.8","gpu_count":4}'
            "]}"
        ),
    )
    monkeypatch.setenv(
        "MINT_DENSE_MODEL_PLACEMENT_JSON",
        '{"Qwen/A":{"replica":0,"node_ip":"10.0.0.9","gpu_count":1}}',
    )

    specs = desired_specs_from_env()

    assert specs[0] == ModelActorSpec(
        domain_key="vllm:Qwen/A",
        base_model="Qwen/A",
        launcher_key="legacy_vllm",
        node_pins=("10.0.0.7", "10.0.0.8"),
        placement_slices=(("replica-0", "10.0.0.7", 4), ("replica-0", "10.0.0.8", 4)),
        gpu_count=4,
    )


def test_issue_593_persistent_specs_reject_worker_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_DESIRED_JSON", raising=False)
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/A")
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        '{"Qwen/A":{"replica":0,"worker_idx":1,"gpu_count":1}}',
    )

    with pytest.raises(ValueError, match="uses worker_index; use node_ip"):
        desired_specs_from_env()


def test_issue_593_supervisor_empty_env_has_no_desired_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_DESIRED_JSON", raising=False)
    monkeypatch.delenv("MINT_PERSISTENT_MODELS", raising=False)

    assert desired_specs_from_env() == [
        ModelActorSpec(
            domain_key="internal:control",
            launcher_key="internal_control",
            gpu_count=0,
        )
    ]


def test_issue_593_placement_reconciler_uses_node_pin_and_removes_owned_orphan_pg() -> None:
    capacity_checks: list[dict[str, object]] = []
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    def _capacity(required, context, ignore_pg_names, namespace):
        capacity_checks.append(
            {
                "required": dict(required),
                "context": context,
                "ignore_pg_names": set(ignore_pg_names),
                "namespace": namespace,
            }
        )

    reconciler = ModelActorPlacementReconciler(
        namespace="tinker",
        actor_exists=lambda _name, _namespace: False,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        actor_lister=lambda: [
            {"name": "mint_model_runtime_old", "namespace": "tinker"},
            {"name": "foreign_actor", "namespace": "other"},
        ],
        capacity_checker=_capacity,
        placement_group_remover=lambda name, namespace: removed_pgs.append((name, namespace)) or True,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Test", "replica-1"): ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-1",
                base_model="Qwen/Test",
                node_pin="10.0.0.17",
                gpu_count=4,
            )
        }
    )

    assert out["ok"] is True
    assert out["node_pins"]["vllm:Qwen/Test::replica-1"] == ["10.0.0.17"]
    assert out["cleaned_actor_names"] == ["mint_model_runtime_old"]
    assert ("mint_model_runtime_old", "tinker", "model_actor_supervisor_undesired_wrapper") in killed
    assert ("tinker_vllm_test_pg", "tinker") in removed_pgs
    assert ("multinode_vllm_test_pg", "tinker") in removed_pgs
    assert capacity_checks == [
        {
            "required": {"10.0.0.17": 4},
            "context": "model_actor_supervisor placement domain='vllm:Qwen/Test' replica='replica-1'",
            "ignore_pg_names": {
                "mint_model_actor_vllm-Qwen-Test_replica-1_pg",
                "tinker_vllm_test_pg",
                "multinode_vllm_test_pg",
            },
            "namespace": "tinker",
        }
    ]


def test_issue_593_placement_reconciler_evicts_foreign_blockers_when_target_absent() -> None:
    capacity_checks: list[dict[str, object]] = []
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    def _capacity(required, context, ignore_pg_names, namespace):
        capacity_checks.append(
            {
                "required": dict(required),
                "context": context,
                "ignore_pg_names": set(ignore_pg_names),
                "namespace": namespace,
            }
        )
        if len(capacity_checks) == 1:
            raise RuntimeError("insufficient GPU on pinned node; blocker=foreign_vllm_pg")

    reconciler = ModelActorPlacementReconciler(
        namespace="tinker",
        actor_exists=lambda _name, _namespace: False,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        capacity_checker=_capacity,
        gpu_actor_lister=lambda: [
            {
                "name": "foreign_gpu_actor",
                "namespace": "foreign",
                "node_ip": "10.0.0.17",
                "gpu": 4,
            }
        ],
        placement_group_lister=lambda: [
            {
                "name": "foreign_vllm_pg",
                "namespace": "foreign",
                "state": "CREATED",
                "node_ips": ["10.0.0.17"],
                "gpu_by_node_ip": {"10.0.0.17": 4},
            }
        ],
        placement_group_remover=lambda name, namespace: removed_pgs.append((name, namespace)) or True,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Test", "replica-1"): ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-1",
                base_model="Qwen/Test",
                node_pin="10.0.0.17",
                gpu_count=4,
            )
        }
    )

    assert out["ok"] is True
    assert len(capacity_checks) == 2
    assert (
        "foreign_gpu_actor",
        "foreign",
        "model_actor_supervisor_exclusive_placement_preempt",
    ) in killed
    assert ("foreign_vllm_pg", "foreign") in removed_pgs
    assert out["evicted_actor_names"] == ["foreign_gpu_actor"]
    assert out["evicted_placement_group_names"] == ["foreign_vllm_pg"]
    assert out["reclaimed_total"] >= 2


def test_issue_593_placement_reconciler_does_not_preempt_on_non_capacity_failure() -> None:
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    reconciler = ModelActorPlacementReconciler(
        namespace="tinker",
        actor_exists=lambda _name, _namespace: False,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        capacity_checker=lambda *_args: (_ for _ in ()).throw(RuntimeError("ray state lookup failed")),
        gpu_actor_lister=lambda: [
            {
                "name": "foreign_gpu_actor",
                "namespace": "foreign",
                "node_ip": "10.0.0.17",
                "gpu": 4,
            }
        ],
        placement_group_lister=lambda: [
            {
                "name": "foreign_vllm_pg",
                "namespace": "foreign",
                "state": "CREATED",
                "node_ips": ["10.0.0.17"],
                "gpu_by_node_ip": {"10.0.0.17": 4},
            }
        ],
        placement_group_remover=lambda name, namespace: removed_pgs.append((name, namespace)) or True,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Test", "replica-1"): ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-1",
                base_model="Qwen/Test",
                node_pin="10.0.0.17",
                gpu_count=4,
            )
        }
    )

    assert out["ok"] is False
    assert "ray state lookup failed" in out["blocked"]["vllm:Qwen/Test::replica-1"]
    assert killed == []
    assert ("foreign_vllm_pg", "foreign") not in removed_pgs
    assert out["evicted_actor_names"] == []
    assert out["evicted_placement_group_names"] == []


def test_issue_593_placement_reconciler_removes_unknown_namespace_pg_as_current_namespace() -> None:
    capacity_checks = 0
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    def _capacity(*_args):
        nonlocal capacity_checks
        capacity_checks += 1
        if capacity_checks == 1:
            raise RuntimeError("pinned node capacity check failed: blocker=unknown_ns_pg")

    reconciler = ModelActorPlacementReconciler(
        namespace="tinker",
        actor_exists=lambda _name, _namespace: False,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        capacity_checker=_capacity,
        gpu_actor_lister=lambda: [],
        placement_group_lister=lambda: [
            {
                "name": "unknown_ns_pg",
                "namespace": "",
                "state": "CREATED",
                "node_ips": ["10.0.0.17"],
                "gpu_by_node_ip": {"10.0.0.17": 4},
            }
        ],
        placement_group_remover=lambda name, namespace: removed_pgs.append((name, namespace)) or True,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Test", "replica-1"): ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-1",
                base_model="Qwen/Test",
                node_pin="10.0.0.17",
                gpu_count=4,
            )
        }
    )

    assert out["ok"] is True
    assert capacity_checks == 2
    assert killed == []
    assert ("unknown_ns_pg", "tinker") in removed_pgs
    assert out["evicted_actor_names"] == []
    assert out["evicted_placement_group_names"] == ["unknown_ns_pg"]


def test_issue_593_placement_reconciler_does_not_evict_foreign_blockers_when_target_started() -> None:
    capacity_checks: list[dict[str, object]] = []
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    def _actor_exists(name: str, _namespace: str) -> bool:
        return name == "mint_model_actor_vllm-Qwen-Test_replica-1"

    def _capacity(required, context, ignore_pg_names, namespace):
        capacity_checks.append(
            {
                "required": dict(required),
                "context": context,
                "ignore_pg_names": set(ignore_pg_names),
                "namespace": namespace,
            }
        )
        raise RuntimeError("insufficient GPU on pinned node; blocker=foreign_vllm_pg")

    reconciler = ModelActorPlacementReconciler(
        namespace="tinker",
        actor_exists=_actor_exists,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        capacity_checker=_capacity,
        gpu_actor_lister=lambda: [
            {
                "name": "foreign_gpu_actor",
                "namespace": "foreign",
                "node_ip": "10.0.0.17",
                "gpu": 4,
            }
        ],
        placement_group_lister=lambda: [
            {
                "name": "foreign_vllm_pg",
                "namespace": "foreign",
                "state": "CREATED",
                "node_ips": ["10.0.0.17"],
                "gpu_by_node_ip": {"10.0.0.17": 4},
            }
        ],
        placement_group_remover=lambda name, namespace: removed_pgs.append((name, namespace)) or True,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Test", "replica-1"): ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-1",
                base_model="Qwen/Test",
                node_pin="10.0.0.17",
                gpu_count=4,
            )
        }
    )

    assert out["ok"] is True
    assert capacity_checks == []
    assert killed == []
    assert ("foreign_vllm_pg", "foreign") not in removed_pgs
    assert out["evicted_actor_names"] == []
    assert out["evicted_placement_group_names"] == []


@pytest.mark.anyio
async def test_issue_593_supervisor_blocks_placement_capacity_failure_without_creating_runtime() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    def _placement(_desired):
        return {
            "ok": False,
            "blocked": {"vllm:Qwen/Test::replica-0": "pinned node capacity check failed: blocker_pg"},
            "node_pins": {"vllm:Qwen/Test::replica-0": ["10.0.0.7"]},
            "reclaimed_total": 0,
        }

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                node_pin="10.0.0.7",
                gpu_count=4,
            )
        ],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
        placement_reconciler=_placement,
    )

    out = await supervisor.reconcile_once()

    assert created == []
    replica = out["snapshot"]["replicas"]["vllm:Qwen/Test::replica-0"]
    assert replica["state"] == "blocked"
    assert replica["node_pins"] == ["10.0.0.7"]
    assert replica["last_action"] == "blocked:placement"
    assert "blocker_pg" in replica["last_error"]
    assert synced[-1][0]["status"] == "blocked"
    assert os.environ.get("MINT_MODEL_ACTOR_DESIRED_JSON") is None
