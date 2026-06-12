from __future__ import annotations

import sqlite3
import sys
import time
import types
from types import SimpleNamespace

import pytest
import yaml

from mint_server.backend.model_actor_inventory import ActorType
from mint_server.backend.cluster_placement_controller import (
    ClusterPlacementController,
    PlacementGroupCreateStatus,
)
from mint_server.backend.model_actor_launchers import (
    ModelActorLauncherRegistry,
    _model_runtime_max_claim_for_spec,
    _model_runtime_token_budget_for_spec,
    launch_model_engine_host,
    placement_env_for_spec,
)
from mint_server.backend.model_actor_supervisor import (
    ControlPlaneDependency,
    ModelActorSpec,
    ModelActorSupervisorClient,
    ModelActorSupervisor,
    desired_specs_from_env,
    domain_key_for_training_base_model,
    domain_key_for_vllm_base_model,
    queue_id_for_replica,
)
from mint_server.backend import model_actor_placement as placement_module
from mint_server.backend.model_actor_placement import ModelActorPlacementReconciler
from mint_server.backend.supervisor_state_store import (
    SupervisorMemoryStateStore,
    SupervisorSQLiteStateStore,
    SupervisorStateOwnerConflictError,
)
from mint_server.backend.topology import (
    ProviderTaskState,
    RayNodeState,
    TopologyConfig,
    TopologyManager,
    TopologyNodeDesired,
)


def _write_supervisor_topology(tmp_path, models: dict[str, object]) -> str:
    path = tmp_path / "topology.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "deployment_env": "dev",
                "cluster_id": "volcano",
                "state_path": str(tmp_path / "topology_state.yaml"),
                "providers": {},
                "nodes": {"desired": []},
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


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
        self.last_error: str | None = None
        self.failed_total = 0
        self.completed_total = 0
        self.processed_total = 0

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
            "last_error": self.last_error,
            "failed_total": self.failed_total,
            "completed_total": self.completed_total,
            "processed_total": self.processed_total,
        }

    def shutdown(self) -> dict:
        self.shutdown_calls += 1
        self.running = False
        return {"ok": True}

    def start(self) -> dict:
        self.start_calls += 1
        self.running = True
        return self.health_snapshot()


class _FakeStartTimeoutRuntimeActor(_FakeRuntimeActor):
    class _StartMethod:
        def __init__(self, actor: "_FakeStartTimeoutRuntimeActor") -> None:
            self._actor = actor

        def remote(self):
            self._actor.start_calls += 1
            return _FakeRayRef({"ok": True})

    def __getattribute__(self, name: str):
        if name == "start":
            return _FakeStartTimeoutRuntimeActor._StartMethod(self)
        return super().__getattribute__(name)


class _OpaqueRuntimeHandle:
    def __init__(self, actor: _FakeRuntimeActor) -> None:
        self._actor = actor

    def health_snapshot(self) -> dict:
        return self._actor.health_snapshot()

    def shutdown(self) -> dict:
        return self._actor.shutdown()

    def start(self) -> dict:
        return self._actor.start()


class _FakeRayRef:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def _never():
            while True:
                await __import__("asyncio").sleep(3600)

        return _never().__await__()


class _FakeRemoteMethod:
    def __init__(self, calls: list[tuple[str, tuple, dict]], name: str, value=None) -> None:
        self._calls = calls
        self._name = name
        self._value = value if value is not None else {"ok": True}

    def remote(self, *args, **kwargs):
        self._calls.append((self._name, args, kwargs))
        return _FakeRayRef(self._value)


class _FakeKillableActorHandle:
    def __init__(self, *, code_identity: str | None) -> None:
        self.code_identity = code_identity
        self.calls: list[tuple[str, tuple, dict]] = []
        self.snapshot = _FakeRemoteMethod(
            self.calls,
            "snapshot",
            {"desired_total": 0, "code_identity": code_identity},
        )
        self.ensure_reconcile_loop_started = _FakeRemoteMethod(
            self.calls,
            "ensure_reconcile_loop_started",
            {"desired_total": 0, "reconcile_loop_running": True, "code_identity": code_identity},
        )


class _FakeActorHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.snapshot = _FakeRemoteMethod(self.calls, "snapshot", {"desired_total": 0})
        self.async_snapshot = _FakeRemoteMethod(
            self.calls,
            "async_snapshot",
            {"desired_total": 0, "async": True},
        )
        self.register = _FakeRemoteMethod(
            self.calls,
            "register",
            {"actor_name": "actor-a", "metadata": {"launcher_contract": "model_actor_supervisor"}},
        )
        self.mark_ready = _FakeRemoteMethod(self.calls, "mark_ready", None)
        self.clear_session = _FakeRemoteMethod(self.calls, "clear_session", 3)
        self.sync_replicas = _FakeRemoteMethod(self.calls, "sync_replicas", {"ok": True})
        self.total_gpus_used = _FakeRemoteMethod(self.calls, "total_gpus_used", 7)
        self.gpus_used_by_node = _FakeRemoteMethod(self.calls, "gpus_used_by_node", {"node-a": 4})


def _disabled_control_plane_kwargs() -> dict:
    return {"control_plane_dependencies": [], "control_plane_enabled": False}


class _FakeRemoteActorClass:
    def __init__(self, created: dict) -> None:
        self.created = created

    def options(self, **options):
        self.created["options"] = options
        return self

    def remote(self, *args, **kwargs):
        self.created["remote_args"] = args
        self.created["remote_kwargs"] = kwargs
        return self.created["actor"]


@pytest.mark.anyio
async def test_issue_593_maybe_await_uses_ray_timeout_for_object_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    calls: list[tuple[object, float | None]] = []
    ref = _FakeRayRef({"ok": True})

    async def _fake_async_get_ray_ref(value, *, timeout_s=None):
        calls.append((value, timeout_s))
        return value.value

    monkeypatch.setattr(supervisor_module, "async_get_ray_ref", _fake_async_get_ray_ref, raising=False)

    out = await supervisor_module._maybe_await(ref)

    assert out == {"ok": True}
    assert calls == [(ref, 10.0)]


def test_issue_593_get_model_actor_supervisor_returns_client_facade() -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    assert isinstance(supervisor_module.get_model_actor_supervisor(), ModelActorSupervisorClient)
    assert supervisor_module.get_model_actor_supervisor() is supervisor_module.model_actor_supervisor
    assert not isinstance(supervisor_module.get_model_actor_supervisor(), ModelActorSupervisor)


def test_issue_593_supervisor_detached_actor_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    actor = _FakeActorHandle()
    created = {"actor": actor, "remote_args": None, "remote_kwargs": None, "remote_decorator": None}

    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"node:10.1.2.3": 1.0, "node:__internal_head__": 1.0},
    )

    def _remote(**kwargs):
        created["remote_decorator"] = kwargs
        return lambda _cls: _FakeRemoteActorClass(created)

    fake_ray.remote = _remote
    fake_ray_util = types.SimpleNamespace(get_node_ip_address=lambda: "10.1.2.3")
    monkeypatch.setitem(sys.modules, "ray.util", fake_ray_util)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("RAY_ADDRESS", "10.1.2.3:6379")
    monkeypatch.setattr(supervisor_module, "PFS_PYTHONPATH", "PFS_PATH", raising=False)
    monkeypatch.setattr(
        supervisor_module,
        "actor_runtime_env",
        lambda **kwargs: {
            "env_vars": {
                "PYTHONPATH": kwargs["pythonpath"],
                **dict(kwargs.get("extra") or {}),
            }
        },
        raising=False,
    )
    monkeypatch.setattr(supervisor_module, "otel_env_vars", lambda: {"OTEL_SERVICE_NAME": "mint-test"}, raising=False)
    monkeypatch.setattr(supervisor_module, "sync_get_ray_ref", lambda ref, **_kwargs: ref.value, raising=False)

    out = supervisor_module._create_ray_actor(require_ready=True)

    assert out is actor
    assert created["remote_decorator"] == {
        "num_cpus": 0,
        "max_concurrency": 128,
        "max_restarts": 0,
    }
    assert created["options"]["name"] == "mint_model_actor_supervisor"
    assert created["options"]["namespace"] == "mint"
    assert created["options"]["lifetime"] == "detached"
    assert created["options"]["get_if_exists"] is True
    assert created["options"]["runtime_env"] == {
        "env_vars": {
            "PYTHONPATH": "PFS_PATH",
            "OTEL_SERVICE_NAME": "mint-test",
            "MINT_GIT_SHA": supervisor_module.CURRENT_CODE_IDENTITY,
        }
    }
    assert created["options"]["resources"] == {"node:__internal_head__": 0.001}
    assert created["remote_kwargs"]["specs"] == desired_specs_from_env()
    assert created["remote_kwargs"]["ray_address"] == "10.1.2.3:6379"
    assert actor.calls == [("snapshot", (), {})]


def test_issue_593_model_runtime_max_claim_uses_training_override_for_megatron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM", "23")

    spec = SimpleNamespace(domain_key="megatron:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert _model_runtime_max_claim_for_spec(spec) == 23


def test_issue_638_supervisor_registers_actor_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    calls = {"count": 0}
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: calls.__setitem__("count", calls["count"] + 1))

    supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )

    assert calls["count"] == 1


def test_issue_638_supervisor_registers_otel_inventory_and_supervisor_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )

    expected = {
        "mint_model_actor_supervisor_desired_total",
        "mint_topology_node_state",
        "mint_topology_node_gpus",
        "mint_node_metrics_daemon_enabled",
        "mint_node_metrics_daemon_desired_total",
        "mint_node_metrics_daemon_managed_total",
        "mint_node_metrics_daemon_state",
        "mint_node_metrics_daemon_sample_count",
        "mint_node_metrics_daemon_error_count",
        "mint_model_actor_supervisor_domain_replicas",
        "mint_model_actor_supervisor_replica_state",
        "mint_model_actor_inventory_actors",
        "mint_model_actor_inventory_actor_gpu_binding",
        "mint_model_actor_inventory_actor_gpu_binding_missing_uuid",
        "mint_model_actor_inventory_actor_idle_time_s",
        "mint_model_actor_inventory_actor_age_s",
        "mint_model_actor_inventory_actor_rss_bytes",
        "mint_model_actor_inventory_actor_rss_sample_age_s",
        "mint_model_actor_inventory_actor_rss_cache_state",
        "mint_model_actor_inventory_group_oldest_idle_time_s",
        "mint_model_actor_inventory_group_oldest_age_s",
        "mint_model_actor_inventory_group_rss_bytes",
        "mint_model_actor_inventory_group_rss_cache_samples",
        "mint_model_actor_inventory_observability_cache_hits_total",
        "mint_model_actor_inventory_observability_cache_stale_total",
        "mint_model_actor_inventory_observability_refresh_success_total",
        "mint_model_actor_inventory_observability_refresh_failures_total",
    }
    assert expected.issubset(set(gauges))


def test_issue_638_supervisor_otel_inventory_callbacks_use_cached_snapshot_without_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_inventory as inventory_module
    import mint_server.backend.model_actor_supervisor as supervisor_module
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

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )
    supervisor.clear(kill_actors=False)
    supervisor.register(
        actor_name="mint_vllm_qwen3_0_6b",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        base_model="Qwen/Qwen3-0.6B",
        metadata={
            "gpu_bindings": [
                {
                    "hostname": "mint-worker-0",
                    "gpu_uuid": "GPU-test-0",
                    "gpu_index": 0,
                    "node_ip": "10.0.0.7",
                    "ray_node_id": "node-high-cardinality",
                    "last_error": "free-form error",
                }
            ]
        },
    )

    def _unexpected_refresh(*_args, **_kwargs):
        raise AssertionError("OTel callbacks must not refresh runtime actor metadata")

    monkeypatch.setattr(inventory_module.ModelActorInventory, "_refresh_entry_metadata", _unexpected_refresh)

    actor_count_obs = gauges["mint_model_actor_inventory_actors"][0](None)
    assert len(actor_count_obs) == 1
    assert actor_count_obs[0].value == 1.0
    assert actor_count_obs[0].attributes["actor_type"] == "vllm"
    assert actor_count_obs[0].attributes["model"] == "Qwen/Qwen3-0.6B"
    assert actor_count_obs[0].attributes["deployment.env"] == "prod"
    assert actor_count_obs[0].attributes["mint.cluster_id"] == "volcano"

    binding_obs = gauges["mint_model_actor_inventory_actor_gpu_binding"][0](None)
    assert len(binding_obs) == 1
    attrs = binding_obs[0].attributes
    assert attrs["actor_name"] == "mint_vllm_qwen3_0_6b"
    assert attrs["workload"] == "sample"
    assert attrs["hostname"] == "mint-worker-0"
    assert attrs["gpu_uuid"] == "GPU-test-0"
    assert "gpu_index" not in attrs
    assert "node_ip" not in attrs
    assert "ray_node_id" not in attrs
    assert "last_error" not in attrs


def test_issue_638_supervisor_otel_reports_missing_gpu_uuid_without_high_cardinality_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )
    supervisor.clear(kill_actors=False)
    supervisor.register(
        actor_name="mint_vllm_no_uuid",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        base_model="Qwen/Qwen3-0.6B",
        metadata={"hostname": "mint-worker-0", "gpu_indices": [0]},
    )

    assert gauges["mint_model_actor_inventory_actor_gpu_binding"][0](None) == []
    missing_obs = gauges["mint_model_actor_inventory_actor_gpu_binding_missing_uuid"][0](None)

    assert len(missing_obs) == 1
    assert missing_obs[0].value == 1.0
    attrs = missing_obs[0].attributes
    assert attrs["actor_name"] == "mint_vllm_no_uuid"
    assert attrs["workload"] == "sample"
    assert attrs["hostname"] == "mint-worker-0"
    assert "gpu_index" not in attrs
    assert "node_ip" not in attrs
    assert "ray_node_id" not in attrs


def test_issue_638_supervisor_rss_snapshot_populates_otel_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_inventory as inventory_module
    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeRemote:
        def remote(self):
            return "rss-ref"

    class _FakeHandle:
        get_rss_bytes = _FakeRemote()

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)
    monkeypatch.setattr(inventory_module.ray, "get", lambda ref, timeout=None: 4096)

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )
    supervisor.clear(kill_actors=False)
    supervisor.register(
        actor_name="mint_vllm_rss",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        actor_handle=_FakeHandle(),
        base_model="Qwen/Qwen3-0.6B",
    )

    assert supervisor.rss_snapshot(timeout_s=1.0)[0]["rss_bytes"] == 4096
    rss_obs = gauges["mint_model_actor_inventory_actor_rss_bytes"][0](None)
    group_rss_obs = gauges["mint_model_actor_inventory_group_rss_bytes"][0](None)

    assert rss_obs[0].value == 4096.0
    assert rss_obs[0].attributes["actor_name"] == "mint_vllm_rss"
    assert group_rss_obs[0].value == 4096.0


def test_issue_638_supervisor_update_metadata_uses_sample_source() -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )
    supervisor.clear(kill_actors=False)
    supervisor.register(
        actor_name="mint_dense_actor",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="Qwen/Qwen3-0.6B",
    )

    assert supervisor.update_metadata(
        "mint_dense_actor",
        {"poisoned": True},
        sample_time=123.0,
        sample_source="dense_retire",
    ) is True
    rec = supervisor.cached_snapshot(refresh_metadata=False)[0]

    assert rec["metadata"] == {"poisoned": True}
    assert rec["metadata_sample_source"] == "dense_retire"


def test_issue_638_supervisor_otel_callbacks_emit_supervisor_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Qwen3-0.6B",
                replica_id="replica-0",
                actor_name="mint_vllm_qwen3_0_6b",
                base_model="Qwen/Qwen3-0.6B",
            )
        ],
        state_store=SupervisorMemoryStateStore(),
        node_metrics_enabled=False,
        **_disabled_control_plane_kwargs(),
    )
    supervisor._states[("vllm:Qwen/Qwen3-0.6B", "replica-0")].update(
        {"state": "healthy", "actor_name": "mint_vllm_qwen3_0_6b", "generation": 7}
    )

    desired_obs = gauges["mint_model_actor_supervisor_desired_total"][0](None)
    assert desired_obs[0].value == 1.0

    domain_obs = gauges["mint_model_actor_supervisor_domain_healthy"][0](None)
    assert domain_obs[0].value == 1.0
    assert domain_obs[0].attributes["domain_key"] == "vllm:Qwen/Qwen3-0.6B"

    replica_obs = gauges["mint_model_actor_supervisor_replica_generation"][0](None)
    assert replica_obs[0].value == 7.0
    assert replica_obs[0].attributes["actor_name"] == "mint_vllm_qwen3_0_6b"
    assert replica_obs[0].attributes["state"] == "healthy"


def test_issue_638_supervisor_otel_callbacks_emit_topology_and_node_daemon_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.logging_context as logging_context

    gauges: dict[str, list] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            gauges[name] = list(kwargs.get("callbacks") or [])

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "init_actor_observability", lambda: None)

    topology = TopologyConfig(
        version=1,
        deployment_env="prod",
        cluster_id="volcano",
        state_path=str(tmp_path / "topology_state.yaml"),
        providers={"volcano": {"templates": {"gpu": {}}}},
        nodes={
            "mint-worker-0": TopologyNodeDesired(
                alias="mint-worker-0",
                provider="volcano",
                template="gpu",
                gpu_count=8,
            )
        },
        models={},
    )

    supervisor = supervisor_module.ModelActorSupervisor(
        specs=[],
        state_store=SupervisorMemoryStateStore(),
        topology_manager=TopologyManager(
            config=topology,
            provider_task_lister=lambda _config: [
                ProviderTaskState(
                    alias="mint-worker-0",
                    provider="volcano",
                    task_name="mint-prod-worker-1",
                    live=True,
                    node_ip="10.0.0.7",
                    gpu_count=8,
                )
            ],
            ray_node_lister=lambda: [
                RayNodeState(
                    node_ip="10.0.0.7",
                    ray_node_id="node-0",
                    alive=True,
                    gpu_count=8,
                )
            ],
        ),
        node_metrics_enabled=True,
        **_disabled_control_plane_kwargs(),
    )
    supervisor._topology_manager.reconcile_once()
    supervisor._node_metric_states["mint-worker-0"] = {
        "state": "healthy",
        "health": {"sample_count": 3, "error_count": 1},
    }
    supervisor._node_metric_actors["mint-worker-0"] = object()

    topology_state = gauges["mint_topology_node_state"][0](None)
    assert topology_state[0].value == 1.0
    assert topology_state[0].attributes["worker_alias"] == "mint-worker-0"
    assert topology_state[0].attributes["provider"] == "volcano"

    topology_gpus = gauges["mint_topology_node_gpus"][0](None)
    assert topology_gpus[0].value == 8.0

    assert gauges["mint_node_metrics_daemon_enabled"][0](None)[0].value == 1.0
    assert gauges["mint_node_metrics_daemon_desired_total"][0](None)[0].value == 1.0
    assert gauges["mint_node_metrics_daemon_managed_total"][0](None)[0].value == 1.0

    daemon_state = gauges["mint_node_metrics_daemon_state"][0](None)
    assert daemon_state[0].value == 1.0
    assert daemon_state[0].attributes["worker_alias"] == "mint-worker-0"
    assert daemon_state[0].attributes["state"] == "healthy"
    assert gauges["mint_node_metrics_daemon_sample_count"][0](None)[0].value == 3.0
    assert gauges["mint_node_metrics_daemon_error_count"][0](None)[0].value == 1.0


def test_issue_593_supervisor_ensure_recreates_stale_code_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    old_actor = _FakeKillableActorHandle(code_identity="old-sha")
    new_actor = _FakeKillableActorHandle(code_identity=supervisor_module.CURRENT_CODE_IDENTITY)
    created = {"count": 0}
    killed: list[tuple[object, bool]] = []

    def _create(*, require_ready=True):
        del require_ready
        created["count"] += 1
        return old_actor if created["count"] == 1 else new_actor

    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(supervisor_module, "_create_ray_actor", _create, raising=False)
    monkeypatch.setattr(
        supervisor_module,
        "_kill_supervisor_actor",
        lambda actor, **_kwargs: killed.append((actor, True)),
        raising=False,
    )
    monkeypatch.setattr(supervisor_module, "sync_get_ray_ref", lambda ref, **_kwargs: ref.value, raising=False)

    out = ModelActorSupervisorClient().ensure_started(timeout_s=2.0)

    assert out["reconcile_loop_running"] is True
    assert created["count"] == 2
    assert killed == [(old_actor, True)]
    assert old_actor.calls == [("snapshot", (), {})]
    assert new_actor.calls == [("ensure_reconcile_loop_started", (), {})]


@pytest.mark.anyio
async def test_issue_593_supervisor_client_forwards_sync_and_async_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module

    actor = _FakeActorHandle()
    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
        get_actor=lambda name, namespace=None: actor,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(supervisor_module, "sync_get_ray_ref", lambda ref, **_kwargs: ref.value, raising=False)

    async def _async_get(ref, **_kwargs):
        return ref.value

    monkeypatch.setattr(supervisor_module, "async_get_ray_ref", _async_get, raising=False)
    client = ModelActorSupervisorClient()

    assert client.snapshot(timeout_s=2.0) == {"desired_total": 0}
    assert await client.async_snapshot(timeout_s=3.0) == {"desired_total": 0, "async": True}
    assert client.total_gpus_used() == 7
    assert client.gpus_used_by_node() == {"node-a": 4}
    assert client.clear_session("session-a", actor_type=ActorType.VLLM) == 3
    assert await client.sync_replicas() == {"ok": True}
    client.mark_ready("actor-a")

    assert ("snapshot", (), {}) in actor.calls
    assert ("async_snapshot", (), {"timeout_s": 3.0}) in actor.calls
    assert ("total_gpus_used", (), {}) in actor.calls
    assert ("gpus_used_by_node", (), {}) in actor.calls
    assert ("clear_session", ("session-a",), {"actor_type": ActorType.VLLM}) in actor.calls
    assert ("sync_replicas", (), {}) in actor.calls
    assert ("mark_ready", ("actor-a",), {}) in actor.calls


def test_issue_593_supervisor_exposes_explicit_inventory_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mint_server.backend.model_actor_inventory.ray.is_initialized", lambda: False)
    supervisor = ModelActorSupervisor(**_disabled_control_plane_kwargs())

    entry = supervisor.register(
        actor_name="vllm-contract-actor",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        base_model="model-a",
        metadata={"launcher_key": "vllm"},
    )
    supervisor.mark_ready("vllm-contract-actor")
    supervisor.mark_inflight("vllm-contract-actor", +1)
    supervisor.set_session("vllm-contract-actor", "session-a")

    assert entry.metadata["launcher_contract"] == "model_actor_supervisor"
    current = supervisor.get("vllm-contract-actor")
    assert current is not None
    assert current.current_session == "session-a"
    assert current.inflight_count == 1
    listed = supervisor.list_actors()
    assert any(row["actor_name"] == "vllm-contract-actor" for row in listed)
    assert supervisor.total_gpus_used() >= 1

    assert supervisor.clear_session("session-a", actor_type=ActorType.VLLM) == 1
    assert supervisor.unregister("vllm-contract-actor") is True


def test_bumblebee_actor_inventory_reports_bumblebee_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mint_server.backend.model_actor_inventory.ray.is_initialized", lambda: False)
    supervisor = ModelActorSupervisor(**_disabled_control_plane_kwargs())
    actor_name = "mint_bumblebee_qwen3_30b_a3b_instruct_2507"

    supervisor.register(
        actor_name=actor_name,
        actor_type=ActorType.MEGATRON,
        num_gpus=4,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    listed = supervisor.list_actors()
    row = next(item for item in listed if item["actor_name"] == actor_name)
    assert row["actor_type"] == "megatron"
    assert row["backend"] == "bumblebee"


def test_issue_593_supervisor_memory_state_backend_owner_and_events() -> None:
    store = SupervisorMemoryStateStore(event_limit=2)
    owner = store.acquire_owner(name="model_actor_supervisor", owner_id="owner-a", ttl_s=30, now=10)
    assert owner == {
        "name": "model_actor_supervisor",
        "owner_id": "owner-a",
        "epoch": 1,
        "started_at": 10.0,
        "last_heartbeat_at": 10.0,
        "lease_until": 40.0,
        "schema_version": 1,
    }
    with pytest.raises(SupervisorStateOwnerConflictError):
        store.acquire_owner(name="model_actor_supervisor", owner_id="owner-b", ttl_s=30, now=11)

    renewed = store.heartbeat_owner(
        name="model_actor_supervisor",
        owner_id="owner-a",
        epoch=1,
        ttl_s=30,
        now=12,
    )
    assert renewed["last_heartbeat_at"] == 12.0
    assert renewed["lease_until"] == 42.0

    store.append_event("a", {"n": 1}, owner=renewed, now=13)
    store.append_event("b", {"n": 2}, owner=renewed, now=14)
    store.append_event("c", {"n": 3}, owner=renewed, now=15)
    assert [event["event_type"] for event in store.list_events(limit=10)] == ["b", "c"]


def test_issue_593_supervisor_sqlite_state_backend_schema_and_generation(tmp_path) -> None:
    db_path = tmp_path / "supervisor_state.sqlite3"
    store = SupervisorSQLiteStateStore(db_path, event_limit=2)
    owner = store.acquire_owner(name="model_actor_supervisor", owner_id="owner-a", ttl_s=30, now=10)
    assert owner["schema_version"] == 1
    assert store.reserve_generation("generation:vllm:model-a::replica-0", floor=100, now=11) == 100
    assert store.reserve_generation("generation:vllm:model-a::replica-0", floor=100, now=12) == 101
    store.append_event("a", {"n": 1}, owner=owner, now=13)
    store.append_event("b", {"n": 2}, owner=owner, now=14)
    store.append_event("c", {"n": 3}, owner=owner, now=15)
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {"kv", "owner", "events"}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(owner)")}
        assert {
            "owner_id",
            "epoch",
            "started_at",
            "last_heartbeat_at",
            "lease_until",
            "schema_version",
        }.issubset(columns)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    finally:
        conn.close()


def test_issue_593_supervisor_uses_sqlite_generation_hint_across_instances(tmp_path) -> None:
    db_path = tmp_path / "supervisor_state.sqlite3"
    store_a = SupervisorSQLiteStateStore(db_path)
    supervisor_a = ModelActorSupervisor(
        specs=[ModelActorSpec(domain_key="vllm:model-a", replica_id="replica-0")],
        state_store=store_a,
        owner_id="owner-a",
        owner_ttl_s=1,
        **_disabled_control_plane_kwargs(),
    )
    generation_a = supervisor_a._next_generation(("vllm:model-a", "replica-0"))
    store_a.close()

    store_b = SupervisorSQLiteStateStore(db_path)
    supervisor_b = ModelActorSupervisor(
        specs=[ModelActorSpec(domain_key="vllm:model-a", replica_id="replica-0")],
        state_store=store_b,
        owner_id="owner-a",
        owner_ttl_s=1,
        **_disabled_control_plane_kwargs(),
    )
    generation_b = supervisor_b._next_generation(("vllm:model-a", "replica-0"))
    store_b.close()

    assert generation_b > generation_a


@pytest.mark.anyio
async def test_issue_593_supervisor_reconciles_control_plane_dependencies() -> None:
    calls: list[tuple[str, str]] = []

    async def _ensure_scheduler() -> dict:
        calls.append(("model_work_scheduler", "ensure"))
        return {"ok": True}

    async def _ping_scheduler() -> dict:
        calls.append(("model_work_scheduler", "ping"))
        return {"ok": True}

    async def _ensure_task_state() -> dict:
        calls.append(("task_state_store", "ensure"))
        return {"ok": True}

    async def _ping_task_state() -> dict:
        calls.append(("task_state_store", "ping"))
        return {"ok": True}

    dependencies = [
        ControlPlaneDependency("model_work_scheduler", _ensure_scheduler, _ping_scheduler),
        ControlPlaneDependency("task_state_store", _ensure_task_state, _ping_task_state),
    ]
    supervisor = ModelActorSupervisor(
        specs=[],
        control_plane_dependencies=dependencies,
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    assert calls == [
        ("model_work_scheduler", "ensure"),
        ("task_state_store", "ensure"),
    ]
    assert out["snapshot"]["control_plane"]["dependencies"]["model_work_scheduler"]["state"] == "ready"
    assert out["snapshot"]["control_plane"]["dependencies"]["task_state_store"]["state"] == "ready"


@pytest.mark.anyio
async def test_issue_593_supervisor_bootstrap_starts_reconcile_loop() -> None:
    calls: list[str] = []

    async def _ensure_dependency() -> dict:
        calls.append("ensure")
        return {"ok": True}

    dependency = ControlPlaneDependency(
        "task_state_store",
        _ensure_dependency,
        lambda: {"ok": True},
    )
    supervisor = ModelActorSupervisor(
        specs=[],
        control_plane_dependencies=[dependency],
        scheduler_sync=lambda _registrations: None,
        reconcile_interval_s=3600.0,
    )

    out = await supervisor.ensure_reconcile_loop_started()
    try:
        assert calls == []
        assert out["reconcile_loop_running"] is True
    finally:
        task = supervisor._reconcile_task
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                pass


@pytest.mark.anyio
async def test_issue_593_supervisor_recovers_stale_reconcile_inflight() -> None:
    calls = 0

    async def _sync(_registrations):
        nonlocal calls
        calls += 1

    supervisor = ModelActorSupervisor(
        specs=[],
        scheduler_sync=_sync,
        reconcile_interval_s=1.0,
        **_disabled_control_plane_kwargs(),
    )
    supervisor._reconcile_inflight = True
    supervisor._reconcile_inflight_started_at = time.time() - 60.0

    out = await supervisor.reconcile_once()

    assert out["ok"] is True
    assert calls == 1
    assert supervisor.snapshot()["reconcile_inflight"] is False
    assert "reconcile_inflight stale" in (supervisor.snapshot()["last_reconcile_loop_error"] or "")


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
        **_disabled_control_plane_kwargs(),
    )

    await supervisor.reconcile_once()

    assert len(launched) == 1
    assert launched[0][0].domain_key == "vllm:model-a"
    snapshot = supervisor.snapshot()["replicas"]
    assert snapshot["vllm:model-a::replica-0"]["launcher_key"] == "test_launcher"


def test_issue_648_vllm_runtime_max_claim_is_high_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM", raising=False)
    monkeypatch.delenv("MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_MAX_CLAIM", raising=False)

    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="vllm:model-a")) == 64
    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="training:model-a")) == 1
    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 16
    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="megatron:model-a")) == 16

    monkeypatch.setenv("MINT_VLLM_MODEL_RUNTIME_MAX_CLAIM", "17")
    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="vllm:model-a")) == 17
    monkeypatch.setenv("MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM", "7")
    assert _model_runtime_max_claim_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 7


def test_issue_648_training_runtime_token_budget_uses_backend_then_training_then_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET",
        "MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET",
        "MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET",
        "MINT_MODEL_RUNTIME_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(key, raising=False)

    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="vllm:model-a")) is None
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 262144
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="megatron:model-a")) is None

    monkeypatch.setenv("MINT_MODEL_RUNTIME_TOKEN_BUDGET", "1000")
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 1000
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="megatron:model-a")) == 1000
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="training:model-a")) == 1000

    monkeypatch.setenv("MINT_TRAINING_MODEL_RUNTIME_TOKEN_BUDGET", "2000")
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 2000
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="megatron:model-a")) == 2000
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="training:model-a")) == 1000

    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET", "262144")
    monkeypatch.setenv("MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET", "131072")
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 262144
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="megatron:model-a")) == 131072

    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET", "0")
    monkeypatch.setenv("MINT_MEGATRON_MODEL_RUNTIME_TOKEN_BUDGET", "not-an-int")
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="bumblebee:model-a")) == 2000
    assert _model_runtime_token_budget_for_spec(ModelActorSpec(domain_key="megatron:model-a")) == 2000


@pytest.mark.anyio
async def test_issue_648_launch_model_engine_host_passes_training_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_get_or_create_model_engine_host(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    runtime_module = types.ModuleType("mint_server.backend.model_engine_host")
    runtime_module.get_or_create_model_engine_host = _fake_get_or_create_model_engine_host
    monkeypatch.setitem(sys.modules, "mint_server.backend.model_engine_host", runtime_module)
    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET", "262144")
    monkeypatch.setenv("MINT_TRAINING_MODEL_RUNTIME_MAX_CLAIM", "16")

    actor = await launch_model_engine_host(
        ModelActorSpec(
            domain_key="bumblebee:Qwen/Qwen3-30B-A3B-Instruct-2507",
            replica_id="replica-0",
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
        generation=12,
    )

    assert actor.ok is True
    assert captured["max_claim"] == 16
    assert captured["token_budget"] == 262144
    assert captured["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.mark.anyio
async def test_issue_593_supervisor_creates_replica_and_syncs_scheduler() -> None:
    created: list[_FakeRuntimeActor] = []
    events: list[str] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        events.append("factory")
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _sync(registrations):
        events.append("sync")
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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    assert out["ok"] is True
    assert len(created) == 1
    assert created[0].start_calls == 1
    assert events[:3] == ["sync", "sync", "factory"]
    assert synced[0][0]["domain_key"] == "vllm:model-a"
    assert synced[0][0]["status"] == "starting"
    assert synced[0][0]["generation"] == 0
    assert synced[1][0]["status"] == "healthy"
    assert synced[1][0]["generation"] >= 1
    assert synced[1][0]["consumer_id"].endswith(f"generation::{synced[1][0]['generation']}")
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
    async_snapshot = await supervisor.async_snapshot()
    assert async_snapshot["replicas"] == supervisor.snapshot()["replicas"]
    assert isinstance(async_snapshot["snapshot_generated_at"], float)


@pytest.mark.anyio
async def test_issue_593_supervisor_keeps_runtime_claimable_when_start_times_out() -> None:
    created: list[_FakeStartTimeoutRuntimeActor] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeStartTimeoutRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

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
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert out["ok"] is True
    assert len(created) == 1
    assert created[0].start_calls == 1
    assert supervisor._actors[("vllm:model-a", "replica-0")] is created[0]
    replica = out["snapshot"]["replicas"][label]
    assert replica["state"] == "starting"
    assert replica["last_action"] == "start_pending:missing"
    assert "GetTimeoutError" in replica["last_error"]
    assert synced[-1][0]["status"] == "healthy"
    assert synced[-1][0]["generation"] == replica["generation"]


@pytest.mark.anyio
async def test_issue_593_supervisor_shuts_down_pending_start_when_sync_fails() -> None:
    created: list[_FakeStartTimeoutRuntimeActor] = []
    sync_calls = 0

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeStartTimeoutRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _sync(_registrations):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 3:
            raise RuntimeError("pending start sync unavailable")

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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert out["ok"] is True
    assert len(created) == 1
    assert created[0].shutdown_calls == 1
    assert supervisor._actors == {}
    assert out["snapshot"]["scheduler_sync_failures_total"] == 1
    assert out["snapshot"]["replicas"][label]["state"] == "dead"
    assert "pending start sync unavailable" in out["snapshot"]["replicas"][label]["last_error"]


@pytest.mark.anyio
async def test_issue_593_supervisor_does_not_start_runtime_when_reserved_sync_fails() -> None:
    created: list[_FakeRuntimeActor] = []
    sync_calls = 0

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _sync(_registrations):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise RuntimeError("scheduler unavailable")

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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert out["ok"] is True
    assert created == []
    assert sync_calls == 3
    assert out["snapshot"]["scheduler_sync_failures_total"] == 1
    assert out["snapshot"]["replicas"][label]["state"] == "dead"
    assert "scheduler unavailable" in out["snapshot"]["replicas"][label]["last_error"]


@pytest.mark.anyio
async def test_issue_593_supervisor_shuts_down_runtime_when_healthy_sync_fails() -> None:
    created: list[_FakeRuntimeActor] = []
    sync_calls = 0

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    async def _sync(_registrations):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 3:
            raise RuntimeError("healthy sync unavailable")

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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert out["ok"] is True
    assert len(created) == 1
    assert created[0].start_calls == 1
    assert created[0].shutdown_calls == 1
    assert created[0].running is False
    assert supervisor._actors == {}
    assert out["snapshot"]["scheduler_sync_failures_total"] == 1
    assert out["snapshot"]["replicas"][label]["state"] == "dead"
    assert "healthy sync unavailable" in out["snapshot"]["replicas"][label]["last_error"]


@pytest.mark.anyio
async def test_issue_593_supervisor_does_not_keep_claimable_status_after_start_failure() -> None:
    synced: list[list[dict]] = []

    class _FailingStartRuntime(_FakeRuntimeActor):
        def start(self) -> dict:
            self.start_calls += 1
            raise RuntimeError("runtime start failed")

    async def _factory(spec: ModelActorSpec, generation: int):
        return _FailingStartRuntime(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert out["ok"] is True
    assert synced[1][0]["status"] == "healthy"
    assert synced[-1][0]["status"] == "dead"
    assert out["snapshot"]["replicas"][label]["state"] == "dead"
    assert "runtime start failed" in out["snapshot"]["replicas"][label]["last_error"]


@pytest.mark.anyio
async def test_issue_593_supervisor_projects_scheduler_backlog_to_desired_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_model = "Qwen/Qwen3-0.6B"
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", base_model)
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {base_model: {"training": {"placement": [{"replica": 0, "node_ip": "10.0.0.8", "gpu_count": 1}]}}},
        ),
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
        **_disabled_control_plane_kwargs(),
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
        **_disabled_control_plane_kwargs(),
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
async def test_issue_593_supervisor_preserves_busy_runtime_on_health_timeout() -> None:
    from ray.exceptions import GetTimeoutError

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
        **_disabled_control_plane_kwargs(),
    )
    await supervisor.reconcile_once()
    first_generation = created[0].generation
    # A health-probe timeout means alive-but-busy, not dead: the runtime must be
    # neither killed nor recreated, and its scheduler registration must stay
    # claimable so the in-flight lease is not requeued as replica_unclaimable.
    created[0].health_errors.append(GetTimeoutError("timed out after 5.000s"))

    out = await supervisor.reconcile_once()

    assert len(created) == 1
    assert supervisor._actors[("vllm:model-a", "replica-0")] is created[0]
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["last_action"] == "health_timeout_preserved"
    assert out["snapshot"]["restarted_total"] == 0
    assert out["snapshot"]["health_timeout_preserved_total"] == 1
    assert synced[-1][0]["status"] == "healthy"
    assert synced[-1][0]["generation"] == first_generation


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
        **_disabled_control_plane_kwargs(),
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
async def test_issue_593_supervisor_ignores_stale_generation_health_failure() -> None:
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
        **_disabled_control_plane_kwargs(),
    )
    await supervisor.reconcile_once()
    stale_actor = created[0]
    stale_generation = stale_actor.generation
    next_generation = stale_generation + 1
    supervisor._generations[stale_actor.domain_key, stale_actor.replica_id] = next_generation
    supervisor._states[stale_actor.domain_key, stale_actor.replica_id] = {
        **supervisor._states[stale_actor.domain_key, stale_actor.replica_id],
        "state": "starting",
        "generation": next_generation,
        "consumer_id": "vllm:model-a::replica-0::generation::next",
        "scheduler_status": "healthy",
        "last_action": "reserve:dead",
    }
    stale_actor.health_errors.append(TimeoutError("metadata timeout"))
    supervisor._actors[stale_actor.domain_key, stale_actor.replica_id] = _OpaqueRuntimeHandle(stale_actor)

    out = await supervisor.reconcile_once()

    label = "vllm:model-a::replica-0"
    assert len(created) == 1
    assert out["snapshot"]["replicas"][label]["state"] == "starting"
    assert out["snapshot"]["replicas"][label]["generation"] == next_generation
    assert out["snapshot"]["replicas"][label]["last_action"] == "reserve:dead"
    assert synced[-1][0]["status"] == "healthy"
    assert synced[-1][0]["generation"] == next_generation


@pytest.mark.anyio
async def test_issue_593_supervisor_restarts_runtime_with_unrecovered_execution_error() -> None:
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
        **_disabled_control_plane_kwargs(),
    )
    await supervisor.reconcile_once()
    first_generation = created[0].generation
    created[0].last_error = "future failed: engine startup failed"
    created[0].failed_total = 2
    created[0].processed_total = 2
    created[0].completed_total = 0

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
async def test_issue_593_supervisor_keeps_runtime_with_recovered_execution_error() -> None:
    created: list[_FakeRuntimeActor] = []

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
        scheduler_sync=lambda _registrations: None,
        **_disabled_control_plane_kwargs(),
    )
    await supervisor.reconcile_once()
    created[0].last_error = "future failed: transient"
    created[0].failed_total = 1
    created[0].processed_total = 2
    created[0].completed_total = 1

    out = await supervisor.reconcile_once()

    assert len(created) == 1
    replica = out["snapshot"]["replicas"]["vllm:model-a::replica-0"]
    assert replica["state"] == "healthy"
    assert replica["last_error"] is None


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
        **_disabled_control_plane_kwargs(),
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
        **_disabled_control_plane_kwargs(),
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
    }


def test_issue_593_supervisor_builds_desired_specs_from_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {"vllm": {}, "training": {}},
                "Qwen/B": {"vllm": {}, "training": {}},
            },
        ),
    )

    specs = desired_specs_from_env()

    assert [spec.domain_key for spec in specs] == [
        "vllm:Qwen/A",
        "training:Qwen/A",
        "vllm:Qwen/B",
        "training:Qwen/B",
        "internal:runtime",
    ]
    assert domain_key_for_vllm_base_model("Qwen/A") == "vllm:Qwen/A"
    assert queue_id_for_replica("vllm:Qwen/A", "replica-2") == "vllm:Qwen/A::replica-2"


def test_issue_593_topology_specs_inherit_runtime_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {
                    "vllm": {"placement": [{"replica": 1, "node_ip": "10.0.0.7", "gpu_count": 2}]},
                    "training": {"placement": [{"replica": 0, "node_ip": "10.0.0.8", "gpu_count": 1}]},
                }
            },
        ),
    )

    specs = desired_specs_from_env()

    assert specs[:2] == [
        ModelActorSpec(
            domain_key="vllm:Qwen/A",
            replica_id="replica-1",
            base_model="Qwen/A",
            launcher_key="vllm",
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


def test_issue_593_topology_specs_accept_bumblebee_training_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                base_model: {
                    "bumblebee": {
                        "placement": [
                            {"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 4},
                            {"replica": 0, "worker_alias": "mint-worker-1", "gpu_count": 4},
                        ]
                    }
                }
            },
        ),
    )

    specs = desired_specs_from_env()
    training_specs = [spec for spec in specs if spec.base_model == base_model and spec.launcher_key == "training"]

    assert training_specs == [
        ModelActorSpec(
            domain_key="bumblebee:mint_megatron_qwen3_30b_a3b_instruct_2507",
            base_model=base_model,
            launcher_key="training",
            worker_aliases=("mint-worker-0", "mint-worker-1"),
            placement_alias_slices=(
                ("replica-0", "mint-worker-0", 4),
                ("replica-0", "mint-worker-1", 4),
            ),
            gpu_count=4,
        )
    ]
    assert training_specs[0].normalized_actor_name().startswith("mint_model_runtime_bumblebee-")


def test_issue_593_topology_specs_accept_qwen35_bumblebee_training_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_model = "Qwen/Qwen3.5-27B"
    monkeypatch.delenv("MINT_QWEN35_TRAINING_BACKEND", raising=False)
    monkeypatch.delenv("MINT_MOE_TRAINING_BACKEND", raising=False)
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                base_model: {
                    "bumblebee": {
                        "placement": [
                            {"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 8},
                        ]
                    }
                }
            },
        ),
    )

    specs = desired_specs_from_env()
    training_specs = [spec for spec in specs if spec.base_model == base_model and spec.launcher_key == "training"]

    assert training_specs == [
        ModelActorSpec(
            domain_key="bumblebee:mint_megatron_qwen3_5_27b",
            base_model=base_model,
            launcher_key="training",
            worker_aliases=("mint-worker-0",),
            placement_alias_slices=(("replica-0", "mint-worker-0", 8),),
            gpu_count=8,
        )
    ]
    assert training_specs[0].normalized_actor_name().startswith("mint_model_runtime_bumblebee-")


def test_topology_legacy_megatron_launcher_follows_selected_moe_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    monkeypatch.delenv("MINT_QWEN3_30B_TRAINING_BACKEND", raising=False)
    monkeypatch.delenv("MINT_MOE_TRAINING_BACKEND", raising=False)
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                base_model: {
                    "megatron": {
                        "placement": [
                            {"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 4},
                            {"replica": 0, "worker_alias": "mint-worker-1", "gpu_count": 4},
                        ]
                    }
                }
            },
        ),
    )

    specs = desired_specs_from_env()
    training_specs = [spec for spec in specs if spec.base_model == base_model and spec.launcher_key == "training"]

    assert training_specs == [
        ModelActorSpec(
            domain_key="bumblebee:mint_megatron_qwen3_30b_a3b_instruct_2507",
            base_model=base_model,
            launcher_key="training",
            worker_aliases=("mint-worker-0", "mint-worker-1"),
            placement_alias_slices=(
                ("replica-0", "mint-worker-0", 4),
                ("replica-0", "mint-worker-1", 4),
            ),
            gpu_count=4,
        )
    ]
    assert training_specs[0].normalized_actor_name().startswith("mint_model_runtime_bumblebee-")


def test_topology_legacy_megatron_launcher_can_roll_back_to_megatron_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    monkeypatch.setenv("MINT_QWEN3_30B_TRAINING_BACKEND", "megatron")
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                base_model: {
                    "megatron": {"placement": [{"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 4}]}
                }
            },
        ),
    )

    specs = desired_specs_from_env()
    training_specs = [spec for spec in specs if spec.base_model == base_model and spec.launcher_key == "training"]

    assert training_specs[0].domain_key == "megatron:mint_megatron_qwen3_30b_a3b_instruct_2507"
    assert training_specs[0].normalized_actor_name().startswith("mint_model_runtime_megatron-")


def test_issue_593_topology_specs_preserve_multi_node_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {
                    "vllm": {
                        "placement": [
                            {"replica": 0, "node_ip": "10.0.0.7", "gpu_count": 4},
                            {"replica": 0, "node_ip": "10.0.0.8", "gpu_count": 4},
                        ]
                    },
                    "training": {"placement": [{"replica": 0, "node_ip": "10.0.0.9", "gpu_count": 1}]},
                }
            },
        ),
    )

    specs = desired_specs_from_env()

    assert specs[0] == ModelActorSpec(
        domain_key="vllm:Qwen/A",
        base_model="Qwen/A",
        launcher_key="vllm",
        node_pins=("10.0.0.7", "10.0.0.8"),
        placement_slices=(("replica-0", "10.0.0.7", 4), ("replica-0", "10.0.0.8", 4)),
        gpu_count=4,
    )


def test_issue_593_topology_specs_split_distinct_replicas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {
                    "vllm": {
                        "placement": [
                            {"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 2},
                            {"replica": 1, "worker_alias": "mint-worker-1", "gpu_count": 2},
                        ]
                    },
                }
            },
        ),
    )

    specs = desired_specs_from_env()

    assert specs[:2] == [
        ModelActorSpec(
            domain_key="vllm:Qwen/A",
            replica_id="replica-0",
            base_model="Qwen/A",
            launcher_key="vllm",
            worker_aliases=("mint-worker-0",),
            placement_alias_slices=(("replica-0", "mint-worker-0", 2),),
            gpu_count=2,
        ),
        ModelActorSpec(
            domain_key="vllm:Qwen/A",
            replica_id="replica-1",
            base_model="Qwen/A",
            launcher_key="vllm",
            worker_aliases=("mint-worker-1",),
            placement_alias_slices=(("replica-1", "mint-worker-1", 2),),
            gpu_count=2,
        ),
    ]


def test_issue_593_topology_placement_without_gpu_count_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {
                    "vllm": {"placement": [{"replica": 0, "node_ip": "10.0.0.7"}]},
                }
            },
        ),
    )

    with pytest.raises(ValueError, match="must include gpu_count"):
        desired_specs_from_env()


def test_issue_593_topology_specs_preserve_disabled_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {
                "Qwen/A": {
                    "vllm": {"enabled": False},
                }
            },
        ),
    )

    specs = desired_specs_from_env()

    assert specs[0] == ModelActorSpec(
        domain_key="vllm:Qwen/A",
        base_model="Qwen/A",
        launcher_key="vllm",
        enabled=False,
    )


def test_issue_593_topology_specs_reject_bad_placement_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MINT_TOPOLOGY_CONFIG_PATH",
        _write_supervisor_topology(
            tmp_path,
            {"Qwen/A": {"vllm": {"placement": "not-a-list"}}},
        ),
    )

    with pytest.raises(ValueError, match="must be a list"):
        desired_specs_from_env()


def test_issue_593_supervisor_empty_env_has_no_desired_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_TOPOLOGY_CONFIG_PATH", raising=False)

    assert desired_specs_from_env() == [
        ModelActorSpec(
            domain_key="internal:runtime",
            launcher_key="cpu_runtime",
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
        namespace="mint",
        actor_exists=lambda _name, _namespace: False,
        actor_killer=lambda name, namespace, reason: killed.append((name, namespace, reason)) or True,
        actor_lister=lambda: [
            {"name": "mint_model_runtime_old", "namespace": "mint"},
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
    assert ("mint_model_runtime_old", "mint", "model_actor_supervisor_undesired_wrapper") in killed
    assert capacity_checks == [
        {
            "required": {"10.0.0.17": 4},
            "context": "model_actor_supervisor placement domain='vllm:Qwen/Test' replica='replica-1'",
            "ignore_pg_names": {
                "mint_model_runtime_vllm-Qwen-Test_replica-1_pg",
                "mint_model_runtime_vllm-Qwen-Test_replica-1_mint_pg",
                "mint_vllm_test_pg",
                "mint_vllm_test_mint_pg",
            },
            "namespace": "mint",
        }
    ]


def test_issue_593_placement_reconciler_kills_orphan_mint_gpu_actor() -> None:
    killed: list[tuple[str, str, str]] = []

    reconciler = ModelActorPlacementReconciler(
        namespace="mint",
        actor_exists=lambda _name, _namespace: False,
        gpu_actor_killer=lambda actor, reason: killed.append(
            (str(actor["name"]), str(actor.get("namespace") or "mint"), reason)
        )
        or True,
        actor_lister=lambda: [],
        gpu_actor_lister=lambda: [
            {
                "name": "mint_openpi_shared_deadbeef",
                "namespace": "mint",
                "node_ip": "10.0.0.8",
                "gpu": 1,
            },
            {
                "name": "mint_openpi_shared_registered",
                "namespace": "mint",
                "node_ip": "10.0.0.8",
                "gpu": 1,
            },
            {
                "name": "foreign_actor",
                "namespace": "mint",
                "node_ip": "10.0.0.8",
                "gpu": 1,
            },
        ],
        placement_group_remover=lambda _name, _namespace: False,
    )

    out = reconciler(
        {},
        protected_actor_names={"mint_openpi_shared_registered"},
    )

    assert out["cleaned_gpu_actor_names"] == ["mint_openpi_shared_deadbeef"]
    assert killed == [
        (
            "mint_openpi_shared_deadbeef",
            "mint",
            "model_actor_supervisor_undesired_gpu_actor",
        )
    ]


def test_issue_593_placement_reconciler_protects_vllm_child_actor_for_desired_runtime() -> None:
    killed: list[tuple[str, str, str]] = []

    reconciler = ModelActorPlacementReconciler(
        namespace="mint",
        actor_exists=lambda _name, _namespace: False,
        gpu_actor_killer=lambda actor, reason: killed.append(
            (str(actor["name"]), str(actor.get("namespace") or "mint"), reason)
        )
        or True,
        actor_lister=lambda: [],
        gpu_actor_lister=lambda: [
            {
                "name": "mint_model_runtime_vllm-Qwen-Qwen3-4B-Instruct-2507_replica-0",
                "namespace": "mint",
                "node_ip": "10.0.0.8",
                "gpu": 1,
            },
            {
                "name": "mint_vllm_qwen3-4b-instruct-2507",
                "namespace": "mint",
                "node_ip": "10.0.0.8",
                "gpu": 1,
            },
        ],
        placement_group_remover=lambda _name, _namespace: False,
    )

    out = reconciler(
        {
            ("vllm:Qwen/Qwen3-4B-Instruct-2507", "replica-0"): ModelActorSpec(
                domain_key="vllm:Qwen/Qwen3-4B-Instruct-2507",
                replica_id="replica-0",
                base_model="Qwen/Qwen3-4B-Instruct-2507",
                gpu_count=1,
            )
        }
    )

    assert out["cleaned_gpu_actor_names"] == []
    assert killed == []


@pytest.mark.anyio
async def test_issue_593_supervisor_passes_inventory_actor_names_to_placement_reconciler() -> None:
    captured: dict[str, set[str]] = {}

    def _placement(_desired, *, protected_actor_names):
        captured["protected"] = set(protected_actor_names)
        return {"ok": True, "reclaimed_total": 0}

    supervisor = ModelActorSupervisor(
        specs=[],
        scheduler_sync=lambda _registrations: None,
        placement_reconciler=_placement,
        **_disabled_control_plane_kwargs(),
    )
    supervisor.register(
        actor_name="mint_openpi_shared_registered",
        actor_type=ActorType.OPENPI,
        num_gpus=1,
    )

    await supervisor.reconcile_once()

    assert "mint_openpi_shared_registered" in captured["protected"]


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
        namespace="mint",
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


def test_issue_593_default_gpu_actor_lister_filters_namespace_and_runtime_actor_prefix(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Row:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def asdict(self) -> dict[str, object]:
            return dict(self._payload)

    class _RayState:
        @staticmethod
        def list_actors(**kwargs):
            calls.append(dict(kwargs))
            return [
                _Row(
                    {
                        "state": "ALIVE",
                        "name": "mint_model_runtime_vllm-Qwen-Test_replica-1",
                        "ray_namespace": "mint-ns",
                        "required_resources": {"GPU": 4},
                        "node_id": "node-1",
                    }
                ),
                _Row(
                    {
                        "state": "ALIVE",
                        "name": "not_mint_runtime",
                        "ray_namespace": "mint-ns",
                        "required_resources": {"GPU": 8},
                        "node_id": "node-1",
                    }
                ),
            ]

    class _RayUtil:
        state = _RayState()

    class _Ray:
        util = _RayUtil()

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def nodes() -> list[dict[str, str]]:
            return [{"NodeID": "node-1", "NodeManagerAddress": "10.0.0.17"}]

    monkeypatch.setenv("MINT_RAY_NAMESPACE", "mint-ns")
    monkeypatch.setitem(sys.modules, "ray", _Ray)
    monkeypatch.setitem(sys.modules, "ray.util", _RayUtil())
    monkeypatch.setitem(sys.modules, "ray.util.state", _RayState())

    out = list(placement_module._default_gpu_actor_lister())

    assert calls == [
        {
            "detail": True,
            "limit": 10000,
            "filters": [("ray_namespace", "=", "mint-ns")],
        }
    ]
    assert out == [
        {
            "name": "mint_model_runtime_vllm-Qwen-Test_replica-1",
            "namespace": "mint-ns",
            "node_ip": "10.0.0.17",
            "gpu": 4.0,
            "node_id": "node-1",
        },
        {
            "name": "not_mint_runtime",
            "namespace": "mint-ns",
            "node_ip": "10.0.0.17",
            "gpu": 8.0,
            "node_id": "node-1",
        },
    ]


def test_issue_593_placement_reconciler_does_not_preempt_on_non_capacity_failure() -> None:
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    reconciler = ModelActorPlacementReconciler(
        namespace="mint",
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
        namespace="mint",
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
    assert ("unknown_ns_pg", "mint") in removed_pgs
    assert out["evicted_actor_names"] == []
    assert out["evicted_placement_group_names"] == ["unknown_ns_pg"]


def test_issue_593_placement_reconciler_does_not_evict_foreign_blockers_when_target_started() -> None:
    capacity_checks: list[dict[str, object]] = []
    removed_pgs: list[tuple[str, str]] = []
    killed: list[tuple[str, str, str]] = []

    def _actor_exists(name: str, _namespace: str) -> bool:
        return name == "mint_model_runtime_vllm-Qwen-Test_replica-1"

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
        namespace="mint",
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
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    assert created == []
    replica = out["snapshot"]["replicas"]["vllm:Qwen/Test::replica-0"]
    assert replica["state"] == "blocked"
    assert replica["node_pins"] == ["10.0.0.7"]
    assert replica["last_action"] == "blocked:placement"
    assert "blocker_pg" in replica["last_error"]
    assert synced[-1][0]["status"] == "blocked"


@pytest.mark.anyio
async def test_issue_593_supervisor_precreates_controller_pg_before_runtime() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []
    pg_calls: list[dict] = []

    class _ReadyPg:
        async def ready(self):
            return self

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        assert pg_calls, "runtime must not be created before controller-created PG is ready"
        return actor

    async def _create_pg(**kwargs):
        pg_calls.append(dict(kwargs))
        return _ReadyPg()

    controller = ClusterPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 1},
        placement_group_factory=_create_pg,
    )
    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                actor_name="mint_model_runtime_vllm-qwen-test_replica-0",
                node_pin="10.0.0.7",
                gpu_count=1,
            )
        ],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
        placement_controller=controller,
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    assert len(created) == 1
    assert len(pg_calls) == 1
    assert pg_calls[0]["name"] == "mint_model_runtime_vllm-qwen-test_replica-0_pg"
    assert pg_calls[0]["bundles"] == ({"CPU": 1, "GPU": 1, "node:10.0.0.7": 0.001}, {"CPU": 1})
    replica = out["snapshot"]["replicas"]["vllm:Qwen/Test::replica-0"]
    assert replica["state"] == "healthy"
    assert replica["node_pins"] == ["10.0.0.7"]
    assert out["snapshot"]["placement_groups_created_total"] == 1
    assert synced[-1][0]["status"] == "healthy"


@pytest.mark.anyio
async def test_issue_593_supervisor_blocks_when_controller_pg_create_blocks() -> None:
    created: list[_FakeRuntimeActor] = []
    synced: list[list[dict]] = []

    class _BlockedPlacementController(ClusterPlacementController):
        async def create_pg(self, request):
            from mint_server.backend.cluster_placement_controller import (
                PlacementBlockReason,
                PlacementGroupCreateResult,
            )

            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.INSUFFICIENT_GPU,
                message="no gpu",
                retry_at=123.0,
            )

    async def _factory(spec: ModelActorSpec, generation: int):
        actor = _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )
        created.append(actor)
        return actor

    controller = _BlockedPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 0},
    )
    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                actor_name="mint_model_runtime_vllm-qwen-test_replica-0",
                node_pin="10.0.0.7",
                gpu_count=1,
            )
        ],
        runtime_factory=_factory,
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
        placement_controller=controller,
        **_disabled_control_plane_kwargs(),
    )

    out = await supervisor.reconcile_once()

    assert created == []
    replica = out["snapshot"]["replicas"]["vllm:Qwen/Test::replica-0"]
    assert replica["state"] == "blocked"
    assert replica["last_action"] == "blocked:placement_group"
    assert "no gpu" in replica["last_error"]
    assert replica["placement_retry_at"] == 123.0
    assert synced[-1][0]["status"] == "blocked"


def test_issue_593_supervisor_defaults_to_cluster_placement_controller_for_real_launchers() -> None:
    supervisor = ModelActorSupervisor(
        specs=[],
        **_disabled_control_plane_kwargs(),
    )

    assert isinstance(supervisor._placement_controller, ClusterPlacementController)
