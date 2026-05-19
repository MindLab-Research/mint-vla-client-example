from __future__ import annotations

import yaml
import pytest

from mint_server.backend.model_actor_supervisor import ModelActorSpec, ModelActorSupervisor, desired_specs_from_env
from mint_server.backend.topology import (
    ProviderTaskState,
    RayNodeState,
    TopologyManager,
    VolcanoTopologyProvider,
    load_topology_config,
    render_volcano_worker_template,
    stable_provider_task_name,
)


class _FakeRuntimeActor:
    def __init__(self, *, actor_name: str, domain_key: str, replica_id: str, generation: int) -> None:
        self.actor_name = actor_name
        self.domain_key = domain_key
        self.replica_id = replica_id
        self.generation = int(generation)

    def start(self) -> dict:
        return {"running": True}

    def health_snapshot(self) -> dict:
        return {
            "actor_name": self.actor_name,
            "domain_key": self.domain_key,
            "replica_id": self.replica_id,
            "actor_generation": self.generation,
            "running": True,
        }


def _write_topology_config(tmp_path, *, state_path=None, desired_nodes=None) -> str:
    path = tmp_path / "topology.yaml"
    payload = {
        "version": 1,
        "deployment_env": "prod",
        "cluster_id": "volcano",
        "state_path": str(state_path or tmp_path / "topology_state.yaml"),
        "providers": {
            "volcano": {
                "templates": {
                    "a800-8gpu-c1": {
                        "template_path": "mint-prod-worker.yaml",
                        "resource_queue_id": "rq-a",
                        "gpu_count": 8,
                    }
                }
            }
        },
        "nodes": {
            "desired": desired_nodes
            if desired_nodes is not None
            else [
                {
                    "alias": "mint-worker-0",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                }
            ]
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_issue_627_topology_config_parses_stable_worker_task_name(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))

    assert list(config.nodes) == ["mint-worker-0"]
    assert stable_provider_task_name(config.deployment_env, "mint-worker-0") == "mint-prod-worker-0"
    assert config.nodes["mint-worker-0"].gpu_count == 8


def test_issue_627_desired_specs_accept_worker_alias_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_MODEL_ACTOR_DESIRED_JSON", raising=False)
    monkeypatch.setenv("MINT_MODEL_ACTOR_INTERNAL_CONTROL", "0")
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/Test")
    monkeypatch.setenv(
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        '{"Qwen/Test":{"replica":0,"worker_alias":"mint-worker-0","gpu_count":4}}',
    )
    monkeypatch.setenv(
        "MINT_DENSE_MODEL_PLACEMENT_JSON",
        '{"Qwen/Test":{"replica":0,"worker_alias":"10.0.0.99","gpu_count":1}}',
    )

    specs = desired_specs_from_env()

    assert specs[0].placement_alias_slices == (("replica-0", "mint-worker-0", 4),)
    assert specs[0].normalized_worker_aliases() == ["mint-worker-0"]
    assert specs[0].normalized_node_pins() == []
    assert specs[1].worker_alias is None
    assert specs[1].normalized_node_pins() == ["10.0.0.99"]


def test_issue_627_topology_manager_writes_ready_state_without_submitting_duplicate(tmp_path) -> None:
    state_path = tmp_path / "runtime" / "topology_state.yaml"
    config = load_topology_config(_write_topology_config(tmp_path, state_path=state_path))
    submitted: list[str] = []
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [
            ProviderTaskState(
                alias="mint-worker-0",
                provider="volcano",
                task_name="mint-prod-worker-0",
                task_id="task-0",
                live=True,
                node_ip="10.0.0.7",
                gpu_count=8,
            )
        ],
        provider_task_submitter=lambda _config, node: submitted.append(node.alias),
        ray_node_lister=lambda: [
            RayNodeState(node_ip="10.0.0.7", ray_node_id="ray-0", alive=True, gpu_count=8)
        ],
    )

    state = manager.reconcile_once()

    assert state is not None
    assert state.nodes["mint-worker-0"].state == "ready"
    assert state.nodes["mint-worker-0"].provider_task_name == "mint-prod-worker-0"
    assert manager.resolve_alias("mint-worker-0") == ("10.0.0.7", None)
    assert submitted == []
    written = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    assert written["nodes"]["mint-worker-0"]["node_ip"] == "10.0.0.7"
    assert written["nodes"]["mint-worker-0"]["ray_node_id"] == "ray-0"


def test_issue_627_topology_manager_submits_missing_task_and_blocks_alias(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    submitted: list[str] = []
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        provider_task_submitter=lambda _config, node: submitted.append(node.alias),
        ray_node_lister=lambda: [],
    )

    state = manager.reconcile_once()

    assert state is not None
    assert state.nodes["mint-worker-0"].state == "provisioning"
    assert submitted == ["mint-worker-0"]
    node_ip, error = manager.resolve_alias("mint-worker-0")
    assert node_ip is None
    assert "not ready" in str(error)


def test_issue_627_volcano_provider_lists_stable_tasks_and_extracts_log_ip(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    calls: list[list[str]] = []

    def _runner(argv: list[str], _timeout_s: float) -> str:
        calls.append(argv)
        if argv[1:3] == ["ml_task", "list"]:
            return (
                "volc banner\n"
                '[{"JobId":"t-1","JobName":"mint-prod-worker-0","Status":"Running",'
                '"TaskRoleSpecs":[{"RoleReplicas":1,"ResourceSpecId":"ml.hpcpni2l.28xlarge"}]},'
                '{"JobId":"t-2","JobName":"other","Status":"Running"}]'
            )
        if argv[1:3] == ["ml_task", "logs"]:
            return "noise\nLocal node IP: 10.0.0.7\n"
        raise AssertionError(argv)

    provider = VolcanoTopologyProvider(command_runner=_runner)

    states = list(provider.list_tasks(config))

    assert len(states) == 1
    assert states[0].alias == "mint-worker-0"
    assert states[0].task_name == "mint-prod-worker-0"
    assert states[0].live is True
    assert states[0].node_ip == "10.0.0.7"
    assert states[0].gpu_count == 8
    assert any(call[1:3] == ["ml_task", "logs"] for call in calls)


def test_issue_627_volcano_provider_renders_template_and_submits(tmp_path) -> None:
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = _write_topology_config(tmp_path)
    config = load_topology_config(config_path)
    provider_cfg = dict(config.providers["volcano"])
    templates = dict(provider_cfg["templates"])
    templates["a800-8gpu-c1"] = {**templates["a800-8gpu-c1"], "template_path": str(template)}
    provider_cfg["templates"] = templates
    config = type(config)(
        version=config.version,
        deployment_env=config.deployment_env,
        cluster_id=config.cluster_id,
        state_path=config.state_path,
        nodes=config.nodes,
        providers={"volcano": provider_cfg},
    )
    submitted: list[list[str]] = []
    submitted_yaml: list[str] = []

    def _runner(argv: list[str], _timeout_s: float) -> str:
        submitted.append(argv)
        submitted_yaml.append(open(argv[argv.index("-c") + 1], encoding="utf-8").read())
        return '{"JobId":"t-new"}'

    provider = VolcanoTopologyProvider(command_runner=_runner)

    provider.submit_task(config, config.nodes["mint-worker-0"])

    assert submitted[0][1:4] == ["ml_task", "submit", "-c"]
    assert 'TaskName: "mint-prod-worker-0"' in submitted_yaml[0]
    assert 'ResourceQueueID: "rq-a"' in submitted_yaml[0]
    assert 'export MINT_WORKER_ALIAS="mint-worker-0"' in submitted_yaml[0]
    assert 'export MINT_DEPLOYMENT_ENV="prod"' in submitted_yaml[0]
    assert 'export MINT_CLUSTER_ID="volcano"' in submitted_yaml[0]


def test_issue_627_render_volcano_template_is_stable(tmp_path) -> None:
    template = tmp_path / "worker.yaml"
    template.write_text(
        'TaskName: "mint-prod-worker"\n'
        'Description: "worker"\n'
        "Entrypoint: |\n"
        "  echo start\n"
        'ResourceQueueID: "<GPU_QUEUE_ID>"\n',
        encoding="utf-8",
    )

    rendered = render_volcano_worker_template(
        template_path=template,
        task_name="mint-prod-worker-0",
        resource_queue_id="q-123",
        worker_alias="mint-worker-0",
        deployment_env="prod",
        cluster_id="volcano",
    )

    assert 'TaskName: "mint-prod-worker-0"' in rendered
    assert 'ResourceQueueID: "q-123"' in rendered
    assert "/runtime/workers/mint-worker-0.env" in rendered


@pytest.mark.anyio
async def test_issue_627_supervisor_resolves_worker_alias_before_launch_and_scheduler_sync(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [
            ProviderTaskState(
                alias="mint-worker-0",
                provider="volcano",
                task_name="mint-prod-worker-0",
                task_id="task-0",
                live=True,
                node_ip="10.0.0.7",
                gpu_count=8,
            )
        ],
        ray_node_lister=lambda: [
            RayNodeState(node_ip="10.0.0.7", ray_node_id="ray-0", alive=True, gpu_count=8)
        ],
    )
    created: list[ModelActorSpec] = []
    synced: list[list[dict]] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        created.append(spec)
        return _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                launcher_key="vllm",
                worker_alias="mint-worker-0",
                gpu_count=4,
            )
        ],
        topology_manager=manager,
        runtime_factory=_factory,
        placement_reconciler=lambda desired: {
            "ok": True,
            "blocked": {},
            "node_pins": {
                f"{spec.domain_key}::{spec.replica_id}": spec.normalized_node_pins()
                for spec in desired.values()
            },
        },
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )

    out = await supervisor.reconcile_once()

    label = "vllm:Qwen/Test::replica-0"
    assert out["snapshot"]["replicas"][label]["worker_aliases"] == ["mint-worker-0"]
    assert out["snapshot"]["replicas"][label]["node_pins"] == ["10.0.0.7"]
    assert created[0].normalized_node_pins() == ["10.0.0.7"]
    assert synced[-1][0]["node_pins"] == ["10.0.0.7"]
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-0"]["state"] == "ready"


@pytest.mark.anyio
async def test_issue_627_supervisor_blocks_until_worker_alias_ready(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [
            ProviderTaskState(
                alias="mint-worker-0",
                provider="volcano",
                task_name="mint-prod-worker-0",
                task_id="task-0",
                live=True,
                node_ip="10.0.0.7",
                gpu_count=8,
            )
        ],
        ray_node_lister=lambda: [],
    )
    created: list[ModelActorSpec] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        created.append(spec)
        return _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                launcher_key="vllm",
                worker_alias="mint-worker-0",
                gpu_count=4,
            )
        ],
        topology_manager=manager,
        runtime_factory=_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    label = "vllm:Qwen/Test::replica-0"
    assert out["snapshot"]["replicas"][label]["state"] == "blocked"
    assert "topology blocked" in out["snapshot"]["replicas"][label]["last_error"]
    assert created == []


@pytest.mark.anyio
async def test_issue_627_raw_ip_worker_alias_is_resolved_without_provider_submit(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path, desired_nodes=[]))
    submitted: list[str] = []
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        provider_task_submitter=lambda _config, node: submitted.append(node.alias),
        ray_node_lister=lambda: [],
    )
    created: list[ModelActorSpec] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        created.append(spec)
        return _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                launcher_key="vllm",
                worker_alias="10.0.0.99",
                gpu_count=4,
            )
        ],
        topology_manager=manager,
        runtime_factory=_factory,
        placement_reconciler=lambda desired: {
            "ok": True,
            "blocked": {},
            "node_pins": {
                f"{spec.domain_key}::{spec.replica_id}": spec.normalized_node_pins()
                for spec in desired.values()
            },
        },
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    label = "vllm:Qwen/Test::replica-0"
    assert out["snapshot"]["replicas"][label]["state"] == "healthy"
    assert out["snapshot"]["replicas"][label]["node_pins"] == ["10.0.0.99"]
    assert created[0].normalized_node_pins() == ["10.0.0.99"]
    assert submitted == []


@pytest.mark.anyio
async def test_issue_627_raw_ip_worker_alias_works_without_topology_config() -> None:
    created: list[ModelActorSpec] = []

    async def _factory(spec: ModelActorSpec, generation: int):
        created.append(spec)
        return _FakeRuntimeActor(
            actor_name=spec.normalized_actor_name(),
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            generation=generation,
        )

    supervisor = ModelActorSupervisor(
        specs=[
            ModelActorSpec(
                domain_key="vllm:Qwen/Test",
                replica_id="replica-0",
                base_model="Qwen/Test",
                launcher_key="vllm",
                worker_alias="10.0.0.99",
                gpu_count=4,
            )
        ],
        topology_manager=TopologyManager(None),
        runtime_factory=_factory,
        placement_reconciler=lambda desired: {
            "ok": True,
            "blocked": {},
            "node_pins": {
                f"{spec.domain_key}::{spec.replica_id}": spec.normalized_node_pins()
                for spec in desired.values()
            },
        },
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    label = "vllm:Qwen/Test::replica-0"
    assert out["snapshot"]["replicas"][label]["state"] == "healthy"
    assert out["snapshot"]["replicas"][label]["node_pins"] == ["10.0.0.99"]
    assert created[0].normalized_node_pins() == ["10.0.0.99"]
