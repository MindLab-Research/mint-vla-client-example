from __future__ import annotations

import json

import yaml
import pytest

from mint_server.backend.model_actor_supervisor import ModelActorSpec, ModelActorSupervisor, desired_specs_from_env
from mint_server.backend.node_metrics_daemon import NodeMetricsDaemonSpec
from mint_server.backend import node_metrics_daemon as node_metrics_daemon_module
from mint_server.backend.topology import (
    ProviderTaskState,
    RayNodeState,
    TopologyManager,
    VolcanoTopologyProvider,
    load_topology_config,
    ray_dashboard_node_lister,
    render_volcano_worker_template,
    stable_provider_task_name,
    worker_alias_index,
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


class _FakeNodeMetricsActor:
    def __init__(self, spec: NodeMetricsDaemonSpec) -> None:
        self.spec = spec

    def health_snapshot(self) -> dict:
        return {
            "running": True,
            "actor_name": self.spec.normalized_actor_name(),
            "worker_alias": self.spec.worker_alias,
            "node_ip": self.spec.node_ip,
            "ray_node_id": self.spec.ray_node_id,
            "deployment_env": self.spec.deployment_env,
            "cluster_id": self.spec.cluster_id,
            "sample_count": 0,
            "error_count": 0,
        }

    def sample_cached(self) -> dict:
        return {
            "worker_alias": self.spec.worker_alias,
            "node_ip": self.spec.node_ip,
            "ray_node_id": self.spec.ray_node_id,
            "deployment_env": self.spec.deployment_env,
            "cluster_id": self.spec.cluster_id,
            "hostname": "worker-host",
            "load_1m": 1.0,
            "load_5m": 2.0,
            "load_15m": 3.0,
            "cpu_utilization_ratio": 0.5,
            "memory_used_bytes": 1024,
            "memory_total_bytes": 2048,
            "disk_used_bytes": 4096,
            "disk_total_bytes": 8192,
            "gpu_count": 1,
            "gpus": [
                {
                    "gpu_uuid": "GPU-test",
                    "memory_used_bytes": 512,
                    "memory_total_bytes": 1024,
                    "utilization_gpu_percent": 77,
                    "utilization_memory_percent": 66,
                    "power_draw_watts": 300,
                    "power_limit_watts": 400,
                    "temperature_celsius": 61,
                    "sm_clock_mhz": 1200,
                    "memory_clock_mhz": 1500,
                    "pcie_link_gen": 4,
                    "pcie_link_width": 16,
                    "processes": [
                        {
                            "process_class": "vllm",
                            "process_count": 2,
                            "memory_used_bytes": 256,
                        }
                    ],
                }
            ],
            "gpu_error": None,
            "host_error": None,
            "sampled_at": 10.0,
            "sample_duration_ms": 12.5,
        }

    def shutdown(self) -> bool:
        return True


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
    assert config.ray_head_ip_path == "/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt"
    assert worker_alias_index("mint-worker-10") == 10


def test_issue_627_ray_config_accepts_dashboard_and_head_ip_path(tmp_path) -> None:
    config_path = _write_topology_config(tmp_path)
    payload = yaml.safe_load(open(config_path, encoding="utf-8").read())
    payload["ray"] = {
        "dashboard_url": "http://10.0.0.1:8265",
        "head_ip_path": "/tmp/ray_head_ip.txt",
    }
    open(config_path, "w", encoding="utf-8").write(yaml.safe_dump(payload))

    config = load_topology_config(config_path)

    assert config.ray_dashboard_url == "http://10.0.0.1:8265"
    assert config.ray_head_ip_path == "/tmp/ray_head_ip.txt"


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


def test_issue_627_dashboard_node_lister_parses_ray_nodes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    head_ip_path = tmp_path / "ray_head_ip.txt"
    head_ip_path.write_text("10.0.0.1\n", encoding="utf-8")
    calls: list[str] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": {
                        "result": {
                            "result": [
                                {
                                    "node_ip": "10.0.0.7",
                                    "node_id": "ray-0",
                                    "state": "ALIVE",
                                    "node_name": "worker-0",
                                    "resources_total": {"GPU": 8.0, "CPU": 108.0},
                                },
                                {
                                    "node_ip": "10.0.0.8",
                                    "node_id": "ray-1",
                                    "state": "DEAD",
                                    "resources_total": {"GPU": 8.0},
                                },
                            ]
                        }
                    }
                }
            ).encode("utf-8")

    def _urlopen(url, timeout):
        calls.append(f"{url} timeout={timeout}")
        return _Response()

    monkeypatch.setattr("mint_server.backend.topology.urllib.request.urlopen", _urlopen)

    nodes = list(ray_dashboard_node_lister(head_ip_path=head_ip_path, timeout_s=3))

    assert calls == ["http://10.0.0.1:8265/api/v0/nodes timeout=3.0"]
    assert nodes[0] == RayNodeState(
        node_ip="10.0.0.7",
        ray_node_id="ray-0",
        alive=True,
        gpu_count=8,
        hostname="worker-0",
    )
    assert nodes[1].alive is False


def test_issue_627_topology_manager_can_use_dashboard_nodes_without_ray_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "runtime" / "topology_state.yaml"
    head_ip_path = tmp_path / "ray_head_ip.txt"
    head_ip_path.write_text("10.0.0.1\n", encoding="utf-8")
    config_path = _write_topology_config(tmp_path, state_path=state_path)
    payload = yaml.safe_load(open(config_path, encoding="utf-8").read())
    payload["ray"] = {"head_ip_path": str(head_ip_path)}
    open(config_path, "w", encoding="utf-8").write(yaml.safe_dump(payload))
    config = load_topology_config(config_path)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": {
                        "result": {
                            "result": [
                                {
                                    "node_ip": "10.0.0.7",
                                    "node_id": "ray-0",
                                    "state": "ALIVE",
                                    "resources_total": {"GPU": 8.0},
                                }
                            ]
                        }
                    }
                }
            ).encode("utf-8")

    monkeypatch.setattr("mint_server.backend.topology.urllib.request.urlopen", lambda *_args, **_kwargs: _Response())
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
    )

    state = manager.reconcile_once()

    assert state is not None
    assert state.nodes["mint-worker-0"].state == "ready"
    assert state.nodes["mint-worker-0"].ray_node_id == "ray-0"
    assert manager.resolve_alias("mint-worker-0") == ("10.0.0.7", None)


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


def test_issue_627_topology_manager_submits_missing_workers_by_idx_order(tmp_path) -> None:
    config = load_topology_config(
        _write_topology_config(
            tmp_path,
            desired_nodes=[
                {"alias": "mint-worker-1", "provider": "volcano", "template": "a800-8gpu-c1"},
                {"alias": "mint-worker-0", "provider": "volcano", "template": "a800-8gpu-c1"},
            ],
        )
    )
    submitted: list[str] = []
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        provider_task_submitter=lambda _config, node: submitted.append(node.alias),
        ray_node_lister=lambda: [],
    )

    state = manager.reconcile_once()

    assert state is not None
    assert submitted == ["mint-worker-0"]
    assert state.nodes["mint-worker-0"].state == "provisioning"
    assert state.nodes["mint-worker-0"].last_error == "missing provider task mint-prod-worker-0"
    assert state.nodes["mint-worker-1"].state == "waiting"
    assert state.nodes["mint-worker-1"].last_error == "waiting for lower worker alias mint-worker-0"


def test_issue_627_topology_manager_submits_next_idx_after_lower_ready(tmp_path) -> None:
    config = load_topology_config(
        _write_topology_config(
            tmp_path,
            desired_nodes=[
                {"alias": "mint-worker-0", "provider": "volcano", "template": "a800-8gpu-c1"},
                {"alias": "mint-worker-1", "provider": "volcano", "template": "a800-8gpu-c1"},
            ],
        )
    )
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
    assert state.nodes["mint-worker-1"].state == "provisioning"
    assert submitted == ["mint-worker-1"]


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


def test_issue_627_volcano_provider_can_run_via_submit_host(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    calls: list[list[str]] = []

    def _runner(argv: list[str], _timeout_s: float) -> str:
        calls.append(argv)
        return (
            "volc banner\n"
            '[{"JobId":"t-1","JobName":"mint-prod-worker-0","Status":"Running",'
            '"TaskRoleSpecs":[{"RoleReplicas":1,"ResourceSpecId":"ml.hpcpni2l.28xlarge"}]}]'
        )

    provider = VolcanoTopologyProvider(command_runner=_runner, submit_host="mint-prod-volcano", fetch_logs=False)

    states = list(provider.list_tasks(config))

    assert states[0].task_name == "mint-prod-worker-0"
    assert calls[0][:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "mint-prod-volcano"]
    assert "/root/.volc/bin/volc ml_task list --output json --limit 200" in calls[0][-1]


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


def test_issue_627_volcano_provider_submit_host_writes_runtime_submit_file(tmp_path) -> None:
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
    state_path = tmp_path / "runtime" / "topology_state.yaml"
    config = load_topology_config(_write_topology_config(tmp_path, state_path=state_path))
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
        ray_dashboard_url=config.ray_dashboard_url,
        ray_head_ip_path=config.ray_head_ip_path,
    )
    calls: list[list[str]] = []

    def _runner(argv: list[str], _timeout_s: float) -> str:
        calls.append(argv)
        submit_path = argv[-1].split(" -c ", 1)[1].split(" ", 1)[0]
        assert submit_path.startswith(str(state_path.parent / "topology-submits"))
        assert open(submit_path, encoding="utf-8").read().startswith('TaskName: "mint-prod-worker-0"')
        return '{"JobId":"t-new"}'

    provider = VolcanoTopologyProvider(command_runner=_runner, submit_host="mint-prod-volcano")

    provider.submit_task(config, config.nodes["mint-worker-0"])

    assert calls[0][:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "mint-prod-volcano"]
    assert "ml_task submit -c " in calls[0][-1]


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
async def test_issue_627_supervisor_reconciles_node_metrics_daemonset_separately(tmp_path) -> None:
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
    daemon_specs: list[NodeMetricsDaemonSpec] = []
    synced: list[list[dict]] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        return _FakeNodeMetricsActor(spec)

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_enabled=True,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda registrations: synced.append(
            [registration.to_dict() for registration in registrations]
        ),
    )

    out = await supervisor.reconcile_once()

    daemon = out["snapshot"]["daemons"]["node_metrics"]
    assert daemon["enabled"] is True
    assert daemon["desired_total"] == 1
    assert daemon["managed_total"] == 1
    assert daemon["nodes"]["mint-worker-0"]["state"] == "healthy"
    assert daemon["nodes"]["mint-worker-0"]["actor_name"] == "mint_daemon_node_metrics_mint-worker-0"
    assert daemon_specs[0].worker_alias == "mint-worker-0"
    assert daemon_specs[0].node_ip == "10.0.0.7"
    assert synced[-1] == []
    assert out["snapshot"]["replicas"] == {}


@pytest.mark.anyio
async def test_issue_627_supervisor_node_metrics_daemonset_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("MINT_NODE_METRICS_DAEMON_ENABLED", raising=False)
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
    daemon_specs: list[NodeMetricsDaemonSpec] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        return _FakeNodeMetricsActor(spec)

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    assert out["snapshot"]["daemons"]["node_metrics"]["enabled"] is True
    assert daemon_specs[0].worker_alias == "mint-worker-0"


@pytest.mark.anyio
async def test_issue_627_supervisor_node_metrics_daemonset_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MINT_NODE_METRICS_DAEMON_ENABLED", "0")
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

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_factory=lambda _spec: (_ for _ in ()).throw(AssertionError("disabled")),
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    assert out["snapshot"]["daemons"]["node_metrics"]["enabled"] is False
    assert out["snapshot"]["daemons"]["node_metrics"]["managed_total"] == 0


def test_issue_627_node_metrics_daemon_registers_expected_otel_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    created: list[str] = []

    class _FakeMeter:
        def create_observable_gauge(self, name, **_kwargs):
            created.append(name)

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())

    actor = node_metrics_daemon_module.NodeMetricsCollectorActor(
        worker_alias="mint-worker-0",
        node_ip="10.0.0.7",
        deployment_env="prod",
        cluster_id="volcano",
    )
    try:
        assert actor.health_snapshot()["otel_enabled"] is True
        assert len(created) == len(set(created))
        for expected in {
            "mint_node_cpu_utilization_ratio",
            "mint_node_load_1m",
            "mint_node_load5",
            "mint_node_load15",
            "mint_node_memory_used_bytes",
            "mint_node_memory_total_bytes",
            "mint_node_disk_used_bytes",
            "mint_node_disk_total_bytes",
            "mint_node_gpu_present",
            "mint_node_gpu_utilization_percent",
            "mint_node_gpu_memory_used_bytes",
            "mint_node_gpu_memory_total_bytes",
            "mint_node_gpu_power_draw_watts",
            "mint_node_gpu_power_limit_watts",
            "mint_node_gpu_temperature_celsius",
            "mint_node_gpu_sm_clock_mhz",
            "mint_node_gpu_memory_clock_mhz",
            "mint_node_gpu_pcie_link_gen",
            "mint_node_gpu_pcie_link_width",
            "mint_node_gpu_processes",
            "mint_node_gpu_process_memory_used_bytes",
            "mint_node_metrics_collector_up",
            "mint_node_metrics_collector_sample_age_s",
            "mint_node_metrics_collector_sample_duration_ms",
            "mint_node_metrics_collector_errors_total",
        }:
            assert expected in created
    finally:
        actor.shutdown()


def test_issue_627_node_metrics_daemon_samples_in_background_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    samples = {"count": 0}
    flushed: list[str] = []

    class _Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def force_flush(self, **_kwargs) -> None:
            flushed.append(self.name)

    monkeypatch.setenv("MINT_NODE_METRICS_SAMPLE_INTERVAL_S", "0.1")
    monkeypatch.setattr(
        node_metrics_daemon_module,
        "_sample_host_metrics",
        lambda: {
            "hostname": "worker-host",
            "load_1m": 1.0,
            "load_5m": 2.0,
            "load_15m": 3.0,
        },
    )

    def _sample_gpu_metrics():
        samples["count"] += 1
        if samples["count"] == 1:
            raise RuntimeError("nvml transient")
        return [], None

    monkeypatch.setattr(node_metrics_daemon_module, "_sample_gpu_metrics", _sample_gpu_metrics)
    monkeypatch.setattr("opentelemetry.metrics.get_meter_provider", lambda: _Provider("metrics"))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: _Provider("trace"))

    actor = node_metrics_daemon_module.NodeMetricsCollectorActor(
        worker_alias="mint-worker-0",
        node_ip="10.0.0.7",
        deployment_env="prod",
        cluster_id="volcano",
    )
    try:
        deadline = time.time() + 1.0
        while samples["count"] < 2 and time.time() < deadline:
            time.sleep(0.02)

        snapshot = actor.health_snapshot()
        assert snapshot["sample_count"] >= 1
        assert snapshot["error_count"] >= 1
        assert snapshot["last_sample"] is not None
    finally:
        assert actor.shutdown() is True

    assert "metrics" in flushed
    assert "trace" in flushed
    assert actor.health_snapshot()["running"] is False


def test_issue_627_node_metrics_disk_path_defaults_to_mint_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_NODE_METRICS_DISK_PATH", raising=False)
    monkeypatch.setenv("MINT_DEPLOYMENT_ENV", "prod")
    assert node_metrics_daemon_module._default_disk_metrics_path() == "/vePFS-Mindverse/share/mint/prod"

    monkeypatch.delenv("MINT_DEPLOYMENT_ENV", raising=False)
    assert node_metrics_daemon_module._default_disk_metrics_path() == "/vePFS-Mindverse/share/mint"


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
