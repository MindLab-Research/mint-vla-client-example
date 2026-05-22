from __future__ import annotations

import json
import sys
import types

import yaml
import pytest

from mint_server.backend.model_actor_supervisor import ModelActorSpec, ModelActorSupervisor, desired_specs_from_env
from mint_server.backend.node_metrics_daemon import NodeMetricsDaemonSpec
from mint_server.backend import node_metrics_daemon as node_metrics_daemon_module
from mint_server.backend.topology import (
    ProviderTaskState,
    RayNodeState,
    TopologyManager,
    UnsupportedTopologyProviderError,
    VolcanoTopologyProvider,
    build_volcano_create_job_request,
    load_topology_config,
    ray_dashboard_node_lister,
    render_volcano_worker_template,
    stable_provider_task_name,
    worker_alias_index,
)


class _SdkModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_fake_volcano_sdk(monkeypatch: pytest.MonkeyPatch):
    sdk = types.SimpleNamespace(
        ListJobsRequest=type("ListJobsRequest", (_SdkModel,), {}),
        ListJobInstancesRequest=type("ListJobInstancesRequest", (_SdkModel,), {}),
        CreateJobRequest=type("CreateJobRequest", (_SdkModel,), {}),
        ResourceConfigForCreateJobInput=type("ResourceConfigForCreateJobInput", (_SdkModel,), {}),
        RuntimeConfigForCreateJobInput=type("RuntimeConfigForCreateJobInput", (_SdkModel,), {}),
        ImageForCreateJobInput=type("ImageForCreateJobInput", (_SdkModel,), {}),
        RoleForCreateJobInput=type("RoleForCreateJobInput", (_SdkModel,), {}),
        ResourceForCreateJobInput=type("ResourceForCreateJobInput", (_SdkModel,), {}),
        StorageConfigForCreateJobInput=type("StorageConfigForCreateJobInput", (_SdkModel,), {}),
        ConvertCredentialForCreateJobInput=type("ConvertCredentialForCreateJobInput", (_SdkModel,), {}),
        StorageForCreateJobInput=type("StorageForCreateJobInput", (_SdkModel,), {}),
        ConfigForCreateJobInput=type("ConfigForCreateJobInput", (_SdkModel,), {}),
        VepfsForCreateJobInput=type("VepfsForCreateJobInput", (_SdkModel,), {}),
        TosForCreateJobInput=type("TosForCreateJobInput", (_SdkModel,), {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "volcenginesdkmlplatform20240701", sdk)
    return sdk


class _FakeVolcanoClient:
    def __init__(self, *, jobs=None, instances=None) -> None:
        self.jobs = jobs or []
        self.instances = instances or {}
        self.list_jobs_requests = []
        self.list_job_instances_requests = []
        self.created_jobs = []

    def list_jobs(self, request):
        self.list_jobs_requests.append(request)
        return _SdkModel(items=self.jobs)

    def list_job_instances(self, request):
        self.list_job_instances_requests.append(request)
        return _SdkModel(items=self.instances.get(request.job_id, []))

    def create_job(self, request):
        self.created_jobs.append(request)
        return _SdkModel(id="created-job")


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
        self.shutdown_requested = False

    def health_snapshot(self) -> dict:
        return {
            "running": True,
            "actor_name": self.spec.normalized_actor_name(),
            "worker_alias": self.spec.worker_alias,
            "node_ip": self.spec.node_ip,
            "ray_node_id": self.spec.ray_node_id,
            "deployment_env": self.spec.deployment_env,
            "cluster_id": self.spec.cluster_id,
            "is_head_node": self.spec.is_head_node,
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
            "is_head_node": self.spec.is_head_node,
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
        self.shutdown_requested = True
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


def test_issue_627_non_volcano_provider_fails_loudly(tmp_path) -> None:
    from mint_server.backend.topology import (
        default_provider_task_lister_for_config,
        default_provider_task_submitter_for_config,
    )

    config_path = _write_topology_config(
        tmp_path,
        desired_nodes=[
            {
                "alias": "mint-worker-0",
                "provider": "aliyun",
                "template": "a800-8gpu-c1",
                "enabled": True,
            }
        ],
    )
    payload = yaml.safe_load(open(config_path, encoding="utf-8").read())
    payload["providers"]["aliyun"] = {"templates": {"a800-8gpu-c1": {"gpu_count": 8}}}
    open(config_path, "w", encoding="utf-8").write(yaml.safe_dump(payload))
    config = load_topology_config(config_path)

    with pytest.raises(UnsupportedTopologyProviderError, match="unsupported topology providers: aliyun"):
        default_provider_task_lister_for_config(config)
    with pytest.raises(UnsupportedTopologyProviderError, match="unsupported topology providers: aliyun"):
        default_provider_task_submitter_for_config(config)


def test_issue_627_default_volcano_provider_uses_sdk_region_not_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.backend import topology as topology_module
    from mint_server.backend.topology import default_provider_task_lister_for_config

    _install_fake_volcano_sdk(monkeypatch)
    config = load_topology_config(_write_topology_config(tmp_path))
    provider_cfg = dict(config.providers["volcano"])
    provider_cfg["region"] = "cn-beijing"
    provider_cfg["connect_timeout_s"] = 3
    provider_cfg["read_timeout_s"] = 5
    provider_cfg["submit_host"] = "must-not-be-used"
    provider_cfg["volc_bin"] = "/must/not/be/used"
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
    client = _FakeVolcanoClient(
        jobs=[
            _SdkModel(
                id="t-1",
                name="mint-prod-worker-0",
                status=_SdkModel(state="Running"),
            )
        ],
        instances={"t-1": [_SdkModel(ips=_SdkModel(primary_ip="10.0.0.7"))]},
    )
    seen_calls: list[tuple[str | None, float | None, float | None]] = []

    def _client_factory(*, region=None, connect_timeout=None, read_timeout=None):
        seen_calls.append((region, connect_timeout, read_timeout))
        return client

    monkeypatch.setattr(topology_module, "_create_volcano_mlplatform_client", _client_factory)

    lister = default_provider_task_lister_for_config(config)
    states = list(lister(config))

    assert seen_calls == [("cn-beijing", 3.0, 5.0)]
    assert states[0].task_name == "mint-prod-worker-0"
    assert states[0].node_ip == "10.0.0.7"
    assert client.list_jobs_requests[0].name_contains == "mint-prod-worker-"


def test_issue_627_sdk_client_can_bridge_legacy_volc_cli_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.backend import topology as topology_module

    class _FakeConfiguration:
        def __init__(self) -> None:
            self.ak = ""
            self.sk = ""
            self.session_token = ""
            self.region = ""
            self.connect_timeout = None
            self.read_timeout = None
            self.debug = True

    class _FakeApiClient:
        def __init__(self, configuration) -> None:
            self.configuration = configuration

    class _FakeApi:
        def __init__(self, api_client) -> None:
            self.api_client = api_client

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VOLCENGINE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_SECRET_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("VOLCENGINE_CLI_CONFIG_FILE", raising=False)
    volc_dir = tmp_path / ".volc"
    volc_dir.mkdir()
    (volc_dir / "config").write_text("[default]\nregion = cn-beijing\n", encoding="utf-8")
    (volc_dir / "credentials").write_text(
        "[default]\naccess_key_id = ak-test\nsecret_access_key = sk-test\n",
        encoding="utf-8",
    )
    core = types.SimpleNamespace(Configuration=_FakeConfiguration, ApiClient=_FakeApiClient)
    sdk = types.SimpleNamespace(MLPLATFORM20240701Api=_FakeApi)
    monkeypatch.setitem(__import__("sys").modules, "volcenginesdkcore", core)
    monkeypatch.setitem(__import__("sys").modules, "volcenginesdkmlplatform20240701", sdk)

    client = topology_module._create_volcano_mlplatform_client(
        region=None,
        connect_timeout=3,
        read_timeout=5,
    )

    configuration = client.api_client.configuration
    assert configuration.ak == "ak-test"
    assert configuration.sk == "sk-test"
    assert configuration.region == "cn-beijing"
    assert configuration.connect_timeout == 3
    assert configuration.read_timeout == 5


def test_issue_627_volcano_sdk_import_falls_back_to_host_venv_site_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.backend import topology as topology_module

    sdk = _install_fake_volcano_sdk(monkeypatch)
    monkeypatch.delitem(sys.modules, "volcenginesdkmlplatform20240701")
    runtime_root = tmp_path / "runtime"
    host_site = tmp_path / "runtime" / "host-venv" / "lib" / "python3.12" / "site-packages"
    host_site.mkdir(parents=True)
    (runtime_root / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_env": {
                    "site_packages_dir": "site-packages",
                    "source_dir": "src",
                    "base_python_dir": "base-python",
                    "host_venv_dir": "host-venv",
                },
                "sources": [{"name": "dummy"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PFS_RUNTIME_ENV_ROOT", str(runtime_root))

    real_import = topology_module.importlib.import_module

    def _import(name: str):
        if name == "volcenginesdkcore":
            return types.SimpleNamespace()
        if name == "volcenginesdkmlplatform20240701":
            if str(host_site) not in sys.path:
                raise ImportError(name)
            return sdk
        return real_import(name)

    monkeypatch.setattr(topology_module.importlib, "import_module", _import)

    assert topology_module._volcano_sdk_module() is sdk
    assert str(host_site) in sys.path


def test_issue_627_sdk_client_prefers_modern_credential_chain_over_legacy_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.backend import topology as topology_module

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOLCENGINE_ACCESS_KEY", "modern-ak")
    volc_dir = tmp_path / ".volc"
    volc_dir.mkdir()
    (volc_dir / "credentials").write_text(
        "[default]\naccess_key_id = legacy-ak\nsecret_access_key = legacy-sk\n",
        encoding="utf-8",
    )

    assert topology_module._legacy_volc_cli_credentials() == {}


def test_issue_627_legacy_volc_cli_credentials_use_explicit_cli_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.backend import topology as topology_module

    monkeypatch.setenv("HOME", str(tmp_path / "not-root"))
    monkeypatch.setenv("VOLC_CLI_HOME", str(tmp_path / "cli-home"))
    monkeypatch.delenv("VOLCENGINE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_SECRET_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("VOLCENGINE_CLI_CONFIG_FILE", raising=False)
    volc_dir = tmp_path / "cli-home"
    volc_dir.mkdir()
    (volc_dir / "config").write_text("[default]\nregion = cn-beijing\n", encoding="utf-8")
    (volc_dir / "credentials").write_text(
        "[default]\naccess_key_id = explicit-ak\nsecret_access_key = explicit-sk\n",
        encoding="utf-8",
    )

    assert topology_module._legacy_volc_cli_credentials() == {
        "ak": "explicit-ak",
        "sk": "explicit-sk",
        "region": "cn-beijing",
    }


def test_issue_627_desired_specs_accept_worker_alias_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MINT_MODEL_ACTOR_INTERNAL_RUNTIME", "0")
    topology_path = _write_topology_config(tmp_path)
    monkeypatch.setenv("MINT_TOPOLOGY_CONFIG_PATH", topology_path)
    payload = yaml.safe_load(open(topology_path, encoding="utf-8"))
    payload["models"] = {
        "Qwen/Test": {
            "vllm": {"placement": [{"replica": 0, "worker_alias": "mint-worker-0", "gpu_count": 4}]},
            "training": {"placement": [{"replica": 0, "worker_alias": "10.0.0.99", "gpu_count": 1}]},
        }
    }
    open(topology_path, "w", encoding="utf-8").write(yaml.safe_dump(payload))

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


def test_issue_627_topology_manager_submits_all_missing_workers_by_idx_order(tmp_path) -> None:
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
    assert set(submitted) == {"mint-worker-0", "mint-worker-1"}
    assert state.nodes["mint-worker-0"].state == "provisioning"
    assert state.nodes["mint-worker-0"].last_error == "missing provider task mint-prod-worker-0"
    assert state.nodes["mint-worker-1"].state == "provisioning"
    assert state.nodes["mint-worker-1"].last_error == "missing provider task mint-prod-worker-1"


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


def test_issue_627_topology_manager_records_parallel_submit_errors(tmp_path) -> None:
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

    def _submit(_config, node) -> None:
        submitted.append(node.alias)
        if node.alias == "mint-worker-1":
            raise RuntimeError("quota exceeded")

    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        provider_task_submitter=_submit,
        ray_node_lister=lambda: [],
    )

    state = manager.reconcile_once()

    assert state is not None
    assert set(submitted) == {"mint-worker-0", "mint-worker-1"}
    assert state.nodes["mint-worker-0"].state == "provisioning"
    assert state.nodes["mint-worker-1"].state == "failed"
    assert "provider task submit failed for mint-worker-1: RuntimeError: quota exceeded" == state.nodes[
        "mint-worker-1"
    ].last_error


def test_issue_627_volcano_provider_lists_stable_tasks_and_extracts_instance_ip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    config = load_topology_config(_write_topology_config(tmp_path))
    client = _FakeVolcanoClient(
        jobs=[
            _SdkModel(
                id="t-1",
                name="mint-prod-worker-0",
                status=_SdkModel(state="Running"),
                resource_config=_SdkModel(
                    roles=[
                        _SdkModel(
                            replicas=1,
                            resource=_SdkModel(instance_type_id="ml.hpcpni2l.28xlarge"),
                        )
                    ]
                ),
            ),
            _SdkModel(id="t-2", name="other", status=_SdkModel(state="Running")),
        ],
        instances={
            "t-1": [
                _SdkModel(ips=_SdkModel(primary_ip="10.0.0.7", host_ip="172.16.0.7")),
            ]
        },
    )

    provider = VolcanoTopologyProvider(client=client)

    states = list(provider.list_tasks(config))

    assert len(states) == 1
    assert states[0].alias == "mint-worker-0"
    assert states[0].task_name == "mint-prod-worker-0"
    assert states[0].live is True
    assert states[0].node_ip == "10.0.0.7"
    assert states[0].gpu_count == 8
    assert client.list_jobs_requests[0].name_contains == "mint-prod-worker-"
    assert client.list_job_instances_requests[0].job_id == "t-1"


def test_issue_627_volcano_provider_treats_deploying_as_live_and_dedupes_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    config = load_topology_config(_write_topology_config(tmp_path))
    client = _FakeVolcanoClient(
        jobs=[
            _SdkModel(id="t-old", name="mint-prod-worker-0", status=_SdkModel(state="Stopped")),
            _SdkModel(
                id="t-live-1",
                name="mint-prod-worker-0",
                status=_SdkModel(state="Deploying"),
                resource_config=_SdkModel(
                    roles=[
                        _SdkModel(
                            replicas=1,
                            resource=_SdkModel(instance_type_id="ml.hpcpni2l.28xlarge"),
                        )
                    ]
                ),
            ),
            _SdkModel(
                id="t-live-2",
                name="mint-prod-worker-0",
                status=_SdkModel(state="Queueing"),
                resource_config=_SdkModel(
                    roles=[
                        _SdkModel(
                            replicas=1,
                            resource=_SdkModel(instance_type_id="ml.hpcpni2l.28xlarge"),
                        )
                    ]
                ),
            ),
        ]
    )

    states = list(VolcanoTopologyProvider(client=client).list_tasks(config))

    assert len(states) == 1
    assert states[0].alias == "mint-worker-0"
    assert states[0].live is True
    assert states[0].task_id == "t-live-2"
    assert states[0].raw_state == "Queueing"


def test_issue_627_volcano_provider_uses_sdk_default_credentials_not_submit_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    config = load_topology_config(_write_topology_config(tmp_path))
    client = _FakeVolcanoClient(
        jobs=[
            _SdkModel(
                id="t-1",
                name="mint-prod-worker-0",
                status=_SdkModel(state="Running"),
                resource_config=_SdkModel(
                    roles=[
                        _SdkModel(
                            replicas=1,
                            resource=_SdkModel(instance_type_id="ml.hpcpni2l.28xlarge"),
                        )
                    ]
                ),
            )
        ]
    )

    provider = VolcanoTopologyProvider(client=client)

    states = list(provider.list_tasks(config))

    assert states[0].task_name == "mint-prod-worker-0"
    assert client.list_jobs_requests[0].page_size == 100


def test_issue_627_volcano_sdk_jobs_list_clamps_page_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.tools import volcano_sdk_jobs

    _install_fake_volcano_sdk(monkeypatch)
    client = _FakeVolcanoClient()
    monkeypatch.setattr(volcano_sdk_jobs, "_create_volcano_mlplatform_client", lambda **_kwargs: client)

    volcano_sdk_jobs._cmd_list(
        types.SimpleNamespace(
            region="cn-beijing",
            connect_timeout=3,
            read_timeout=5,
            name_contains="mint-dev-worker-",
            limit=200,
            state=None,
        )
    )

    assert client.list_jobs_requests[0].page_size == 100


def test_issue_627_volcano_provider_renders_template_and_submits_sdk_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ImageUrl: "image"',
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
                "TaskRoleSpecs:",
                '  - RoleName: "worker"',
                "    RoleReplicas: 1",
                '    Flavor: "ml.hpcpni2l.28xlarge"',
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
    client = _FakeVolcanoClient()

    provider = VolcanoTopologyProvider(client=client)

    provider.submit_task(config, config.nodes["mint-worker-0"])

    request = client.created_jobs[0]
    assert request.name == "mint-prod-worker-0"
    assert request.resource_config.resource_queue_id == "rq-a"
    assert request.runtime_config.framework == "Custom"
    assert request.runtime_config.image.type == "Public"
    assert request.runtime_config.image.url == "image"
    assert request.resource_config.roles[0].name == "worker"
    assert request.resource_config.roles[0].replicas == 1
    assert request.resource_config.roles[0].resource.instance_type_id == "ml.hpcpni2l.28xlarge"
    assert 'export MINT_WORKER_ALIAS="mint-worker-0"' in request.runtime_config.command
    assert 'export MINT_DEPLOYMENT_ENV="prod"' in request.runtime_config.command
    assert 'export MINT_CLUSTER_ID="volcano"' in request.runtime_config.command


def test_issue_627_build_volcano_create_job_request_converts_storages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ImageUrl: "image"',
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
                "TaskRoleSpecs:",
                '  - RoleName: "worker"',
                "    RoleReplicas: 1",
                '    Flavor: "ml.hpcpni2l.28xlarge"',
                "Storages:",
                '  - Type: "Vepfs"',
                '    Id: "vepfs-test"',
                '    MountPath: "/vePFS-Mindverse/share"',
                '    SubPath: "share"',
                "    ReadOnly: false",
                '  - Type: "TosFuse"',
                '    MountPath: "/tos-mindverse"',
                '    Bucket: "tos-mindverse"',
                '    Prefix: "/"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_topology_config(_write_topology_config(tmp_path))
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

    request = build_volcano_create_job_request(config, config.nodes["mint-worker-0"])

    assert len(request.storage_config.storages) == 2
    assert request.storage_config.storages[0].type == "Vepfs"
    assert request.storage_config.storages[0].config.vepfs.id == "vepfs-test"
    assert request.storage_config.storages[0].config.vepfs.sub_path == "share"
    assert request.storage_config.storages[1].type == "Tos"
    assert request.storage_config.storages[1].config.tos.bucket == "tos-mindverse"
    assert request.storage_config.credential.use_service_linked_role is True


def test_issue_627_build_volcano_create_job_request_requires_vepfs_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ImageUrl: "image"',
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
                "TaskRoleSpecs:",
                '  - RoleName: "worker"',
                "    RoleReplicas: 1",
                '    Flavor: "ml.hpcpni2l.28xlarge"',
                "Storages:",
                '  - Type: "Vepfs"',
                '    MountPath: "/vePFS-Mindverse/share"',
                '    SubPath: "share"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_topology_config(_write_topology_config(tmp_path))
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

    with pytest.raises(ValueError, match="Vepfs storage must include Id or FileSystemName"):
        build_volcano_create_job_request(config, config.nodes["mint-worker-0"])


def test_issue_627_build_volcano_create_job_request_maps_volcengine_cr_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ImageUrl: "image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:16-sm80"',
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
                "TaskRoleSpecs:",
                '  - RoleName: "worker"',
                "    RoleReplicas: 1",
                '    Flavor: "ml.hpcpni2l.28xlarge"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_topology_config(_write_topology_config(tmp_path))
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

    request = build_volcano_create_job_request(config, config.nodes["mint-worker-0"])

    assert request.runtime_config.image.type == "VolcEngine"


def test_issue_627_build_volcano_create_job_request_allows_explicit_image_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_volcano_sdk(monkeypatch)
    template = tmp_path / "worker.yaml"
    template.write_text(
        "\n".join(
            [
                'TaskName: "mint-prod-worker"',
                'Description: "worker"',
                "Entrypoint: |",
                "  echo start",
                'ImageType: "Prebuild"',
                'ImageUrl: "prebuild-image-name"',
                'ResourceQueueID: "<GPU_QUEUE_ID>"',
                "TaskRoleSpecs:",
                '  - RoleName: "worker"',
                "    RoleReplicas: 1",
                '    Flavor: "ml.hpcpni2l.28xlarge"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_topology_config(_write_topology_config(tmp_path))
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

    request = build_volcano_create_job_request(config, config.nodes["mint-worker-0"])

    assert request.runtime_config.image.type == "Prebuild"


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
    assert daemon_specs[0].is_head_node is False
    assert synced[-1] == []
    assert out["snapshot"]["replicas"] == {}


@pytest.mark.anyio
async def test_issue_638_supervisor_marks_head_node_metrics_daemon_spec(tmp_path) -> None:
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
            RayNodeState(
                node_ip="10.0.0.7",
                ray_node_id="ray-0",
                alive=True,
                gpu_count=8,
                is_head_node=True,
            )
        ],
    )
    daemon_specs: list[NodeMetricsDaemonSpec] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        return _FakeNodeMetricsActor(spec)

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_enabled=True,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    assert daemon_specs[0].is_head_node is True
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-0"]["is_head_node"] is True
    assert out["snapshot"]["daemons"]["node_metrics"]["nodes"]["mint-worker-0"]["state"] == "healthy"


@pytest.mark.anyio
async def test_issue_638_supervisor_adds_observed_ray_head_node_metrics_daemon(
    tmp_path,
) -> None:
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
            RayNodeState(
                node_ip="10.0.0.1",
                ray_node_id="ray-head",
                alive=True,
                gpu_count=0,
                is_head_node=True,
            ),
            RayNodeState(node_ip="10.0.0.7", ray_node_id="ray-0", alive=True, gpu_count=8),
        ],
    )
    daemon_specs: list[NodeMetricsDaemonSpec] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        return _FakeNodeMetricsActor(spec)

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_enabled=True,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    specs_by_alias = {spec.worker_alias: spec for spec in daemon_specs}
    assert set(specs_by_alias) == {"mint-head", "mint-worker-0"}
    assert specs_by_alias["mint-head"].node_ip == "10.0.0.1"
    assert specs_by_alias["mint-head"].is_head_node is True
    assert out["snapshot"]["topology"]["nodes"]["mint-head"]["role"] == "head"
    assert out["snapshot"]["topology"]["nodes"]["mint-head"]["is_head_node"] is True


@pytest.mark.anyio
async def test_issue_638_mint_head_alias_is_not_valid_model_placement(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path, desired_nodes=[]))
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        ray_node_lister=lambda: [
            RayNodeState(
                node_ip="10.0.0.1",
                ray_node_id="ray-head",
                alive=True,
                gpu_count=0,
                is_head_node=True,
            )
        ],
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
                worker_alias="mint-head",
                gpu_count=1,
            )
        ],
        topology_manager=manager,
        runtime_factory=_factory,
        node_metrics_enabled=False,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    label = "vllm:Qwen/Test::replica-0"
    assert out["snapshot"]["topology"]["nodes"]["mint-head"]["role"] == "head"
    assert out["snapshot"]["replicas"][label]["state"] == "blocked"
    assert "not valid for model placement" in out["snapshot"]["replicas"][label]["last_error"]
    assert created == []


@pytest.mark.anyio
async def test_issue_638_node_metrics_daemon_recreated_when_spec_changes(
    tmp_path,
) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    current_ray_nodes = [
        RayNodeState(node_ip="10.0.0.7", ray_node_id="ray-0", alive=True, gpu_count=8)
    ]
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [
            ProviderTaskState(
                alias="mint-worker-0",
                provider="volcano",
                task_name="mint-prod-worker-0",
                task_id="task-0",
                live=True,
                node_ip=current_ray_nodes[0].node_ip,
                gpu_count=8,
            )
        ],
        ray_node_lister=lambda: list(current_ray_nodes),
    )
    daemon_specs: list[NodeMetricsDaemonSpec] = []
    actors: list[_FakeNodeMetricsActor] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        actor = _FakeNodeMetricsActor(spec)
        actors.append(actor)
        return actor

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_enabled=True,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    await supervisor.reconcile_once()
    current_ray_nodes[:] = [
        RayNodeState(
            node_ip="10.0.0.7",
            ray_node_id="ray-0",
            alive=True,
            gpu_count=8,
            is_head_node=True,
        )
    ]
    out = await supervisor.reconcile_once()

    assert [spec.is_head_node for spec in daemon_specs] == [False, True]
    assert len(actors) == 2
    assert actors[0].shutdown_requested is True
    assert out["snapshot"]["daemons"]["node_metrics"]["nodes"]["mint-worker-0"]["health"]["is_head_node"] is True


@pytest.mark.anyio
async def test_issue_627_supervisor_node_metrics_daemonset_filters_ineligible_nodes(tmp_path) -> None:
    config = load_topology_config(
        _write_topology_config(
            tmp_path,
            desired_nodes=[
                {
                    "alias": "mint-worker-0",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                    "role": "gpu",
                    "gpu_count": 8,
                },
                {
                    "alias": "mint-worker-1",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                    "role": "cpu",
                    "gpu_count": 8,
                },
                {
                    "alias": "mint-worker-2",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                    "role": "gpu",
                    "gpu_count": 0,
                },
                {
                    "alias": "mint-worker-3",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                    "role": "gpu",
                    "gpu_count": 8,
                    "mount_ok": False,
                },
                {
                    "alias": "mint-worker-4",
                    "provider": "volcano",
                    "template": "a800-8gpu-c1",
                    "enabled": True,
                    "role": "gpu",
                    "gpu_count": 8,
                    "runtime_env_ok": False,
                },
            ],
        )
    )
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [
            ProviderTaskState(
                alias=f"mint-worker-{idx}",
                provider="volcano",
                task_name=f"mint-prod-worker-{idx}",
                task_id=f"task-{idx}",
                live=True,
                node_ip=f"10.0.0.{idx + 7}",
                gpu_count=8 if idx != 2 else 0,
            )
            for idx in range(5)
        ],
        ray_node_lister=lambda: [
            RayNodeState(
                node_ip=f"10.0.0.{idx + 7}",
                ray_node_id=f"ray-{idx}",
                alive=True,
                gpu_count=8 if idx != 2 else 0,
            )
            for idx in range(5)
        ],
    )
    daemon_specs: list[NodeMetricsDaemonSpec] = []

    async def _node_metrics_factory(spec: NodeMetricsDaemonSpec):
        daemon_specs.append(spec)
        return _FakeNodeMetricsActor(spec)

    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        node_metrics_enabled=True,
        node_metrics_factory=_node_metrics_factory,
        placement_reconciler=lambda _desired: {"ok": True, "blocked": {}},
        scheduler_sync=lambda _registrations: None,
    )

    out = await supervisor.reconcile_once()

    daemon = out["snapshot"]["daemons"]["node_metrics"]
    assert [spec.worker_alias for spec in daemon_specs] == ["mint-worker-0"]
    assert daemon["desired_total"] == 1
    assert set(daemon["nodes"]) == {"mint-worker-0"}
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-1"]["role"] == "cpu"
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-3"]["mount_ok"] is False
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-4"]["runtime_env_ok"] is False


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


def test_issue_638_node_metrics_collector_up_reflects_otel_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    callbacks: dict[str, object] = {}

    class _FailingMeter:
        def create_observable_gauge(self, name, **kwargs):
            callbacks[name] = kwargs["callbacks"][0]
            if name == "mint_node_metrics_collector_sample_age_s":
                raise RuntimeError("otel gauge registration failed")

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FailingMeter())

    actor = node_metrics_daemon_module.NodeMetricsCollectorActor(
        worker_alias="mint-worker-0",
        node_ip="10.0.0.7",
        deployment_env="prod",
        cluster_id="volcano",
    )
    try:
        snapshot = actor.health_snapshot()
        assert snapshot["otel_enabled"] is False
        assert "otel gauge registration failed" in str(snapshot["otel_error"])
        actor.sample_once()
        obs = callbacks["mint_node_metrics_collector_up"](None)
        assert obs[0].value == 0.0
    finally:
        actor.shutdown()


def test_issue_638_node_metrics_daemon_registers_head_ray_global_otel_gauges(
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
        worker_alias="mint-head",
        node_ip="10.0.0.1",
        deployment_env="prod",
        cluster_id="volcano",
        is_head_node=True,
    )
    try:
        assert actor.health_snapshot()["is_head_node"] is True
        for expected in {
            "mint_ray_cluster_up",
            "mint_ray_cluster_nodes",
            "mint_ray_cluster_placement_groups_pending_gpu",
            "mint_ray_cluster_probe_latency_ms",
            "mint_ray_gcs_metrics_bridge_up",
            "mint_ray_gcs_metrics_bridge_sample_count",
            "mint_ray_gcs_raw_gcs_actors_count",
            "mint_ray_gcs_gcs_task_manager_task_events_drop_ratio",
        }:
            assert expected in created
    finally:
        actor.shutdown()


def test_issue_638_node_metrics_daemon_skips_ray_global_gauges_on_worker(
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
        is_head_node=False,
    )
    try:
        assert actor.health_snapshot()["is_head_node"] is False
        assert "mint_ray_cluster_up" not in created
        assert "mint_ray_gcs_metrics_bridge_up" not in created
    finally:
        actor.shutdown()


def test_issue_638_ray_global_otel_callbacks_use_bounded_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    callbacks: dict[str, object] = {}

    class _FakeMeter:
        def create_observable_gauge(self, name, **kwargs):
            callbacks[name] = kwargs["callbacks"][0]

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("MINT_RAY_NAMESPACE", "mint")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(
        node_metrics_daemon_module.NodeMetricsCollectorActor,
        "_start_sampling_loop",
        lambda self: None,
    )
    monkeypatch.setattr(
        node_metrics_daemon_module,
        "_ray_cluster_snapshot",
        lambda: {
            "up": True,
            "nodes": {
                "alive": 2,
                "dead": 1,
                "dead_missing_heartbeats": 1,
                "dead_missing_heartbeat_ips": ["10.0.0.9"],
            },
            "placement_groups": {
                "pending_gpu": 1,
                "pending_gpu_names": ["secret-high-cardinality-pg"],
            },
            "probes": {
                "nodes": {"ok": True, "latency_ms": 1.5, "error": "do-not-label"},
                "placement_groups": {"ok": False, "latency_ms": 2.5, "error": "do-not-label"},
            },
        },
    )
    monkeypatch.setattr(
        node_metrics_daemon_module,
        "_ray_gcs_snapshot",
        lambda: {
            "up": True,
            "sample_count": 3,
            "aggregates": {"gcs_actors_count": 7.0},
            "derived": {"gcs_task_manager_task_events_drop_ratio": 0.25},
            "scrape_errors": [{"address": "10.0.0.1:8080", "error": "do-not-label"}],
        },
    )

    actor = node_metrics_daemon_module.NodeMetricsCollectorActor(
        worker_alias="mint-head",
        node_ip="10.0.0.1",
        deployment_env="prod",
        cluster_id="volcano",
        is_head_node=True,
    )
    try:
        actor.sample_once(collect_ray_global=True)
        monkeypatch.setattr(
            node_metrics_daemon_module,
            "_ray_cluster_snapshot",
            lambda: (_ for _ in ()).throw(AssertionError("OTel callback must not probe Ray")),
        )
        monkeypatch.setattr(
            node_metrics_daemon_module,
            "_ray_gcs_snapshot",
            lambda: (_ for _ in ()).throw(AssertionError("OTel callback must not scrape GCS")),
        )
        observations = []
        for name in (
            "mint_ray_cluster_nodes",
            "mint_ray_cluster_dead_nodes_missing_heartbeats",
            "mint_ray_cluster_placement_groups_pending_gpu",
            "mint_ray_cluster_probe_success",
            "mint_ray_cluster_probe_latency_ms",
            "mint_ray_gcs_metrics_bridge_sample_count",
            "mint_ray_gcs_raw_gcs_actors_count",
            "mint_ray_gcs_gcs_task_manager_task_events_drop_ratio",
        ):
            observations.extend(callbacks[name](None))
        observed_by_name = {
            name: callbacks[name](None)
            for name in (
                "mint_ray_gcs_raw_gcs_actors_count",
                "mint_ray_gcs_gcs_task_manager_task_events_drop_ratio",
            )
        }
        assert observed_by_name["mint_ray_gcs_raw_gcs_actors_count"][0].value == 7.0
        assert observed_by_name["mint_ray_gcs_gcs_task_manager_task_events_drop_ratio"][0].value == 0.25
        for obs in observations:
            attrs = dict(obs.attributes)
            assert set(attrs) <= {
                "worker_alias",
                "deployment.env",
                "mint.cluster_id",
                "ray_namespace",
                "probe",
                "state",
            }
            assert "10.0.0.9" not in attrs.values()
            assert "secret-high-cardinality-pg" not in attrs.values()
            assert "do-not-label" not in attrs.values()
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


@pytest.mark.anyio
async def test_issue_627_supervisor_reconciles_desired_nodes_without_model_placement(tmp_path) -> None:
    config = load_topology_config(_write_topology_config(tmp_path))
    submitted: list[str] = []
    manager = TopologyManager(
        config,
        provider_task_lister=lambda _config: [],
        provider_task_submitter=lambda _config, node: submitted.append(node.alias),
        ray_node_lister=lambda: [],
    )
    supervisor = ModelActorSupervisor(
        specs=[],
        topology_manager=manager,
        runtime_factory=lambda _spec, _generation: None,
        placement_reconciler=lambda desired: {"ok": True, "blocked": {}, "node_pins": {}},
        scheduler_sync=lambda _registrations: None,
        control_plane_dependencies=[],
    )

    out = await supervisor.reconcile_once()

    assert submitted == ["mint-worker-0"]
    assert out["snapshot"]["topology"]["nodes"]["mint-worker-0"]["state"] == "provisioning"
