from __future__ import annotations

import ipaddress
import importlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MINT_WORKER_ALIAS_RE = re.compile(r"^mint-worker-(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class TopologyNodeDesired:
    alias: str
    provider: str
    template: str
    role: str = "gpu"
    enabled: bool = True
    mount_ok: bool = True
    runtime_env_ok: bool = True
    labels: dict[str, str] = field(default_factory=dict)
    gpu_count: int | None = None


@dataclass(frozen=True)
class TopologyConfig:
    version: int
    deployment_env: str
    cluster_id: str
    state_path: str
    nodes: dict[str, TopologyNodeDesired]
    models: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    ray_dashboard_url: str | None = None
    ray_head_ip_path: str | None = None


@dataclass(frozen=True)
class ProviderTaskState:
    alias: str
    provider: str
    task_name: str
    live: bool
    task_id: str | None = None
    node_ip: str | None = None
    gpu_count: int | None = None
    raw_state: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RayNodeState:
    node_ip: str
    ray_node_id: str | None = None
    alive: bool = True
    gpu_count: int | None = None
    hostname: str | None = None


@dataclass(frozen=True)
class TopologyNodeRuntime:
    alias: str
    state: str
    provider: str
    provider_task_name: str
    template: str
    enabled: bool
    role: str = "gpu"
    mount_ok: bool = True
    runtime_env_ok: bool = True
    node_ip: str | None = None
    ray_node_id: str | None = None
    provider_task_id: str | None = None
    gpu_count: int | None = None
    validated_at: float | None = None
    last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready" and bool(self.node_ip)


@dataclass(frozen=True)
class TopologyRuntimeState:
    version: int
    deployment_env: str
    cluster_id: str
    observed_at: float
    source: str
    nodes: dict[str, TopologyNodeRuntime]

    def ready_node_ip(self, alias: str) -> str | None:
        node = self.nodes.get(alias)
        if node is None or not node.ready:
            return None
        return node.node_ip


ProviderTaskLister = Callable[[TopologyConfig], Iterable[ProviderTaskState]]
ProviderTaskSubmitter = Callable[[TopologyConfig, TopologyNodeDesired], Any]
RayNodeLister = Callable[[], Iterable[RayNodeState]]

LIVE_PROVIDER_TASK_STATES = {
    "Deploying",
    "Initialized",
    "Queue",
    "Queueing",
    "Running",
    "Staging",
}
TERMINAL_PROVIDER_TASK_STATES = {"Succeeded", "Failed", "Cancelled", "Stopped", "Killing", "Terminated"}


class UnsupportedTopologyProviderError(ValueError):
    pass


def is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(str(value).strip())
        return True
    except ValueError:
        return False


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return float(raw)


def _validate_alias(alias: str) -> str:
    value = str(alias or "").strip()
    if not MINT_WORKER_ALIAS_RE.fullmatch(value):
        raise ValueError(f"topology node alias must match mint-worker-{{idx}}, got {alias!r}")
    return value


def stable_provider_task_name(deployment_env: str, alias: str) -> str:
    alias = _validate_alias(alias)
    env = str(deployment_env or "").strip()
    if not env:
        raise ValueError("deployment_env is required")
    idx = alias.rsplit("-", 1)[-1]
    return f"mint-{env}-worker-{idx}"


def worker_alias_index(alias: str) -> int:
    alias = _validate_alias(alias)
    return int(alias.rsplit("-", 1)[-1])


def default_topology_state_path(deployment_env: str) -> str:
    env = str(deployment_env or "").strip() or "dev"
    return f"/vePFS-Mindverse/share/mint/{env}/runtime/topology_state.yaml"


def default_ray_head_ip_path(deployment_env: str) -> str:
    env = str(deployment_env or "").strip() or "dev"
    return f"/vePFS-Mindverse/share/mint/{env}/ray/head-address/ray_head_ip.txt"


def load_topology_config(path: str | os.PathLike[str]) -> TopologyConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    root = _require_mapping(payload, context="topology config")
    version = int(root.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported topology config version: {version}")
    deployment_env = str(root.get("deployment_env") or os.environ.get("MINT_DEPLOYMENT_ENV") or "dev").strip()
    cluster_id = str(root.get("cluster_id") or os.environ.get("MINT_CLUSTER_ID") or "volcano").strip()
    if not deployment_env:
        raise ValueError("topology config deployment_env is required")
    if not cluster_id:
        raise ValueError("topology config cluster_id is required")
    state_path = str(
        os.environ.get("MINT_TOPOLOGY_STATE_PATH")
        or root.get("state_path")
        or default_topology_state_path(deployment_env)
    ).strip()
    nodes_root = _require_mapping(root.get("nodes") or {}, context="topology config nodes")
    desired_raw = nodes_root.get("desired") or []
    if not isinstance(desired_raw, list):
        raise ValueError("topology config nodes.desired must be a list")
    providers = _require_mapping(root.get("providers") or {}, context="topology config providers")
    models = _require_mapping(root.get("models") or {}, context="topology config models")
    ray_root = _require_mapping(root.get("ray") or {}, context="topology config ray")
    ray_dashboard_url = str(
        os.environ.get("MINT_RAY_DASHBOARD_URL")
        or ray_root.get("dashboard_url")
        or ""
    ).strip() or None
    ray_head_ip_path = str(
        os.environ.get("MINT_RAY_HEAD_ADDRESS_PATH")
        or ray_root.get("head_ip_path")
        or default_ray_head_ip_path(deployment_env)
    ).strip()
    nodes: dict[str, TopologyNodeDesired] = {}
    for idx, item in enumerate(desired_raw):
        item_map = _require_mapping(item, context=f"topology config nodes.desired[{idx}]")
        alias = _validate_alias(str(item_map.get("alias") or ""))
        if alias in nodes:
            raise ValueError(f"duplicate topology node alias: {alias}")
        provider = str(item_map.get("provider") or "").strip()
        template = str(item_map.get("template") or "").strip()
        if not provider:
            raise ValueError(f"topology node {alias} provider is required")
        if provider not in providers:
            raise ValueError(f"topology node {alias} references unknown provider {provider!r}")
        provider_templates = (
            providers.get(provider, {}).get("templates", {})
            if isinstance(providers.get(provider), dict)
            else {}
        )
        if template and isinstance(provider_templates, dict) and template not in provider_templates:
            raise ValueError(f"topology node {alias} references unknown template {template!r}")
        labels_raw = item_map.get("labels") or {}
        if not isinstance(labels_raw, dict):
            raise ValueError(f"topology node {alias} labels must be a mapping")
        node_gpu_count = item_map.get("gpu_count")
        if node_gpu_count is None and template and isinstance(provider_templates, dict):
            template_cfg = provider_templates.get(template) or {}
            if isinstance(template_cfg, dict):
                node_gpu_count = template_cfg.get("gpu_count")
        node = TopologyNodeDesired(
            alias=alias,
            provider=provider,
            template=template,
            role=str(item_map.get("role") or "gpu"),
            enabled=bool(item_map.get("enabled", True)),
            mount_ok=bool(item_map.get("mount_ok", True)),
            runtime_env_ok=bool(item_map.get("runtime_env_ok", True)),
            labels={str(k): str(v) for k, v in labels_raw.items()},
            gpu_count=None if node_gpu_count is None else int(node_gpu_count),
        )
        nodes[alias] = node
    return TopologyConfig(
        version=version,
        deployment_env=deployment_env,
        cluster_id=cluster_id,
        state_path=state_path,
        nodes=nodes,
        models=models,
        providers=providers,
        ray_dashboard_url=ray_dashboard_url,
        ray_head_ip_path=ray_head_ip_path,
    )


def load_topology_config_from_env() -> TopologyConfig | None:
    path = str(os.environ.get("MINT_TOPOLOGY_CONFIG_PATH") or "").strip()
    if not path:
        return None
    return load_topology_config(path)


def default_ray_node_lister() -> Iterable[RayNodeState]:
    try:
        import ray

        if not ray.is_initialized():
            return []
        rows = ray.nodes()
    except Exception:
        return []
    nodes: list[RayNodeState] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        resources = row.get("Resources") or {}
        gpu_count = None
        if isinstance(resources, dict) and "GPU" in resources:
            try:
                gpu_count = int(resources.get("GPU") or 0)
            except Exception:
                gpu_count = None
        node_ip = str(row.get("NodeManagerAddress") or "").strip()
        if not node_ip:
            continue
        nodes.append(
            RayNodeState(
                node_ip=node_ip,
                ray_node_id=str(row.get("NodeID") or "").strip() or None,
                alive=bool(row.get("Alive", False)),
                gpu_count=gpu_count,
                hostname=str(row.get("NodeName") or "").strip() or None,
            )
        )
    return nodes


def _ray_nodes_from_dashboard_payload(payload: Any) -> list[RayNodeState]:
    if not isinstance(payload, dict):
        return []
    result = payload.get("data")
    if isinstance(result, dict):
        result = result.get("result", result)
    if isinstance(result, dict):
        rows = result.get("result", [])
    elif isinstance(result, list):
        rows = result
    else:
        rows = []
    if not isinstance(rows, list):
        return []
    nodes: list[RayNodeState] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        node_ip = str(row.get("node_ip") or row.get("nodeName") or "").strip()
        if not node_ip:
            continue
        resources = row.get("resources_total") or row.get("resourcesTotal") or {}
        gpu_count = None
        if isinstance(resources, dict) and "GPU" in resources:
            try:
                gpu_count = int(resources.get("GPU") or 0)
            except Exception:
                gpu_count = None
        state = str(row.get("state") or "").strip().upper()
        nodes.append(
            RayNodeState(
                node_ip=node_ip,
                ray_node_id=str(row.get("node_id") or row.get("nodeId") or "").strip() or None,
                alive=state == "ALIVE" if state else bool(row.get("alive", True)),
                gpu_count=gpu_count,
                hostname=str(row.get("node_name") or row.get("hostname") or "").strip() or None,
            )
        )
    return nodes


def ray_dashboard_node_lister(
    *,
    dashboard_url: str | None = None,
    head_ip_path: str | os.PathLike[str] | None = None,
    timeout_s: float = 5.0,
) -> Iterable[RayNodeState]:
    url = str(dashboard_url or "").strip().rstrip("/")
    if not url and head_ip_path:
        try:
            head_ip = Path(head_ip_path).read_text(encoding="utf-8").strip()
        except OSError:
            head_ip = ""
        if head_ip:
            url = f"http://{head_ip}:8265"
    if not url:
        return []
    try:
        with urllib.request.urlopen(f"{url}/api/v0/nodes", timeout=float(timeout_s)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return _ray_nodes_from_dashboard_payload(payload)


def default_ray_node_lister_for_config(config: TopologyConfig | None) -> RayNodeLister:
    def _list_nodes() -> Iterable[RayNodeState]:
        by_ip: dict[str, RayNodeState] = {}
        for node in default_ray_node_lister():
            by_ip[node.node_ip] = node
        if config is not None:
            for node in ray_dashboard_node_lister(
                dashboard_url=config.ray_dashboard_url,
                head_ip_path=config.ray_head_ip_path,
            ):
                by_ip[node.node_ip] = node
        return list(by_ip.values())

    return _list_nodes


def _object_get(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict):
            for key in (name, _camel_to_pascal(name), _pascal_to_snake(name)):
                if key in value:
                    return value[key]
        if hasattr(value, name):
            return getattr(value, name)
        snake = _pascal_to_snake(name)
        if hasattr(value, snake):
            return getattr(value, snake)
    return None


def _camel_to_pascal(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    if "_" in raw:
        return "".join(part[:1].upper() + part[1:] for part in raw.split("_") if part)
    return raw[:1].upper() + raw[1:]


def _pascal_to_snake(value: str) -> str:
    raw = str(value or "")
    out = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    out = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", out)
    return out.lower()


def _task_gpu_count(task: Any) -> int | None:
    flavor_gpus = {
        "ml.hpcpni2l.28xlarge": 8,
        "ml.hpcpni2l.14xlarge": 4,
        "ml.hpcpni2l.7xlarge": 2,
        "ml.r3i.4xlarge": 0,
    }
    resource_config = _object_get(task, "ResourceConfig", "resource_config")
    specs = _object_get(resource_config, "Roles", "roles") if resource_config is not None else None
    if specs is None:
        specs = _object_get(task, "TaskRoleSpecs", "task_role_specs") or []
    total = 0
    seen = False
    for spec in specs or []:
        try:
            replicas = int(_object_get(spec, "RoleReplicas", "replicas") or 0)
        except Exception:
            replicas = 0
        resource = _object_get(spec, "ResourceSpec", "resource")
        flavor = str(_object_get(spec, "ResourceSpecId", "instance_type_id") or _object_get(resource, "FlavorID", "instance_type_id") or "")
        if flavor in flavor_gpus:
            seen = True
            total += replicas * flavor_gpus[flavor]
    return total if seen else None


def _job_state(task: Any) -> str:
    status = _object_get(task, "Status", "status")
    return str(_object_get(status, "State", "state") or status or "").strip()


def _volcano_image_type_for_create_job(raw_type: Any, image_url: str) -> str:
    image_type = str(raw_type or "").strip()
    if image_type:
        return image_type
    if ".cr.volces.com/" in image_url:
        return "VolcEngine"
    return "Public"


def _job_id(task: Any) -> str | None:
    return str(_object_get(task, "JobId", "Id", "id") or "").strip() or None


def _job_name(task: Any) -> str:
    return str(_object_get(task, "JobName", "Name", "name") or "").strip()


def _provider_task_sort_key(state: ProviderTaskState) -> tuple[int, str]:
    return (1 if state.live else 0, state.task_id or "")


def _extract_instance_node_ip(instance: Any) -> str | None:
    ips = _object_get(instance, "Ips", "ips")
    for name in ("PrimaryIp", "primary_ip", "HostIp", "host_ip"):
        value = str(_object_get(ips, name) or "").strip()
        if value and is_ip_address(value):
            return value
    return None


class VolcanoTopologyProvider:
    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        region: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._client_factory = client_factory or (
            lambda: _create_volcano_mlplatform_client(
                region=region,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def list_tasks(self, config: TopologyConfig) -> Iterable[ProviderTaskState]:
        sdk = _volcano_sdk_module()
        response = self.client.list_jobs(
            sdk.ListJobsRequest(
                name_contains=f"mint-{config.deployment_env}-worker-",
                page_number=1,
                page_size=100,
            )
        )
        tasks = _object_get(response, "Items", "items") or []
        task_names = {
            stable_provider_task_name(config.deployment_env, alias): alias
            for alias in config.nodes
        }
        states_by_alias: dict[str, ProviderTaskState] = {}
        for task in tasks:
            task_name = _job_name(task)
            alias = task_names.get(task_name)
            if alias is None:
                continue
            raw_state = _job_state(task)
            live = raw_state in LIVE_PROVIDER_TASK_STATES
            task_id = _job_id(task)
            node_ip = None
            error = None
            if live and task_id:
                try:
                    instance_response = self.client.list_job_instances(
                        sdk.ListJobInstancesRequest(job_id=task_id, page_number=1, page_size=20)
                    )
                    for instance in _object_get(instance_response, "Items", "items") or []:
                        node_ip = _extract_instance_node_ip(instance)
                        if node_ip:
                            break
                except Exception as e:
                    error = f"volcano list_job_instances failed: {type(e).__name__}: {e}"
            state = ProviderTaskState(
                alias=alias,
                provider="volcano",
                task_name=task_name,
                live=live,
                task_id=task_id,
                node_ip=node_ip,
                gpu_count=_task_gpu_count(task),
                raw_state=raw_state,
                error=error,
            )
            previous = states_by_alias.get(alias)
            if previous is None or _provider_task_sort_key(state) > _provider_task_sort_key(previous):
                states_by_alias[alias] = state
        return states_by_alias.values()

    def submit_task(self, config: TopologyConfig, node: TopologyNodeDesired) -> None:
        request = build_volcano_create_job_request(config, node)
        self.client.create_job(request)


def _volcano_sdk_module() -> Any:
    try:
        return importlib.import_module("volcenginesdkmlplatform20240701")
    except ImportError as e:
        if _prepend_host_venv_site_packages():
            try:
                return importlib.import_module("volcenginesdkmlplatform20240701")
            except ImportError:
                pass
        raise RuntimeError("volcengine-python-sdk is required for Volcano topology node management") from e


def _import_volcano_sdk_modules() -> tuple[Any, Any]:
    try:
        return (
            importlib.import_module("volcenginesdkcore"),
            importlib.import_module("volcenginesdkmlplatform20240701"),
        )
    except ImportError as e:
        if _prepend_host_venv_site_packages():
            try:
                return (
                    importlib.import_module("volcenginesdkcore"),
                    importlib.import_module("volcenginesdkmlplatform20240701"),
                )
            except ImportError:
                pass
        raise RuntimeError("volcengine-python-sdk is required for Volcano topology node management") from e


def _prepend_host_venv_site_packages() -> bool:
    env_root = str(os.environ.get("PFS_RUNTIME_ENV_ROOT") or "").strip()
    if not env_root:
        return False
    try:
        from ..runtime_env import host_venv_site_packages

        path = host_venv_site_packages(env_root)
    except Exception:
        return False
    if not path or not Path(path).is_dir():
        return False
    if path not in sys.path:
        sys.path.insert(0, path)
    return True


def _legacy_volc_cli_credentials() -> dict[str, str]:
    """Read legacy Volcano CLI credentials for SDK compatibility.

    The Volcano SDK default chain reads ~/.volcengine/config.json, while the
    older ml_task CLI used ~/.volc/config plus ~/.volc/credentials. Keep this
    bridge local to the driver process and never expose the returned values in
    topology config, logs, metrics, or state snapshots.
    """

    has_modern_source = any(
        str(os.environ.get(name) or "").strip()
        for name in (
            "VOLCENGINE_ACCESS_KEY",
            "VOLCENGINE_SECRET_KEY",
            "VOLCENGINE_SESSION_TOKEN",
            "VOLCENGINE_CLI_CONFIG_FILE",
        )
    ) or Path(os.path.expanduser("~/.volcengine/config.json")).is_file()
    if has_modern_source:
        return {}

    volc_cli_home = _legacy_volc_cli_home()
    if volc_cli_home is None:
        return {}
    cred_path = volc_cli_home / "credentials"
    if not cred_path.is_file():
        return {}
    creds = ConfigParser()
    creds.read(cred_path, encoding="utf-8")
    section = os.environ.get("VOLC_PROFILE") or os.environ.get("VOLCENGINE_PROFILE") or "default"
    if not creds.has_section(section):
        return {}
    ak = str(creds.get(section, "access_key_id", fallback="")).strip()
    sk = str(creds.get(section, "secret_access_key", fallback="")).strip()
    token = str(creds.get(section, "session_token", fallback="")).strip()
    if not ak or not sk:
        return {}

    config_path = volc_cli_home / "config"
    region = ""
    if config_path.is_file():
        cfg = ConfigParser()
        cfg.read(config_path, encoding="utf-8")
        if cfg.has_section(section):
            region = str(cfg.get(section, "region", fallback="")).strip()
        if not region and cfg.has_section("default"):
            region = str(cfg.get("default", "region", fallback="")).strip()

    out = {"ak": ak, "sk": sk}
    if token:
        out["session_token"] = token
    if region:
        out["region"] = region
    return out


def _legacy_volc_cli_home() -> Path | None:
    for name in ("VOLC_CLI_HOME", "VOLCENGINE_LEGACY_CLI_HOME"):
        raw = str(os.environ.get(name) or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.is_dir():
                return path

    candidates = [Path(os.path.expanduser("~/.volc")), Path("/root/.volc")]
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            return path
    return None


def _create_volcano_mlplatform_client(
    *,
    region: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
) -> Any:
    volcenginesdkcore, sdk = _import_volcano_sdk_modules()
    configuration = volcenginesdkcore.Configuration()
    if region:
        configuration.region = str(region)
    legacy_credentials = _legacy_volc_cli_credentials()
    if legacy_credentials:
        configuration.ak = legacy_credentials["ak"]
        configuration.sk = legacy_credentials["sk"]
        if legacy_credentials.get("session_token"):
            configuration.session_token = legacy_credentials["session_token"]
        if not region and legacy_credentials.get("region"):
            configuration.region = legacy_credentials["region"]
    if connect_timeout is not None:
        configuration.connect_timeout = float(connect_timeout)
    if read_timeout is not None:
        configuration.read_timeout = float(read_timeout)
    configuration.debug = False
    return sdk.MLPLATFORM20240701Api(volcenginesdkcore.ApiClient(configuration))


def build_volcano_create_job_request(config: TopologyConfig, node: TopologyNodeDesired) -> Any:
    sdk = _volcano_sdk_module()
    provider_cfg = config.providers.get(node.provider) or {}
    if not isinstance(provider_cfg, dict):
        raise ValueError(f"provider config for {node.provider!r} must be a mapping")
    template_cfg = (provider_cfg.get("templates") or {}).get(node.template) or {}
    if not isinstance(template_cfg, dict):
        raise ValueError(f"template config for {node.template!r} must be a mapping")
    template_path = str(template_cfg.get("template_path") or "").strip()
    queue_id = str(template_cfg.get("resource_queue_id") or template_cfg.get("ResourceQueueID") or "").strip()
    if not template_path:
        raise ValueError(f"topology node {node.alias} template_path is required")
    if not queue_id:
        raise ValueError(f"topology node {node.alias} resource_queue_id is required")
    task_name = stable_provider_task_name(config.deployment_env, node.alias)
    rendered = render_volcano_worker_template(
        template_path=template_path,
        task_name=task_name,
        resource_queue_id=queue_id,
        worker_alias=node.alias,
        deployment_env=config.deployment_env,
        cluster_id=config.cluster_id,
    )
    payload = yaml.safe_load(rendered) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"rendered Volcano worker template for {node.alias} must be a mapping")
    role_specs = []
    for raw_role in payload.get("TaskRoleSpecs") or []:
        if not isinstance(raw_role, dict):
            raise ValueError(f"TaskRoleSpecs item for {node.alias} must be a mapping")
        role_name = str(raw_role.get("RoleName") or "").strip()
        replicas = int(raw_role.get("RoleReplicas") or 1)
        flavor = str(raw_role.get("Flavor") or raw_role.get("ResourceSpecId") or "").strip()
        if not role_name or not flavor:
            raise ValueError(f"TaskRoleSpecs item for {node.alias} must include RoleName and Flavor")
        role_specs.append(
            sdk.RoleForCreateJobInput(
                name=role_name,
                replicas=replicas,
                resource=sdk.ResourceForCreateJobInput(instance_type_id=flavor),
            )
        )
    storage_specs = []
    uses_tos_storage = False
    for raw_storage in payload.get("Storages") or []:
        if not isinstance(raw_storage, dict):
            raise ValueError(f"Storages item for {node.alias} must be a mapping")
        raw_storage_type = str(raw_storage.get("Type") or raw_storage.get("type") or "").strip()
        if raw_storage_type in {"Tos", "TosFuse"}:
            uses_tos_storage = True
        storage_specs.append(_volcano_storage_for_create_job(sdk, raw_storage))
    image_url = str(payload.get("ImageUrl") or "").strip()
    if not image_url:
        raise ValueError(f"rendered Volcano worker template for {node.alias} must include ImageUrl")
    image_type = _volcano_image_type_for_create_job(payload.get("ImageType"), image_url)
    command = str(payload.get("Entrypoint") or "").strip()
    if not command:
        raise ValueError(f"rendered Volcano worker template for {node.alias} must include Entrypoint")
    return sdk.CreateJobRequest(
        name=task_name,
        description=str(payload.get("Description") or f"{task_name} topology worker"),
        resource_config=sdk.ResourceConfigForCreateJobInput(
            resource_queue_id=queue_id,
            max_runtime_seconds=int(payload.get("ActiveDeadlineSeconds") or 0),
            roles=role_specs,
        ),
        runtime_config=sdk.RuntimeConfigForCreateJobInput(
            command=command,
            framework=str(payload.get("Framework") or "Custom"),
            image=sdk.ImageForCreateJobInput(type=image_type, url=image_url),
        ),
        storage_config=sdk.StorageConfigForCreateJobInput(
            credential=(
                sdk.ConvertCredentialForCreateJobInput(use_service_linked_role=True)
                if uses_tos_storage
                else None
            ),
            storages=storage_specs,
        )
        if storage_specs
        else None,
    )


def _volcano_storage_for_create_job(sdk: Any, raw: dict[str, Any]) -> Any:
    storage_type = str(raw.get("Type") or raw.get("type") or "").strip()
    mount_path = str(raw.get("MountPath") or raw.get("mount_path") or "").strip()
    read_only = bool(raw.get("ReadOnly", raw.get("read_only", False)))
    if not storage_type or not mount_path:
        raise ValueError("Volcano storage item must include Type and MountPath")
    config_obj = None
    sdk_storage_type = storage_type
    if storage_type == "Vepfs":
        vepfs_id = str(raw.get("Id") or "").strip() or None
        file_system_name = str(raw.get("FileSystemName") or "").strip() or None
        if not vepfs_id and not file_system_name:
            raise ValueError(
                "Volcano Vepfs storage must include Id or FileSystemName; "
                f"mount_path={mount_path!r}"
            )
        config_obj = sdk.ConfigForCreateJobInput(
            vepfs=sdk.VepfsForCreateJobInput(
                id=vepfs_id,
                file_system_name=file_system_name,
                sub_path=str(raw.get("SubPath") or "").strip() or None,
                host_path=str(raw.get("HostPath") or "").strip() or None,
            )
        )
    elif storage_type == "TosFuse":
        sdk_storage_type = "Tos"
        config_obj = sdk.ConfigForCreateJobInput(
            tos=sdk.TosForCreateJobInput(
                bucket=str(raw.get("Bucket") or "").strip() or None,
                prefix=str(raw.get("Prefix") or "").strip() or None,
            )
        )
    else:
        raise ValueError(f"unsupported Volcano storage type in topology worker template: {storage_type!r}")
    return sdk.StorageForCreateJobInput(
        type=sdk_storage_type,
        mount_path=mount_path,
        read_only=read_only,
        config=config_obj,
    )


def render_volcano_worker_template(
    *,
    template_path: str | os.PathLike[str],
    task_name: str,
    resource_queue_id: str,
    worker_alias: str,
    deployment_env: str,
    cluster_id: str,
) -> str:
    path = Path(template_path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'^TaskName:\s*"[^"]*"', f'TaskName: "{task_name}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(
        r'^Description:\s*"[^"]*"',
        f'Description: "{task_name} topology worker"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = text.replace("<GPU_QUEUE_ID>", resource_queue_id)
    text = re.sub(
        r'^ResourceQueueID:\s*"[^"]*"',
        f'ResourceQueueID: "{resource_queue_id}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    injection = (
        "  export MINT_WORKER_ALIAS=\"{alias}\"\n"
        "  export MINT_DEPLOYMENT_ENV=\"{env}\"\n"
        "  export MINT_CLUSTER_ID=\"{cluster}\"\n"
    ).format(alias=worker_alias, env=deployment_env, cluster=cluster_id)
    marker = (
        "\n  mkdir -p /vePFS-Mindverse/share/mint/{env}/runtime/workers\n"
        "  printf '%s\\n' \"alias={alias}\" \"task_name={task}\" \"deployment_env={env}\" \"cluster_id={cluster}\" "
        "> /vePFS-Mindverse/share/mint/{env}/runtime/workers/{alias}.env\n"
    ).format(alias=worker_alias, task=task_name, env=deployment_env, cluster=cluster_id)
    if "  export MINT_WORKER_ALIAS=" not in text:
        text = text.replace("Entrypoint: |\n", "Entrypoint: |\n" + injection, 1)
    if f"/runtime/workers/{worker_alias}.env" not in text:
        text = text.replace(injection, injection + marker, 1)
    return text


def default_provider_task_lister_for_config(config: TopologyConfig) -> ProviderTaskLister:
    providers = {node.provider for node in config.nodes.values()}
    if providers == {"volcano"}:
        provider_cfg = config.providers.get("volcano") if isinstance(config.providers.get("volcano"), dict) else {}
        provider = VolcanoTopologyProvider(
            region=str(provider_cfg.get("region") or "").strip() or None,
            connect_timeout=_optional_float(provider_cfg.get("connect_timeout_s")),
            read_timeout=_optional_float(provider_cfg.get("read_timeout_s")),
        )
        return provider.list_tasks
    if providers:
        raise UnsupportedTopologyProviderError(
            f"unsupported topology providers: {', '.join(sorted(providers))}; supported providers: volcano"
        )
    return empty_provider_task_lister


def default_provider_task_submitter_for_config(config: TopologyConfig) -> ProviderTaskSubmitter:
    providers = {node.provider for node in config.nodes.values()}
    if providers == {"volcano"}:
        provider_cfg = config.providers.get("volcano") if isinstance(config.providers.get("volcano"), dict) else {}
        region = str(provider_cfg.get("region") or "").strip() or None
        connect_timeout = _optional_float(provider_cfg.get("connect_timeout_s"))
        read_timeout = _optional_float(provider_cfg.get("read_timeout_s"))

        def _submit(config: TopologyConfig, node: TopologyNodeDesired) -> None:
            provider = VolcanoTopologyProvider(
                region=region,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            provider.submit_task(config, node)

        return _submit
    if providers:
        raise UnsupportedTopologyProviderError(
            f"unsupported topology providers: {', '.join(sorted(providers))}; supported providers: volcano"
        )
    return noop_provider_task_submitter


def empty_provider_task_lister(_config: TopologyConfig) -> Iterable[ProviderTaskState]:
    return []


def noop_provider_task_submitter(_config: TopologyConfig, _node: TopologyNodeDesired) -> None:
    return None


def _topology_submit_concurrency() -> int:
    raw = str(os.environ.get("MINT_TOPOLOGY_SUBMIT_CONCURRENCY") or "").strip()
    if not raw:
        return 8
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


class TopologyManager:
    def __init__(
        self,
        config: TopologyConfig | None = None,
        *,
        provider_task_lister: ProviderTaskLister | None = None,
        provider_task_submitter: ProviderTaskSubmitter | None = None,
        ray_node_lister: RayNodeLister | None = None,
    ) -> None:
        self._config = config if config is not None else load_topology_config_from_env()
        self._provider_task_lister = provider_task_lister or (
            default_provider_task_lister_for_config(self._config)
            if self._config is not None
            else empty_provider_task_lister
        )
        self._provider_task_submitter = provider_task_submitter or (
            default_provider_task_submitter_for_config(self._config)
            if self._config is not None
            else noop_provider_task_submitter
        )
        self._ray_node_lister = ray_node_lister or default_ray_node_lister_for_config(self._config)
        self._state: TopologyRuntimeState | None = None
        self._submit_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._config is not None

    @property
    def state(self) -> TopologyRuntimeState | None:
        return self._state

    @property
    def config(self) -> TopologyConfig | None:
        return self._config

    def reconcile_once(self) -> TopologyRuntimeState | None:
        config = self._config
        if config is None:
            return None
        provider_tasks = {task.alias: task for task in self._provider_task_lister(config)}
        ray_nodes_by_ip = {node.node_ip: node for node in self._ray_node_lister() if node.node_ip}
        runtime_nodes: dict[str, TopologyNodeRuntime] = {}
        submit_candidates: list[TopologyNodeDesired] = []
        for alias, desired in sorted(config.nodes.items(), key=lambda item: worker_alias_index(item[0])):
            task_name = stable_provider_task_name(config.deployment_env, alias)
            task = provider_tasks.get(alias)
            if not desired.enabled:
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="disabled",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=False,
                    role=desired.role,
                    mount_ok=desired.mount_ok,
                    runtime_env_ok=desired.runtime_env_ok,
                    last_error=None,
                )
                continue
            if task is None or not task.live:
                submit_candidates.append(desired)
                last_error = (
                    task.error
                    or f"provider task {task.task_name} is not live: state={task.raw_state}"
                    if task
                    else f"missing provider task {task_name}"
                )
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="provisioning",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=True,
                    role=desired.role,
                    mount_ok=desired.mount_ok,
                    runtime_env_ok=desired.runtime_env_ok,
                    provider_task_id=task.task_id if task else None,
                    node_ip=task.node_ip if task else None,
                    gpu_count=task.gpu_count if task else desired.gpu_count,
                    last_error=last_error,
                )
                continue
            node_ip = str(task.node_ip or "").strip()
            if not node_ip:
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="provisioning",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=True,
                    role=desired.role,
                    mount_ok=desired.mount_ok,
                    runtime_env_ok=desired.runtime_env_ok,
                    provider_task_id=task.task_id,
                    gpu_count=task.gpu_count or desired.gpu_count,
                    last_error=f"provider task {task.task_name} has no node_ip",
                )
                continue
            ray_node = ray_nodes_by_ip.get(node_ip)
            if ray_node is None or not ray_node.alive:
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="provisioning",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=True,
                    role=desired.role,
                    mount_ok=desired.mount_ok,
                    runtime_env_ok=desired.runtime_env_ok,
                    provider_task_id=task.task_id,
                    node_ip=node_ip,
                    gpu_count=task.gpu_count or desired.gpu_count,
                    last_error=f"Ray node {node_ip} is not alive",
                )
                continue
            gpu_count = ray_node.gpu_count if ray_node.gpu_count is not None else task.gpu_count
            if desired.gpu_count is not None and gpu_count is not None and int(gpu_count) < int(desired.gpu_count):
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="failed",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=True,
                    role=desired.role,
                    mount_ok=desired.mount_ok,
                    runtime_env_ok=desired.runtime_env_ok,
                    provider_task_id=task.task_id,
                    node_ip=node_ip,
                    ray_node_id=ray_node.ray_node_id,
                    gpu_count=gpu_count,
                    last_error=f"Ray node GPU count {gpu_count} < desired {desired.gpu_count}",
                )
                continue
            runtime_nodes[alias] = TopologyNodeRuntime(
                alias=alias,
                state="ready",
                provider=desired.provider,
                provider_task_name=task_name,
                template=desired.template,
                enabled=True,
                role=desired.role,
                mount_ok=desired.mount_ok,
                runtime_env_ok=desired.runtime_env_ok,
                provider_task_id=task.task_id,
                node_ip=node_ip,
                ray_node_id=ray_node.ray_node_id,
                gpu_count=gpu_count,
                validated_at=time.time(),
                last_error=None,
            )
        submit_errors = self._submit_missing_nodes(config, submit_candidates)
        for alias, error in submit_errors.items():
            node = runtime_nodes.get(alias)
            if node is None:
                continue
            runtime_nodes[alias] = TopologyNodeRuntime(
                alias=node.alias,
                state="failed",
                provider=node.provider,
                provider_task_name=node.provider_task_name,
                template=node.template,
                enabled=node.enabled,
                role=node.role,
                mount_ok=node.mount_ok,
                runtime_env_ok=node.runtime_env_ok,
                provider_task_id=node.provider_task_id,
                node_ip=node.node_ip,
                ray_node_id=node.ray_node_id,
                gpu_count=node.gpu_count,
                validated_at=node.validated_at,
                last_error=error,
            )
        state = TopologyRuntimeState(
            version=config.version,
            deployment_env=config.deployment_env,
            cluster_id=config.cluster_id,
            observed_at=time.time(),
            source=config.cluster_id,
            nodes=runtime_nodes,
        )
        self._state = state
        write_topology_state(config.state_path, state)
        return state

    def _submit_missing_nodes(
        self,
        config: TopologyConfig,
        candidates: list[TopologyNodeDesired],
    ) -> dict[str, str]:
        if not candidates:
            return {}
        candidates = sorted(candidates, key=lambda node: worker_alias_index(node.alias))
        max_workers = min(len(candidates), _topology_submit_concurrency())
        errors: dict[str, str] = {}
        with self._submit_lock:
            if max_workers <= 1:
                for node in candidates:
                    try:
                        self._provider_task_submitter(config, node)
                    except Exception as e:
                        errors[node.alias] = (
                            f"provider task submit failed for {node.alias}: {type(e).__name__}: {e}"
                        )
                return errors
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mint-topology-submit") as pool:
                futures = {
                    pool.submit(self._provider_task_submitter, config, node): node.alias
                    for node in candidates
                }
                for future in as_completed(futures):
                    alias = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        errors[alias] = (
                            f"provider task submit failed for {alias}: {type(e).__name__}: {e}"
                        )
        return errors

    def resolve_alias(self, alias: str) -> tuple[str | None, str | None]:
        if is_ip_address(alias):
            return alias, None
        state = self._state
        if state is None:
            return None, "topology has not been reconciled"
        node = state.nodes.get(alias)
        if node is None:
            return None, f"unknown worker_alias {alias!r}"
        if not node.ready:
            return None, f"worker_alias {alias!r} is not ready: state={node.state} error={node.last_error or ''}".strip()
        return node.node_ip, None

    def snapshot(self) -> dict[str, Any]:
        return topology_state_to_dict(self._state) if self._state is not None else {}


def topology_state_to_dict(state: TopologyRuntimeState | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "version": state.version,
        "deployment_env": state.deployment_env,
        "cluster_id": state.cluster_id,
        "observed_at": state.observed_at,
        "source": state.source,
        "nodes": {
            alias: {
                "state": node.state,
                "provider": node.provider,
                "provider_task_id": node.provider_task_id,
                "provider_task_name": node.provider_task_name,
                "template": node.template,
                "enabled": node.enabled,
                "role": node.role,
                "mount_ok": node.mount_ok,
                "runtime_env_ok": node.runtime_env_ok,
                "node_ip": node.node_ip,
                "ray_node_id": node.ray_node_id,
                "gpu_count": node.gpu_count,
                "validated_at": node.validated_at,
                "last_error": node.last_error,
            }
            for alias, node in sorted(state.nodes.items())
        },
    }


def write_topology_state(path: str | os.PathLike[str], state: TopologyRuntimeState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = topology_state_to_dict(state)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_name = f.name
        yaml.safe_dump(payload, f, sort_keys=True)
    os.replace(tmp_name, target)
