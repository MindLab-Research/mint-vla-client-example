from __future__ import annotations

import ipaddress
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
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
    labels: dict[str, str] = field(default_factory=dict)
    gpu_count: int | None = None


@dataclass(frozen=True)
class TopologyConfig:
    version: int
    deployment_env: str
    cluster_id: str
    state_path: str
    nodes: dict[str, TopologyNodeDesired]
    providers: dict[str, Any] = field(default_factory=dict)


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


def default_topology_state_path(deployment_env: str) -> str:
    env = str(deployment_env or "").strip() or "dev"
    return f"/vePFS-Mindverse/share/mint/{env}/runtime/topology_state.yaml"


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
        providers=providers,
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


def empty_provider_task_lister(_config: TopologyConfig) -> Iterable[ProviderTaskState]:
    return []


def noop_provider_task_submitter(_config: TopologyConfig, _node: TopologyNodeDesired) -> None:
    return None


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
        self._provider_task_lister = provider_task_lister or empty_provider_task_lister
        self._provider_task_submitter = provider_task_submitter or noop_provider_task_submitter
        self._ray_node_lister = ray_node_lister or default_ray_node_lister
        self._state: TopologyRuntimeState | None = None

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
        for alias, desired in sorted(config.nodes.items()):
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
                    last_error=None,
                )
                continue
            if task is None or not task.live:
                self._provider_task_submitter(config, desired)
                runtime_nodes[alias] = TopologyNodeRuntime(
                    alias=alias,
                    state="provisioning",
                    provider=desired.provider,
                    provider_task_name=task_name,
                    template=desired.template,
                    enabled=True,
                    provider_task_id=task.task_id if task else None,
                    node_ip=task.node_ip if task else None,
                    gpu_count=task.gpu_count if task else desired.gpu_count,
                    last_error=task.error if task else "provider task is not live",
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
                    provider_task_id=task.task_id,
                    gpu_count=task.gpu_count or desired.gpu_count,
                    last_error="provider task has no node_ip",
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
                    provider_task_id=task.task_id,
                    node_ip=node_ip,
                    gpu_count=task.gpu_count or desired.gpu_count,
                    last_error="Ray node is not alive",
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
                provider_task_id=task.task_id,
                node_ip=node_ip,
                ray_node_id=ray_node.ray_node_id,
                gpu_count=gpu_count,
                validated_at=time.time(),
                last_error=None,
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
