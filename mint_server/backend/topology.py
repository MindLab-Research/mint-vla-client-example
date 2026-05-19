from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
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
CommandRunner = Callable[[list[str], float], str]

LIVE_PROVIDER_TASK_STATES = {"Queue", "Staging", "Running", "Initialized"}
TERMINAL_PROVIDER_TASK_STATES = {"Succeeded", "Failed", "Cancelled", "Stopped", "Killing", "Terminated"}


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


def _strip_volc_json_banner(output: str) -> Any:
    text = str(output or "")
    starts = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
    if not starts:
        raise ValueError("volc output did not contain JSON payload")
    return json.loads(text[min(starts):])


def _default_command_runner(argv: list[str], timeout_s: float) -> str:
    result = subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(timeout_s),
    )
    return result.stdout


def _task_gpu_count(task: dict[str, Any]) -> int | None:
    flavor_gpus = {
        "ml.hpcpni2l.28xlarge": 8,
        "ml.hpcpni2l.14xlarge": 4,
        "ml.hpcpni2l.7xlarge": 2,
        "ml.r3i.4xlarge": 0,
    }
    total = 0
    seen = False
    for spec in task.get("TaskRoleSpecs") or []:
        if not isinstance(spec, dict):
            continue
        try:
            replicas = int(spec.get("RoleReplicas") or 0)
        except Exception:
            replicas = 0
        flavor = str(
            spec.get("ResourceSpecId")
            or (spec.get("ResourceSpec") or {}).get("FlavorID")
            or ""
        )
        if flavor in flavor_gpus:
            seen = True
            total += replicas * flavor_gpus[flavor]
    return total if seen else None


def _extract_node_ip_from_worker_logs(logs: str) -> str | None:
    for pattern in (
        r"Local node IP:\s*([0-9]+(?:\.[0-9]+){3})",
        r"published IP:\s*([0-9]+(?:\.[0-9]+){3})",
        r"Ray head IP:\s*([0-9]+(?:\.[0-9]+){3})",
    ):
        match = re.search(pattern, logs)
        if match:
            return match.group(1)
    return None


class VolcanoTopologyProvider:
    def __init__(
        self,
        *,
        volc_bin: str = "/root/.volc/bin/volc",
        command_runner: CommandRunner | None = None,
        timeout_s: float = 30.0,
        fetch_logs: bool = True,
    ) -> None:
        self._volc_bin = str(volc_bin or "/root/.volc/bin/volc")
        self._command_runner = command_runner or _default_command_runner
        self._timeout_s = float(timeout_s)
        self._fetch_logs = bool(fetch_logs)

    def _run(self, argv: list[str], *, timeout_s: float | None = None) -> str:
        return self._command_runner(argv, self._timeout_s if timeout_s is None else float(timeout_s))

    def list_tasks(self, config: TopologyConfig) -> Iterable[ProviderTaskState]:
        output = self._run(
            [self._volc_bin, "ml_task", "list", "--output", "json", "--limit", "200"]
        )
        payload = _strip_volc_json_banner(output)
        if not isinstance(payload, list):
            raise ValueError("volc ml_task list JSON payload must be a list")
        task_names = {
            stable_provider_task_name(config.deployment_env, alias): alias
            for alias in config.nodes
        }
        states: list[ProviderTaskState] = []
        for task in payload:
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("JobName") or "").strip()
            alias = task_names.get(task_name)
            if alias is None:
                continue
            raw_state = str(task.get("Status") or "").strip()
            live = raw_state in LIVE_PROVIDER_TASK_STATES
            task_id = str(task.get("JobId") or "").strip() or None
            node_ip = None
            error = None
            if live and task_id and self._fetch_logs:
                try:
                    logs = self._run(
                        [self._volc_bin, "ml_task", "logs", "-t", task_id, "-i", "worker_0"],
                        timeout_s=10.0,
                    )
                    node_ip = _extract_node_ip_from_worker_logs(logs)
                except Exception as e:
                    error = f"volc logs failed: {type(e).__name__}: {e}"
            states.append(
                ProviderTaskState(
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
            )
        return states

    def submit_task(self, config: TopologyConfig, node: TopologyNodeDesired) -> None:
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
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            f.write(rendered)
        try:
            self._run([self._volc_bin, "ml_task", "submit", "-c", temp_path, "--output", "json"])
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


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
        provider = VolcanoTopologyProvider(
            volc_bin=str((config.providers.get("volcano") or {}).get("volc_bin") or "/root/.volc/bin/volc")
            if isinstance(config.providers.get("volcano"), dict)
            else "/root/.volc/bin/volc"
        )
        return provider.list_tasks
    return empty_provider_task_lister


def default_provider_task_submitter_for_config(config: TopologyConfig) -> ProviderTaskSubmitter:
    providers = {node.provider for node in config.nodes.values()}
    if providers == {"volcano"}:
        provider = VolcanoTopologyProvider(
            volc_bin=str((config.providers.get("volcano") or {}).get("volc_bin") or "/root/.volc/bin/volc")
            if isinstance(config.providers.get("volcano"), dict)
            else "/root/.volc/bin/volc"
        )
        return provider.submit_task
    return noop_provider_task_submitter


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
