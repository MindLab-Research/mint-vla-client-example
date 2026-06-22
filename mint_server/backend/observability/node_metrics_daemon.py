from __future__ import annotations

import asyncio
import contextlib
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mint_server.config import PFS_PYTHONPATH, PFS_CONTROL_PLANE_PYTHONPATH, actor_runtime_env, otel_env_vars
from mint_server.ray.runtime_env import TIER_CPU
from mint_server.ray.runtime_env import env_nonempty


def sanitize_worker_alias_for_actor_name(worker_alias: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(worker_alias or "").strip()).strip("-")
    return value or "unknown"


def node_metrics_actor_name(worker_alias: str) -> str:
    return f"mint_daemon_node_metrics_{sanitize_worker_alias_for_actor_name(worker_alias)}"


@dataclass(frozen=True)
class NodeMetricsDaemonSpec:
    worker_alias: str
    node_ip: str
    ray_node_id: str | None = None
    gpu_count: int | None = None
    deployment_env: str | None = None
    cluster_id: str | None = None
    actor_name: str | None = None
    is_head_node: bool = False

    def normalized_actor_name(self) -> str:
        return self.actor_name or node_metrics_actor_name(self.worker_alias)


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _sample_interval_s() -> float:
    raw = os.getenv("MINT_NODE_METRICS_SAMPLE_INTERVAL_S", "5")
    try:
        value = float(raw)
    except Exception:
        value = 5.0
    return max(0.1, value)


def _sample_host_metrics() -> dict[str, Any]:
    load_1m = None
    load_5m = None
    load_15m = None
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except Exception:
        pass
    sample: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
    }
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        sample.update(
            {
                "cpu_utilization_ratio": float(psutil.cpu_percent(interval=None)) / 100.0,
                "memory_used_bytes": int(memory.used),
                "memory_total_bytes": int(memory.total),
            }
        )
        disk_path = os.getenv("MINT_NODE_METRICS_DISK_PATH") or _default_disk_metrics_path()
        if Path(disk_path).exists():
            disk = psutil.disk_usage(disk_path)
            sample.update(
                {
                    "disk_path": str(disk_path),
                    "disk_used_bytes": int(disk.used),
                    "disk_total_bytes": int(disk.total),
                }
            )
    except Exception as e:
        sample["host_error"] = f"{type(e).__name__}: {e}"
    return sample


def _default_disk_metrics_path() -> str:
    env = (os.getenv("MINT_DEPLOYMENT_ENV") or "").strip()
    if env:
        return f"/vePFS-Mindverse/share/mint/{env}"
    return "/vePFS-Mindverse/share/mint"


def _nvml_value(fn, handle, *, scale: float = 1.0) -> int | float | None:
    try:
        value = fn(handle)
    except Exception:
        return None
    try:
        out = float(value) / float(scale)
    except Exception:
        return None
    return int(out) if out.is_integer() else out


def _classify_gpu_process(command: str) -> str:
    lowered = str(command or "").lower()
    if "vllm" in lowered:
        return "vllm"
    if "megatron" in lowered:
        return "megatron"
    if "ray" in lowered and "python" in lowered:
        return "ray_python"
    if "python" in lowered:
        return "python"
    return "other"


def _sample_gpu_metrics() -> tuple[list[dict[str, Any]], str | None]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        gpus: list[dict[str, Any]] = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu = {
                "gpu_uuid": str(uuid),
                "memory_used_bytes": int(mem.used),
                "memory_total_bytes": int(mem.total),
                "utilization_gpu_percent": int(util.gpu),
                "utilization_memory_percent": int(util.memory),
            }
            power_draw = _nvml_value(pynvml.nvmlDeviceGetPowerUsage, handle, scale=1000.0)
            if power_draw is not None:
                gpu["power_draw_watts"] = power_draw
            power_limit = _nvml_value(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, scale=1000.0)
            if power_limit is not None:
                gpu["power_limit_watts"] = power_limit
            temperature = _nvml_value(
                lambda h: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                handle,
            )
            if temperature is not None:
                gpu["temperature_celsius"] = temperature
            sm_clock = _nvml_value(
                lambda h: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                handle,
            )
            if sm_clock is not None:
                gpu["sm_clock_mhz"] = sm_clock
            memory_clock = _nvml_value(
                lambda h: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                handle,
            )
            if memory_clock is not None:
                gpu["memory_clock_mhz"] = memory_clock
            pcie_gen = _nvml_value(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle)
            if pcie_gen is not None:
                gpu["pcie_link_gen"] = pcie_gen
            pcie_width = _nvml_value(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle)
            if pcie_width is not None:
                gpu["pcie_link_width"] = pcie_width
            processes_by_class: dict[str, dict[str, int]] = {}
            for process_getter in (
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
                getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
            ):
                if not callable(process_getter):
                    continue
                try:
                    processes = process_getter(handle)
                except Exception:
                    continue
                for proc in processes or []:
                    pid = int(getattr(proc, "pid", 0) or 0)
                    command = ""
                    if pid > 0:
                        try:
                            import psutil  # type: ignore[import-not-found]

                            command = " ".join(psutil.Process(pid).cmdline())
                        except Exception:
                            command = ""
                    process_class = _classify_gpu_process(command)
                    bucket = processes_by_class.setdefault(
                        process_class,
                        {"process_count": 0, "memory_used_bytes": 0},
                    )
                    bucket["process_count"] += 1
                    bucket["memory_used_bytes"] += int(getattr(proc, "usedGpuMemory", 0) or 0)
            if processes_by_class:
                gpu["processes"] = [
                    {"process_class": process_class, **values}
                    for process_class, values in sorted(processes_by_class.items())
                ]
            gpus.append(
                gpu
            )
        return gpus, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


class NodeMetricsCollectorActor:
    def __init__(
        self,
        *,
        worker_alias: str,
        node_ip: str,
        ray_node_id: str | None = None,
        gpu_count: int | None = None,
        deployment_env: str | None = None,
        cluster_id: str | None = None,
        actor_name: str | None = None,
        is_head_node: bool = False,
    ) -> None:
        from mint_server.observability.logging_context import init_actor_observability

        init_actor_observability()
        self._spec = NodeMetricsDaemonSpec(
            worker_alias=str(worker_alias),
            node_ip=str(node_ip),
            ray_node_id=ray_node_id,
            gpu_count=gpu_count,
            deployment_env=deployment_env or os.getenv("MINT_DEPLOYMENT_ENV") or "",
            cluster_id=cluster_id or os.getenv("MINT_CLUSTER_ID") or "",
            actor_name=actor_name,
            is_head_node=bool(is_head_node),
        )
        self._started_at = time.time()
        self._sample_count = 0
        self._error_count = 0
        self._last_sample_at: float | None = None
        self._last_sample_duration_ms: float | None = None
        self._last_error: str | None = None
        self._last_sample: dict[str, Any] | None = None
        self._otel_enabled = False
        self._otel_error: str | None = None
        self._sample_interval_s = _sample_interval_s()
        self._shutdown_requested = False
        self._stop_event: asyncio.Event | None = None
        self._sampling_task: asyncio.Task | None = None
        self._sampling_loop: asyncio.AbstractEventLoop | None = None
        self._sampling_thread: threading.Thread | None = None
        self._init_otel_metrics()
        self._start_sampling_loop()

    async def _sampling_loop_main(self) -> None:
        self._stop_event = asyncio.Event()
        while not self._shutdown_requested:
            started_at = time.perf_counter()
            try:
                self.sample_once()
            except Exception as e:
                self._record_sample_exception(e, started_at=started_at)
            if self._shutdown_requested:
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._sample_interval_s)
            except TimeoutError:
                continue
            except asyncio.TimeoutError:
                continue
            break

    def _start_sampling_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            self._sampling_loop = loop

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                self._sampling_task = loop.create_task(self._sampling_loop_main())
                try:
                    loop.run_until_complete(self._sampling_task)
                finally:
                    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                    for task in pending:
                        task.cancel()
                    if pending:
                        with contextlib.suppress(Exception):
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()

            self._sampling_thread = threading.Thread(
                target=_run_loop,
                name=f"mint-node-metrics-{self._spec.worker_alias}",
                daemon=True,
            )
            self._sampling_thread.start()
            return

        self._sampling_loop = loop
        self._sampling_task = loop.create_task(self._sampling_loop_main())

    def _record_sample_exception(self, error: Exception, *, started_at: float) -> None:
        self._error_count += 1
        self._last_error = f"{type(error).__name__}: {error}"
        self._last_sample_at = time.time()
        self._last_sample_duration_ms = (time.perf_counter() - started_at) * 1000.0

    def _metric_attrs(self) -> dict[str, str]:
        attrs = {
            "worker_alias": self._spec.worker_alias,
            "deployment.env": self._spec.deployment_env or "",
            "mint.cluster_id": self._spec.cluster_id or "",
        }
        return {k: v for k, v in attrs.items() if v}

    def _init_otel_metrics(self) -> None:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation

            meter = metrics.get_meter("mint.node_metrics")

            def _observe_node_metric(field: str):
                sample = self.sample_cached()
                value = sample.get(field)
                if value is None:
                    return []
                return [Observation(float(value), self._metric_attrs())]

            def _observe_gpu_metric(field: str):
                sample = self.sample_cached()
                return [
                    Observation(
                        float(gpu.get(field) or 0.0),
                        {**self._metric_attrs(), "gpu_uuid": str(gpu.get("gpu_uuid") or "")},
                    )
                    for gpu in sample.get("gpus", [])
                    if gpu.get("gpu_uuid")
                ]

            def _observe_gpu_present(_options):
                sample = self.sample_cached()
                return [
                    Observation(
                        1.0,
                        {**self._metric_attrs(), "gpu_uuid": str(gpu.get("gpu_uuid") or "")},
                    )
                    for gpu in sample.get("gpus", [])
                    if gpu.get("gpu_uuid")
                ]

            def _observe_gpu_process_metric(field: str):
                sample = self.sample_cached()
                observations = []
                for gpu in sample.get("gpus", []):
                    gpu_uuid = str(gpu.get("gpu_uuid") or "")
                    if not gpu_uuid:
                        continue
                    for proc in gpu.get("processes", []):
                        process_class = str(proc.get("process_class") or "other")
                        value = proc.get(field)
                        if value is None:
                            continue
                        observations.append(
                            Observation(
                                float(value),
                                {
                                    **self._metric_attrs(),
                                    "gpu_uuid": gpu_uuid,
                                    "process_class": process_class,
                                },
                            )
                        )
                return observations

            def _collector_up(_options):
                return [Observation(0.0 if self._last_error or self._otel_error else 1.0, self._metric_attrs())]

            def _sample_age(_options):
                if self._last_sample_at is None:
                    return []
                return [Observation(max(0.0, time.time() - self._last_sample_at), self._metric_attrs())]

            def _sample_duration(_options):
                if self._last_sample_duration_ms is None:
                    return []
                return [Observation(float(self._last_sample_duration_ms), self._metric_attrs())]

            def _errors_total(_options):
                return [Observation(float(self._error_count), self._metric_attrs())]

            def _gauge(name: str, callback, *, unit: str | None = None, description: str = "") -> None:
                kwargs: dict[str, Any] = {"callbacks": [callback]}
                if unit:
                    kwargs["unit"] = unit
                if description:
                    kwargs["description"] = description
                meter.create_observable_gauge(name, **kwargs)

            meter.create_observable_gauge(
                "mint_node_load_1m",
                callbacks=[lambda options: _observe_node_metric("load_1m")],
                description="Node 1-minute load average observed by mint node metrics daemon",
            )
            _gauge("mint_node_load5", lambda options: _observe_node_metric("load_5m"))
            _gauge("mint_node_load15", lambda options: _observe_node_metric("load_15m"))
            _gauge("mint_node_cpu_utilization_ratio", lambda options: _observe_node_metric("cpu_utilization_ratio"))
            _gauge("mint_node_memory_used_bytes", lambda options: _observe_node_metric("memory_used_bytes"), unit="By")
            _gauge("mint_node_memory_total_bytes", lambda options: _observe_node_metric("memory_total_bytes"), unit="By")
            _gauge("mint_node_disk_used_bytes", lambda options: _observe_node_metric("disk_used_bytes"), unit="By")
            _gauge("mint_node_disk_total_bytes", lambda options: _observe_node_metric("disk_total_bytes"), unit="By")
            _gauge("mint_node_gpu_present", _observe_gpu_present)
            _gauge(
                "mint_node_gpu_utilization_percent",
                lambda options: _observe_gpu_metric("utilization_gpu_percent"),
                unit="%",
                description="GPU utilization by UUID observed by mint node metrics daemon",
            )
            _gauge("mint_node_gpu_memory_used_bytes", lambda options: _observe_gpu_metric("memory_used_bytes"), unit="By")
            _gauge("mint_node_gpu_memory_total_bytes", lambda options: _observe_gpu_metric("memory_total_bytes"), unit="By")
            _gauge("mint_node_gpu_power_draw_watts", lambda options: _observe_gpu_metric("power_draw_watts"), unit="W")
            _gauge("mint_node_gpu_power_limit_watts", lambda options: _observe_gpu_metric("power_limit_watts"), unit="W")
            _gauge("mint_node_gpu_temperature_celsius", lambda options: _observe_gpu_metric("temperature_celsius"), unit="Cel")
            _gauge("mint_node_gpu_sm_clock_mhz", lambda options: _observe_gpu_metric("sm_clock_mhz"), unit="MHz")
            _gauge("mint_node_gpu_memory_clock_mhz", lambda options: _observe_gpu_metric("memory_clock_mhz"), unit="MHz")
            _gauge("mint_node_gpu_pcie_link_gen", lambda options: _observe_gpu_metric("pcie_link_gen"))
            _gauge("mint_node_gpu_pcie_link_width", lambda options: _observe_gpu_metric("pcie_link_width"))
            _gauge("mint_node_gpu_processes", lambda options: _observe_gpu_process_metric("process_count"))
            _gauge(
                "mint_node_gpu_process_memory_used_bytes",
                lambda options: _observe_gpu_process_metric("memory_used_bytes"),
                unit="By",
            )
            _gauge("mint_node_metrics_collector_up", _collector_up)
            _gauge("mint_node_metrics_collector_sample_age_s", _sample_age, unit="s")
            _gauge("mint_node_metrics_collector_sample_duration_ms", _sample_duration, unit="ms")
            _gauge("mint_node_metrics_collector_errors_total", _errors_total)
            self._otel_enabled = True
        except Exception as e:
            self._otel_error = f"{type(e).__name__}: {e}"

    def sample_once(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        host = _sample_host_metrics()
        gpus, gpu_error = _sample_gpu_metrics()
        self._sample_count += 1
        self._last_sample_at = time.time()
        self._last_sample_duration_ms = (time.perf_counter() - started_at) * 1000.0
        host_error = host.get("host_error")
        if gpu_error or host_error:
            self._error_count += 1
            self._last_error = "; ".join(str(e) for e in (gpu_error, host_error) if e)
        else:
            self._last_error = None
        sample = {
            "worker_alias": self._spec.worker_alias,
            "node_ip": self._spec.node_ip,
            "ray_node_id": self._spec.ray_node_id,
            "deployment_env": self._spec.deployment_env,
            "cluster_id": self._spec.cluster_id,
            "hostname": host.get("hostname"),
            "load_1m": host.get("load_1m"),
            "load_5m": host.get("load_5m"),
            "load_15m": host.get("load_15m"),
            "cpu_utilization_ratio": host.get("cpu_utilization_ratio"),
            "memory_used_bytes": host.get("memory_used_bytes"),
            "memory_total_bytes": host.get("memory_total_bytes"),
            "disk_path": host.get("disk_path"),
            "disk_used_bytes": host.get("disk_used_bytes"),
            "disk_total_bytes": host.get("disk_total_bytes"),
            "gpu_count": len(gpus),
            "gpus": gpus,
            "gpu_error": gpu_error,
            "host_error": host_error,
            "sampled_at": self._last_sample_at,
            "sample_duration_ms": self._last_sample_duration_ms,
        }
        self._last_sample = dict(sample)
        return sample

    def sample_cached(self, *, max_age_s: float = 5.0) -> dict[str, Any]:
        if (
            self._last_sample is not None
            and self._last_sample_at is not None
            and time.time() - self._last_sample_at <= float(max_age_s)
        ):
            return dict(self._last_sample)
        return self.sample_once()

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "actor_name": self._spec.normalized_actor_name(),
            "worker_alias": self._spec.worker_alias,
            "node_ip": self._spec.node_ip,
            "ray_node_id": self._spec.ray_node_id,
            "deployment_env": self._spec.deployment_env,
            "cluster_id": self._spec.cluster_id,
            "is_head_node": self._spec.is_head_node,
            "expected_gpu_count": self._spec.gpu_count,
            "started_at": self._started_at,
            "sample_count": self._sample_count,
            "error_count": self._error_count,
            "last_sample_at": self._last_sample_at,
            "last_sample_duration_ms": self._last_sample_duration_ms,
            "last_error": self._last_error,
            "last_sample": self._last_sample,
            "otel_enabled": self._otel_enabled,
            "otel_error": self._otel_error,
            "sample_interval_s": self._sample_interval_s,
            "running": not self._shutdown_requested,
        }

    def shutdown(self) -> bool:
        self._shutdown_requested = True
        loop = self._sampling_loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(stop_event.set)
        thread = self._sampling_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._flush_otel_best_effort()
        return True

    def _flush_otel_best_effort(self) -> None:
        provider_getters = []
        try:
            from opentelemetry import metrics

            provider_getters.append(metrics.get_meter_provider)
        except Exception:
            pass
        try:
            from opentelemetry import trace

            provider_getters.append(trace.get_tracer_provider)
        except Exception:
            pass
        try:
            from opentelemetry import _logs

            provider_getters.append(_logs.get_logger_provider)
        except Exception:
            pass

        for get_provider in provider_getters:
            try:
                provider = get_provider()
                force_flush = getattr(provider, "force_flush", None)
                if not callable(force_flush):
                    continue
                try:
                    force_flush(timeout_millis=5000)
                except TypeError:
                    force_flush()
            except Exception:
                pass


def get_or_create_node_metrics_collector_actor(spec: NodeMetricsDaemonSpec) -> Any:
    import ray

    name = spec.normalized_actor_name()
    namespace = _ray_namespace()
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass
    remote_cls = ray.remote(num_cpus=0, max_concurrency=4, max_restarts=2)(NodeMetricsCollectorActor)
    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        "resources": {f"node:{spec.node_ip}": 0.001},
        "runtime_env": actor_runtime_env(
            pythonpath=PFS_CONTROL_PLANE_PYTHONPATH if spec.is_head_node else PFS_PYTHONPATH,
            tier=TIER_CPU if spec.is_head_node else None,
            extra={
                **otel_env_vars(),
                "OTEL_SERVICE_NAME": "mint-node-metrics",
                "MINT_WORKER_ALIAS": spec.worker_alias,
                "MINT_NODE_IP": spec.node_ip,
                **({"MINT_RAY_NODE_ID": spec.ray_node_id} if spec.ray_node_id else {}),
                **({"MINT_DEPLOYMENT_ENV": spec.deployment_env} if spec.deployment_env else {}),
                **({"MINT_CLUSTER_ID": spec.cluster_id} if spec.cluster_id else {}),
            },
            include_ray_attach_hints=False,
        ),
    }
    return remote_cls.options(**options).remote(
        worker_alias=spec.worker_alias,
        node_ip=spec.node_ip,
        ray_node_id=spec.ray_node_id,
        gpu_count=spec.gpu_count,
        deployment_env=spec.deployment_env,
        cluster_id=spec.cluster_id,
        actor_name=name,
        is_head_node=spec.is_head_node,
    )
