#!/usr/bin/env python3
"""Start or stop detached Ray actors that export NVIDIA GPU metrics through OTLP.

Run this from the Mint driver host, for example:

    ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/code/tinker-server-auth && python3 scripts/tools/start_nvml_otel_probe.py start'

The probe is read-only. It requests CPU only and never requests Ray GPU resources.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray


PROBE_PREFIX = "mint_nvml_probe"
METER_NAME = "mint.nvml_probe"
DEFAULT_INTERVAL_S = 5.0


@dataclass(frozen=True)
class GpuRow:
    hostname: str
    node_ip: str
    gpu_index: str
    gpu_uuid: str
    gpu_name: str
    values: dict[str, float]


def _node_ip() -> str:
    try:
        import ray.util  # type: ignore[attr-defined]

        return str(ray.util.get_node_ip_address())
    except Exception:
        return "unknown"


def _float_or_none(value: str) -> float | None:
    text = str(value).strip()
    if not text or text == "[Not Supported]" or text.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _process_class(pid: str, process_name: str) -> str:
    name = os.path.basename(str(process_name)).lower()
    cmdline = ""
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", "replace"
        ).lower()
    except Exception:
        pass
    text = f"{name} {cmdline}"
    if "vllm" in text:
        return "vllm"
    if "megatron" in text:
        return "megatron"
    if "ray::" in text or ("ray" in text and "python" in name):
        return "ray_python"
    if "python" in name:
        return "python"
    return "other"


def _run_compute_apps() -> dict[tuple[str, str], dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_uuid, pid, process_name, used_memory_mib = parts
        klass = _process_class(pid, process_name)
        key = (str(gpu_uuid), klass)
        row = agg.setdefault(key, {"count": 0.0, "memory_used_bytes": 0.0})
        row["count"] += 1.0
        mem = _float_or_none(used_memory_mib)
        if mem is not None:
            row["memory_used_bytes"] += float(mem) * 1024.0 * 1024.0
    return agg


def _run_nvidia_smi() -> list[GpuRow]:
    query = (
        "index,uuid,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,power.limit,"
        "temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,"
        "pcie.link.width.current"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, timeout=10)
    hostname = socket.gethostname()
    node_ip = _node_ip()
    rows: list[GpuRow] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 14:
            continue
        (
            gpu_index,
            gpu_uuid,
            gpu_name,
            gpu_util_pct,
            mem_util_pct,
            mem_used_mib,
            mem_total_mib,
            power_draw_w,
            power_limit_w,
            temp_c,
            sm_clock_mhz,
            mem_clock_mhz,
            pcie_gen,
            pcie_width,
        ) = parts
        gpu_util = _float_or_none(gpu_util_pct)
        mem_util = _float_or_none(mem_util_pct)
        mem_used = _float_or_none(mem_used_mib)
        mem_total = _float_or_none(mem_total_mib)
        raw = {
            "gpu_utilization_ratio": None if gpu_util is None else gpu_util / 100.0,
            "memory_utilization_ratio": None if mem_util is None else mem_util / 100.0,
            "memory_used_bytes": None if mem_used is None else mem_used * 1024.0 * 1024.0,
            "memory_total_bytes": None if mem_total is None else mem_total * 1024.0 * 1024.0,
            "power_draw_watts": _float_or_none(power_draw_w),
            "power_limit_watts": _float_or_none(power_limit_w),
            "temperature_celsius": _float_or_none(temp_c),
            "sm_clock_mhz": _float_or_none(sm_clock_mhz),
            "memory_clock_mhz": _float_or_none(mem_clock_mhz),
            "pcie_link_gen": _float_or_none(pcie_gen),
            "pcie_link_width": _float_or_none(pcie_width),
        }
        values = {k: float(v) for k, v in raw.items() if v is not None}
        rows.append(
            GpuRow(
                hostname=hostname,
                node_ip=node_ip,
                gpu_index=str(gpu_index),
                gpu_uuid=str(gpu_uuid),
                gpu_name=str(gpu_name),
                values=values,
            )
        )
    return rows


def _parse_headers(raw: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not raw:
        return out
    for pair in str(raw).split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def _configure_otel() -> Any:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        raise RuntimeError("OTEL_EXPORTER_OTLP_ENDPOINT is not set")

    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    attrs: dict[str, str] = {"service.name": (os.getenv("OTEL_SERVICE_NAME") or "mint").strip() or "mint"}
    if os.getenv("MINT_NVML_PROBE_RESOURCE_ATTRS_JSON"):
        attrs.update(json.loads(os.environ["MINT_NVML_PROBE_RESOURCE_ATTRS_JSON"]))
    attrs.setdefault("mint.component", "nvml_probe")

    headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))
    app_key = (os.getenv("MINT_APMPLUS_APP_KEY") or os.getenv("OTEL_APMPLUS_APP_KEY") or "").strip()
    if app_key and "x-byteapm-appkey" not in headers:
        headers["x-byteapm-appkey"] = app_key

    interval_ms = max(1000, int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000")))
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers=headers or None,
        insecure=_parse_bool(os.getenv("OTEL_EXPORTER_OTLP_INSECURE"), True),
    )
    provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(exporter=exporter, export_interval_millis=interval_ms)],
        resource=Resource(attributes=attrs),
    )
    metrics.set_meter_provider(provider)
    return metrics.get_meter(METER_NAME), provider


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@ray.remote(num_cpus=0.05, num_gpus=0)
class NvmlOtelProbe:
    def __init__(self, *, interval_s: float = DEFAULT_INTERVAL_S, runtime_env_vars: dict[str, str] | None = None) -> None:
        if runtime_env_vars:
            os.environ.update({str(k): str(v) for k, v in runtime_env_vars.items()})
        self._interval_s = max(1.0, float(interval_s))
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error = ""
        self._last_emit_ts = 0.0
        self._samples = 0
        self._meter, self._provider = _configure_otel()
        self._gpu_present = self._meter.create_gauge(
            "mint_nvml_gpu_present",
            unit="1",
            description="NVIDIA GPU device presence sampled by Mint NVML probe",
        )
        self._process_count = self._meter.create_gauge(
            "mint_nvml_gpu_processes",
            unit="{process}",
            description="NVIDIA GPU compute process count by bounded process class sampled by Mint NVML probe",
        )
        self._process_memory = self._meter.create_gauge(
            "mint_nvml_gpu_process_memory_used_bytes",
            unit="By",
            description="NVIDIA GPU compute process memory by bounded process class sampled by Mint NVML probe",
        )
        self._instruments = {
            "gpu_utilization_ratio": self._meter.create_gauge(
                "mint_nvml_gpu_utilization_ratio",
                unit="1",
                description="NVIDIA GPU SM utilization ratio sampled by Mint NVML probe",
            ),
            "memory_utilization_ratio": self._meter.create_gauge(
                "mint_nvml_memory_utilization_ratio",
                unit="1",
                description="NVIDIA GPU memory controller utilization ratio sampled by Mint NVML probe",
            ),
            "memory_used_bytes": self._meter.create_gauge(
                "mint_nvml_memory_used_bytes",
                unit="By",
                description="NVIDIA framebuffer memory used sampled by Mint NVML probe",
            ),
            "memory_total_bytes": self._meter.create_gauge(
                "mint_nvml_memory_total_bytes",
                unit="By",
                description="NVIDIA framebuffer memory total sampled by Mint NVML probe",
            ),
            "power_draw_watts": self._meter.create_gauge(
                "mint_nvml_power_draw_watts",
                unit="W",
                description="NVIDIA GPU power draw sampled by Mint NVML probe",
            ),
            "power_limit_watts": self._meter.create_gauge(
                "mint_nvml_power_limit_watts",
                unit="W",
                description="NVIDIA GPU power limit sampled by Mint NVML probe",
            ),
            "temperature_celsius": self._meter.create_gauge(
                "mint_nvml_temperature_celsius",
                unit="Cel",
                description="NVIDIA GPU temperature sampled by Mint NVML probe",
            ),
            "sm_clock_mhz": self._meter.create_gauge(
                "mint_nvml_sm_clock_mhz",
                unit="MHz",
                description="NVIDIA GPU SM clock sampled by Mint NVML probe",
            ),
            "memory_clock_mhz": self._meter.create_gauge(
                "mint_nvml_memory_clock_mhz",
                unit="MHz",
                description="NVIDIA GPU memory clock sampled by Mint NVML probe",
            ),
            "pcie_link_gen": self._meter.create_gauge(
                "mint_nvml_pcie_link_gen",
                unit="1",
                description="NVIDIA GPU current PCIe link generation sampled by Mint NVML probe",
            ),
            "pcie_link_width": self._meter.create_gauge(
                "mint_nvml_pcie_link_width",
                unit="1",
                description="NVIDIA GPU current PCIe link width sampled by Mint NVML probe",
            ),
        }
        self._probe_up = self._meter.create_gauge(
            "mint_nvml_probe_up",
            unit="1",
            description="Mint NVML probe health, 1 for last scrape ok and 0 for last scrape failed",
        )

    def status(self) -> dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "node_ip": _node_ip(),
            "running": self._running,
            "last_error": self._last_error,
            "last_emit_ts": self._last_emit_ts,
            "samples": self._samples,
            "interval_s": self._interval_s,
        }

    def start_loop(self) -> dict[str, Any]:
        if self._thread is not None and self._thread.is_alive():
            return self.status()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="mint-nvml-probe", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self._interval_s + 1.0))
        try:
            self._provider.force_flush(timeout_millis=5000)
            self._provider.shutdown(timeout_millis=5000)
        except Exception:
            pass
        return self.status()

    def sample_once(self) -> dict[str, Any]:
        return self._sample_once()

    def _run_loop(self) -> None:
        while self._running:
            started = time.monotonic()
            self._sample_once()
            sleep_s = max(0.0, self._interval_s - (time.monotonic() - started))
            time.sleep(sleep_s)

    def _sample_once(self) -> dict[str, Any]:
        base_attrs = {"hostname": socket.gethostname(), "node_ip": _node_ip()}
        try:
            rows = _run_nvidia_smi()
            process_agg = _run_compute_apps()
            for row in rows:
                attrs = {
                    "hostname": row.hostname,
                    "node_ip": row.node_ip,
                    "gpu_index": row.gpu_index,
                    "gpu_uuid": row.gpu_uuid,
                    "gpu_name": row.gpu_name,
                }
                self._gpu_present.set(1, attributes=attrs)
                for name, value in row.values.items():
                    self._instruments[name].set(value, attributes=attrs)
                for (process_gpu_uuid, process_class), rec in process_agg.items():
                    if process_gpu_uuid != row.gpu_uuid:
                        continue
                    proc_attrs = {**attrs, "process_class": process_class}
                    self._process_count.set(float(rec.get("count", 0.0)), attributes=proc_attrs)
                    self._process_memory.set(float(rec.get("memory_used_bytes", 0.0)), attributes=proc_attrs)
            self._probe_up.set(1, attributes=base_attrs)
            self._last_error = ""
            self._last_emit_ts = time.time()
            self._samples += 1
            return {"ok": True, "gpu_count": len(rows), **self.status()}
        except Exception as exc:
            self._last_error = repr(exc)
            self._probe_up.set(0, attributes=base_attrs)
            self._last_emit_ts = time.time()
            self._samples += 1
            return {"ok": False, "error": self._last_error, **self.status()}


def _maybe_reexec_runtime_python() -> None:
    env_root = (os.getenv("PFS_RUNTIME_ENV_ROOT") or "").strip()
    if not env_root:
        return
    target = Path(env_root) / "host-venv" / "bin" / "python"
    if not target.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        wanted = target.resolve()
    except Exception:
        return
    if current == wanted:
        return
    os.execv(str(wanted), [str(wanted), *sys.argv])


def _connect_ray(address: str, namespace: str) -> None:
    ray.init(address=address, namespace=namespace, ignore_reinit_error=True, log_to_driver=False)


def _probe_runtime_env() -> dict[str, object] | None:
    try:
        from tinker_server.config import PFS_PYTHONPATH, actor_runtime_env, otel_env_vars

        env_vars = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars()).get("env_vars", {})
    except Exception:
        env_vars = {k: v for k, v in os.environ.items() if k.startswith("OTEL_")}
    for key in (
        "MINT_APMPLUS_APP_KEY",
        "OTEL_APMPLUS_APP_KEY",
        "RAY_ADDRESS",
        "PFS_RUNTIME_ENV_ROOT",
        "PFS_TINKER_PATH",
        "PFS_HF_MODULES_PATH",
    ):
        value = os.getenv(key)
        if value:
            env_vars.setdefault(key, value)
    return {"env_vars": env_vars}


def _gpu_nodes() -> list[dict[str, Any]]:
    return [n for n in ray.nodes() if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0.0)) > 0]


def _actor_name(node_ip: str) -> str:
    safe = node_ip.replace(".", "_").replace(":", "_")
    return f"{PROBE_PREFIX}_{safe}"


def start(args: argparse.Namespace) -> int:
    _connect_ray(args.ray_address, args.namespace)
    nodes = _gpu_nodes()
    actors: dict[str, Any] = {}
    for node in nodes:
        node_ip = str(node["NodeManagerAddress"])
        name = _actor_name(node_ip)
        actor = NvmlOtelProbe.options(
            name=name,
            namespace=args.namespace,
            lifetime="detached",
            get_if_exists=True,
            resources={f"node:{node_ip}": 0.001},
        ).remote(interval_s=args.interval_s, runtime_env_vars=_probe_runtime_env().get("env_vars", {}))
        actors[name] = actor

    if not args.no_start_loop:
        ray.get([actor.start_loop.remote() for actor in actors.values()], timeout=args.timeout_s)

    statuses = ray.get([actor.status.remote() for actor in actors.values()], timeout=args.timeout_s)
    print(json.dumps({"started": sorted(actors), "statuses": statuses}, indent=2, sort_keys=True))
    return 0


def sample(args: argparse.Namespace) -> int:
    _connect_ray(args.ray_address, args.namespace)
    nodes = _gpu_nodes()
    refs = []
    missing = []
    for node in nodes:
        name = _actor_name(str(node["NodeManagerAddress"]))
        try:
            actor = ray.get_actor(name, namespace=args.namespace)
        except ValueError:
            missing.append(name)
            continue
        refs.append(actor.sample_once.remote())
    results = ray.get(refs, timeout=args.timeout_s) if refs else []
    print(json.dumps({"results": results, "missing": missing}, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    _connect_ray(args.ray_address, args.namespace)
    out = []
    for node in _gpu_nodes():
        name = _actor_name(str(node["NodeManagerAddress"]))
        try:
            actor = ray.get_actor(name, namespace=args.namespace)
            out.append(ray.get(actor.status.remote(), timeout=args.timeout_s))
        except ValueError:
            out.append({"name": name, "missing": True})
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def stop(args: argparse.Namespace) -> int:
    _connect_ray(args.ray_address, args.namespace)
    stopped = []
    for node in _gpu_nodes():
        name = _actor_name(str(node["NodeManagerAddress"]))
        try:
            actor = ray.get_actor(name, namespace=args.namespace)
        except ValueError:
            continue
        try:
            ray.get(actor.stop.remote(), timeout=args.timeout_s)
        finally:
            ray.kill(actor, no_restart=True)
        stopped.append(name)
    print(json.dumps({"stopped": stopped}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    _maybe_reexec_runtime_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "sample", "status", "stop"))
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--namespace", default=os.getenv("MINT_RAY_NAMESPACE") or os.getenv("TINKER_RAY_NAMESPACE") or "tinker")
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--no-start-loop", action="store_true", help="Create actors but do not start run_forever loops")
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
