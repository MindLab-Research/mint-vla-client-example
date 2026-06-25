from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

CLAIMABLE_REPLICA_STATUSES = frozenset({"healthy", "ready"})


def otel_metric_attrs(*, ray_namespace: str) -> dict[str, str]:
    attrs = {
        "deployment.env": os.getenv("MINT_DEPLOYMENT_ENV", "").strip(),
        "mint.cluster_id": os.getenv("MINT_CLUSTER_ID", "").strip(),
        "ray_namespace": ray_namespace,
    }
    return {key: value for key, value in attrs.items() if value}


def metric_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scheduler_domain_base_model(domain_key: object) -> str | None:
    from mint_server.backend.actors.domain_keys import base_model_from_domain_key

    return base_model_from_domain_key(str(domain_key or ""))


class SchedulerMetrics:
    def __init__(
        self,
        *,
        ray_namespace: Callable[[], str],
        stats_snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._ray_namespace = ray_namespace
        self._stats_snapshot = stats_snapshot
        self.enabled = False
        self.error: str | None = None

    def _attrs(self, **extra: object) -> dict[str, str]:
        attrs = otel_metric_attrs(ray_namespace=self._ray_namespace())
        for key, value in extra.items():
            text = str(value if value is not None else "").strip()
            if text:
                attrs[key] = text
        return attrs

    def init_otel_metrics(self) -> None:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation

            meter = metrics.get_meter("mint.model_work_scheduler")

            def _gauge(name: str, callback, *, unit: str | None = None) -> None:
                kwargs: dict[str, Any] = {"callbacks": [callback]}
                if unit:
                    kwargs["unit"] = unit
                meter.create_observable_gauge(name, **kwargs)

            def _scalar(field: str):
                def _callback(_options):
                    value = metric_number(self._stats_snapshot().get(field))
                    if value is None:
                        return []
                    return [Observation(value, self._attrs())]

                return _callback

            _gauge("mint_model_work_scheduler_depth", _scalar("depth"))
            _gauge("mint_model_work_scheduler_backlog_depth", _scalar("backlog_depth"))

            def _counter(field: str):
                def _callback(_options):
                    counters = self._stats_snapshot().get("counters")
                    if not isinstance(counters, dict):
                        return []
                    value = metric_number(counters.get(field))
                    if value is None:
                        return []
                    return [Observation(value, self._attrs())]

                return _callback

            for key in (
                "appended",
                "assigned",
                "claimed",
                "completed",
                "failed",
                "requeued",
                "stale_dropped",
            ):
                _gauge(f"mint_model_work_scheduler_{key}_total", _counter(key))

            def _domain_backlog(_options):
                backlog_by_domain = self._stats_snapshot().get("backlog_depth_by_domain")
                if not isinstance(backlog_by_domain, dict):
                    return []
                observations = []
                for domain_key, depth in sorted(backlog_by_domain.items()):
                    value = metric_number(depth)
                    if value is None:
                        continue
                    observations.append(Observation(value, self._attrs(domain_key=domain_key)))
                return observations

            _gauge("mint_model_work_scheduler_domain_backlog_depth", _domain_backlog)

            def _replica_queue_depth(_options):
                replica_queues = self._stats_snapshot().get("replica_queues")
                if not isinstance(replica_queues, dict):
                    return []
                observations = []
                for queue_id, rec in sorted(replica_queues.items()):
                    if not isinstance(rec, dict):
                        continue
                    value = metric_number(rec.get("depth"))
                    if value is None:
                        continue
                    observations.append(
                        Observation(
                            value,
                            self._attrs(
                                domain_key=rec.get("domain_key") or "unknown",
                                replica_id=rec.get("replica_id") or "unknown",
                                queue_id=queue_id,
                                status=rec.get("status") or "unknown",
                            ),
                        )
                    )
                return observations

            _gauge("mint_model_work_scheduler_replica_queue_depth", _replica_queue_depth)

            def _leases(_options):
                leases = self._stats_snapshot().get("leases")
                if not isinstance(leases, list):
                    return []
                return [Observation(float(len(leases)), self._attrs())]

            _gauge("mint_model_work_scheduler_leases", _leases)

            def _sample_model_load(metric: str):
                def _callback(_options):
                    stats = self._stats_snapshot()
                    load: dict[str, dict[str, float]] = {}
                    replica_queues = stats.get("replica_queues")
                    if isinstance(replica_queues, dict):
                        for rec in replica_queues.values():
                            if not isinstance(rec, dict):
                                continue
                            base_model = scheduler_domain_base_model(rec.get("domain_key"))
                            if not base_model:
                                continue
                            bucket = load.setdefault(
                                base_model,
                                {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                            )
                            bucket["pending_requests"] += float(metric_number(rec.get("depth")) or 0.0)
                            if str(rec.get("status") or "").lower() in CLAIMABLE_REPLICA_STATUSES:
                                bucket["capacity_workers"] += 1.0
                    leases = stats.get("leases")
                    if isinstance(leases, list):
                        for lease in leases:
                            if not isinstance(lease, dict):
                                continue
                            item = lease.get("item") if isinstance(lease.get("item"), dict) else {}
                            assert item is not None
                            base_model = scheduler_domain_base_model(item.get("domain_key") or lease.get("domain_key"))
                            if not base_model:
                                continue
                            bucket = load.setdefault(
                                base_model,
                                {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                            )
                            bucket["inflight_workers"] += 1.0
                    observations = []
                    for base_model, bucket in sorted(load.items()):
                        capacity = float(bucket.get("capacity_workers", 0.0))
                        values = {
                            "pending_requests": float(bucket.get("pending_requests", 0.0)),
                            "inflight_workers": float(bucket.get("inflight_workers", 0.0)),
                            "capacity_workers": capacity,
                            "load_pct": 0.0
                            if capacity <= 0.0
                            else 100.0 * float(bucket.get("inflight_workers", 0.0)) / capacity,
                        }
                        observations.append(
                            Observation(
                                values[metric],
                                self._attrs(base_model=base_model, workload="sample"),
                            )
                        )
                    return observations

                return _callback

            _gauge("mint_model_load_pct", _sample_model_load("load_pct"))
            _gauge("mint_model_pending_requests", _sample_model_load("pending_requests"))
            _gauge("mint_model_inflight_workers", _sample_model_load("inflight_workers"))
            _gauge("mint_model_capacity_workers", _sample_model_load("capacity_workers"))

            def _sampling_inflight_by_domain(_options):
                sampling = self._stats_snapshot().get("sampling_inflight")
                by_domain = sampling.get("by_domain") if isinstance(sampling, dict) else None
                if not isinstance(by_domain, dict):
                    return []
                observations = []
                for domain_key, count in sorted(by_domain.items()):
                    value = metric_number(count)
                    if value is None:
                        continue
                    observations.append(Observation(value, self._attrs(domain_key=domain_key)))
                return observations

            _gauge("mint_sampling_inflight_by_domain", _sampling_inflight_by_domain)

            def _sampling_inflight_principal_domain_max(_options):
                sampling = self._stats_snapshot().get("sampling_inflight")
                max_by_domain = sampling.get("principal_domain_max_by_domain") if isinstance(sampling, dict) else None
                if not isinstance(max_by_domain, dict):
                    return []
                observations = []
                for domain_key, count in sorted(max_by_domain.items()):
                    value = metric_number(count)
                    if value is None:
                        continue
                    observations.append(Observation(value, self._attrs(domain_key=domain_key)))
                return observations

            _gauge(
                "mint_sampling_inflight_principal_domain_max",
                _sampling_inflight_principal_domain_max,
            )

            def _sampling_admission_counter(decision: str):
                def _callback(_options):
                    stats = self._stats_snapshot().get("sampling_admission_counters")
                    records = stats.get(decision) if isinstance(stats, dict) else None
                    if not isinstance(records, list):
                        return []
                    observations = []
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        value = metric_number(record.get("count"))
                        if value is None:
                            continue
                        observations.append(
                            Observation(
                                value,
                                self._attrs(
                                    domain_key=record.get("domain_key"),
                                    reason=record.get("reason"),
                                ),
                            )
                        )
                    return observations

                return _callback

            _gauge(
                "mint_sampling_admission_would_reject_total",
                _sampling_admission_counter("would_reject"),
            )
            _gauge(
                "mint_sampling_admission_reject_total",
                _sampling_admission_counter("reject"),
            )

            self.enabled = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
