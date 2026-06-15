from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .model_placement_topology import (
    EnginePlacementTopology,
    ParallelTopology,
    PlacementBundle,
)

CLUSTER_GPU_KEY = "__cluster__"


class PlacementReservationStatus(StrEnum):
    RESERVED = "reserved"
    BLOCKED = "blocked"
    RELEASED = "released"
    NOT_FOUND = "not_found"


class PlacementGroupCreateStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class PlacementBlockReason(StrEnum):
    INSUFFICIENT_GPU = "insufficient_gpu"
    UNKNOWN_RESERVATION = "unknown_reservation"
    PG_PENDING_TIMEOUT = "pg_pending_timeout"
    PG_READY_FAILED = "pg_ready_failed"
    BACKOFF_ACTIVE = "backoff_active"


GpuByNode = tuple[tuple[str, int], ...]
BUNDLE = tuple[tuple[str, int | float], ...]
ObservedGpuSource = Callable[[], Mapping[str, int] | Awaitable[Mapping[str, int]]]
PlacementGroupTableSource = Callable[[], Mapping[str, Any] | Iterable[Any] | Awaitable[Mapping[str, Any] | Iterable[Any]]]
PlacementGroupFactory = Callable[..., Any | Awaitable[Any]]
PlacementGroupRemover = Callable[[Any], Any | Awaitable[Any]]
TopologyResolver = Callable[[dict[tuple[str, str], Any]], Any | Awaitable[dict[str, Any]]]
PlacementReconciler = Callable[..., Any | Awaitable[dict[str, Any]]]
SpecResolver = Callable[[Any, list[str] | None, tuple[tuple[str, str, int], ...] | None], Any]


@dataclass(frozen=True)
class PlacementReservationRequest:
    replica_key: str
    required_gpus_by_node: GpuByNode
    placement_group_name: str | None = None
    namespace: str | None = None

    @classmethod
    def from_mapping(
        cls,
        *,
        replica_key: str,
        required_gpus_by_node: Mapping[str, int],
        placement_group_name: str | None = None,
        namespace: str | None = None,
    ) -> PlacementReservationRequest:
        return cls(
            replica_key=str(replica_key),
            required_gpus_by_node=_normalize_gpu_by_node(required_gpus_by_node),
            placement_group_name=placement_group_name,
            namespace=namespace,
        )


@dataclass(frozen=True)
class PlacementReservationToken:
    token_id: str
    replica_key: str
    required_gpus_by_node: GpuByNode
    placement_group_name: str | None = None
    namespace: str | None = None


@dataclass(frozen=True)
class PlacementGroupCreateRequest:
    replica_key: str
    placement_group_name: str
    required_gpus_by_node: GpuByNode
    bundles: tuple[BUNDLE, ...]
    namespace: str | None = None
    strategy: str = "PACK"
    lifetime: str = "detached"
    ready_timeout_s: float = 60.0

    @classmethod
    def from_mapping(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        required_gpus_by_node: Mapping[str, int],
        bundles: Iterable[Mapping[str, int | float]],
        namespace: str | None = None,
        strategy: str = "PACK",
        lifetime: str = "detached",
        ready_timeout_s: float = 60.0,
    ) -> PlacementGroupCreateRequest:
        return cls(
            replica_key=str(replica_key),
            placement_group_name=str(placement_group_name),
            required_gpus_by_node=_normalize_gpu_by_node(required_gpus_by_node),
            bundles=_normalize_bundles(bundles),
            namespace=namespace,
            strategy=str(strategy),
            lifetime=str(lifetime),
            ready_timeout_s=float(ready_timeout_s),
        )

    @property
    def bundle_dicts(self) -> list[dict[str, int | float]]:
        # Ray's placement_group API requires a concrete list, while this module
        # keeps immutable tuples internally for typed snapshots and equality.
        return [dict(bundle) for bundle in self.bundles]


@dataclass(frozen=True)
class PlacementGroupBundleRequest:
    replica_key: str
    placement_group_name: str
    required_gpus_by_node: GpuByNode
    bundles: tuple[BUNDLE, ...]
    controller_bundle_index: int | None = None
    namespace: str | None = None

    @classmethod
    def for_dense(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        node_ip: str | None = None,
        namespace: str | None = None,
    ) -> PlacementGroupBundleRequest:
        topology = EnginePlacementTopology(
            gpu_count=1,
            placement_slices=() if node_ip is None else ((replica_key, node_ip, 1),),
        )
        return cls.from_topology(
            replica_key=replica_key,
            placement_group_name=placement_group_name,
            topology=topology,
            namespace=namespace,
        )

    @classmethod
    def for_vllm(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        worker_gpus: int,
        node_ips: Iterable[str] = (),
        namespace: str | None = None,
    ) -> PlacementGroupBundleRequest:
        placement_slices: list[tuple[str, str, int]] = []
        node_counts: dict[str, int] = {}
        for raw_node_ip in node_ips:
            node_ip = str(raw_node_ip).strip()
            if not node_ip:
                continue
            node_counts[node_ip] = node_counts.get(node_ip, 0) + 1
        for node_ip, gpu_count in node_counts.items():
            placement_slices.append((replica_key, node_ip, gpu_count))
        topology = EnginePlacementTopology(
            gpu_count=int(worker_gpus),
            placement_slices=tuple(placement_slices),
        )
        layout = topology.with_controller_bundle(cpu=1, cpu_per_gpu=1)
        return cls.from_layout(
            replica_key=replica_key,
            placement_group_name=placement_group_name,
            bundles=layout.bundles,
            controller_bundle_index=layout.controller_bundle_index,
            namespace=namespace,
        )

    @classmethod
    def for_distributed_training(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        parallel: ParallelTopology,
        placement_slices: Iterable[tuple[str, str, int]] = (),
        namespace: str | None = None,
    ) -> PlacementGroupBundleRequest:
        topology = EnginePlacementTopology(
            parallel=parallel,
            placement_slices=tuple(placement_slices),
        )
        return cls.from_topology(
            replica_key=replica_key,
            placement_group_name=placement_group_name,
            topology=topology,
            namespace=namespace,
        )

    @classmethod
    def from_topology(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        topology: EnginePlacementTopology,
        namespace: str | None = None,
    ) -> PlacementGroupBundleRequest:
        return cls.from_layout(
            replica_key=replica_key,
            placement_group_name=placement_group_name,
            bundles=topology.gpu_bundles(cpu_per_gpu=1),
            namespace=namespace,
        )

    @classmethod
    def from_layout(
        cls,
        *,
        replica_key: str,
        placement_group_name: str,
        bundles: Iterable[PlacementBundle],
        controller_bundle_index: int | None = None,
        namespace: str | None = None,
    ) -> PlacementGroupBundleRequest:
        normalized_bundles = _normalize_bundles(bundles)
        return cls(
            replica_key=str(replica_key),
            placement_group_name=str(placement_group_name),
            required_gpus_by_node=_required_gpus_by_node_from_bundles(normalized_bundles),
            bundles=normalized_bundles,
            controller_bundle_index=controller_bundle_index,
            namespace=namespace,
        )

    def to_create_request(self, *, ready_timeout_s: float = 60.0) -> PlacementGroupCreateRequest:
        return PlacementGroupCreateRequest(
            replica_key=self.replica_key,
            placement_group_name=self.placement_group_name,
            required_gpus_by_node=self.required_gpus_by_node,
            bundles=self.bundles,
            namespace=self.namespace,
            ready_timeout_s=float(ready_timeout_s),
        )


@dataclass(frozen=True)
class PlacementReservationResult:
    status: PlacementReservationStatus
    token: PlacementReservationToken | None = None
    reason: PlacementBlockReason | None = None
    message: str | None = None
    available_gpus_by_node: GpuByNode = ()
    reserved_gpus_by_node: GpuByNode = ()

    @property
    def ok(self) -> bool:
        return self.status in {PlacementReservationStatus.RESERVED, PlacementReservationStatus.RELEASED}


@dataclass(frozen=True)
class PlacementGroupCreateResult:
    status: PlacementGroupCreateStatus
    placement_group_name: str
    placement_group: Any | None = None
    reason: PlacementBlockReason | None = None
    message: str | None = None
    retry_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is PlacementGroupCreateStatus.READY


@dataclass(frozen=True)
class PlacementReservationSnapshot:
    in_flight_gpus_by_node: GpuByNode
    rebuilt_gpus_by_node: GpuByNode
    available_gpus_by_node: GpuByNode
    active_reservation_count: int
    blocked_replicas: tuple[tuple[str, PlacementBlockReason, float], ...] = ()


@dataclass(frozen=True)
class PlacementReconcileRequest:
    desired: Mapping[tuple[str, str], Any]
    protected_actor_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PlacementReconcileResult:
    ok: bool
    node_pins: tuple[tuple[str, tuple[str, ...]], ...] = ()
    blocked: tuple[tuple[str, str], ...] = ()
    placement_group_requests: tuple[PlacementGroupCreateRequest, ...] = ()
    protected_actor_names: frozenset[str] = frozenset()
    error: str | None = None

    @property
    def node_pins_by_label(self) -> dict[str, list[str]]:
        return {label: list(pins) for label, pins in self.node_pins}

    @property
    def blocked_by_label(self) -> dict[str, str]:
        return dict(self.blocked)

    def to_legacy_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": bool(self.ok),
            "blocked": self.blocked_by_label,
            "node_pins": self.node_pins_by_label,
            "placement_group_requests": list(self.placement_group_requests),
            "protected_actor_names": sorted(self.protected_actor_names),
        }
        if self.error is not None:
            out["error"] = self.error
        return out


class ClusterPlacementController:
    """In-process placement reservation authority for model actor placement."""

    def __init__(
        self,
        *,
        observed_free_gpus_by_node: ObservedGpuSource,
        placement_group_table: PlacementGroupTableSource | None = None,
        placement_group_factory: PlacementGroupFactory | None = None,
        placement_group_remover: PlacementGroupRemover | None = None,
        topology_resolver: TopologyResolver | None = None,
        placement_reconciler: PlacementReconciler | None = None,
        spec_resolver: SpecResolver | None = None,
        namespace: str | None = None,
        monotonic: Callable[[], float] | None = None,
        initial_backoff_s: float = 5.0,
        max_backoff_s: float = 60.0,
    ) -> None:
        self._observed_free_gpus_by_node = observed_free_gpus_by_node
        self._placement_group_table = placement_group_table
        self._placement_group_factory = placement_group_factory
        self._placement_group_remover = placement_group_remover
        self._topology_resolver = topology_resolver
        self._placement_reconciler = placement_reconciler
        self._spec_resolver = spec_resolver
        self._namespace = namespace
        self._monotonic = monotonic or time.monotonic
        self._initial_backoff_s = max(0.0, float(initial_backoff_s))
        self._max_backoff_s = max(self._initial_backoff_s, float(max_backoff_s))
        self._lock = asyncio.Lock()
        self._reservations: dict[str, PlacementReservationToken] = {}
        self._rebuilt_gpus_by_node: dict[str, int] = {}
        self._blocked: dict[str, tuple[PlacementBlockReason, float, float]] = {}

    async def reconcile(self, request: PlacementReconcileRequest) -> PlacementReconcileResult:
        desired = dict(request.desired)
        topology_blocked: dict[str, str] = {}
        topology_node_pins: dict[str, list[str]] = {}
        topology_placement_slices: dict[str, tuple[tuple[str, str, int], ...]] = {}
        if self._topology_resolver is not None:
            raw_topology = self._topology_resolver(dict(desired))
            topology_out = await raw_topology if inspect.isawaitable(raw_topology) else raw_topology
            if isinstance(topology_out, dict):
                topology_blocked = _string_dict(topology_out.get("blocked"))
                topology_node_pins = _list_dict(topology_out.get("node_pins"))
                topology_placement_slices = _placement_slices_dict(topology_out.get("placement_slices"))
                desired = self._apply_resolved_topology(
                    desired,
                    node_pins=topology_node_pins,
                    placement_slices=topology_placement_slices,
                )

        placement_blocked: dict[str, str] = {}
        placement_node_pins: dict[str, list[str]] = {}
        if self._placement_reconciler is not None:
            try:
                raw_placement = self._placement_reconciler(
                    dict(desired),
                    protected_actor_names=set(request.protected_actor_names),
                )
            except TypeError as exc:
                if "protected_actor_names" not in str(exc):
                    raise
                raw_placement = self._placement_reconciler(dict(desired))
            placement_out = await raw_placement if inspect.isawaitable(raw_placement) else raw_placement
            if isinstance(placement_out, dict):
                placement_blocked = _string_dict(placement_out.get("blocked"))
                placement_node_pins = _list_dict(placement_out.get("node_pins"))

        node_pins: dict[str, tuple[str, ...]] = {}
        blocked: dict[str, str] = dict(topology_blocked)
        blocked.update(placement_blocked)
        create_requests: list[PlacementGroupCreateRequest] = []
        for key, spec in sorted(desired.items(), key=lambda item: _label(item[0])):
            label = _label(key)
            try:
                bundle_request = placement_group_bundle_request_for_spec(spec)
            except ValueError:
                continue
            except Exception as exc:
                blocked[label] = f"{type(exc).__name__}: {exc}"
                continue
            pins = tuple(placement_node_pins.get(label) or topology_node_pins.get(label) or list(_spec_node_pins(spec)))
            if pins:
                node_pins[label] = pins
            if label not in blocked:
                create_requests.append(bundle_request.to_create_request())
        return PlacementReconcileResult(
            ok=not blocked,
            node_pins=tuple(sorted(node_pins.items())),
            blocked=tuple(sorted(blocked.items())),
            placement_group_requests=tuple(create_requests),
            protected_actor_names=frozenset(str(name) for name in request.protected_actor_names),
        )

    def _apply_resolved_topology(
        self,
        desired: dict[tuple[str, str], Any],
        *,
        node_pins: Mapping[str, list[str]],
        placement_slices: Mapping[str, tuple[tuple[str, str, int], ...]],
    ) -> dict[tuple[str, str], Any]:
        if self._spec_resolver is None:
            return desired
        resolved: dict[tuple[str, str], Any] = {}
        for key, spec in desired.items():
            label = _label(key)
            if label in node_pins or label in placement_slices:
                resolved[key] = self._spec_resolver(
                    spec,
                    node_pins.get(label),
                    placement_slices.get(label),
                )
            else:
                resolved[key] = spec
        return resolved

    async def create_pg(self, request: PlacementGroupCreateRequest) -> PlacementGroupCreateResult:
        blocked = self._blocked.get(request.replica_key)
        now = float(self._monotonic())
        if blocked is not None and now < blocked[1]:
            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.BACKOFF_ACTIVE,
                retry_at=blocked[1],
                message=f"placement backoff active until {blocked[1]:.3f}",
            )
        reservation = await self.reserve(
            PlacementReservationRequest(
                replica_key=request.replica_key,
                required_gpus_by_node=request.required_gpus_by_node,
                placement_group_name=request.placement_group_name,
                namespace=request.namespace or self._namespace,
            )
        )
        if reservation.status is PlacementReservationStatus.BLOCKED:
            retry_at = self._record_blocked(request.replica_key, PlacementBlockReason.INSUFFICIENT_GPU)
            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.INSUFFICIENT_GPU,
                message=reservation.message,
                retry_at=retry_at,
            )
        if reservation.token is None:
            retry_at = self._record_blocked(request.replica_key, PlacementBlockReason.PG_READY_FAILED)
            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.PG_READY_FAILED,
                message="placement reservation did not return a token",
                retry_at=retry_at,
            )

        pg = None
        try:
            pg = await self._create_placement_group(request)
            await self._await_pg_ready(pg, timeout_s=request.ready_timeout_s)
        except TimeoutError as exc:
            await self._remove_placement_group(pg)
            await self.release(reservation.token)
            retry_at = self._record_blocked(request.replica_key, PlacementBlockReason.PG_PENDING_TIMEOUT)
            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.PG_PENDING_TIMEOUT,
                message=f"placement group ready timed out after {request.ready_timeout_s:g}s: {exc}",
                retry_at=retry_at,
            )
        except Exception as exc:
            existing_pg = await self._existing_active_placement_group(request)
            if existing_pg is not None and _is_placement_group_name_conflict(exc, request.placement_group_name):
                await self.release(reservation.token)
                self._blocked.pop(request.replica_key, None)
                return PlacementGroupCreateResult(
                    status=PlacementGroupCreateStatus.READY,
                    placement_group_name=request.placement_group_name,
                    placement_group=existing_pg,
                )
            await self._remove_placement_group(pg)
            await self.release(reservation.token)
            retry_at = self._record_blocked(request.replica_key, PlacementBlockReason.PG_READY_FAILED)
            return PlacementGroupCreateResult(
                status=PlacementGroupCreateStatus.BLOCKED,
                placement_group_name=request.placement_group_name,
                reason=PlacementBlockReason.PG_READY_FAILED,
                message=f"{type(exc).__name__}: {exc}",
                retry_at=retry_at,
            )

        await self.release(reservation.token)
        self._blocked.pop(request.replica_key, None)
        return PlacementGroupCreateResult(
            status=PlacementGroupCreateStatus.READY,
            placement_group_name=request.placement_group_name,
            placement_group=pg,
        )

    async def reserve(self, request: PlacementReservationRequest) -> PlacementReservationResult:
        async with self._lock:
            available = await self._available_gpus_by_node_unlocked()
            required = dict(request.required_gpus_by_node)
            blockers = {
                node_ip: required_gpus
                for node_ip, required_gpus in required.items()
                if not _has_available_gpus(available, node_ip=node_ip, required_gpus=int(required_gpus))
            }
            if blockers:
                return PlacementReservationResult(
                    status=PlacementReservationStatus.BLOCKED,
                    reason=PlacementBlockReason.INSUFFICIENT_GPU,
                    message=f"insufficient GPU for replica_key={request.replica_key!r} blockers={blockers!r}",
                    available_gpus_by_node=_gpu_by_node_tuple(available),
                    reserved_gpus_by_node=self._in_flight_gpus_by_node_unlocked(),
                )

            token = PlacementReservationToken(
                token_id=uuid.uuid4().hex,
                replica_key=request.replica_key,
                required_gpus_by_node=request.required_gpus_by_node,
                placement_group_name=request.placement_group_name,
                namespace=request.namespace or self._namespace,
            )
            self._reservations[token.token_id] = token
            return PlacementReservationResult(
                status=PlacementReservationStatus.RESERVED,
                token=token,
                available_gpus_by_node=_gpu_by_node_tuple(available),
                reserved_gpus_by_node=self._in_flight_gpus_by_node_unlocked(),
            )

    async def release(self, token: PlacementReservationToken) -> PlacementReservationResult:
        async with self._lock:
            removed = self._reservations.pop(token.token_id, None)
            if removed is None:
                return PlacementReservationResult(
                    status=PlacementReservationStatus.NOT_FOUND,
                    reason=PlacementBlockReason.UNKNOWN_RESERVATION,
                    message=f"unknown placement reservation token_id={token.token_id!r}",
                    available_gpus_by_node=_gpu_by_node_tuple(
                        await self._available_gpus_by_node_unlocked()
                    ),
                    reserved_gpus_by_node=self._in_flight_gpus_by_node_unlocked(),
                )
            return PlacementReservationResult(
                status=PlacementReservationStatus.RELEASED,
                token=removed,
                available_gpus_by_node=_gpu_by_node_tuple(await self._available_gpus_by_node_unlocked()),
                reserved_gpus_by_node=self._in_flight_gpus_by_node_unlocked(),
            )

    async def _existing_active_placement_group(self, request: PlacementGroupCreateRequest) -> Any | None:
        if self._placement_group_table is None:
            return None
        raw = self._placement_group_table()
        table = await raw if inspect.isawaitable(raw) else raw
        target_namespace = request.namespace or self._namespace
        for row in _placement_group_rows(table):
            if str(row.get("name") or "") != request.placement_group_name:
                continue
            if not _placement_group_is_active(row):
                continue
            if not _placement_group_namespace_matches(row, target_namespace):
                continue
            return row.get("placement_group") or row.get("pg") or row.get("handle") or row
        return None

    async def rebuild_from_placement_group_table(self) -> PlacementReservationSnapshot:
        if self._placement_group_table is None:
            table: Mapping[str, Any] | Iterable[Any] = ()
        else:
            raw = self._placement_group_table()
            table = await raw if inspect.isawaitable(raw) else raw
        rebuilt: dict[str, int] = {}
        for row in _placement_group_rows(table):
            if not _placement_group_is_active(row):
                continue
            if not _placement_group_namespace_matches(row, self._namespace):
                continue
            for node_ip, gpu_count in _gpu_by_pinned_node_from_pg(row).items():
                rebuilt[node_ip] = rebuilt.get(node_ip, 0) + int(gpu_count)
        async with self._lock:
            self._rebuilt_gpus_by_node = rebuilt
            return await self._snapshot_unlocked()

    async def snapshot(self) -> PlacementReservationSnapshot:
        async with self._lock:
            return await self._snapshot_unlocked()

    async def _snapshot_unlocked(self) -> PlacementReservationSnapshot:
        return PlacementReservationSnapshot(
            in_flight_gpus_by_node=self._in_flight_gpus_by_node_unlocked(),
            rebuilt_gpus_by_node=_gpu_by_node_tuple(self._rebuilt_gpus_by_node),
            available_gpus_by_node=_gpu_by_node_tuple(await self._available_gpus_by_node_unlocked()),
            active_reservation_count=len(self._reservations),
            blocked_replicas=tuple(
                sorted((replica_key, reason, retry_at) for replica_key, (reason, retry_at, _backoff) in self._blocked.items())
            ),
        )

    async def _available_gpus_by_node_unlocked(self) -> dict[str, int]:
        raw = self._observed_free_gpus_by_node()
        observed = await raw if inspect.isawaitable(raw) else raw
        available = dict(_normalize_gpu_by_node(observed))
        for node_ip, gpu_count in self._rebuilt_gpus_by_node.items():
            available = _subtract_available_gpus(
                available,
                node_ip=node_ip,
                gpu_count=int(gpu_count),
            )
        for node_ip, gpu_count in self._in_flight_gpus_by_node_unlocked():
            available = _subtract_available_gpus(
                available,
                node_ip=node_ip,
                gpu_count=int(gpu_count),
            )
        return available

    def _in_flight_gpus_by_node_unlocked(self) -> GpuByNode:
        totals: dict[str, int] = {}
        for token in self._reservations.values():
            for node_ip, gpu_count in token.required_gpus_by_node:
                totals[node_ip] = totals.get(node_ip, 0) + int(gpu_count)
        return _gpu_by_node_tuple(totals)

    async def _create_placement_group(self, request: PlacementGroupCreateRequest) -> Any:
        factory = self._placement_group_factory or _ray_placement_group_factory
        out = factory(
            bundles=request.bundle_dicts,
            strategy=request.strategy,
            name=request.placement_group_name,
            lifetime=request.lifetime,
            namespace=request.namespace or self._namespace,
        )
        return await out if inspect.isawaitable(out) else out

    async def _await_pg_ready(self, pg: Any, *, timeout_s: float) -> None:
        ready = getattr(pg, "ready", None)
        if not callable(ready):
            return
        out = ready()
        if inspect.isawaitable(out):
            await asyncio.wait_for(out, timeout=max(0.001, float(timeout_s)))
            return
        await asyncio.wait_for(
            asyncio.to_thread(_ray_get, out),
            timeout=max(0.001, float(timeout_s)),
        )

    async def _remove_placement_group(self, pg: Any | None) -> None:
        if pg is None:
            return
        remover = self._placement_group_remover or _ray_placement_group_remover
        out = remover(pg)
        if inspect.isawaitable(out):
            await out

    def _record_blocked(self, replica_key: str, reason: PlacementBlockReason) -> float:
        _old_reason, _old_retry_at, old_backoff = self._blocked.get(
            replica_key,
            (reason, 0.0, 0.0),
        )
        backoff = self._initial_backoff_s if old_backoff <= 0 else min(self._max_backoff_s, old_backoff * 2.0)
        retry_at = float(self._monotonic()) + backoff
        self._blocked[replica_key] = (reason, retry_at, backoff)
        return retry_at


def placement_group_bundle_request_for_spec(spec: Any) -> PlacementGroupBundleRequest:
    launcher_key = str(getattr(spec, "launcher_key", "") or "").strip().lower()
    domain_key = str(getattr(spec, "domain_key", "") or "")
    replica_id = str(getattr(spec, "replica_id", "") or "replica-0")
    replica_key = f"{domain_key}::{replica_id}"
    base_model = _base_model_from_spec(spec)
    node_pins = _spec_node_pins(spec)
    placement_slices = _spec_placement_slices(spec)
    namespace = _ray_namespace()
    actor_name = str(_call_spec_method(spec, "normalized_actor_name") or _default_model_actor_name(domain_key, replica_id))

    if launcher_key == "cpu_runtime" or int(getattr(spec, "gpu_count", 1) or 0) <= 0:
        raise ValueError(f"spec does not require GPU placement: {replica_key}")

    if launcher_key == "dense" or domain_key.startswith("dense:"):
        dense_actor_name = _dense_actor_name(base_model or domain_key)
        return PlacementGroupBundleRequest.for_dense(
            replica_key=replica_key,
            placement_group_name=_namespace_actor_pg_name(dense_actor_name, namespace),
            node_ip=node_pins[0] if node_pins else None,
            namespace=namespace,
        )

    if launcher_key in {"megatron", "bumblebee"} or domain_key.startswith(("megatron:", "bumblebee:")):
        training_actor_name = (
            _bumblebee_actor_name(base_model or domain_key)
            if launcher_key == "bumblebee" or domain_key.startswith("bumblebee:")
            else _megatron_actor_name(base_model or domain_key)
        )
        return PlacementGroupBundleRequest.for_distributed_training(
            replica_key=replica_key,
            placement_group_name=_namespace_actor_pg_name(training_actor_name, namespace),
            parallel=_parallel_topology_from_spec(spec),
            placement_slices=placement_slices,
            namespace=namespace,
        )

    worker_gpus = int(getattr(spec, "gpu_count", None) or 1)
    return PlacementGroupBundleRequest.for_vllm(
        replica_key=replica_key,
        placement_group_name=f"{actor_name}_pg",
        worker_gpus=worker_gpus,
        node_ips=node_pins,
        namespace=namespace,
    )


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _call_spec_method(spec: Any, method_name: str) -> Any | None:
    method = getattr(spec, method_name, None)
    if callable(method):
        return method()
    return None


def _base_model_from_spec(spec: Any) -> str | None:
    raw = getattr(spec, "base_model", None)
    if raw:
        return str(raw)
    domain_key = str(getattr(spec, "domain_key", "") or "")
    for prefix in ("vllm:", "training:", "megatron:", "bumblebee:", "dense:"):
        if domain_key.startswith(prefix):
            model = domain_key.removeprefix(prefix).strip()
            return model or None
    return None


def _spec_node_pins(spec: Any) -> tuple[str, ...]:
    raw = _call_spec_method(spec, "normalized_node_pins")
    if raw is None:
        raw = getattr(spec, "node_pins", ())
        node_pin = getattr(spec, "node_pin", None)
        if node_pin:
            raw = (*tuple(raw or ()), node_pin)
    return tuple(dict.fromkeys(str(pin) for pin in (raw or ()) if str(pin).strip()))


def _spec_placement_slices(spec: Any) -> tuple[tuple[str, str, int], ...]:
    out: list[tuple[str, str, int]] = []
    for raw in getattr(spec, "placement_slices", ()) or ():
        if len(raw) != 3:
            continue
        replica_id, node_ip, gpu_count = raw
        out.append((str(replica_id), str(node_ip), int(gpu_count)))
    return tuple(out)


def _default_model_actor_name(domain_key: str, replica_id: str) -> str:
    return f"mint_model_runtime_{_sanitize_actor_name_part(domain_key).lower()}_{_sanitize_actor_name_part(replica_id).lower()}"


def _sanitize_actor_name_part(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return out or "unknown"


def _sanitize_pg_component(value: str | None) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def _namespace_actor_pg_name(actor_name: str, namespace: str) -> str:
    return f"{actor_name}_{_sanitize_pg_component(namespace)}_pg"


def _dense_actor_name(base_model: str) -> str:
    model_key = str(base_model or "").strip()
    if model_key.startswith("/"):
        model_key = model_key.split("/")[-1]
    else:
        model_key = model_key.replace("/", "__")
    model_key = (
        model_key.replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .lower()
    )
    return f"mint_dense_{model_key or 'unknown'}"


def _megatron_actor_name(base_model: str) -> str:
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots", str(base_model))
    if match:
        model_name = match.group(2).lower().replace("-", "_").replace(".", "_")
    else:
        model_name = str(base_model).split("/")[-1].lower().replace("-", "_").replace(".", "_")
    return f"mint_megatron_{model_name}"


def _bumblebee_actor_name(base_model: str) -> str:
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots", str(base_model))
    model_name = match.group(2) if match else str(base_model).split("/")[-1]
    model_name = model_name.lower().replace("-", "_").replace(".", "_")
    return f"mint_bumblebee_{model_name}"


def _parallel_topology_from_spec(spec: Any) -> ParallelTopology:
    explicit_fields = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
    )
    if not any(hasattr(spec, field) for field in explicit_fields):
        gpu_count = int(getattr(spec, "gpu_count", None) or 1)
        if gpu_count > 1:
            return ParallelTopology(tensor_parallel_size=gpu_count)
    return ParallelTopology(
        tensor_parallel_size=int(getattr(spec, "tensor_parallel_size", 1) or 1),
        pipeline_parallel_size=int(getattr(spec, "pipeline_parallel_size", 1) or 1),
        expert_parallel_size=int(getattr(spec, "expert_parallel_size", 1) or 1),
        expert_tensor_parallel_size=_optional_positive_int(getattr(spec, "expert_tensor_parallel_size", None)),
        context_parallel_size=int(getattr(spec, "context_parallel_size", 1) or 1),
    )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    out = int(value)
    return out if out > 0 else None


def _ray_namespace() -> str:
    env_ns = None
    try:
        import os

        env_ns = os.environ.get("MINT_RAY_NAMESPACE")
    except Exception:
        env_ns = None
    if env_ns:
        return str(env_ns)
    try:
        from ..config import RAY_NAMESPACE

        return str(RAY_NAMESPACE)
    except Exception:
        return "mint"


def _normalize_gpu_by_node(values: Mapping[str, int]) -> GpuByNode:
    normalized: dict[str, int] = {}
    for raw_node_ip, raw_gpu_count in values.items():
        node_ip = str(raw_node_ip).strip()
        if not node_ip:
            raise ValueError("node IP must be non-empty")
        gpu_count = int(raw_gpu_count)
        if gpu_count <= 0:
            raise ValueError(f"GPU count for node {node_ip!r} must be positive, got {raw_gpu_count!r}")
        normalized[node_ip] = normalized.get(node_ip, 0) + gpu_count
    return _gpu_by_node_tuple(normalized)


def _normalize_bundles(values: Iterable[Mapping[str, int | float]]) -> tuple[BUNDLE, ...]:
    bundles: list[BUNDLE] = []
    for raw_bundle in values:
        bundle: dict[str, int | float] = {}
        for key, value in raw_bundle.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"placement bundle resource key must be non-empty str, got {key!r}")
            resource = float(value)
            if resource <= 0:
                continue
            bundle[key] = int(resource) if resource.is_integer() else resource
        if not bundle:
            raise ValueError("placement bundle must contain at least one positive resource")
        bundles.append(tuple(sorted(bundle.items())))
    if not bundles:
        raise ValueError("bundles must not be empty")
    return tuple(bundles)


def _gpu_by_node_tuple(values: Mapping[str, int]) -> GpuByNode:
    return tuple(sorted((str(node_ip), int(gpu_count)) for node_ip, gpu_count in values.items()))


def _required_gpus_by_node_from_bundles(bundles: Iterable[BUNDLE]) -> GpuByNode:
    required: dict[str, int] = {}
    unpinned_gpus = 0
    for bundle in bundles:
        bundle_dict = dict(bundle)
        gpu_count = int(float(bundle_dict.get("GPU", 0) or 0))
        if gpu_count <= 0:
            continue
        pinned_nodes = [
            str(key).split("node:", 1)[1]
            for key, value in bundle_dict.items()
            if isinstance(key, str) and key.startswith("node:") and float(value or 0) > 0
        ]
        if pinned_nodes:
            for node_ip in pinned_nodes:
                required[node_ip] = required.get(node_ip, 0) + gpu_count
        else:
            unpinned_gpus += gpu_count
    if unpinned_gpus > 0:
        required[CLUSTER_GPU_KEY] = required.get(CLUSTER_GPU_KEY, 0) + unpinned_gpus
    return _gpu_by_node_tuple(required)


def _has_available_gpus(available: Mapping[str, int], *, node_ip: str, required_gpus: int) -> bool:
    if node_ip == CLUSTER_GPU_KEY:
        return sum(int(value) for value in available.values()) >= int(required_gpus)
    return int(available.get(node_ip, 0)) >= int(required_gpus)


def _subtract_available_gpus(available: Mapping[str, int], *, node_ip: str, gpu_count: int) -> dict[str, int]:
    out = dict(available)
    remaining = int(gpu_count)
    if remaining <= 0:
        return out
    if node_ip != CLUSTER_GPU_KEY:
        out[node_ip] = max(0, int(out.get(node_ip, 0)) - remaining)
        return out
    for current_node_ip in sorted(out):
        if remaining <= 0:
            break
        current = int(out.get(current_node_ip, 0))
        consumed = min(current, remaining)
        out[current_node_ip] = current - consumed
        remaining -= consumed
    return out


def _ray_placement_group_factory(**kwargs: Any) -> Any:
    import ray

    try:
        return ray.util.placement_group(**kwargs)
    except TypeError:
        kwargs.pop("namespace", None)
        return ray.util.placement_group(**kwargs)


def _ray_placement_group_remover(pg: Any) -> None:
    import ray

    ray.util.remove_placement_group(pg)


def _ray_get(ref: Any) -> Any:
    import ray

    return ray.get(ref)


def _string_dict(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _list_dict(raw: object) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Iterable):
            values = [str(item) for item in value if str(item).strip()]
        else:
            values = []
        out[str(key)] = values
    return out


def _placement_slices_dict(raw: object) -> dict[str, tuple[tuple[str, str, int], ...]]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, tuple[tuple[str, str, int], ...]] = {}
    for key, value in raw.items():
        slices: list[tuple[str, str, int]] = []
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            for item in value:
                if isinstance(item, Iterable) and not isinstance(item, str | bytes):
                    parts = tuple(item)
                    if len(parts) == 3:
                        replica_id, node_ip, gpu_count = parts
                        slices.append((str(replica_id), str(node_ip), int(gpu_count)))
        out[str(key)] = tuple(slices)
    return out


def _placement_group_rows(table: Mapping[str, Any] | Iterable[Any]) -> tuple[dict[str, Any], ...]:
    if isinstance(table, Mapping):
        values = table.values()
    else:
        values = table
    rows: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "asdict"):
            value = value.asdict()
        if isinstance(value, dict):
            rows.append(dict(value))
    return tuple(rows)


def _placement_group_is_active(row: Mapping[str, Any]) -> bool:
    return str(row.get("state") or row.get("State") or "").upper() != "REMOVED"


def _placement_group_namespace_matches(row: Mapping[str, Any], namespace: str | None) -> bool:
    if namespace is None:
        return True
    row_namespace = row.get("namespace") or row.get("ray_namespace") or row.get("rayNamespace")
    return row_namespace is None or str(row_namespace) == str(namespace)


def _is_placement_group_name_conflict(exc: Exception, placement_group_name: str) -> bool:
    message = str(exc)
    return placement_group_name in message and "already exists" in message


def _placement_group_bundles(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = row.get("bundles") or row.get("Bundles") or ()
    if isinstance(raw, Mapping):
        values = raw.values()
    elif isinstance(raw, list | tuple):
        values = raw
    else:
        values = ()
    return tuple(dict(bundle) for bundle in values if isinstance(bundle, dict))


def _gpu_by_pinned_node_from_pg(row: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for bundle in _placement_group_bundles(row):
        try:
            gpu_count = int(float(bundle.get("GPU", 0) or 0))
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count <= 0:
            continue
        pinned_nodes = [
            str(key).split("node:", 1)[1]
            for key, value in bundle.items()
            if isinstance(key, str) and key.startswith("node:") and float(value or 0) > 0
        ]
        if pinned_nodes:
            for node_ip in pinned_nodes:
                out[node_ip] = out.get(node_ip, 0) + gpu_count
        else:
            out[CLUSTER_GPU_KEY] = out.get(CLUSTER_GPU_KEY, 0) + gpu_count
    return out


__all__ = [
    "ClusterPlacementController",
    "PlacementReconcileRequest",
    "PlacementReconcileResult",
    "PlacementGroupBundleRequest",
    "PlacementBlockReason",
    "PlacementGroupCreateRequest",
    "PlacementGroupCreateResult",
    "PlacementGroupCreateStatus",
    "PlacementReservationRequest",
    "PlacementReservationResult",
    "PlacementReservationSnapshot",
    "PlacementReservationStatus",
    "PlacementReservationToken",
    "placement_group_bundle_request_for_spec",
]
