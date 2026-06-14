from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable, cast

from fastapi import Request, Response
from mint_server.backend.control_plane_contracts import (
    AsyncSchedulerControlPlane,
    AsyncSchedulerQueue,
    AsyncTaskLedger,
    CancelTaskResult,
    ClaimResult,
    ContainsResult,
    ExecutorOutcome,
    InProcessSchedulerQueueAdapter,
    ModelWorkTaskGateway,
    SyncReplicasResult,
)
from mint_server.backend.model_work_admission import ModelWorkAdmissionResult
from mint_server.backend.model_actor_supervisor import (
    ModelActorSpec,
    ModelActorSupervisorCore,
    consumer_id_for_replica,
    queue_id_for_replica,
)
from mint_server.backend.model_engine_host import ModelEngineHost
from mint_server.backend.model_work_admission import enqueue_model_work
from mint_server.backend.model_work_scheduler import _ModelWorkSchedulerActor
from mint_server.backend.model_work_task_gateway import SchedulerModelWorkTaskGateway
from mint_server.backend.task_payload_store import TaskPayloadStore
from mint_server.backend.task_state_store import FutureStatus, TaskFutureService, TaskStateStore
from mint_server.models.types import FutureCancelRequest, FutureRetrieveRequest
from mint_server.routes import futures as futures_route

from .faults import FaultController
from .scenarios import DEFAULT_DOMAIN, DEFAULT_GENERATION, DEFAULT_REPLICA, sampling_meta


class LocalAsyncTaskStateClient:
    def __init__(self, store: TaskStateStore, faults: FaultController | None = None) -> None:
        self.store = store
        self.faults = faults or FaultController()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        method = str(method)
        self.calls.append((method, {"args": args, **dict(kwargs)}))
        await self.faults.before_call(f"task_state.{method}", **kwargs)
        out = await asyncio.to_thread(getattr(self.store, method), *args, **kwargs)
        await self.faults.before_call(f"task_state.{method}.after", **kwargs)
        return out

    async def async_ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        _ = timeout_s, create_if_missing
        return await self.async_ping()

    async def async_ensure_started(self) -> None:
        return None

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        _ = timeout_s
        return await asyncio.to_thread(self.store.ping)

    async def async_acquire_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("acquire_scheduler_owner", **kwargs)

    async def async_renew_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("renew_scheduler_owner", **kwargs)

    async def async_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("create_task", **kwargs)

    async def async_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        try:
            record = await self.async_get_task(request_id=str(kwargs["request_id"]))
            return {"ok": True, "created": False, "record": record}
        except Exception:
            return await self.async_create_task(**kwargs)

    async def async_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("assign_task", **kwargs)

    async def async_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("claim_task", **kwargs)

    async def async_renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("renew_lease", **kwargs)

    async def async_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("begin_finalize", **kwargs)

    async def async_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("commit_finalize_success", **kwargs)

    async def async_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("commit_finalize_failure", **kwargs)

    async def async_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("requeue_task", **kwargs)

    async def async_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("forget_task", **kwargs)

    async def async_get_task(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if args:
            if len(args) != 1 or "request_id" in kwargs:
                raise TypeError("async_get_task accepts request_id as either positional or keyword")
            kwargs["request_id"] = args[0]
        return await self._call("get_task", **kwargs)

    async def async_list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call("list_active_tasks", **kwargs)

    async def async_list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call("list_tasks_by_metadata", **kwargs)

    async def async_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("update_task_metadata", **kwargs)

    async def async_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("stage_payload", **kwargs)

    async def async_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("complete_task_success", **kwargs)

    async def async_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("complete_task_failure", **kwargs)

    async def async_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("mark_task_retrieved", **kwargs)

    async def async_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        return await self._call("expire_active_tasks", **kwargs)

    async def async_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call("list_terminal_payloads_for_eviction", **kwargs)

    async def async_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("mark_payload_evicted", **kwargs)

    async def async_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        return await self._call("delete_expired_tombstones", **kwargs)

    async def async_wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        from mint_server.backend.task_state_store import TERMINAL_TASK_STATUSES

        deadline = time.time() + max(0.0, float(timeout_s))
        last_record: dict[str, Any] | None = None
        while True:
            try:
                record = await asyncio.to_thread(self.store.get_task, str(request_id))
            except KeyError:
                return {"changed": False, "timeout": False, "missing": True}
            last_record = record
            status = str(record.get("status") or "")
            if (terminal_only and status in TERMINAL_TASK_STATUSES) or not terminal_only:
                return {
                    "changed": True,
                    "timeout": False,
                    "missing": False,
                    "record": record,
                }
            if time.time() >= deadline:
                return {
                    "changed": False,
                    "timeout": True,
                    "missing": False,
                    "record": last_record,
                }
            await asyncio.sleep(0.005)


class FaultingTaskPayloadStore(TaskPayloadStore):
    def __init__(self, *args: Any, faults: FaultController, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.faults = faults
        self.fail_writes = False

    def write_json_payload(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_writes:
            raise RuntimeError("synthetic payload write failure")
        return super().write_json_payload(**kwargs)


@dataclass
class SchedulerComponentWorld:
    tmp_path: Path
    domain_key: str = DEFAULT_DOMAIN
    replica_id: str = DEFAULT_REPLICA
    generation: int = DEFAULT_GENERATION
    task_store: TaskStateStore | None = None

    def __post_init__(self) -> None:
        self.faults = FaultController()
        if self.task_store is None:
            self.task_store = TaskStateStore.in_memory()
        self.task_state = LocalAsyncTaskStateClient(self.task_store, self.faults)
        self.payload_store = FaultingTaskPayloadStore(
            root_dir=self.tmp_path / "payloads",
            faults=self.faults,
        )
        self.future_service = TaskFutureService(
            task_state_client=cast(Any, self.task_state),
            future_state_client=cast(Any, self.task_state),
            payload_store=self.payload_store,
        )
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id="component-scheduler",
        )
        scheduler_adapter = InProcessSchedulerQueueAdapter(self.scheduler_actor)
        self.runtime_queue: AsyncSchedulerQueue = scheduler_adapter
        self.scheduler: AsyncSchedulerControlPlane = scheduler_adapter
        task_ledger = self.scheduler_actor._task_state_store
        assert task_ledger is not None
        self.task_ledger: AsyncTaskLedger = task_ledger
        self.task_gateway: ModelWorkTaskGateway = SchedulerModelWorkTaskGateway(
            scheduler_client=self.scheduler,
            task_ledger_client=self.task_ledger,
        )
        self.event_log: list[tuple[str, dict[str, Any]]] = []

    def replace_scheduler(self, *, owner_id: str) -> None:
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id=owner_id,
        )
        scheduler_adapter = InProcessSchedulerQueueAdapter(self.scheduler_actor)
        self.runtime_queue = scheduler_adapter
        self.scheduler = scheduler_adapter
        task_ledger = self.scheduler_actor._task_state_store
        assert task_ledger is not None
        self.task_ledger = task_ledger
        self.task_gateway = SchedulerModelWorkTaskGateway(
            scheduler_client=self.scheduler,
            task_ledger_client=self.task_ledger,
        )

    @property
    def consumer_id(self) -> str:
        return consumer_id_for_replica(self.domain_key, self.replica_id, self.generation)

    def replica(
        self,
        *,
        status: str = "healthy",
        generation: int | None = None,
        replica_id: str | None = None,
        capacity: int = 4,
    ) -> dict[str, Any]:
        generation = self.generation if generation is None else int(generation)
        replica_id = self.replica_id if replica_id is None else str(replica_id)
        return {
            "domain_key": self.domain_key,
            "replica_id": replica_id,
            "consumer_id": consumer_id_for_replica(self.domain_key, replica_id, generation),
            "generation": generation,
            "status": status,
            "queue_id": queue_id_for_replica(self.domain_key, replica_id),
            "capacity": int(capacity),
            "actor_name": f"component-runtime-{replica_id}",
            "node_pins": ["127.0.0.1"],
            "updated_at": time.time(),
        }

    async def start(self) -> None:
        await self.scheduler.sync_replicas([self.replica(status="healthy")])

    async def enqueue_sampling(
        self,
        request_id: str,
        *,
        assign: bool = True,
        affinity_group: str = "lora:session-a:generation:1",
        ordering_key: str | None = None,
        token_cost: int = 1,
        request_json: bytes = b'{"prompt":"hello"}',
    ) -> ModelWorkAdmissionResult:
        return await enqueue_model_work(
            request_id=request_id,
            op="sampling.asample",
            request_json=request_json,
            domain_key=self.domain_key,
            queued_meta=sampling_meta(self.domain_key),
            user_id="user-a",
            apikey_id="key-a",
            throttle_principal="apikey:key-a",
            webhook_url=None,
            affinity_group=affinity_group,
            ordering_key=ordering_key,
            token_cost=token_cost,
            assign=assign,
            assign_max_items=1,
            gateway_client=self.task_gateway,
        )

    async def claim_one(
        self,
        *,
        lease_ttl_s: float = 30.0,
        replica_id: str | None = None,
        consumer_id: str | None = None,
        consumer_generation: int | None = None,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        replica_id = self.replica_id if replica_id is None else str(replica_id)
        consumer_generation = self.generation if consumer_generation is None else int(consumer_generation)
        consumer_id = (
            consumer_id
            if consumer_id is not None
            else consumer_id_for_replica(self.domain_key, replica_id, consumer_generation)
        )
        claimed = await self.runtime_queue.claim(
            domain_key=self.domain_key,
            replica_id=replica_id,
            consumer_id=consumer_id,
            consumer_generation=consumer_generation,
            max_items=1,
            token_budget=token_budget,
            lease_ttl_s=lease_ttl_s,
        )
        leases = claimed.leases
        assert isinstance(leases, list)
        assert len(leases) == 1
        return leases[0]

    async def claim_none(self, *, lease_ttl_s: float = 30.0) -> ClaimResult:
        claimed = await self.runtime_queue.claim(
            domain_key=self.domain_key,
            replica_id=self.replica_id,
            consumer_id=self.consumer_id,
            consumer_generation=self.generation,
            max_items=1,
            lease_ttl_s=lease_ttl_s,
        )
        assert claimed.leases == []
        return claimed

    async def acquire_owner(
        self,
        *,
        owner_id: str,
        ttl_s: float = 30.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        return await self.task_state.async_acquire_owner(
            owner_id=str(owner_id),
            ttl_s=float(ttl_s),
            now=now,
        )

    async def runtime_once(
        self,
        *,
        executor: Callable[[dict[str, Any]], Awaitable[ExecutorOutcome | None]] | None = None,
        lease_ttl_s: float = 1.0,
    ) -> ModelEngineHost:
        async def _default_executor(lease: dict[str, Any]) -> ExecutorOutcome:
            request_id = str(lease["item"]["request_id"])
            await self.future_service.async_resolve(request_id, {"ok": True, "request_id": request_id})
            return ExecutorOutcome(kind="success")

        actor = ModelEngineHost(
            domain_key=self.domain_key,
            replica_id=self.replica_id,
            actor_name=f"component-runtime-{self.replica_id}",
            actor_generation=self.generation,
            poll_interval_s=0.01,
            lease_ttl_s=lease_ttl_s,
            max_claim=1,
            scheduler_client=cast(Any, self.runtime_queue),
            task_futures_client=self.future_service,
            task_state_store_client=self.task_state,
            payload_store=self.payload_store,
            executor=executor or _default_executor,
        )
        await actor.run_once()
        return actor

    async def observe_task(self, request_id: str) -> dict[str, Any]:
        return await self.task_state.async_get_task(request_id=str(request_id))

    async def observe_scheduler(self, request_id: str) -> ContainsResult:
        return await self.scheduler.contains(request_id=str(request_id))

    async def observe_future_status(self, request_id: str) -> FutureStatus:
        return await self.future_service.async_get_status(str(request_id))

    async def assert_consistent(
        self,
        *,
        terminal_request_ids: Iterable[str] = (),
        terminal_payloads: bool = False,
    ) -> None:
        from .invariants import (
            assert_every_terminal_has_payload_ref,
            assert_lease_consistency,
            assert_no_double_lease,
            assert_no_orphan_assigned,
            assert_terminal_not_scheduled,
        )

        await assert_no_double_lease(self)
        await assert_lease_consistency(self)
        await assert_no_orphan_assigned(self)
        if terminal_payloads:
            await assert_every_terminal_has_payload_ref(self)
        for request_id in terminal_request_ids:
            await assert_terminal_not_scheduled(self, str(request_id))

    def inject_payload_write_failure(self, enabled: bool) -> None:
        self.payload_store.fail_writes = bool(enabled)

    async def retrieve(
        self,
        request_id: str,
        monkeypatch: Any,
        *,
        admin: bool = True,
        wait_timeout_s: float = 0.0,
        scheduler_override: Any | None = None,
    ) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(futures_route, "task_futures", self.future_service)
        import mint_server.backend.model_work_scheduler as scheduler_module
        import mint_server.backend.model_work_task_gateway as gateway_module

        monkeypatch.setattr(scheduler_module, "model_work_scheduler", scheduler_override or self.scheduler)
        monkeypatch.setattr(
            gateway_module,
            "model_work_task_gateway",
            SchedulerModelWorkTaskGateway(
                scheduler_client=scheduler_override or self.scheduler,
                task_ledger_client=self.task_ledger if scheduler_override is None else self.task_state,
            ),
        )
        monkeypatch.setattr(futures_route, "task_payload_store", self.payload_store)
        monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: float(wait_timeout_s))
        monkeypatch.setattr(futures_route, "_is_privileged", lambda _request: bool(admin))
        response = SimpleNamespace(status_code=200, headers={})
        user_id = "admin" if admin else "user-a"
        request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": user_id}), headers={})
        payload = await futures_route.retrieve_future(
            FutureRetrieveRequest(request_id=request_id),
            cast(Request, request),
            cast(Response, response),
        )
        return int(response.status_code), payload

    async def cancel(
        self,
        request_id: str,
        monkeypatch: Any,
        *,
        reason: str = "cancelled",
        scheduler_override: Any | None = None,
    ) -> CancelTaskResult:
        monkeypatch.setattr(futures_route, "task_futures", self.future_service)
        import mint_server.backend.model_work_scheduler as scheduler_module
        import mint_server.backend.model_work_task_gateway as gateway_module

        monkeypatch.setattr(scheduler_module, "model_work_scheduler", scheduler_override or self.scheduler)
        monkeypatch.setattr(
            gateway_module,
            "model_work_task_gateway",
            SchedulerModelWorkTaskGateway(
                scheduler_client=scheduler_override or self.scheduler,
                task_ledger_client=self.task_ledger if scheduler_override is None else self.task_state,
            ),
        )
        payload = await futures_route.cancel_future(
            FutureCancelRequest(request_id=request_id, reason=reason)
        )
        return CancelTaskResult.from_wire(payload)

    def supervisor(self) -> ModelActorSupervisorCore:
        async def _runtime_factory(spec: ModelActorSpec, generation: int) -> Any:
            return SimpleNamespace(
                actor_name=spec.normalized_actor_name(),
                start=lambda: {"running": True},
                health_snapshot=lambda: {
                    "running": True,
                    "actor_generation": generation,
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                },
                shutdown=lambda: {"ok": True},
            )

        async def _sync(registrations: list[Any]) -> SyncReplicasResult:
            return await self.scheduler.sync_replicas([registration.to_dict() for registration in registrations])

        return ModelActorSupervisorCore(
            specs=[
                ModelActorSpec(
                    domain_key=self.domain_key,
                    replica_id=self.replica_id,
                    base_model="model-a",
                    gpu_count=1,
                )
            ],
            runtime_factory=_runtime_factory,
            scheduler_sync=_sync,
            control_plane_dependencies=[],
            control_plane_enabled=False,
            node_metrics_enabled=False,
        )

    def supervisor_with_factory(
        self,
        runtime_factory: Callable[[ModelActorSpec, int], Awaitable[Any]],
    ) -> ModelActorSupervisorCore:
        async def _sync(registrations: list[Any]) -> SyncReplicasResult:
            return await self.scheduler.sync_replicas([registration.to_dict() for registration in registrations])

        return ModelActorSupervisorCore(
            specs=[
                ModelActorSpec(
                    domain_key=self.domain_key,
                    replica_id=self.replica_id,
                    base_model="model-a",
                    gpu_count=1,
                )
            ],
            runtime_factory=runtime_factory,
            scheduler_sync=_sync,
            control_plane_dependencies=[],
            control_plane_enabled=False,
            node_metrics_enabled=False,
        )

    def close(self) -> None:
        self.assert_basic_invariants_sync()
        manager_task = getattr(self.scheduler_actor, "_background_loop_manager_task", None)
        if manager_task is not None and not manager_task.done():
            manager_task.cancel()
        task_store = self.task_store
        assert task_store is not None
        task_store.close()

    def assert_basic_invariants_sync(self) -> None:
        """Run global scheduler invariants at component-world teardown."""
        stats = self.scheduler_actor.stats()
        leases = stats.get("leases") or []
        assert isinstance(leases, list)

        lease_ids: set[str] = set()
        request_ids: set[str] = set()
        for lease in leases:
            assert isinstance(lease, dict)
            lease_id = str(lease.get("lease_id") or "")
            raw_item = lease.get("item")
            item = raw_item if isinstance(raw_item, dict) else {}
            request_id = str(item.get("request_id") or lease.get("request_id") or "")
            assert lease_id
            assert request_id
            assert lease_id not in lease_ids
            assert request_id not in request_ids
            lease_ids.add(lease_id)
            request_ids.add(request_id)

            assert self.task_store is not None
            record = self.task_store.get_task(request_id)
            assert record["status"] in {"leased", "finalizing"}
            assert str(record.get("lease_id") or "") == lease_id
            assert str(record.get("attempt_id") or "") == str(lease.get("attempt_id") or "")
            assert int(record.get("scheduler_epoch") or 0) == int(lease.get("scheduler_epoch") or 0)

        active_records = self.task_store.list_active_tasks(limit=1000) if self.task_store is not None else []
        assigned_count_by_queue: dict[str, int] = {}
        for record in active_records:
            if str(record.get("status") or "") != "assigned":
                continue
            queue_id = str(record.get("subqueue_id") or "")
            assigned_count_by_queue[queue_id] = assigned_count_by_queue.get(queue_id, 0) + 1

        replica_queues = stats.get("replica_queues") or {}
        assert isinstance(replica_queues, dict)
        for queue_id, queue in replica_queues.items():
            assert isinstance(queue, dict)
            depth = int(queue.get("depth") or 0)
            assert assigned_count_by_queue.get(str(queue_id), 0) == depth
