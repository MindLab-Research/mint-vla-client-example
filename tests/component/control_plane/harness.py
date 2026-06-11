from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from mint_server.backend.control_plane_contracts import (
    InProcessSchedulerQueueAdapter,
)
from mint_server.backend.model_actor_supervisor import (
    ModelActorSpec,
    ModelActorSupervisorCore,
    consumer_id_for_replica,
    queue_id_for_replica,
)
from mint_server.backend.model_runtime_actor import ModelRuntimeActor
from mint_server.backend.model_work_admission import enqueue_model_work
from mint_server.backend.model_work_scheduler import _ModelWorkSchedulerActor
from mint_server.backend.task_payload_store import TaskPayloadStore
from mint_server.backend.task_state_store import FutureStatus, TaskFutureService, TaskStateStore
from mint_server.models.types import FutureRetrieveRequest
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
            task_state_client=self.task_state,
            future_state_client=self.task_state,
            payload_store=self.payload_store,
        )
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id="component-scheduler",
        )
        self.scheduler = InProcessSchedulerQueueAdapter(self.scheduler_actor)
        self.event_log: list[tuple[str, dict[str, Any]]] = []

    def replace_scheduler(self, *, owner_id: str) -> None:
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id=owner_id,
        )
        self.scheduler = InProcessSchedulerQueueAdapter(self.scheduler_actor)

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
    ) -> dict[str, Any]:
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
            scheduler_client=self.scheduler,
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
        claimed = await self.scheduler.claim(
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

    async def claim_none(self, *, lease_ttl_s: float = 30.0) -> dict[str, Any]:
        claimed = await self.scheduler.claim(
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
        executor: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        lease_ttl_s: float = 1.0,
    ) -> ModelRuntimeActor:
        async def _default_executor(lease: dict[str, Any]) -> None:
            request_id = str(lease["item"]["request_id"])
            await self.future_service.async_resolve(request_id, {"ok": True, "request_id": request_id})

        actor = ModelRuntimeActor(
            domain_key=self.domain_key,
            replica_id=self.replica_id,
            actor_name=f"component-runtime-{self.replica_id}",
            actor_generation=self.generation,
            poll_interval_s=0.01,
            lease_ttl_s=lease_ttl_s,
            max_claim=1,
            scheduler_client=self.scheduler,
            task_futures_client=self.future_service,
            task_state_store_client=self.task_state,
            payload_store=self.payload_store,
            executor=executor or _default_executor,
        )
        await actor.run_once()
        return actor

    async def observe_task(self, request_id: str) -> dict[str, Any]:
        return await self.task_state.async_get_task(request_id=str(request_id))

    async def observe_scheduler(self, request_id: str) -> dict[str, Any]:
        return await self.scheduler.contains(request_id=str(request_id))

    async def observe_future_status(self, request_id: str) -> FutureStatus:
        return await self.future_service.async_get_status(str(request_id))

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

        monkeypatch.setattr(scheduler_module, "model_work_scheduler", scheduler_override or self.scheduler)
        monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: float(wait_timeout_s))
        monkeypatch.setattr(futures_route, "_is_privileged", lambda _request: bool(admin))
        response = SimpleNamespace(status_code=200, headers={})
        user_id = "admin" if admin else "user-a"
        request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": user_id}), headers={})
        payload = await futures_route.retrieve_future(
            FutureRetrieveRequest(request_id=request_id),
            request,
            response,
        )
        return int(response.status_code), payload

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

        async def _sync(registrations: list[Any]) -> dict[str, Any]:
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
        async def _sync(registrations: list[Any]) -> dict[str, Any]:
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
        self.task_store.close()
