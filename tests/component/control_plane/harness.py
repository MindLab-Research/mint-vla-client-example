from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

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
from mint_server.backend.task_state_store import TaskFutureService, TaskStateStore
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
        return getattr(self.store, method)(*args, **kwargs)

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
        return self.store.ping()

    async def async_wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        _ = terminal_only
        await asyncio.sleep(max(0.0, min(float(timeout_s), 0.01)))
        try:
            return {
                "changed": False,
                "timeout": True,
                "missing": False,
                "record": self.store.get_task(str(request_id)),
            }
        except KeyError:
            return {"changed": False, "timeout": False, "missing": True}

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("async_future_"):
            method = name[len("async_future_") :]
        elif name.startswith("async_"):
            method = name[len("async_") :]
        else:
            raise AttributeError(name)
        if method == "wait_task_status_change":
            return self.async_wait_task_status_change

        async def _method(*args: Any, **kwargs: Any) -> Any:
            return await self._call(method, *args, **kwargs)

        return _method


class SchedulerClient:
    def __init__(self, actor: _ModelWorkSchedulerActor) -> None:
        self.actor = actor

    async def append(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        user_id: str | None,
        apikey_id: str | None,
        throttle_principal: str | None,
        webhook_url: str | None,
        domain_key: str,
        affinity_group: str | None = None,
        ordering_key: str | None = None,
        token_cost: int = 1,
        extra: dict[str, Any] | None = None,
        assign: bool = False,
        assign_max_items: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        item = {
            "request_id": request_id,
            "op": op,
            "request_json": request_json,
            "user_id": user_id,
            "apikey_id": apikey_id,
            "throttle_principal": throttle_principal,
            "webhook_url": webhook_url,
            "extra": dict(extra or {}),
            "created_at": time.time(),
            "domain_key": domain_key,
            "affinity_group": affinity_group,
            "ordering_key": ordering_key,
            "token_cost": token_cost,
        }
        return await self.actor.append(
            item,
            assign=assign,
            assign_max_items=assign_max_items,
        )

    async def sync_replicas(self, replicas: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
        return await self.actor.sync_replicas(replicas)

    async def assign_pending(self, *, max_items: int | None = None, **_kwargs: Any) -> dict[str, Any]:
        return await self.actor.assign_pending(max_items=max_items)

    async def claim_from_replica_queue(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.claim_from_replica_queue(**kwargs)

    async def renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.renew_lease(**kwargs)

    async def begin_finalize_lease(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.begin_finalize_lease(**kwargs)

    async def complete_lease(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.complete_lease(**kwargs)

    async def fail_lease(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.fail_lease(**kwargs)

    async def validate_lease(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.validate_lease(**kwargs)

    async def contains_request(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.contains_request(**kwargs)

    async def expire_leases(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.expire_leases(**kwargs)

    async def stats(self, **_kwargs: Any) -> dict[str, Any]:
        return self.actor.stats()


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

    def __post_init__(self) -> None:
        self.faults = FaultController()
        self.task_store = TaskStateStore.in_memory()
        self.task_state = LocalAsyncTaskStateClient(self.task_store, self.faults)
        self.payload_store = FaultingTaskPayloadStore(
            root_dir=self.tmp_path / "payloads",
            faults=self.faults,
        )
        self.future_service = TaskFutureService(
            task_state_client=self.task_state,
            payload_store=self.payload_store,
        )
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id="component-scheduler",
        )
        self.scheduler = SchedulerClient(self.scheduler_actor)
        self.event_log: list[tuple[str, dict[str, Any]]] = []

    def replace_scheduler(self, *, owner_id: str) -> None:
        self.scheduler_actor = _ModelWorkSchedulerActor(
            use_task_state_store=True,
            task_state_store=self.task_state,
            owner_id=owner_id,
        )
        self.scheduler = SchedulerClient(self.scheduler_actor)

    @property
    def consumer_id(self) -> str:
        return consumer_id_for_replica(self.domain_key, self.replica_id, self.generation)

    def replica(self, *, status: str = "healthy", generation: int | None = None) -> dict[str, Any]:
        generation = self.generation if generation is None else int(generation)
        return {
            "domain_key": self.domain_key,
            "replica_id": self.replica_id,
            "consumer_id": consumer_id_for_replica(self.domain_key, self.replica_id, generation),
            "generation": generation,
            "status": status,
            "queue_id": queue_id_for_replica(self.domain_key, self.replica_id),
            "capacity": 4,
            "actor_name": f"component-runtime-{self.replica_id}",
            "node_pins": ["127.0.0.1"],
            "updated_at": time.time(),
        }

    async def start(self) -> None:
        await self.scheduler.sync_replicas([self.replica(status="healthy")])

    async def enqueue_sampling(self, request_id: str, *, assign: bool = True) -> dict[str, Any]:
        return await enqueue_model_work(
            request_id=request_id,
            op="sampling.asample",
            request_json=b'{"prompt":"hello"}',
            domain_key=self.domain_key,
            queued_meta=sampling_meta(self.domain_key),
            user_id="user-a",
            apikey_id="key-a",
            throttle_principal="apikey:key-a",
            webhook_url=None,
            affinity_group="lora:session-a:generation:1",
            ordering_key="session:session-a",
            token_cost=1,
            assign=assign,
            assign_max_items=1,
            scheduler_client=self.scheduler,
        )

    async def claim_one(self, *, lease_ttl_s: float = 30.0) -> dict[str, Any]:
        claimed = await self.scheduler.claim_from_replica_queue(
            domain_key=self.domain_key,
            replica_id=self.replica_id,
            consumer_id=self.consumer_id,
            consumer_generation=self.generation,
            max_items=1,
            lease_ttl_s=lease_ttl_s,
        )
        leases = claimed.get("leases") if isinstance(claimed, dict) else None
        assert isinstance(leases, list)
        assert len(leases) == 1
        return leases[0]

    async def claim_none(self, *, lease_ttl_s: float = 30.0) -> dict[str, Any]:
        claimed = await self.scheduler.claim_from_replica_queue(
            domain_key=self.domain_key,
            replica_id=self.replica_id,
            consumer_id=self.consumer_id,
            consumer_generation=self.generation,
            max_items=1,
            lease_ttl_s=lease_ttl_s,
        )
        assert claimed.get("leases") == []
        return claimed

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

    async def retrieve(self, request_id: str, monkeypatch: Any) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(futures_route, "task_futures", self.future_service)
        import mint_server.backend.model_work_scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "model_work_scheduler", self.scheduler)
        monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: 0.0)
        response = SimpleNamespace(status_code=200, headers={})
        request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})
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

    def close(self) -> None:
        self.task_store.close()
