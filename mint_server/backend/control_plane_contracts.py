from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class TaskLedgerOp(StrEnum):
    ACQUIRE_OWNER = "acquire_owner"
    RENEW_OWNER = "renew_owner"
    CREATE_TASK = "create_task"
    ASSIGN_TASK = "assign_task"
    CLAIM_TASK = "claim_task"
    RENEW_LEASE = "renew_lease"
    BEGIN_FINALIZE = "begin_finalize"
    COMMIT_FINALIZE_SUCCESS = "commit_finalize_success"
    COMMIT_FINALIZE_FAILURE = "commit_finalize_failure"
    REQUEUE_TASK = "requeue_task"
    FORGET_TASK = "forget_task"
    GET_TASK = "get_task"
    LIST_ACTIVE_TASKS = "list_active_tasks"
    WAIT_TASK_STATUS_CHANGE = "wait_task_status_change"
    UPDATE_TASK_METADATA = "update_task_metadata"


class SchedulerQueueOp(StrEnum):
    APPEND = "append"
    SYNC_REPLICAS = "sync_replicas"
    ASSIGN_PENDING = "assign_pending"
    CLAIM = "claim"
    RENEW = "renew"
    BEGIN_FINALIZE = "begin_finalize"
    COMPLETE = "complete"
    FAIL = "fail"
    VALIDATE = "validate"
    EXPIRE = "expire"
    CONTAINS = "contains"
    STATS = "stats"


@runtime_checkable
class AsyncTaskLedger(Protocol):
    async def ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]: ...

    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]: ...

    async def acquire_owner(
        self,
        *,
        owner_id: str,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def renew_owner(
        self,
        *,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def create_task(
        self,
        *,
        request_id: str,
        op: str,
        domain_key: str,
        request_json: bytes,
        payload_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def assign_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        scheduler_epoch: int,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def claim_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        lease_id: str,
        attempt_id: str,
        consumer_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        lease_ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def renew_lease(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        lease_ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def begin_finalize(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        finalize_ttl_s: float,
        staged_payload_path: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def commit_finalize_success(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        result_path: str,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def commit_finalize_failure(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        error: str,
        result_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def requeue_task(
        self,
        *,
        request_id: str,
        scheduler_epoch: int,
        reason: str,
        now: float | None = None,
    ) -> dict[str, Any]: ...

    async def forget_task(self, *, request_id: str) -> dict[str, Any]: ...

    async def get_task(self, *, request_id: str) -> dict[str, Any]: ...

    async def list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]: ...

    async def wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> dict[str, Any]: ...

    async def update_task_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class AsyncSchedulerQueue(Protocol):
    async def append_work(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        domain_key: str,
        user_id: str | None = None,
        apikey_id: str | None = None,
        throttle_principal: str | None = None,
        webhook_url: str | None = None,
        extra: dict[str, Any] | None = None,
        created_at: float | None = None,
        affinity_group: str | None = None,
        ordering_key: str | None = None,
        token_cost: int = 1,
        assign: bool = False,
        assign_max_items: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def sync_replicas(self, replicas: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    async def assign_pending(
        self,
        *,
        max_items: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def claim(
        self,
        *,
        domain_key: str,
        replica_id: str,
        consumer_id: str,
        consumer_generation: int,
        max_items: int = 1,
        token_budget: int | None = None,
        lease_ttl_s: float = 30.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def renew(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float = 30.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def begin_finalize(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def complete(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def fail(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        requeue: bool = True,
        reason: str = "failed",
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def validate(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def expire(
        self,
        *,
        now: float | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    async def contains(self, *, request_id: str, timeout_s: float | None = None) -> dict[str, Any]: ...

    async def stats(self, **kwargs: Any) -> dict[str, Any]: ...


class TaskStateStoreLedgerAdapter:
    """Async task-ledger contract over TaskStateStoreClient-like objects."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def _call_dict(self, method: str, **kwargs: Any) -> dict[str, Any]:
        out = await self._call(method, **kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"TaskLedger.{method} returned non-dict: {type(out)}")
        return out

    async def _call_list(self, method: str, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call(method, **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskLedger.{method} returned non-list: {type(out)}")
        return out

    async def _call(self, method: str, **kwargs: Any) -> Any:
        async_method = getattr(self._client, f"async_{method}", None)
        if callable(async_method):
            return await async_method(**kwargs)
        raise AttributeError(f"Task ledger client missing async_{method}")

    async def ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "ensure_ready",
            timeout_s=timeout_s,
            create_if_missing=create_if_missing,
        )

    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self._call_dict("ping", timeout_s=timeout_s)

    async def acquire_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("acquire_owner", **kwargs)

    async def renew_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("renew_owner", **kwargs)

    async def create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("create_task", **kwargs)

    async def assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("assign_task", **kwargs)

    async def claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("claim_task", **kwargs)

    async def renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("renew_lease", **kwargs)

    async def begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("begin_finalize", **kwargs)

    async def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("commit_finalize_success", **kwargs)

    async def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("commit_finalize_failure", **kwargs)

    async def requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("requeue_task", **kwargs)

    async def forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("forget_task", **kwargs)

    async def get_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("get_task", **kwargs)

    async def list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call_list("list_active_tasks", **kwargs)

    async def wait_task_status_change(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("wait_task_status_change", **kwargs)

    async def update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("update_task_metadata", **kwargs)


_ASYNC_TASK_LEDGER_METHODS = (
    "ensure_ready",
    "ping",
    "acquire_owner",
    "renew_owner",
    "create_task",
    "assign_task",
    "claim_task",
    "renew_lease",
    "begin_finalize",
    "commit_finalize_success",
    "commit_finalize_failure",
    "requeue_task",
    "forget_task",
    "get_task",
    "list_active_tasks",
    "wait_task_status_change",
    "update_task_metadata",
)


def _has_methods(client: Any, names: tuple[str, ...]) -> bool:
    return all(callable(getattr(client, name, None)) for name in names)


def as_task_ledger(client: Any) -> AsyncTaskLedger:
    if _has_methods(client, _ASYNC_TASK_LEDGER_METHODS):
        return client
    missing_async = [name for name in _ASYNC_TASK_LEDGER_METHODS if not callable(getattr(client, name, None))]
    has_async_client_surface = any(callable(getattr(client, f"async_{name}", None)) for name in _ASYNC_TASK_LEDGER_METHODS)
    if not has_async_client_surface:
        raise TypeError(
            "task ledger client does not implement AsyncTaskLedger; "
            f"missing_async={missing_async[:5]}"
        )
    return TaskStateStoreLedgerAdapter(client)


class InProcessSchedulerQueueAdapter:
    """Async scheduler contract over an in-process scheduler actor implementation."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def append_work(self, **kwargs: Any) -> dict[str, Any]:
        item = {
            "request_id": kwargs["request_id"],
            "op": kwargs["op"],
            "request_json": kwargs["request_json"],
            "user_id": kwargs.get("user_id"),
            "apikey_id": kwargs.get("apikey_id"),
            "throttle_principal": kwargs.get("throttle_principal"),
            "webhook_url": kwargs.get("webhook_url"),
            "extra": dict(kwargs.get("extra") or {}),
            "created_at": kwargs.get("created_at", time.time()),
            "domain_key": kwargs["domain_key"],
            "affinity_group": kwargs.get("affinity_group"),
            "ordering_key": kwargs.get("ordering_key"),
            "token_cost": kwargs.get("token_cost", 1),
        }
        return await self.actor.append(
            item,
            assign=bool(kwargs.get("assign", False)),
            assign_max_items=kwargs.get("assign_max_items"),
        )

    async def sync_replicas(self, replicas: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
        return await self.actor.sync_replicas(replicas)

    async def assign_pending(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.assign_pending(**kwargs)

    async def claim(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.claim_from_replica_queue(**kwargs)

    async def renew(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.renew_lease(**kwargs)

    async def begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.begin_finalize_lease(**kwargs)

    async def complete(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.complete_lease(**kwargs)

    async def fail(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.fail_lease(**kwargs)

    async def validate(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.validate_lease(**kwargs)

    async def expire(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.expire_leases(**kwargs)

    async def contains(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.contains_request(**kwargs)

    async def stats(self, **_kwargs: Any) -> dict[str, Any]:
        return self.actor.stats()
