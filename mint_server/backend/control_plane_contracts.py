from __future__ import annotations

import asyncio
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

    async def acquire_owner(self, **kwargs: Any) -> dict[str, Any]: ...

    async def renew_owner(self, **kwargs: Any) -> dict[str, Any]: ...

    async def create_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def assign_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def claim_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def renew_lease(self, **kwargs: Any) -> dict[str, Any]: ...

    async def begin_finalize(self, **kwargs: Any) -> dict[str, Any]: ...

    async def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]: ...

    async def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]: ...

    async def requeue_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def forget_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def get_task(self, **kwargs: Any) -> dict[str, Any]: ...

    async def list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    async def wait_task_status_change(self, **kwargs: Any) -> dict[str, Any]: ...

    async def update_task_metadata(self, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class AsyncSchedulerQueue(Protocol):
    async def append_work(self, **kwargs: Any) -> dict[str, Any]: ...

    async def sync_replicas(self, replicas: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    async def assign_pending(self, **kwargs: Any) -> dict[str, Any]: ...

    async def claim(self, **kwargs: Any) -> dict[str, Any]: ...

    async def renew(self, **kwargs: Any) -> dict[str, Any]: ...

    async def begin_finalize(self, **kwargs: Any) -> dict[str, Any]: ...

    async def complete(self, **kwargs: Any) -> dict[str, Any]: ...

    async def fail(self, **kwargs: Any) -> dict[str, Any]: ...

    async def validate(self, **kwargs: Any) -> dict[str, Any]: ...

    async def expire(self, **kwargs: Any) -> dict[str, Any]: ...

    async def contains(self, **kwargs: Any) -> dict[str, Any]: ...

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


class InProcessTaskLedgerAdapter:
    """Async task-ledger contract over an in-process TaskStateStore implementation."""

    def __init__(self, store: Any) -> None:
        self._store = store

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
        sync_method = getattr(self._store, method)
        return await asyncio.to_thread(sync_method, **kwargs)

    async def ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        _ = timeout_s, create_if_missing
        return await self.ping()

    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        _ = timeout_s
        return await self._call_dict("ping")

    async def acquire_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("acquire_scheduler_owner", **kwargs)

    async def renew_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call_dict("renew_scheduler_owner", **kwargs)

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


def as_task_ledger(client: Any) -> AsyncTaskLedger:
    if callable(getattr(client, "acquire_owner", None)) and callable(
        getattr(client, "commit_finalize_success", None)
    ):
        return client
    if callable(getattr(client, "acquire_scheduler_owner", None)) and callable(
        getattr(client, "commit_finalize_success", None)
    ):
        return InProcessTaskLedgerAdapter(client)
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
