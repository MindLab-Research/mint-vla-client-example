from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable


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


class ConflictReason(StrEnum):
    ADMISSION_REJECTED = "admission_rejected"
    ALREADY_TERMINAL = "already_terminal"
    CANCELLED_AFTER_REQUEUE_COMMIT = "cancelled_after_requeue_commit"
    DOMAIN_INFLIGHT_LIMIT_EXCEEDED = "domain_inflight_limit_exceeded"
    DOMAIN_TOKEN_BUDGET_EXCEEDED = "domain_token_budget_exceeded"
    DUPLICATE_MISMATCH = "duplicate_mismatch"
    DUPLICATE_REQUEST_ID = "duplicate_request_id"
    FINALIZE_IN_PROGRESS = "finalize_in_progress"
    FINALIZE_INFLIGHT = "finalize_inflight"
    LEASE_EXPIRED = "lease_expired"
    NOT_FINALIZING = "not_finalizing"
    NOT_FOUND = "not_found"
    NOT_PENDING = "not_pending"
    NOT_TERMINAL = "not_terminal"
    OWNER_ACTIVE = "owner_active"
    PAYLOAD_CHANGED = "payload_changed"
    PRINCIPAL_DOMAIN_INFLIGHT_LIMIT_EXCEEDED = "principal_domain_inflight_limit_exceeded"
    PRINCIPAL_DOMAIN_TOKEN_BUDGET_EXCEEDED = "principal_domain_token_budget_exceeded"
    RETRY_REQUIRED = "retry_required"
    STALE_CONSUMER = "stale_consumer"
    STALE_EPOCH = "stale_epoch"
    STALE_GENERATION = "stale_generation"
    STALE_OWNER = "stale_owner"
    TASK_STATE_INVALID = "task_state_invalid"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"
    UNKNOWN_LEASE = "unknown_lease"


def _reason_from_wire(value: Any) -> ConflictReason | None:
    if value is None:
        return None
    if isinstance(value, ConflictReason):
        return value
    try:
        return ConflictReason(str(value))
    except ValueError:
        return ConflictReason.UNKNOWN


@dataclass(frozen=True)
class LeaseToken:
    request_id: str
    lease_id: str
    attempt_id: str
    scheduler_epoch: int
    consumer_id: str
    consumer_generation: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "lease_id": self.lease_id,
            "attempt_id": self.attempt_id,
            "scheduler_epoch": self.scheduler_epoch,
            "consumer_id": self.consumer_id,
            "consumer_generation": self.consumer_generation,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "LeaseToken":
        return cls(
            request_id=str(data["request_id"]),
            lease_id=str(data["lease_id"]),
            attempt_id=str(data["attempt_id"]),
            scheduler_epoch=int(data["scheduler_epoch"]),
            consumer_id=str(data["consumer_id"]),
            consumer_generation=int(data["consumer_generation"]),
        )


@dataclass(frozen=True)
class SubmitTaskResult:
    ok: bool
    request_id: str
    created: bool = False
    assigned: bool = False
    reason: ConflictReason | None = None
    record: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "created": self.created,
            "assigned": self.assigned,
            "reason": self.reason.value if self.reason is not None else None,
            "record": self.record,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "SubmitTaskResult":
        assigned = data.get("assigned", False)
        if isinstance(assigned, dict):
            assigned = bool(assigned.get("assigned"))
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]),
            created=bool(data.get("created", not bool(data.get("idempotent", False)))),
            assigned=bool(assigned),
            reason=_reason_from_wire(data.get("reason")),
            record=data.get("record") if isinstance(data.get("record"), dict) else None,
        )


@dataclass(frozen=True)
class CancelTaskResult:
    ok: bool
    request_id: str
    was_terminal: bool = False
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "was_terminal": self.was_terminal,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "CancelTaskResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]),
            was_terminal=bool(data.get("was_terminal", False)),
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class RetrieveTaskResult:
    status: Literal["ready", "failed", "pending", "unknown", "unavailable"]
    request_id: str
    result_path: str | None = None
    result_checksum: str | None = None
    result_size_bytes: int | None = None
    error: dict[str, Any] | None = None
    retry_after_s: float | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "result_path": self.result_path,
            "result_checksum": self.result_checksum,
            "result_size_bytes": self.result_size_bytes,
            "error": self.error,
            "retry_after_s": self.retry_after_s,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "RetrieveTaskResult":
        return cls(
            status=data["status"],
            request_id=str(data["request_id"]),
            result_path=data.get("result_path"),
            result_checksum=data.get("result_checksum"),
            result_size_bytes=(
                int(data["result_size_bytes"]) if data.get("result_size_bytes") is not None else None
            ),
            error=data.get("error") if isinstance(data.get("error"), dict) else None,
            retry_after_s=float(data["retry_after_s"]) if data.get("retry_after_s") is not None else None,
        )


@dataclass(frozen=True)
class TaskRecord:
    request_id: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return dict(self.data)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "TaskRecord":
        return cls(
            request_id=str(data["request_id"]),
            status=str(data["status"]),
            data=dict(data),
        )


@dataclass(frozen=True)
class OwnerLeaseResult:
    ok: bool
    owner_id: str | None = None
    epoch: int | None = None
    expires_at: float | None = None
    fencing_token: str | None = None
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "owner_id": self.owner_id,
            "epoch": self.epoch,
            "expires_at": self.expires_at,
            "fencing_token": self.fencing_token,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "OwnerLeaseResult":
        return cls(
            ok=bool(data["ok"]),
            owner_id=str(data["owner_id"]) if data.get("owner_id") is not None else None,
            epoch=int(data["epoch"]) if data.get("epoch") is not None else None,
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            fencing_token=str(data["fencing_token"]) if data.get("fencing_token") is not None else None,
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class CreateTaskResult:
    ok: bool
    created: bool
    record: dict[str, Any]
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "created": self.created,
            "record": self.record,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "CreateTaskResult":
        return cls(
            ok=bool(data["ok"]),
            created=bool(data["created"]),
            record=dict(data["record"]),
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class TaskMutationResult:
    ok: bool
    record: dict[str, Any] | None = None
    reason: ConflictReason | None = None
    idempotent: bool = False
    retry_required: bool = False
    deleted: bool | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "record": self.record,
            "reason": self.reason.value if self.reason is not None else None,
            "idempotent": self.idempotent,
            "retry_required": self.retry_required,
            "deleted": self.deleted,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "TaskMutationResult":
        return cls(
            ok=bool(data["ok"]),
            record=data.get("record") if isinstance(data.get("record"), dict) else None,
            reason=_reason_from_wire(data.get("reason")),
            idempotent=bool(data.get("idempotent", False)),
            retry_required=bool(data.get("retry_required", False)),
            deleted=bool(data["deleted"]) if data.get("deleted") is not None else None,
        )


AssignTaskResult = TaskMutationResult
ClaimTaskResult = TaskMutationResult
RenewLeaseResult = TaskMutationResult
BeginFinalizeResult = TaskMutationResult
CommitFinalizeResult = TaskMutationResult
RequeueTaskResult = TaskMutationResult


@dataclass(frozen=True)
class TaskStatusChange:
    changed: bool
    request_id: str
    record: dict[str, Any] | None = None
    missing: bool = False
    timeout: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "request_id": self.request_id,
            "record": self.record,
            "missing": self.missing,
            "timeout": self.timeout,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "TaskStatusChange":
        record = data.get("record") if isinstance(data.get("record"), dict) else None
        request_id = str(data.get("request_id") or (record or {}).get("request_id"))
        return cls(
            changed=bool(data["changed"]),
            request_id=request_id,
            record=record,
            missing=bool(data.get("missing", False)),
            timeout=bool(data.get("timeout", False)),
        )


@dataclass(frozen=True)
class AppendWorkResult:
    ok: bool
    request_id: str | None = None
    domain_key: str | None = None
    scheduler_instance_id: str | None = None
    backlog_depth: int | None = None
    assigned: dict[str, Any] | None = None
    idempotent: bool = False
    reason: ConflictReason | None = None
    retry_after_s: float | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "domain_key": self.domain_key,
            "scheduler_instance_id": self.scheduler_instance_id,
            "backlog_depth": self.backlog_depth,
            "assigned": self.assigned,
            "idempotent": self.idempotent,
            "reason": self.reason.value if self.reason is not None else None,
            "retry_after_s": self.retry_after_s,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "AppendWorkResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]) if data.get("request_id") is not None else None,
            domain_key=str(data["domain_key"]) if data.get("domain_key") is not None else None,
            scheduler_instance_id=(
                str(data["scheduler_instance_id"]) if data.get("scheduler_instance_id") is not None else None
            ),
            backlog_depth=int(data["backlog_depth"]) if data.get("backlog_depth") is not None else None,
            assigned=data.get("assigned") if isinstance(data.get("assigned"), dict) else None,
            idempotent=bool(data.get("idempotent", False)),
            reason=_reason_from_wire(data.get("reason")),
            retry_after_s=float(data["retry_after_s"]) if data.get("retry_after_s") is not None else None,
        )


@dataclass(frozen=True)
class AssignPendingResult:
    ok: bool
    assigned: int
    skipped_domains: list[str] = field(default_factory=list)
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "assigned": self.assigned,
            "skipped_domains": list(self.skipped_domains),
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "AssignPendingResult":
        return cls(
            ok=bool(data["ok"]),
            assigned=int(data.get("assigned") or 0),
            skipped_domains=[str(value) for value in data.get("skipped_domains") or []],
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class ClaimResult:
    ok: bool
    leases: list[dict[str, Any]] = field(default_factory=list)
    remaining_queue_depth: int = 0
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "leases": list(self.leases),
            "remaining_queue_depth": self.remaining_queue_depth,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ClaimResult":
        return cls(
            ok=bool(data["ok"]),
            leases=list(data.get("leases") or []),
            remaining_queue_depth=int(data.get("remaining_queue_depth") or 0),
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class LeaseResult:
    ok: bool
    lease: dict[str, Any] | None = None
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "lease": self.lease,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "LeaseResult":
        return cls(
            ok=bool(data["ok"]),
            lease=data.get("lease") if isinstance(data.get("lease"), dict) else None,
            reason=_reason_from_wire(data.get("reason")),
        )


RenewResult = LeaseResult


@dataclass(frozen=True)
class FinishResult:
    ok: bool
    request_id: str | None = None
    status: str | None = None
    reason: ConflictReason | None = None
    idempotent: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "status": self.status,
            "reason": self.reason.value if self.reason is not None else None,
            "idempotent": self.idempotent,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "FinishResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]) if data.get("request_id") is not None else None,
            status=str(data["status"]) if data.get("status") is not None else None,
            reason=_reason_from_wire(data.get("reason")),
            idempotent=bool(data.get("idempotent", False)),
        )


@dataclass(frozen=True)
class FailLeaseResult:
    ok: bool
    request_id: str | None = None
    requeued: bool = False
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "requeued": self.requeued,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "FailLeaseResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]) if data.get("request_id") is not None else None,
            requeued=bool(data.get("requeued", False)),
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class ValidateLeaseResult:
    ok: bool
    request_id: str | None = None
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ValidateLeaseResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]) if data.get("request_id") is not None else None,
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class ExpireResult:
    ok: bool
    expired: int

    def to_wire(self) -> dict[str, Any]:
        return {"ok": self.ok, "expired": self.expired}

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ExpireResult":
        return cls(ok=bool(data["ok"]), expired=int(data.get("expired") or 0))


@dataclass(frozen=True)
class ContainsResult:
    ok: bool
    request_id: str
    present: bool
    location: str | None = None
    lease_id: str | None = None
    scheduler_instance_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "present": self.present,
            "location": self.location,
            "lease_id": self.lease_id,
            "scheduler_instance_id": self.scheduler_instance_id,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ContainsResult":
        return cls(
            ok=bool(data["ok"]),
            request_id=str(data["request_id"]),
            present=bool(data["present"]),
            location=str(data["location"]) if data.get("location") is not None else None,
            lease_id=str(data["lease_id"]) if data.get("lease_id") is not None else None,
            scheduler_instance_id=(
                str(data["scheduler_instance_id"]) if data.get("scheduler_instance_id") is not None else None
            ),
        )


@dataclass(frozen=True)
class SyncReplicasResult:
    ok: bool
    registered: int = 0
    removed: int = 0
    expired: int = 0
    assigned: dict[str, Any] | None = None
    reason: ConflictReason | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "registered": self.registered,
            "removed": self.removed,
            "expired": self.expired,
            "assigned": self.assigned,
            "reason": self.reason.value if self.reason is not None else None,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "SyncReplicasResult":
        return cls(
            ok=bool(data["ok"]),
            registered=int(data.get("registered") or 0),
            removed=int(data.get("removed") or 0),
            expired=int(data.get("expired") or 0),
            assigned=data.get("assigned") if isinstance(data.get("assigned"), dict) else None,
            reason=_reason_from_wire(data.get("reason")),
        )


@dataclass(frozen=True)
class ExecutorOutcome:
    kind: Literal["success", "retryable_failure", "fatal_backend_death", "user_error"]
    payload: Any | None = None
    error: str | None = None
    billing_observations: list[dict[str, Any]] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "error": self.error,
            "billing_observations": self.billing_observations,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ExecutorOutcome":
        return cls(
            kind=data["kind"],
            payload=data.get("payload"),
            error=data.get("error"),
            billing_observations=(
                list(data["billing_observations"])
                if data.get("billing_observations") is not None
                else None
            ),
        )


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
    ) -> OwnerLeaseResult: ...

    async def renew_owner(
        self,
        *,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        now: float | None = None,
    ) -> OwnerLeaseResult: ...

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
    ) -> CreateTaskResult: ...

    async def assign_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        scheduler_epoch: int,
        now: float | None = None,
    ) -> AssignTaskResult: ...

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
    ) -> ClaimTaskResult: ...

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
    ) -> RenewLeaseResult: ...

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
    ) -> BeginFinalizeResult: ...

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
        billing_observations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> CommitFinalizeResult: ...

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
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> CommitFinalizeResult: ...

    async def complete_task_failure(
        self,
        *,
        request_id: str,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> CommitFinalizeResult: ...

    async def requeue_task(
        self,
        *,
        request_id: str,
        scheduler_epoch: int,
        reason: str,
        now: float | None = None,
    ) -> RequeueTaskResult: ...

    async def forget_task(self, *, request_id: str) -> TaskMutationResult: ...

    async def get_task(self, *, request_id: str) -> TaskRecord: ...

    async def list_active_tasks(self, *, limit: int | None = None) -> list[TaskRecord]: ...

    async def wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> TaskStatusChange: ...

    async def update_task_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any],
        now: float | None = None,
    ) -> TaskMutationResult: ...


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
    ) -> AppendWorkResult: ...

    async def sync_replicas(self, replicas: list[dict[str, Any]], **kwargs: Any) -> SyncReplicasResult: ...

    async def assign_pending(
        self,
        *,
        max_items: int | None = None,
        timeout_s: float | None = None,
    ) -> AssignPendingResult: ...

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
    ) -> ClaimResult: ...

    async def renew(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        lease_ttl_s: float = 30.0,
        timeout_s: float | None = None,
    ) -> RenewResult: ...

    async def begin_finalize(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        finalize_ttl_s: float = 30.0,
        staged_payload_path: str | None = None,
        timeout_s: float | None = None,
    ) -> LeaseResult: ...

    async def complete(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float | None = None,
    ) -> FinishResult: ...

    async def finish_success(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        consumer_id: str,
        consumer_generation: int,
        result_path: str,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
        timeout_s: float | None = None,
    ) -> FinishResult: ...

    async def finish_failure(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        consumer_id: str,
        consumer_generation: int,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        timeout_s: float | None = None,
    ) -> FinishResult: ...

    async def fail(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        requeue: bool = True,
        reason: str = "failed",
        abort_finalize: bool = False,
        timeout_s: float | None = None,
    ) -> FailLeaseResult: ...

    async def validate(
        self,
        *,
        lease_id: str,
        consumer_id: str,
        consumer_generation: int,
        timeout_s: float | None = None,
    ) -> ValidateLeaseResult: ...

    async def expire(
        self,
        *,
        now: float | None = None,
        timeout_s: float | None = None,
    ) -> ExpireResult: ...

    async def contains(self, *, request_id: str, timeout_s: float | None = None) -> ContainsResult: ...

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

    async def acquire_owner(self, **kwargs: Any) -> OwnerLeaseResult:
        return OwnerLeaseResult.from_wire(await self._call_dict("acquire_owner", **kwargs))

    async def renew_owner(self, **kwargs: Any) -> OwnerLeaseResult:
        return OwnerLeaseResult.from_wire(await self._call_dict("renew_owner", **kwargs))

    async def create_task(self, **kwargs: Any) -> CreateTaskResult:
        return CreateTaskResult.from_wire(await self._call_dict("create_task", **kwargs))

    async def assign_task(self, **kwargs: Any) -> AssignTaskResult:
        return AssignTaskResult.from_wire(await self._call_dict("assign_task", **kwargs))

    async def claim_task(self, **kwargs: Any) -> ClaimTaskResult:
        return ClaimTaskResult.from_wire(await self._call_dict("claim_task", **kwargs))

    async def renew_lease(self, **kwargs: Any) -> RenewLeaseResult:
        return RenewLeaseResult.from_wire(await self._call_dict("renew_lease", **kwargs))

    async def begin_finalize(self, **kwargs: Any) -> BeginFinalizeResult:
        return BeginFinalizeResult.from_wire(await self._call_dict("begin_finalize", **kwargs))

    async def commit_finalize_success(self, **kwargs: Any) -> CommitFinalizeResult:
        return CommitFinalizeResult.from_wire(await self._call_dict("commit_finalize_success", **kwargs))

    async def commit_finalize_failure(self, **kwargs: Any) -> CommitFinalizeResult:
        return CommitFinalizeResult.from_wire(await self._call_dict("commit_finalize_failure", **kwargs))

    async def complete_task_failure(self, **kwargs: Any) -> CommitFinalizeResult:
        return CommitFinalizeResult.from_wire(await self._call_dict("complete_task_failure", **kwargs))

    async def requeue_task(self, **kwargs: Any) -> RequeueTaskResult:
        return RequeueTaskResult.from_wire(await self._call_dict("requeue_task", **kwargs))

    async def forget_task(self, **kwargs: Any) -> TaskMutationResult:
        return TaskMutationResult.from_wire(await self._call_dict("forget_task", **kwargs))

    async def get_task(self, **kwargs: Any) -> TaskRecord:
        return TaskRecord.from_wire(await self._call_dict("get_task", **kwargs))

    async def list_active_tasks(self, **kwargs: Any) -> list[TaskRecord]:
        return [TaskRecord.from_wire(item) for item in await self._call_list("list_active_tasks", **kwargs)]

    async def wait_task_status_change(self, **kwargs: Any) -> TaskStatusChange:
        out = await self._call_dict("wait_task_status_change", **kwargs)
        out.setdefault("request_id", kwargs.get("request_id"))
        return TaskStatusChange.from_wire(out)

    async def update_task_metadata(self, **kwargs: Any) -> TaskMutationResult:
        return TaskMutationResult.from_wire(await self._call_dict("update_task_metadata", **kwargs))


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
    "complete_task_failure",
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
    missing_client_async = [
        f"async_{name}"
        for name in _ASYNC_TASK_LEDGER_METHODS
        if not callable(getattr(client, f"async_{name}", None))
    ]
    if missing_client_async:
        raise TypeError(
            "task ledger client does not implement AsyncTaskLedger; "
            f"missing_async={(missing_async + missing_client_async)[:5]}"
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

    async def finish_success(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.finish_lease_success(**kwargs)

    async def finish_failure(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("timeout_s", None)
        return await self.actor.finish_lease_failure(**kwargs)

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

    async def contains_request(self, **kwargs: Any) -> dict[str, Any]:
        return await self.contains(**kwargs)

    async def cancel_request(self, **kwargs: Any) -> dict[str, Any]:
        return await self.actor.cancel_request(**kwargs)

    async def stats(self, **_kwargs: Any) -> dict[str, Any]:
        return self.actor.stats()
