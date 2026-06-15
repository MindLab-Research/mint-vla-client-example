from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from ..config import (
    PFS_PYTHONPATH,
    actor_runtime_env_vars,
    apply_detached_actor_resources,
    otel_env_vars,
    preferred_vllm_python_executable,
)
from ..runtime_env import env_nonempty
from ..logging_context import (
    classify_failure_reason,
    extract_trace_id_from_traceparent,
    get_otel_tracer,
    record_scheduler_decision_otel,
    set_request_id,
    set_trace_id,
)
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref
from .control_plane_contracts import ExecutorOutcome, LeaseToken, as_task_ledger
from .engine_adapter import EngineHealth, EngineHealthStatus, EngineObservability
from .engine_liveness import EngineLivenessPush
from .execution_context import ExecutionContext, bind_execution_context
from .model_actor_supervisor import consumer_id_for_replica, queue_id_for_replica
from .model_work_scheduler import ModelWorkSchedulerClient, model_work_scheduler
from .model_work_execution_context import ModelWorkFinalizeBuffer, model_work_execution_context
from .task_payload_store import TaskPayloadStore
from .task_state_store import FutureStatus, TaskStateNotFoundError, task_futures, task_state_store

logger = logging.getLogger(__name__)

ModelWorkExecutor = Callable[[dict[str, Any]], Awaitable[ExecutorOutcome | None] | ExecutorOutcome | None]
TokenBudgetProvider = Callable[[], Awaitable[int | None]]
EngineLifecycleHook = Any
SelfExitHook = Callable[[str], Awaitable[None] | None]
LivenessPushHook = Callable[[EngineLivenessPush], Awaitable[None] | None]
_EXECUTION_BINDINGS: ExecutionContext | None = None
VLLM_TOKEN_BUDGET_RATIO = 0.95
VLLM_TOKEN_BUDGET_REFRESH_S = 60.0
VLLM_TOKEN_BUDGET_QUERY_TIMEOUT_S = 1.0
ENGINE_RESTART_TIMEOUT_S = 30.0 * 60.0
_RAY_ATTACH_ENV_KEYS = (
    "RAY_CLIENT_ADDRESS",
    "MINT_RAY_CLIENT_ADDRESS",
    "MINT_RAY_HEAD_ADDRESS_PATH",
    "MINT_RAY_NODE_IP_ADDRESS",
    "MINT_RAY_TEMP_DIR",
    "RAY_TMPDIR",
    "TMPDIR",
    "TMP",
    "TEMP",
)


class EngineDeathLeaseDisposition(StrEnum):
    TERMINAL_FAILED = "terminal_failed"
    RESUMED_SUCCESS = "resumed_success"


def _scheduler_result_ok(result: Any) -> bool:
    ok = getattr(result, "ok", None)
    if ok is not None:
        return bool(ok)
    get_result = getattr(result, "get", None)
    return bool(get_result("ok")) if callable(get_result) else False


def _blank_process_ray_attach_hints() -> None:
    os.environ.pop("RAY_ADDRESS", None)
    for key in _RAY_ATTACH_ENV_KEYS:
        os.environ[key] = ""


def _scheduler_claim_leases(result: Any) -> list[dict[str, Any]] | None:
    leases = getattr(result, "leases", None)
    if leases is None:
        get_result = getattr(result, "get", None)
        leases = get_result("leases") if callable(get_result) else None
    return leases if isinstance(leases, list) else None


def _lease_wire_to_token(lease: dict[str, Any], *, require_full: bool) -> LeaseToken:
    item = lease.get("item") if isinstance(lease, dict) else None
    if require_full and not isinstance(item, dict):
        raise RuntimeError(f"model work lease missing item: {lease!r}")
    item_wire = item if isinstance(item, dict) else {}
    token_wire = {
        "request_id": item_wire["request_id"] if require_full else item_wire.get("request_id", ""),
        "lease_id": lease["lease_id"],
        "attempt_id": lease["attempt_id"] if require_full else lease.get("attempt_id", ""),
        "scheduler_epoch": lease["scheduler_epoch"] if require_full else lease.get("scheduler_epoch", 0),
        "consumer_id": lease.get("consumer_id", ""),
        "consumer_generation": lease.get("consumer_generation", 0),
    }
    return LeaseToken.from_wire(token_wire)


def _lease_token(lease: dict[str, Any]) -> LeaseToken:
    return _lease_wire_to_token(lease, require_full=True)


def _legacy_lease_token(lease: dict[str, Any]) -> LeaseToken:
    return _lease_wire_to_token(lease, require_full=False)


def _scheduler_lease_token(lease: dict[str, Any]) -> LeaseToken:
    try:
        return _lease_token(lease)
    except Exception:
        return _legacy_lease_token(lease)


def _lease_item_wire(lease: dict[str, Any]) -> dict[str, Any]:
    item = lease.get("item") if isinstance(lease, dict) else None
    if not isinstance(item, dict):
        raise RuntimeError(f"model work lease missing item: {lease!r}")
    return item


def _outcome_from_finalize_buffer(finalize_buffer: ModelWorkFinalizeBuffer) -> ExecutorOutcome | None:
    finalization = finalize_buffer.finalization
    if finalization is None:
        return None
    if finalization.kind == "resolve":
        return ExecutorOutcome(
            kind="success",
            payload=finalization.payload,
            billing_observations=finalization.billing_observations,
        )
    if finalization.kind == "fail":
        return ExecutorOutcome(kind="user_error", error=str(finalization.payload))
    raise RuntimeError(f"unknown model work finalization kind: {finalization.kind!r}")


def _finalization_from_outcome(
    *,
    request_id: str,
    outcome: ExecutorOutcome,
) -> SimpleNamespace:
    if outcome.kind == "success":
        return SimpleNamespace(
            kind="resolve",
            request_id=str(request_id),
            payload=outcome.payload,
            billing_observations=outcome.billing_observations,
        )
    if outcome.kind == "user_error":
        return SimpleNamespace(
            kind="fail",
            request_id=str(request_id),
            payload=str(outcome.error or "Task failed"),
            billing_observations=None,
        )
    raise RuntimeError(f"cannot convert executor outcome to terminal finalization: {outcome.kind!r}")


@dataclass(frozen=True)
class ModelEngineHostConfig:
    domain_key: str
    replica_id: str
    actor_name: str
    actor_generation: int
    base_model: str | None = None
    poll_interval_s: float = 0.2
    lease_ttl_s: float = 30.0
    max_claim: int = 1
    token_budget: int | None = None
    execution_timeout_s: float | None = None
    engine_restart_timeout_s: float = ENGINE_RESTART_TIMEOUT_S
    runtime_env_fingerprint: str | None = None

    @property
    def consumer_id(self) -> str:
        return consumer_id_for_replica(self.domain_key, self.replica_id, self.actor_generation)

    @property
    def queue_id(self) -> str:
        return queue_id_for_replica(self.domain_key, self.replica_id)


def _sanitize_actor_name_part(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return out or "unknown"


def default_model_engine_host_name(domain_key: str, replica_id: str) -> str:
    return f"mint_model_runtime_{_sanitize_actor_name_part(domain_key).lower()}_{_sanitize_actor_name_part(replica_id).lower()}"


def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _runtime_env_fingerprint(runtime_env_extra: dict[str, str] | None) -> str:
    payload = {
        str(key): str(value)
        for key, value in sorted((runtime_env_extra or {}).items())
        if value is not None
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_or_create_model_engine_host(
    *,
    domain_key: str,
    replica_id: str = "replica-0",
    actor_name: str | None = None,
    actor_generation: int = 0,
    base_model: str | None = None,
    poll_interval_s: float | None = None,
    lease_ttl_s: float | None = None,
    max_claim: int = 1,
    token_budget: int | None = None,
    execution_timeout_s: float | None = None,
    engine_restart_timeout_s: float | None = None,
    runtime_env_extra: dict[str, str] | None = None,
    ray_address: str | None = None,
) -> Any:
    import ray

    name = str(actor_name or default_model_engine_host_name(domain_key, replica_id))
    expected_runtime_env_fingerprint = _runtime_env_fingerprint(runtime_env_extra)
    try:
        existing = ray.get_actor(name, namespace=_ray_namespace())
        health = sync_get_ray_ref(existing.health_snapshot.remote(), timeout_s=5.0)
        expected_token_budget = None if token_budget is None else int(token_budget)
        if (
            isinstance(health, dict)
            and str(health.get("domain_key")) == str(domain_key)
            and str(health.get("replica_id")) == str(replica_id)
            and int(health.get("actor_generation") or -1) == int(actor_generation)
            and int(health.get("max_claim") or 0) == max(1, int(max_claim))
            and health.get("token_budget") == expected_token_budget
            and str(health.get("runtime_env_fingerprint") or "") == expected_runtime_env_fingerprint
        ):
            return existing
        logger.warning(
            "[model_runtime] killing stale detached actor name=%s expected_domain=%s expected_replica=%s expected_generation=%s expected_max_claim=%s expected_token_budget=%s expected_runtime_env_fingerprint=%s health=%s",
            name,
            domain_key,
            replica_id,
            actor_generation,
            max_claim,
            expected_token_budget,
            expected_runtime_env_fingerprint,
            health,
        )
        ray.kill(existing, no_restart=True)
    except ValueError:
        pass
    except Exception as e:
        logger.warning(
            "[model_runtime] existing actor health check failed name=%s error_type=%s error=%s; recreating",
            name,
            type(e).__name__,
            e,
        )
        try:
            existing = ray.get_actor(name, namespace=_ray_namespace())
            ray.kill(existing, no_restart=True)
        except Exception:
            pass

    from ..ray_utils import strict_ray_gcs_address

    resolved_ray_address = str(ray_address or "").strip() or strict_ray_gcs_address()
    if resolved_ray_address is None:
        raise RuntimeError("MINT_RAY_GCS_ADDRESS is required")

    preferred_python = (preferred_vllm_python_executable() or "").strip()
    runtime_env_extra_payload = {
        **otel_env_vars(),
        **(runtime_env_extra or {}),
    }
    if preferred_python:
        runtime_env_extra_payload.setdefault("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", preferred_python)
        runtime_env_extra_payload.setdefault("MINT_ENABLE_VLLM_IMPORT_PATCHES", "1")
    runtime_env: dict[str, Any] = {
        "env_vars": actor_runtime_env_vars(
            pythonpath=PFS_PYTHONPATH,
            extra=runtime_env_extra_payload,
            include_ray_attach_hints=False,
        )
    }

    remote_cls = ray.remote(num_cpus=0, max_concurrency=64, max_restarts=-1)(ModelEngineHost)
    options: dict[str, Any] = {
        "name": name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_task_retries": -1,
        "runtime_env": runtime_env,
    }
    apply_detached_actor_resources(options, ray)
    return remote_cls.options(**options).remote(
        domain_key=str(domain_key),
        replica_id=str(replica_id),
        actor_name=name,
        actor_generation=int(actor_generation),
        base_model=base_model,
        poll_interval_s=poll_interval_s,
        lease_ttl_s=lease_ttl_s,
        max_claim=max_claim,
        token_budget=token_budget,
        execution_timeout_s=execution_timeout_s,
        engine_restart_timeout_s=engine_restart_timeout_s,
        runtime_env_fingerprint=expected_runtime_env_fingerprint,
        ray_address=resolved_ray_address,
    )


async def _default_executor(lease: dict[str, Any]) -> ExecutorOutcome:
    context = await _ensure_execution_bindings()
    item = _lease_item_wire(lease)
    op = str(item.get("op") or "")
    from .model_work_dispatch import execute_model_work_item

    finalize_buffer = ModelWorkFinalizeBuffer()
    token = _lease_token(lease)
    with bind_execution_context(context), model_work_execution_context(
        lease_id=token.lease_id,
        consumer_id=token.consumer_id,
        consumer_generation=token.consumer_generation,
        finalize_buffer=finalize_buffer,
    ):
        outcome = await execute_model_work_item(
            SimpleNamespace(
                request_id=str(item["request_id"]),
                op=op,
                request_json=bytes(item.get("request_json") or b""),
                user_id=None if item.get("user_id") is None else str(item.get("user_id")),
                apikey_id=None if item.get("apikey_id") is None else str(item.get("apikey_id")),
                throttle_principal=(
                    None
                    if item.get("throttle_principal") is None
                    else str(item.get("throttle_principal"))
                ),
                webhook_url=None if item.get("webhook_url") is None else str(item.get("webhook_url")),
                extra=dict(item.get("extra") or {}),
                created_at=float(item.get("created_at") or time.time()),
            ),
            component="model_engine_host",
        )
    if outcome.kind == "success":
        return _outcome_from_finalize_buffer(finalize_buffer) or outcome
    return outcome


async def _ensure_execution_bindings(*, force_refresh: bool = False) -> ExecutionContext:
    global _EXECUTION_BINDINGS
    if _EXECUTION_BINDINGS is not None and not force_refresh:
        return _EXECUTION_BINDINGS
    from .execution_bindings import initialize_execution_bindings

    _EXECUTION_BINDINGS = ExecutionContext(**await initialize_execution_bindings())
    return _EXECUTION_BINDINGS


async def _refresh_execution_bindings() -> ExecutionContext:
    return await _ensure_execution_bindings(force_refresh=True)


async def _default_liveness_push(payload: EngineLivenessPush) -> None:
    from .model_actor_supervisor import get_model_actor_supervisor

    await get_model_actor_supervisor().push_liveness(payload)


class ModelEngineHost:
    """Model-owned pull executor bound to one scheduler ReplicaQueue."""

    def __init__(
        self,
        *,
        domain_key: str,
        replica_id: str = "replica-0",
        actor_name: str | None = None,
        actor_generation: int = 0,
        base_model: str | None = None,
        poll_interval_s: float | None = None,
        lease_ttl_s: float | None = None,
        max_claim: int = 1,
        token_budget: int | None = None,
        execution_timeout_s: float | None = None,
        runtime_env_fingerprint: str | None = None,
        scheduler_client: ModelWorkSchedulerClient | None = None,
        task_futures_client: Any | None = None,
        task_state_store_client: Any | None = None,
        payload_store: TaskPayloadStore | None = None,
        executor: ModelWorkExecutor | None = None,
        token_budget_provider: TokenBudgetProvider | None = None,
        engine_lifecycle: EngineLifecycleHook | None = None,
        engine_restart_timeout_s: float | None = None,
        liveness_push: LivenessPushHook | None = None,
        self_exit: SelfExitHook | None = None,
        ray_address: str | None = None,
    ) -> None:
        self._ray_address = str(ray_address or "").strip() or None
        _blank_process_ray_attach_hints()
        domain = str(domain_key).strip()
        replica = str(replica_id).strip()
        if not domain:
            raise ValueError("domain_key is required")
        if not replica:
            raise ValueError("replica_id is required")
        self._config = ModelEngineHostConfig(
            domain_key=domain,
            replica_id=replica,
            actor_name=str(actor_name or default_model_engine_host_name(domain, replica)),
            actor_generation=int(actor_generation),
            base_model=None if base_model is None else str(base_model),
            poll_interval_s=max(
                0.01,
                float(
                    poll_interval_s
                    if poll_interval_s is not None
                    else os.environ.get("MINT_MODEL_RUNTIME_POLL_INTERVAL_S", "0.2")
                ),
            ),
            lease_ttl_s=max(
                0.1,
                float(
                    lease_ttl_s
                    if lease_ttl_s is not None
                    else os.environ.get("MINT_MODEL_RUNTIME_LEASE_TTL_S", "30.0")
                ),
            ),
            max_claim=max(1, int(max_claim)),
            token_budget=None if token_budget is None else int(token_budget),
            execution_timeout_s=self._normalize_optional_timeout_s(
                execution_timeout_s
                if execution_timeout_s is not None
                else os.environ.get("MINT_MODEL_RUNTIME_EXECUTION_TIMEOUT_S")
            ),
            engine_restart_timeout_s=max(
                0.001,
                float(
                    engine_restart_timeout_s
                    if engine_restart_timeout_s is not None
                    else os.environ.get(
                        "MINT_MODEL_RUNTIME_ENGINE_RESTART_TIMEOUT_S",
                        str(ENGINE_RESTART_TIMEOUT_S),
                    )
                ),
            ),
            runtime_env_fingerprint=runtime_env_fingerprint,
        )
        self._scheduler = scheduler_client if scheduler_client is not None else model_work_scheduler
        self._task_futures = task_futures_client if task_futures_client is not None else task_futures
        self._task_state_store = as_task_ledger(
            task_state_store_client if task_state_store_client is not None else task_state_store
        )
        self._payload_store = payload_store if payload_store is not None else TaskPayloadStore()
        self._executor = executor if executor is not None else _default_executor
        self._token_budget_provider = (
            token_budget_provider if token_budget_provider is not None else self._default_token_budget_provider
        )
        default_lifecycle = engine_lifecycle is None and executor is None
        if default_lifecycle:
            from .engine_lifecycle import ExecutionContextEngineLifecycle

            engine_lifecycle = ExecutionContextEngineLifecycle(
                _ensure_execution_bindings,
                refresh_context_factory=_refresh_execution_bindings,
            )
        self._engine_lifecycle = engine_lifecycle
        self._liveness_push = (
            liveness_push
            if liveness_push is not None
            else (_default_liveness_push if default_lifecycle else None)
        )
        self._self_exit = self_exit if self_exit is not None else self._ray_actor_self_exit

        self._running = False
        self._draining = False
        self._engine_restarting = False
        self._engine_ready: bool | None = None
        self._engine_restart_deadline_at: float | None = None
        self._engine_restart_count = 0
        self._engine_failure_epoch = 0
        self._engine_restart_timed_out = False
        self._self_exit_requested = False
        self._engine_death_lock = asyncio.Lock()
        self._loop_task: asyncio.Task | None = None
        self._active_request_id: str | None = None
        self._active_lease_id: str | None = None
        self._active_leases: dict[str, dict[str, Any]] = {}
        self._started_at = time.time()
        self._last_claimed_at: float | None = None
        self._last_completed_at: float | None = None
        self._last_renewed_at: float | None = None
        self._max_renew_rpc_latency_s = 0.0
        self._consecutive_renew_failures = 0
        self._last_renew_deadline_slack_s: float | None = None
        self._last_error: str | None = None
        self._last_error_traceback: str | None = None
        self._dynamic_token_budget: int | None = None
        self._dynamic_token_capacity_tokens: int | None = None
        self._dynamic_token_budget_ratio: float | None = None
        self._dynamic_token_budget_updated_at: float | None = None
        self._dynamic_token_budget_error: str | None = None
        self._processed_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._requeued_total = 0
        self._renewed_total = 0
        self._empty_polls_total = 0

    async def _task_state_future_call(self, method: str, **kwargs: Any) -> Any:
        async_method = getattr(self._task_state_store, method)
        return await async_method(**kwargs)

    @property
    def domain_key(self) -> str:
        return self._config.domain_key

    @property
    def replica_id(self) -> str:
        return self._config.replica_id

    @property
    def actor_name(self) -> str:
        return self._config.actor_name

    @property
    def actor_generation(self) -> int:
        return self._config.actor_generation

    @property
    def consumer_id(self) -> str:
        return self._config.consumer_id

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "actor_name": self._config.actor_name,
            "domain_key": self._config.domain_key,
            "replica_id": self._config.replica_id,
            "queue_id": self._config.queue_id,
            "consumer_id": self._config.consumer_id,
            "base_model": self._config.base_model,
            "actor_generation": int(self._config.actor_generation),
            "running": bool(self._running),
            "draining": bool(self._draining),
            "active_request_id": self._active_request_id,
            "active_lease_id": self._active_lease_id,
            "active_request_ids": [
                str(_lease_item_wire(lease)["request_id"]) for lease in self._active_leases.values()
            ],
            "active_lease_ids": list(self._active_leases.keys()),
            "active_lease_count": len(self._active_leases),
            "engine_ready": self._engine_ready,
            "engine_restarting": bool(self._engine_restarting),
            "engine_restart_deadline_at": self._engine_restart_deadline_at,
            "engine_restart_count": int(self._engine_restart_count),
            "engine_restart_timeout_s": float(self._config.engine_restart_timeout_s),
            "engine_restart_timed_out": bool(self._engine_restart_timed_out),
            "engine_failure_epoch": int(self._engine_failure_epoch),
            "self_exit_requested": bool(self._self_exit_requested),
            "started_at": float(self._started_at),
            "last_claimed_at": self._last_claimed_at,
            "last_completed_at": self._last_completed_at,
            "last_renewed_at": self._last_renewed_at,
            "max_renew_rpc_latency_s": self._max_renew_rpc_latency_s,
            "consecutive_renew_failures": self._consecutive_renew_failures,
            "last_renew_deadline_slack_s": self._last_renew_deadline_slack_s,
            "last_error": self._last_error,
            "last_error_traceback": self._last_error_traceback,
            "processed_total": int(self._processed_total),
            "completed_total": int(self._completed_total),
            "failed_total": int(self._failed_total),
            "requeued_total": int(self._requeued_total),
            "renewed_total": int(self._renewed_total),
            "empty_polls_total": int(self._empty_polls_total),
            "poll_interval_s": float(self._config.poll_interval_s),
            "lease_ttl_s": float(self._config.lease_ttl_s),
            "max_claim": int(self._config.max_claim),
            "token_budget": self._config.token_budget,
            "runtime_env_fingerprint": self._config.runtime_env_fingerprint,
            "dynamic_token_budget": self._dynamic_token_budget,
            "dynamic_token_capacity_tokens": self._dynamic_token_capacity_tokens,
            "dynamic_token_budget_ratio": self._dynamic_token_budget_ratio,
            "dynamic_token_budget_updated_at": self._dynamic_token_budget_updated_at,
            "dynamic_token_budget_error": self._dynamic_token_budget_error,
            "execution_timeout_s": self._config.execution_timeout_s,
        }

    async def start(self) -> dict[str, Any]:
        self._draining = False
        if self._running and self._loop_task is not None and not self._loop_task.done():
            return self.health_snapshot()
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        return self.health_snapshot()

    async def drain(self) -> dict[str, Any]:
        self._draining = True
        return self.health_snapshot()

    async def shutdown(self) -> dict[str, Any]:
        self._running = False
        self._draining = True
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.health_snapshot()

    async def _run_loop(self) -> None:
        while self._running:
            try:
                if self._draining:
                    await asyncio.sleep(self._config.poll_interval_s)
                    continue
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._record_error(e)
                logger.error(
                    "[model_runtime] loop failed actor=%s domain=%s replica=%s error_type=%s failure_reason=%s",
                    self._config.actor_name,
                    self._config.domain_key,
                    self._config.replica_id,
                    type(e).__name__,
                    classify_failure_reason(e),
                )
            await asyncio.sleep(self._config.poll_interval_s)

    async def run_once(self) -> dict[str, Any]:
        if self._draining:
            await self._push_liveness()
            return {"claimed": 0, "executed": 0, "draining": True}
        if await self._engine_restart_deadline_expired():
            await self._push_liveness()
            return {"claimed": 0, "executed": 0, "engine_ready": False, "self_exit": True}
        engine_ready = await self._engine_is_ready()
        if not engine_ready:
            if await self._engine_restart_deadline_expired():
                await self._push_liveness()
                return {"claimed": 0, "executed": 0, "engine_ready": False, "self_exit": True}
            self._empty_polls_total += 1
            await self._push_liveness()
            return {"claimed": 0, "executed": 0, "engine_ready": False}
        max_items, token_budget = await self._claim_limits()
        try:
            claimed = await self._scheduler.claim(
                domain_key=self._config.domain_key,
                replica_id=self._config.replica_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                max_items=max_items,
                token_budget=token_budget,
                lease_ttl_s=self._config.lease_ttl_s,
            )
        except Exception as exc:
            if self._is_scheduler_not_claimable_error(exc):
                self._empty_polls_total += 1
                await self._push_liveness()
                return {"claimed": 0, "executed": 0}
            raise
        leases = _scheduler_claim_leases(claimed)
        if not leases:
            self._empty_polls_total += 1
            self._clear_transient_scheduler_error()
            await self._push_liveness()
            return {"claimed": 0, "executed": 0}

        self._last_claimed_at = time.time()
        valid_leases = [lease for lease in leases if isinstance(lease, dict)]
        if len(valid_leases) == 1:
            await self._execute_lease(valid_leases[0])
        elif valid_leases and self._executes_leases_concurrently():
            await asyncio.gather(*(self._execute_lease(lease) for lease in valid_leases))
        else:
            await self._execute_leases_sequentially(valid_leases)
        await self._push_liveness()
        return {"claimed": len(leases), "executed": len(valid_leases)}

    async def _claim_limits(self) -> tuple[int, int | None]:
        if self._config.token_budget is not None:
            return int(self._config.max_claim), int(self._config.token_budget)
        if not self._is_vllm_domain():
            return int(self._config.max_claim), None
        token_budget = await self._refresh_dynamic_token_budget()
        if token_budget is None:
            return 1, None
        return int(self._config.max_claim), int(token_budget)

    def _is_vllm_domain(self) -> bool:
        return str(self._config.domain_key).startswith("vllm:")

    def _executes_leases_concurrently(self) -> bool:
        return self._is_vllm_domain()

    async def _execute_leases_sequentially(self, leases: list[dict[str, Any]]) -> None:
        lease_tokens = [_scheduler_lease_token(lease) for lease in leases if isinstance(lease, dict)]
        pending_leases = {token.lease_id: token for token in lease_tokens}
        renew_task: asyncio.Task | None = None
        if len(pending_leases) > 1:
            renew_task = asyncio.create_task(self._renew_pending_leases_until_done(pending_leases))
        try:
            for lease in leases:
                lease_id = _scheduler_lease_token(lease).lease_id
                await self._execute_lease(lease)
                pending_leases.pop(lease_id, None)
        finally:
            pending_leases.clear()
            if renew_task is not None:
                renew_task.cancel()
                await asyncio.gather(renew_task, return_exceptions=True)

    async def _refresh_dynamic_token_budget(self) -> int | None:
        now = time.time()
        if self._dynamic_token_budget is not None and self._dynamic_token_budget_updated_at is not None:
            if now - float(self._dynamic_token_budget_updated_at) < VLLM_TOKEN_BUDGET_REFRESH_S:
                return int(self._dynamic_token_budget)
        try:
            budget = await self._token_budget_provider()
        except Exception as e:
            self._dynamic_token_budget_error = f"{type(e).__name__}: {e}"
            logger.debug(
                "[model_runtime] dynamic token budget refresh failed actor=%s domain=%s error=%s",
                self._config.actor_name,
                self._config.domain_key,
                self._dynamic_token_budget_error,
            )
            return self._dynamic_token_budget
        if budget is None:
            return self._dynamic_token_budget
        budget_i = max(1, int(budget))
        self._dynamic_token_budget = budget_i
        self._dynamic_token_budget_updated_at = now
        self._dynamic_token_budget_error = None
        return budget_i

    async def _default_token_budget_provider(self) -> int | None:
        if not self._is_vllm_domain():
            return None
        actor_name = self._vllm_actor_name()
        if not actor_name:
            return None
        import ray

        actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        getter = getattr(actor, "get_observability_binding", None)
        if not callable(getter):
            return None
        timeout_s = VLLM_TOKEN_BUDGET_QUERY_TIMEOUT_S
        payload = await async_get_ray_ref(getter.remote(), timeout_s=timeout_s)
        if not isinstance(payload, dict):
            return None
        capacity = self._positive_int(payload.get("kv_cache_capacity_tokens"))
        if capacity is None:
            kv_debug = getattr(actor, "get_kv_debug_info", None)
            if callable(kv_debug):
                payload = await async_get_ray_ref(kv_debug.remote(), timeout_s=timeout_s)
                if isinstance(payload, dict):
                    capacity = self._positive_int(payload.get("kv_cache_capacity_tokens"))
        if capacity is None:
            return None
        ratio = VLLM_TOKEN_BUDGET_RATIO
        self._dynamic_token_capacity_tokens = int(capacity)
        self._dynamic_token_budget_ratio = float(ratio)
        return max(1, int(float(capacity) * ratio))

    def _vllm_actor_name(self) -> str | None:
        base_model = str(self._config.base_model or "").strip()
        if not base_model and self._is_vllm_domain():
            base_model = str(self._config.domain_key).removeprefix("vllm:").strip()
        if not base_model:
            return None
        model_part = base_model.split("/")[-1] if "/" in base_model else base_model
        return f"mint_vllm_{model_part.lower().replace(' ', '_')}"

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            out = int(value)
        except (TypeError, ValueError):
            return None
        return out if out > 0 else None

    def _restore_item_context(self, lease: dict[str, Any]) -> None:
        item = _lease_item_wire(lease)
        extra = item.get("extra") if isinstance(item, dict) else {}
        extra = extra if isinstance(extra, dict) else {}
        trace_id = extra.get("_trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            traceparent = extra.get("_traceparent")
            trace_id = extract_trace_id_from_traceparent(traceparent) if isinstance(traceparent, str) else None
        set_trace_id(trace_id if isinstance(trace_id, str) else None)
        request_id = item.get("request_id") if isinstance(item, dict) else None
        if request_id is not None:
            set_request_id(str(request_id))

    def _clear_active_lease(self, lease_id: str) -> None:
        self._active_leases.pop(str(lease_id), None)
        next_lease = next(iter(self._active_leases.values()), None)
        self._active_request_id = (
            str(_lease_item_wire(next_lease)["request_id"]) if isinstance(next_lease, dict) else None
        )
        self._active_lease_id = next(iter(self._active_leases.keys()), None)

    def _record_error(self, e: BaseException) -> None:
        self._last_error = f"{type(e).__name__}: {e}"
        self._last_error_traceback = traceback.format_exc()

    @staticmethod
    def _is_missing_future_state_error(exc: BaseException) -> bool:
        if isinstance(exc, (KeyError, TaskStateNotFoundError)):
            return True
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, (KeyError, TaskStateNotFoundError)):
            return True
        context = getattr(exc, "__context__", None)
        return isinstance(context, (KeyError, TaskStateNotFoundError))

    @staticmethod
    def _is_scheduler_not_claimable_error(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return "modelworkschedulerconflicterror" in text and "not claimable" in text

    def _clear_transient_scheduler_error(self) -> None:
        error = str(self._last_error or "").lower()
        if not error:
            return
        if "modelworkschedulerconflicterror" in error or "consumer_id mismatch" in error:
            self._last_error = None
            self._last_error_traceback = None

    @staticmethod
    def _normalize_optional_timeout_s(value: Any) -> float | None:
        if value is None:
            return None
        try:
            timeout_s = float(value)
        except (TypeError, ValueError):
            return None
        if timeout_s <= 0:
            return None
        return max(0.1, timeout_s)

    def _default_save_lora_timeout_s(self) -> float:
        base_model = str(self._config.base_model or "")
        try:
            from .model_registry import get_model_config

            train_gpus = int(get_model_config(base_model).train_gpus) if base_model else 1
        except Exception:
            train_gpus = 1

        if train_gpus >= 32:
            return 3600.0
        if train_gpus >= 16:
            return 1800.0
        if train_gpus >= 4:
            return 600.0
        return 300.0

    def _recovered_stale_generation_error(self, lease: dict[str, Any]) -> RuntimeError | None:
        item = _lease_item_wire(lease)
        extra = item.get("extra") if isinstance(item, dict) and isinstance(item.get("extra"), dict) else {}
        raw = extra.get("actor_generation")
        if raw is None:
            return None
        try:
            recovered_generation = int(raw)
        except (TypeError, ValueError):
            return None
        current_generation = int(self._config.actor_generation)
        if recovered_generation == current_generation:
            return None
        return RuntimeError(
            "model work request recovered from stale runtime generation "
            f"actor_generation={recovered_generation} current_actor_generation={current_generation}; "
            "request must be retried"
        )

    def _execution_timeout_s_for_lease(self, lease: dict[str, Any]) -> float | None:
        if self._config.execution_timeout_s is not None:
            return self._config.execution_timeout_s
        item = _lease_item_wire(lease)
        op = str(item.get("op") if isinstance(item, dict) else "")
        if op != "training.save_weights_for_sampler":
            return None

        explicit = self._normalize_optional_timeout_s(
            os.environ.get("MINT_MODEL_RUNTIME_SAVE_WEIGHTS_TIMEOUT_S")
        )
        if explicit is not None:
            return explicit

        save_timeout_s = self._default_save_lora_timeout_s()
        grace_s = self._normalize_optional_timeout_s(
            os.environ.get("MINT_MODEL_RUNTIME_EXECUTION_TIMEOUT_GRACE_S")
        )
        if grace_s is None:
            grace_s = 60.0
        return save_timeout_s + grace_s

    async def _cancel_executor_task(self, task: asyncio.Task | None, *, reason: str) -> None:
        if task is None or task.done():
            return
        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=5.0)
        for done_task in done:
            try:
                done_task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if pending:
            logger.warning(
                "[model_runtime] executor task did not stop after cancellation actor=%s reason=%s",
                self._config.actor_name,
                reason,
            )

    async def _status_is_pending(self, lease: dict[str, Any]) -> bool:
        item = _lease_item_wire(lease)
        request_id = str(item["request_id"])
        try:
            status = await self._task_futures.async_get_status(request_id)
        except KeyError:
            await self._finish_non_pending_scheduler_lease(
                lease,
                status=None,
                error="model work task disappeared before runtime execution",
            )
            return False
        if status is not None and status != FutureStatus.PENDING:
            await self._finish_non_pending_scheduler_lease(
                lease,
                status=status,
                error=f"model work task is already terminal: {status.value}",
            )
            return False
        return True

    async def _finish_non_pending_scheduler_lease(
        self,
        lease: dict[str, Any],
        *,
        status: FutureStatus | None,
        error: str,
    ) -> None:
        token = _scheduler_lease_token(lease)
        if status == FutureStatus.DONE and self._task_state_finalize_enabled(lease):
            finished = await self._scheduler.finish_success(
                lease=_lease_token(lease),
                result_path="",
                result_checksum=None,
                result_size_bytes=None,
                billing_observations=None,
            )
            if _scheduler_result_ok(finished):
                return
        if status == FutureStatus.FAILED and self._task_state_finalize_enabled(lease):
            finished = await self._scheduler.finish_failure(
                lease=_lease_token(lease),
                error=error,
            )
            if _scheduler_result_ok(finished):
                return
        await self._scheduler.fail(
            lease=token,
            reason="task_not_pending",
            requeue=False,
            abort_finalize=True,
        )

    async def _fail_lost_lease_if_still_pending(
        self,
        request_id: str,
        error: str,
        *,
        attempt_id: str | None = None,
    ) -> bool:
        expected_meta = {"model_work_attempt_id": attempt_id} if attempt_id else None
        out = await self._task_futures.async_fail_if_pending_meta_matches(
            request_id,
            error,
            expected_meta=expected_meta,
        )
        return bool(out.get("failed")) if isinstance(out, dict) else False

    def _task_state_finalize_enabled(self, lease: dict[str, Any]) -> bool:
        try:
            _lease_token(lease)
        except Exception:
            return False
        return True

    def _require_task_state_finalize(self, lease: dict[str, Any]) -> None:
        if self._task_state_finalize_enabled(lease):
            return
        raise RuntimeError(
            "model work lease missing TaskStateStore finalize metadata "
            "(attempt_id and scheduler_epoch are required)"
        )

    def _lease_attempt_id(self, lease: dict[str, Any]) -> str | None:
        attempt_id = _scheduler_lease_token(lease).attempt_id or None
        if attempt_id:
            return attempt_id
        item = _lease_item_wire(lease)
        extra = item.get("extra") if isinstance(item, dict) and isinstance(item.get("extra"), dict) else {}
        return str(extra.get("model_work_attempt_id") or "") or None

    def _payload_attempt_id_for_lease(self, lease: dict[str, Any]) -> str:
        token = _lease_token(lease)
        return f"{token.attempt_id}__{token.lease_id}"

    def _staged_payload_path_for_lease(self, lease: dict[str, Any]) -> str:
        item = _lease_item_wire(lease)
        return str(
            self._payload_store.payload_path(
                request_id=str(item["request_id"]),
                attempt_id=self._payload_attempt_id_for_lease(lease),
            )
        )

    async def _write_success_payload(
        self,
        lease: dict[str, Any],
        *,
        payload: Any,
    ) -> dict[str, Any]:
        self._require_task_state_finalize(lease)
        item = _lease_item_wire(lease)
        request_id = str(item["request_id"])
        return await asyncio.to_thread(
            self._payload_store.write_json_payload,
            request_id=request_id,
            attempt_id=self._payload_attempt_id_for_lease(lease),
            payload=payload,
        )

    async def _engine_is_ready(self) -> bool:
        if self._engine_lifecycle is None:
            self._engine_ready = True
            return True
        checker = getattr(self._engine_lifecycle, "is_ready", None)
        if not callable(checker):
            self._engine_ready = True
            return True
        try:
            ready = checker()
            if inspect.isawaitable(ready):
                ready = await ready
            self._engine_ready = bool(ready)
            return bool(ready)
        except Exception as exc:
            self._record_error(exc)
            self._engine_ready = False
            return False

    async def _engine_health(self) -> EngineHealth:
        if self._engine_lifecycle is None:
            return EngineHealth(status=EngineHealthStatus.READY)
        getter = getattr(self._engine_lifecycle, "health", None)
        if callable(getter):
            try:
                health = getter()
                if inspect.isawaitable(health):
                    health = await health
                if isinstance(health, EngineHealth):
                    return health
                if isinstance(health, dict):
                    return EngineHealth.from_wire(health)
            except Exception as exc:
                self._record_error(exc)
                return EngineHealth(
                    status=EngineHealthStatus.UNHEALTHY,
                    reason="health_error",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
        if self._engine_ready is False:
            return EngineHealth(
                status=EngineHealthStatus.UNHEALTHY,
                reason="not_ready",
                last_error=self._last_error,
            )
        if self._engine_restarting:
            return EngineHealth(
                status=EngineHealthStatus.RESTARTING,
                restart_count=int(self._engine_restart_count),
                last_error=self._last_error,
            )
        return EngineHealth(status=EngineHealthStatus.READY, last_error=self._last_error)

    async def _engine_observability(self) -> EngineObservability:
        if self._engine_lifecycle is None:
            return EngineObservability()
        getter = getattr(self._engine_lifecycle, "get_observability_binding", None)
        if not callable(getter):
            return EngineObservability()
        try:
            observability = getter()
            if inspect.isawaitable(observability):
                observability = await observability
            if isinstance(observability, EngineObservability):
                return observability
            if isinstance(observability, dict):
                return EngineObservability.from_wire(observability)
        except Exception as exc:
            self._record_error(exc)
            logger.debug(
                "[model_runtime] engine observability lookup failed actor=%s domain=%s error_type=%s error=%s",
                self._config.actor_name,
                self._config.domain_key,
                type(exc).__name__,
                exc,
            )
        return EngineObservability()

    async def _push_liveness(self) -> None:
        if self._liveness_push is None:
            return
        payload = EngineLivenessPush(
            actor_name=self._config.actor_name,
            domain_key=self._config.domain_key,
            replica_id=self._config.replica_id,
            consumer_id=self._config.consumer_id,
            actor_generation=int(self._config.actor_generation),
            running=bool(self._running),
            engine_ready=bool(self._engine_ready),
            engine_health=await self._engine_health(),
            observability=await self._engine_observability(),
            active_request_id=self._active_request_id,
            active_lease_id=self._active_lease_id,
            active_lease_count=len(self._active_leases),
            pushed_at=time.time(),
            last_error=self._last_error,
        )
        try:
            out = self._liveness_push(payload)
            if inspect.isawaitable(out):
                await out
        except Exception as exc:
            self._record_error(exc)
            logger.warning(
                "[model_runtime] liveness push failed actor=%s domain=%s replica=%s error_type=%s error=%s",
                self._config.actor_name,
                self._config.domain_key,
                self._config.replica_id,
                type(exc).__name__,
                exc,
            )

    async def _ray_actor_self_exit(self, reason: str) -> None:
        logger.error(
            "[model_runtime] self exiting actor=%s domain=%s replica=%s reason=%s",
            self._config.actor_name,
            self._config.domain_key,
            self._config.replica_id,
            reason,
        )
        try:
            import ray

            actor_module = getattr(ray, "actor", None)
            exit_actor = getattr(actor_module, "exit_actor", None)
            if callable(exit_actor):
                exit_actor()
        except Exception:
            pass

    async def _request_self_exit(self, reason: str) -> None:
        if self._self_exit_requested:
            return
        self._self_exit_requested = True
        self._engine_restart_timed_out = True
        self._running = False
        self._draining = True
        self._record_error(RuntimeError(reason))
        out = self._self_exit(str(reason))
        if inspect.isawaitable(out):
            await out

    async def _engine_restart_deadline_expired(self) -> bool:
        deadline = self._engine_restart_deadline_at
        if deadline is None:
            return False
        if self._engine_ready is not False:
            return False
        if time.time() < float(deadline):
            return False
        await self._request_self_exit(
            "engine restart deadline exceeded "
            f"timeout_s={self._config.engine_restart_timeout_s:.2f}"
        )
        return True

    async def _mark_engine_unhealthy(self, reason: str) -> None:
        self._engine_ready = False
        marker = getattr(self._engine_lifecycle, "mark_unhealthy", None)
        if callable(marker):
            out = marker(str(reason))
            if inspect.isawaitable(out):
                await out

    async def _restart_engine(self) -> None:
        self._engine_restarting = True
        timeout_s = float(self._config.engine_restart_timeout_s)
        self._engine_restart_deadline_at = time.time() + timeout_s
        self._engine_restart_count += 1
        restarter = getattr(self._engine_lifecycle, "restart", None)
        try:
            if callable(restarter):
                out = restarter()
                if inspect.isawaitable(out):
                    await asyncio.wait_for(out, timeout=timeout_s)
            self._engine_ready = await self._engine_is_ready()
        except TimeoutError:
            self._engine_ready = False
            await self._request_self_exit(f"engine restart exceeded {timeout_s:g}s")
        finally:
            self._engine_restarting = False

    async def _settle_lease_for_engine_death(
        self,
        lease: dict[str, Any],
        *,
        error: str,
    ) -> EngineDeathLeaseDisposition | None:
        token = _scheduler_lease_token(lease)
        if not token.request_id or not token.attempt_id or not token.scheduler_epoch:
            failed = await self._scheduler.fail(
                lease=token,
                reason="gpu_actor_died",
                requeue=False,
                abort_finalize=True,
            )
            return EngineDeathLeaseDisposition.TERMINAL_FAILED if _scheduler_result_ok(failed) else None
        staged_payload = await self._resume_finalizing_payload(token)
        if staged_payload is not None:
            finished = await self._scheduler.finish_success(
                lease=token,
                result_path=str(staged_payload["result_path"]),
                result_checksum=(
                    None
                    if staged_payload.get("result_checksum") is None
                    else str(staged_payload.get("result_checksum"))
                ),
                result_size_bytes=(
                    None
                    if staged_payload.get("result_size_bytes") is None
                    else int(staged_payload.get("result_size_bytes") or 0)
                ),
                billing_observations=staged_payload.get("billing_observations"),
            )
            if _scheduler_result_ok(finished):
                return EngineDeathLeaseDisposition.RESUMED_SUCCESS
        begin_finalize = await self._scheduler.begin_finalize(
            lease=token,
            finalize_ttl_s=self._config.lease_ttl_s,
        )
        if not _scheduler_result_ok(begin_finalize):
            failed = await self._scheduler.fail(
                lease=_legacy_lease_token(lease),
                reason="gpu_actor_died",
                requeue=False,
                abort_finalize=True,
            )
            return EngineDeathLeaseDisposition.TERMINAL_FAILED if _scheduler_result_ok(failed) else None
        finished = await self._scheduler.finish_failure(
            lease=token,
            error=f"engine died: {error}",
        )
        return EngineDeathLeaseDisposition.TERMINAL_FAILED if _scheduler_result_ok(finished) else None

    async def _resume_finalizing_payload(self, token: LeaseToken) -> dict[str, Any] | None:
        hook = getattr(self._scheduler, "resume_finalizing_payload", None)
        if callable(hook):
            out = hook(token)
            if inspect.isawaitable(out):
                out = await out
            return dict(out) if isinstance(out, dict) else None
        try:
            record = await self._task_state_store.get_task(request_id=token.request_id)
        except Exception:
            return None
        data = record if isinstance(record, dict) else getattr(record, "data", None)
        if not isinstance(data, dict):
            return None
        status = str(data.get("status") or "")
        if status != "finalizing":
            return None
        if str(data.get("lease_id") or "") != token.lease_id:
            return None
        result_path = data.get("staged_payload_path") or data.get("result_path")
        if result_path is None:
            return None
        return {
            "result_path": str(result_path),
            "result_checksum": data.get("result_checksum"),
            "result_size_bytes": data.get("result_size_bytes"),
            "billing_observations": data.get("billing_observations"),
        }

    async def _handle_engine_death(self, *, error: str) -> None:
        async with self._engine_death_lock:
            self._engine_failure_epoch += 1
            self._record_error(RuntimeError(error))
            await self._mark_engine_unhealthy(error)
            leases = list(self._active_leases.values())
            if not leases:
                await self._restart_engine()
                return
            for lease in leases:
                lease_id = str(lease.get("lease_id"))
                try:
                    disposition = await self._settle_lease_for_engine_death(lease, error=error)
                    if disposition == EngineDeathLeaseDisposition.TERMINAL_FAILED:
                        self._failed_total += 1
                    elif disposition == EngineDeathLeaseDisposition.RESUMED_SUCCESS:
                        self._completed_total += 1
                except Exception as exc:
                    self._record_error(exc)
                    try:
                        await self._scheduler.fail(
                            lease=_legacy_lease_token(lease),
                            reason="gpu_actor_died_terminal_fail_failed",
                            requeue=False,
                            abort_finalize=True,
                        )
                    except Exception:
                        pass
                finally:
                    self._clear_active_lease(lease_id)
            await self._restart_engine()

    async def _mark_running(self, lease: dict[str, Any]) -> None:
        item = _lease_item_wire(lease)
        extra = item.get("extra") if isinstance(item, dict) else {}
        extra = extra if isinstance(extra, dict) else {}
        running_at = time.time()
        queued_at = extra.get("queued_at")
        queue_wait_s = None
        if isinstance(queued_at, (int, float)):
            queue_wait_s = max(0.0, running_at - float(queued_at))
        op = str(item.get("op") or "unknown")
        await self._task_futures.async_mark_running(
            str(item["request_id"]),
            meta={
                "consumer_id": self._config.consumer_id,
                "actor_name": self._config.actor_name,
                "actor_generation": int(self._config.actor_generation),
                "domain_key": self._config.domain_key,
                "replica_id": self._config.replica_id,
                "queue_id": self._config.queue_id,
                "lease_id": _scheduler_lease_token(lease).lease_id,
                "op": op,
                "queue_state": "running",
                "stage": "prefill",
                "queued_at": queued_at,
                "dequeue_at": running_at,
                "running_at": running_at,
                "executor_started_at": running_at,
                "queue_wait_s": queue_wait_s,
            },
        )
        if queue_wait_s is not None:
            record_scheduler_decision_otel(
                op=op,
                backend="model_work",
                queue_kind="model_work_scheduler",
                reason="lease_claimed",
                queue_wait_s=queue_wait_s,
                switched=False,
            )

    async def _renew_until_done(
        self,
        lease: dict[str, Any],
        task: asyncio.Task,
        *,
        execution_timeout_s: float | None = None,
    ) -> None:
        interval_s = max(0.1, min(float(self._config.lease_ttl_s) / 3.0, 10.0))
        started = time.monotonic()
        while not task.done():
            wait_s = interval_s
            if execution_timeout_s is not None:
                elapsed_s = time.monotonic() - started
                remaining_s = float(execution_timeout_s) - elapsed_s
                if remaining_s <= 0:
                    item = _lease_item_wire(lease)
                    op = str(item.get("op") if isinstance(item, dict) else "unknown")
                    raise TimeoutError(
                        f"model work executor timed out after {execution_timeout_s:.1f}s op={op}"
                    )
                wait_s = min(wait_s, remaining_s)
            try:
                async with asyncio.timeout(wait_s):
                    await asyncio.shield(task)
                return
            except asyncio.TimeoutError:
                if (
                    execution_timeout_s is not None
                    and time.monotonic() - started >= float(execution_timeout_s)
                    and not task.done()
                ):
                    item = _lease_item_wire(lease)
                    op = str(item.get("op") if isinstance(item, dict) else "unknown")
                    raise TimeoutError(
                        f"model work executor timed out after {execution_timeout_s:.1f}s op={op}"
                    )
                renew_started = time.monotonic()
                try:
                    result = await self._scheduler.renew(
                        lease=_scheduler_lease_token(lease),
                        lease_ttl_s=self._config.lease_ttl_s,
                    )
                except Exception:
                    self._record_renew_result(
                        ok=False,
                        latency_s=time.monotonic() - renew_started,
                        lease_ttl_s=self._config.lease_ttl_s,
                    )
                    raise
                self._record_renew_result(
                    ok=_scheduler_result_ok(result),
                    latency_s=time.monotonic() - renew_started,
                    lease_ttl_s=self._config.lease_ttl_s,
                )
                if _scheduler_result_ok(result):
                    await self._push_liveness()
                    continue
                raise RuntimeError(f"model work lease renew failed: {result!r}")

    async def _renew_pending_leases_until_done(self, pending_leases: dict[str, LeaseToken]) -> None:
        interval_s = max(0.1, min(float(self._config.lease_ttl_s) / 3.0, 10.0))
        while pending_leases:
            await asyncio.sleep(interval_s)
            for lease_id, token in list(pending_leases.items()):
                renew_started = time.monotonic()
                try:
                    result = await self._scheduler.renew(
                        lease=token,
                        lease_ttl_s=self._config.lease_ttl_s,
                    )
                except Exception as e:
                    self._record_renew_result(
                        ok=False,
                        latency_s=time.monotonic() - renew_started,
                        lease_ttl_s=self._config.lease_ttl_s,
                    )
                    self._record_error(e)
                    logger.warning(
                        "[model_runtime] pending lease renew failed actor=%s lease_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        lease_id,
                        type(e).__name__,
                        e,
                    )
                    continue
                self._record_renew_result(
                    ok=_scheduler_result_ok(result),
                    latency_s=time.monotonic() - renew_started,
                    lease_ttl_s=self._config.lease_ttl_s,
                )
                if _scheduler_result_ok(result):
                    continue

    def _record_renew_result(self, *, ok: bool, latency_s: float, lease_ttl_s: float) -> None:
        latency = max(0.0, float(latency_s))
        self._max_renew_rpc_latency_s = max(self._max_renew_rpc_latency_s, latency)
        if ok:
            self._last_renewed_at = time.time()
            self._renewed_total += 1
            self._consecutive_renew_failures = 0
            self._last_renew_deadline_slack_s = max(0.0, float(lease_ttl_s) - latency)
            return
        self._consecutive_renew_failures += 1

    async def _call_executor(self, lease: dict[str, Any]) -> ExecutorOutcome | None:
        result = await asyncio.to_thread(self._executor, lease)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_executor(self, lease: dict[str, Any]) -> ExecutorOutcome:
        item = _lease_item_wire(lease)
        tracer = get_otel_tracer()
        if tracer is None:
            return await self._call_executor(lease) or ExecutorOutcome(kind="success")
        try:
            from opentelemetry.propagate import extract
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except Exception:
            return await self._call_executor(lease) or ExecutorOutcome(kind="success")

        span_context = None
        extra = item.get("extra") if isinstance(item, dict) else {}
        traceparent = extra.get("_traceparent") if isinstance(extra, dict) else None
        if isinstance(traceparent, str) and traceparent:
            try:
                span_context = extract({"traceparent": traceparent})
            except Exception:
                span_context = None
        with tracer.start_as_current_span(
            "model_runtime.execute",
            kind=SpanKind.INTERNAL,
            context=span_context,
        ) as span:
            span.set_attribute("component", "model_engine_host")
            span.set_attribute("op", str(item.get("op") if isinstance(item, dict) else "unknown"))
            span.set_attribute("request_id", str(item.get("request_id") if isinstance(item, dict) else "unknown"))
            span.set_attribute("domain_key", self._config.domain_key)
            span.set_attribute("replica_id", self._config.replica_id)
            span.set_attribute("actor_name", self._config.actor_name)
            try:
                out = await self._call_executor(lease)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            return out or ExecutorOutcome(kind="success")

    async def _execute_lease(self, lease: dict[str, Any]) -> None:
        item = _lease_item_wire(lease)
        request_id = str(item["request_id"])
        legacy_token = _legacy_lease_token(lease)
        lease_id = legacy_token.lease_id
        token: LeaseToken | None = None
        attempt_id = self._lease_attempt_id(lease)
        self._active_leases[lease_id] = lease
        self._active_request_id = request_id
        self._active_lease_id = lease_id
        self._restore_item_context(lease)
        if not await self._status_is_pending(lease):
            self._clear_active_lease(lease_id)
            return

        try:
            await self._mark_running(lease)
        except Exception as e:
            if self._is_missing_future_state_error(e):
                self._record_error(e)
                await self._finish_non_pending_scheduler_lease(
                    lease,
                    status=FutureStatus.FAILED,
                    error=(
                        "external future state is missing; "
                        "terminating scheduler lease instead of requeueing stale model work"
                    ),
                )
                self._processed_total += 1
                self._failed_total += 1
                self._last_completed_at = time.time()
                self._clear_active_lease(lease_id)
                return
            self._record_error(e)
            try:
                await self._scheduler.fail(
                    lease=legacy_token,
                    reason="mark_running_failed",
                    requeue=True,
                )
                self._requeued_total += 1
            except Exception:
                pass
            self._clear_active_lease(lease_id)
            return

        stale_generation_error = self._recovered_stale_generation_error(lease)
        if stale_generation_error is not None:
            token = _lease_token(lease)
            self._record_error(stale_generation_error)
            try:
                begin_finalize = await self._scheduler.begin_finalize(
                    lease=token,
                    finalize_ttl_s=self._config.lease_ttl_s,
                )
                if _scheduler_result_ok(begin_finalize):
                    await self._scheduler.finish_failure(
                        lease=token,
                        error=f"executor failed: {stale_generation_error}",
                    )
                    self._processed_total += 1
                    self._failed_total += 1
                else:
                    await self._fail_lost_lease_if_still_pending(
                        request_id,
                        "model work scheduler lost stale-generation lease; request must be retried",
                        attempt_id=attempt_id,
                    )
                    self._requeued_total += 1
            except Exception as e:
                self._record_error(e)
                logger.error(
                    "[model_runtime] stale-generation lease failure failed actor=%s request_id=%s error_type=%s error=%s",
                    self._config.actor_name,
                    request_id,
                    type(e).__name__,
                    e,
                )
                try:
                    await self._scheduler.fail(
                        lease=token,
                        reason="stale_runtime_generation_finalize_failed",
                        requeue=True,
                    )
                    self._requeued_total += 1
                except Exception:
                    pass
            self._clear_active_lease(lease_id)
            return

        task: asyncio.Task | None = None
        lease_engine_failure_epoch = int(self._engine_failure_epoch)
        try:
            executor_started_at = time.time()
            with model_work_execution_context(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_buffer=None,
            ):
                task = asyncio.create_task(self._run_executor(lease))
                await self._renew_until_done(
                    lease,
                    task,
                    execution_timeout_s=self._execution_timeout_s_for_lease(lease),
                )
                outcome = await task
            executor_done_at = time.time()
            try:
                await self._task_futures.async_update_meta(
                    request_id,
                    {
                        "executor_done_at": executor_done_at,
                        "executor_exec_s": max(0.0, executor_done_at - executor_started_at),
                    },
                )
            except Exception:
                pass
            if outcome.kind in {"retryable_failure", "fatal_backend_death"}:
                error = outcome.error or f"executor returned {outcome.kind}"
                if outcome.kind == "fatal_backend_death":
                    if int(self._engine_failure_epoch) != lease_engine_failure_epoch:
                        return
                    await self._handle_engine_death(error=error)
                    return
                self._record_error(RuntimeError(error))
                failed = await self._scheduler.fail(
                    lease=legacy_token,
                    reason="executor_retryable_failure",
                    requeue=True,
                    abort_finalize=True,
                )
                if not _scheduler_result_ok(failed):
                    logger.warning(
                        "[model_runtime] retryable executor failure requeue rejected actor=%s request_id=%s result=%s",
                        self._config.actor_name,
                        request_id,
                        failed,
                    )
                self._requeued_total += 1
                return
            if int(self._engine_failure_epoch) != lease_engine_failure_epoch:
                return
            finalization = _finalization_from_outcome(request_id=request_id, outcome=outcome)
            if finalization.request_id != request_id:
                raise RuntimeError(
                    f"model work executor finalized wrong request_id: {finalization.request_id!r}"
                )
            if finalization.kind not in {"resolve", "fail"}:
                raise RuntimeError(f"unknown model work finalization kind: {finalization.kind!r}")
            if not self._task_state_finalize_enabled(lease):
                if finalization.kind == "resolve":
                    await self._task_futures.async_resolve(
                        request_id,
                        finalization.payload,
                        billing_observations=finalization.billing_observations,
                    )
                    failed = await self._scheduler.fail(
                        lease=legacy_token,
                        reason="future_resolved_without_finalize_identity",
                        requeue=False,
                        abort_finalize=True,
                    )
                    if not _scheduler_result_ok(failed):
                        logger.warning(
                            "[model_runtime] legacy lease release rejected after future resolve actor=%s request_id=%s result=%s",
                            self._config.actor_name,
                            request_id,
                            failed,
                        )
                    self._processed_total += 1
                    self._completed_total += 1
                    self._last_completed_at = time.time()
                    self._last_error = None
                    self._last_error_traceback = None
                    return
                await self._task_futures.async_fail(request_id, str(finalization.payload))
                failed = await self._scheduler.fail(
                    lease=legacy_token,
                    reason="future_failed",
                    requeue=False,
                )
                if not _scheduler_result_ok(failed):
                    logger.warning(
                        "[model_runtime] legacy lease fail rejected after future failure actor=%s request_id=%s result=%s",
                        self._config.actor_name,
                        request_id,
                        failed,
                    )
                self._processed_total += 1
                self._failed_total += 1
                self._last_error = f"future failed: {finalization.payload}"
                self._last_error_traceback = None
                self._last_completed_at = time.time()
                return
            token = _lease_token(lease)
            finalization_started_at = time.time()
            try:
                await self._task_futures.async_update_meta(
                    request_id,
                    {
                        "stage": "finalizing",
                        "finalization_started_at": finalization_started_at,
                    },
                )
            except Exception:
                pass
            begin_finalize = await self._scheduler.begin_finalize(
                lease=token,
                finalize_ttl_s=self._config.lease_ttl_s,
                staged_payload_path=(
                    self._staged_payload_path_for_lease(lease)
                    if finalization.kind == "resolve" and self._task_state_finalize_enabled(lease)
                    else None
                ),
            )
            if not _scheduler_result_ok(begin_finalize):
                logger.warning(
                    "[model_runtime] lease finalize rejected actor=%s request_id=%s result=%s",
                    self._config.actor_name,
                    request_id,
                    begin_finalize,
                )
                try:
                    failed = await self._fail_lost_lease_if_still_pending(
                        request_id,
                        "model work scheduler lost active lease; request must be retried",
                        attempt_id=attempt_id,
                    )
                    if failed:
                        self._failed_total += 1
                except Exception as e:
                    logger.error(
                        "[model_runtime] lost-lease task_futures.fail failed actor=%s request_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        request_id,
                        type(e).__name__,
                        e,
                    )
                self._requeued_total += 1
                return
            try:
                finalization_done_at = time.time()
                try:
                    await self._task_futures.async_update_meta(
                        request_id,
                        {
                            "finalization_done_at": finalization_done_at,
                            "finalization_s": max(0.0, finalization_done_at - finalization_started_at),
                        },
                    )
                except Exception:
                    pass
                if finalization.kind == "resolve":
                    payload_meta = await self._write_success_payload(
                        lease,
                        payload=finalization.payload,
                    )
                    finished = await self._scheduler.finish_success(
                        lease=token,
                        result_path=str(payload_meta["path"]),
                        result_checksum=str(payload_meta["checksum"]),
                        result_size_bytes=int(payload_meta["size_bytes"]),
                        billing_observations=finalization.billing_observations,
                    )
                else:
                    finished = await self._scheduler.finish_failure(
                        lease=token,
                        error=str(finalization.payload),
                    )
                if not _scheduler_result_ok(finished):
                    logger.warning(
                        "[model_runtime] lease finish rejected actor=%s request_id=%s result=%s",
                        self._config.actor_name,
                        request_id,
                        finished,
                    )
                    self._requeued_total += 1
                    return
            except Exception:
                try:
                    await self._scheduler.fail(
                        lease=token,
                        reason="task_state_finalize_failed",
                        requeue=True,
                        abort_finalize=True,
                    )
                    self._requeued_total += 1
                except Exception:
                    pass
                return
            if finalization.kind == "resolve":
                self._processed_total += 1
                self._completed_total += 1
                self._last_completed_at = time.time()
                self._last_error = None
                self._last_error_traceback = None
                return
            self._processed_total += 1
            self._failed_total += 1
            self._last_error = f"future failed: {finalization.payload}"
            self._last_error_traceback = None
            self._last_completed_at = time.time()
        except asyncio.CancelledError:
            await self._cancel_executor_task(task, reason="runtime_cancelled")
            try:
                await self._scheduler.fail(
                    lease=legacy_token,
                    reason="model_runtime_cancelled",
                    requeue=True,
                    abort_finalize=True,
                )
                self._requeued_total += 1
            except Exception:
                pass
            raise
        except Exception as e:
            await self._cancel_executor_task(task, reason=type(e).__name__)
            self._record_error(e)
            logger.error(
                "[model_runtime] executor failed actor=%s request_id=%s op=%s error_type=%s failure_reason=%s",
                self._config.actor_name,
                request_id,
                item.get("op"),
                type(e).__name__,
                classify_failure_reason(e),
            )
            try:
                if isinstance(e, TimeoutError):
                    token = _lease_token(lease)
                    begin_finalize = await self._scheduler.begin_finalize(
                        lease=token,
                        finalize_ttl_s=self._config.lease_ttl_s,
                    )
                    if not _scheduler_result_ok(begin_finalize):
                        logger.warning(
                            "[model_runtime] timeout lease finalize rejected actor=%s request_id=%s result=%s",
                            self._config.actor_name,
                            request_id,
                            begin_finalize,
                        )
                        failed = await self._scheduler.fail(
                            lease=legacy_token,
                            reason="executor_retryable_failure",
                            requeue=True,
                            abort_finalize=True,
                        )
                        if not _scheduler_result_ok(failed):
                            logger.warning(
                                "[model_runtime] timeout fallback requeue rejected actor=%s request_id=%s result=%s",
                                self._config.actor_name,
                                request_id,
                                failed,
                            )
                        self._requeued_total += 1
                        return
                    finished = await self._scheduler.finish_failure(
                        lease=token,
                        error=f"executor failed: {e}",
                    )
                    if not _scheduler_result_ok(finished):
                        logger.warning(
                            "[model_runtime] timeout lease finish rejected actor=%s request_id=%s result=%s",
                            self._config.actor_name,
                            request_id,
                            finished,
                        )
                        self._requeued_total += 1
                        return
                    self._processed_total += 1
                    self._failed_total += 1
                    return
                failed = await self._scheduler.fail(
                    lease=legacy_token,
                    reason="executor_retryable_failure",
                    requeue=True,
                    abort_finalize=True,
                )
                if not _scheduler_result_ok(failed):
                    logger.warning(
                        "[model_runtime] executor exception requeue rejected actor=%s request_id=%s result=%s",
                        self._config.actor_name,
                        request_id,
                        failed,
                    )
                self._requeued_total += 1
                return
            except Exception as e2:
                logger.error(
                    "[model_runtime] executor exception requeue failed actor=%s request_id=%s error_type=%s error=%s",
                    self._config.actor_name,
                    request_id,
                    type(e2).__name__,
                    e2,
                )
                try:
                    await self._scheduler.fail(
                        lease=legacy_token,
                        reason="task_state_finalize_failed",
                        requeue=True,
                        abort_finalize=True,
                    )
                    self._requeued_total += 1
                except Exception:
                    pass
                return
            self._processed_total += 1
            self._failed_total += 1
        finally:
            self._clear_active_lease(lease_id)
