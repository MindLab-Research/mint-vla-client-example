from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from ..config import PFS_PYTHONPATH, actor_runtime_env_vars, apply_detached_actor_resources, otel_env_vars
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
from .control_plane_contracts import as_task_ledger
from .execution_context import ExecutionContext, bind_execution_context
from .model_actor_supervisor import (
    _is_ray_get_timeout_error,
    consumer_id_for_replica,
    queue_id_for_replica,
)
from .model_work_scheduler import ModelWorkSchedulerClient, model_work_scheduler
from .model_work_execution_context import ModelWorkFinalizeBuffer, model_work_execution_context
from .task_payload_store import TaskPayloadStore
from .task_state_store import FutureStatus, task_futures, task_state_store

logger = logging.getLogger(__name__)

ModelWorkExecutor = Callable[[dict[str, Any]], Awaitable[None]]
TokenBudgetProvider = Callable[[], Awaitable[int | None]]
_EXECUTION_BINDINGS: ExecutionContext | None = None
VLLM_TOKEN_BUDGET_RATIO = 0.95
VLLM_TOKEN_BUDGET_REFRESH_S = 60.0
VLLM_TOKEN_BUDGET_QUERY_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class ModelRuntimeActorConfig:
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


def default_model_runtime_actor_name(domain_key: str, replica_id: str) -> str:
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


def get_or_create_model_runtime_actor(
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
    runtime_env_extra: dict[str, str] | None = None,
) -> Any:
    import ray

    name = str(actor_name or default_model_runtime_actor_name(domain_key, replica_id))
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
        if _is_ray_get_timeout_error(e):
            # The actor exists but its health_snapshot did not return within the
            # probe window: it is alive and busy (e.g. mid save_weights_for_sampler),
            # not dead. Killing it here would abort an in-flight lease and trigger a
            # kill/recreate loop. Reuse the existing handle and let the caller retry.
            logger.warning(
                "[model_runtime] existing actor health probe timed out name=%s "
                "error_type=%s error=%s; assuming busy, reusing existing actor",
                name,
                type(e).__name__,
                e,
            )
            try:
                return ray.get_actor(name, namespace=_ray_namespace())
            except Exception:
                pass
        else:
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

    ray_address = env_nonempty(os.environ, "RAY_ADDRESS")
    if ray_address is None:
        raise RuntimeError("RAY_ADDRESS is required")

    remote_cls = ray.remote(num_cpus=0, max_concurrency=64, max_restarts=-1)(ModelRuntimeActor)
    options: dict[str, Any] = {
        "name": name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_task_retries": -1,
        "runtime_env": {
            "env_vars": actor_runtime_env_vars(
                pythonpath=PFS_PYTHONPATH,
                extra={
                    **otel_env_vars(),
                    **(runtime_env_extra or {}),
                },
                include_ray_attach_hints=False,
            )
        },
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
        runtime_env_fingerprint=expected_runtime_env_fingerprint,
        ray_address=ray_address,
    )


async def _default_executor(lease: dict[str, Any]) -> None:
    context = await _ensure_execution_bindings()
    item = lease.get("item")
    if not isinstance(item, dict):
        raise RuntimeError(f"model work lease missing item: {lease!r}")
    op = str(item.get("op") or "")
    from .model_work_dispatch import execute_model_work_item

    with bind_execution_context(context):
        await execute_model_work_item(
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
            component="model_runtime_actor",
        )


async def _ensure_execution_bindings() -> ExecutionContext:
    global _EXECUTION_BINDINGS
    if _EXECUTION_BINDINGS is not None:
        return _EXECUTION_BINDINGS
    from .execution_bindings import initialize_execution_bindings

    _EXECUTION_BINDINGS = ExecutionContext(**await initialize_execution_bindings())
    return _EXECUTION_BINDINGS


class ModelRuntimeActor:
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
        ray_address: str | None = None,
    ) -> None:
        domain = str(domain_key).strip()
        replica = str(replica_id).strip()
        if not domain:
            raise ValueError("domain_key is required")
        if not replica:
            raise ValueError("replica_id is required")
        ray_address_value = str(ray_address or "").strip()
        if ray_address_value:
            os.environ["RAY_ADDRESS"] = ray_address_value
        self._config = ModelRuntimeActorConfig(
            domain_key=domain,
            replica_id=replica,
            actor_name=str(actor_name or default_model_runtime_actor_name(domain, replica)),
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

        self._running = False
        self._draining = False
        self._loop_task: asyncio.Task | None = None
        self._active_request_id: str | None = None
        self._active_lease_id: str | None = None
        self._active_leases: dict[str, str] = {}
        self._started_at = time.time()
        self._last_claimed_at: float | None = None
        self._last_completed_at: float | None = None
        self._last_renewed_at: float | None = None
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
            "active_request_ids": list(self._active_leases.values()),
            "active_lease_ids": list(self._active_leases.keys()),
            "active_lease_count": len(self._active_leases),
            "started_at": float(self._started_at),
            "last_claimed_at": self._last_claimed_at,
            "last_completed_at": self._last_completed_at,
            "last_renewed_at": self._last_renewed_at,
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
            return {"claimed": 0, "executed": 0, "draining": True}
        max_items, token_budget = await self._claim_limits()
        claimed = await self._scheduler.claim(
            domain_key=self._config.domain_key,
            replica_id=self._config.replica_id,
            consumer_id=self._config.consumer_id,
            consumer_generation=self._config.actor_generation,
            max_items=max_items,
            token_budget=token_budget,
            lease_ttl_s=self._config.lease_ttl_s,
        )
        get_claimed = getattr(claimed, "get", None)
        leases = get_claimed("leases") if callable(get_claimed) else None
        if not leases:
            self._empty_polls_total += 1
            self._clear_transient_scheduler_error()
            return {"claimed": 0, "executed": 0}

        self._last_claimed_at = time.time()
        valid_leases = [lease for lease in leases if isinstance(lease, dict)]
        if len(valid_leases) == 1:
            await self._execute_lease(valid_leases[0])
        elif valid_leases and self._executes_leases_concurrently():
            await asyncio.gather(*(self._execute_lease(lease) for lease in valid_leases))
        else:
            await self._execute_leases_sequentially(valid_leases)
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
        pending_lease_ids = {
            str(lease.get("lease_id"))
            for lease in leases
            if isinstance(lease, dict) and lease.get("lease_id") is not None
        }
        renew_task: asyncio.Task | None = None
        if len(pending_lease_ids) > 1:
            renew_task = asyncio.create_task(self._renew_pending_leases_until_done(pending_lease_ids))
        try:
            for lease in leases:
                lease_id = str(lease.get("lease_id"))
                await self._execute_lease(lease)
                pending_lease_ids.discard(lease_id)
        finally:
            pending_lease_ids.clear()
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
        item = lease.get("item") if isinstance(lease, dict) else {}
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
        self._active_request_id = next(iter(self._active_leases.values()), None)
        self._active_lease_id = next(iter(self._active_leases.keys()), None)

    def _record_error(self, e: BaseException) -> None:
        self._last_error = f"{type(e).__name__}: {e}"
        self._last_error_traceback = traceback.format_exc()

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
        item = lease.get("item") if isinstance(lease, dict) else {}
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
        item = lease.get("item") if isinstance(lease, dict) else {}
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
        item = lease["item"]
        request_id = str(item["request_id"])
        lease_id = str(lease["lease_id"])
        try:
            status = await self._task_futures.async_get_status(request_id)
        except KeyError:
            await self._scheduler.complete(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
            )
            return False
        if status is not None and status != FutureStatus.PENDING:
            await self._scheduler.complete(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
            )
            return False
        return True

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
        return lease.get("scheduler_epoch") is not None and bool(lease.get("attempt_id"))

    def _require_task_state_finalize(self, lease: dict[str, Any]) -> None:
        if self._task_state_finalize_enabled(lease):
            return
        raise RuntimeError(
            "model work lease missing TaskStateStore finalize metadata "
            "(attempt_id and scheduler_epoch are required)"
        )

    def _lease_attempt_id(self, lease: dict[str, Any]) -> str | None:
        attempt_id = str(lease.get("attempt_id") or "") or None
        if attempt_id:
            return attempt_id
        item = lease.get("item") if isinstance(lease, dict) else {}
        extra = item.get("extra") if isinstance(item, dict) and isinstance(item.get("extra"), dict) else {}
        return str(extra.get("model_work_attempt_id") or "") or None

    def _payload_attempt_id_for_lease(self, lease: dict[str, Any]) -> str:
        attempt_id = str(lease["attempt_id"])
        lease_id = str(lease["lease_id"])
        return f"{attempt_id}__{lease_id}"

    def _staged_payload_path_for_lease(self, lease: dict[str, Any]) -> str:
        item = lease["item"]
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
        item = lease["item"]
        request_id = str(item["request_id"])
        return await asyncio.to_thread(
            self._payload_store.write_json_payload,
            request_id=request_id,
            attempt_id=self._payload_attempt_id_for_lease(lease),
            payload=payload,
        )

    async def _mark_running(self, lease: dict[str, Any]) -> None:
        item = lease["item"]
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
                "lease_id": str(lease["lease_id"]),
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
                    item = lease.get("item") if isinstance(lease, dict) else {}
                    op = str(item.get("op") if isinstance(item, dict) else "unknown")
                    raise TimeoutError(
                        f"model work executor timed out after {execution_timeout_s:.1f}s op={op}"
                    )
                wait_s = min(wait_s, remaining_s)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait_s)
                return
            except asyncio.TimeoutError:
                if (
                    execution_timeout_s is not None
                    and time.monotonic() - started >= float(execution_timeout_s)
                    and not task.done()
                ):
                    item = lease.get("item") if isinstance(lease, dict) else {}
                    op = str(item.get("op") if isinstance(item, dict) else "unknown")
                    raise TimeoutError(
                        f"model work executor timed out after {execution_timeout_s:.1f}s op={op}"
                    )
                result = await self._scheduler.renew(
                    lease_id=str(lease["lease_id"]),
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    lease_ttl_s=self._config.lease_ttl_s,
                )
                if bool(getattr(result, "get", lambda _key, _default=None: _default)("ok")):
                    self._last_renewed_at = time.time()
                    self._renewed_total += 1
                    continue
                raise RuntimeError(f"model work lease renew failed: {result!r}")

    async def _renew_pending_leases_until_done(self, pending_lease_ids: set[str]) -> None:
        interval_s = max(0.1, min(float(self._config.lease_ttl_s) / 3.0, 10.0))
        while pending_lease_ids:
            await asyncio.sleep(interval_s)
            for lease_id in list(pending_lease_ids):
                try:
                    result = await self._scheduler.renew(
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                        lease_ttl_s=self._config.lease_ttl_s,
                    )
                except Exception as e:
                    self._record_error(e)
                    logger.warning(
                        "[model_runtime] pending lease renew failed actor=%s lease_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        lease_id,
                        type(e).__name__,
                        e,
                    )
                    continue
                if bool(getattr(result, "get", lambda _key, _default=None: _default)("ok")):
                    self._last_renewed_at = time.time()
                    self._renewed_total += 1

    async def _run_executor(self, lease: dict[str, Any]) -> None:
        item = lease.get("item") if isinstance(lease, dict) else {}
        tracer = get_otel_tracer()
        if tracer is None:
            await self._executor(lease)
            return
        try:
            from opentelemetry.propagate import extract
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except Exception:
            await self._executor(lease)
            return

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
            span.set_attribute("component", "model_runtime_actor")
            span.set_attribute("op", str(item.get("op") if isinstance(item, dict) else "unknown"))
            span.set_attribute("request_id", str(item.get("request_id") if isinstance(item, dict) else "unknown"))
            span.set_attribute("domain_key", self._config.domain_key)
            span.set_attribute("replica_id", self._config.replica_id)
            span.set_attribute("actor_name", self._config.actor_name)
            try:
                await self._executor(lease)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    async def _execute_lease(self, lease: dict[str, Any]) -> None:
        item = lease["item"]
        request_id = str(item["request_id"])
        lease_id = str(lease["lease_id"])
        attempt_id = self._lease_attempt_id(lease)
        self._active_leases[lease_id] = request_id
        self._active_request_id = request_id
        self._active_lease_id = lease_id
        self._restore_item_context(lease)
        if not await self._status_is_pending(lease):
            self._clear_active_lease(lease_id)
            return

        try:
            await self._mark_running(lease)
        except Exception as e:
            self._record_error(e)
            try:
                await self._scheduler.fail(
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
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
            self._record_error(stale_generation_error)
            try:
                begin_finalize = await self._scheduler.begin_finalize(
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    finalize_ttl_s=self._config.lease_ttl_s,
                )
                if bool(getattr(begin_finalize, "get", lambda _key, _default=None: _default)("ok")):
                    await self._scheduler.finish_failure(
                        request_id=request_id,
                        lease_id=lease_id,
                        attempt_id=str(lease["attempt_id"]),
                        scheduler_epoch=int(lease["scheduler_epoch"]),
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
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
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                        reason="stale_runtime_generation_finalize_failed",
                        requeue=True,
                    )
                    self._requeued_total += 1
                except Exception:
                    pass
            self._clear_active_lease(lease_id)
            return

        task: asyncio.Task | None = None
        try:
            finalize_buffer = ModelWorkFinalizeBuffer()
            executor_started_at = time.time()
            with model_work_execution_context(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_buffer=finalize_buffer,
            ):
                task = asyncio.create_task(self._run_executor(lease))
                await self._renew_until_done(
                    lease,
                    task,
                    execution_timeout_s=self._execution_timeout_s_for_lease(lease),
                )
                await task
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
            finalization = finalize_buffer.finalization
            if finalization is None:
                raise RuntimeError("model work executor finished without resolving or failing future")
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
                    completed = await self._scheduler.complete(
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                    )
                    if not bool(getattr(completed, "get", lambda _key, _default=None: _default)("ok")):
                        logger.warning(
                            "[model_runtime] legacy lease complete rejected after future resolve actor=%s request_id=%s result=%s",
                            self._config.actor_name,
                            request_id,
                            completed,
                        )
                    self._processed_total += 1
                    self._completed_total += 1
                    self._last_completed_at = time.time()
                    self._last_error = None
                    self._last_error_traceback = None
                    return
                await self._task_futures.async_fail(request_id, str(finalization.payload))
                failed = await self._scheduler.fail(
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    reason="future_failed",
                    requeue=False,
                )
                if not bool(getattr(failed, "get", lambda _key, _default=None: _default)("ok")):
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
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_ttl_s=self._config.lease_ttl_s,
                staged_payload_path=(
                    self._staged_payload_path_for_lease(lease)
                    if finalization.kind == "resolve" and self._task_state_finalize_enabled(lease)
                    else None
                ),
            )
            if not bool(getattr(begin_finalize, "get", lambda _key, _default=None: _default)("ok")):
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
                        request_id=request_id,
                        lease_id=lease_id,
                        attempt_id=str(lease["attempt_id"]),
                        scheduler_epoch=int(lease["scheduler_epoch"]),
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                        result_path=str(payload_meta["path"]),
                        result_checksum=str(payload_meta["checksum"]),
                        result_size_bytes=int(payload_meta["size_bytes"]),
                        billing_observations=finalization.billing_observations,
                    )
                else:
                    finished = await self._scheduler.finish_failure(
                        request_id=request_id,
                        lease_id=lease_id,
                        attempt_id=str(lease["attempt_id"]),
                        scheduler_epoch=int(lease["scheduler_epoch"]),
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                        error=str(finalization.payload),
                    )
                if not bool(getattr(finished, "get", lambda _key, _default=None: _default)("ok")):
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
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
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
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
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
            begin_finalize = await self._scheduler.begin_finalize(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_ttl_s=self._config.lease_ttl_s,
            )
            if not bool(getattr(begin_finalize, "get", lambda _key, _default=None: _default)("ok")):
                logger.warning(
                    "[model_runtime] failure lease finalize rejected actor=%s request_id=%s result=%s",
                    self._config.actor_name,
                    request_id,
                    begin_finalize,
                )
                try:
                    failed = await self._fail_lost_lease_if_still_pending(
                        request_id,
                        "model work scheduler lost active lease after executor failure; request must be retried",
                        attempt_id=attempt_id,
                    )
                    if failed:
                        self._failed_total += 1
                except Exception as e2:
                    logger.error(
                        "[model_runtime] lost-failure-lease task_futures.fail failed actor=%s request_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        request_id,
                        type(e2).__name__,
                        e2,
                    )
                self._requeued_total += 1
                return
            try:
                finished = await self._scheduler.finish_failure(
                    request_id=request_id,
                    lease_id=lease_id,
                    attempt_id=str(lease["attempt_id"]),
                    scheduler_epoch=int(lease["scheduler_epoch"]),
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    error=f"executor failed: {e}",
                )
                if not bool(getattr(finished, "get", lambda _key, _default=None: _default)("ok")):
                    logger.warning(
                        "[model_runtime] failure lease finish rejected actor=%s request_id=%s result=%s",
                        self._config.actor_name,
                        request_id,
                        finished,
                    )
                    self._requeued_total += 1
                    return
            except Exception as e2:
                logger.error(
                    "[model_runtime] task_state failure finalize failed actor=%s request_id=%s error_type=%s error=%s",
                    self._config.actor_name,
                    request_id,
                    type(e2).__name__,
                    e2,
                )
                try:
                    await self._scheduler.fail(
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
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
