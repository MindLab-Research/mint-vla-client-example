from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .async_ray_control import async_get_ray_ref
from .model_actor_placement import model_actor_placement_reconciler
from .model_work_scheduler import ModelReplicaRegistration, ModelWorkSchedulerClient, model_work_scheduler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelActorSpec:
    domain_key: str
    replica_id: str = "replica-0"
    base_model: str | None = None
    actor_name: str | None = None
    launcher_key: str = "legacy_vllm"
    worker_index: int | None = None
    node_pin: str | None = None
    node_pins: tuple[str, ...] = ()
    gpu_count: int | None = None
    enabled: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return str(self.domain_key), str(self.replica_id)

    def normalized_actor_name(self) -> str:
        if self.actor_name:
            return str(self.actor_name)
        return default_model_actor_name(self.domain_key, self.replica_id)

    def normalized_node_pins(self) -> list[str]:
        pins = [str(pin) for pin in self.node_pins if str(pin).strip()]
        if self.node_pin and str(self.node_pin).strip() and str(self.node_pin) not in pins:
            pins.append(str(self.node_pin))
        return pins


RuntimeFactory = Callable[[ModelActorSpec, int], Any | Awaitable[Any]]
NodeInventory = Callable[[], set[str] | None | Awaitable[set[str] | None]]
SchedulerSync = Callable[[list[ModelReplicaRegistration]], Any | Awaitable[Any]]
OrphanPlacementGroupCleaner = Callable[[dict[tuple[str, str], ModelActorSpec]], Any | Awaitable[Any]]
PlacementReconciler = Callable[[dict[tuple[str, str], ModelActorSpec]], Any | Awaitable[dict[str, Any]]]


def domain_key_for_vllm_base_model(base_model: str) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    return f"vllm:{model}"


def _normalize_megatron_domain_key(base_model: str) -> str:
    model_name = str(base_model or "").split("/")[-1]
    model_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return f"megatron_{model_name}" if model_name else "megatron_model"


def domain_key_for_training_base_model(base_model: str) -> str:
    model = str(base_model).strip()
    if not model:
        raise ValueError("base_model is required")
    try:
        from .model_registry import get_model_config

        if bool(getattr(get_model_config(model), "is_moe", False)):
            return f"megatron:{_normalize_megatron_domain_key(model)}"
    except Exception:
        logger.debug("training domain model config lookup failed for %s", model, exc_info=True)
    return f"training:{model}"


def domain_key_for_internal_control() -> str:
    return "internal:control"


def default_model_actor_name(domain_key: str, replica_id: str) -> str:
    raw = f"mint_model_actor_{domain_key}_{replica_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")


def _replica_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"replica-{int(value or 0)}"


def _base_model_from_spec(spec: ModelActorSpec) -> str | None:
    if spec.base_model:
        return str(spec.base_model)
    if spec.domain_key.startswith("vllm:"):
        model = spec.domain_key.removeprefix("vllm:").strip()
        return model or None
    return None


def _replica_int(replica_id: str) -> int:
    raw = str(replica_id).strip()
    if raw.startswith("replica-"):
        raw = raw.removeprefix("replica-")
    try:
        return int(raw)
    except Exception:
        return 0


def _placement_env_for_spec(spec: ModelActorSpec) -> dict[str, str]:
    base_model = _base_model_from_spec(spec)
    if not base_model or spec.gpu_count is None:
        return {}
    if spec.worker_index is not None:
        payload = {
            base_model: {
                "replica": _replica_int(spec.replica_id),
                "worker_index": int(spec.worker_index),
                "gpu_count": int(spec.gpu_count),
            }
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {
            "MINT_MODEL_PLACEMENT_JSON": raw,
            "MINT_VLLM_MODEL_PLACEMENT_JSON": raw,
            "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        }
    node_pins = spec.normalized_node_pins()
    if len(node_pins) != 1:
        return {}
    placement_raw = json.dumps(
        {
            base_model: {
                "replica": _replica_int(spec.replica_id),
                "node_ip": node_pins[0],
                "gpu_count": int(spec.gpu_count),
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    pinned_raw = json.dumps({base_model: node_pins[0]}, sort_keys=True, separators=(",", ":"))
    nodes_raw = json.dumps({base_model: node_pins}, sort_keys=True, separators=(",", ":"))
    return {
        "MINT_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_VLLM_MODEL_PLACEMENT_JSON": placement_raw,
        "MINT_MODEL_ACTOR_REPLICA_ID": spec.replica_id,
        "MINT_VLLM_PINNED_NODE_IP_JSON": pinned_raw,
        "MINT_VLLM_MODEL_NODE_IPS_JSON": nodes_raw,
    }


def _spec_from_obj(obj: Any) -> ModelActorSpec:
    if isinstance(obj, str):
        return ModelActorSpec(domain_key=domain_key_for_vllm_base_model(obj), base_model=obj)
    if not isinstance(obj, dict):
        raise TypeError(f"model actor spec must be dict or str, got {type(obj)}")

    base_model = obj.get("base_model") or obj.get("model") or obj.get("model_id")
    domain_key = obj.get("domain_key")
    if domain_key is None:
        if base_model is None:
            raise ValueError(f"model actor spec missing domain_key/base_model: {obj!r}")
        domain_key = domain_key_for_vllm_base_model(str(base_model))
    raw_node_pins = obj.get("node_pins")
    if raw_node_pins is None:
        raw_node_pins = obj.get("node_pin")
    if isinstance(raw_node_pins, str):
        node_pins = tuple(pin.strip() for pin in raw_node_pins.split(",") if pin.strip())
    else:
        node_pins = tuple(str(pin) for pin in (raw_node_pins or []) if str(pin).strip())
    worker_index = obj.get("worker_index")
    return ModelActorSpec(
        domain_key=str(domain_key),
        replica_id=_replica_id(obj.get("replica_id", obj.get("replica", 0))),
        base_model=None if base_model is None else str(base_model),
        actor_name=None if obj.get("actor_name") is None else str(obj["actor_name"]),
        launcher_key=str(obj.get("launcher_key") or "legacy_vllm"),
        worker_index=None if worker_index is None else int(worker_index),
        node_pin=None if obj.get("node_pin") is None else str(obj["node_pin"]),
        node_pins=node_pins,
        gpu_count=None if obj.get("gpu_count") is None else int(obj["gpu_count"]),
        enabled=bool(obj.get("enabled", True)),
    )


def desired_specs_from_env() -> list[ModelActorSpec]:
    def _with_internal_control(specs: list[ModelActorSpec]) -> list[ModelActorSpec]:
        enabled = str(os.environ.get("MINT_MODEL_ACTOR_INTERNAL_CONTROL", "1")).strip().lower()
        if enabled in ("0", "false", "no", "n", "off"):
            return specs
        domain_key = domain_key_for_internal_control()
        if any(spec.domain_key == domain_key for spec in specs):
            return specs
        return [
            *specs,
            ModelActorSpec(
                domain_key=domain_key,
                launcher_key="internal_control",
                gpu_count=0,
            ),
        ]

    raw = os.environ.get("MINT_MODEL_ACTOR_DESIRED_JSON", "").strip()
    if raw:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            items = payload.get("models") or payload.get("actors") or payload.get("items")
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError("MINT_MODEL_ACTOR_DESIRED_JSON must be a list or contain models/actors/items")
        return _with_internal_control([_spec_from_obj(item) for item in items])

    legacy_raw = os.environ.get("MINT_MODEL_RUNTIME_DESIRED_JSON", "").strip()
    if legacy_raw:
        payload = json.loads(legacy_raw)
        if isinstance(payload, dict):
            items = payload.get("models") or payload.get("runtimes") or payload.get("items")
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError("MINT_MODEL_RUNTIME_DESIRED_JSON must be a list or contain models/runtimes/items")
        return _with_internal_control([_spec_from_obj(item) for item in items])

    persistent = os.environ.get("MINT_PERSISTENT_MODELS", "").strip()
    if not persistent:
        return _with_internal_control([])
    specs: list[ModelActorSpec] = []
    for model in (item.strip() for item in persistent.split(",")):
        if not model:
            continue
        specs.append(_spec_from_obj(model))
        specs.append(
            ModelActorSpec(
                domain_key=domain_key_for_training_base_model(model),
                base_model=model,
                launcher_key="training",
            )
        )
    return _with_internal_control(specs)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    try:
        return await async_get_ray_ref(value, timeout_s=10.0)
    except TypeError:
        return value


async def _invoke_actor(actor: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return await _maybe_await(remote(*args, **kwargs))
    return await _maybe_await(method(*args, **kwargs))


async def _default_runtime_factory(spec: ModelActorSpec, generation: int) -> Any:
    from .model_runtime_actor import get_or_create_model_runtime_actor

    base_model = _base_model_from_spec(spec)
    return get_or_create_model_runtime_actor(
        domain_key=spec.domain_key,
        replica_id=spec.replica_id,
        actor_name=spec.normalized_actor_name(),
        actor_generation=int(generation),
        base_model=base_model,
        # RuntimeActor executes claimed work sequentially and only renews the
        # active lease. Keep claims single-item until multi-lease renewal and
        # concurrent execution are implemented.
        max_claim=1,
        runtime_env_extra=_placement_env_for_spec(spec),
    )


class ModelActorSupervisor:
    def __init__(
        self,
        *,
        specs: list[ModelActorSpec] | None = None,
        runtime_factory: RuntimeFactory | None = None,
        node_inventory: NodeInventory | None = None,
        scheduler: ModelWorkSchedulerClient | None = None,
        scheduler_sync: SchedulerSync | None = None,
        orphan_pg_cleaner: OrphanPlacementGroupCleaner | None = None,
        placement_reconciler: PlacementReconciler | None = None,
    ) -> None:
        self._desired: dict[tuple[str, str], ModelActorSpec] = {}
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._node_inventory = node_inventory
        self._scheduler = scheduler or model_work_scheduler
        self._scheduler_sync = scheduler_sync
        self._orphan_pg_cleaner = orphan_pg_cleaner
        self._placement_reconciler = placement_reconciler or model_actor_placement_reconciler
        self._actors: dict[tuple[str, str], Any] = {}
        self._generations: dict[tuple[str, str], int] = {}
        self._states: dict[tuple[str, str], dict[str, Any]] = {}
        self._reconcile_total = 0
        self._created_total = 0
        self._restarted_total = 0
        self._blocked_total = 0
        self._busy_recycle_skipped_total = 0
        self._scheduler_sync_failures_total = 0
        self._placement_reconcile_failures_total = 0
        self._placement_reclaimed_total = 0
        self._last_reconcile_at: float | None = None
        self._last_scheduler_sync_at: float | None = None
        self._last_placement_reconcile: dict[str, Any] | None = None
        for spec in specs or []:
            self.set_desired(spec)

    def set_desired(self, spec: ModelActorSpec) -> None:
        if not spec.domain_key:
            raise ValueError("domain_key is required")
        if not spec.replica_id:
            raise ValueError("replica_id is required")
        key = spec.key
        self._desired[key] = spec
        self._states.setdefault(
            key,
            {
                "domain_key": spec.domain_key,
                "replica_id": spec.replica_id,
                "state": "desired",
                "actor_name": spec.normalized_actor_name(),
                "generation": 0,
                "crash_count": 0,
                "last_error": None,
                "last_action": "set_desired",
                "last_action_at": time.time(),
            },
        )

    def remove_desired(self, *, domain_key: str, replica_id: str) -> None:
        self._desired.pop(_key(domain_key, replica_id), None)

    async def _available_nodes(self) -> set[str] | None:
        if self._node_inventory is None:
            return None
        nodes = await _maybe_await(self._node_inventory())
        if nodes is None:
            return None
        return {str(node) for node in nodes}

    async def _node_pins_possible(
        self,
        spec: ModelActorSpec,
        *,
        resolved_node_pins: list[str] | None = None,
    ) -> bool:
        pins = list(resolved_node_pins) if resolved_node_pins is not None else spec.normalized_node_pins()
        if not pins:
            return True
        nodes = await self._available_nodes()
        if nodes is None:
            return True
        return all(pin in nodes for pin in pins)

    async def _actor_health(self, actor: Any) -> dict[str, Any]:
        out = await _invoke_actor(actor, "health_snapshot")
        if not isinstance(out, dict):
            raise TypeError(f"runtime actor health_snapshot returned non-dict: {type(out)}")
        return out

    def _next_generation(self, key: tuple[str, str]) -> int:
        previous = int(self._generations.get(key, 0))
        now = int(time.time())
        return max(now, previous + 1)

    async def _create_runtime(self, spec: ModelActorSpec, *, reason: str) -> Any:
        key = spec.key
        generation = self._next_generation(key)
        actor = await _maybe_await(self._runtime_factory(spec, generation))
        start_result = await _invoke_actor(actor, "start")
        if isinstance(start_result, dict) and start_result.get("running") is False:
            raise RuntimeError(f"runtime actor did not start: {start_result!r}")
        self._actors[key] = actor
        self._generations[key] = generation
        self._created_total += 1
        if reason != "missing":
            self._restarted_total += 1
        self._states[key] = {
            "domain_key": spec.domain_key,
            "replica_id": spec.replica_id,
            "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
            "state": "healthy",
            "actor_name": spec.normalized_actor_name(),
            "launcher_key": spec.launcher_key,
            "generation": generation,
            "consumer_id": consumer_id_for_replica(spec.domain_key, spec.replica_id, generation),
            "crash_count": int(self._states.get(key, {}).get("crash_count", 0)),
            "last_error": None,
            "last_action": f"create:{reason}",
            "last_action_at": time.time(),
            "node_pins": spec.normalized_node_pins(),
            "worker_index": spec.worker_index,
            "gpu_count": spec.gpu_count,
        }
        return actor

    def _replica_registration_for_state(
        self,
        spec: ModelActorSpec,
        state: dict[str, Any],
    ) -> ModelReplicaRegistration:
        generation = int(state.get("generation") or self._generations.get(spec.key, 0) or 0)
        status = str(state.get("state") or "starting")
        if status == "running":
            status = "healthy"
        if status in {"blocked", "disabled", "dead", "unhealthy"}:
            claim_status = status
        else:
            claim_status = "healthy" if status == "healthy" else "starting"
        return ModelReplicaRegistration(
            domain_key=spec.domain_key,
            replica_id=spec.replica_id,
            consumer_id=str(
                state.get("consumer_id")
                or consumer_id_for_replica(spec.domain_key, spec.replica_id, generation)
            ),
            generation=generation,
            status=claim_status,
            queue_id=queue_id_for_replica(spec.domain_key, spec.replica_id),
            capacity=max(1, int(spec.gpu_count or 1)),
            actor_name=spec.normalized_actor_name(),
            node_pins=spec.normalized_node_pins(),
            updated_at=float(state.get("last_action_at") or time.time()),
        )

    async def _sync_scheduler(self) -> None:
        registrations = [
            self._replica_registration_for_state(spec, self._states.get(key, {}))
            for key, spec in sorted(self._desired.items())
        ]
        try:
            if self._scheduler_sync is not None:
                await _maybe_await(self._scheduler_sync(registrations))
            else:
                await self._scheduler.sync_replicas(registrations)
            self._last_scheduler_sync_at = time.time()
        except Exception as e:
            self._scheduler_sync_failures_total += 1
            logger.warning(
                "[model_actor_supervisor] scheduler sync failed: %s: %s",
                type(e).__name__,
                e,
            )

    async def reconcile_once(self) -> dict[str, Any]:
        self._reconcile_total += 1
        self._last_reconcile_at = time.time()
        if self._orphan_pg_cleaner is not None:
            await _maybe_await(self._orphan_pg_cleaner(dict(self._desired)))
        placement_out: dict[str, Any] = {}
        if self._placement_reconciler is not None:
            try:
                candidate = await _maybe_await(self._placement_reconciler(dict(self._desired)))
                placement_out = candidate if isinstance(candidate, dict) else {"ok": True, "result": candidate}
                self._last_placement_reconcile = dict(placement_out)
                self._placement_reclaimed_total += int(placement_out.get("reclaimed_total") or 0)
            except Exception as e:
                self._placement_reconcile_failures_total += 1
                placement_out = {"ok": False, "error": f"{type(e).__name__}: {e}", "blocked": {}}
                self._last_placement_reconcile = dict(placement_out)
                logger.warning(
                    "[model_actor_supervisor] placement reconcile failed: %s: %s",
                    type(e).__name__,
                    e,
                )
        placement_blocked = placement_out.get("blocked") if isinstance(placement_out, dict) else {}
        if not isinstance(placement_blocked, dict):
            placement_blocked = {}
        placement_node_pins = placement_out.get("node_pins") if isinstance(placement_out, dict) else {}
        if not isinstance(placement_node_pins, dict):
            placement_node_pins = {}

        results: dict[str, Any] = {}
        for key, spec in sorted(self._desired.items()):
            label = _label(key)
            resolved_node_pins = [
                str(pin)
                for pin in placement_node_pins.get(label, spec.normalized_node_pins())
                if str(pin).strip()
            ]
            if not spec.enabled:
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "disabled",
                    "actor_name": spec.normalized_actor_name(),
                    "last_action": "disabled",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            placement_error = placement_blocked.get(label)
            if placement_error:
                self._blocked_total += 1
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "blocked",
                    "actor_name": spec.normalized_actor_name(),
                    "node_pins": resolved_node_pins,
                    "worker_index": spec.worker_index,
                    "gpu_count": spec.gpu_count,
                    "last_error": f"placement blocked: {placement_error}",
                    "last_action": "blocked:placement",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            if not await self._node_pins_possible(spec, resolved_node_pins=resolved_node_pins):
                self._blocked_total += 1
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "blocked",
                    "actor_name": spec.normalized_actor_name(),
                    "node_pins": resolved_node_pins,
                    "worker_index": spec.worker_index,
                    "gpu_count": spec.gpu_count,
                    "last_error": f"node pin unavailable: {','.join(resolved_node_pins)}",
                    "last_action": "blocked:node_pin_unavailable",
                    "last_action_at": time.time(),
                }
                results[label] = self._states[key]
                continue

            actor = self._actors.get(key)
            if actor is None:
                await self._create_runtime(spec, reason="missing")
                results[label] = self._states[key]
                continue

            try:
                health = await self._actor_health(actor)
                state = "healthy" if bool(health.get("running", True)) else "unhealthy"
                if state == "unhealthy":
                    previous = self._states.get(key, {})
                    crash_count = int(previous.get("crash_count", 0)) + 1
                    self._states[key] = {
                        **previous,
                        "domain_key": spec.domain_key,
                        "replica_id": spec.replica_id,
                        "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                        "state": "unhealthy",
                        "actor_name": spec.normalized_actor_name(),
                        "launcher_key": spec.launcher_key,
                        "generation": int(self._generations.get(key, 0)),
                        "consumer_id": consumer_id_for_replica(
                            spec.domain_key,
                            spec.replica_id,
                            int(self._generations.get(key, 0)),
                        ),
                        "health": health,
                        "crash_count": crash_count,
                        "last_error": "runtime actor not running",
                        "last_action": "health_unhealthy",
                        "last_action_at": time.time(),
                        "node_pins": resolved_node_pins,
                        "worker_index": spec.worker_index,
                        "gpu_count": spec.gpu_count,
                    }
                    self._actors.pop(key, None)
                    await self._create_runtime(spec, reason="unhealthy")
                    self._states[key]["crash_count"] = crash_count
                    results[label] = self._states[key]
                    continue
                self._states[key] = {
                    **self._states.get(key, {}),
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "queue_id": queue_id_for_replica(spec.domain_key, spec.replica_id),
                    "state": state,
                    "actor_name": spec.normalized_actor_name(),
                    "launcher_key": spec.launcher_key,
                    "generation": int(self._generations.get(key, 0)),
                    "consumer_id": consumer_id_for_replica(
                        spec.domain_key,
                        spec.replica_id,
                        int(self._generations.get(key, 0)),
                    ),
                    "health": health,
                    "last_error": None if state == "healthy" else "runtime actor not running",
                    "last_action": "health_check",
                    "last_action_at": time.time(),
                    "node_pins": resolved_node_pins,
                    "worker_index": spec.worker_index,
                    "gpu_count": spec.gpu_count,
                }
                results[label] = self._states[key]
            except Exception as e:
                previous = self._states.get(key, {})
                crash_count = int(previous.get("crash_count", 0)) + 1
                self._states[key] = {
                    **previous,
                    "domain_key": spec.domain_key,
                    "replica_id": spec.replica_id,
                    "state": "dead",
                    "actor_name": spec.normalized_actor_name(),
                    "crash_count": crash_count,
                    "last_error": f"{type(e).__name__}: {e}",
                    "last_action": "health_failed",
                    "last_action_at": time.time(),
                    "node_pins": resolved_node_pins,
                    "worker_index": spec.worker_index,
                    "gpu_count": spec.gpu_count,
                }
                self._actors.pop(key, None)
                await self._create_runtime(spec, reason="dead")
                self._states[key]["crash_count"] = crash_count
                results[label] = self._states[key]

        await self._sync_scheduler()
        return {"ok": True, "replicas": results, "snapshot": self.snapshot()}

    async def recycle(self, *, domain_key: str, replica_id: str, force: bool = False) -> dict[str, Any]:
        key = _key(domain_key, replica_id)
        actor = self._actors.get(key)
        spec = self._desired.get(key)
        if spec is None:
            return {"ok": False, "domain_key": domain_key, "replica_id": replica_id, "reason": "unknown_replica"}
        if actor is None:
            return {"ok": True, "domain_key": domain_key, "replica_id": replica_id, "recycled": False, "reason": "missing"}
        health: dict[str, Any] = {}
        try:
            health = await self._actor_health(actor)
        except Exception:
            health = {}
        if not force and health.get("active_request_id"):
            self._busy_recycle_skipped_total += 1
            return {
                "ok": False,
                "domain_key": domain_key,
                "replica_id": replica_id,
                "reason": "busy",
                "active_request_id": health.get("active_request_id"),
            }
        try:
            await _invoke_actor(actor, "shutdown")
        except Exception as e:
            logger.warning(
                "[model_actor_supervisor] runtime shutdown failed domain=%s replica=%s error_type=%s error=%s",
                domain_key,
                replica_id,
                type(e).__name__,
                e,
            )
        self._actors.pop(key, None)
        self._states[key] = {
            **self._states.get(key, {}),
            "domain_key": domain_key,
            "replica_id": replica_id,
            "state": "dead",
            "actor_name": spec.normalized_actor_name(),
            "last_action": "recycle",
            "last_action_at": time.time(),
        }
        await self._sync_scheduler()
        return {"ok": True, "domain_key": domain_key, "replica_id": replica_id, "recycled": True}

    def snapshot(self) -> dict[str, Any]:
        replicas: dict[str, dict[str, Any]] = {}
        domains: dict[str, dict[str, Any]] = {}
        for key in sorted(set(self._states) | set(self._desired)):
            spec = self._desired.get(key)
            state = dict(self._states.get(key, {}))
            domain_key, replica_id = key
            state.setdefault("domain_key", domain_key)
            state.setdefault("replica_id", replica_id)
            state.setdefault("queue_id", queue_id_for_replica(domain_key, replica_id))
            if spec is not None:
                if not state.get("node_pins"):
                    state["node_pins"] = spec.normalized_node_pins()
                state.update(
                    {
                        "base_model": spec.base_model,
                        "desired_actor_name": spec.normalized_actor_name(),
                        "desired_enabled": bool(spec.enabled),
                        "launcher_key": spec.launcher_key,
                        "worker_index": spec.worker_index,
                        "gpu_count": spec.gpu_count,
                    }
                )
            replicas[_label(key)] = state
            domain = domains.setdefault(domain_key, {"replicas": 0, "healthy": 0, "unhealthy": 0})
            domain["replicas"] += 1
            if state.get("state") == "healthy":
                domain["healthy"] += 1
            elif state.get("state") in {"dead", "unhealthy", "blocked"}:
                domain["unhealthy"] += 1
        return {
            "desired_total": int(len(self._desired)),
            "managed_total": int(len(self._actors)),
            "domain_total": int(len(domains)),
            "reconcile_total": int(self._reconcile_total),
            "created_total": int(self._created_total),
            "restarted_total": int(self._restarted_total),
            "blocked_total": int(self._blocked_total),
            "busy_recycle_skipped_total": int(self._busy_recycle_skipped_total),
            "scheduler_sync_failures_total": int(self._scheduler_sync_failures_total),
            "placement_reconcile_failures_total": int(self._placement_reconcile_failures_total),
            "placement_reclaimed_total": int(self._placement_reclaimed_total),
            "last_reconcile_at": self._last_reconcile_at,
            "last_scheduler_sync_at": self._last_scheduler_sync_at,
            "last_placement_reconcile": self._last_placement_reconcile,
            "domains": domains,
            "replicas": replicas,
        }

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        _ = timeout_s
        return self.snapshot()


def _key(domain_key: str, replica_id: str) -> tuple[str, str]:
    return str(domain_key), str(replica_id)


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def queue_id_for_replica(domain_key: str, replica_id: str) -> str:
    return f"{domain_key}::{replica_id}"


def consumer_id_for_replica(domain_key: str, replica_id: str, generation: int) -> str:
    return f"{domain_key}::{replica_id}::generation::{int(generation)}"


model_actor_supervisor = ModelActorSupervisor(specs=desired_specs_from_env())
