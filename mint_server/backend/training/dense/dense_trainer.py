"""PEFT (dense) trainer actor helpers.

This module replaces the previous DenseTrainerPool tracking dictionary with:
- deterministic actor naming
- a small inflight-creation guard keyed by Ray actor identity
- ModelActorSupervisor inventory publication for lifecycle and observability
"""

from __future__ import annotations

import os
import threading
import structlog
import re
import time
from dataclasses import dataclass
from typing import Any

import ray
from ray.exceptions import GetTimeoutError, RayActorError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import mint_server.backend.ray_cluster.ray_kill as ray_kill
from mint_server.backend.ray_cluster.ray_placement_groups import get_named_placement_group, get_or_create_named_placement_group
from mint_server.backend.actors.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor
from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor
from mint_server.backend.actors.node_placement import parse_model_gpu_placement
from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE

logger = structlog.get_logger(__name__)


PERSISTENT_DENSE_NAMESPACE = RAY_NAMESPACE
# Naming clarifies actor role/back-end.
PERSISTENT_DENSE_ACTOR_PREFIX = "mint_dense_"
PFS_PYTHONPATH_DENSE = PFS_PYTHONPATH

DEFAULT_MAX_LORA_RANK = 64
DEFAULT_NUM_GPUS = 1

DENSE_POISONED_KEY = "poisoned"
DENSE_POISON_REASON_KEY = "poison_reason"
DENSE_POISONED_AT_KEY = "poisoned_at"
DENSE_POISONED_SESSION_KEY = "poisoned_session_id"
DENSE_LAST_FATAL_OP_KEY = "last_fatal_op"
DENSE_LAST_FATAL_REQUEST_ID_KEY = "last_fatal_request_id"

_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_inflight_errors: dict[str, str] = {}


@dataclass(frozen=True)
class DenseTrainerHandle:
    actor: Any
    actor_name: str
    base_model: str
    max_lora_rank: int


def _normalize_model_key_for_actor_name(model_key: str) -> str:
    k = (model_key or "").strip()
    if k.startswith("/"):
        k = k.split("/")[-1]
    else:
        # Keep org in "org/model" for uniqueness, but make it Ray-name safe.
        k = k.replace("/", "__")
    k = (
        k.replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .lower()
    )
    return k or "unknown"


def _make_actor_name(*, model_key: str, max_rank: int) -> str:
    model_name = _normalize_model_key_for_actor_name(model_key)
    return f"{PERSISTENT_DENSE_ACTOR_PREFIX}{model_name}"


def _sanitize_pg_component(value: str | None) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def _pg_name(actor_name: str) -> str:
    return f"{actor_name}_{_sanitize_pg_component(PERSISTENT_DENSE_NAMESPACE)}_pg"


def _base_dense_metadata(*, actual_rank: int, max_lora_rank: int, model_key: str | None) -> dict[str, object]:
    return {
        "max_lora_rank": int(max_lora_rank),
        "actual_rank": int(actual_rank),
        "model_key": model_key,
        DENSE_POISONED_KEY: False,
        DENSE_POISON_REASON_KEY: None,
        DENSE_POISONED_AT_KEY: None,
        DENSE_POISONED_SESSION_KEY: None,
        DENSE_LAST_FATAL_OP_KEY: None,
        DENSE_LAST_FATAL_REQUEST_ID_KEY: None,
    }


def _poison_metadata(
    metadata: dict[str, object] | None,
    *,
    reason: str,
    session_id: str | None,
    fatal_op: str | None = None,
    request_id: str | None = None,
) -> dict[str, object]:
    poisoned = dict(metadata or {})
    already_poisoned = bool(poisoned.get(DENSE_POISONED_KEY))

    poisoned[DENSE_POISONED_KEY] = True
    if not already_poisoned or poisoned.get(DENSE_POISON_REASON_KEY) is None:
        poisoned[DENSE_POISON_REASON_KEY] = str(reason)
    if not already_poisoned or poisoned.get(DENSE_POISONED_AT_KEY) is None:
        poisoned[DENSE_POISONED_AT_KEY] = time.time()
    if not already_poisoned or poisoned.get(DENSE_POISONED_SESSION_KEY) is None:
        poisoned[DENSE_POISONED_SESSION_KEY] = session_id
    if not already_poisoned or poisoned.get(DENSE_LAST_FATAL_OP_KEY) is None:
        poisoned[DENSE_LAST_FATAL_OP_KEY] = None if fatal_op is None else str(fatal_op)
    if not already_poisoned or poisoned.get(DENSE_LAST_FATAL_REQUEST_ID_KEY) is None:
        poisoned[DENSE_LAST_FATAL_REQUEST_ID_KEY] = None if request_id is None else str(request_id)
    return poisoned


def _dense_reuse_block_reason_from_metadata(metadata: dict | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    if not bool(metadata.get(DENSE_POISONED_KEY)):
        return None
    reason = str(metadata.get(DENSE_POISON_REASON_KEY) or "dense_actor_poisoned")
    return reason


def dense_trainer_reuse_block_reason(actor_name: str) -> str | None:
    entry = get_model_actor_supervisor().get(actor_name)
    if entry is None:
        return None
    return _dense_reuse_block_reason_from_metadata(entry.metadata)


def _preferred_worker_node_ip_for_model(model_key: str | None, base_model: str) -> str | None:
    lookup_keys: list[str] = []
    for key in (model_key, base_model):
        if not key:
            continue
        lookup_keys.extend((key, str(key).lower()))

    for env_name in ("MINT_DENSE_MODEL_PLACEMENT_JSON", "MINT_MODEL_PLACEMENT_JSON"):
        placement = parse_model_gpu_placement(
            raw_json=os.environ.get(env_name),
            lookup_keys=lookup_keys,
            env_var_name=env_name,
            context=f"DenseTrainer placement model={model_key or base_model}",
            replica=0,
        )
        if placement is None:
            continue
        if len(placement.slices) != 1:
            raise RuntimeError(
                f"DenseTrainer placement model={model_key or base_model} expected exactly 1 placement slice, "
                f"got {len(placement.slices)}"
            )
        if placement.total_gpus != 1:
            raise RuntimeError(
                f"DenseTrainer placement model={model_key or base_model} expected exactly 1 GPU, "
                f"got {placement.total_gpus}"
            )
        return placement.slices[0].node_ip
    return None


def _get_or_create_pg(actor_name: str, *, model_key: str | None, base_model: str) -> Any:
    """Get or create the detached 1-GPU placement group for this actor."""
    pg_name = _pg_name(actor_name)
    bundle: dict[str, float] = {"GPU": 1.0, "CPU": 1.0}
    preferred_ip = _preferred_worker_node_ip_for_model(model_key, base_model)
    if preferred_ip:
        bundle[f"node:{preferred_ip}"] = 0.001
        logger.info("dense_trainer_pin", model=model_key or base_model, node_ip=preferred_ip)
    return get_or_create_named_placement_group(
        pg_name,
        namespace=PERSISTENT_DENSE_NAMESPACE,
        bundles=[bundle],
    )


def _remove_pg(actor_name: str) -> None:
    pg_name = _pg_name(actor_name)
    try:
        pg = get_named_placement_group(pg_name, namespace=PERSISTENT_DENSE_NAMESPACE)
    except Exception:
        return
    try:
        ray.util.remove_placement_group(pg)
        logger.warning("removed", placement_group=pg_name, actor_name=actor_name)
    except Exception:
        logger.warning(
            "[dense_trainer] failed remove placement_group=%s actor_name=%s",
            pg_name,
            actor_name,
            exc_info=True,
        )


def clear_dense_trainer_session(session_id: str) -> int:
    """Clear supervisor-owned current_session pointers for dense trainers."""
    return get_model_actor_supervisor().clear_session(session_id, actor_type=ActorType.DENSE)


def retire_dense_trainer(
    *,
    actor_name: str,
    reason: str,
    base_model: str,
    session_id: str | None = None,
    fatal_op: str | None = None,
    request_id: str | None = None,
    namespace: str = PERSISTENT_DENSE_NAMESPACE,
    actor: Any | None = None,
) -> str:
    """Poison a dense trainer so it cannot be reused after fatal GPU failures."""
    from mint_server.backend.observability.runtime_observability import runtime_observability

    pool = get_model_actor_supervisor()
    entry = pool.get(actor_name)
    retire_error: str | None = None
    auxiliary_failures: list[str] = []

    if entry is not None:
        try:
            pool.update_metadata(
                actor_name,
                metadata=_poison_metadata(
                    entry.metadata,
                    reason=reason,
                    session_id=session_id,
                    fatal_op=fatal_op,
                    request_id=request_id,
                ),
                sample_source="dense_retire",
            )
        except Exception:
            auxiliary_failures.append("metadata_update_failed")
            logger.warning(
                "[dense_trainer] failed to persist poison metadata actor_name=%s reason=%s",
                actor_name,
                reason,
                exc_info=True,
            )
    if session_id is not None:
        try:
            pool.clear_session(session_id, actor_type=ActorType.DENSE)
        except Exception:
            auxiliary_failures.append("clear_session_failed")
            logger.warning(
                "[dense_trainer] failed clear_session actor_name=%s session_id=%s",
                actor_name,
                session_id,
                exc_info=True,
            )
    try:
        pool.set_session(actor_name, None)
    except Exception:
        auxiliary_failures.append("clear_binding_failed")
        logger.warning(
            "[dense_trainer] failed to clear bound session actor_name=%s",
            actor_name,
            exc_info=True,
        )

    actor_absent = False
    if actor is None:
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            actor_absent = True
        except Exception:
            retire_error = "actor_lookup_failed"
            logger.warning(
                "[dense_trainer] failed actor lookup during retire actor_name=%s reason=%s",
                actor_name,
                reason,
                exc_info=True,
            )

    if actor is not None:
        try:
            shutdown = getattr(actor, "shutdown", None)
            if shutdown is not None:
                try:
                    ray.get(shutdown.remote(), timeout=30)
                except Exception:
                    pass
            ray_kill.kill(
                actor,
                reason="dense_trainer_retire",
                actor_name=actor_name,
                namespace=namespace,
                no_restart=True,
                verify_absent=True,
                base_model=base_model,
                session_id=session_id,
                retire_reason=reason,
            )
            actor_absent = True
        except Exception:
            retire_error = "kill_failed"
            logger.warning(
                "[dense_trainer] failed retire actor_name=%s reason=%s",
                actor_name,
                reason,
                exc_info=True,
            )

    _remove_pg(actor_name)

    if actor_absent:
        try:
            pool.unregister(actor_name)
        except Exception:
            auxiliary_failures.append("unregister_failed")
            logger.warning(
                "[dense_trainer] failed unregister after retire actor_name=%s reason=%s",
                actor_name,
                reason,
                exc_info=True,
            )
    elif retire_error is None:
        retire_error = "actor_still_present"

    retire_outcome = retire_error or "ok"
    incident_context: dict[str, object] = {"outcome": retire_outcome}
    if auxiliary_failures:
        incident_context["auxiliary_failures"] = list(auxiliary_failures)

    runtime_observability.record_dense_actor_retire(base_model=base_model, outcome=retire_outcome)
    runtime_observability.record_training_incident(
        kind="dense_actor_retire",
        base_model=base_model,
        backend="peft",
        op=str(fatal_op or "retire"),
        status="ok" if retire_outcome == "ok" else "error",
        failure_class="none" if retire_outcome == "ok" else retire_outcome,
        actor_name=actor_name,
        request_id=None if request_id is None else str(request_id),
        session_id=None if session_id is None else str(session_id),
        detail=str(reason),
        context=incident_context,
    )
    for auxiliary_failure in auxiliary_failures:
        runtime_observability.record_training_incident(
            kind="dense_actor_retire_auxiliary_failure",
            base_model=base_model,
            backend="peft",
            op=str(fatal_op or "retire"),
            status="error",
            failure_class=auxiliary_failure,
            actor_name=actor_name,
            request_id=None if request_id is None else str(request_id),
            session_id=None if session_id is None else str(session_id),
            detail=str(reason),
            context={"retire_outcome": retire_outcome},
        )
    return retire_outcome


def get_or_create_dense_trainer(
    *,
    training_worker_cls,
    base_model: str,
    model_key: str | None = None,
    lora_rank: int,
    learning_rate: float,
    session_id: str | None = None,
    max_lora_rank: int | None = None,
) -> DenseTrainerHandle:
    """Get an existing dense trainer actor or create a new one.

    The returned actor is published through ModelActorSupervisor inventory metadata:
      - max_lora_rank
      - actual_rank (current session rank)
    """
    wait_timeout_s = float(os.environ.get("MINT_DENSE_INFLIGHT_WAIT_S", "1800"))
    effective_max_rank = max(int(lora_rank), int(max_lora_rank or 0), int(DEFAULT_MAX_LORA_RANK))
    name_key = model_key or base_model
    actor_name = _make_actor_name(model_key=name_key, max_rank=effective_max_rank)
    key = actor_name

    while True:
        with _lock:
            inflight = _inflight.get(key)
            if inflight is None:
                inflight = threading.Event()
                _inflight[key] = inflight
                _inflight_errors.pop(key, None)
                creator = True
            else:
                creator = False

        if not creator:
            if not inflight.wait(timeout=wait_timeout_s):
                raise TimeoutError(
                    f"dense_trainer inflight create timed out after {wait_timeout_s}s "
                    f"actor_name={actor_name} base_model={base_model}"
                )
            with _lock:
                err = _inflight_errors.get(key)
            if err:
                raise RuntimeError(err)
            continue

        try:
            bind_decision = "create"

            from mint_server.backend.core.model_registry import is_topology_desired_model
            from mint_server.backend.observability.runtime_observability import runtime_observability

            is_topology_desired = is_topology_desired_model(base_model)

            reuse_block_reason = dense_trainer_reuse_block_reason(actor_name)

            try:
                actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                if reuse_block_reason is not None:
                    bind_decision = "recreate_poisoned"
                    retire_outcome = retire_dense_trainer(
                        actor_name=actor_name,
                        actor=actor,
                        reason=f"reuse_blocked:{reuse_block_reason}",
                        base_model=base_model,
                        session_id=session_id,
                    )
                    if retire_outcome != "ok":
                        raise RuntimeError(
                            f"dense_trainer retire failed before recreate "
                            f"actor_name={actor_name} outcome={retire_outcome}"
                        )
                    actor = None
                if actor is not None:
                    bind_decision = "reuse"
                    # Heartbeat: if busy, we still treat as alive.
                    heartbeat: dict[str, Any] | None = None
                    try:
                        heartbeat_result = ray.get(actor.heartbeat.remote(), timeout=5)
                        if isinstance(heartbeat_result, dict):
                            heartbeat = heartbeat_result
                    except RayActorError:
                        raise
                    except GetTimeoutError:
                        pass
                    observed_max_rank = None
                    if heartbeat is not None:
                        observed_max_rank = heartbeat.get("max_lora_rank")
                    if not isinstance(observed_max_rank, int) or isinstance(observed_max_rank, bool):
                        entry = get_model_actor_supervisor().get(actor_name)
                        metadata = entry.metadata if entry is not None else None
                        if isinstance(metadata, dict):
                            observed_max_rank = metadata.get("max_lora_rank")
                    if not isinstance(observed_max_rank, int) or isinstance(observed_max_rank, bool):
                        raise RuntimeError(
                            f"dense_trainer cannot verify max_lora_rank for actor_name={actor_name}; "
                            "refusing to reuse a shape-opaque detached actor"
                        )
                    if int(observed_max_rank) != int(effective_max_rank):
                        bind_decision = "recreate_max_lora_rank_mismatch"
                        retire_outcome = retire_dense_trainer(
                            actor_name=actor_name,
                            actor=actor,
                            reason=(
                                "max_lora_rank_mismatch:"
                                f"observed={int(observed_max_rank)} expected={int(effective_max_rank)}"
                            ),
                            base_model=base_model,
                            session_id=session_id,
                        )
                        if retire_outcome != "ok":
                            raise RuntimeError(
                                f"dense_trainer retire failed before recreate "
                                f"actor_name={actor_name} outcome={retire_outcome}"
                            )
                        actor = None
            except RayActorError:
                # Dead actor name: prove the Ray name is absent or kill the remaining named actor before recreate.
                try:
                    actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                except ValueError:
                    actor = None
                except Exception as e:
                    raise RuntimeError(
                        f"dense_trainer dead actor lookup failed before recreate actor_name={actor_name}"
                    ) from e
                if actor is not None:
                    try:
                        ray_kill.kill(
                            actor,
                            reason="dense_trainer_dead",
                            actor_name=actor_name,
                            namespace=PERSISTENT_DENSE_NAMESPACE,
                            no_restart=True,
                            verify_absent=True,
                            base_model=base_model,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"dense_trainer dead actor cleanup failed before recreate actor_name={actor_name}"
                        ) from e
                _remove_pg(actor_name)
                actor = None
            except ValueError:
                actor = None

            if actor is None:
                from mint_server.config import actor_runtime_env_vars, otel_env_vars
                runtime_env = {
                    "env_vars": actor_runtime_env_vars(
                        pythonpath=PFS_PYTHONPATH_DENSE,
                        extra={
                            "USE_TORCH": "1",
                            "USE_TF": "0",
                            "USE_FLAX": "0",
                            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                            **otel_env_vars(),
                        },
                        include_ray_attach_hints=False,
                    )
                }
                pg = _get_or_create_pg(actor_name, model_key=name_key, base_model=base_model)
                actor = training_worker_cls.options(
                    name=actor_name,
                    namespace=PERSISTENT_DENSE_NAMESPACE,
                    lifetime="detached",
                    num_gpus=1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,
                    ),
                    runtime_env=runtime_env,
                ).remote(
                    base_model=base_model,
                    lora_rank=effective_max_rank,
                    learning_rate=learning_rate,
                )

                # Register immediately (creating=True) to account for GPU usage
                # and protect the actor from the placement reconciler's
                # undesired-GPU-actor reaper during the (up to 600s) init window.
                # The richer publish below marks it ready (creating=False) once
                # __ray_ready__ completes. Mirrors the megatron create path.
                # Best-effort: the authoritative publish after readiness remains
                # load-bearing, so a transient supervisor RPC failure here must
                # not abort trainer creation.
                try:
                    publish_backend_model_actor(
                        BackendModelActorLaunch(
                            actor_name=actor_name,
                            actor_type=ActorType.DENSE,
                            num_gpus=DEFAULT_NUM_GPUS,
                            actor_handle=actor,
                            namespace=PERSISTENT_DENSE_NAMESPACE,
                            base_model=base_model,
                            session_id=session_id,
                            protected=is_topology_desired,
                            metadata={
                                **_base_dense_metadata(
                                    actual_rank=int(lora_rank),
                                    max_lora_rank=effective_max_rank,
                                    model_key=name_key,
                                ),
                            },
                        ),
                        ready=False,
                        refresh_observability=False,
                    )
                except Exception as reg_error:
                    logger.warning(
                        "[dense_trainer] early creating-registration failed actor_name=%s "
                        "error_type=%s error=%s; actor unprotected during init",
                        actor_name,
                        type(reg_error).__name__,
                        reg_error,
                    )

                init_timeout_s = float(os.environ.get("MINT_DENSE_ACTOR_INIT_TIMEOUT_S", "600"))
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=init_timeout_s)
                except GetTimeoutError:
                    try:
                        get_model_actor_supervisor().unregister(actor_name)
                    except Exception:
                        pass
                    try:
                        ray_kill.kill(
                            actor,
                            reason="dense_trainer_init_timeout",
                            actor_name=actor_name,
                            namespace=PERSISTENT_DENSE_NAMESPACE,
                            no_restart=True,
                            base_model=base_model,
                            timeout_s=init_timeout_s,
                        )
                    except Exception:
                        pass
                    _remove_pg(actor_name)
                    raise

            entry = publish_backend_model_actor(BackendModelActorLaunch(
                actor_name=actor_name,
                actor_type=ActorType.DENSE,
                num_gpus=DEFAULT_NUM_GPUS,
                actor_handle=actor,
                namespace=PERSISTENT_DENSE_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
                protected=is_topology_desired,
                metadata={
                    **_base_dense_metadata(
                        actual_rank=int(lora_rank),
                        max_lora_rank=effective_max_rank,
                        model_key=name_key,
                    ),
                },
            ),
                observability_wins=True,
            )
            entry.current_session = session_id
            runtime_observability.record_dense_actor_bind_decision(base_model=base_model, decision=bind_decision)
            runtime_observability.record_training_incident(
                kind="dense_actor_bind_decision",
                base_model=base_model,
                backend="peft",
                op="bind",
                status="ok",
                failure_class="none",
                actor_name=actor_name,
                session_id=None if session_id is None else str(session_id),
                detail=bind_decision,
                context={"protected": bool(is_topology_desired)},
            )

            return DenseTrainerHandle(
                actor=actor,
                actor_name=actor_name,
                base_model=base_model,
                max_lora_rank=effective_max_rank,
            )
        except Exception as e:
            with _lock:
                _inflight_errors[key] = str(e)
            raise
        finally:
            with _lock:
                ev = _inflight.pop(key, None)
            if ev is not None:
                ev.set()


def remove_dense_trainers(*, base_model: str, kill_actor: bool = True) -> int:
    """Remove dense trainer actors for base_model from supervisor inventory and optionally Ray."""
    pool = get_model_actor_supervisor()
    targets = [e for e in pool.iter_entries() if e.actor_type == ActorType.DENSE and e.base_model == base_model]
    if not targets:
        return 0

    killed = 0
    for e in targets:
        if kill_actor:
            try:
                actor = ray.get_actor(e.actor_name, namespace=e.namespace)
                try:
                    ray.get(actor.shutdown.remote(), timeout=30)
                except Exception:
                    pass
                ray_kill.kill(
                    actor,
                    reason="dense_trainer_remove",
                    actor_name=e.actor_name,
                    namespace=e.namespace,
                    no_restart=True,
                    base_model=e.base_model,
                )
            except Exception:
                pass
            _remove_pg(e.actor_name)
        pool.unregister(e.actor_name)
        killed += 1
    return killed
