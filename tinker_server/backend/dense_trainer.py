"""PEFT (dense) trainer actor helpers (ResourcePool is the source of truth).

This module replaces the previous DenseTrainerPool tracking dictionary with:
- deterministic actor naming
- a small inflight-creation guard (per base_model)
- ResourcePool registration for lifecycle and eviction
"""

from __future__ import annotations

import os
import threading
import logging
from dataclasses import dataclass

import ray

from . import ray_kill
from .resource_pool import ActorType, get_resource_pool
from ..config import PFS_PYTHONPATH, RAY_NAMESPACE, otel_env_vars

logger = logging.getLogger(__name__)


PERSISTENT_DENSE_NAMESPACE = RAY_NAMESPACE
# Naming clarifies actor role/back-end.
PERSISTENT_DENSE_ACTOR_PREFIX = "peft_trainer_"
PFS_PYTHONPATH_DENSE = PFS_PYTHONPATH

DEFAULT_MAX_LORA_RANK = 64
DEFAULT_NUM_GPUS = 1

_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_inflight_errors: dict[str, str] = {}


@dataclass(frozen=True)
class DenseTrainerHandle:
    actor: ray.actor.ActorHandle
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
    return f"{PERSISTENT_DENSE_ACTOR_PREFIX}{model_name}_maxr{int(max_rank)}"


def _pg_name(actor_name: str) -> str:
    return f"{actor_name}_pg"


def _get_or_create_pg(actor_name: str) -> ray.util.placement_group.PlacementGroup:
    """Ensure a detached 1-GPU placement group exists for this actor."""
    pg_name = _pg_name(actor_name)
    try:
        pg = ray.util.get_placement_group(pg_name)
    except Exception:
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}],
            strategy="PACK",
            name=pg_name,
            lifetime="detached",
        )
    ray.get(pg.ready())
    return pg


def _remove_pg(actor_name: str) -> None:
    pg_name = _pg_name(actor_name)
    try:
        pg = ray.util.get_placement_group(pg_name)
    except Exception:
        return
    try:
        ray.util.remove_placement_group(pg)
    except Exception:
        pass


def clear_dense_trainer_session(session_id: str) -> int:
    """Clear ResourcePool current_session pointers for dense trainers."""
    return get_resource_pool().clear_session(session_id, actor_type=ActorType.DENSE)


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

    The returned actor is registered in ResourcePool with metadata:
      - max_lora_rank
      - actual_rank (current session rank)
    """
    wait_timeout_s = float(os.environ.get("MINT_DENSE_INFLIGHT_WAIT_S", "1800"))
    key = base_model

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
                raise TimeoutError(f"dense_trainer inflight create timed out after {wait_timeout_s}s base_model={base_model}")
            with _lock:
                err = _inflight_errors.get(key)
            if err:
                raise RuntimeError(err)
            continue

        try:
            effective_max_rank = max(int(lora_rank), int(max_lora_rank or 0), int(DEFAULT_MAX_LORA_RANK))
            name_key = model_key or base_model
            actor_name = _make_actor_name(model_key=name_key, max_rank=effective_max_rank)

            pool = get_resource_pool()
            from .model_registry import is_persistent_model

            is_persistent = is_persistent_model(base_model)

            try:
                actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                # Heartbeat: if busy, we still treat as alive.
                try:
                    ray.get(actor.heartbeat.remote(), timeout=5)
                except ray.exceptions.RayActorError:
                    raise
                except ray.exceptions.GetTimeoutError:
                    pass
            except ray.exceptions.RayActorError:
                # Dead actor name: best-effort kill to free name, then recreate.
                try:
                    actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                    ray_kill.kill(
                        actor,
                        reason="dense_trainer_dead",
                        actor_name=actor_name,
                        namespace=PERSISTENT_DENSE_NAMESPACE,
                        no_restart=True,
                        base_model=base_model,
                    )
                except Exception:
                    pass
                _remove_pg(actor_name)
                actor = None
            except ValueError:
                actor = None

            if actor is None:
                pool.ensure_gpus_available(DEFAULT_NUM_GPUS)
                from ..config import actor_runtime_env_vars, otel_env_vars
                runtime_env = {
                    "env_vars": actor_runtime_env_vars(
                        pythonpath=PFS_PYTHONPATH_DENSE,
                        extra={
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        **otel_env_vars(),
                        },
                    )
                }
                pg = _get_or_create_pg(actor_name)
                actor = training_worker_cls.options(
                    name=actor_name,
                    namespace=PERSISTENT_DENSE_NAMESPACE,
                    lifetime="detached",
                    num_gpus=1,
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,
                    ),
                    runtime_env=runtime_env,
                ).remote(
                    base_model=base_model,
                    lora_rank=effective_max_rank,
                    learning_rate=learning_rate,
                )

                init_timeout_s = float(os.environ.get("MINT_DENSE_ACTOR_INIT_TIMEOUT_S", "600"))
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=init_timeout_s)
                except ray.exceptions.GetTimeoutError:
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

            entry = pool.register(
                actor_name=actor_name,
                actor_type=ActorType.DENSE,
                num_gpus=DEFAULT_NUM_GPUS,
                actor_handle=actor,
                namespace=PERSISTENT_DENSE_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
                protected=is_persistent,
                metadata={
                    "max_lora_rank": effective_max_rank,
                    "actual_rank": int(lora_rank),
                    "model_key": name_key,
                },
            )
            pool.mark_ready(actor_name)
            entry.current_session = session_id

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
    """Remove dense trainer actors for base_model from ResourcePool (and optionally Ray)."""
    pool = get_resource_pool()
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
