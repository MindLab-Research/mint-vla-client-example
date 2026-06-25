"""Qwen3.6-27B trainer actor lifecycle management.

Mirrors :mod:`dense_trainer` structure: deterministic actor naming,
inflight-creation guard, ModelActorSupervisor inventory publication.
Uses isolated transformers v5 + peft >=0.19.0 PYTHONPATH via runtime_env.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import structlog
import ray
from ray.exceptions import GetTimeoutError, RayActorError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import mint_server.backend.ray_cluster.ray_kill as ray_kill
from mint_server.backend.actors.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor
from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor
from mint_server.backend.ray_cluster.ray_placement_groups import get_named_placement_group, get_or_create_named_placement_group
from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE
from mint_server.config import config as server_config
from mint_server.ray.runtime_env import join_pythonpath, sanitize_worker_pythonpath

logger = structlog.get_logger(__name__)

PERSISTENT_NAMESPACE = RAY_NAMESPACE
PERSISTENT_ACTOR_PREFIX = "mint_qwen36_"
DEFAULT_MAX_LORA_RANK = 64
DEFAULT_NUM_GPUS = 2

# Isolated dependency directory for transformers v5 + peft >=0.19.0
QWEN36_DEPS_PATH = os.environ.get(
    "MINT_QWEN36_DEPS_PATH",
    "/vePFS-Mindverse/share/mint/dev/runtime/gpu_rl/qwen36-deps",
)

_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_inflight_errors: dict[str, str] = {}


@dataclass(frozen=True)
class Qwen36TrainerHandle:
    actor: Any
    actor_name: str
    base_model: str
    max_lora_rank: int


def _normalize_model_key(model_key: str) -> str:
    k = (model_key or "").strip()
    if k.startswith("/"):
        k = k.split("/")[-1]
    else:
        k = k.replace("/", "__")
    k = re.sub(r"[^0-9A-Za-z_]+", "_", k).strip("_").lower()
    return k or "unknown"


def _make_actor_name(*, model_key: str) -> str:
    return f"{PERSISTENT_ACTOR_PREFIX}{_normalize_model_key(model_key)}"


def _pg_name(actor_name: str) -> str:
    return f"{actor_name}_pg"


def _get_or_create_pg(actor_name: str) -> Any:
    pg_name = _pg_name(actor_name)
    bundle: dict[str, float] = {"GPU": 1.0, "CPU": 2.0}
    return get_or_create_named_placement_group(
        pg_name,
        namespace=PERSISTENT_NAMESPACE,
        bundles=[bundle],
    )


def _remove_pg(actor_name: str) -> None:
    pg_name = _pg_name(actor_name)
    try:
        pg = get_named_placement_group(pg_name, namespace=PERSISTENT_NAMESPACE)
        ray.util.remove_placement_group(pg)
    except Exception:
        pass


def _build_runtime_env() -> dict[str, Any]:
    """Build runtime_env with isolated transformers v5 PYTHONPATH.

    qwen36-deps is prepended so `import transformers` finds v5 first.
    torch/accelerate/numpy fall through to the global site-packages.
    """
    env_root = os.environ.get("PFS_RUNTIME_ENV_ROOT")
    global_pythonpath = sanitize_worker_pythonpath(PFS_PYTHONPATH, env_root=env_root)

    # Prepend isolated deps
    pythonpath = join_pythonpath(QWEN36_DEPS_PATH, global_pythonpath)

    from mint_server.config import actor_runtime_env_vars, otel_env_vars

    env_vars = actor_runtime_env_vars(
        pythonpath=pythonpath,
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
    return {"env_vars": env_vars}


def get_or_create_qwen36_trainer(
    *,
    base_model: str,
    model_key: str | None = None,
    lora_rank: int,
    learning_rate: float,
    session_id: str | None = None,
    max_lora_rank: int | None = None,
) -> Qwen36TrainerHandle:
    """Get an existing Qwen3.6 trainer actor or create a new one.

    Same lifecycle pattern as ``get_or_create_dense_trainer``:
    - Check for existing detached actor by name
    - Create new if absent or dead
    - Publish through ModelActorSupervisor
    """
    wait_timeout_s = float(os.environ.get("MINT_QWEN36_INFLIGHT_WAIT_S", "1800"))
    effective_max_rank = max(int(lora_rank), int(max_lora_rank or 0), int(DEFAULT_MAX_LORA_RANK))
    name_key = model_key or base_model
    actor_name = _make_actor_name(model_key=name_key)
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
                    f"qwen36_trainer inflight create timed out after {wait_timeout_s}s "
                    f"actor_name={actor_name} base_model={base_model}"
                )
            with _lock:
                err = _inflight_errors.get(key)
            if err:
                raise RuntimeError(err)
            continue

        try:
            # Try to reuse existing actor
            try:
                actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
                # Health check
                try:
                    ray.get(actor.heartbeat.remote(), timeout=10)
                except RayActorError:
                    # Dead, kill and recreate
                    try:
                        ray_kill.kill(actor, reason="qwen36_dead", actor_name=actor_name,
                                      namespace=PERSISTENT_NAMESPACE, no_restart=True)
                    except Exception:
                        pass
                    _remove_pg(actor_name)
                    actor = None
                except GetTimeoutError:
                    # Busy, reuse
                    pass

                if actor is not None:
                    # Verify max_lora_rank matches (matches dense_trainer.py pattern)
                    heartbeat: dict[str, Any] | None = None
                    try:
                        hb_result = ray.get(actor.heartbeat.remote(), timeout=5)
                        if isinstance(hb_result, dict):
                            heartbeat = hb_result
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
                            f"qwen36_trainer cannot verify max_lora_rank for actor_name={actor_name}; "
                            "refusing to reuse a shape-opaque detached actor"
                        )
                    if int(observed_max_rank) != int(effective_max_rank):
                        logger.warning(
                            "qwen36_trainer_rank_mismatch_recreating",
                            actor_name=actor_name,
                            observed=observed_max_rank,
                            expected=effective_max_rank,
                        )
                        try:
                            ray_kill.kill(actor, reason="qwen36_rank_mismatch",
                                          actor_name=actor_name,
                                          namespace=PERSISTENT_NAMESPACE, no_restart=True)
                        except Exception:
                            pass
                        _remove_pg(actor_name)
                        actor = None

                if actor is not None:
                    entry = publish_backend_model_actor(BackendModelActorLaunch(
                        actor_name=actor_name,
                        actor_type=ActorType.DENSE,  # reuse DENSE type for 1-GPU trainer
                        num_gpus=DEFAULT_NUM_GPUS,
                        actor_handle=actor,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=base_model,
                        session_id=session_id,
                    ))
                    entry.current_session = session_id
                    return Qwen36TrainerHandle(
                        actor=actor,
                        actor_name=actor_name,
                        base_model=base_model,
                        max_lora_rank=effective_max_rank,
                    )
            except ValueError:
                actor = None

            # Create new actor
            from mint_server.backend.training.qwen36.qwen36_trainer import Qwen36TrainingWorker

            runtime_env = _build_runtime_env()
            pg = _get_or_create_pg(actor_name)

            actor = Qwen36TrainingWorker.options(
                name=actor_name,
                namespace=PERSISTENT_NAMESPACE,
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

            # Register early for GPU accounting
            try:
                publish_backend_model_actor(BackendModelActorLaunch(
                    actor_name=actor_name,
                    actor_type=ActorType.DENSE,
                    num_gpus=DEFAULT_NUM_GPUS,
                    actor_handle=actor,
                    namespace=PERSISTENT_NAMESPACE,
                    base_model=base_model,
                    session_id=session_id,
                ), ready=False)
            except Exception:
                pass

            init_timeout_s = float(os.environ.get("MINT_QWEN36_ACTOR_INIT_TIMEOUT_S", "600"))
            try:
                ray.get(actor.__ray_ready__.remote(), timeout=init_timeout_s)
            except GetTimeoutError:
                try:
                    ray_kill.kill(actor, reason="qwen36_init_timeout", actor_name=actor_name,
                                  namespace=PERSISTENT_NAMESPACE, no_restart=True)
                except Exception:
                    pass
                _remove_pg(actor_name)
                raise

            entry = publish_backend_model_actor(BackendModelActorLaunch(
                actor_name=actor_name,
                actor_type=ActorType.DENSE,
                num_gpus=DEFAULT_NUM_GPUS,
                actor_handle=actor,
                namespace=PERSISTENT_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
            ))
            entry.current_session = session_id

            logger.info("qwen36_trainer_created", actor_name=actor_name, model=base_model)

            return Qwen36TrainerHandle(
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


def remove_qwen36_trainers(*, base_model: str, kill_actor: bool = True) -> int:
    """Remove Qwen3.6 trainer actors for base_model."""
    pool = get_model_actor_supervisor()
    targets = [
        e for e in pool.iter_entries()
        if e.actor_type == ActorType.DENSE
        and e.base_model == base_model
        and e.actor_name.startswith(PERSISTENT_ACTOR_PREFIX)
    ]
    if not targets:
        return 0

    killed = 0
    for e in targets:
        if kill_actor:
            try:
                actor = ray.get_actor(e.actor_name, namespace=e.namespace)
                ray_kill.kill(actor, reason="qwen36_remove", actor_name=e.actor_name,
                              namespace=e.namespace, no_restart=True, base_model=e.base_model)
            except Exception:
                pass
            _remove_pg(e.actor_name)
        pool.unregister(e.actor_name)
        killed += 1
    return killed
