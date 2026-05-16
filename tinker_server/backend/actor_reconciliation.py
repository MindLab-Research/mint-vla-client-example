from __future__ import annotations

import logging
import os

from ..config import config
from ..ray_utils import init_ray

logger = logging.getLogger(__name__)


async def cleanup_stale_actors_once() -> dict[str, int]:
    """Cleanup stale Ray actors and register alive ones with ModelActorInventory.

    Returns a summary dict so callers can surface observability and health.
    """
    if config.skip_actor_cleanup:
        logger.info("Skipping actor cleanup (MINT_SKIP_ACTOR_CLEANUP=1)")
        return {"cleaned": 0, "registered": 0}

    import ray

    from . import ray_kill
    from .multi_lora_engine import PERSISTENT_NAMESPACE
    from .model_actor_supervisor import ActorType, get_model_actor_supervisor

    if not ray.is_initialized():
        init_ray(namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    def _normalize_model_part(s: str) -> str:
        return s.lower().replace("-", "_").replace(".", "_")

    def _lookup_model_config(model_part: str):
        try:
            from tinker_server.backend.model_registry import MODEL_CONFIGS
        except Exception:
            return "", None

        needle = _normalize_model_part(model_part)
        for model_name, cfg in MODEL_CONFIGS.items():
            if _normalize_model_part(model_name.split("/")[-1]) == needle:
                return model_name, cfg
        return "", None

    def _openpi_actor_diag(actor) -> tuple[str, str | None, str | None, dict[str, object]]:
        diag = ray.get(actor.describe.remote(), timeout=10)
        if not isinstance(diag, dict):
            raise TypeError(f"openpi actor describe payload must be dict, got {type(diag)}")

        base_model = str(diag.get("base_model", "") or "")
        session_id = diag.get("action_session_id")
        pool_key = diag.get("pool_key")
        if isinstance(pool_key, dict):
            if not base_model:
                base_model = str(pool_key.get("base_model", "") or "")
            session_id = diag.get("current_session_id")

        node_id = diag.get("node_id")
        metadata: dict[str, object] = {
            "worker_module": str(diag.get("worker_module") or ""),
            "actor_id": diag.get("actor_id"),
            "node_ip": diag.get("node_ip"),
            "pid": diag.get("pid"),
            "cuda_visible_devices": diag.get("cuda_visible_devices"),
        }
        if isinstance(pool_key, dict):
            metadata["pool_key"] = dict(pool_key)
        if diag.get("action_session_id") is not None:
            metadata["action_session_id"] = str(diag.get("action_session_id"))
        return base_model, str(session_id) if session_id else None, str(node_id) if node_id else None, metadata

    actors = ray.util.list_named_actors(all_namespaces=True)
    tinker_actors = [a for a in actors if a.get("namespace") == PERSISTENT_NAMESPACE]
    if not tinker_actors:
        logger.info("No actors found in namespace %s", PERSISTENT_NAMESPACE)
        return {"cleaned": 0, "registered": 0}

    logger.info("Found %s actors in namespace %s, checking status...", len(tinker_actors), PERSISTENT_NAMESPACE)
    model_actor_inventory = get_model_actor_supervisor()
    cleaned = 0
    registered = 0
    ready_timeout_s = float(os.environ.get("MINT_STARTUP_RECONCILE_READY_TIMEOUT_S", "5"))

    def _pg_total_gpus(actor_name: str) -> int | None:
        try:
            pg = ray.util.get_placement_group(f"{actor_name}_pg")
            info = ray.util.placement_group_table(pg)
        except Exception:
            return None
        bundles = info.get("bundles") or {}
        total = sum(int(b.get("GPU", 0) or 0) for b in bundles.values() if isinstance(b, dict))
        return total or None

    for actor_info in tinker_actors:
        name = actor_info["name"]
        try:
            actor = ray.get_actor(name, namespace=PERSISTENT_NAMESPACE)
            if name.startswith("dense_trainer_pool_"):
                try:
                    ray_kill.kill(
                        actor,
                        reason="legacy_dense_trainer_prefix",
                        actor_name=name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                    cleaned += 1
                except Exception as kill_err:
                    logger.warning("Failed to kill legacy dense trainer actor %s: %s", name, kill_err)
                try:
                    model_actor_inventory.unregister(name)
                except Exception:
                    pass
                continue

            try:
                ray.get(actor.__ray_ready__.remote(), timeout=ready_timeout_s)
                session_id: str | None = None
                node_id: str | None = None
                metadata: dict[str, object] = {}
                if name.startswith("tinker_vllm_") or name.startswith("multinode_vllm_"):
                    actor_type = ActorType.VLLM
                    base_model = ""
                    num_gpus: int | None = None
                    model_part = name[len("tinker_vllm_"):] if name.startswith("tinker_vllm_") else name[len("multinode_vllm_"):]
                    model_name, cfg = _lookup_model_config(model_part)
                    if cfg is not None:
                        base_model = model_name
                        num_gpus = cfg.total_gpus
                    num_gpus = _pg_total_gpus(name) or num_gpus
                    if num_gpus is None:
                        logger.warning("Skipping restored vLLM actor with unknown GPU count: actor=%s", name)
                        continue
                elif name.startswith("peft_trainer_"):
                    actor_type = ActorType.DENSE
                    num_gpus = 1
                    base_model = ""
                elif name.startswith("megatron_"):
                    actor_type = ActorType.MEGATRON
                    base_model = ""
                    num_gpus: int | None = None
                    model_part = name[len("megatron_"):]
                    model_name, cfg = _lookup_model_config(model_part)
                    if cfg is not None:
                        base_model = model_name
                        num_gpus = cfg.train_gpus
                    try:
                        diag = ray.get(actor.get_diagnostics.remote(), timeout=10)
                        num_gpus = int(diag.get("world_size", num_gpus))
                        base_model = diag.get("base_model", "") or base_model
                    except Exception:
                        pass
                    num_gpus = _pg_total_gpus(name) or num_gpus
                    if num_gpus is None:
                        logger.warning("Skipping restored Megatron actor with unknown GPU count: actor=%s", name)
                        continue
                elif name.startswith("openpi_shared_runtime_") or name.startswith("openpi_action_runtime_"):
                    actor_type = ActorType.OPENPI
                    num_gpus = 1
                    base_model, session_id, node_id, metadata = _openpi_actor_diag(actor)
                else:
                    logger.debug("Unknown actor type for %s, skipping registration", name)
                    continue

                from tinker_server.backend.model_registry import is_persistent_model

                model_actor_inventory.register(
                    actor_name=name,
                    actor_type=actor_type,
                    num_gpus=num_gpus,
                    actor_handle=actor,
                    namespace=PERSISTENT_NAMESPACE,
                    base_model=base_model,
                    session_id=session_id,
                    node_id=node_id,
                    protected=bool(actor_type != ActorType.OPENPI and base_model and is_persistent_model(base_model)),
                    metadata=metadata,
                )
                model_actor_inventory.mark_ready(name)
                registered += 1
                logger.info("Registered existing actor: %s (%s, %s GPUs)", name, actor_type.value, num_gpus)
            except ray.exceptions.RayActorError:
                logger.info("Cleaning up dead actor: %s", name)
                try:
                    ray_kill.kill(
                        actor,
                        reason="startup_cleanup_dead_actor",
                        actor_name=name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                    cleaned += 1
                except Exception as kill_err:
                    logger.warning("Failed to kill actor %s: %s", name, kill_err)
            except ray.exceptions.GetTimeoutError:
                logger.warning("Actor %s __ray_ready__ timed out; registering without marking ready", name)
                try:
                    session_id: str | None = None
                    node_id: str | None = None
                    metadata: dict[str, object] = {"startup_reconcile": "__ray_ready__timeout"}
                    if name.startswith("tinker_vllm_") or name.startswith("multinode_vllm_"):
                        actor_type = ActorType.VLLM
                        num_gpus: int | None = None
                        base_model = ""
                        model_part = name[len("tinker_vllm_"):] if name.startswith("tinker_vllm_") else name[len("multinode_vllm_"):]
                        model_name, cfg = _lookup_model_config(model_part)
                        if cfg is not None:
                            base_model = model_name
                            num_gpus = cfg.total_gpus
                        num_gpus = _pg_total_gpus(name) or num_gpus
                        if num_gpus is None:
                            continue
                    elif name.startswith("peft_trainer_"):
                        actor_type = ActorType.DENSE
                        num_gpus = 1
                        base_model = ""
                    elif name.startswith("megatron_"):
                        actor_type = ActorType.MEGATRON
                        base_model = ""
                        num_gpus: int | None = None
                        model_part = name[len("megatron_"):]
                        model_name, cfg = _lookup_model_config(model_part)
                        if cfg is not None:
                            base_model = model_name
                            num_gpus = cfg.train_gpus
                        num_gpus = _pg_total_gpus(name) or num_gpus
                        if num_gpus is None:
                            continue
                    elif name.startswith("openpi_shared_runtime_") or name.startswith("openpi_action_runtime_"):
                        actor_type = ActorType.OPENPI
                        num_gpus = 1
                        base_model, session_id, node_id, metadata = _openpi_actor_diag(actor)
                        metadata = {**metadata, "startup_reconcile": "__ray_ready__timeout"}
                    else:
                        continue

                    from tinker_server.backend.model_registry import is_persistent_model

                    model_actor_inventory.register(
                        actor_name=name,
                        actor_type=actor_type,
                        num_gpus=num_gpus,
                        actor_handle=actor,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=base_model,
                        session_id=session_id,
                        node_id=node_id,
                        protected=bool(actor_type != ActorType.OPENPI and base_model and is_persistent_model(base_model)),
                        metadata=metadata,
                    )
                    registered += 1
                    logger.info("Registered busy actor (not ready): %s (%s, %s GPUs)", name, actor_type.value, num_gpus)
                except Exception as reg_err:
                    logger.warning("Failed to register busy actor %s: %s", name, reg_err)
        except ValueError:
            logger.debug("Actor %s not found (name registered but no actor)", name)
            try:
                model_actor_inventory.unregister(name)
            except Exception:
                pass
            try:
                pg_name = f"{name}_pg"
                pg = ray.util.get_placement_group(pg_name)
                ray.util.remove_placement_group(pg)
                logger.warning("Removed orphan placement_group=%s", pg_name)
            except Exception:
                pass

    logger.info("Actor cleanup complete: %s cleaned, %s registered", cleaned, registered)
    return {"cleaned": cleaned, "registered": registered}
