from __future__ import annotations

import structlog
import time

import os
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any

from mint_server.runtime_env import env_nonempty
from mint_server.backend.ray_cluster.model_actor_pg_names import actor_placement_group_names

logger = structlog.get_logger(__name__)


def _undesired_gpu_actor_grace_s() -> float:
    """Grace period before the reaper kills an undesired mint GPU actor.

    Configurable via MINT_UNDESIRED_GPU_ACTOR_GRACE_S (seconds, float).
    Default 120 s: long enough to outlast a supervisor control-plane restart
    and FIX-D adoption (which runs before the first reconcile), but short
    enough to eventually reap genuinely-orphaned actors.
    """
    raw = str(os.environ.get("MINT_UNDESIRED_GPU_ACTOR_GRACE_S") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning(
                "[model_actor_placement] invalid MINT_UNDESIRED_GPU_ACTOR_GRACE_S=%r; using default",
                raw,
            )
    return 120.0


ActorExists = Callable[[str, str], bool]
ActorKiller = Callable[[str, str, str], bool]
GpuActorKiller = Callable[[dict[str, Any], str], bool]
ActorLister = Callable[[], Iterable[dict[str, Any]]]
CapacityChecker = Callable[[dict[str, int], str, set[str], str], None]
PlacementGroupRemover = Callable[[str, str], bool]
PlacementGroupLister = Callable[[], Iterable[dict[str, Any]]]


def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _base_model_from_spec(spec: Any) -> str | None:
    base_model = getattr(spec, "base_model", None)
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()
    domain_key = str(getattr(spec, "domain_key", "") or "")
    if domain_key.startswith("vllm:"):
        model = domain_key.removeprefix("vllm:").strip()
        return model or None
    return None


def _owned_actor_names_for_spec(spec: Any) -> set[str]:
    name = str(spec.normalized_actor_name())
    names = {name} if name else set()
    domain_key = str(getattr(spec, "domain_key", "") or "")
    if domain_key.startswith("vllm:"):
        base_model = _base_model_from_spec(spec)
        if base_model:
            model_part = base_model.split("/")[-1] if "/" in base_model else base_model
            legacy_name = f"mint_vllm_{model_part.lower().replace(' ', '_')}".strip()
            if legacy_name != "mint_vllm_":
                names.add(legacy_name)
    return names


def _is_supervisor_wrapper_actor_name(name: str) -> bool:
    return name.startswith("mint_model_runtime_")


_MINT_GPU_ACTOR_PREFIXES = (
    "mint_model_runtime_",
    "mint_vllm_",
    "mint_dense_",
    "mint_megatron_",
    "mint_openpi_shared_",
    "mint_openpi_action_",
    "openpi_shared_runtime_",
)


def _is_mint_gpu_actor_name(name: str) -> bool:
    return name.startswith(_MINT_GPU_ACTOR_PREFIXES)


def _default_actor_exists(actor_name: str, namespace: str) -> bool:
    try:
        import ray

        if not ray.is_initialized():
            return True
        ray.get_actor(actor_name, namespace=namespace)
        return True
    except ValueError:
        return False
    except Exception as e:
        logger.warning(
            "[model_actor_placement] actor lookup failed actor=%s namespace=%s error_type=%s error=%s; assuming present",
            actor_name,
            namespace,
            type(e).__name__,
            e,
        )
        return True


def _default_actor_lister() -> Iterable[dict[str, Any]]:
    try:
        import ray

        if not ray.is_initialized():
            return []
        actors = ray.util.list_named_actors(all_namespaces=True)
        return [dict(actor) for actor in actors if isinstance(actor, dict)]
    except Exception:
        return []


@lru_cache(maxsize=1)
def _kill_actor_by_pid_remote():
    import ray

    @ray.remote(num_cpus=0)
    def _task(node_id: str | None, actor_id: str | None, pid: int, actor_name: str, reason: str) -> bool:
        _ = node_id, actor_id, actor_name, reason
        import os
        import signal

        os.kill(int(pid), signal.SIGTERM)
        return True

    return _task


def _actor_row_asdict(row: Any) -> dict[str, Any]:
    if hasattr(row, "asdict"):
        candidate = row.asdict()
        return candidate if isinstance(candidate, dict) else {}
    return row if isinstance(row, dict) else {}


def _actor_required_resources(actor: dict[str, Any]) -> dict[str, Any]:
    resources = (
        actor.get("required_resources")
        or actor.get("RequiredResources")
        or actor.get("requiredResources")
        or {}
    )
    return resources if isinstance(resources, dict) else {}


def _actor_name(actor: dict[str, Any]) -> str:
    return str(actor.get("name") or actor.get("Name") or actor.get("actor_name") or "").strip()


def _actor_namespace(actor: dict[str, Any]) -> str:
    return str(
        actor.get("ray_namespace")
        or actor.get("namespace")
        or actor.get("rayNamespace")
        or actor.get("RayNamespace")
        or ""
    ).strip()


def _actor_state(actor: dict[str, Any]) -> str:
    return str(actor.get("state") or actor.get("State") or "").strip()


def _actor_node_id(actor: dict[str, Any]) -> str:
    address = actor.get("address") or actor.get("Address") or {}
    if not isinstance(address, dict):
        address = {}
    return str(
        actor.get("node_id")
        or actor.get("nodeId")
        or actor.get("NodeID")
        or address.get("raylet_id")
        or address.get("RayletID")
        or address.get("node_id")
        or ""
    ).strip()


def _actor_node_ip(actor: dict[str, Any], node_id_to_ip: dict[str, str]) -> str:
    address = actor.get("address") or actor.get("Address") or {}
    if not isinstance(address, dict):
        address = {}
    node_id = _actor_node_id(actor)
    return str(
        actor.get("node_ip")
        or actor.get("node_manager_address")
        or actor.get("ipAddress")
        or actor.get("IPAddress")
        or address.get("ip_address")
        or address.get("IPAddress")
        or (node_id_to_ip.get(node_id, "") if node_id else "")
        or ""
    ).strip()


def _actor_pid(actor: dict[str, Any]) -> int | None:
    for key in ("pid", "Pid", "worker_pid", "WorkerPid"):
        value = actor.get(key)
        if value is None:
            continue
        try:
            pid = int(value)
        except Exception:
            continue
        if pid > 0:
            return pid
    return None


def _gpu_actor_records_from_rows(
    rows: Iterable[Any],
    *,
    node_id_to_ip: dict[str, str],
    include_mint_named_without_resource: bool = False,
) -> list[dict[str, Any]]:
    actors: list[dict[str, Any]] = []
    for row in rows:
        actor = _actor_row_asdict(row)
        if not actor:
            continue
        if _actor_state(actor) != "ALIVE":
            continue
        name = _actor_name(actor)
        if not name:
            continue
        resources = _actor_required_resources(actor)
        try:
            gpu = float(resources.get("GPU", 0) or 0)
        except Exception:
            gpu = 0.0
        if gpu <= 0:
            if not include_mint_named_without_resource or not _is_mint_gpu_actor_name(name):
                continue
            gpu = 1.0
        node_id = _actor_node_id(actor)
        node_ip = _actor_node_ip(actor, node_id_to_ip)
        if not node_ip:
            continue
        record = {
            "name": name,
            "namespace": _actor_namespace(actor),
            "node_ip": node_ip,
            "gpu": gpu,
        }
        actor_id = str(actor.get("actor_id") or actor.get("actorId") or actor.get("ActorID") or "").strip()
        if actor_id:
            record["actor_id"] = actor_id
        if node_id:
            record["node_id"] = node_id
        pid = _actor_pid(actor)
        if pid is not None:
            record["pid"] = pid
        actors.append(record)
    return actors


def _default_gpu_actor_lister() -> Iterable[dict[str, Any]]:
    try:
        import ray

        if not ray.is_initialized():
            return []
        from ray.util import state as ray_state

        namespace = _ray_namespace()
        node_id_to_ip = {
            str(node.get("NodeID") or ""): str(node.get("NodeManagerAddress") or "")
            for node in ray.nodes()
            if node.get("NodeID") and node.get("NodeManagerAddress")
        }
        try:
            rows = ray_state.list_actors(
                detail=True,
                limit=10000,
                raise_on_missing_output=False,
                filters=[("ray_namespace", "=", namespace)],
            )
            return _gpu_actor_records_from_rows(rows, node_id_to_ip=node_id_to_ip)
        except Exception as exc:
            logger.warning(
                "[model_actor_placement] GPU actor state listing failed"
                " namespace=%s error_type=%s error=%s; skipping surviving actor adoption",
                namespace,
                type(exc).__name__,
                exc,
            )
    except Exception:
        logger.debug("GPU actor lister unavailable", exc_info=True)
    return []

def _iter_pg_bundle_items(bundles: object) -> list[tuple[str, dict[str, object]]]:
    if isinstance(bundles, dict):
        iterator = bundles.items()
    elif isinstance(bundles, list):
        iterator = ((str(i), bundle) for i, bundle in enumerate(bundles))
    else:
        iterator = ()
    return [
        (str(bundle_key), bundle)
        for bundle_key, bundle in iterator
        if isinstance(bundle, dict)
    ]


def _default_placement_group_lister() -> Iterable[dict[str, Any]]:
    try:
        import ray

        if not ray.is_initialized():
            return []
        table = ray.util.placement_group_table()
        node_id_to_ip = {
            str(node.get("NodeID") or ""): str(node.get("NodeManagerAddress") or "")
            for node in ray.nodes()
            if node.get("NodeID") and node.get("NodeManagerAddress")
        }
    except Exception:
        return []

    if isinstance(table, dict):
        infos = table.values()
    elif isinstance(table, list):
        infos = table
    else:
        infos = ()

    groups: list[dict[str, Any]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        state = str(info.get("state") or "")
        if state == "REMOVED":
            continue
        bundles_to_node_id = info.get("bundles_to_node_id") or {}
        if not isinstance(bundles_to_node_id, dict):
            bundles_to_node_id = {}

        total_gpu = 0.0
        node_ips: set[str] = set()
        pinned_ips: set[str] = set()
        gpu_by_node_ip: dict[str, float] = {}
        for bundle_key, bundle in _iter_pg_bundle_items(info.get("bundles") or {}):
            try:
                bundle_gpu = float(str(bundle.get("GPU", 0) or 0))
            except Exception:
                bundle_gpu = 0.0
            if bundle_gpu <= 0:
                continue
            total_gpu += bundle_gpu

            bundle_idx = int(bundle_key) if bundle_key.isdigit() else None
            node_id = str(
                bundles_to_node_id.get(bundle_key)
                or (bundles_to_node_id.get(bundle_idx) if bundle_idx is not None else "")
                or ""
            )
            node_ip = node_id_to_ip.get(node_id, "") if node_id else ""
            if node_ip:
                node_ips.add(node_ip)
                gpu_by_node_ip[node_ip] = gpu_by_node_ip.get(node_ip, 0.0) + bundle_gpu

            for key, value in bundle.items():
                if not isinstance(key, str) or not key.startswith("node:"):
                    continue
                try:
                    pinned = float(str(value or 0)) > 0
                except Exception:
                    pinned = False
                if not pinned:
                    continue
                pinned_ip = key.split("node:", 1)[1]
                if not pinned_ip:
                    continue
                pinned_ips.add(pinned_ip)
                node_ips.add(pinned_ip)
                gpu_by_node_ip[pinned_ip] = gpu_by_node_ip.get(pinned_ip, 0.0) + bundle_gpu

        if total_gpu <= 0:
            continue
        groups.append(
            {
                "name": str(info.get("name") or "").strip(),
                "namespace": str(
                    info.get("ray_namespace")
                    or info.get("namespace")
                    or info.get("rayNamespace")
                    or ""
                ),
                "state": state or "<unknown>",
                "node_ips": sorted(node_ips),
                "pinned_ips": sorted(pinned_ips),
                "gpu_by_node_ip": dict(sorted(gpu_by_node_ip.items())),
            }
        )
    return groups


def _default_actor_killer(actor_name: str, namespace: str, reason: str) -> bool:
    try:
        import ray

        import mint_server.backend.ray_cluster.ray_kill as ray_kill

        if not ray.is_initialized():
            return False
        actor = ray.get_actor(actor_name, namespace=namespace)
        ray_kill.kill(
            actor,
            reason=reason,
            actor_name=actor_name,
            namespace=namespace,
            no_restart=True,
            verify_absent=True,
        )
        return True
    except ValueError:
        return False
    except Exception as e:
        logger.warning(
            "[model_actor_placement] failed to kill actor=%s namespace=%s reason=%s error_type=%s error=%s",
            actor_name,
            namespace,
            reason,
            type(e).__name__,
            e,
        )
        return False


def _default_gpu_actor_killer(actor_info: dict[str, Any], reason: str) -> bool:
    name = str(actor_info.get("name") or "").strip()
    namespace = str(actor_info.get("namespace") or _ray_namespace()).strip() or _ray_namespace()
    if name and _default_actor_killer(name, namespace, reason):
        return True

    try:
        import ray

        if not ray.is_initialized():
            return False
        pid = actor_info.get("pid")
        if pid is None:
            return False
        from mint_server.backend.ray_cluster.async_ray_control import control_plane_task_runtime_env

        options: dict[str, Any] = {}
        if actor_info.get("node_id"):
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=str(actor_info.get("node_id") or ""),
                soft=False,
            )
        options["runtime_env"] = control_plane_task_runtime_env()
        ref = _kill_actor_by_pid_remote().options(**options).remote(
            str(actor_info.get("node_id") or ""),
            str(actor_info.get("actor_id") or ""),
            int(pid),
            name,
            reason,
        )
        ray.get(ref, timeout=5.0)
        return True
    except Exception as e:
        logger.warning(
            "[model_actor_placement] failed to kill gpu actor=%s namespace=%s reason=%s error_type=%s error=%s",
            name,
            namespace,
            reason,
            type(e).__name__,
            e,
        )
        return False


def _default_pg_remover(pg_name: str, namespace: str) -> bool:
    try:
        from mint_server.backend.ray_cluster.ray_placement_groups import remove_named_placement_group

        return bool(remove_named_placement_group(pg_name, namespace=namespace))
    except Exception as e:
        logger.warning(
            "[model_actor_placement] failed to remove placement_group=%s namespace=%s error_type=%s error=%s",
            pg_name,
            namespace,
            type(e).__name__,
            e,
        )
        return False


def _default_capacity_checker(
    required_gpus_by_node_ip: dict[str, int],
    context: str,
    ignore_placement_group_names: set[str],
    ignore_placement_group_namespace: str,
) -> None:
    from mint_server.backend.actors.node_placement import assert_node_ip_capacity

    assert_node_ip_capacity(
        required_gpus_by_node_ip=required_gpus_by_node_ip,
        context=context,
        ignore_placement_group_names=ignore_placement_group_names,
        ignore_placement_group_namespace=ignore_placement_group_namespace,
    )


def _is_capacity_block_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "pinned node capacity check failed" in msg
        or "insufficient gpu" in msg
        or "insufficient pinned gpu" in msg
    )


class ModelActorPlacementReconciler:
    """In-process placement reconciliation for ModelActorSupervisor.

    This layer deliberately does not create a Ray actor. It gives the supervisor a
    single place to resolve worker pins, remove stale owned resources, and fail
    closed when pinned GPU capacity is blocked by non-owned Ray state.
    """

    def __init__(
        self,
        *,
        namespace: str | None = None,
        actor_exists: ActorExists | None = None,
        actor_killer: ActorKiller | None = None,
        gpu_actor_killer: GpuActorKiller | None = None,
        actor_lister: ActorLister | None = None,
        capacity_checker: CapacityChecker | None = None,
        gpu_actor_lister: ActorLister | None = None,
        placement_group_lister: PlacementGroupLister | None = None,
        placement_group_remover: PlacementGroupRemover | None = None,
    ) -> None:
        self._namespace = namespace or _ray_namespace()
        self._actor_exists = actor_exists or _default_actor_exists
        self._actor_killer = actor_killer or _default_actor_killer
        if gpu_actor_killer is not None:
            self._gpu_actor_killer = gpu_actor_killer
        elif actor_killer is not None:
            self._gpu_actor_killer = lambda actor_info, reason: self._actor_killer(
                str(actor_info.get("name") or ""),
                str(actor_info.get("namespace") or self._namespace),
                reason,
            )
        else:
            self._gpu_actor_killer = _default_gpu_actor_killer
        self._actor_lister = actor_lister or _default_actor_lister
        self._capacity_checker = capacity_checker or _default_capacity_checker
        self._gpu_actor_lister = gpu_actor_lister or _default_gpu_actor_lister
        self._placement_group_lister = placement_group_lister or _default_placement_group_lister
        self._placement_group_remover = placement_group_remover or _default_pg_remover
        # Grace-period tracking for the undesired GPU actor reaper (Fix B / #727).
        # Maps actor name -> first time it was seen as undesired (time.monotonic()).
        self._undesired_first_seen: dict[str, float] = {}

    def _resolved_node_pins(self, spec: Any, *, context: str) -> list[str]:
        _ = context
        pins = [str(pin) for pin in spec.normalized_node_pins() if str(pin).strip()]
        if pins:
            return list(dict.fromkeys(pins))
        return []

    def _required_gpus_by_node_ip(self, spec: Any, node_pins: list[str]) -> dict[str, int]:
        placement_slices = getattr(spec, "placement_slices", ())
        if placement_slices:
            required: dict[str, int] = {}
            for _replica_id, node_ip, gpu_count in placement_slices:
                node = str(node_ip).strip()
                if not node:
                    continue
                required[node] = required.get(node, 0) + int(gpu_count)
            return required
        gpu_count = getattr(spec, "gpu_count", None)
        if gpu_count is None:
            return {}
        gpus = int(gpu_count)
        if gpus <= 0 or not node_pins:
            return {}
        if len(node_pins) == 1:
            return {node_pins[0]: gpus}
        # ModelActorSpec is currently replica-level rather than slice-level. For
        # multi-node pins, fail closed by requiring the declared replica shape on
        # every pinned node until a richer topology object owns per-node slices.
        return {node_ip: gpus for node_ip in node_pins}

    def _cleanup_undesired_wrapper_actors(self, desired_actor_names: set[str]) -> list[str]:
        cleaned: list[str] = []
        for actor_info in self._actor_lister():
            if not isinstance(actor_info, dict):
                continue
            if actor_info.get("namespace") != self._namespace:
                continue
            name = actor_info.get("name")
            if not isinstance(name, str) or not name:
                continue
            if not _is_supervisor_wrapper_actor_name(name) or name in desired_actor_names:
                continue
            if self._actor_killer(name, self._namespace, "model_actor_supervisor_undesired_wrapper"):
                cleaned.append(name)
            for pg_name in actor_placement_group_names(name, self._namespace):
                self._placement_group_remover(pg_name, self._namespace)
        return cleaned

    def _cleanup_undesired_mint_gpu_actors(self, keep_actor_names: set[str]) -> list[str]:
        """Kill mint GPU actors that are not in *keep_actor_names*.

        Grace-period logic (Fix B / #727): an actor is not killed on first
        sight. It must remain absent from *keep_actor_names* for at least
        ``_undesired_gpu_actor_grace_s()`` seconds before the reaper acts.
        This lets a newly-restarted supervisor re-adopt still-alive dense
        workers (Fix D) before the first reconcile fires the reaper.
        """
        cleaned: list[str] = []
        reason = "model_actor_supervisor_undesired_gpu_actor"
        seen: set[tuple[str, str]] = set()
        now = time.monotonic()
        grace = _undesired_gpu_actor_grace_s()
        listed_names: set[str] = set()

        for actor_info in self._gpu_actor_lister():
            if not isinstance(actor_info, dict):
                continue
            name = str(actor_info.get("name") or "").strip()
            if not name or not _is_mint_gpu_actor_name(name):
                continue
            namespace = str(actor_info.get("namespace") or self._namespace).strip() or self._namespace
            if namespace != self._namespace:
                continue
            key = (namespace, name)
            if key in seen:
                continue
            seen.add(key)
            listed_names.add(name)

            if name in keep_actor_names:
                # Actor is desired/protected — clear any pending grace timer.
                self._undesired_first_seen.pop(name, None)
                continue

            # Actor is NOT in keep set — apply grace period before killing.
            first_seen = self._undesired_first_seen.get(name)
            if first_seen is None:
                # First time we notice this actor is undesired: start the clock.
                self._undesired_first_seen[name] = now
                logger.info(
                    "[model_actor_placement] undesired mint GPU actor grace started"
                    " actor=%s namespace=%s grace_s=%.1f",
                    name,
                    namespace,
                    grace,
                )
                continue
            elapsed = now - first_seen
            if elapsed < grace:
                logger.debug(
                    "[model_actor_placement] undesired mint GPU actor still in grace"
                    " actor=%s elapsed_s=%.1f grace_s=%.1f",
                    name,
                    elapsed,
                    grace,
                )
                continue

            # Grace expired — kill the actor.  Logged at WARNING: reaching this
            # point is an anomalous condition, and if the kill keeps failing the
            # line repeats every reconcile tick, which should surface as a
            # warning rather than INFO noise.
            logger.warning(
                "[model_actor_placement] killing undesired mint GPU actor after grace"
                " actor=%s namespace=%s elapsed_s=%.1f grace_s=%.1f",
                name,
                namespace,
                elapsed,
                grace,
            )
            killed = self._gpu_actor_killer(actor_info, reason)
            if killed:
                # Only drop the grace-timer entry on a successful kill.  If the
                # kill fails the entry stays in place so the NEXT reconcile tick
                # finds the clock already expired and retries immediately instead
                # of restarting a fresh grace window.
                self._undesired_first_seen.pop(name, None)
                cleaned.append(name)
            for pg_name in actor_placement_group_names(name, namespace):
                self._placement_group_remover(pg_name, namespace)

        # Purge stale first_seen entries for actors no longer returned by the lister
        # (they may have been killed externally or adopted into keep_actor_names).
        stale = set(self._undesired_first_seen) - listed_names
        for name in stale:
            self._undesired_first_seen.pop(name, None)

        return cleaned

    def _cleanup_orphan_owned_pgs(self, owned_actor_names: set[str]) -> list[str]:
        removed: list[str] = []
        for actor_name in sorted(owned_actor_names):
            if self._actor_exists(actor_name, self._namespace):
                continue
            for pg_name in actor_placement_group_names(actor_name, self._namespace):
                if self._placement_group_remover(pg_name, self._namespace):
                    removed.append(pg_name)
        return removed

    def _target_actor_started(self, owned_actor_names: set[str]) -> bool:
        return any(self._actor_exists(actor_name, self._namespace) for actor_name in owned_actor_names)

    @staticmethod
    def _placement_group_node_ips(pg_info: dict[str, Any]) -> set[str]:
        node_ips: set[str] = set()
        for key in ("node_ips", "pinned_ips"):
            value = pg_info.get(key) or []
            if isinstance(value, (list, tuple, set)):
                node_ips.update(str(item) for item in value if str(item).strip())
        gpu_by_node_ip = pg_info.get("gpu_by_node_ip") or pg_info.get("gpu_by_pinned_ip") or {}
        if isinstance(gpu_by_node_ip, dict):
            for node_ip, gpu in gpu_by_node_ip.items():
                try:
                    has_gpu = float(gpu or 0) > 0
                except Exception:
                    has_gpu = False
                if has_gpu and str(node_ip).strip():
                    node_ips.add(str(node_ip))
        return node_ips

    def _blocking_placement_groups(
        self,
        *,
        required_gpus_by_node_ip: dict[str, int],
        ignore_placement_group_names: set[str],
        ignore_placement_group_namespace: str,
    ) -> list[dict[str, Any]]:
        required_nodes = set(required_gpus_by_node_ip)
        blockers: list[dict[str, Any]] = []
        for pg_info in self._placement_group_lister():
            if not isinstance(pg_info, dict):
                continue
            pg_name = str(pg_info.get("name") or "").strip()
            if not pg_name:
                continue
            pg_namespace = str(pg_info.get("namespace") or "").strip() or self._namespace
            if (
                pg_namespace == ignore_placement_group_namespace
                and pg_name in ignore_placement_group_names
            ):
                continue
            if not (self._placement_group_node_ips(pg_info) & required_nodes):
                continue
            blockers.append({**pg_info, "name": pg_name, "namespace": pg_namespace})
        return blockers

    def _blocking_gpu_actors(
        self,
        *,
        required_gpus_by_node_ip: dict[str, int],
        protected_actor_names: set[str],
    ) -> list[dict[str, Any]]:
        required_nodes = set(required_gpus_by_node_ip)
        blockers: list[dict[str, Any]] = []
        for actor_info in self._gpu_actor_lister():
            if not isinstance(actor_info, dict):
                continue
            name = str(actor_info.get("name") or "").strip()
            if not name:
                continue
            namespace = str(actor_info.get("namespace") or self._namespace)
            if namespace == self._namespace and name in protected_actor_names:
                continue
            node_ip = str(actor_info.get("node_ip") or "").strip()
            if node_ip not in required_nodes:
                continue
            blockers.append({**actor_info, "name": name, "namespace": namespace})
        return blockers

    def _preempt_exclusive_blockers(
        self,
        *,
        required_gpus_by_node_ip: dict[str, int],
        owned_actor_names: set[str],
        protected_actor_names: set[str],
        ignore_placement_group_names: set[str],
        context: str,
    ) -> dict[str, list[str]]:
        reason = "model_actor_supervisor_exclusive_placement_preempt"
        evicted_actor_names: list[str] = []
        evicted_placement_group_names: list[str] = []
        seen_actor_keys: set[tuple[str, str]] = set()

        blocking_gpu_actors = self._blocking_gpu_actors(
            required_gpus_by_node_ip=required_gpus_by_node_ip,
            protected_actor_names=protected_actor_names,
        )
        blocking_actor_keys = {
            (str(actor_info["namespace"]), str(actor_info["name"]))
            for actor_info in blocking_gpu_actors
        }
        for actor_info in blocking_gpu_actors:
            actor_key = (str(actor_info["namespace"]), str(actor_info["name"]))
            if actor_key in seen_actor_keys:
                continue
            seen_actor_keys.add(actor_key)
            if self._gpu_actor_killer(actor_info, reason):
                evicted_actor_names.append(actor_key[1])

        for pg_info in self._blocking_placement_groups(
            required_gpus_by_node_ip=required_gpus_by_node_ip,
            ignore_placement_group_names=ignore_placement_group_names,
            ignore_placement_group_namespace=self._namespace,
        ):
            pg_name = str(pg_info["name"])
            pg_namespace = str(pg_info["namespace"] or self._namespace)
            actor_name = pg_name.removesuffix("_pg") if pg_name.endswith("_pg") else ""
            if actor_name and (pg_namespace, actor_name) in blocking_actor_keys:
                actor_key = (pg_namespace, actor_name)
                if actor_key not in seen_actor_keys and (
                    pg_namespace != self._namespace or actor_name not in owned_actor_names
                ):
                    seen_actor_keys.add(actor_key)
                    if self._actor_killer(actor_name, pg_namespace, reason):
                        evicted_actor_names.append(actor_name)
            if self._placement_group_remover(pg_name, pg_namespace):
                evicted_placement_group_names.append(pg_name)

        if evicted_actor_names or evicted_placement_group_names:
            logger.warning(
                "[model_actor_placement] preempted blockers for exclusive placement context=%s actors=%s placement_groups=%s required_gpus_by_node_ip=%s",
                context,
                sorted(set(evicted_actor_names)),
                sorted(set(evicted_placement_group_names)),
                required_gpus_by_node_ip,
            )
        return {
            "evicted_actor_names": sorted(set(evicted_actor_names)),
            "evicted_placement_group_names": sorted(set(evicted_placement_group_names)),
        }

    def __call__(
        self,
        desired: dict[tuple[str, str], Any],
        protected_actor_names: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        desired_actor_names: set[str] = set()
        for spec in desired.values():
            if bool(getattr(spec, "enabled", True)):
                desired_actor_names.update(_owned_actor_names_for_spec(spec))

        protected_actor_name_set = {
            str(name).strip()
            for name in (protected_actor_names or ())
            if str(name).strip()
        }
        keep_actor_names = set(desired_actor_names) | protected_actor_name_set
        cleaned_actors = self._cleanup_undesired_wrapper_actors(desired_actor_names)
        cleaned_gpu_actors = self._cleanup_undesired_mint_gpu_actors(keep_actor_names)
        removed_pgs: list[str] = []
        evicted_actor_names: list[str] = []
        evicted_pgs: list[str] = []
        blocked: dict[str, str] = {}
        node_pins_by_label: dict[str, list[str]] = {}

        for key, spec in sorted(desired.items()):
            label = _label(key)
            if not bool(getattr(spec, "enabled", True)):
                continue
            context = (
                f"model_actor_supervisor placement domain={getattr(spec, 'domain_key', '')!r} "
                f"replica={getattr(spec, 'replica_id', '')!r}"
            )
            try:
                node_pins = self._resolved_node_pins(spec, context=context)
                if node_pins:
                    node_pins_by_label[label] = node_pins
                owned_actor_names = _owned_actor_names_for_spec(spec)
                removed_pgs.extend(self._cleanup_orphan_owned_pgs(owned_actor_names))
                protected_for_spec = keep_actor_names | owned_actor_names
                required = self._required_gpus_by_node_ip(spec, node_pins)
                if required:
                    if self._target_actor_started(owned_actor_names):
                        continue
                    ignore_pg_names = {
                        pg_name
                        for name in owned_actor_names
                        for pg_name in actor_placement_group_names(name, self._namespace)
                    }
                    try:
                        self._capacity_checker(
                            required,
                            context,
                            ignore_pg_names,
                            self._namespace,
                        )
                    except Exception as capacity_error:
                        if not _is_capacity_block_error(capacity_error):
                            raise
                        preempted = self._preempt_exclusive_blockers(
                            required_gpus_by_node_ip=required,
                            owned_actor_names=owned_actor_names,
                            protected_actor_names=protected_for_spec,
                            ignore_placement_group_names=ignore_pg_names,
                            context=context,
                        )
                        evicted_actor_names.extend(preempted["evicted_actor_names"])
                        evicted_pgs.extend(preempted["evicted_placement_group_names"])
                        self._capacity_checker(
                            required,
                            context,
                            ignore_pg_names,
                            self._namespace,
                        )
            except Exception as e:
                blocked[label] = f"{type(e).__name__}: {e}"

        return {
            "ok": not blocked,
            "namespace": self._namespace,
            "blocked": blocked,
            "node_pins": node_pins_by_label,
            "cleaned_actor_names": cleaned_actors,
            "cleaned_gpu_actor_names": cleaned_gpu_actors,
            "evicted_actor_names": sorted(set(evicted_actor_names)),
            "evicted_placement_group_names": sorted(set(evicted_pgs)),
            "removed_placement_group_names": sorted(set(removed_pgs)),
            "reclaimed_total": int(
                len(cleaned_actors)
                + len(set(cleaned_gpu_actors))
                + len(set(removed_pgs))
                + len(set(evicted_actor_names))
                + len(set(evicted_pgs))
            ),
        }


model_actor_placement_reconciler = ModelActorPlacementReconciler()
