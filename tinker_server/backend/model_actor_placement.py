from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

ActorExists = Callable[[str, str], bool]
ActorKiller = Callable[[str, str, str], bool]
ActorLister = Callable[[], Iterable[dict[str, Any]]]
CapacityChecker = Callable[[dict[str, int], str, set[str], str], None]
PlacementGroupRemover = Callable[[str, str], bool]
PlacementGroupLister = Callable[[], Iterable[dict[str, Any]]]
WorkerResolver = Callable[[list[int], str], list[str]]


def _ray_namespace() -> str:
    v = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


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


def _legacy_vllm_actor_name(base_model: str) -> str:
    try:
        from .multi_lora_engine import _model_to_actor_name

        return _model_to_actor_name(base_model)
    except Exception:
        model_part = base_model.split("/")[-1] if "/" in base_model else base_model
        return f"tinker_vllm_{model_part.lower().replace(' ', '_')}"


def _legacy_multinode_actor_name(base_model: str) -> str:
    model_part = base_model.split("/")[-1] if "/" in base_model else base_model
    return f"multinode_vllm_{model_part.lower()}"


def _owned_actor_names_for_spec(spec: Any) -> set[str]:
    names = {str(spec.normalized_actor_name())}
    base_model = _base_model_from_spec(spec)
    if base_model:
        names.add(_legacy_vllm_actor_name(base_model))
        names.add(_legacy_multinode_actor_name(base_model))
    return {name for name in names if name}


def _is_supervisor_wrapper_actor_name(name: str) -> bool:
    return name.startswith("mint_model_actor_") or name.startswith("mint_model_runtime_")


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


def _default_gpu_actor_lister() -> Iterable[dict[str, Any]]:
    try:
        import ray

        if not ray.is_initialized():
            return []
        from ray.util import state as ray_state

        node_id_to_ip = {
            str(node.get("NodeID") or ""): str(node.get("NodeManagerAddress") or "")
            for node in ray.nodes()
            if node.get("NodeID") and node.get("NodeManagerAddress")
        }
        rows = ray_state.list_actors(detail=True, limit=10000)
    except Exception:
        return []

    actors: list[dict[str, Any]] = []
    for row in rows:
        actor = row.asdict() if hasattr(row, "asdict") else row
        if not isinstance(actor, dict):
            continue
        if str(actor.get("state") or "") != "ALIVE":
            continue
        name = str(actor.get("name") or "").strip()
        if not name:
            continue
        resources = actor.get("required_resources") or {}
        if not isinstance(resources, dict):
            continue
        try:
            gpu = float(resources.get("GPU", 0) or 0)
        except Exception:
            gpu = 0.0
        if gpu <= 0:
            continue
        node_id = str(actor.get("node_id") or "")
        node_ip = str(actor.get("node_ip") or actor.get("node_manager_address") or "")
        if not node_ip and node_id:
            node_ip = node_id_to_ip.get(node_id, "")
        if not node_ip:
            continue
        actors.append(
            {
                "name": name,
                "namespace": str(
                    actor.get("ray_namespace")
                    or actor.get("namespace")
                    or actor.get("rayNamespace")
                    or ""
                ),
                "node_ip": node_ip,
                "gpu": gpu,
            }
        )
    return actors


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
                bundle_gpu = float(bundle.get("GPU", 0) or 0)
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
                    pinned = float(value or 0) > 0
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

        from . import ray_kill

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


def _default_pg_remover(pg_name: str, namespace: str) -> bool:
    try:
        from .ray_placement_groups import remove_named_placement_group

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
    from .volc_placement import assert_node_ip_capacity

    assert_node_ip_capacity(
        required_gpus_by_node_ip=required_gpus_by_node_ip,
        context=context,
        ignore_placement_group_names=ignore_placement_group_names,
        ignore_placement_group_namespace=ignore_placement_group_namespace,
    )


def _default_worker_resolver(worker_indices: list[int], context: str) -> list[str]:
    from .volc_placement import resolve_worker_indices_to_node_ips

    return resolve_worker_indices_to_node_ips(worker_indices=worker_indices, context=context)


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
        actor_lister: ActorLister | None = None,
        capacity_checker: CapacityChecker | None = None,
        gpu_actor_lister: ActorLister | None = None,
        placement_group_lister: PlacementGroupLister | None = None,
        placement_group_remover: PlacementGroupRemover | None = None,
        worker_resolver: WorkerResolver | None = None,
    ) -> None:
        self._namespace = namespace or _ray_namespace()
        self._actor_exists = actor_exists or _default_actor_exists
        self._actor_killer = actor_killer or _default_actor_killer
        self._actor_lister = actor_lister or _default_actor_lister
        self._capacity_checker = capacity_checker or _default_capacity_checker
        self._gpu_actor_lister = gpu_actor_lister or _default_gpu_actor_lister
        self._placement_group_lister = placement_group_lister or _default_placement_group_lister
        self._placement_group_remover = placement_group_remover or _default_pg_remover
        self._worker_resolver = worker_resolver or _default_worker_resolver

    def _resolved_node_pins(self, spec: Any, *, context: str) -> list[str]:
        pins = [str(pin) for pin in spec.normalized_node_pins() if str(pin).strip()]
        if pins:
            return list(dict.fromkeys(pins))
        worker_index = getattr(spec, "worker_index", None)
        if worker_index is None:
            return []
        return list(dict.fromkeys(self._worker_resolver([int(worker_index)], context)))

    def _required_gpus_by_node_ip(self, spec: Any, node_pins: list[str]) -> dict[str, int]:
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
            self._placement_group_remover(f"{name}_pg", self._namespace)
        return cleaned

    def _cleanup_orphan_owned_pgs(self, owned_actor_names: set[str]) -> list[str]:
        removed: list[str] = []
        for actor_name in sorted(owned_actor_names):
            if self._actor_exists(actor_name, self._namespace):
                continue
            pg_name = f"{actor_name}_pg"
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
        owned_actor_names: set[str],
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
            if namespace == self._namespace and name in owned_actor_names:
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
        ignore_placement_group_names: set[str],
        context: str,
    ) -> dict[str, list[str]]:
        reason = "model_actor_supervisor_exclusive_placement_preempt"
        evicted_actor_names: list[str] = []
        evicted_placement_group_names: list[str] = []
        seen_actor_keys: set[tuple[str, str]] = set()

        blocking_gpu_actors = self._blocking_gpu_actors(
            required_gpus_by_node_ip=required_gpus_by_node_ip,
            owned_actor_names=owned_actor_names,
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
            if self._actor_killer(actor_key[1], actor_key[0], reason):
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

    def __call__(self, desired: dict[tuple[str, str], Any]) -> dict[str, Any]:
        desired_actor_names: set[str] = set()
        for spec in desired.values():
            if bool(getattr(spec, "enabled", True)):
                desired_actor_names.add(str(spec.normalized_actor_name()))

        cleaned_actors = self._cleanup_undesired_wrapper_actors(desired_actor_names)
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
                required = self._required_gpus_by_node_ip(spec, node_pins)
                if required:
                    ignore_pg_names = {f"{name}_pg" for name in owned_actor_names}
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
                        if self._target_actor_started(owned_actor_names):
                            raise
                        preempted = self._preempt_exclusive_blockers(
                            required_gpus_by_node_ip=required,
                            owned_actor_names=owned_actor_names,
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
            "evicted_actor_names": sorted(set(evicted_actor_names)),
            "evicted_placement_group_names": sorted(set(evicted_pgs)),
            "removed_placement_group_names": sorted(set(removed_pgs)),
            "reclaimed_total": int(
                len(cleaned_actors)
                + len(set(removed_pgs))
                + len(set(evicted_actor_names))
                + len(set(evicted_pgs))
            ),
        }


model_actor_placement_reconciler = ModelActorPlacementReconciler()
