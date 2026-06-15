from __future__ import annotations

import os
from typing import Any

import ray

from mint_server.runtime_env import env_nonempty


class PlacementGroupMismatchError(RuntimeError):
    def __init__(self, pg: Any, message: str) -> None:
        super().__init__(message)
        self.pg = pg


class PlacementGroupNotFoundError(ValueError):
    pass


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _bundle_dicts(raw: object) -> tuple[dict[str, object], ...]:
    if isinstance(raw, dict):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = ()
    return tuple(bundle for bundle in values if isinstance(bundle, dict))


def _bundle_signature(raw: object) -> tuple[tuple[tuple[str, float], ...], ...]:
    signature: list[tuple[tuple[str, float], ...]] = []
    for bundle in _bundle_dicts(raw):
        gpu = float(bundle.get("GPU", 0) or 0)
        node_affinity = tuple(
            sorted(
                (key, float(value or 0))
                for key, value in bundle.items()
                if isinstance(key, str) and key.startswith("node:") and float(value or 0) > 0
            )
        )
        if gpu <= 0 and not node_affinity:
            continue
        signature.append((("GPU", gpu), *node_affinity))
    return tuple(sorted(signature))


def _namespace_from_pg_info(info: object) -> str | None:
    if not isinstance(info, dict):
        return None
    for key in ("ray_namespace", "namespace", "rayNamespace"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _pinned_ips(raw: object) -> tuple[str, ...]:
    pinned: set[str] = set()
    for bundle in _bundle_dicts(raw):
        for key, value in bundle.items():
            if isinstance(key, str) and key.startswith("node:") and float(value or 0) > 0:
                pinned.add(key.split("node:", 1)[1])
    return tuple(sorted(pinned))


def _lookup_named_pg_info(name: str, namespace: str) -> dict[str, object] | None:
    try:
        table = ray.util.placement_group_table()
    except Exception:
        return None

    exact_matches: list[dict[str, object]] = []
    unknown_ns_matches: list[dict[str, object]] = []
    for info in table.values():
        if not isinstance(info, dict) or info.get("name") != name:
            continue
        if str(info.get("state") or "").upper() == "REMOVED":
            continue
        info_ns = _namespace_from_pg_info(info)
        if info_ns == namespace:
            exact_matches.append(info)
        elif info_ns is None:
            unknown_ns_matches.append(info)

    if exact_matches:
        return exact_matches[0]
    if unknown_ns_matches:
        return unknown_ns_matches[0]
    return None


def _placement_group_from_info(info: dict[str, object]) -> Any | None:
    raw_id = info.get("placement_group_id") or info.get("id")
    if raw_id is None:
        return None
    try:
        from ray._raylet import PlacementGroupID
        from ray.util.placement_group import PlacementGroup

        return PlacementGroup(PlacementGroupID.from_hex(str(raw_id)))
    except Exception:
        return None


def get_named_placement_group(
    name: str,
    *,
    namespace: str | None = None,
    expected_bundles: object | None = None,
) -> Any:
    target_namespace = namespace or _ray_namespace()
    pg = None
    try:
        try:
            pg = ray.util.get_placement_group(name, namespace=target_namespace)
        except TypeError:
            pg = ray.util.get_placement_group(name)
    except Exception:
        info = _lookup_named_pg_info(name, target_namespace)
        if info is None:
            raise PlacementGroupNotFoundError(f"placement group {name!r} not found") from None
        pg = _placement_group_from_info(info)
        if pg is None:
            raise

    info = None
    try:
        info = ray.util.placement_group_table(pg)
    except Exception:
        info = _lookup_named_pg_info(name, target_namespace)

    info_namespace = _namespace_from_pg_info(info)
    if info_namespace is not None and info_namespace != target_namespace:
        raise ValueError(
            f"placement group {name!r} exists in namespace={info_namespace!r}, "
            f"not target_namespace={target_namespace!r}"
        )

    if expected_bundles is not None and info is not None:
        actual_signature = _bundle_signature(info.get("bundles"))
        expected_signature = _bundle_signature(expected_bundles)
        if actual_signature != expected_signature:
            raise PlacementGroupMismatchError(
                pg,
                f"placement group {name!r} has incompatible bundle shape: "
                f"actual_pinned_ips={list(_pinned_ips(info.get('bundles')))!r} "
                f"expected_pinned_ips={list(_pinned_ips(expected_bundles))!r}",
            )

    return pg


def remove_named_placement_group(name: str, *, namespace: str | None = None) -> bool:
    target_namespace = namespace or _ray_namespace()
    try:
        pg = get_named_placement_group(name, namespace=target_namespace)
    except PlacementGroupNotFoundError:
        return False
    except Exception:
        info = _lookup_named_pg_info(name, target_namespace)
        if info is None:
            raise
        pg = _placement_group_from_info(info)
        if pg is None:
            return False
    ray.util.remove_placement_group(pg)
    return True
