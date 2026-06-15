from __future__ import annotations

import hashlib

DEFAULT_ACTOR_PG_NAMESPACE = "mint"


def namespace_pg_suffix(namespace: str | None) -> str:
    raw = str(namespace or "").strip().lower()
    if not raw:
        return "default"
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    if len(sanitized) <= 24:
        return sanitized or "default"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:15]}_{digest}"


def legacy_actor_placement_group_name(actor_name: str) -> str:
    return f"{actor_name}_pg"


def namespace_actor_placement_group_name(actor_name: str, namespace: str | None) -> str:
    return f"{actor_name}_{namespace_pg_suffix(namespace)}_pg"


def actor_placement_group_names(actor_name: str | None, namespace: str | None) -> tuple[str, ...]:
    name = str(actor_name or "").strip()
    if not name:
        return ()
    scoped_namespace = str(namespace or "").strip() or DEFAULT_ACTOR_PG_NAMESPACE
    candidates = [
        legacy_actor_placement_group_name(name),
        namespace_actor_placement_group_name(name, scoped_namespace),
    ]
    return tuple(dict.fromkeys(candidates))
