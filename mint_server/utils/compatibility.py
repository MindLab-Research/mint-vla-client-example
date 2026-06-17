"""External compatibility helpers kept at the API boundary."""

from __future__ import annotations

from typing import Any


def rewrite_legacy_tinker_uri(value: str) -> tuple[str, bool]:
    if value.startswith("tinker://"):
        return "mint://" + value[len("tinker://") :], True
    return value, False


def rewrite_legacy_tinker_uris(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        return rewrite_legacy_tinker_uri(value)
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            rewritten, item_changed = rewrite_legacy_tinker_uris(item)
            changed = changed or item_changed
            out.append(rewritten)
        return out, changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            rewritten, item_changed = rewrite_legacy_tinker_uris(item)
            changed = changed or item_changed
            out[key] = rewritten
        return out, changed
    return value, False
