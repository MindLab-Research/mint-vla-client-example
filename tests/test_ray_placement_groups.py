from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _import_pg_module(monkeypatch: pytest.MonkeyPatch, *, pg_info: dict[str, object]) -> object:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")

    handle = object()

    def _get_placement_group(name: str):
        if name != pg_info["name"]:
            raise ValueError("pg not found")
        return handle

    def _placement_group_table(arg=None):
        if arg is None:
            return {"pg-id": pg_info}
        if arg is handle:
            return pg_info
        raise AssertionError("unexpected placement_group_table arg")

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.remove_placement_group = lambda _pg: None
    ray_util_module.placement_group_table = _placement_group_table
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    sys.modules.pop("tinker_server.backend.ray_placement_groups", None)
    return importlib.import_module("tinker_server.backend.ray_placement_groups")


def test_get_named_placement_group_rejects_wrong_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_pg_module(
        monkeypatch,
        pg_info={
            "name": "megatron_qwen_pg",
            "namespace": "ns-b",
            "bundles": [{"GPU": 1, "node:192.168.38.38": 0.001}],
        },
    )

    with pytest.raises(ValueError, match="target_namespace='ns-a'"):
        module.get_named_placement_group("megatron_qwen_pg", namespace="ns-a")


def test_get_named_placement_group_rejects_incompatible_pinned_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_pg_module(
        monkeypatch,
        pg_info={
            "name": "megatron_qwen_pg",
            "namespace": "ns-a",
            "bundles": [{"GPU": 1, "CPU": 1, "node:192.168.38.38": 0.001}] * 8,
        },
    )

    with pytest.raises(module.PlacementGroupMismatchError, match="192.168.37.240"):
        module.get_named_placement_group(
            "megatron_qwen_pg",
            namespace="ns-a",
            expected_bundles=[{"GPU": 1, "CPU": 1, "node:192.168.37.240": 0.001}] * 8,
        )
