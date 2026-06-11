from __future__ import annotations

import importlib
import sys
import types

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
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    return importlib.import_module("mint_server.backend.ray_placement_groups")


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


def test_get_named_placement_group_passes_namespace_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")
    calls: list[tuple[str, str | None]] = []
    handle = object()

    def _get_placement_group(name: str, namespace: str | None = None):
        calls.append((name, namespace))
        return handle

    def _placement_group_table(arg=None):
        assert arg is handle
        return {
            "name": "megatron_qwen_pg",
            "namespace": "ns-a",
            "bundles": [{"GPU": 1, "node:192.168.38.38": 0.001}],
        }

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.remove_placement_group = lambda _pg: None
    ray_util_module.placement_group_table = _placement_group_table
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    module = importlib.import_module("mint_server.backend.ray_placement_groups")

    assert module.get_named_placement_group("megatron_qwen_pg", namespace="ns-a") is handle
    assert calls == [("megatron_qwen_pg", "ns-a")]


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


def test_remove_named_placement_group_falls_back_to_table_id_for_unknown_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")
    placement_group_module = types.ModuleType("ray.util.placement_group")
    raylet_module = types.ModuleType("ray._raylet")
    removed: list[object] = []

    class _FakePlacementGroupID:
        @classmethod
        def from_hex(cls, value: str) -> str:
            return f"pgid:{value}"

    class _FakePlacementGroup:
        def __init__(self, pg_id: object) -> None:
            self.pg_id = pg_id

    def _get_placement_group(_name: str):
        raise ValueError("pg not found in namespace")

    def _placement_group_table(arg=None):
        if arg is not None:
            raise AssertionError("unexpected placement_group_table arg")
        return {
            "pg-id": {
                "name": "megatron_qwen_pg",
                "placement_group_id": "abcd0000",
                "bundles": [{"GPU": 1, "node:192.168.38.38": 0.001}],
            }
        }

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.remove_placement_group = lambda pg: removed.append(pg)
    ray_util_module.placement_group_table = _placement_group_table
    placement_group_module.PlacementGroup = _FakePlacementGroup
    raylet_module.PlacementGroupID = _FakePlacementGroupID
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement_group_module)
    monkeypatch.setitem(sys.modules, "ray._raylet", raylet_module)
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    module = importlib.import_module("mint_server.backend.ray_placement_groups")

    assert module.remove_named_placement_group("megatron_qwen_pg", namespace="ns-a") is True
    assert len(removed) == 1
    assert removed[0].pg_id == "pgid:abcd0000"


def test_remove_named_placement_group_ignores_removed_table_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")
    removed: list[object] = []

    def _get_placement_group(_name: str):
        raise ValueError("pg not found in namespace")

    def _placement_group_table(arg=None):
        if arg is not None:
            raise AssertionError("unexpected placement_group_table arg")
        return {
            "pg-id": {
                "name": "megatron_qwen_pg",
                "placement_group_id": "abcd0000",
                "state": "REMOVED",
                "bundles": [{"GPU": 1, "node:192.168.38.38": 0.001}],
            }
        }

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.remove_placement_group = lambda pg: removed.append(pg)
    ray_util_module.placement_group_table = _placement_group_table
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    module = importlib.import_module("mint_server.backend.ray_placement_groups")

    assert module.remove_named_placement_group("megatron_qwen_pg", namespace="ns-a") is False
    assert removed == []


def test_remove_named_placement_group_handles_old_ray_namespace_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")
    removed: list[object] = []
    calls: list[tuple[object, ...]] = []

    def _get_placement_group(*args, **kwargs):
        calls.append((args, kwargs))
        if "namespace" in kwargs:
            raise TypeError("get_placement_group() got an unexpected keyword argument 'namespace'")
        raise ValueError("pg not found in namespace")

    def _placement_group_table(arg=None):
        if arg is not None:
            raise AssertionError("unexpected placement_group_table arg")
        return {}

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.remove_placement_group = lambda pg: removed.append(pg)
    ray_util_module.placement_group_table = _placement_group_table
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    module = importlib.import_module("mint_server.backend.ray_placement_groups")

    assert module.remove_named_placement_group("missing_pg", namespace="ns-a") is False
    assert calls == [
        (("missing_pg",), {"namespace": "ns-a"}),
        (("missing_pg",), {}),
    ]
    assert removed == []


def test_get_named_placement_group_classifies_missing_old_ray_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")

    def _get_placement_group(*args, **kwargs):
        if "namespace" in kwargs:
            raise TypeError("get_placement_group() got an unexpected keyword argument 'namespace'")
        raise ValueError("Failed to look up placement group with name: missing_pg")

    ray_util_module.get_placement_group = _get_placement_group
    ray_util_module.placement_group_table = lambda: {}
    ray_module.util = ray_util_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    sys.modules.pop("mint_server.backend.ray_placement_groups", None)
    module = importlib.import_module("mint_server.backend.ray_placement_groups")

    with pytest.raises(module.PlacementGroupNotFoundError):
        module.get_named_placement_group("missing_pg", namespace="ns-a")
