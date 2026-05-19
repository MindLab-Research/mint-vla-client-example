from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module_with_fake_ray(monkeypatch):
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")

    ray_util_module.list_named_actors = lambda *args, **kwargs: [{"name": "mint_vllm_qwen3-235b", "namespace": "other-ns"}]
    ray_util_module.get_placement_group = lambda *args, **kwargs: object()
    ray_util_module.placement_group_table = lambda *_args, **_kwargs: {
        "bundles_to_node_id": {"0": "node-a", "1": "node-b"},
        "bundles": {
            "0": {"GPU": 8},
            "1": {"GPU": 8},
        },
        "state": "CREATED",
    }

    ray_module.util = ray_util_module
    ray_module.get_actor = lambda *args, **kwargs: object()
    ray_module.nodes = lambda: [
        {"NodeID": "node-a", "NodeManagerAddress": "10.0.0.1"},
        {"NodeID": "node-b", "NodeManagerAddress": "10.0.0.2"},
    ]

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "check_node_usage.py"
    spec = importlib.util.spec_from_file_location("check_node_usage", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_actor_placements_counts_multi_node_gpus_per_node(monkeypatch) -> None:
    module = _load_module_with_fake_ray(monkeypatch)

    placements = module.get_actor_placements()

    assert placements["10.0.0.1"] == [
        {
            "actor_name": "mint_vllm_qwen3-235b",
            "namespace": "other-ns",
            "gpus": 8,
            "pg_state": "CREATED",
        }
    ]
    assert placements["10.0.0.2"] == [
        {
            "actor_name": "mint_vllm_qwen3-235b",
            "namespace": "other-ns",
            "gpus": 8,
            "pg_state": "CREATED",
        }
    ]
