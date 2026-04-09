from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _import_volc_placement(monkeypatch: pytest.MonkeyPatch):
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.nodes = lambda: []  # type: ignore[attr-defined]
    ray.util = SimpleNamespace(placement_group_table=lambda: {})
    monkeypatch.setitem(sys.modules, "ray", ray)

    ray_private = types.ModuleType("ray._private")
    ray_private.state = SimpleNamespace(available_resources_per_node=lambda: {})
    monkeypatch.setitem(sys.modules, "ray._private", ray_private)

    sys.modules.pop("tinker_server.backend.volc_placement", None)
    return importlib.import_module("tinker_server.backend.volc_placement")


def test_parse_model_node_ip_list_rejects_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    vp = _import_volc_placement(monkeypatch)

    with pytest.raises(RuntimeError, match="MINT_MODEL_NODE_IPS_JSON is not valid JSON"):
        vp.parse_model_node_ip_list(
            raw_json="{bad json",
            lookup_keys=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            env_var_name="MINT_MODEL_NODE_IPS_JSON",
            context="test pin parse",
        )


def test_assert_node_ip_capacity_reports_pg_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp,
        "_list_alive_gpu_nodes",
        lambda: [
            vp.VolcGpuNode(
                node_id="node-1",
                node_ip="10.0.0.7",
                hostname="worker-7",
                total_gpus=8,
                available_gpus=0,
                volc_job_id=None,
                volc_resource_queue_id=None,
            )
        ],
    )
    monkeypatch.setattr(
        vp,
        "_gpu_placement_groups",
        lambda: [
            {
                "name": "megatron_qwen_pg",
                "state": "CREATED",
                "pinned_ips": ["10.0.0.7"],
                "node_ids": ["node-1"],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="pinned node capacity check failed"):
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.7": 8},
            context="megatron pin preflight",
        )

    with pytest.raises(RuntimeError) as exc_info:
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.7": 8},
            context="megatron pin preflight",
        )
    msg = str(exc_info.value)

    assert "10.0.0.7" in msg
    assert "available_gpus" in msg
    assert "used_or_reserved_gpus" in msg
    assert "megatron_qwen_pg:CREATED" in msg


def test_parse_model_single_node_ip_rejects_non_string_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    with pytest.raises(RuntimeError, match="must be a non-empty node IP string"):
        vp.parse_model_single_node_ip(
            raw_json='{"Qwen/Qwen3-30B-A3B-Instruct-2507":["10.0.0.8"]}',
            lookup_keys=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            env_var_name="MINT_VLLM_PINNED_NODE_IP_JSON",
            context="single-node vllm pin",
        )


def test_assert_node_ip_capacity_reports_missing_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(vp, "_list_alive_gpu_nodes", lambda: [])
    monkeypatch.setattr(vp, "_gpu_placement_groups", lambda: [])

    with pytest.raises(RuntimeError, match="missing_nodes=\\['10.0.0.9'\\]"):
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.9": 8},
            context="single-node vllm pin",
        )


def test_list_alive_gpu_nodes_falls_back_when_private_state_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.nodes = lambda: [  # type: ignore[attr-defined]
        {
            "Alive": True,
            "NodeID": "node-1",
            "NodeManagerAddress": "10.0.0.7",
            "NodeManagerHostname": "worker-7",
            "Resources": {"GPU": 8.0},
        }
    ]
    ray.util = SimpleNamespace(placement_group_table=lambda: {})
    monkeypatch.setitem(sys.modules, "ray", ray)

    ray_private = types.ModuleType("ray._private")
    ray_private.state = SimpleNamespace(
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("client mode"))
    )
    monkeypatch.setitem(sys.modules, "ray._private", ray_private)

    sys.modules.pop("tinker_server.backend.volc_placement", None)
    vp = importlib.import_module("tinker_server.backend.volc_placement")

    nodes = vp._list_alive_gpu_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_ip == "10.0.0.7"
    assert nodes[0].available_gpus == 8
