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
    ray.util = SimpleNamespace(
        placement_group_table=lambda: {},
        state=SimpleNamespace(list_actors=lambda **kwargs: []),
    )
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


def test_parse_model_worker_index_list_rejects_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    vp = _import_volc_placement(monkeypatch)

    with pytest.raises(RuntimeError, match="MINT_MODEL_WORKER_INDICES_JSON is not valid JSON"):
        vp.parse_model_worker_index_list(
            raw_json="{bad json",
            lookup_keys=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            env_var_name="MINT_MODEL_WORKER_INDICES_JSON",
            context="test worker pin parse",
        )


def test_parse_model_worker_index_list_rejects_negative_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    with pytest.raises(RuntimeError, match="non-negative worker indexes"):
        vp.parse_model_worker_index_list(
            raw_json='{"Qwen/Qwen3-30B-A3B-Instruct-2507":[-1]}',
            lookup_keys=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            env_var_name="MINT_MODEL_WORKER_INDICES_JSON",
            context="test worker pin parse",
        )


def test_resolve_worker_indices_to_node_ips_uses_volcano_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp,
        "_list_alive_gpu_nodes",
        lambda: [
            vp.VolcGpuNode(
                node_id="node-1",
                node_ip="10.0.0.7",
                hostname="t-abc-worker-1",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            ),
            vp.VolcGpuNode(
                node_id="node-2",
                node_ip="10.0.0.8",
                hostname="t-abc-worker-2",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            ),
        ],
    )

    assert vp.resolve_worker_indices_to_node_ips(worker_indices=[2, 1], context="test") == [
        "10.0.0.8",
        "10.0.0.7",
    ]


def test_resolve_worker_indices_to_node_ips_reports_missing_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp,
        "_list_alive_gpu_nodes",
        lambda: [
            vp.VolcGpuNode(
                node_id="node-1",
                node_ip="10.0.0.7",
                hostname="t-abc-worker-1",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="missing_worker_indices=\\[2\\]"):
        vp.resolve_worker_indices_to_node_ips(worker_indices=[2], context="test")


def test_parse_model_gpu_placement_resolves_worker_gpu_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp,
        "_list_alive_gpu_nodes",
        lambda: [
            vp.VolcGpuNode(
                node_id="node-1",
                node_ip="10.0.0.7",
                hostname="t-abc-worker-1",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            ),
            vp.VolcGpuNode(
                node_id="node-2",
                node_ip="10.0.0.8",
                hostname="t-abc-worker-2",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            ),
        ],
    )

    placement = vp.parse_model_gpu_placement(
        raw_json=(
            '{"Qwen/Qwen3-235B-A22B-Instruct-2507":['
            '{"replica":0,"worker_index":1,"gpu_count":2},'
            '{"replica":0,"worker_index":2,"gpu_count":3}'
            "]}"
        ),
        lookup_keys=["Qwen/Qwen3-235B-A22B-Instruct-2507"],
        env_var_name="MINT_VLLM_MODEL_PLACEMENT_JSON",
        context="test placement",
    )

    assert placement is not None
    assert placement.node_ips == ["10.0.0.7", "10.0.0.8"]
    assert placement.total_gpus == 5
    assert placement.required_gpus_by_node_ip() == {"10.0.0.7": 2, "10.0.0.8": 3}


def test_parse_model_gpu_placement_rejects_out_of_range_gpu_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp,
        "_list_alive_gpu_nodes",
        lambda: [
            vp.VolcGpuNode(
                node_id="node-1",
                node_ip="10.0.0.7",
                hostname="t-abc-worker-1",
                total_gpus=8,
                available_gpus=8,
                volc_job_id=None,
                volc_resource_queue_id=None,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="exceeds node GPU count"):
        vp.parse_model_gpu_placement(
            raw_json='{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"worker_index":1,"gpu_count":9}}',
            lookup_keys=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            env_var_name="MINT_VLLM_MODEL_PLACEMENT_JSON",
            context="test placement",
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
                "namespace": "tinker",
                "state": "CREATED",
                "pinned_ips": ["10.0.0.7"],
                "gpu_by_pinned_ip": {"10.0.0.7": 8.0},
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


def test_assert_node_ip_capacity_ignores_owned_pg_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "namespace": "tinker",
                "state": "CREATED",
                "pinned_ips": ["10.0.0.7"],
                "gpu_by_pinned_ip": {"10.0.0.7": 8.0},
                "node_ids": ["node-1"],
            }
        ],
    )

    vp.assert_node_ip_capacity(
        required_gpus_by_node_ip={"10.0.0.7": 8},
        context="megatron pin preflight",
        ignore_placement_group_names={"megatron_qwen_pg"},
        ignore_placement_group_namespace="tinker",
    )


def test_assert_node_ip_capacity_does_not_ignore_same_name_pg_from_other_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                "namespace": "other",
                "state": "CREATED",
                "pinned_ips": ["10.0.0.7"],
                "gpu_by_pinned_ip": {"10.0.0.7": 8.0},
                "node_ids": ["node-1"],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="megatron_qwen_pg:CREATED"):
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.7": 8},
            context="megatron pin preflight",
            ignore_placement_group_names={"megatron_qwen_pg"},
            ignore_placement_group_namespace="tinker",
        )


def test_assert_node_ip_capacity_ignores_namespace_suffixed_pg_when_ray_table_has_no_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                "name": "megatron_qwen_tinker_pg",
                "state": "CREATED",
                "pinned_ips": ["10.0.0.7"],
                "gpu_by_pinned_ip": {"10.0.0.7": 8.0},
                "node_ids": ["node-1"],
            }
        ],
    )

    vp.assert_node_ip_capacity(
        required_gpus_by_node_ip={"10.0.0.7": 8},
        context="megatron pin preflight",
        ignore_placement_group_names={"megatron_qwen_tinker_pg"},
        ignore_placement_group_namespace="tinker",
    )


def test_assert_node_ip_capacity_does_not_ignore_unsuffixed_pg_when_ray_table_has_no_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                "gpu_by_pinned_ip": {"10.0.0.7": 8.0},
                "node_ids": ["node-1"],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="megatron_qwen_pg:CREATED"):
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.7": 8},
            context="megatron pin preflight",
            ignore_placement_group_names={"megatron_qwen_pg"},
            ignore_placement_group_namespace="tinker",
        )


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


def test_list_alive_gpu_nodes_fails_closed_when_state_and_pg_fallback_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(
        vp.ray.util,
        "placement_group_table",
        lambda: (_ for _ in ()).throw(RuntimeError("pg down")),
    )
    monkeypatch.setattr(
        vp.ray.util.state,
        "list_actors",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("actors down")),
    )
    monkeypatch.setattr(
        vp.ray.util,
        "list_named_actors",
        lambda all_namespaces=True: [{"name": "foreign-actor", "namespace": "other"}],
        raising=False,
    )

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].node_ip == "10.0.0.7"
    assert nodes[0].available_gpus == 0


def test_list_alive_gpu_nodes_ray_client_mode_treats_missing_state_entry_as_schedulable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(vp.ray.util, "placement_group_table", lambda: {})

    client_ray = SimpleNamespace(is_connected=lambda: True)
    monkeypatch.setattr(vp.ray.util, "client", SimpleNamespace(ray=client_ray), raising=False)

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].node_ip == "10.0.0.7"
    assert nodes[0].available_gpus == 8


def test_list_alive_gpu_nodes_accounts_pending_list_bundles_via_pinned_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(
        vp.ray.util,
        "placement_group_table",
        lambda: {
            "pg-1": {
                "state": "PENDING",
                "bundles": [{"GPU": 4, "node:10.0.0.7": 0.001}],
                "bundles_to_node_id": {},
            }
        },
    )

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].node_ip == "10.0.0.7"
    assert nodes[0].available_gpus == 4


def test_select_free_nodes_from_allowed_ips_uses_fail_closed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(vp.ray.util, "placement_group_table", lambda: {})
    monkeypatch.setattr(
        vp.ray.util.state,
        "list_actors",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("actors down")),
    )
    monkeypatch.setattr(
        vp.ray.util,
        "list_named_actors",
        lambda all_namespaces=True: [{"name": "foreign-actor", "namespace": "other"}],
        raising=False,
    )

    with pytest.raises(RuntimeError, match="insufficient free nodes within allowlist"):
        vp.select_free_nodes_from_allowed_ips(allowed_node_ips=["10.0.0.7"], required_gpus=1)


def test_list_alive_gpu_nodes_treats_missing_state_entry_as_zero_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(available_resources_per_node=lambda: {})  # type: ignore[attr-defined]

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].available_gpus == 0


def test_list_alive_gpu_nodes_uses_actor_state_fallback_when_private_state_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.39.87:10001")
    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(vp.ray.util, "placement_group_table", lambda: {})
    monkeypatch.setattr(vp.ray.util.state, "list_actors", lambda **kwargs: [])

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].node_ip == "10.0.0.7"
    assert nodes[0].available_gpus == 8


def test_list_alive_gpu_nodes_actor_state_fallback_accounts_alive_gpu_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.39.87:10001")
    monkeypatch.setattr(
        vp.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-1",
                "NodeManagerAddress": "10.0.0.7",
                "NodeManagerHostname": "worker-7",
                "Resources": {"GPU": 8},
            }
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private",
        types.ModuleType("ray._private"),
    )
    sys.modules["ray._private"].state = SimpleNamespace(  # type: ignore[attr-defined]
        available_resources_per_node=lambda: (_ for _ in ()).throw(RuntimeError("state down"))
    )
    monkeypatch.setattr(vp.ray.util, "placement_group_table", lambda: {})
    monkeypatch.setattr(
        vp.ray.util.state,
        "list_actors",
        lambda **kwargs: [
            {
                "state": "ALIVE",
                "node_id": "node-1",
                "required_resources": {"GPU": 4},
            }
        ],
    )

    nodes = vp._list_alive_gpu_nodes()

    assert len(nodes) == 1
    assert nodes[0].available_gpus == 4


def test_actor_state_fallback_omits_address_when_ray_is_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.39.87:10001")

    seen: dict[str, object] = {}

    def _list_actors(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(vp.ray.util.state, "list_actors", _list_actors)

    used, ok = vp._actor_used_gpus_by_node_from_state_api(context="test")

    assert ok is True
    assert used == {}
    assert "address" not in seen
    assert seen["detail"] is True
    assert seen["limit"] == 10000


def test_actor_state_fallback_uses_subprocess_after_direct_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vp = _import_volc_placement(monkeypatch)

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.39.87:10001")

    calls: list[dict[str, object]] = []

    def _list_actors(**kwargs):
        calls.append(dict(kwargs))
        raise RuntimeError("not initialized in this thread")

    monkeypatch.setattr(vp.ray.util.state, "list_actors", _list_actors)
    subprocess_calls: list[list[str]] = []

    def _check_output(cmd, text, timeout, env):
        subprocess_calls.append(list(cmd))
        assert text is True
        assert timeout == 60
        assert "RAY_ADDRESS" not in env
        assert "MINT_RAY_CLIENT_ADDRESS" not in env
        assert "RAY_CLIENT_ADDRESS" not in env
        return "[]"

    monkeypatch.setattr(vp.subprocess, "check_output", _check_output)

    used, ok = vp._actor_used_gpus_by_node_from_state_api(context="test")

    assert ok is True
    assert used == {}
    assert len(calls) == 1
    assert "address" not in calls[0]
    assert len(subprocess_calls) == 1
    assert subprocess_calls[0][-1] == "192.168.39.87:6379"


def test_assert_node_ip_capacity_handles_list_shaped_pg_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        vp.ray.util,
        "placement_group_table",
        lambda: [
            {
                "name": "megatron_qwen_pg",
                "state": "CREATED",
                "bundles": [{"GPU": 8, "node:10.0.0.7": 0.001}],
                "bundles_to_node_id": {0: "node-1"},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="megatron_qwen_pg:CREATED"):
        vp.assert_node_ip_capacity(
            required_gpus_by_node_ip={"10.0.0.7": 8},
            context="single-node vllm pin",
        )
