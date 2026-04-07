from __future__ import annotations

from types import SimpleNamespace


def test_issue_439_megatron_pg_recreates_incompatible_named_group(monkeypatch) -> None:
    from tinker_server.backend import megatron_distributed as dist
    from tinker_server.backend.ray_placement_groups import PlacementGroupMismatchError

    old_pg = object()
    new_pg = object()
    removed: list[object] = []

    def _raise_mismatch(pg_name: str, namespace: str, expected_bundles):
        assert pg_name == "megatron_qwen_pg"
        assert namespace == dist.PERSISTENT_NAMESPACE
        assert expected_bundles == [{"GPU": 1, "CPU": 1, "node:192.168.38.175": 0.001}]
        raise PlacementGroupMismatchError(old_pg, "mismatch")

    monkeypatch.setattr(dist, "get_named_placement_group", _raise_mismatch)
    monkeypatch.setattr(dist.ray.util, "remove_placement_group", lambda pg: removed.append(pg), raising=False)
    monkeypatch.setattr(dist.ray.util, "placement_group", lambda *args, **kwargs: new_pg, raising=False)

    out = dist._get_or_create_megatron_placement_group(
        pg_name="megatron_qwen_pg",
        bundles=[{"GPU": 1, "CPU": 1, "node:192.168.38.175": 0.001}],
    )

    assert removed == [old_pg]
    assert out is new_pg


def test_issue_439_node_affinity_resources_follow_bundle_pin() -> None:
    from tinker_server.backend import megatron_distributed as dist

    bundle = {"GPU": 1, "CPU": 1, "node:192.168.38.175": 0.001}
    assert dist._bundle_node_ip(bundle) == "192.168.38.175"
    assert dist._node_affinity_resources(dist._bundle_node_ip(bundle)) == {
        "node:192.168.38.175": 0.001,
    }
    assert dist._node_affinity_resources(None) == {}


def test_issue_439_megatron_pg_name_is_namespace_scoped() -> None:
    from tinker_server.backend import megatron_distributed as dist

    name_a = dist._make_megatron_pg_name(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        namespace="issue432-e2e-1775119249-2523697",
    )
    name_b = dist._make_megatron_pg_name(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        namespace="issue432-e2e-1775118421-2517026",
    )

    assert name_a != name_b
    assert name_a.startswith("megatron_qwen3_30b_a3b_instruct_2507_")
    assert name_a.endswith("_pg")


def test_issue_439_megatron_diagnostics_report_requested_nodes() -> None:
    from tinker_server.backend import megatron_distributed as dist

    cls = dist.MegatronWorkerGroup.__ray_metadata__.modified_class
    group = cls.__new__(cls)
    group.config = SimpleNamespace(world_size=4, tensor_parallel_size=4, pipeline_parallel_size=1, expert_parallel_size=1)
    group.workers = [object(), object(), object(), object()]
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.lora_rank = 8
    group._placement_bundle_node_ips = ["192.168.38.175"] * 4
    group._placement_requested_node_ips = ["192.168.38.175"] * 4

    out = cls.__dict__["get_diagnostics"](group)

    assert out["placement_bundle_node_ips"] == ["192.168.38.175"] * 4
    assert out["placement_requested_node_ips"] == ["192.168.38.175"]
