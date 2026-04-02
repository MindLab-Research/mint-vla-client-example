from __future__ import annotations


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
    monkeypatch.setattr(dist.ray.util, "remove_placement_group", lambda pg: removed.append(pg))
    monkeypatch.setattr(dist.ray.util, "placement_group", lambda *args, **kwargs: new_pg)

    out = dist._get_or_create_megatron_placement_group(
        pg_name="megatron_qwen_pg",
        bundles=[{"GPU": 1, "CPU": 1, "node:192.168.38.175": 0.001}],
    )

    assert removed == [old_pg]
    assert out is new_pg
