from mint_server.backend.ray_cluster.multinode_resources import compute_multinode_engine_resources


def test_issue_82_multinode_controller_does_not_reserve_extra_gpu() -> None:
    r = compute_multinode_engine_resources(worker_gpus=16)
    assert r.controller_gpus == 0
    assert r.total_required_gpus == 16
    assert len([b for b in r.pg_bundles if b.get("GPU", 0) > 0]) == 16
    assert r.pg_bundles[r.controller_bundle_index].get("GPU", 0) == 0
