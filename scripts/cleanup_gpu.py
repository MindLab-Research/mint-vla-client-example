#!/usr/bin/env python3
"""Cleanup stale GPU resources in Ray cluster."""
import ray
from ray.util.placement_group import PlacementGroup

ray.init(address="auto", ignore_reinit_error=True)

# Get placement groups
pgs = ray.util.placement_group_table()
for pg_id, pg_info in pgs.items():
    state = pg_info.get("state")
    if state != "REMOVED":
        bundles = pg_info.get("bundles", {})
        total_gpus = 0
        for bundle_id, bundle in bundles.items():
            total_gpus += bundle.get("GPU", 0)
        print(f"PG {pg_id[:16]}: state={state}, GPUs={total_gpus}")

        if total_gpus > 0:
            try:
                pg = PlacementGroup(ray.PlacementGroupID.from_hex(pg_id))
                ray.util.remove_placement_group(pg)
                print("  Removed!")
            except Exception as e:
                print(f"  Cannot remove: {e}")

# Check after
import time
time.sleep(2)
r = ray.available_resources()
t = ray.cluster_resources()
g = "GPU"
print(f"\nGPUs: {r.get(g, 0):.0f} / {t.get(g, 0):.0f}")
