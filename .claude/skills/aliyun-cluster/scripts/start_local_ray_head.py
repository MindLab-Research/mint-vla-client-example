#!/usr/bin/env python3
from __future__ import annotations

import os
import time

import ray
from ray._private.node import Node
from ray._private.parameter import RayParams


def main() -> None:
    ip = ray.util.get_node_ip_address()
    gcs_address = (os.environ.get("RAY_GCS_ADDRESS") or "").strip()
    start_head = (os.environ.get("RAY_START_HEAD") or "").strip() in ("1", "true", "yes")
    port = int(os.environ.get("RAY_HEAD_PORT", "6379"))
    num_cpus = int(os.environ.get("RAY_HEAD_NUM_CPUS", "4"))
    env_vars: dict[str, str] = {}
    pythonpath = (os.environ.get("PYTHONPATH") or "").strip()
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath

    print("RAY_VERSION", ray.__version__, flush=True)
    print("RAY_NODE_IP", ip, flush=True)
    print("RAY_GCS_ADDRESS", gcs_address, flush=True)
    print("RAY_START_HEAD", int(start_head), flush=True)
    print("RAY_HEAD_PORT", port, flush=True)
    print("RAY_NODE_NUM_CPUS", num_cpus, flush=True)
    print("RAY_ENV_PYTHONPATH_SET", int(bool(pythonpath)), flush=True)

    if start_head:
        ray_params = RayParams(
            num_cpus=num_cpus,
            num_gpus=0,
            include_dashboard=False,
            gcs_server_port=port,
            node_ip_address=ip,
            env_vars=env_vars or None,
        )
        Node(ray_params, head=True, shutdown_at_exit=False, spawn_reaper=False)
    else:
        if not gcs_address:
            raise SystemExit("RAY_GCS_ADDRESS is empty (expected '<head_ip>:6379' or similar)")
        ray_params = RayParams(
            num_cpus=num_cpus,
            num_gpus=0,
            include_dashboard=False,
            gcs_address=gcs_address,
            node_ip_address=ip,
            env_vars=env_vars or None,
        )
        Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
