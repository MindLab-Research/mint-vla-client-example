import os


def _maybe_set_vllm_host_ip() -> None:
    # vLLM uses VLLM_HOST_IP for cross-node rendezvous and also for validating
    # that each Ray node reports a unique IP. When unset, vLLM's get_ip()
    # probes the default route, which can be inconsistent across Ray worker
    # processes on the same node.
    #
    # Ray sets RAY_NODE_IP_ADDRESS for each worker process; use it to pin a
    # stable per-node IP without requiring cluster-wide env configuration.
    if os.environ.get("VLLM_HOST_IP"):
        return
    ray_node_ip = os.environ.get("RAY_NODE_IP_ADDRESS")
    if ray_node_ip:
        os.environ["VLLM_HOST_IP"] = ray_node_ip


_maybe_set_vllm_host_ip()

