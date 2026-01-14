#!/usr/bin/env python3
"""Patch system vLLM on specific node to add debug logging."""
import ray
import sys

PATCH_CODE = '''
        # === DEBUG: Log environment at worker start ===
        import os as _os, sys as _sys
        _debug_file = "/vePFS-Mindverse/share/code/vllm_worker_env.txt"
        with open(_debug_file, "a") as _f:
            _f.write(f"\\n=== Worker PID {_os.getpid()} started ===\\n")
            _f.write(f"PYTHONPATH: {_os.environ.get('PYTHONPATH', 'NOT_SET')[:300]}\\n")
            _f.write(f"sys.path[:5]: {_sys.path[:5]}\\n")
            import vllm as _vllm
            _f.write(f"vllm.__file__: {_vllm.__file__}\\n")
        # === END DEBUG ===
'''


def main():
    target_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.32.213"

    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    # Find node ID
    target_node_id = None
    for n in ray.nodes():
        if n.get("NodeManagerAddress") == target_ip and n["Alive"]:
            target_node_id = n["NodeID"]
            break

    if not target_node_id:
        print(f"Node {target_ip} not found")
        return

    print(f"Target node: {target_ip} -> {target_node_id}")

    @ray.remote(
        num_cpus=0.1,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=target_node_id,
            soft=False,
        ),
    )
    def patch_on_target():
        import os

        path = "/usr/local/lib/python3.10/dist-packages/vllm/v1/executor/multiproc_executor.py"
        hostname = os.uname().nodename

        with open(path, "r") as f:
            content = f.read()

        if "vllm_worker_env.txt" in content:
            return {"hostname": hostname, "status": "already_patched"}

        # Find worker_main
        marker = "def worker_main(*args, **kwargs):"
        marker_pos = content.find(marker)
        if marker_pos == -1:
            return {"hostname": hostname, "status": "error", "msg": "marker not found"}

        # Find the docstring end (triple quotes)
        docstring_marker = '"""'
        docstring_start = content.find(docstring_marker, marker_pos)
        docstring_end = content.find(docstring_marker, docstring_start + 3)
        insert_pos = docstring_end + 3

        new_content = content[:insert_pos] + PATCH_CODE + content[insert_pos:]
        with open(path, "w") as f:
            f.write(new_content)

        return {"hostname": hostname, "status": "patched"}

    result = ray.get(patch_on_target.remote())
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
