#!/usr/bin/env python3
"""Patch system vLLM on GPU nodes to add debug logging."""
import ray

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

@ray.remote(num_gpus=0.01)
def patch_worker_main():
    """Add debug logging at start of worker_main."""
    import os
    path = "/usr/local/lib/python3.10/dist-packages/vllm/v1/executor/multiproc_executor.py"
    hostname = os.uname().nodename

    with open(path, "r") as f:
        content = f.read()

    if "vllm_worker_env.txt" in content:
        return {"hostname": hostname, "status": "already_patched"}

    # Find worker_main and add debug code right after the docstring
    marker = '@staticmethod\n    def worker_main(*args, **kwargs):'

    if marker not in content:
        # Try alternative
        marker = 'def worker_main(*args, **kwargs):'
        if marker not in content:
            return {"hostname": hostname, "status": "error", "msg": "Could not find worker_main"}

    # Find position after the docstring
    marker_pos = content.find(marker)
    docstring_start = content.find('"""', marker_pos)
    docstring_end = content.find('"""', docstring_start + 3)
    insert_pos = docstring_end + 3

    new_content = content[:insert_pos] + PATCH_CODE + content[insert_pos:]

    with open(path, "w") as f:
        f.write(new_content)

    return {"hostname": hostname, "status": "patched"}


@ray.remote(num_gpus=0.01)
def check_patch_status():
    """Check if patch is applied."""
    import os
    path = "/usr/local/lib/python3.10/dist-packages/vllm/v1/executor/multiproc_executor.py"
    hostname = os.uname().nodename

    with open(path, "r") as f:
        content = f.read()

    is_patched = "vllm_worker_env.txt" in content
    return {"hostname": hostname, "is_patched": is_patched}


def main():
    import sys
    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    # Get all GPU nodes
    nodes = [n for n in ray.nodes() if n.get("Resources", {}).get("GPU", 0) > 0 and n["Alive"]]
    print(f"Found {len(nodes)} GPU nodes")

    action = sys.argv[1] if len(sys.argv) > 1 else "--check"

    if action == "--patch":
        # Patch multiple nodes (tasks may land on different nodes)
        futures = [patch_worker_main.remote() for _ in range(min(20, len(nodes) * 2))]
        results = ray.get(futures)

        patched_hosts = {}
        for r in results:
            h = r["hostname"]
            if h not in patched_hosts:
                patched_hosts[h] = r["status"]
                print(f"{h}: {r['status']}")
        print(f"\nPatched {len(patched_hosts)} unique nodes")

    elif action == "--check":
        futures = [check_patch_status.remote() for _ in range(min(20, len(nodes) * 2))]
        results = ray.get(futures)

        checked_hosts = {}
        for r in results:
            h = r["hostname"]
            if h not in checked_hosts:
                checked_hosts[h] = r["is_patched"]
                print(f"{h}: {'patched' if r['is_patched'] else 'not patched'}")
        print(f"\nChecked {len(checked_hosts)} unique nodes")


if __name__ == "__main__":
    main()
