#!/usr/bin/env python3
"""Test vLLM version in Ray worker with PFS PYTHONPATH."""

import ray

VLLM_PFS_PATH = "/vePFS-Mindverse/share/code/vllm-0.12.0"

@ray.remote
def check_vllm_version():
    import vllm
    return {
        "version": vllm.__version__,
        "path": vllm.__file__,
    }

def main():
    ray.init(address="auto", ignore_reinit_error=True)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{VLLM_PFS_PATH}:$PYTHONPATH",
        }
    }

    result = ray.get(
        check_vllm_version.options(runtime_env=runtime_env).remote()
    )

    print(f"vLLM version: {result['version']}")
    print(f"vLLM path: {result['path']}")

    expected = "0.12.0"
    if result["version"] == expected:
        print(f"OK: Version matches expected {expected}")
    else:
        print(f"FAIL: Expected {expected}, got {result['version']}")

if __name__ == "__main__":
    main()
