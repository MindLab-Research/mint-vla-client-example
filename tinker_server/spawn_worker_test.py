"""Module-level worker function for spawn test."""
import os
import sys

RESULT_FILE = "/vePFS-Mindverse/share/code/spawn_worker_result.txt"

def spawn_worker_check():
    """Worker function that checks vLLM import location."""
    import vllm
    with open(RESULT_FILE, "w") as f:
        f.write(f"vllm.__file__: {vllm.__file__}\n")
        f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')[:200]}\n")
        f.write(f"sys.path[:5]: {sys.path[:5]}\n")
