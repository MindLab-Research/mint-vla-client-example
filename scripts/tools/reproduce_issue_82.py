import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tinker_server.backend.multinode_resources import compute_multinode_engine_resources


def main() -> int:
    worker_gpus = 16
    r = compute_multinode_engine_resources(worker_gpus)

    if r.controller_gpus != 0:
        print(f"FAIL: controller_gpus={r.controller_gpus} (expected 0)", file=sys.stderr)
        return 1
    if r.total_required_gpus != worker_gpus:
        print(
            f"FAIL: total_required_gpus={r.total_required_gpus} (expected {worker_gpus})",
            file=sys.stderr,
        )
        return 1
    if r.controller_bundle_index != worker_gpus:
        print(
            f"FAIL: controller_bundle_index={r.controller_bundle_index} (expected {worker_gpus})",
            file=sys.stderr,
        )
        return 1
    if len(r.pg_bundles) != worker_gpus + 1:
        print(
            f"FAIL: len(pg_bundles)={len(r.pg_bundles)} (expected {worker_gpus + 1})",
            file=sys.stderr,
        )
        return 1
    gpu_bundles = [b for b in r.pg_bundles if b.get("GPU", 0) > 0]
    if len(gpu_bundles) != worker_gpus:
        print(
            f"FAIL: GPU bundles={len(gpu_bundles)} (expected {worker_gpus})",
            file=sys.stderr,
        )
        return 1
    if r.pg_bundles[r.controller_bundle_index].get("GPU", 0) != 0:
        print("FAIL: controller bundle unexpectedly reserves GPU", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

