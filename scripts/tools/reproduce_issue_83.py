import os
import sys
from pathlib import Path

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    # This issue is about local invariants (namespace + PYTHONPATH config), not HTTP behavior.
    _ = BASE_URL, API_KEY

    expected_ns = os.environ.get("TINKER_RAY_NAMESPACE", "tinker")

    try:
        import tinker_server.config as cfg
    except Exception as e:
        return _fail(f"import tinker_server.config failed: {e}")

    ray_ns = getattr(cfg, "RAY_NAMESPACE", None)
    if ray_ns != expected_ns:
        return _fail(f"tinker_server.config.RAY_NAMESPACE={ray_ns!r} expected {expected_ns!r}")

    # Namespace propagation across modules that create/get detached actors.
    required = {
        "tinker_server/backend/multi_lora_engine.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "tinker_server/backend/megatron_distributed.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "tinker_server/backend/multinode_inference.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "tinker_server/backend/verl_training.py": "PERSISTENT_DENSE_NAMESPACE = RAY_NAMESPACE",
    }
    for rel, needle in required.items():
        txt = (_repo_root / rel).read_text(encoding="utf-8")
        if needle not in txt:
            return _fail(f"{rel} missing {needle!r}")

    # Per-run code root support: PFS_TINKER_PATH should flow into PFS_PYTHONPATH.
    pfs_tinker = os.environ.get("PFS_TINKER_PATH")
    if pfs_tinker:
        pfs_pythonpath = getattr(cfg, "PFS_PYTHONPATH", "")
        if pfs_tinker not in pfs_pythonpath.split(":"):
            return _fail(f"PFS_TINKER_PATH not present in PFS_PYTHONPATH: {pfs_tinker!r}")

    # Regression guard: no hard-coded shared dev root in worker PYTHONPATH.
    for rel in (
        "tinker_server/backend/verl_inference.py",
        "tinker_server/backend/verl_training.py",
    ):
        txt = (_repo_root / rel).read_text(encoding="utf-8")
        if "/vePFS-Mindverse/share/code/tinker-server" in txt:
            return _fail(f"hard-coded '/vePFS-Mindverse/share/code/tinker-server' found in {rel}")
        if 'namespace="tinker"' in txt or "namespace='tinker'" in txt:
            return _fail(f"hard-coded Ray namespace 'tinker' found in {rel}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
