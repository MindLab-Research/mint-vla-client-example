import os
import sys
from pathlib import Path

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    # This issue is about local invariants (namespace + PYTHONPATH config), not HTTP behavior.
    _ = BASE_URL, API_KEY

    expected_ns = os.environ.get("MINT_RAY_NAMESPACE", "mint")

    try:
        import mint_server.config as cfg
    except Exception as e:
        return _fail(f"import mint_server.config failed: {e}")

    ray_ns = getattr(cfg, "RAY_NAMESPACE", None)
    if ray_ns != expected_ns:
        return _fail(f"mint_server.config.RAY_NAMESPACE={ray_ns!r} expected {expected_ns!r}")

    # Namespace propagation across modules that create/get detached actors.
    required = {
        "mint_server/backend/inference/multi_lora_engine.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "mint_server/backend/training/megatron/megatron_distributed.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "mint_server/backend/inference/multinode_inference.py": "PERSISTENT_NAMESPACE = RAY_NAMESPACE",
        "mint_server/backend/training/verl/verl_training.py": "PERSISTENT_DENSE_NAMESPACE = RAY_NAMESPACE",
    }
    for rel, needle in required.items():
        txt = (_repo_root / rel).read_text(encoding="utf-8")
        if needle not in txt:
            return _fail(f"{rel} missing {needle!r}")

    # Per-run code root support: MINT_CODE_ROOT should flow into PFS_PYTHONPATH.
    mint_code_root = os.environ.get("MINT_CODE_ROOT")
    if mint_code_root:
        pfs_pythonpath = getattr(cfg, "PFS_PYTHONPATH", "")
        if mint_code_root not in pfs_pythonpath.split(":"):
            return _fail(f"MINT_CODE_ROOT not present in PFS_PYTHONPATH: {mint_code_root!r}")

    # Regression guard: no hard-coded shared dev root in worker PYTHONPATH.
    for rel in (
        "mint_server/backend/training/verl/verl_inference.py",
        "mint_server/backend/training/verl/verl_training.py",
    ):
        txt = (_repo_root / rel).read_text(encoding="utf-8")
        if "/vePFS-Mindverse/share/code/mint-server" in txt:
            return _fail(f"hard-coded '/vePFS-Mindverse/share/code/mint-server' found in {rel}")
        if 'namespace="mint"' in txt or "namespace='mint'" in txt:
            return _fail(f"hard-coded Ray namespace 'mint' found in {rel}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
