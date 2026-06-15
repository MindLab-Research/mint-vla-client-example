import os
import sys
import types
import importlib.machinery
from pathlib import Path


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _install_ray_stub():
    if "ray" in sys.modules:
        del sys.modules["ray"]

    calls: list[dict] = []
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def init(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    ray.init = init  # type: ignore[attr-defined]
    sys.modules["ray"] = ray
    return calls


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from mint_server.ray_utils import init_ray  # noqa: E402

    calls = _install_ray_stub()

    os.environ["MINT_RAY_GCS_ADDRESS"] = "192.168.37.185:6379"
    os.environ.pop("MINT_RAY_LOG_TO_DRIVER", None)
    init_ray(namespace="ns", ignore_reinit_error=True)
    if calls[-1].get("log_to_driver") is not False:
        return _fail(f"log_to_driver expected False when disabled: {calls[-1]!r}")

    os.environ["MINT_RAY_LOG_TO_DRIVER"] = "1"
    init_ray(namespace="ns", ignore_reinit_error=True)
    if calls[-1].get("log_to_driver") is not True:
        return _fail(f"log_to_driver not forwarded when enabled: {calls[-1]!r}")

    init_ray(address="127.0.0.1:6379", namespace="ns", ignore_reinit_error=True, log_to_driver=False)
    if calls[-1].get("log_to_driver") is not False:
        return _fail(f"explicit log_to_driver override not respected: {calls[-1]!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
