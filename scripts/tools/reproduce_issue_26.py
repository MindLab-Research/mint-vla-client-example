import sys
import importlib.machinery
import types
import importlib
from pathlib import Path


def _install_stub(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


def _ensure_ray_stubbed() -> None:
    try:
        import ray  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def remote(**_kwargs):
        def deco(obj):
            return obj

        return deco

    ray.remote = remote  # type: ignore[attr-defined]
    ray.kill = lambda *_a, **_k: None  # type: ignore[attr-defined]

    ray_util = types.ModuleType("ray.util")
    ray_util.__spec__ = importlib.machinery.ModuleSpec("ray.util", loader=None)
    ray.util = ray_util  # type: ignore[attr-defined]

    _install_stub("ray", ray)
    _install_stub("ray.util", ray_util)


def _ensure_peft_stubbed() -> None:
    try:
        import peft  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    peft = types.ModuleType("peft")
    peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class LoraConfig:  # noqa: D401
        """peft.LoraConfig stub (tests-only)."""

        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    def get_peft_model(model, _config):
        return model

    peft.LoraConfig = LoraConfig  # type: ignore[attr-defined]
    peft.TaskType = TaskType  # type: ignore[attr-defined]
    peft.get_peft_model = get_peft_model  # type: ignore[attr-defined]

    _install_stub("peft", peft)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    _ensure_ray_stubbed()
    _ensure_peft_stubbed()

    dist = importlib.import_module("mint_server.backend.megatron_distributed")
    if hasattr(dist, "MegatronActorPool"):
        print("FAIL: megatron_distributed still exposes MegatronActorPool", file=sys.stderr)
        return 1
    if hasattr(dist, "MegatronActorEntry"):
        print("FAIL: megatron_distributed still exposes MegatronActorEntry", file=sys.stderr)
        return 1
    if hasattr(dist, "get_megatron_actor_pool") or hasattr(dist, "_megatron_actor_pool"):
        print("FAIL: megatron_distributed still exposes MegatronActorPool globals", file=sys.stderr)
        return 1

    module_doc = dist.__doc__ or ""
    if "MegatronTrainingWorker" in module_doc:
        print(
            "FAIL: megatron_distributed module docstring still mentions MegatronTrainingWorker",
            file=sys.stderr,
        )
        return 1

    vt = importlib.import_module("mint_server.backend.verl_training")
    doc = (vt.VerlTrainingEngine.create_training_session.__doc__ or "")
    if "MegatronTrainingWorker" in doc:
        print(
            "FAIL: create_training_session docstring still mentions MegatronTrainingWorker",
            file=sys.stderr,
        )
        return 1
    if "MegatronWorkerGroup" not in doc:
        print(
            "FAIL: create_training_session docstring does not mention MegatronWorkerGroup",
            file=sys.stderr,
        )
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
