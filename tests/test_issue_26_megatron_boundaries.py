import importlib
import importlib.machinery
import sys
import types


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


def test_issue_26_megatron_distributed_no_longer_contains_megatron_actor_pool() -> None:
    _ensure_ray_stubbed()
    _ensure_peft_stubbed()

    dist = importlib.import_module("mint_server.backend.training.megatron.megatron_distributed")
    assert not hasattr(dist, "MegatronActorPool")
    assert not hasattr(dist, "MegatronActorEntry")
    assert not hasattr(dist, "get_megatron_actor_pool")
    assert not hasattr(dist, "_megatron_actor_pool")


def test_issue_26_megatron_docstrings_do_not_reference_deprecated_worker_path() -> None:
    _ensure_ray_stubbed()
    _ensure_peft_stubbed()

    dist = importlib.import_module("mint_server.backend.training.megatron.megatron_distributed")
    assert "MegatronTrainingWorker" not in (dist.__doc__ or "")

    vt = importlib.import_module("mint_server.backend.training.verl.verl_training")
    doc = vt.VerlTrainingEngine.create_training_session.__doc__
    assert doc is not None
    assert "MegatronTrainingWorker" not in doc
    assert "MegatronWorkerGroup" in doc
