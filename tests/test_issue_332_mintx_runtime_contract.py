from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import torch


def _install_fake_ray(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def remote(**_kwargs):
        class _RemoteWrapper:
            def __call__(self, obj):
                obj.__ray_metadata__ = types.SimpleNamespace(modified_class=obj)
                return obj

        return _RemoteWrapper()

    ray.remote = remote  # type: ignore[attr-defined]
    ray.kill = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ray.get = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ray.get_actor = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.nodes = lambda: []  # type: ignore[attr-defined]
    ray.actor = types.SimpleNamespace(ActorHandle=object)  # type: ignore[attr-defined]

    ray_util = types.ModuleType("ray.util")
    ray_util.__spec__ = importlib.machinery.ModuleSpec("ray.util", loader=None)
    ray_util.placement_group_table = lambda: {}
    ray.util = ray_util  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)

    tensordict = types.ModuleType("tensordict")
    tensordict.TensorDict = dict  # type: ignore[attr-defined]
    tensordict.__spec__ = importlib.machinery.ModuleSpec("tensordict", loader=None)
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)

    tensordict_tensorclass = types.ModuleType("tensordict.tensorclass")
    tensordict_tensorclass.NonTensorData = object  # type: ignore[attr-defined]
    tensordict_tensorclass.__spec__ = importlib.machinery.ModuleSpec("tensordict.tensorclass", loader=None)
    monkeypatch.setitem(sys.modules, "tensordict.tensorclass", tensordict_tensorclass)

    verl = types.ModuleType("verl")
    verl.__spec__ = importlib.machinery.ModuleSpec("verl", loader=None)
    verl_utils = types.ModuleType("verl.utils")
    verl_utils.__spec__ = importlib.machinery.ModuleSpec("verl.utils", loader=None)
    td_utils = types.ModuleType("verl.utils.tensordict_utils")
    td_utils.__spec__ = importlib.machinery.ModuleSpec("verl.utils.tensordict_utils", loader=None)
    td_utils.get_non_tensor_data = lambda data, key, default=None: data.get(key, default)
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.utils", verl_utils)
    monkeypatch.setitem(sys.modules, "verl.utils.tensordict_utils", td_utils)


def _import_megatron_modules(monkeypatch):
    _install_fake_ray(monkeypatch)
    sys.modules.pop("mint_server.backend.megatron_training", None)
    sys.modules.pop("mint_server.backend.megatron_distributed", None)
    training = importlib.import_module("mint_server.backend.megatron_training")
    distributed = importlib.import_module("mint_server.backend.megatron_distributed")
    return training, distributed


def test_issue_332_vocab_parallel_extractor_does_not_require_log_probs(monkeypatch) -> None:
    training, _ = _import_megatron_modules(monkeypatch)
    extractor = training.create_vocab_parallel_logits_extractor_fn()
    local_logits = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float32)
    loss, metrics = extractor(
        {"vocab_parallel_logits": local_logits},
        {"loss_mask": torch.tensor([1.0], dtype=torch.float32)},
    )

    assert torch.equal(loss, local_logits.new_zeros(()))
    assert metrics["num_tokens"] == 1
    assert torch.equal(metrics["vocab_parallel_logits"], local_logits)
    assert "log_probs" not in metrics


def test_issue_332_megatron_group_forward_preserves_top_level_log_probs(monkeypatch) -> None:
    _, distributed = _import_megatron_modules(monkeypatch)
    log_probs = {
        "data": [-1.0, -2.0],
        "shape": [2],
        "dtype": "float32",
    }
    log_probs_tensor = torch.tensor([-1.0, -2.0], dtype=torch.float32)

    class _RemoteForward:
        def remote(self, *args, **kwargs):
            return "fake-future"

    dummy_group = types.SimpleNamespace(
        _bind_traceparent=lambda traceparent: None,
        _resolve_required_session_id=lambda session_id, op: "sess-1",
        _ensure_session_loaded=lambda *args, **kwargs: {},
        _ensure_session_for_request=lambda **_kwargs: ("sess-1", {}),
        _ray_get_group_results=lambda futures, **_kwargs: distributed.ray.get(futures),
        workers=[types.SimpleNamespace(forward=_RemoteForward())],
    )

    monkeypatch.setattr(
        distributed.ray,
        "get",
        lambda futures: [
            {
                "loss_value": 0.5,
                "loss_sum_value": 1.0,
                "num_tokens": 2,
                "valid_count": 1,
                "loss_fn_outputs": [
                    {
                        "loss": {
                            "data": [0.5],
                            "shape": [1],
                            "dtype": "float32",
                        },
                        "logprobs": log_probs,
                    }
                ],
                "log_probs": log_probs_tensor,
            }
        ],
    )

    result = distributed.MegatronWorkerGroup.forward(
        dummy_group,
        data_items=[{"model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]}}],
        session_id="sess-1",
    )

    assert result["log_probs"] == {"data": [-1.0, -2.0], "shape": [2], "dtype": "torch.float32"}
    assert "loss:sum" not in result["metrics"]
    assert result["metrics"]["loss:mean"] == 0.5
