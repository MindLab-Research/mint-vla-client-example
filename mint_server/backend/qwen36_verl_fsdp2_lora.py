"""Runtime patches for Qwen3.6 dense text veRL FSDP2 LoRA jobs.

This module productizes the Qwen3.6-27B smoke-run shims as a controlled
MinT-owned overlay.  It is intentionally inert until
``install_qwen36_verl_fsdp2_lora_patches()`` is called, normally from
``sitecustomize.py`` under ``MINT_QWEN36_VERL_FSDP2_LORA_PATCHES=1``.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
import importlib
import importlib.abc
import inspect
import os
import pickle
import sys
import textwrap
from types import ModuleType
from typing import Any


QWEN36_VERL_FSDP2_LORA_BACKEND = "verl_fsdp2_lora"
QWEN36_MODEL_ID = "Qwen/Qwen3.6-27B"
QWEN36_PATCH_ENV = "MINT_QWEN36_VERL_FSDP2_LORA_PATCHES"
QWEN36_TEXT_ONLY_SKIP_DUMMY_VISUAL_ENV = "MINT_QWEN36_TEXT_ONLY_SKIP_DUMMY_VISUAL"
QWEN36_FORCE_PLACEMENT_NODE_IP_ENV = "MINT_QWEN36_FORCE_PLACEMENT_NODE_IP"

_PATCH_MARKER = "_mint_qwen36_verl_fsdp2_lora_patch"
_INSTALLED_HOOKS: set[str] = set()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_qwen36_model(model: str | None) -> bool:
    return "qwen3.6-27b" in str(model or "").lower()


def qwen36_model_path_override(model: str | None) -> str | None:
    if not is_qwen36_model(model):
        return None
    return os.environ.get("MINT_QWEN36_MODEL_PATH") or os.environ.get("QWEN36_MODEL_PATH")


def _target_node_ip() -> str | None:
    return (
        os.environ.get(QWEN36_FORCE_PLACEMENT_NODE_IP_ENV)
        or os.environ.get("QWEN36_FORCE_PLACEMENT_NODE_IP")
        # Kept for compatibility with the smoke scripts that used the Qwen3.5
        # name while validating Qwen3.6 through the qwen3_5 HF architecture.
        or os.environ.get("QWEN35_FORCE_PLACEMENT_NODE_IP")
        or os.environ.get("APPROVED_NODE_IP")
    )


def _target_node_resources() -> dict[str, float]:
    target_ip = _target_node_ip()
    if not target_ip:
        return {}
    return {f"node:{target_ip}": 1e-4}


def _target_alive_node_id(ray_module: Any, *, purpose: str) -> str:
    target_ip = _target_node_ip()
    target_node_ids = [
        node["NodeID"]
        for node in ray_module.nodes()
        if node.get("Alive")
        and node.get("NodeManagerAddress") == target_ip
        and node.get("Resources", {}).get("CPU", 0) > 0
    ]
    if not target_node_ids:
        raise RuntimeError(f"No alive Ray CPU node found for forced {purpose} node {target_ip}")
    return str(target_node_ids[0])


def _mark(obj: Any, suffix: str = "") -> None:
    try:
        setattr(obj, f"{_PATCH_MARKER}{suffix}", True)
    except Exception:
        pass


def _is_marked(obj: Any, suffix: str = "") -> bool:
    return bool(getattr(obj, f"{_PATCH_MARKER}{suffix}", False))


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader, patch: Callable[[ModuleType], None]):
        self._wrapped_loader = wrapped_loader
        self._patch = patch

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        create = getattr(self._wrapped_loader, "create_module", None)
        if callable(create):
            return create(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped_loader.exec_module(module)  # type: ignore[attr-defined]
        self._patch(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch: Callable[[ModuleType], None]):
        self.target = target
        self.patch = patch

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if fullname != self.target:
            return None

        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, self.patch)
            return spec
        return None


def _install_import_patch(target: str, patch: Callable[[ModuleType], None]) -> None:
    module = sys.modules.get(target)
    if isinstance(module, ModuleType):
        patch(module)
        return

    key = f"{target}:{id(patch)}"
    if key in _INSTALLED_HOOKS:
        return
    if not any(isinstance(finder, _PatchFinder) and finder.target == target for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchFinder(target, patch))
    _INSTALLED_HOOKS.add(key)


def _patch_transformers_auto_model_vision2seq_alias() -> None:
    try:
        import transformers
    except Exception:
        return

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return

    image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if image_text_cls is None:
        return

    transformers.AutoModelForVision2Seq = image_text_cls
    print(
        "MinT Qwen3.6 veRL FSDP2 LoRA patch: aliased "
        "transformers.AutoModelForVision2Seq to AutoModelForImageTextToText",
        file=sys.stderr,
        flush=True,
    )


def _patch_accelerate_init_on_device() -> None:
    try:
        import torch
        import torch.nn as nn
        import accelerate
        import accelerate.big_modeling as bm
    except Exception:
        return

    original = getattr(bm, "init_on_device", None)
    if original is None or _is_marked(original, "_accelerate_init_on_device"):
        return

    try:
        if 'kwargs.pop("_is_hf_initialized"' in inspect.getsource(original):
            return
    except Exception:
        pass

    @contextmanager
    def init_on_device(device: torch.device, include_buffers: bool | None = None):
        if include_buffers is None:
            include_buffers = bm.parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

        if include_buffers:
            with device:
                yield
            return

        old_register_parameter = nn.Module.register_parameter

        def register_empty_parameter(module, name, param):  # type: ignore[no-untyped-def]
            old_register_parameter(module, name, param)
            if param is not None:
                param_cls = type(module._parameters[name])
                kwargs = module._parameters[name].__dict__
                kwargs["requires_grad"] = param.requires_grad
                _is_hf_initialized = kwargs.pop("_is_hf_initialized", None)
                module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
                if _is_hf_initialized is not None:
                    module._parameters[name]._is_hf_initialized = _is_hf_initialized

        try:
            nn.Module.register_parameter = register_empty_parameter
            yield
        finally:
            nn.Module.register_parameter = old_register_parameter

    _mark(init_on_device, "_accelerate_init_on_device")
    bm.init_on_device = init_on_device
    accelerate.init_on_device = init_on_device


def _patch_verl_attention_utils_module(attention_utils: ModuleType) -> None:
    if _is_marked(attention_utils, "_attention_utils"):
        return

    try:
        import torch
        import torch.nn.functional as F
        from einops import rearrange as einops_rearrange
    except Exception:
        return

    def index_first_axis(input_tensor, indices):  # type: ignore[no-untyped-def]
        indices = indices.to(device=input_tensor.device, dtype=torch.long)
        return torch.index_select(input_tensor, 0, indices)

    def pad_input(hidden_states, indices, batch, seqlen):  # type: ignore[no-untyped-def]
        indices = indices.to(device=hidden_states.device, dtype=torch.long)
        output_shape = (batch * seqlen, *hidden_states.shape[1:])
        output = hidden_states.new_zeros(output_shape)
        output.index_copy_(0, indices, hidden_states)
        return output.reshape(batch, seqlen, *hidden_states.shape[1:])

    def unpad_input(hidden_states, attention_mask, unused_mask=None):  # type: ignore[no-untyped-def]
        if hidden_states.ndim < 2:
            raise ValueError(f"hidden_states must have batch and sequence dimensions, got {hidden_states.shape}")
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask must match hidden_states batch/sequence dimensions: "
                f"hidden_states={tuple(hidden_states.shape)}, attention_mask={tuple(attention_mask.shape)}"
            )

        valid_mask = attention_mask.bool()
        seqused = valid_mask.sum(dim=-1, dtype=torch.int32)
        if unused_mask is not None:
            valid_mask = valid_mask | unused_mask.bool()

        seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten().to(torch.long)
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
        flat_hidden_states = hidden_states.reshape(-1, *hidden_states.shape[2:])
        return index_first_axis(flat_hidden_states, indices), indices, cu_seqlens, max_seqlen_in_batch, seqused

    def rearrange(*args, **kwargs):  # type: ignore[no-untyped-def]
        return einops_rearrange(*args, **kwargs)

    def _get_attention_functions():
        return index_first_axis, pad_input, rearrange, unpad_input

    for obj in (index_first_axis, pad_input, unpad_input, rearrange):
        _mark(obj, "_attention_utils")

    attention_utils.index_first_axis = index_first_axis
    attention_utils.pad_input = pad_input
    attention_utils.unpad_input = unpad_input
    attention_utils.rearrange = rearrange
    attention_utils._get_attention_functions = _get_attention_functions
    attention_utils._index_first_axis = index_first_axis
    attention_utils._pad_input = pad_input
    attention_utils._rearrange = rearrange
    attention_utils._unpad_input = unpad_input
    _mark(attention_utils, "_attention_utils")


def _patch_torch_fsdp2_cpu_offload_validation_module(fsdp_param_group: ModuleType) -> None:
    try:
        import torch
    except Exception:
        return

    CPUOffloadPolicy = getattr(fsdp_param_group, "CPUOffloadPolicy", None)
    cls = getattr(fsdp_param_group, "FSDPParamGroup", None)
    original = getattr(cls, "_validate_cpu_offload_params", None) if cls is not None else None
    if original is None or _is_marked(original, "_fsdp2_cpu_offload_validation"):
        return

    def _is_cpu_offload_policy(policy: Any) -> bool:
        if policy is None:
            return False
        if CPUOffloadPolicy is not None and isinstance(policy, CPUOffloadPolicy):
            return True
        return type(policy).__name__ == "CPUOffloadPolicy"

    def _tensor_storage_device(tensor: Any) -> Any:
        local = getattr(tensor, "_local_tensor", None)
        if local is not None:
            return getattr(local, "device", None)
        try:
            untyped_storage = tensor.untyped_storage()
            return untyped_storage.device
        except Exception:
            return getattr(tensor, "device", None)

    def _move_param_data_to_cpu(param: Any) -> bool:
        moved = False
        sharded = getattr(param, "_local_tensor", None)
        if sharded is not None and getattr(sharded, "device", None) is not None and sharded.device.type != "cpu":
            sharded.data = sharded.data.to("cpu", non_blocking=True)
            moved = True
        elif getattr(param, "device", None) is not None and param.device.type != "cpu":
            param.data = param.data.to("cpu", non_blocking=True)
            moved = True

        fsdp_param = getattr(param, "_fsdp_param", None)
        sharded_param = getattr(fsdp_param, "sharded_param", None)
        local = getattr(sharded_param, "_local_tensor", None)
        if local is not None and getattr(local, "device", None) is not None and local.device.type != "cpu":
            local.data = local.data.to("cpu", non_blocking=True)
            moved = True
        return moved

    def _validate_cpu_offload_params(self):  # type: ignore[no-untyped-def]
        policy = getattr(self, "cpu_offload_policy", None)
        if _is_cpu_offload_policy(policy):
            moved = 0
            non_cpu = 0
            for fsdp_param in getattr(self, "fsdp_params", []):
                for param_name in ("sharded_param", "to_shard_param"):
                    param = getattr(fsdp_param, param_name, None)
                    if param is None:
                        continue
                    if _move_param_data_to_cpu(param):
                        moved += 1
                    device = _tensor_storage_device(param)
                    if device is not None and device.type != "cpu":
                        non_cpu += 1

            if moved:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            if non_cpu:
                raise AssertionError(
                    "FSDP2 CPU-offload validation still found non-CPU parameter storage "
                    f"after repair: count={non_cpu}"
                )
        return original(self)

    _mark(_validate_cpu_offload_params, "_fsdp2_cpu_offload_validation")
    _validate_cpu_offload_params._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls._validate_cpu_offload_params = _validate_cpu_offload_params


def _patch_verl_ray_resource_pool_module(ray_base: ModuleType) -> None:
    cls = getattr(ray_base, "RayResourcePool", None)
    original = getattr(cls, "get_placement_groups", None) if cls is not None else None
    if original is None or _is_marked(original, "_verl_resource_pool"):
        return

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):  # type: ignore[no-untyped-def]
        target_ip = _target_node_ip()
        if not target_ip:
            return original(self, strategy=strategy, name=name, device_name=device_name)

        from ray.util.placement_group import placement_group

        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = f"{name}_" if name else ""
        lifetime = "detached" if self.use_ray_remote else None
        pgs = [
            placement_group(
                bundles=[
                    {
                        device_name: self.max_colocate_count,
                        f"node:{target_ip}": 1e-4,
                    }
                ]
                * self.process_on_nodes[i],
                strategy=strategy,
                name=pg_name_prefix + str(i),
                lifetime=lifetime,
            )
            for i in range(len(self._store))
        ]
        ray_base.ready_with_progress_bar(pgs)
        self.pgs = ray_base.sort_placement_group_by_node_ip(pgs)
        print(
            "MinT Qwen3.6 veRL FSDP2 LoRA patch: forced veRL placement groups "
            f"onto node {target_ip}",
            file=sys.stderr,
            flush=True,
        )
        return self.pgs

    _mark(get_placement_groups, "_verl_resource_pool")
    get_placement_groups._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls.get_placement_groups = get_placement_groups


def _patch_verl_fsdp2_lora_backup_module(fsdp_utils: ModuleType) -> None:
    if (
        _is_marked(getattr(fsdp_utils, "backup_base_model_weights", None), "_fsdp2_lora")
        and _is_marked(getattr(fsdp_utils, "_merge_or_unmerge_lora_", None), "_fsdp2_lora")
        and _is_marked(getattr(fsdp_utils, "layered_summon_lora_params", None), "_fsdp2_lora")
    ):
        return

    try:
        import torch
    except Exception:
        return

    def _is_dtensor_like(tensor: Any) -> bool:
        return hasattr(tensor, "to_local") and hasattr(tensor, "_spec")

    def _local_tensor(tensor: Any) -> Any:
        local = getattr(tensor, "_local_tensor", None)
        if local is not None:
            return local
        return tensor.to_local()

    def _clone_to_cpu(param: Any) -> tuple[str, Any]:
        if _is_dtensor_like(param):
            local = _local_tensor(param)
            return "dtensor_local", local.detach().clone().cpu()
        tensor = param.detach() if hasattr(param, "detach") else param
        return "tensor", tensor.clone().cpu()

    def _copy_from_cpu(param: Any, saved: tuple[str, Any]) -> None:
        kind, cpu_tensor = saved
        if kind == "dtensor_local" and _is_dtensor_like(param):
            local = _local_tensor(param)
            local.copy_(cpu_tensor.to(device=local.device, dtype=local.dtype))
            return

        device = getattr(param, "device", None)
        dtype = getattr(param, "dtype", None)
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        param.copy_(cpu_tensor.to(**kwargs))

    def _lora_state_key(name: str, adapter_name: str = "default") -> str | None:
        if "lora_" not in name:
            return None
        if f".{adapter_name}." not in name and not name.endswith(f".{adapter_name}"):
            return None
        if name.endswith(f".{adapter_name}"):
            return name[: -len(f".{adapter_name}")]
        key, _, suffix = name.rpartition(".")
        key = key.removesuffix(f".{adapter_name}")
        return f"{key}.{suffix}"

    def _to_lora_cpu_tensor(param: Any) -> Any:
        if _is_dtensor_like(param):
            return param.full_tensor().detach().cpu()
        tensor = param.detach() if hasattr(param, "detach") else param
        return tensor.cpu()

    def _patch_layered_summon_lora_params() -> None:
        from collections import OrderedDict

        def _prefix_submodules(module: Any, prefix: str):
            for name, submodule in module.named_modules():
                if name.startswith(prefix) and "." not in name[len(prefix) :]:
                    yield name, submodule

        def layered_summon_lora_params(fsdp_module) -> OrderedDict:  # type: ignore[type-arg]
            lora_params = OrderedDict()
            prefix_list = [
                "_fsdp_wrapped_module.base_model.model.",
                "_fsdp_wrapped_module.base_model.model.model.",
                "_fsdp_wrapped_module.base_model.model.model.layers.",
                "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
                "base_model.model.",
                "base_model.model.model.",
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            ]

            for prefix in prefix_list:
                for name, submodule in _prefix_submodules(fsdp_module, prefix):
                    output_prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
                    if name.endswith(".model") or name.endswith(".layers"):
                        continue
                    if fsdp_utils.fsdp_version(submodule) <= 0:
                        continue

                    with fsdp_utils.FSDP.summon_full_params(submodule, writeback=False):
                        for param_name, param in submodule.named_parameters():
                            lora_key = _lora_state_key(param_name)
                            if lora_key is None:
                                continue
                            lora_params[f"{output_prefix}.{lora_key}"] = _to_lora_cpu_tensor(param)
                        submodule._is_root = False
                    fsdp_utils.get_torch_device().empty_cache()

            print(
                "MinT Qwen3.6 veRL FSDP2 LoRA patch: collected LoRA params "
                f"without submodule.state_dict() (count={len(lora_params)})",
                file=sys.stderr,
                flush=True,
            )
            return lora_params

        _mark(layered_summon_lora_params, "_fsdp2_lora")
        fsdp_utils.layered_summon_lora_params = layered_summon_lora_params

    def _inplace_tensor(tensor: Any) -> Any:
        if _is_dtensor_like(tensor):
            return _local_tensor(tensor)
        return tensor

    def _dtensor_mesh(tensor: Any) -> Any:
        mesh = getattr(tensor, "device_mesh", None)
        if mesh is not None:
            return mesh
        spec = getattr(tensor, "_spec", None)
        return getattr(spec, "mesh", None)

    def _dtensor_placements(tensor: Any) -> tuple[Any, ...]:
        placements = getattr(tensor, "placements", None)
        if placements is not None:
            return tuple(placements)
        spec = getattr(tensor, "_spec", None)
        return tuple(getattr(spec, "placements", ()) or ())

    def _mesh_size(mesh: Any, mesh_dim: int) -> int:
        try:
            return int(mesh.size(mesh_dim=mesh_dim))
        except TypeError:
            return int(mesh.size(mesh_dim))

    def _mesh_coordinate(mesh: Any) -> tuple[int, ...] | None:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return None
        if isinstance(coordinate, (list, tuple)):
            return tuple(int(v) for v in coordinate)
        return (int(coordinate),)

    def _local_shard_size_and_offset(placement: Any, size_on_dim: int, num_chunks: int, rank: int) -> tuple[int, int]:
        size_and_offset = getattr(placement, "_local_shard_size_and_offset", None)
        if size_and_offset is not None:
            return size_and_offset(size_on_dim, num_chunks, rank)

        size_on_dim_fn = getattr(placement, "_local_shard_size_on_dim", None)
        if size_on_dim_fn is not None:
            return size_on_dim_fn(size_on_dim, num_chunks, rank, return_offset=True)

        if size_on_dim % num_chunks == 0:
            local_size = size_on_dim // num_chunks
            return local_size, local_size * rank

        chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
        offset = chunk_size * rank
        if size_on_dim < offset:
            return 0, size_on_dim
        return min(size_on_dim, offset + chunk_size) - offset, offset

    def _slice_delta_like_dtensor_local_shard(delta: Any, target_dtensor: Any, target_local: Any) -> Any:
        if tuple(delta.shape) == tuple(target_local.shape):
            return delta

        mesh = _dtensor_mesh(target_dtensor)
        placements = _dtensor_placements(target_dtensor)
        coordinate = _mesh_coordinate(mesh) if mesh is not None else None
        if mesh is None or coordinate is None:
            raise RuntimeError(
                "Cannot align LoRA delta with DTensor local shard because the target DTensor "
                f"has no usable mesh coordinate: delta={tuple(delta.shape)}, "
                f"target_local={tuple(target_local.shape)}"
            )

        sliced = delta
        target_shape = tuple(target_dtensor.shape)
        for mesh_dim, placement in enumerate(placements):
            shard_dim = getattr(placement, "dim", None)
            if shard_dim is None:
                continue
            if shard_dim < 0:
                shard_dim += len(target_shape)
            num_chunks = _mesh_size(mesh, mesh_dim)
            rank = coordinate[mesh_dim]
            local_size, offset = _local_shard_size_and_offset(placement, target_shape[shard_dim], num_chunks, rank)
            sliced = sliced.narrow(shard_dim, offset, local_size)

        if tuple(sliced.shape) != tuple(target_local.shape):
            raise RuntimeError(
                "Cannot align LoRA delta with DTensor local shard: "
                f"delta={tuple(delta.shape)}, sliced={tuple(sliced.shape)}, "
                f"target_global={target_shape}, target_local={tuple(target_local.shape)}, "
                f"placements={placements}, coordinate={coordinate}"
            )
        return sliced

    def _add_to_tensor_(tensor: Any, delta: Any, *, alpha: int = 1) -> None:
        target = _inplace_tensor(tensor)
        delta = _inplace_tensor(delta)
        if _is_dtensor_like(tensor):
            delta = _slice_delta_like_dtensor_local_shard(delta, tensor, target)
        target.add_(delta.to(device=target.device, dtype=target.dtype), alpha=alpha)

    def _merge_lora_layer_(layer: Any, check_adapters_to_merge: Callable[..., Any]) -> None:
        adapter_names = check_adapters_to_merge(layer, None)
        if not adapter_names:
            return

        lora_a = getattr(layer, "lora_A", {})
        lora_variants = getattr(layer, "lora_variant", {})
        lora_bias = getattr(layer, "lora_bias", {})
        for active_adapter in adapter_names:
            if active_adapter not in lora_a:
                continue
            if active_adapter in lora_variants:
                raise RuntimeError("MinT Qwen3.6 DTensor LoRA merge shim only supports vanilla LoRA adapters")

            base_layer = layer.get_base_layer()
            _add_to_tensor_(base_layer.weight, layer.get_delta_weight(active_adapter))

            if lora_bias.get(active_adapter, False):
                bias = getattr(base_layer, "bias", None)
                if bias is None:
                    raise RuntimeError("Impossible to merge LoRA with lora_bias=True because the base layer has no bias")
                _add_to_tensor_(bias, layer.lora_B[active_adapter].bias * layer.scaling[active_adapter])

            layer.merged_adapters.append(active_adapter)

    def _unmerge_lora_layer_(layer: Any) -> None:
        if not getattr(layer, "merged", False):
            return

        lora_a = getattr(layer, "lora_A", {})
        lora_variants = getattr(layer, "lora_variant", {})
        lora_bias = getattr(layer, "lora_bias", {})
        while layer.merged_adapters:
            active_adapter = layer.merged_adapters.pop()
            if active_adapter not in lora_a:
                continue
            if active_adapter in lora_variants:
                raise RuntimeError("MinT Qwen3.6 DTensor LoRA unmerge shim only supports vanilla LoRA adapters")

            base_layer = layer.get_base_layer()
            _add_to_tensor_(base_layer.weight, layer.get_delta_weight(active_adapter), alpha=-1)

            if lora_bias.get(active_adapter, False):
                bias = getattr(base_layer, "bias", None)
                if bias is not None:
                    _add_to_tensor_(bias, layer.lora_B[active_adapter].bias * layer.scaling[active_adapter], alpha=-1)

    def _merge_or_unmerge_lora_(module: Any, merge: bool) -> None:
        from peft.tuners.lora import LoraLayer
        from peft.tuners.tuners_utils import check_adapters_to_merge

        with torch.no_grad():
            for layer in module.modules():
                if not isinstance(layer, LoraLayer):
                    continue
                is_merged = getattr(layer, "merged", False)
                if merge and not is_merged:
                    _merge_lora_layer_(layer, check_adapters_to_merge)
                elif (not merge) and is_merged:
                    _unmerge_lora_layer_(layer)

    def backup_base_model_weights(module: Any) -> dict[str, tuple[str, Any]]:
        from peft import PeftModel

        backup = {}
        with torch.no_grad():
            context = module.disable_adapter() if isinstance(module, PeftModel) else nullcontext()
            with context:
                for name, param in module.named_parameters():
                    if isinstance(module, PeftModel) and "lora" in name.lower():
                        continue
                    backup[name] = _clone_to_cpu(param)
        return backup

    def restore_base_model_weights(module: Any, backup: dict[str, tuple[str, Any]]) -> None:
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in backup:
                    _copy_from_cpu(param, backup[name])

    _mark(backup_base_model_weights, "_fsdp2_lora")
    _mark(restore_base_model_weights, "_fsdp2_lora")
    _mark(_merge_or_unmerge_lora_, "_fsdp2_lora")
    fsdp_utils.backup_base_model_weights = backup_base_model_weights
    fsdp_utils.restore_base_model_weights = restore_base_model_weights
    fsdp_utils._merge_or_unmerge_lora_ = _merge_or_unmerge_lora_
    _patch_layered_summon_lora_params()

    original_fsdp2_load_full_state_dict = getattr(fsdp_utils, "fsdp2_load_full_state_dict", None)
    if callable(original_fsdp2_load_full_state_dict) and not _is_marked(
        original_fsdp2_load_full_state_dict, "_fsdp2_cpu_offload_load"
    ):

        def fsdp2_load_full_state_dict(model, full_state, device_mesh=None, cpu_offload=None):  # type: ignore[no-untyped-def]
            original_fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload)
            if cpu_offload is None:
                return

            moved = 0
            with torch.no_grad():
                for param in model.parameters():
                    if getattr(param, "device", None) is not None and param.device.type != "cpu":
                        param.data = param.data.to("cpu", non_blocking=True)
                        moved += 1

            print(
                "MinT Qwen3.6 veRL FSDP2 LoRA patch: ensured FSDP2 CPU-offload "
                f"params on CPU after full-state load (moved={moved})",
                file=sys.stderr,
                flush=True,
            )

        _mark(fsdp2_load_full_state_dict, "_fsdp2_cpu_offload_load")
        fsdp2_load_full_state_dict._mint_qwen36_original = original_fsdp2_load_full_state_dict  # type: ignore[attr-defined]
        fsdp_utils.fsdp2_load_full_state_dict = fsdp2_load_full_state_dict


def _patch_verl_fsdp_transformer_impl_module(transformer_impl: ModuleType) -> None:
    cls = getattr(transformer_impl, "FSDPEngine", None)
    original = getattr(cls, "get_per_tensor_param", None) if cls is not None else None
    if original is None or _is_marked(original, "_fsdp_transformer_impl"):
        return

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):  # type: ignore[no-untyped-def]
        merge_lora = self.model_config.lora.get("merge", False)
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not (merge_lora and hasattr(peft_model, "peft_config")):
            return original(self, layered_summon=layered_summon, base_sync_done=base_sync_done, **kwargs)

        from torch.distributed.tensor import DTensor
        from verl.utils.fsdp_utils import collect_merged_lora_params

        transformer_impl.logger.warning(
            "MinT Qwen3.6 veRL FSDP2 LoRA patch: collecting merged LoRA params without module.state_dict()"
        )
        transformer_impl.log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=transformer_impl.logger)
        transformer_impl.load_fsdp_model_to_gpu(self.module)
        transformer_impl.log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=transformer_impl.logger)

        try:
            params = collect_merged_lora_params(self.module)
            params = transformer_impl.normalize_peft_param_name(params)
            params = transformer_impl.convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
        finally:
            transformer_impl.log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=transformer_impl.logger)
            if self._is_offload_param:
                transformer_impl.offload_fsdp_model_to_cpu(self.module)
            transformer_impl.log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=transformer_impl.logger)

        device = transformer_impl.get_device_id()

        def _to_sglang_update_tensor(param):  # type: ignore[no-untyped-def]
            if isinstance(param, DTensor):
                param = param.to(device, non_blocking=True).full_tensor()
            else:
                param = param.to(device, non_blocking=True)
            return param.detach().to(transformer_impl.torch.bfloat16, non_blocking=True)

        per_tensor_param = ((name, _to_sglang_update_tensor(param)) for name, param in params.items())

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = transformer_impl.torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=transformer_impl.torch.device(transformer_impl.get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=transformer_impl.torch.device("cpu"),
            )

        peft_config = getattr(peft_model, "peft_config", {}).get("default")
        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    _mark(get_per_tensor_param, "_fsdp_transformer_impl")
    get_per_tensor_param._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls.get_per_tensor_param = get_per_tensor_param


def _patch_verl_qwen35_text_only_module(qwen35_module: ModuleType) -> None:
    original = getattr(qwen35_module, "_get_input_embeds", None)
    if original is None or _is_marked(original, "_qwen35_text_only"):
        return

    def _get_input_embeds(
        model,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        skip_dummy_visual = _env_flag(
            QWEN36_TEXT_ONLY_SKIP_DUMMY_VISUAL_ENV,
            default=_env_flag("QWEN35_TEXT_ONLY_SKIP_DUMMY_VISUAL", default=True),
        )
        if not skip_dummy_visual or pixel_values is not None or pixel_values_videos is not None:
            return original(
                model,
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        inputs_embeds = model.get_input_embeddings()(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    _mark(_get_input_embeds, "_qwen35_text_only")
    _get_input_embeds._mint_qwen36_original = original  # type: ignore[attr-defined]
    qwen35_module._get_input_embeds = _get_input_embeds


def _patch_sglang_scheduler_update_weights_mixin_module(scheduler_update_weights_mixin: ModuleType) -> None:
    cls = getattr(scheduler_update_weights_mixin, "SchedulerUpdateWeightsMixin", None)
    original = getattr(cls, "resume_memory_occupation", None) if cls is not None else None
    if original is None or _is_marked(original, "_sglang_resume_memory"):
        return

    def resume_memory_occupation(self, recv_req):  # type: ignore[no-untyped-def]
        import torch

        tags = recv_req.tags
        if tags is None or len(tags) == 0:
            tags = scheduler_update_weights_mixin.GPU_MEMORY_ALL_TYPES
        tags = list(tags)

        offload_tags = getattr(self, "offload_tags", set())
        active_tags = [tag for tag in tags if tag in offload_tags]
        missing_tags = [tag for tag in tags if tag not in offload_tags]
        if missing_tags:
            print(
                "MinT Qwen3.6 veRL FSDP2 LoRA patch: SGLang "
                f"resume_memory_occupation skipped non-offloaded tags {missing_tags} "
                f"(requested={tags}, active={sorted(offload_tags)})",
                file=sys.stderr,
                flush=True,
            )

        for tag in active_tags:
            offload_tags.discard(tag)

        if not active_tags:
            return scheduler_update_weights_mixin.ResumeMemoryOccupationReqOutput()

        if scheduler_update_weights_mixin.GPU_MEMORY_TYPE_CUDA_GRAPH in active_tags:
            self.memory_saver_adapter.resume(scheduler_update_weights_mixin.GPU_MEMORY_TYPE_CUDA_GRAPH)

        if scheduler_update_weights_mixin.GPU_MEMORY_TYPE_WEIGHTS in active_tags:
            self.memory_saver_adapter.resume(scheduler_update_weights_mixin.GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            stashed_model_static_state = getattr(self, "stashed_model_static_state", None)
            if stashed_model_static_state is not None:
                scheduler_update_weights_mixin._import_static_state(
                    self.tp_worker.model_runner.model,
                    stashed_model_static_state,
                )
                del self.stashed_model_static_state
            else:
                print(
                    "MinT Qwen3.6 veRL FSDP2 LoRA patch: SGLang weights tag "
                    "was active but no stashed static state existed",
                    file=sys.stderr,
                    flush=True,
                )

        if scheduler_update_weights_mixin.GPU_MEMORY_TYPE_KV_CACHE in active_tags:
            self.memory_saver_adapter.resume(scheduler_update_weights_mixin.GPU_MEMORY_TYPE_KV_CACHE)

        return scheduler_update_weights_mixin.ResumeMemoryOccupationReqOutput()

    _mark(resume_memory_occupation, "_sglang_resume_memory")
    resume_memory_occupation._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls.resume_memory_occupation = resume_memory_occupation


def _patch_verl_sglang_rollout_module(sglang_rollout: ModuleType) -> None:
    cls = getattr(sglang_rollout, "ServerAdapter", None)
    original = getattr(cls, "wrap_lora_params", None) if cls is not None else None
    if original is None or _is_marked(original, "_sglang_rollout_wrap_lora"):
        return

    def _jsonable_peft_config(peft_config: Any) -> dict[str, Any]:
        from dataclasses import asdict, is_dataclass

        if isinstance(peft_config, dict):
            peft_config_json = dict(peft_config)
        elif hasattr(peft_config, "to_dict"):
            peft_config_json = dict(peft_config.to_dict())
        elif is_dataclass(peft_config):
            peft_config_json = asdict(peft_config)
        else:
            raise TypeError(
                "Unsupported PEFT config type for SGLang LoRA tensor load: "
                f"{type(peft_config).__module__}.{type(peft_config).__qualname__}"
            )

        for key in ("task_type", "peft_type"):
            value = peft_config_json.get(key)
            if hasattr(value, "value"):
                peft_config_json[key] = value.value
            elif value is None:
                peft_config_json[key] = None
            else:
                peft_config_json[key] = str(value)

        target_modules = peft_config_json.get("target_modules")
        if target_modules is None:
            peft_config_json["target_modules"] = []
        elif isinstance(target_modules, (set, frozenset)):
            peft_config_json["target_modules"] = sorted(target_modules)
        else:
            peft_config_json["target_modules"] = list(target_modules)

        return peft_config_json

    def _serialize_lora_tensors_by_value(processed_weights: dict[str, Any]) -> str:
        payload = pickle.dumps(processed_weights, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(payload).decode("utf-8")

    def wrap_lora_params(self, peft_config, weights):  # type: ignore[no-untyped-def]
        peft_config_json = _jsonable_peft_config(peft_config)
        processed_weights = {}
        for name, tensor in weights:
            processed = sglang_rollout._preprocess_tensor_for_update_weights(tensor.detach())
            processed_weights[name] = processed.detach().cpu().contiguous()

        serialized_tensors = _serialize_lora_tensors_by_value(processed_weights)
        print(
            "MinT Qwen3.6 veRL FSDP2 LoRA patch: serialized PEFT config for "
            f"SGLang LoRA tensor load (tensors={len(processed_weights)}, "
            f"r={peft_config_json.get('r')}, alpha={peft_config_json.get('lora_alpha')}, "
            "serialized_tensors_mode=pickle_by_value_cpu_base64)",
            file=sys.stderr,
            flush=True,
        )
        return peft_config_json, serialized_tensors

    _mark(wrap_lora_params, "_sglang_rollout_wrap_lora")
    wrap_lora_params._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls.wrap_lora_params = wrap_lora_params


def _patch_verl_sglang_http_server_engine_module(http_server_engine: ModuleType) -> None:
    cls = getattr(http_server_engine, "AsyncHttpServerAdapter", None)
    original = getattr(cls, "_make_async_request", None) if cls is not None else None
    if original is None or _is_marked(original, "_sglang_http_error_body"):
        return

    async def _make_async_request(
        self,
        endpoint,
        payload=None,
        method="POST",
        timeout=None,
        only_master=True,
    ):
        if timeout is None:
            timeout = getattr(http_server_engine, "DEFAULT_TIMEOUT", 60.0)
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        import aiohttp
        import asyncio

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            if response.status >= 400:
                                body = await http_server_engine._read_async_response(response)
                                print(
                                    "MinT Qwen3.6 veRL FSDP2 LoRA patch: HTTP error body "
                                    f"endpoint={endpoint}, status={response.status}, body={body}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                response.raise_for_status()
                            return await http_server_engine._read_async_response(response)
                    async with session.post(url, json=payload or {}, timeout=timeout) as response:
                        if response.status >= 400:
                            body = await http_server_engine._read_async_response(response)
                            payload_summary = {}
                            if isinstance(payload, dict):
                                payload_summary = {
                                    key: (
                                        f"list(len={len(value)})"
                                        if isinstance(value, list)
                                        else f"dict(keys={sorted(value)[:12]})"
                                        if isinstance(value, dict)
                                        else type(value).__name__
                                    )
                                    for key, value in payload.items()
                                }
                            print(
                                "MinT Qwen3.6 veRL FSDP2 LoRA patch: HTTP error body "
                                f"endpoint={endpoint}, status={response.status}, "
                                f"payload_summary={payload_summary}, body={body}",
                                file=sys.stderr,
                                flush=True,
                            )
                            response.raise_for_status()
                        return await http_server_engine._read_async_response(response)
            except asyncio.TimeoutError:
                http_server_engine.logger.warning("async_request_timeout", endpoint=endpoint, attempt=attempt + 1)
            except aiohttp.ClientConnectorError:
                http_server_engine.logger.warning("connection_error", endpoint=endpoint, attempt=attempt + 1)
            except aiohttp.ClientResponseError as e:
                http_server_engine.logger.error("http_error", endpoint=endpoint, error=str(e))
                raise
            except Exception as e:
                http_server_engine.logger.error("unexpected_error", endpoint=endpoint, error=str(e))
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

    _mark(_make_async_request, "_sglang_http_error_body")
    _make_async_request._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls._make_async_request = _make_async_request


def _patch_verl_llm_server_module(llm_server: ModuleType) -> None:
    cls = getattr(llm_server, "LLMServerManager", None)
    original = getattr(cls, "_init_global_load_balancer", None) if cls is not None else None
    if original is None or _is_marked(original, "_llm_server_pin"):
        return

    async def _init_global_load_balancer(self):  # type: ignore[no-untyped-def]
        resources = _target_node_resources()
        load_balancer_cls = llm_server.GlobalRequestLoadBalancer
        if resources:
            load_balancer_cls = load_balancer_cls.options(resources=resources)
            print(
                "MinT Qwen3.6 veRL FSDP2 LoRA patch: pinning "
                f"GlobalRequestLoadBalancer to resources {resources}",
                file=sys.stderr,
                flush=True,
            )

        self.global_load_balancer = load_balancer_cls.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=llm_server.DEFAULT_ROUTING_CACHE_SIZE,
        )

    _mark(_init_global_load_balancer, "_llm_server_pin")
    _init_global_load_balancer._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls._init_global_load_balancer = _init_global_load_balancer


def _patch_verl_agent_loop_module(agent_loop: ModuleType) -> None:
    cls = getattr(agent_loop, "AgentLoopManager", None)
    original = getattr(cls, "_init_agent_loop_workers", None) if cls is not None else None
    if original is None or _is_marked(original, "_agent_loop_pin"):
        return

    async def _init_agent_loop_workers(self):  # type: ignore[no-untyped-def]
        resources = _target_node_resources()
        if not resources:
            return await original(self)

        import ray

        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        target_ip = _target_node_ip()
        node_id = _target_alive_node_id(ray, purpose="agent loop")
        for i in range(num_workers):
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{agent_loop.uuid4().hex[:8]}",
                    resources=resources,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

        print(
            "MinT Qwen3.6 veRL FSDP2 LoRA patch: pinned AgentLoopWorker actors "
            f"to {target_ip} with resources {resources} (count={num_workers})",
            file=sys.stderr,
            flush=True,
        )

    _mark(_init_agent_loop_workers, "_agent_loop_pin")
    _init_agent_loop_workers._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls._init_agent_loop_workers = _init_agent_loop_workers


def _patch_verl_reward_loop_module(reward_loop: ModuleType) -> None:
    cls = getattr(reward_loop, "RewardLoopManager", None)
    original = getattr(cls, "_init_reward_loop_workers", None) if cls is not None else None
    if original is None or _is_marked(original, "_reward_loop_pin"):
        return

    def _init_reward_loop_workers(self):  # type: ignore[no-untyped-def]
        resources = _target_node_resources()
        if not resources:
            return original(self)

        import ray

        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        target_ip = _target_node_ip()
        node_id = _target_alive_node_id(ray, purpose="reward loop")

        for i in range(num_workers):
            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    name=f"reward_loop_worker_{i}",
                    resources=resources,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

        print(
            "MinT Qwen3.6 veRL FSDP2 LoRA patch: pinned RewardLoopWorker actors "
            f"to {target_ip} with resources {resources} (count={num_workers})",
            file=sys.stderr,
            flush=True,
        )

    _mark(_init_reward_loop_workers, "_reward_loop_pin")
    _init_reward_loop_workers._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls._init_reward_loop_workers = _init_reward_loop_workers


def _patch_verl_sglang_async_server_module(async_sglang_server: ModuleType) -> None:
    cls = getattr(async_sglang_server, "SGLangHttpServer", None)
    original = getattr(cls, "generate", None) if cls is not None else None
    if original is None or _is_marked(original, "_sglang_generate_lora"):
        return

    try:
        source = textwrap.dedent(inspect.getsource(original))
    except Exception:
        source = ""

    patched = None
    marker = "if self.model_config.lora_rank > 0:"
    if marker in source:
        namespace = dict(async_sglang_server.__dict__)
        exec(source.replace(marker, "if self.lora_as_adapter:"), namespace)
        patched = namespace.get("generate")

    if patched is None:

        async def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            model_lora = getattr(self.model_config, "lora", {}) or {}
            merge_lora = bool(model_lora.get("merge", False))
            if merge_lora and getattr(self.model_config, "lora_rank", 0) > 0:
                original_lora_rank = self.model_config.lora_rank
                try:
                    self.model_config.lora_rank = 0
                    return await original(self, *args, **kwargs)
                finally:
                    self.model_config.lora_rank = original_lora_rank

            return await original(self, *args, **kwargs)

        patched._mint_qwen36_patch_mode = "temporary_lora_rank_zero"
    else:
        patched._mint_qwen36_patch_mode = "source_lora_as_adapter"

    _mark(patched, "_sglang_generate_lora")
    patched._mint_qwen36_original = original  # type: ignore[attr-defined]
    cls.generate = patched
    print(
        "MinT Qwen3.6 veRL FSDP2 LoRA patch: SGLangHttpServer.generate "
        f"patched ({patched._mint_qwen36_patch_mode})",
        file=sys.stderr,
        flush=True,
    )


def install_qwen36_verl_fsdp2_lora_patches() -> None:
    """Install all Qwen3.6 veRL FSDP2 LoRA runtime patches."""

    os.environ.setdefault(QWEN36_TEXT_ONLY_SKIP_DUMMY_VISUAL_ENV, "1")
    _patch_transformers_auto_model_vision2seq_alias()
    _patch_accelerate_init_on_device()

    _install_import_patch("verl.utils.attention_utils", _patch_verl_attention_utils_module)
    _install_import_patch("torch.distributed.fsdp._fully_shard._fsdp_param_group", _patch_torch_fsdp2_cpu_offload_validation_module)
    _install_import_patch("verl.single_controller.base.ray", _patch_verl_ray_resource_pool_module)
    _install_import_patch("verl.utils.fsdp_utils", _patch_verl_fsdp2_lora_backup_module)
    _install_import_patch("verl.workers.engine.fsdp.transformer_impl", _patch_verl_fsdp_transformer_impl_module)
    _install_import_patch("verl.models.transformers.qwen3_5", _patch_verl_qwen35_text_only_module)
    _install_import_patch("sglang.srt.managers.scheduler_update_weights_mixin", _patch_sglang_scheduler_update_weights_mixin_module)
    _install_import_patch("verl.workers.rollout.sglang_rollout.sglang_rollout", _patch_verl_sglang_rollout_module)
    _install_import_patch("verl.workers.rollout.sglang_rollout.http_server_engine", _patch_verl_sglang_http_server_engine_module)
    _install_import_patch("verl.workers.rollout.llm_server", _patch_verl_llm_server_module)
    _install_import_patch("verl.experimental.agent_loop.agent_loop", _patch_verl_agent_loop_module)
    _install_import_patch("verl.experimental.reward_loop.reward_loop", _patch_verl_reward_loop_module)
    _install_import_patch("verl.workers.rollout.sglang_rollout.async_sglang_server", _patch_verl_sglang_async_server_module)


__all__ = [
    "QWEN36_MODEL_ID",
    "QWEN36_PATCH_ENV",
    "QWEN36_VERL_FSDP2_LORA_BACKEND",
    "install_qwen36_verl_fsdp2_lora_patches",
    "is_qwen36_model",
    "qwen36_model_path_override",
]
