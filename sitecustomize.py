"""Process-startup patches for Ray/vLLM worker subprocesses.

Python automatically imports `sitecustomize` (if present on sys.path) on
interpreter startup. We use this to patch code paths that run in vLLM
subprocesses spawned with the `spawn` method, where in-process monkey patches
from the parent process do not propagate.

This file activates only when explicitly enabled via environment variables
propagated into vLLM worker processes.
"""

from __future__ import annotations

import importlib.util
import multiprocessing.spawn as _mp_spawn
import os
import sys
import sysconfig


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _is_cv2_package_dir(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    norm = os.path.normpath(path)
    return norm.endswith("/site-packages/cv2")


def _is_cv2_typing_file(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    norm = os.path.normpath(path)
    return norm.endswith("/site-packages/cv2/typing/__init__.py")


def _sanitize_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        if _is_cv2_package_dir(p):
            continue
        if p in out:
            continue
        out.append(p)
    return out


def _patch_cv2_typing_shadow() -> None:
    """Prevent accidental import of `cv2/typing` as top-level `typing`.

    Some worker subprocesses inherit polluted sys.path containing
    `.../site-packages/cv2`, which can shadow stdlib `typing.py` and crash with
    `ImportError: libxcb.so.1`.
    """
    if _env_flag("MINT_DISABLE_CV2_TYPING_PATCH", default=False):
        return

    # 1) Clean current process paths.
    sys.path[:] = _sanitize_paths(list(sys.path))

    raw_py = os.environ.get("PYTHONPATH", "")
    if raw_py:
        parts = [p.strip() for p in raw_py.split(":")]
        os.environ["PYTHONPATH"] = ":".join(_sanitize_paths(parts))

    # 2) Ensure multiprocessing spawn does not reintroduce bad paths.
    orig = _mp_spawn.get_preparation_data
    if not getattr(orig, "__mint_cv2_typing_patched__", False):
        def _mint_get_preparation_data(*args, **kwargs):
            data = orig(*args, **kwargs)
            raw = data.get("sys_path")
            if isinstance(raw, list):
                data["sys_path"] = _sanitize_paths(raw)
            return data

        _mint_get_preparation_data.__mint_cv2_typing_patched__ = True  # type: ignore[attr-defined]
        _mp_spawn.get_preparation_data = _mint_get_preparation_data  # type: ignore[assignment]

    # 3) Pin top-level typing module to stdlib implementation.
    try:
        import typing as _typing  # noqa: F401
    except Exception:
        _typing = None  # type: ignore[assignment]

    if _typing is None or _is_cv2_typing_file(getattr(_typing, "__file__", None)):
        stdlib_typing = os.path.join(sysconfig.get_path("stdlib"), "typing.py")
        spec = importlib.util.spec_from_file_location("typing", stdlib_typing)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["typing"] = mod


def _patch_vllm_fused_moe_slice_for_fully_sharded_loras() -> None:
    import vllm.lora.layers.fused_moe as fused_moe_mod

    def _patch_cls(cls: type) -> None:
        original = getattr(cls, "_slice_w13_a", None)
        if original is None:
            raise RuntimeError(f"vLLM class {cls.__name__} has no _slice_w13_a")
        if getattr(original, "_tinker_patched_fully_sharded", False):
            return

        def _slice_w13_a(self, w13_lora_a):  # type: ignore[no-untyped-def]
            if self.tp_size == 1 or not self.fully_sharded:
                return w13_lora_a

            if getattr(w13_lora_a, "ndim", None) != 3:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a expects 3D w13_lora_a: "
                    f"w13_lora_a.ndim={getattr(w13_lora_a, 'ndim', None)}"
                )

            # Robust rank-dimension inference for K2: the LoRA rank is the smallest
            # non-expert dimension (typically R=64). Fully-sharded LoRA shards that
            # rank across TP, so each rank sees R_local = R / tp_size.
            d1, d2 = int(w13_lora_a.shape[1]), int(w13_lora_a.shape[2])
            rank_dim = 1 if d1 <= d2 else 2
            global_rank = min(d1, d2)
            if global_rank <= 0:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a invalid global_rank: "
                    f"w13_lora_a.shape={tuple(w13_lora_a.shape)}"
                )
            if global_rank > 4096:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a suspicious global_rank (refusing to slice): "
                    f"w13_lora_a.shape={tuple(w13_lora_a.shape)} tp_size={self.tp_size}"
                )
            if global_rank % int(self.tp_size) != 0:
                # Already sharded (or unexpected rank); do not touch.
                return w13_lora_a

            local_rank = global_rank // int(self.tp_size)
            start_idx = int(self.tp_rank) * local_rank
            end_idx = (int(self.tp_rank) + 1) * local_rank
            if rank_dim == 1:
                return w13_lora_a[:, start_idx:end_idx, :]
            return w13_lora_a[:, :, start_idx:end_idx]

        _slice_w13_a._tinker_patched_fully_sharded = True  # type: ignore[attr-defined]
        cls._slice_w13_a = _slice_w13_a  # type: ignore[method-assign]

    for name in ("FusedMoEWithLoRA", "FusedMoE3DWithLoRA"):
        cls = getattr(fused_moe_mod, name, None)
        if cls is None:
            raise RuntimeError(f"vLLM fused_moe is missing class {name}")
        _patch_cls(cls)


def _patch_vllm_skip_dummy_lora_setup_when_inactive() -> None:
    """Avoid expensive dummy-LoRA warmup during profiling runs.

    In vLLM v1, `GPUModelRunner.profile_run()` calls `_dummy_run(..., is_profile=True)`
    with `activate_lora=False` by default. However, `maybe_dummy_run_with_lora(...)`
    can still create dummy LoRAs unconditionally, which is extremely expensive for
    large MoE models (e.g. K2) and can stall engine startup.

    This patch keeps the existing behavior when LoRA is actually activated, but
    skips dummy-LoRA setup/selection when `activate_lora=False`.
    """

    from contextlib import contextmanager

    import vllm.v1.worker.lora_model_runner_mixin as mixin_mod

    cls = getattr(mixin_mod, "LoRAModelRunnerMixin", None)
    if cls is None:
        raise RuntimeError("vLLM missing LoRAModelRunnerMixin")

    original = getattr(cls, "maybe_dummy_run_with_lora", None)
    if original is None:
        raise RuntimeError("vLLM LoRAModelRunnerMixin missing maybe_dummy_run_with_lora")
    if getattr(original, "_tinker_patched_skip_dummy_inactive", False):
        return

    @contextmanager
    def maybe_dummy_run_with_lora(  # type: ignore[no-untyped-def]
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        activate_lora: bool = True,
        remove_lora: bool = True,
    ):
        if lora_config is not None and not activate_lora:
            self.maybe_remove_all_loras(lora_config)
            yield
            return

        with original(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            yield

    maybe_dummy_run_with_lora._tinker_patched_skip_dummy_inactive = True  # type: ignore[attr-defined]
    cls.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora  # type: ignore[method-assign]


def _patch_vllm_pack_moe_sparse_ok() -> None:
    """Patch vLLM MoE LoRA packing to tolerate missing expert adapters.

    vLLM's PackedLoRALayerWeights.pack_moe can assume every expert has (w1,w2,w3)
    LoRA weights present. Our adapter export can be shared across experts (export
    expert 0 only). Missing experts should be treated as sharing the base expert
    weights, without materializing full per-expert tensors when possible.
    """

    from vllm.lora import lora_weights as lw  # type: ignore

    Packed = getattr(lw, "PackedLoRALayerWeights", None)
    LoRALayerWeights = getattr(lw, "LoRALayerWeights", None)
    if Packed is None or LoRALayerWeights is None:
        raise RuntimeError("vLLM lora_weights types missing; cannot patch pack_moe")

    cm = Packed.__dict__.get("pack_moe")
    orig_fn = getattr(cm, "__func__", None)
    if orig_fn is None:
        raise RuntimeError("vLLM pack_moe not found; cannot patch")
    if getattr(orig_fn, "__mint_sparse_ok__", False):
        return

    import inspect

    try:
        sig = inspect.signature(orig_fn)
    except Exception as e:
        raise RuntimeError(
            f"Unable to inspect vLLM PackedLoRALayerWeights.pack_moe signature: {type(e).__name__}: {e}"
        ) from e

    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if "is_non_gated_moe" not in sig.parameters and not has_kwargs:
        import vllm  # type: ignore

        raise RuntimeError(
            "vLLM PackedLoRALayerWeights.pack_moe signature mismatch for sparse-ok patch: "
            f"expected 'is_non_gated_moe' kwarg (or **kwargs), got signature={sig}. "
            f"installed_vllm_version={getattr(vllm, '__version__', 'unknown')!r}"
        )

    def pack_moe_sparse_ok(cls, loras, module_name: str, is_non_gated_moe: bool = False):  # type: ignore[no-untyped-def]
        if loras and all(l is not None for l in loras):
            return orig_fn(cls, loras, module_name, is_non_gated_moe=is_non_gated_moe)

        if not loras or (len(loras) % 3) != 0:
            raise RuntimeError(
                f"Unexpected MoE LoRA pack_moe inputs for module={module_name!r}: len(loras)={len(loras)}"
            )

        n_experts = len(loras) // 3

        base_any = next((l for l in loras if l is not None), None)
        if base_any is None:
            raise RuntimeError(f"MoE LoRA pack_moe got all-None loras for module={module_name!r}")
        rank = int(getattr(base_any, "rank"))
        lora_alpha = int(getattr(base_any, "lora_alpha"))

        base_w1 = next((loras[i * 3] for i in range(n_experts) if loras[i * 3] is not None), None)
        base_w2 = next((loras[i * 3 + 1] for i in range(n_experts) if loras[i * 3 + 1] is not None), None)
        base_w3 = next((loras[i * 3 + 2] for i in range(n_experts) if loras[i * 3 + 2] is not None), None)
        if base_w1 is None or base_w2 is None or base_w3 is None:
            raise RuntimeError(f"MoE LoRA pack_moe missing base weight(s) for module={module_name!r}")

        only_expert0 = True
        for eid in range(1, n_experts):
            if (
                loras[eid * 3] is not None
                or loras[eid * 3 + 1] is not None
                or loras[eid * 3 + 2] is not None
            ):
                only_expert0 = False
                break

        if only_expert0:
            # Shared-expert export: avoid materializing [num_experts, ...] tensors.
            #
            # vLLM later calls `optimize()` (in-place scaling merge) on packed LoRA
            # weights. `expand(...)` returns a view with overlapping storage, so
            # in-place ops (e.g., `lora_b *= scaling`) crash with:
            #   "unsupported operation: more than one element ... refers to a single
            #    memory location"
            #
            # Keep the non-materialized representation, but pre-apply scaling
            # out-of-place and mark scaling=1 so vLLM's in-place optimize is a no-op.
            w1_lora_a = base_w1.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w1.lora_a.shape))
            w2_lora_a = base_w2.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w2.lora_a.shape))
            w3_lora_a = base_w3.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w3.lora_a.shape))
            w1_lora_b_base = base_w1.lora_b * float(getattr(base_w1, "scaling", lora_alpha / rank))
            w2_lora_b_base = base_w2.lora_b * float(getattr(base_w2, "scaling", lora_alpha / rank))
            w3_lora_b_base = base_w3.lora_b * float(getattr(base_w3, "scaling", lora_alpha / rank))
            w1_lora_b = w1_lora_b_base.unsqueeze(0).expand((n_experts,) + tuple(w1_lora_b_base.shape))
            w2_lora_b = w2_lora_b_base.unsqueeze(0).expand((n_experts,) + tuple(w2_lora_b_base.shape))
            w3_lora_b = w3_lora_b_base.unsqueeze(0).expand((n_experts,) + tuple(w3_lora_b_base.shape))
            packed_scaling = [1.0, 1.0, 1.0]
        else:
            # General sparse case: default missing experts to base weights, but allow
            # explicitly-provided experts to override.
            w1_lora_a = base_w1.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w1.lora_a.shape)).clone()
            w2_lora_a = base_w2.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w2.lora_a.shape)).clone()
            w3_lora_a = base_w3.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w3.lora_a.shape)).clone()
            w1_lora_b = base_w1.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w1.lora_b.shape)).clone()
            w2_lora_b = base_w2.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w2.lora_b.shape)).clone()
            w3_lora_b = base_w3.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w3.lora_b.shape)).clone()

            for eid in range(n_experts):
                w1 = loras[eid * 3]
                w2 = loras[eid * 3 + 1]
                w3 = loras[eid * 3 + 2]
                if w1 is not None:
                    w1_lora_a[eid].copy_(w1.lora_a)
                    w1_lora_b[eid].copy_(w1.lora_b)
                if w2 is not None:
                    w2_lora_a[eid].copy_(w2.lora_a)
                    w2_lora_b[eid].copy_(w2.lora_b)
                if w3 is not None:
                    w3_lora_a[eid].copy_(w3.lora_a)
                    w3_lora_b[eid].copy_(w3.lora_b)
            packed_scaling = None

        return cls(
            module_name,
            rank,
            [lora_alpha, lora_alpha, lora_alpha],
            [w1_lora_a, w2_lora_a, w3_lora_a],
            [w1_lora_b, w2_lora_b, w3_lora_b],
            scaling=packed_scaling,
        )

    pack_moe_sparse_ok.__mint_sparse_ok__ = True  # type: ignore[attr-defined]
    Packed.pack_moe = classmethod(pack_moe_sparse_ok)


def _patch_vllm_lora_optimize_overlap_safe() -> None:
    """Avoid in-place LoRA optimize on overlapping tensors.

    vLLM merges scaling into `lora_b` via in-place ops (`*= scaling`) inside
    LoRA `optimize()`. This fails for tensors with internal overlap (commonly
    produced by `expand(...)`), raising:
      "unsupported operation: more than one element ... refers to a single
       memory location"

    When overlap is detected, use out-of-place multiplication and replace the
    tensor reference to preserve semantics without invalid writes.
    """

    try:
        import torch
        from vllm.lora import lora_weights as lw  # type: ignore
    except Exception:
        return

    LoRA = getattr(lw, "LoRALayerWeights", None)
    Packed = getattr(lw, "PackedLoRALayerWeights", None)
    if LoRA is None or Packed is None:
        return

    def _has_internal_overlap(t: "torch.Tensor") -> bool:  # type: ignore[name-defined]
        try:
            return bool(torch._C._debug_has_internal_overlap(t))  # type: ignore[attr-defined]
        except Exception:
            return False

    orig_opt = getattr(LoRA, "optimize", None)
    if callable(orig_opt) and not getattr(orig_opt, "_tinker_overlap_safe", False):

        def optimize(self):  # type: ignore[no-untyped-def]
            if getattr(self, "scaling", 1) == 1:
                return self
            lb = getattr(self, "lora_b", None)
            if lb is None:
                return self
            if _has_internal_overlap(lb):
                self.lora_b = lb * float(self.scaling)
            else:
                self.lora_b *= float(self.scaling)
            self.scaling = 1
            return self

        optimize._tinker_overlap_safe = True  # type: ignore[attr-defined]
        LoRA.optimize = optimize  # type: ignore[method-assign]

    orig_popt = getattr(Packed, "optimize", None)
    if callable(orig_popt) and not getattr(orig_popt, "_tinker_overlap_safe", False):

        def optimize(self):  # type: ignore[no-untyped-def]
            for i in range(len(self.lora_b)):
                if self.scaling[i] == 1 or self.lora_b[i] is None:  # type: ignore
                    continue
                lb = self.lora_b[i]  # type: ignore
                if _has_internal_overlap(lb):
                    self.lora_b[i] = lb * float(self.scaling[i])  # type: ignore
                else:
                    self.lora_b[i] *= float(self.scaling[i])  # type: ignore
                self.scaling[i] = 1  # type: ignore
            return self

        optimize._tinker_overlap_safe = True  # type: ignore[attr-defined]
        Packed.optimize = optimize  # type: ignore[method-assign]


def _patch_vllm_lora_pin_memory_overlap_safe() -> None:
    """Avoid pin_memory() crash on overlapping MoE LoRA tensors.

    vLLM pins LoRA tensors after packing/merging inside
    `LoRAModelManager._create_merged_loras_inplace`:
      lora.lora_b[index] = lora.lora_b[index].pin_memory()

    When MoE LoRA weights are represented as `expand(...)` views (to avoid
    materializing [num_experts, ...] tensors for shared-expert exports), calling
    `pin_memory()` can fail with:
      "unsupported operation: more than one element ... refers to a single
       memory location"

    Do not materialize the expanded tensor. Instead, leave it unpinned when
    pinning fails due to internal overlap.
    """

    try:
        import torch
        from vllm.lora.model_manager import LoRAModelManager  # type: ignore
    except Exception:
        return

    orig = getattr(LoRAModelManager, "_create_merged_loras_inplace", None)
    if not callable(orig) or getattr(orig, "_tinker_pin_memory_overlap_safe", False):
        return

    def _create_merged_loras_inplace(self, lora_model):  # type: ignore[no-untyped-def]
        orig_pin = torch.Tensor.pin_memory  # type: ignore[attr-defined]

        def _safe_pin_memory(t, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return orig_pin(t, *args, **kwargs)
            except RuntimeError as e:
                msg = str(e)
                if "more than one element of the written-to tensor refers to a single memory location" in msg:
                    return t
                raise

        torch.Tensor.pin_memory = _safe_pin_memory  # type: ignore[assignment]
        try:
            return orig(self, lora_model)
        finally:
            torch.Tensor.pin_memory = orig_pin  # type: ignore[assignment]

    _create_merged_loras_inplace._tinker_pin_memory_overlap_safe = True  # type: ignore[attr-defined]
    LoRAModelManager._create_merged_loras_inplace = _create_merged_loras_inplace  # type: ignore[method-assign]


def _patch_vllm_ray_env_carry_over_pythonpath() -> None:
    """Ensure vLLM Ray workers start with our PYTHONPATH.

    vLLM's Ray backend uses `vllm.ray.ray_env.get_env_vars_to_copy()` to decide
    which env vars are propagated via Ray runtime_env at actor startup.

    By default it only carries vLLM-defined env vars; it does not include
    `PYTHONPATH`, so worker processes (EngineCore, TP workers, etc.) may not
    import `sitecustomize.py`, and thus miss our vLLM monkey patches.
    """

    try:
        import vllm.ray.ray_env as ray_env  # type: ignore
    except Exception:
        return

    orig = getattr(ray_env, "get_env_vars_to_copy", None)
    if not callable(orig) or getattr(orig, "_tinker_pythonpath_carryover", False):
        return

    def get_env_vars_to_copy(  # type: ignore[no-untyped-def]
        exclude_vars=None,
        additional_vars=None,
        destination=None,
    ):
        extra = {
            "PYTHONPATH",
            "MINT_ENABLE_VLLM_IMPORT_PATCHES",
            "VLLM_USE_V1",
            "MINT_VLLM_DISABLE_MOE_LORA_PACKING",
            "MINT_VLLM_FULLY_SHARDED_LORAS",
            "MINT_VLLM_DISABLE_TORCH_DIST_TP",
            "TVM_FFI_DISABLE_TORCH_C_DLPACK",
        }
        if additional_vars is None:
            additional_vars2 = set(extra)
        else:
            additional_vars2 = set(additional_vars) | set(extra)
        return orig(
            exclude_vars=exclude_vars,
            additional_vars=additional_vars2,
            destination=destination,
        )

    get_env_vars_to_copy._tinker_pythonpath_carryover = True  # type: ignore[attr-defined]
    ray_env.get_env_vars_to_copy = get_env_vars_to_copy  # type: ignore[method-assign]


def _patch_vllm_fused_moe_lora_use_torch_dist_tp_collectives() -> None:
    """Force torch.distributed collectives in fused_moe_lora TP path."""

    if _env_flag("MINT_VLLM_DISABLE_TORCH_DIST_TP", default=False):
        return

    try:
        import importlib

        import torch
        import torch.distributed as dist

        op = importlib.import_module("vllm.lora.ops.triton_ops.fused_moe_lora_op")
    except Exception:
        return

    if getattr(op, "_tinker_patched_fused_moe_lora_torch_dist_tp", False):
        return

    def _get_tp_process_group():  # type: ignore[no-untyped-def]
        try:
            import vllm.distributed.parallel_state as ps

            if not ps.model_parallel_is_initialized():
                return None
            tp = ps.get_tp_group()
            for attr in ("process_group", "pg", "group", "device_group", "_group"):
                g = getattr(tp, attr, None)
                if g is not None and not isinstance(g, bool):
                    return g
            for meth in ("get_process_group", "get_group", "get_device_group"):
                m = getattr(tp, meth, None)
                if callable(m):
                    try:
                        g = m()
                    except Exception:
                        continue
                    if g is not None and not isinstance(g, bool):
                        return g
        except Exception:
            return None
        return None

    def tensor_model_parallel_all_reduce(x):  # type: ignore[no-untyped-def]
        pg = _get_tp_process_group()
        if pg is None:
            raise RuntimeError("vLLM TP process group not found for all_reduce")
        y = x.contiguous().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=pg)
        return y

    def tensor_model_parallel_all_gather(x):  # type: ignore[no-untyped-def]
        if not hasattr(dist, "all_gather_into_tensor"):
            raise RuntimeError("torch.distributed.all_gather_into_tensor unavailable")
        pg = _get_tp_process_group()
        if pg is None:
            raise RuntimeError("vLLM TP process group not found for all_gather")
        world_size = dist.get_world_size(group=pg)
        x2 = x.contiguous()
        flat = x2.view(-1, x2.shape[-1])
        out = torch.empty(
            (flat.shape[0], flat.shape[1] * world_size),
            device=flat.device,
            dtype=flat.dtype,
        )
        dist.all_gather_into_tensor(out, flat, group=pg)
        return out.view(*x2.shape[:-1], x2.shape[-1] * world_size)

    setattr(op, "tensor_model_parallel_all_reduce", tensor_model_parallel_all_reduce)
    setattr(op, "tensor_model_parallel_all_gather", tensor_model_parallel_all_gather)
    setattr(op, "_tinker_patched_fused_moe_lora_torch_dist_tp", True)


def _apply_vllm_worker_patches() -> None:
    if not _env_flag("MINT_ENABLE_VLLM_IMPORT_PATCHES", default=False):
        return
    if "VLLM_USE_V1" not in os.environ:
        return

    # Prevent vLLM workers from spawning repeated optional builds of Torch C DLPack
    # bindings via tvm_ffi (observed as many `_build_optional_torch_c_dlpack.py`
    # processes on Ray worker nodes).
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

    _patch_vllm_ray_env_carry_over_pythonpath()
    if not _env_flag("MINT_VLLM_DISABLE_MOE_LORA_PACKING", default=False):
        _patch_vllm_pack_moe_sparse_ok()
    _patch_vllm_lora_optimize_overlap_safe()
    _patch_vllm_lora_pin_memory_overlap_safe()
    if _env_flag("MINT_VLLM_FULLY_SHARDED_LORAS", default=False):
        _patch_vllm_fused_moe_slice_for_fully_sharded_loras()
        _patch_vllm_skip_dummy_lora_setup_when_inactive()
        _patch_vllm_fused_moe_lora_use_torch_dist_tp_collectives()


_patch_cv2_typing_shadow()
_apply_vllm_worker_patches()
