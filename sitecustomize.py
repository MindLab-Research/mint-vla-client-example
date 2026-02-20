"""Process-startup patches for Ray/vLLM worker subprocesses.

Python automatically imports `sitecustomize` (if present on sys.path) on
interpreter startup. We use this to patch code paths that run in vLLM
subprocesses spawned with the `spawn` method, where in-process monkey patches
from the parent process do not propagate.

This file activates only when explicitly enabled via environment variables
propagated into vLLM worker processes.
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


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

    def pack_moe_sparse_ok(cls, loras, module_name: str):  # type: ignore[no-untyped-def]
        try:
            if loras and all(l is not None for l in loras):
                return orig_fn(cls, loras, module_name)
        except Exception:
            return orig_fn(cls, loras, module_name)

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
            w1_lora_a = base_w1.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w1.lora_a.shape))
            w2_lora_a = base_w2.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w2.lora_a.shape))
            w3_lora_a = base_w3.lora_a.unsqueeze(0).expand((n_experts,) + tuple(base_w3.lora_a.shape))
            w1_lora_b = base_w1.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w1.lora_b.shape))
            w2_lora_b = base_w2.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w2.lora_b.shape))
            w3_lora_b = base_w3.lora_b.unsqueeze(0).expand((n_experts,) + tuple(base_w3.lora_b.shape))
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

        return cls(
            module_name,
            rank,
            [lora_alpha, lora_alpha, lora_alpha],
            [w1_lora_a, w2_lora_a, w3_lora_a],
            [w1_lora_b, w2_lora_b, w3_lora_b],
        )

    pack_moe_sparse_ok.__mint_sparse_ok__ = True  # type: ignore[attr-defined]
    Packed.pack_moe = classmethod(pack_moe_sparse_ok)


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

    _patch_vllm_pack_moe_sparse_ok()
    if _env_flag("MINT_VLLM_FULLY_SHARDED_LORAS", default=False):
        _patch_vllm_fused_moe_slice_for_fully_sharded_loras()
        _patch_vllm_skip_dummy_lora_setup_when_inactive()
        _patch_vllm_fused_moe_lora_use_torch_dist_tp_collectives()


_apply_vllm_worker_patches()
