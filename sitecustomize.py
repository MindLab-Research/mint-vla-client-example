"""Process-wide Python hooks for Mint/Tinker deployments.

Python automatically imports `sitecustomize` on interpreter startup (if found on
sys.path). In Mint deployments, Ray actors inherit `PYTHONPATH` that includes the
tinker-server code root, so this file runs in API server, Megatron ranks, and
vLLM Ray workers.

Keep this file lightweight and defensive: it must not fail imports for
environments that do not have optional dependencies installed.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import threading
from typing import Any

_MINT_REQ_CTX = threading.local()


def _set_active_req_ids(req_ids: tuple[str, ...]) -> None:
    try:
        _MINT_REQ_CTX.req_ids = tuple(str(r) for r in req_ids if r is not None)
    except Exception:
        _MINT_REQ_CTX.req_ids = ()


def _clear_active_req_ids() -> None:
    try:
        if hasattr(_MINT_REQ_CTX, "req_ids"):
            delattr(_MINT_REQ_CTX, "req_ids")
    except Exception:
        pass


def _get_active_req_ids() -> tuple[str, ...]:
    try:
        req_ids = getattr(_MINT_REQ_CTX, "req_ids", ())
        if not req_ids:
            return ()
        return tuple(str(r) for r in req_ids)
    except Exception:
        return ()


def _maybe_set_vllm_host_ip() -> None:
    # vLLM uses VLLM_HOST_IP for cross-node rendezvous and also for validating
    # that each Ray node reports a unique IP. When unset, vLLM's get_ip()
    # probes the default route, which can be inconsistent across Ray worker
    # processes on the same node.
    #
    # Ray sets RAY_NODE_IP_ADDRESS for each worker process; use it to pin a
    # stable per-node IP without requiring cluster-wide env configuration.
    if os.environ.get("VLLM_HOST_IP"):
        return
    ray_node_ip = os.environ.get("RAY_NODE_IP_ADDRESS")
    if ray_node_ip:
        os.environ["VLLM_HOST_IP"] = ray_node_ip


def _patch_vllm_lora_pack_moe(module: Any) -> None:
    """Patch vLLM MoE LoRA packing to tolerate sparse expert weights.

    vLLM's `PackedLoRALayerWeights.pack_moe()` historically assumes every expert
    has LoRA weights present (len(loras) == num_experts * 3, and each entry is
    non-None). For large MoE models, exporting per-expert PEFT weights can be
    prohibitively large on disk and during loading.

    This patch allows exporting only a subset of experts (typically one
    representative per EP shard) and fills missing experts by repeating the most
    recent non-missing expert weights, then delegates to vLLM's original
    implementation to build contiguous stacked tensors.
    """
    try:
        if os.environ.get("MINT_VLLM_DISABLE_PACK_MOE_PATCH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
    except Exception:
        pass

    try:
        PackedLoRALayerWeights = getattr(module, "PackedLoRALayerWeights")
        LoRALayerWeights = getattr(module, "LoRALayerWeights")
    except Exception:
        return

    # Idempotent patching.
    try:
        already = getattr(PackedLoRALayerWeights.pack_moe, "__mint_sparse_ok__", False)
    except Exception:
        already = False
    if already:
        return

    try:
        orig_cm = PackedLoRALayerWeights.__dict__["pack_moe"]
        orig_fn = orig_cm.__func__
    except Exception:
        return

    def _is_none(x: object) -> bool:
        return x is None

    def pack_moe_sparse_ok(cls, loras, module_name: str):  # type: ignore[no-untyped-def]
        # Fast path: no missing entries.
        try:
            if loras and all(not _is_none(x) for x in loras):
                return orig_fn(cls, loras, module_name)
        except Exception:
            # Fall back to original behavior on unexpected types.
            return orig_fn(cls, loras, module_name)

        if not loras or (len(loras) % 3) != 0:
            return orig_fn(cls, loras, module_name)

        n_experts = len(loras) // 3
        if n_experts <= 0:
            return orig_fn(cls, loras, module_name)

        def triple_at(e: int) -> tuple[object, object, object]:
            i = e * 3
            return loras[i], loras[i + 1], loras[i + 2]

        def triple_complete(t: tuple[object, object, object]) -> bool:
            return all(not _is_none(x) for x in t)

        def triple_empty(t: tuple[object, object, object]) -> bool:
            return all(_is_none(x) for x in t)

        # Identify a base complete triple (for filling).
        base: tuple[object, object, object] | None = None
        for e in range(n_experts):
            t = triple_at(e)
            if triple_empty(t):
                continue
            if not triple_complete(t):
                # Partial expert triple: preserve strictness.
                return orig_fn(cls, loras, module_name)
            if base is None:
                base = t
                break

        if base is None:
            return orig_fn(cls, loras, module_name)

        def _zero_like_lora(obj: object) -> object | None:
            try:
                import torch

                if not isinstance(obj, LoRALayerWeights):
                    return None
                lora_a = getattr(obj, "lora_a", None)
                lora_b = getattr(obj, "lora_b", None)
                if lora_a is None or lora_b is None:
                    return None
                return LoRALayerWeights(
                    module_name=str(getattr(obj, "module_name", module_name)),
                    rank=int(getattr(obj, "rank", 0)),
                    lora_alpha=int(getattr(obj, "lora_alpha", 1)),
                    lora_a=torch.zeros_like(lora_a),
                    lora_b=torch.zeros_like(lora_b),
                    scaling=1.0,
                )
            except Exception:
                return None

        zero_triple = (
            _zero_like_lora(base[0]),
            _zero_like_lora(base[1]),
            _zero_like_lora(base[2]),
        )
        if any(z is None for z in zero_triple):
            return orig_fn(cls, loras, module_name)

        # Fill missing experts with zero LoRA deltas, then use vLLM's original
        # implementation (torch.stack), which expects all experts to be present
        # and produces contiguous tensors for kernels.
        filled = list(loras)
        for e in range(n_experts):
            i = e * 3
            t = (filled[i], filled[i + 1], filled[i + 2])
            if triple_complete(t):
                continue
            if triple_empty(t):
                filled[i] = zero_triple[0]
                filled[i + 1] = zero_triple[1]
                filled[i + 2] = zero_triple[2]
                continue
            return orig_fn(cls, loras, module_name)

        return orig_fn(cls, filled, module_name)

    pack_moe_sparse_ok.__mint_sparse_ok__ = True  # type: ignore[attr-defined]
    PackedLoRALayerWeights.pack_moe = classmethod(pack_moe_sparse_ok)


def _patch_vllm_punica_moe_lora_align(module: Any) -> None:
    """Replace vLLM's MoE LoRA align op with a torch implementation.

    vLLM's `_moe_C.moe_lora_align_block_size` (called from
    `PunicaWrapperGPU.moe_lora_align_block_size`) has produced uninitialized
    `sorted_ids` (negative/garbage) for K2 + per-expert MLP LoRA. Those indices
    reach fused MoE LoRA Triton kernels and can trigger CUDA illegal memory
    access.

    This patch computes `sorted_ids/expert_ids/num_tokens_post_pad` using torch
    ops (sort + bincount + scatter), avoiding the custom extension.
    """
    try:
        if os.environ.get("MINT_VLLM_DISABLE_PUNICA_PATCH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
    except Exception:
        pass

    try:
        PunicaWrapperGPU = getattr(module, "PunicaWrapperGPU")
    except Exception:
        return

    try:
        already = getattr(
            PunicaWrapperGPU.moe_lora_align_block_size,
            "__mint_safe_moe_lora_align__",
            False,
        )
    except Exception:
        already = False
    if already:
        return

    try:
        orig_fn = PunicaWrapperGPU.__dict__["moe_lora_align_block_size"]
    except Exception:
        return

    def moe_lora_align_block_size_safe(  # type: ignore[no-untyped-def]
        self,
        topk_ids,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled,
        expert_map=None,
        pad_sorted_ids: bool = False,
    ):
        import torch

        # Flattened tokens are addressed as token_idx * top_k + topk_slot.
        if topk_ids.dim() != 2:
            return orig_fn(
                self,
                topk_ids,
                num_tokens,
                block_size,
                num_experts,
                max_loras,
                adapter_enabled,
                expert_map,
                pad_sorted_ids,
            )
        top_k = int(topk_ids.shape[1])
        num_valid = int(topk_ids.numel())

        max_num_tokens_padded = num_valid + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = (
                (max_num_tokens_padded + block_size - 1) // block_size
            ) * block_size
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

        # Default-fill with safe sentinels: token_id=num_valid (masked out),
        # expert_id=-1 (kernel early-exit), num_tokens_post_pad=0.
        sorted_ids = torch.full(
            (max_loras * max_num_tokens_padded,),
            fill_value=num_valid,
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_ids = torch.full(
            (max_loras * max_num_m_blocks,),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_ids.device,
        )
        num_tokens_post_pad = torch.zeros(
            (max_loras,), dtype=torch.int32, device=topk_ids.device
        )

        token_lora_mapping, _, _, _, _, _ = self.token_mapping_meta.meta_args(num_tokens)

        active = torch.unique(token_lora_mapping)
        active = active[(active >= 0) & (active < max_loras)]
        if active.numel() == 0:
            return sorted_ids, expert_ids, num_tokens_post_pad

        slot = torch.arange(top_k, device=topk_ids.device, dtype=torch.int64)

        for i in range(int(active.numel())):
            lora_id = int(active[i].item())
            if lora_id < 0 or lora_id >= max_loras:
                continue
            if lora_id >= int(adapter_enabled.numel()):
                continue
            if int(adapter_enabled[lora_id].item()) == 0:
                continue

            tok = torch.nonzero(token_lora_mapping == lora_id, as_tuple=False).flatten()
            if tok.numel() == 0:
                continue

            # (T, top_k) -> (T*top_k)
            expert_flat = topk_ids.index_select(0, tok).reshape(-1).to(torch.int64)
            token_rep = tok.to(torch.int64).repeat_interleave(top_k)
            slot_rep = slot.repeat(int(tok.numel()))
            token_flat = token_rep * top_k + slot_rep

            perm = torch.argsort(expert_flat)
            expert_sorted = expert_flat.index_select(0, perm)
            token_sorted = token_flat.index_select(0, perm).to(torch.int32)

            counts = torch.bincount(expert_sorted, minlength=num_experts)
            padded_counts = ((counts + (block_size - 1)) // block_size) * block_size
            total_padded = int(padded_counts.sum().item())
            if total_padded <= 0:
                continue

            start_sorted = torch.cumsum(counts, dim=0) - counts
            start_out = torch.cumsum(padded_counts, dim=0) - padded_counts
            delta = start_out - start_sorted

            base = torch.arange(
                int(token_sorted.numel()), device=topk_ids.device, dtype=torch.int64
            )
            out_pos = base + delta.index_select(0, expert_sorted)

            row_off = lora_id * max_num_tokens_padded
            row = sorted_ids[row_off : row_off + max_num_tokens_padded]
            row.index_put_((out_pos,), token_sorted)

            blocks_per_expert = padded_counts // block_size
            block_expert_ids = torch.repeat_interleave(
                torch.arange(num_experts, device=topk_ids.device, dtype=torch.int32),
                blocks_per_expert.to(torch.int64),
            )
            n_blocks = int(block_expert_ids.numel())

            exp_off = lora_id * max_num_m_blocks
            exp_row = expert_ids[exp_off : exp_off + max_num_m_blocks]
            exp_row[:n_blocks].copy_(block_expert_ids, non_blocking=True)
            num_tokens_post_pad[lora_id] = total_padded

        if expert_map is not None:
            invalid = expert_ids < 0
            expert_ids_safe = torch.where(
                invalid, torch.zeros_like(expert_ids), expert_ids
            )
            expert_ids = expert_map[expert_ids_safe]
            expert_ids = torch.where(invalid, torch.full_like(expert_ids, -1), expert_ids)
            try:
                if not getattr(self, "_mint_logged_expert_map_stats", False):
                    setattr(self, "_mint_logged_expert_map_stats", True)
                    num_local_experts = int(num_experts)
                    exp_min = int(expert_ids.min().item())
                    exp_max = int(expert_ids.max().item())
                    exp_neg = int((expert_ids < 0).sum().item())
                    exp_oob_local = int((expert_ids >= num_local_experts).sum().item())
                    print(
                        "[punica][align_expert_map] local_num_experts=%s expert_ids.min=%s expert_ids.max=%s "
                        "expert_ids.neg_count=%s expert_ids.ge_local_count=%s expert_map.shape=%s"
                        % (
                            str(num_local_experts),
                            str(exp_min),
                            str(exp_max),
                            str(exp_neg),
                            str(exp_oob_local),
                            str(tuple(expert_map.shape)),
                        ),
                        flush=True,
                    )
            except Exception:
                pass

        return sorted_ids, expert_ids, num_tokens_post_pad

    moe_lora_align_block_size_safe.__mint_safe_moe_lora_align__ = True  # type: ignore[attr-defined]
    PunicaWrapperGPU.moe_lora_align_block_size = moe_lora_align_block_size_safe

    # Instrument fused MoE LoRA calls (once per process) to capture the kernel
    # config and tensor shapes for debugging CUDA illegal memory access.
    try:
        already_add = getattr(
            PunicaWrapperGPU.add_lora_fused_moe,
            "__mint_log_fused_moe_lora_cfg__",
            False,
        )
    except Exception:
        already_add = False
    if already_add:
        return

    try:
        orig_add = PunicaWrapperGPU.__dict__["add_lora_fused_moe"]
    except Exception:
        return

    def add_lora_fused_moe_logged(  # type: ignore[no-untyped-def]
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
    ):
        try:
            callsite = "w2" if bool(mul_routed_weight) else "w13"
            active_flag = f"_mint_logged_fused_moe_lora_cfg_active_{callsite}"
            no_lora_flag = f"_mint_logged_fused_moe_lora_cfg_no_lora_{callsite}"
            lora_ids_cpu = None
            has_lora = True
            try:
                _, _, _, _, lora_ids, _ = self.token_mapping_meta.meta_args(int(x.size(0)))
                lora_ids_cpu = lora_ids.detach().cpu().tolist()
                has_lora = any(int(v) >= 0 for v in lora_ids_cpu)
            except Exception:
                lora_ids_cpu = None
                has_lora = True

            if has_lora:
                if not getattr(self, active_flag, False):
                    setattr(self, active_flag, True)
                    a0 = lora_a_stacked[0] if lora_a_stacked else None
                    b0 = lora_b_stacked[0] if lora_b_stacked else None
                    try:
                        ntpp = num_tokens_post_padded.detach().cpu().tolist()
                    except Exception:
                        ntpp = None
                    try:
                        ae = adapter_enabled.detach().cpu().tolist()
                        ae = ae if isinstance(ae, list) and len(ae) <= 8 else None
                    except Exception:
                        ae = None
                    try:
                        num_tokens_cap = int(x.shape[0]) * int(top_k_num)
                        sorted_min = int(sorted_token_ids.min().item())
                        sorted_max = int(sorted_token_ids.max().item())
                        sorted_neg = int((sorted_token_ids < 0).sum().item())
                        sorted_eq_cap = int((sorted_token_ids == num_tokens_cap).sum().item())
                        sorted_gt_cap = int((sorted_token_ids > num_tokens_cap).sum().item())
                    except Exception:
                        num_tokens_cap = -1
                        sorted_min = None
                        sorted_max = None
                        sorted_neg = None
                        sorted_eq_cap = None
                        sorted_gt_cap = None
                    try:
                        local_num_experts = int(a0.shape[1]) if a0 is not None else -1
                        expert_min = int(expert_ids.min().item())
                        expert_max = int(expert_ids.max().item())
                        expert_neg = int((expert_ids < 0).sum().item())
                        expert_ge_local = (
                            int((expert_ids >= local_num_experts).sum().item())
                            if local_num_experts > 0
                            else None
                        )
                    except Exception:
                        local_num_experts = -1
                        expert_min = None
                        expert_max = None
                        expert_neg = None
                        expert_ge_local = None
                    print(
                        "[punica][fused_moe_lora_active] callsite=%s mul_routed_weight=%s num_slices=%s "
                        "fully_sharded=%s offset=%s max_lora_rank=%s top_k=%s "
                        "shrink_split_k=%s expand_split_k=%s "
                        "lora_ids=%s num_tokens_post_padded.shape=%s num_tokens_post_padded=%s adapter_enabled=%s "
                        "a0.shape=%s b0.shape=%s y.shape=%s x.shape=%s sorted.shape=%s expert_ids.shape=%s "
                        "num_tokens_cap=%s sorted.min=%s sorted.max=%s sorted.neg_count=%s sorted.eq_cap_count=%s sorted.gt_cap_count=%s "
                        "local_num_experts=%s expert.min=%s expert.max=%s expert.neg_count=%s expert.ge_local_count=%s"
                        % (
                            str(callsite),
                            str(bool(mul_routed_weight)),
                            str(int(len(lora_a_stacked))),
                            str(bool(fully_sharded)),
                            str(int(offset)),
                            str(int(max_lora_rank)),
                            str(int(top_k_num)),
                            str(int(shrink_config.get("SPLIT_K", -1))),
                            str(int(expand_config.get("SPLIT_K", -1))),
                            str(lora_ids_cpu),
                            str(tuple(num_tokens_post_padded.shape)),
                            str(ntpp),
                            str(ae),
                            str(tuple(a0.shape) if a0 is not None else None),
                            str(tuple(b0.shape) if b0 is not None else None),
                            str(tuple(y.shape)),
                            str(tuple(x.shape)),
                            str(tuple(sorted_token_ids.shape)),
                            str(tuple(expert_ids.shape)),
                            str(num_tokens_cap),
                            str(sorted_min),
                            str(sorted_max),
                            str(sorted_neg),
                            str(sorted_eq_cap),
                            str(sorted_gt_cap),
                            str(local_num_experts),
                            str(expert_min),
                            str(expert_max),
                            str(expert_neg),
                            str(expert_ge_local),
                        ),
                        flush=True,
                    )
            else:
                if not getattr(self, no_lora_flag, False):
                    setattr(self, no_lora_flag, True)
        except Exception:
            pass
        return orig_add(
            self,
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_lora_rank,
            top_k_num,
            shrink_config,
            expand_config,
            adapter_enabled,
            mul_routed_weight,
            fully_sharded,
            offset,
        )

    add_lora_fused_moe_logged.__mint_log_fused_moe_lora_cfg__ = True  # type: ignore[attr-defined]
    PunicaWrapperGPU.add_lora_fused_moe = add_lora_fused_moe_logged

    # Localize CUDA illegal memory access by forcing sync boundaries around
    # fused MoE LoRA phases (shrink -> TP gather/reduce -> expand). Enable only
    # on K2 vLLM workers (they set MINT_VLLM_ENABLE_PREFIX_CACHING in runtime_env).
    try:
        if os.environ.get("MINT_VLLM_ENABLE_PREFIX_CACHING") is None:
            return
        stage_sync = os.environ.get("MINT_VLLM_STAGE_SYNC", "1").strip().lower()
        if stage_sync in {"0", "false", "off", "no"}:
            return
    except Exception:
        return

    try:
        already = getattr(module.fused_moe_lora, "__mint_stage_sync__", False)
    except Exception:
        already = False
    if already:
        return

    try:
        import torch
        from vllm.distributed import (
            tensor_model_parallel_all_gather,
            tensor_model_parallel_all_reduce,
        )
        from vllm.lora.ops.triton_ops import fused_moe_lora_op as fml
    except Exception:
        return

    def fused_moe_lora_stage_sync(  # type: ignore[no-untyped-def]
        output,
        qcurr_hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_lora_rank: int,
        top_k_num: int,
        lora_ids,
        adapter_enabled,
        shrink_block_size_m: int,
        shrink_block_size_n: int,
        shrink_block_size_k: int,
        shrink_group_size_m: int,
        shrink_num_warps: int,
        shrink_num_stages: int,
        shrink_split_k: int,
        expand_block_size_m: int,
        expand_block_size_n: int,
        expand_block_size_k: int,
        expand_group_size_m: int,
        expand_num_warps: int,
        expand_num_stages: int,
        expand_split_k: int,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
    ):
        callsite = "w2" if bool(mul_routed_weight) else "w13"
        trace_nonfinite = os.environ.get("MINT_VLLM_TRACE_NONFINITE", "1") != "0"
        trace_stage_stats = (
            os.environ.get("MINT_VLLM_TRACE_STAGE_STATS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        trace_stage_stats_req_suffix = os.environ.get(
            "MINT_VLLM_TRACE_STAGE_STATS_REQ_SUFFIX", "_s0"
        ).strip()
        preflight_finite_gate = (
            os.environ.get("MINT_VLLM_PREFLIGHT_FINITE_GATE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        sanitize_w13_after_expand = (
            os.environ.get("MINT_VLLM_SANITIZE_W13_AFTER_EXPAND", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        def _stage_stats_trace(stage: str, tensor_name: str, tensor) -> None:
            if not trace_stage_stats:
                return
            try:
                seen = getattr(
                    fused_moe_lora_stage_sync, "__mint_stage_stats_seen__", None
                )
                if seen is None:
                    seen = set()
                    setattr(
                        fused_moe_lora_stage_sync,
                        "__mint_stage_stats_seen__",
                        seen,
                    )
                req_ids = _get_active_req_ids()
                req_preview = list(req_ids[:8])
                if trace_stage_stats_req_suffix and not any(
                    str(req).endswith(trace_stage_stats_req_suffix) for req in req_ids
                ):
                    return
                tensor_shape = tuple(getattr(tensor, "shape", ()))
                key = (callsite, stage, tensor_name, tensor_shape, req_ids)
                if key in seen:
                    return
                flat = tensor.reshape(-1)
                finite_mask = torch.isfinite(flat)
                nonfinite_count = int((~finite_mask).sum().item())
                nan_count = int(torch.isnan(flat).sum().item())
                if hasattr(torch, "isposinf"):
                    posinf_count = int(torch.isposinf(flat).sum().item())
                    neginf_count = int(torch.isneginf(flat).sum().item())
                else:
                    posinf_count = int((flat == float("inf")).sum().item())
                    neginf_count = int((flat == float("-inf")).sum().item())
                finite_min, finite_max, finite_count = _finite_minmax_info(tensor)
                print(
                    "[punica][fused_moe_lora_stage_stats] callsite=%s stage=%s tensor=%s "
                    "shape=%s dtype=%s nonfinite_count=%s nan_count=%s posinf_count=%s neginf_count=%s "
                    "finite_count=%s finite_min=%s finite_max=%s req_ids=%s"
                    % (
                        str(callsite),
                        stage,
                        tensor_name,
                        str(tensor_shape),
                        str(getattr(tensor, "dtype", None)),
                        str(nonfinite_count),
                        str(nan_count),
                        str(posinf_count),
                        str(neginf_count),
                        str(finite_count),
                        str(finite_min),
                        str(finite_max),
                        str(req_preview),
                    ),
                    flush=True,
                )
                seen.add(key)
            except Exception as stats_err:
                print(
                    "[punica][fused_moe_lora_stage_stats_error] callsite=%s stage=%s tensor=%s req_ids=%s err_type=%s err=%s"
                    % (
                        str(callsite),
                        stage,
                        tensor_name,
                        str(list(_get_active_req_ids()[:8])),
                        type(stats_err).__name__,
                        str(stats_err),
                    ),
                    flush=True,
                )

        def _routing_trace(stage: str) -> None:
            if not trace_stage_stats:
                return
            try:
                req_ids = _get_active_req_ids()
                req_preview = list(req_ids[:8])
                if trace_stage_stats_req_suffix and not any(
                    str(req).endswith(trace_stage_stats_req_suffix) for req in req_ids
                ):
                    return
                seen = getattr(
                    fused_moe_lora_stage_sync, "__mint_routing_stats_seen__", None
                )
                if seen is None:
                    seen = set()
                    setattr(
                        fused_moe_lora_stage_sync,
                        "__mint_routing_stats_seen__",
                        seen,
                    )
                key = (
                    callsite,
                    stage,
                    tuple(getattr(topk_weights, "shape", ())),
                    tuple(getattr(expert_ids, "shape", ())),
                    tuple(getattr(sorted_token_ids, "shape", ())),
                    req_ids,
                )
                if key in seen:
                    return

                topk_flat = topk_weights.reshape(-1).to(torch.float32)
                topk_finite_mask = torch.isfinite(topk_flat)
                topk_nonfinite_count = int((~topk_finite_mask).sum().item())
                if bool(topk_finite_mask.any()):
                    finite_topk = topk_flat[topk_finite_mask]
                    topk_finite_min = float(finite_topk.min().item())
                    topk_finite_max = float(finite_topk.max().item())
                else:
                    topk_finite_min = None
                    topk_finite_max = None

                expert_flat = expert_ids.reshape(-1).to(torch.int64)
                if expert_flat.numel() > 0:
                    expert_min = int(expert_flat.min().item())
                    expert_max = int(expert_flat.max().item())
                    expert_unique = torch.unique(expert_flat)
                    expert_unique_count = int(expert_unique.numel())
                    expert_unique_preview = [
                        int(x) for x in expert_unique[:16].tolist()
                    ]
                else:
                    expert_min = None
                    expert_max = None
                    expert_unique_count = 0
                    expert_unique_preview = []

                sorted_token_flat = sorted_token_ids.reshape(-1).to(torch.int64)
                if sorted_token_flat.numel() > 0:
                    sorted_token_min = int(sorted_token_flat.min().item())
                    sorted_token_max = int(sorted_token_flat.max().item())
                else:
                    sorted_token_min = None
                    sorted_token_max = None

                num_tokens_flat = num_tokens_post_padded.reshape(-1).to(torch.int64)
                num_tokens_vals = [int(x) for x in num_tokens_flat.tolist()[:16]]
                num_tokens_sum = int(num_tokens_flat.sum().item()) if num_tokens_flat.numel() else 0

                print(
                    "[punica][fused_moe_lora_routing] callsite=%s stage=%s req_ids=%s "
                    "topk_shape=%s topk_nonfinite_count=%s topk_finite_min=%s topk_finite_max=%s "
                    "expert_shape=%s expert_min=%s expert_max=%s expert_unique_count=%s expert_unique_preview=%s "
                    "sorted_token_shape=%s sorted_token_min=%s sorted_token_max=%s "
                    "num_tokens_shape=%s num_tokens_sum=%s num_tokens_values=%s"
                    % (
                        str(callsite),
                        stage,
                        str(req_preview),
                        str(tuple(topk_weights.shape)),
                        str(topk_nonfinite_count),
                        str(topk_finite_min),
                        str(topk_finite_max),
                        str(tuple(expert_ids.shape)),
                        str(expert_min),
                        str(expert_max),
                        str(expert_unique_count),
                        str(expert_unique_preview),
                        str(tuple(sorted_token_ids.shape)),
                        str(sorted_token_min),
                        str(sorted_token_max),
                        str(tuple(num_tokens_post_padded.shape)),
                        str(num_tokens_sum),
                        str(num_tokens_vals),
                    ),
                    flush=True,
                )
                seen.add(key)
            except Exception as route_err:
                print(
                    "[punica][fused_moe_lora_routing_error] callsite=%s stage=%s req_ids=%s err_type=%s err=%s"
                    % (
                        str(callsite),
                        stage,
                        str(list(_get_active_req_ids()[:8])),
                        type(route_err).__name__,
                        str(route_err),
                    ),
                    flush=True,
                )

        def _stage_nonfinite_trace(stage: str, tensor_name: str, tensor) -> None:
            if not trace_nonfinite:
                return
            try:
                seen = getattr(fused_moe_lora_stage_sync, "__mint_nonfinite_seen__", None)
                if seen is None:
                    seen = set()
                    setattr(fused_moe_lora_stage_sync, "__mint_nonfinite_seen__", seen)
                req_ids = _get_active_req_ids()
                req_preview = list(req_ids[:8])
                key = (callsite, stage, tensor_name, req_ids)
                if key in seen:
                    return

                flat = tensor.reshape(-1)
                finite_mask = torch.isfinite(flat)
                if bool(finite_mask.all()):
                    return

                nonfinite_mask = ~finite_mask
                bad_flat = torch.nonzero(nonfinite_mask, as_tuple=False).flatten()[:3]
                bad_idx = [int(i.item()) for i in bad_flat]
                bad_vals = [str(float(flat[i].item())) for i in bad_flat]
                nan_count = int(torch.isnan(flat).sum().item())
                if hasattr(torch, "isposinf"):
                    posinf_count = int(torch.isposinf(flat).sum().item())
                    neginf_count = int(torch.isneginf(flat).sum().item())
                else:
                    posinf_count = int((flat == float("inf")).sum().item())
                    neginf_count = int((flat == float("-inf")).sum().item())

                print(
                    "[punica][fused_moe_lora_nonfinite] callsite=%s mul_routed_weight=%s stage=%s tensor=%s "
                    "shape=%s dtype=%s nonfinite_count=%s nan_count=%s posinf_count=%s neginf_count=%s "
                    "bad_flat_idx=%s bad_vals=%s req_count=%s req_ids=%s"
                    % (
                        str(callsite),
                        str(bool(mul_routed_weight)),
                        stage,
                        tensor_name,
                        str(tuple(tensor.shape)),
                        str(getattr(tensor, "dtype", None)),
                        str(int(nonfinite_mask.sum().item())),
                        str(nan_count),
                        str(posinf_count),
                        str(neginf_count),
                        str(bad_idx),
                        str(bad_vals),
                        str(len(req_ids)),
                        str(req_preview),
                    ),
                    flush=True,
                )
                seen.add(key)
            except Exception as e:
                try:
                        print(
                        "[punica][fused_moe_lora_nonfinite_probe_error] callsite=%s stage=%s tensor=%s req_ids=%s err_type=%s err=%s"
                        % (
                            str(callsite),
                            stage,
                            tensor_name,
                            str(list(_get_active_req_ids()[:8])),
                            type(e).__name__,
                            str(e).replace("\r", "\\r").replace("\n", "\\n"),
                        ),
                        flush=True,
                    )
                except Exception:
                    pass

        def _finite_minmax_info(tensor):
            try:
                flat = tensor.reshape(-1)
                finite_mask = torch.isfinite(flat)
                finite_count = int(finite_mask.sum().item())
                if finite_count <= 0:
                    return None, None, 0
                finite_vals = flat[finite_mask].float()
                return (
                    float(finite_vals.min().item()),
                    float(finite_vals.max().item()),
                    finite_count,
                )
            except Exception:
                return None, None, 0

        def _stage_sync(stage: str) -> None:
            try:
                torch.cuda.synchronize()
            except Exception as e:
                try:
                    err_text = str(e).replace("\r", "\\r").replace("\n", "\\n")
                    print(
                        "[punica][fused_moe_lora_sync_error] callsite=%s mul_routed_weight=%s stage=%s err_type=%s err=%s "
                        "fully_sharded=%s offset=%s num_slices=%s M=%s top_k_num=%s EM=%s K=%s N=%s "
                        "max_lora_rank=%s w1_output_dim_size=%s output_shape=%s hidden_shape=%s req_ids=%s"
                        % (
                            str(callsite),
                            str(bool(mul_routed_weight)),
                            stage,
                            type(e).__name__,
                            err_text,
                            str(bool(fully_sharded)),
                            str(int(offset)),
                            str(int(num_slices)),
                            str(int(M)),
                            str(int(top_k_num)),
                            str(int(EM)),
                            str(int(K)),
                            str(int(N)),
                            str(int(max_lora_rank)),
                            str(int(w1_output_dim_size)),
                            str(tuple(output.shape)),
                            str(tuple(qcurr_hidden_states.shape)),
                            str(list(_get_active_req_ids()[:8])),
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
                raise

        assert len(lora_a_stacked) == len(lora_b_stacked) > 0
        assert (
            sorted_token_ids.dim()
            == expert_ids.dim()
            == topk_weights.dim()
            == qcurr_hidden_states.dim()
            == 2
        )
        assert (
            sorted_token_ids.shape[0]
            == expert_ids.shape[0]
            == num_tokens_post_padded.shape[0]
        )
        assert output.shape[0] == topk_weights.shape[0]
        assert top_k_num == topk_weights.shape[1]

        device = qcurr_hidden_states.device
        num_slices = len(lora_a_stacked)
        w1_lora_b_stacked = lora_b_stacked[0]
        num_experts = lora_a_stacked[0].shape[1]
        N = max_lora_rank
        M = topk_weights.shape[0]
        EM = sorted_token_ids.shape[1]
        K = qcurr_hidden_states.shape[1]
        num_tokens = M * top_k_num
        w1_output_dim_size = w1_lora_b_stacked.shape[2]

        _stage_nonfinite_trace("pre_shrink", "qcurr_hidden_states", qcurr_hidden_states)
        _stage_stats_trace("pre_shrink", "qcurr_hidden_states", qcurr_hidden_states)
        _stage_stats_trace("pre_shrink", "topk_weights", topk_weights)
        _routing_trace("pre_shrink")
        _stage_nonfinite_trace("pre_shrink", "topk_weights", topk_weights)
        _stage_nonfinite_trace("pre_shrink", "lora_a_stacked_0", lora_a_stacked[0])
        _stage_nonfinite_trace("pre_shrink", "lora_b_stacked_0", lora_b_stacked[0])
        if len(lora_a_stacked) > 1:
            _stage_nonfinite_trace("pre_shrink", "lora_a_stacked_1", lora_a_stacked[1])
        if len(lora_b_stacked) > 1:
            _stage_nonfinite_trace("pre_shrink", "lora_b_stacked_1", lora_b_stacked[1])
        if preflight_finite_gate:
            gate_tensors = [
                ("qcurr_hidden_states", qcurr_hidden_states),
                ("topk_weights", topk_weights),
                ("lora_a_stacked_0", lora_a_stacked[0]),
                ("lora_b_stacked_0", lora_b_stacked[0]),
            ]
            if len(lora_a_stacked) > 1:
                gate_tensors.append(("lora_a_stacked_1", lora_a_stacked[1]))
            if len(lora_b_stacked) > 1:
                gate_tensors.append(("lora_b_stacked_1", lora_b_stacked[1]))
            for gate_name, gate_tensor in gate_tensors:
                flat = gate_tensor.reshape(-1)
                finite_mask = torch.isfinite(flat)
                if bool(finite_mask.all()):
                    continue
                nonfinite_mask = ~finite_mask
                nan_count = int(torch.isnan(flat).sum().item())
                if hasattr(torch, "isposinf"):
                    posinf_count = int(torch.isposinf(flat).sum().item())
                    neginf_count = int(torch.isneginf(flat).sum().item())
                else:
                    posinf_count = int((flat == float("inf")).sum().item())
                    neginf_count = int((flat == float("-inf")).sum().item())
                finite_min, finite_max, finite_count = _finite_minmax_info(gate_tensor)
                req_preview = list(_get_active_req_ids()[:8])
                try:
                    topk_flat = topk_weights.reshape(-1).to(torch.float32)
                    topk_finite_mask = torch.isfinite(topk_flat)
                    topk_nonfinite_count = int((~topk_finite_mask).sum().item())
                    if bool(topk_finite_mask.any()):
                        finite_topk = topk_flat[topk_finite_mask]
                        topk_finite_min = float(finite_topk.min().item())
                        topk_finite_max = float(finite_topk.max().item())
                    else:
                        topk_finite_min = None
                        topk_finite_max = None

                    expert_flat = expert_ids.reshape(-1).to(torch.int64)
                    if expert_flat.numel() > 0:
                        expert_min = int(expert_flat.min().item())
                        expert_max = int(expert_flat.max().item())
                        expert_unique = torch.unique(expert_flat)
                        expert_unique_count = int(expert_unique.numel())
                        expert_unique_preview = [int(x) for x in expert_unique[:16].tolist()]
                    else:
                        expert_min = None
                        expert_max = None
                        expert_unique_count = 0
                        expert_unique_preview = []

                    sorted_token_flat = sorted_token_ids.reshape(-1).to(torch.int64)
                    if sorted_token_flat.numel() > 0:
                        sorted_token_min = int(sorted_token_flat.min().item())
                        sorted_token_max = int(sorted_token_flat.max().item())
                    else:
                        sorted_token_min = None
                        sorted_token_max = None

                    num_tokens_flat = num_tokens_post_padded.reshape(-1).to(torch.int64)
                    num_tokens_sum = int(num_tokens_flat.sum().item()) if num_tokens_flat.numel() else 0
                    num_tokens_values = [int(x) for x in num_tokens_flat.tolist()[:16]]
                except Exception:
                    topk_nonfinite_count = None
                    topk_finite_min = None
                    topk_finite_max = None
                    expert_min = None
                    expert_max = None
                    expert_unique_count = None
                    expert_unique_preview = None
                    sorted_token_min = None
                    sorted_token_max = None
                    num_tokens_sum = None
                    num_tokens_values = None
                print(
                    "[punica][fused_moe_lora_preflight_nonfinite] callsite=%s tensor=%s "
                    "shape=%s dtype=%s nonfinite_count=%s nan_count=%s posinf_count=%s neginf_count=%s "
                    "finite_count=%s finite_min=%s finite_max=%s req_ids=%s "
                    "topk_shape=%s topk_nonfinite_count=%s topk_finite_min=%s topk_finite_max=%s "
                    "expert_shape=%s expert_min=%s expert_max=%s expert_unique_count=%s expert_unique_preview=%s "
                    "sorted_token_shape=%s sorted_token_min=%s sorted_token_max=%s "
                    "num_tokens_shape=%s num_tokens_sum=%s num_tokens_values=%s"
                    % (
                        str(callsite),
                        gate_name,
                        str(tuple(gate_tensor.shape)),
                        str(getattr(gate_tensor, "dtype", None)),
                        str(int(nonfinite_mask.sum().item())),
                        str(nan_count),
                        str(posinf_count),
                        str(neginf_count),
                        str(finite_count),
                        str(finite_min),
                        str(finite_max),
                        str(req_preview),
                        str(tuple(topk_weights.shape)),
                        str(topk_nonfinite_count),
                        str(topk_finite_min),
                        str(topk_finite_max),
                        str(tuple(expert_ids.shape)),
                        str(expert_min),
                        str(expert_max),
                        str(expert_unique_count),
                        str(expert_unique_preview),
                        str(tuple(sorted_token_ids.shape)),
                        str(sorted_token_min),
                        str(sorted_token_max),
                        str(tuple(num_tokens_post_padded.shape)),
                        str(num_tokens_sum),
                        str(num_tokens_values),
                    ),
                    flush=True,
                )
                raise RuntimeError(
                    "fused_moe_lora_preflight_nonfinite callsite=%s tensor=%s req_ids=%s"
                    % (str(callsite), gate_name, str(req_preview))
                )

        a_intermediate_cache1 = torch.zeros(
            (num_slices, M, top_k_num, max_lora_rank),
            dtype=output.dtype,
            device=device,
        )

        fml._fused_moe_lora_shrink(
            a_intermediate_cache1,
            qcurr_hidden_states,
            lora_a_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_experts,
            num_slices,
            shrink_block_size_m,
            shrink_block_size_n,
            shrink_block_size_k,
            shrink_group_size_m,
            shrink_num_warps,
            shrink_num_stages,
            shrink_split_k,
            mul_routed_weight,
        )
        _stage_sync("after_shrink")
        _stage_nonfinite_trace("after_shrink", "a_intermediate_cache1", a_intermediate_cache1)
        _stage_stats_trace("after_shrink", "a_intermediate_cache1", a_intermediate_cache1)

        if fully_sharded:
            if max_lora_rank == w1_lora_b_stacked.shape[-1]:
                a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                    a_intermediate_cache1
                )
            else:
                a_intermediate_cache1 = tensor_model_parallel_all_gather(
                    a_intermediate_cache1
                )
                max_lora_rank = a_intermediate_cache1.shape[-1]
            _stage_sync("after_tp_comm")
            _stage_nonfinite_trace("after_tp_comm", "a_intermediate_cache1", a_intermediate_cache1)
            _stage_stats_trace("after_tp_comm", "a_intermediate_cache1", a_intermediate_cache1)

        fml._fused_moe_lora_expand(
            output,
            a_intermediate_cache1,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_experts,
            num_slices,
            max_lora_rank,
            w1_output_dim_size,
            expand_block_size_m,
            expand_block_size_n,
            expand_block_size_k,
            expand_group_size_m,
            expand_num_warps,
            expand_num_stages,
            expand_split_k,
            mul_routed_weight,
            offset,
        )
        _stage_sync("after_expand")
        if sanitize_w13_after_expand and callsite == "w13":
            try:
                flat = output.reshape(-1)
                finite_mask = torch.isfinite(flat)
                bad_count = int((~finite_mask).sum().item())
                if bad_count > 0:
                    nan_count = int(torch.isnan(flat).sum().item())
                    if hasattr(torch, "isposinf"):
                        posinf_count = int(torch.isposinf(flat).sum().item())
                        neginf_count = int(torch.isneginf(flat).sum().item())
                    else:
                        posinf_count = int((flat == float("inf")).sum().item())
                        neginf_count = int((flat == float("-inf")).sum().item())
                    req_preview = list(_get_active_req_ids()[:8])
                    print(
                        "[punica][fused_moe_lora_sanitize] callsite=%s stage=after_expand tensor=output "
                        "shape=%s dtype=%s bad_count=%s nan_count=%s posinf_count=%s neginf_count=%s req_ids=%s"
                        % (
                            str(callsite),
                            str(tuple(output.shape)),
                            str(getattr(output, "dtype", None)),
                            str(bad_count),
                            str(nan_count),
                            str(posinf_count),
                            str(neginf_count),
                            str(req_preview),
                        ),
                        flush=True,
                    )
                    output.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as sanitize_err:
                print(
                    "[punica][fused_moe_lora_sanitize_error] callsite=%s stage=after_expand err_type=%s err=%s req_ids=%s"
                    % (
                        str(callsite),
                        type(sanitize_err).__name__,
                        str(sanitize_err).replace("\r", "\\r").replace("\n", "\\n"),
                        str(list(_get_active_req_ids()[:8])),
                    ),
                    flush=True,
                )
        _stage_nonfinite_trace("after_expand", "output", output)
        _stage_stats_trace("after_expand", "output", output)

    fused_moe_lora_stage_sync.__mint_stage_sync__ = True  # type: ignore[attr-defined]
    module.fused_moe_lora = fused_moe_lora_stage_sync


def _patch_vllm_gpu_model_runner_request_context(module: Any) -> None:
    try:
        GPUModelRunner = getattr(module, "GPUModelRunner")
    except Exception:
        return

    try:
        already = getattr(GPUModelRunner.execute_model, "__mint_req_ctx__", False)
    except Exception:
        already = False
    if already:
        return

    try:
        orig_execute = GPUModelRunner.__dict__["execute_model"]
    except Exception:
        return

    def execute_model_with_req_ctx(  # type: ignore[no-untyped-def]
        self,
        scheduler_output,
        intermediate_tensors=None,
    ):
        req_ids: tuple[str, ...] = ()
        try:
            req_map = getattr(scheduler_output, "num_scheduled_tokens", None)
            if req_map is not None:
                req_ids = tuple(str(r) for r in req_map.keys())
            if not req_ids:
                req_data = getattr(scheduler_output, "scheduled_cached_reqs", None)
                req_list = getattr(req_data, "req_ids", None) if req_data is not None else None
                if req_list:
                    req_ids = tuple(str(r) for r in req_list)
        except Exception:
            req_ids = ()

        _set_active_req_ids(req_ids)
        try:
            return orig_execute(self, scheduler_output, intermediate_tensors)
        finally:
            _clear_active_req_ids()

    execute_model_with_req_ctx.__mint_req_ctx__ = True  # type: ignore[attr-defined]
    GPUModelRunner.execute_model = execute_model_with_req_ctx


class _PatchFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that patches vLLM after importing specific modules."""

    _TARGETS = {
        "vllm.lora.lora_weights": _patch_vllm_lora_pack_moe,
        "vllm.lora.punica_wrapper.punica_gpu": _patch_vllm_punica_moe_lora_align,
        "vllm.v1.worker.gpu_model_runner": _patch_vllm_gpu_model_runner_request_context,
    }

    def find_spec(self, fullname: str, path, target=None):  # type: ignore[no-untyped-def]
        patcher = self._TARGETS.get(fullname)
        if patcher is None:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec

        orig_loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):  # type: ignore[no-untyped-def]
                if hasattr(orig_loader, "create_module"):
                    return orig_loader.create_module(spec)  # type: ignore[misc]
                return None

            def exec_module(self, module):  # type: ignore[no-untyped-def]
                orig_loader.exec_module(module)  # type: ignore[misc]
                try:
                    patcher(module)
                except Exception:
                    # Never fail interpreter startup because of a patch.
                    pass

        spec.loader = _Loader()
        return spec


def _install_import_patches() -> None:
    # Patch immediately if already imported (rare but safe).
    mod = sys.modules.get("vllm.lora.lora_weights")
    if mod is not None:
        try:
            _patch_vllm_lora_pack_moe(mod)
        except Exception:
            pass
    mod = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if mod is not None:
        try:
            _patch_vllm_gpu_model_runner_request_context(mod)
        except Exception:
            pass

    # Install finder once.
    for finder in sys.meta_path:
        if isinstance(finder, _PatchFinder):
            return
    sys.meta_path.insert(0, _PatchFinder())


_install_import_patches()

_maybe_set_vllm_host_ip()
