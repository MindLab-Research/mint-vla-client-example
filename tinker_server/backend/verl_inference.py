"""Wrapper for verl's vLLMHttpServer for inference.

Uses verl's Ray-based vLLM infrastructure for scalable inference.
"""

from __future__ import annotations

import os

# Required for vLLM multiprocessing in Ray actors (prevents fork-related hangs)
# Must be set before vLLM is imported anywhere in the process
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray

from tinker_server.backend.model_registry import get_model_config
from tinker_server.config import PFS_PYTHONPATH, RAY_NAMESPACE
from tinker_server.config import config as server_config
from tinker_server.logging_context import (
    get_current_traceparent,
    init_actor_observability,
    restore_trace_id_from_traceparent,
    traced_async_from_traceparent,
)
from tinker_server.ray_utils import init_ray

from . import ray_kill

if TYPE_CHECKING:
    import torch
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = logging.getLogger(__name__)


def _mint_present_expert_ids_from_keys(keys: list[str]) -> set[int]:
    import re

    expert_pat = re.compile(r"\.mlp\.experts\.(\d+)\.")
    out: set[int] = set()
    for k in keys:
        m = expert_pat.search(k)
        if m is not None:
            out.add(int(m.group(1)))
    return out


def _mint_present_expert_ids_from_adapter_dir(adapter_dir: str) -> set[int]:
    from safetensors import safe_open

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        return _mint_present_expert_ids_from_keys(list(f.keys()))


def _mint_expected_num_experts_from_base_model(base_model_name_or_path: object) -> int | None:
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path:
        return None
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
        )
    except Exception:
        return None
    value = getattr(hf_config, "num_experts", None) or getattr(hf_config, "n_routed_experts", None)
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _mint_expected_num_experts_from_adapter_dir(adapter_dir: str) -> int | None:
    import json

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    return _mint_expected_num_experts_from_base_model(cfg.get("base_model_name_or_path"))


def _mint_sparse_expert_ids_need_patch(
    present_expert_ids: set[int],
    *,
    expected_num_experts: int | None,
) -> bool:
    if not present_expert_ids:
        return False
    if expected_num_experts is None or expected_num_experts <= 0:
        # Fail-open only for the legacy shared-expert export that we can identify
        # without model metadata. Avoid patching full per-expert adapters
        # unnecessarily in hardened environments.
        return present_expert_ids == {0}
    expected = set(range(expected_num_experts))
    return present_expert_ids != expected


# vLLM MoE LoRA support: tolerate shared-expert sparse adapters.
#
# vLLM's PackedLoRALayerWeights.pack_moe currently asserts that every expert has
# (w1,w2,w3) LoRA weights present. Our adapter export can be sparse across
# experts when using "shared expert" export (expert 0 only): the exported expert
# weights are intended to be broadcast to all experts at load time.
#
# We patch pack_moe inside vLLM TP worker processes via EngineCore.collective_rpc.
def _mint_vllm_patch_pack_moe_sparse_ok(_worker: Any) -> str:
    import inspect
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
        return "ok:already"

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
        if loras and all(lora_item is not None for lora_item in loras):
            return orig_fn(cls, loras, module_name, is_non_gated_moe=is_non_gated_moe)

        if not loras or (len(loras) % 3) != 0:
            raise RuntimeError(
                f"Unexpected MoE LoRA pack_moe inputs for module={module_name!r}: len(loras)={len(loras)}"
            )

        n_experts = len(loras) // 3

        base_any = next((lora_item for lora_item in loras if lora_item is not None), None)
        if base_any is None:
            raise RuntimeError(
                f"MoE LoRA pack_moe got all-None loras for module={module_name!r}"
            )
        rank = int(getattr(base_any, "rank"))
        lora_alpha = int(getattr(base_any, "lora_alpha"))

        base_w1 = next(
            (loras[i * 3] for i in range(n_experts) if loras[i * 3] is not None), None
        )
        base_w2 = next(
            (loras[i * 3 + 1] for i in range(n_experts) if loras[i * 3 + 1] is not None),
            None,
        )
        base_w3 = next(
            (loras[i * 3 + 2] for i in range(n_experts) if loras[i * 3 + 2] is not None),
            None,
        )
        if base_w1 is None or base_w2 is None or base_w3 is None:
            raise RuntimeError(
                f"MoE LoRA pack_moe missing base weight(s) for module={module_name!r}"
            )

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
            # Shared-expert export (expert 0 only): expand across experts without
            # materializing full per-expert tensors.
            #
            # vLLM later calls optimize() (in-place scaling merge) on packed weights.
            # expand(...) returns a view with internal overlap; avoid in-place ops by
            # pre-applying scaling out-of-place and setting scaling=1.
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
            import torch

            present_eids: set[int] = set()
            for eid in range(n_experts):
                if (
                    loras[eid * 3] is not None
                    or loras[eid * 3 + 1] is not None
                    or loras[eid * 3 + 2] is not None
                ):
                    present_eids.add(eid)

            # Sparse export mode: one representative expert per EP shard (linear placement).
            #
            # Export format expectation:
            # - loras has length n_experts * 3
            # - only representative experts are present (non-None); all other experts are None
            # - representatives are the first expert of each linear EP shard:
            #     rep_eid = start(ep_rank) where:
            #         base = n_experts // ep_size
            #         rem  = n_experts % ep_size
            #         start(r) = r * base + min(r, rem)
            #
            # We materialize full per-expert tensors directly on the target device (prefer GPU)
            # to avoid host-CPU OOM when expanding a large adapter.
            ep_size = len(present_eids)
            if ep_size > 1:
                base = n_experts // ep_size
                rem = n_experts % ep_size
                expected_reps = {
                    (r * base + min(r, rem))
                    for r in range(ep_size)
                    if (base + (1 if r < rem else 0)) > 0
                }
                if present_eids == expected_reps:
                    # Device selection: materialize on same device as base weights if possible.
                    # If base weights are on CPU, move representatives to current CUDA device.
                    device = getattr(base_w1.lora_a, "device", torch.device("cpu"))
                    if device.type == "cpu":
                        device = torch.device("cuda")

                    def _to_dev(t: torch.Tensor) -> torch.Tensor:
                        return t if t.device == device else t.to(device)

                    # Pre-apply scaling and set packed_scaling=1 to avoid vLLM in-place optimize
                    # touching overlapped views. We will materialize distinct per-expert tensors.
                    w1_lora_a_out = torch.empty(
                        (n_experts,) + tuple(base_w1.lora_a.shape),
                        device=device,
                        dtype=base_w1.lora_a.dtype,
                    )
                    w2_lora_a_out = torch.empty(
                        (n_experts,) + tuple(base_w2.lora_a.shape),
                        device=device,
                        dtype=base_w2.lora_a.dtype,
                    )
                    w3_lora_a_out = torch.empty(
                        (n_experts,) + tuple(base_w3.lora_a.shape),
                        device=device,
                        dtype=base_w3.lora_a.dtype,
                    )

                    w1_lora_b_out = torch.empty(
                        (n_experts,) + tuple(base_w1.lora_b.shape),
                        device=device,
                        dtype=base_w1.lora_b.dtype,
                    )
                    w2_lora_b_out = torch.empty(
                        (n_experts,) + tuple(base_w2.lora_b.shape),
                        device=device,
                        dtype=base_w2.lora_b.dtype,
                    )
                    w3_lora_b_out = torch.empty(
                        (n_experts,) + tuple(base_w3.lora_b.shape),
                        device=device,
                        dtype=base_w3.lora_b.dtype,
                    )

                    for r in range(ep_size):
                        start = r * base + min(r, rem)
                        local = base + (1 if r < rem else 0)
                        end = min(n_experts, start + local)
                        if start >= end:
                            continue

                        rep_eid = start
                        rep_w1 = loras[rep_eid * 3]
                        rep_w2 = loras[rep_eid * 3 + 1]
                        rep_w3 = loras[rep_eid * 3 + 2]
                        if rep_w1 is None or rep_w2 is None or rep_w3 is None:
                            raise RuntimeError(
                                "Sparse EP-shard MoE LoRA missing representative expert tensors: "
                                f"module={module_name!r} rep_eid={rep_eid} present_expert_ids={sorted(present_eids)}"
                            )

                        rep_w1_a = _to_dev(rep_w1.lora_a)
                        rep_w2_a = _to_dev(rep_w2.lora_a)
                        rep_w3_a = _to_dev(rep_w3.lora_a)

                        rep_w1_b = _to_dev(rep_w1.lora_b) * float(
                            getattr(rep_w1, "scaling", lora_alpha / rank)
                        )
                        rep_w2_b = _to_dev(rep_w2.lora_b) * float(
                            getattr(rep_w2, "scaling", lora_alpha / rank)
                        )
                        rep_w3_b = _to_dev(rep_w3.lora_b) * float(
                            getattr(rep_w3, "scaling", lora_alpha / rank)
                        )

                        sl = slice(start, end)
                        w1_lora_a_out[sl].copy_(rep_w1_a.unsqueeze(0).expand((end - start,) + rep_w1_a.shape))
                        w2_lora_a_out[sl].copy_(rep_w2_a.unsqueeze(0).expand((end - start,) + rep_w2_a.shape))
                        w3_lora_a_out[sl].copy_(rep_w3_a.unsqueeze(0).expand((end - start,) + rep_w3_a.shape))

                        w1_lora_b_out[sl].copy_(rep_w1_b.unsqueeze(0).expand((end - start,) + rep_w1_b.shape))
                        w2_lora_b_out[sl].copy_(rep_w2_b.unsqueeze(0).expand((end - start,) + rep_w2_b.shape))
                        w3_lora_b_out[sl].copy_(rep_w3_b.unsqueeze(0).expand((end - start,) + rep_w3_b.shape))

                    w1_lora_a, w2_lora_a, w3_lora_a = w1_lora_a_out, w2_lora_a_out, w3_lora_a_out
                    w1_lora_b, w2_lora_b, w3_lora_b = w1_lora_b_out, w2_lora_b_out, w3_lora_b_out
                    packed_scaling = [1.0, 1.0, 1.0]
                else:
                    raise RuntimeError(
                        "Unsupported sparse MoE LoRA adapter sparsity pattern for vLLM pack_moe. "
                        "Supported patterns: (1) all experts present, (2) expert-0-only shared-expert export, "
                        "(3) one-representative-per-EP-shard (linear placement). "
                        f"module={module_name!r} n_experts={n_experts} ep_size={ep_size} "
                        f"present_expert_ids={sorted(present_eids)} expected_reps={sorted(expected_reps)}"
                    )
            else:
                raise RuntimeError(
                    "Unsupported sparse MoE LoRA adapter sparsity pattern for vLLM pack_moe. "
                    "Supported patterns: (1) all experts present, (2) expert-0-only shared-expert export. "
                    f"module={module_name!r} n_experts={n_experts} present_expert_ids={sorted(present_eids)}"
                )

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
    return "ok:patched"


# Do not apply TensorLoRARequest hijacks at module import time.
# The API server can be CPU-only while Ray actors run on GPU nodes with the full
# runtime, and this codepath already loads LoRAs from on-node adapter paths.
# The older hijack path also breaks newer vLLM engine-core startup.


# Extended vLLMHttpServer with add_lora support
# Defined lazily so actor-local runtime setup happens inside the Ray process.
def _create_extended_server_class(
    max_loras: int = 1,
    max_cpu_loras: int = 0,
):
    """Create extended vLLMHttpServer class with add_lora method.

    Args:
        max_loras: Maximum LoRAs in a single batch (default: 1).
                   Set > 1 for multi-LoRA concurrent inference.
        max_cpu_loras: Maximum LoRAs in CPU cache for swap (default: 0).

    Returns:
        Ray actor class (result of ray.remote() applied to the extended class).
    """
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
    def _unwrap_ray_actor_class(cls):
        """Return the underlying Python class if `cls` is a Ray ActorClass."""
        try:
            md = getattr(cls, "__ray_metadata__", None)
            modified = getattr(md, "modified_class", None)
            if isinstance(modified, type):
                return modified
        except Exception:
            pass
        return cls

    _BaseVLLMHttpServer = _unwrap_ray_actor_class(vLLMHttpServer)

    # Capture in closure
    _max_loras = max_loras
    _max_cpu_loras = max_cpu_loras

    # Define the class WITHOUT @ray.remote decorator
    class ExtendedVLLMHttpServer(_BaseVLLMHttpServer):
        """Extended vLLMHttpServer with hot LoRA loading support."""

        # Class-level config for multi-LoRA (captured from factory)
        MULTI_LORA_MAX_LORAS = _max_loras
        MULTI_LORA_MAX_CPU_LORAS = _max_cpu_loras

        def __init__(self, *args, **kwargs):
            """Initialize with VLLMHijack applied first."""
            init_actor_observability()
            # Set PYTHONPATH in OS environment so vLLM's TP workers inherit it
            # Ray's runtime_env only sets it for this process, not multiprocessing children
            import os
            import inspect
            import sys
            # Allow EngineCoreClient to send function objects for collective_rpc.
            #
            # This is used only for internal worker patching (e.g. MoE LoRA pack_moe
            # sparse/shared-expert support) and is not exposed as an HTTP surface.
            #
            # Security note:
            # - vLLM names this flag "insecure" because enabling it relaxes
            #   serialization restrictions for RPC payloads.
            # - We keep it enabled by default for backwards compatibility, but allow
            #   explicitly disabling it via MINT_VLLM_ALLOW_INSECURE_SERIALIZATION=0.
            allow_insecure = os.environ.get("MINT_VLLM_ALLOW_INSECURE_SERIALIZATION", "1").strip().lower()
            if allow_insecure not in {"0", "false", "no", "off"}:
                os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
            rollout_mode = kwargs.get("rollout_mode")
            if isinstance(rollout_mode, str):
                from verl.workers.rollout.replica import RolloutMode

                kwargs["rollout_mode"] = RolloutMode(rollout_mode)
            pfs_pythonpath = PFS_PYTHONPATH
            os.environ["PYTHONPATH"] = pfs_pythonpath + ":" + os.environ.get("PYTHONPATH", "")
            for p in reversed(pfs_pythonpath.split(":")):
                if p and p not in sys.path:
                    sys.path.insert(0, p)
            print(f"[ExtendedVLLMHttpServer] Set PYTHONPATH, vLLM path: {sys.path[0]}", flush=True)

            # Always initialize through current MRO parent. Calling __init__ on a
            # separately re-imported class can break class identity and trigger:
            # "super(type, obj): obj must be an instance or subtype of type".
            _parent_cls = type(self).__mro__[1]
            parent_file = inspect.getsourcefile(_parent_cls)
            print(
                f"[ExtendedVLLMHttpServer] parent_cls={_parent_cls.__module__}.{_parent_cls.__qualname__} file={parent_file}",
                flush=True,
            )
            try:
                sig = inspect.signature(_parent_cls.__init__)
            except (TypeError, ValueError):
                sig = None
            print(f"[ExtendedVLLMHttpServer] vLLMHttpServer.__init__ sig={sig}", flush=True)
            print(f"[ExtendedVLLMHttpServer] Calling with args={args}, kwargs keys={list(kwargs.keys())}", flush=True)
            if 'cuda_visible_devices' in kwargs:
                print(f"[ExtendedVLLMHttpServer] cuda_visible_devices={kwargs['cuda_visible_devices']}", flush=True)
            else:
                print("[ExtendedVLLMHttpServer] WARNING: cuda_visible_devices NOT in kwargs!", flush=True)
            call_kwargs = dict(kwargs)
            rollout_cfg = call_kwargs.get("config")
            if rollout_cfg is not None:
                from dataclasses import asdict, is_dataclass
                from verl.utils.config import omega_conf_to_dataclass
                from verl.workers.config import RolloutConfig as VerlRolloutConfig

                if isinstance(rollout_cfg, dict):
                    rollout_cfg = omega_conf_to_dataclass(
                        rollout_cfg, dataclass_type=VerlRolloutConfig
                    )
                elif is_dataclass(rollout_cfg):
                    rollout_cfg = omega_conf_to_dataclass(
                        asdict(rollout_cfg), dataclass_type=VerlRolloutConfig
                    )
                elif not isinstance(rollout_cfg, VerlRolloutConfig):
                    rollout_cfg = omega_conf_to_dataclass(
                        dict(rollout_cfg), dataclass_type=VerlRolloutConfig
                    )
                print(
                    f"[ExtendedVLLMHttpServer] rollout config normalized type={type(rollout_cfg).__name__} _target_={getattr(rollout_cfg, '_target_', None)!r}",
                    flush=True,
                )
                call_kwargs["config"] = rollout_cfg

            if sig is not None:
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if not accepts_var_kw:
                    accepted = {
                        name for name, p in sig.parameters.items()
                        if name != "self"
                        and p.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    }
                    dropped = [k for k in list(call_kwargs.keys()) if k not in accepted]
                    for k in dropped:
                        call_kwargs.pop(k, None)
                    if dropped:
                        logger.warning(
                            "[ExtendedVLLMHttpServer] dropping unsupported kwargs for base __init__: %s",
                            dropped,
                        )
            super(ExtendedVLLMHttpServer, self).__init__(*args, **call_kwargs)
            # Track local paths for multi-LoRA (needed for GPU/CPU swap)
            self._lora_paths: dict[int, str] = {}
            self._timing = os.environ.get("MINT_VLLM_REQUEST_TIMING", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            self._enable_rollout_routing_replay = bool(
                getattr(self.config, "enable_rollout_routing_replay", False)
            )
            self._mint_pack_moe_patched = False
            self._progress_last: dict[str, float] = {}

        def _progress_meta(self, tokens_generated: int, max_tokens: int) -> dict[str, Any]:
            return {
                "tokens_generated": int(tokens_generated),
                "max_tokens": int(max_tokens),
                "last_progress_at": time.time(),
            }

        def _bind_traceparent(self, traceparent: str | None) -> None:
            if isinstance(traceparent, str) and traceparent:
                restore_trace_id_from_traceparent(traceparent)

        async def _update_progress(
            self,
            *,
            request_id: str,
            tokens_generated: int,
            max_tokens: int,
            stage: str = "decode",
            min_interval_s: float = 5.0,
        ) -> None:
            now = time.time()
            last = self._progress_last.get(request_id)
            if last is not None and (now - last) < min_interval_s:
                return
            self._progress_last[request_id] = now
            try:
                from tinker_server.backend.future_store import future_store

                future_store.update_meta(
                    request_id,
                    meta={
                        "stage": stage,
                        "progress": self._progress_meta(tokens_generated, max_tokens),
                        "last_progress_at": time.time(),
                    },
                )
            except Exception as e:
                logger.warning(
                    "verl_vllm_progress_update_failed request_id=%s err=%s: %s",
                    request_id,
                    type(e).__name__,
                    e,
                )

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        async def is_engine_ready(self) -> bool:
            """Check if vLLM engine is properly initialized.

            __ray_ready__ only checks if Ray actor is alive, not engine status.
            This method verifies the engine was successfully created.
            """
            try:
                if not hasattr(self, "engine") or self.engine is None:
                    return False
                # list_loras() requires engine to be initialized
                await self.engine.list_loras()
                return True
            except Exception:
                return False

        async def is_ready(self) -> bool:
            """Alias for is_engine_ready() for compatibility with multinode_inference.py."""
            return await self.is_engine_ready()

        async def _ensure_pack_moe_patched(self) -> None:
            if self._mint_pack_moe_patched:
                return
            if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "").strip() not in {"1", "true", "yes"}:
                raise RuntimeError(
                    "vLLM worker patching requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 "
                    "(EngineCore collective_rpc sends function objects). "
                    "Set MINT_VLLM_ALLOW_INSECURE_SERIALIZATION=1 (default) or export full per-expert adapters "
                    "by setting MINT_MOE_LORA_SPARSE_EXPERT_EXPORT=0."
                )
            engine_core = getattr(self.engine, "engine_core", None)
            rpc = getattr(engine_core, "collective_rpc_async", None)
            if rpc is None:
                raise RuntimeError("vLLM engine_core.collective_rpc_async missing; cannot patch workers")
            results = await rpc(_mint_vllm_patch_pack_moe_sparse_ok)
            if not results or any((not isinstance(r, str)) or (not r.startswith("ok:")) for r in results):
                raise RuntimeError(f"vLLM worker pack_moe patch failed: {results}")
            self._mint_pack_moe_patched = True

        async def _maybe_ensure_pack_moe_patched_for_adapter_dir(self, adapter_dir: object) -> None:
            if not isinstance(adapter_dir, str) or not adapter_dir:
                raise RuntimeError(f"vLLM LoRARequest missing lora_path: {adapter_dir!r}")
            present = _mint_present_expert_ids_from_adapter_dir(adapter_dir)
            expected_num_experts = _mint_expected_num_experts_from_adapter_dir(adapter_dir)
            if _mint_sparse_expert_ids_need_patch(
                present,
                expected_num_experts=expected_num_experts,
            ):
                await self._ensure_pack_moe_patched()

        async def _maybe_ensure_pack_moe_patched_for_state_dict(
            self,
            state_dict: dict,
            *,
            base_model_name_or_path: object = None,
        ) -> None:
            present = _mint_present_expert_ids_from_keys([str(k) for k in state_dict.keys()])
            expected_num_experts = _mint_expected_num_experts_from_base_model(base_model_name_or_path)
            if _mint_sparse_expert_ids_need_patch(
                present,
                expected_num_experts=expected_num_experts,
            ):
                await self._ensure_pack_moe_patched()

        def _patch_lora_args(self, args):
            """Patch args Namespace with multi-LoRA config.

            verl hardcodes max_loras=1. This modifies the parsed args to use
            our configured values before AsyncEngineArgs.from_cli_args().
            """
            import logging
            _logger = logging.getLogger(__name__)

            if self.MULTI_LORA_MAX_LORAS > 1:
                args.max_loras = self.MULTI_LORA_MAX_LORAS
                _logger.info(f"Multi-LoRA: overriding max_loras={args.max_loras}")

            if self.MULTI_LORA_MAX_CPU_LORAS > 0:
                args.max_cpu_loras = self.MULTI_LORA_MAX_CPU_LORAS
                _logger.info(f"Multi-LoRA: overriding max_cpu_loras={args.max_cpu_loras}")

        async def run_server(self, args):
            """Override to inject multi-LoRA config (rank-0 node)."""
            self._patch_lora_args(args)
            return await super().run_server(args)

        async def run_headless(self, args):
            """Override to inject multi-LoRA config (non-rank-0 nodes)."""
            self._patch_lora_args(args)
            return await super().run_headless(args)

        async def add_lora(self, lora_request) -> None:
            """Add LoRA adapter to running engine.

            Args:
                lora_request: LoRARequest with lora_path pointing to adapter directory.
            """
            # Remove existing LoRA first if present
            try:
                loaded = await self.engine.list_loras()
                if VLLM_LORA_INT_ID in loaded:
                    await self.engine.remove_lora(VLLM_LORA_INT_ID)
            except Exception:
                pass  # May not have any LoRA loaded

            # Add new LoRA
            await self._maybe_ensure_pack_moe_patched_for_adapter_dir(getattr(lora_request, "lora_path", None))
            await self.engine.add_lora(lora_request)

        async def add_lora_from_tensors(
            self,
            state_dict: dict,
            peft_config: dict,
            traceparent: str | None = None,
        ) -> str:
            """Add LoRA from tensors by saving to temp dir on worker node.

            Receives tensors via Ray object store, saves to local temp file,
            then loads via file-based LoRARequest. This handles distributed
            deployments where API server and inference worker are on different nodes.

            Args:
                state_dict: LoRA weight tensors (already on CPU).
                peft_config: PEFT adapter config dict.

            Returns:
                Path where adapter was saved on worker node.
            """
            self._bind_traceparent(traceparent)
            import json
            import os
            import tempfile

            from safetensors.torch import save_file
            from vllm.lora.request import LoRARequest

            from verl.workers.rollout.vllm_rollout.utils import (
                VLLM_LORA_INT_ID,
                VLLM_LORA_NAME,
            )

            # Create temp dir on THIS worker node
            temp_dir = tempfile.mkdtemp(prefix="tinker_lora_")
            adapter_path = temp_dir

            # Save adapter files locally on worker node
            save_file(state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, indent=2)

            # Now load from local path
            lora_request = LoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=adapter_path,
            )

            # Remove existing and add new
            try:
                loaded = await self.engine.list_loras()
                if VLLM_LORA_INT_ID in loaded:
                    await self.engine.remove_lora(VLLM_LORA_INT_ID)
            except Exception:
                pass

            await self._maybe_ensure_pack_moe_patched_for_state_dict(
                state_dict,
                base_model_name_or_path=peft_config.get("base_model_name_or_path"),
            )
            await self.engine.add_lora(lora_request)
            return adapter_path

        async def list_loras(self) -> set[int]:
            """List loaded LoRA adapter IDs."""
            return await self.engine.list_loras()

        async def remove_lora(self, lora_int_id: int) -> None:
            """Remove a LoRA adapter by ID.

            Args:
                lora_int_id: The LoRA adapter ID to remove.
            """
            await self.engine.remove_lora(lora_int_id)
            # Clean up path tracking
            self._lora_paths.pop(lora_int_id, None)

        async def add_lora_with_id(
            self,
            lora_int_id: int,
            state_dict: dict,
            peft_config: dict,
            traceparent: str | None = None,
        ) -> str:
            """Add LoRA from tensors with specific lora_int_id.

            For multi-LoRA: each sampling session gets unique lora_int_id
            with frozen weights.

            Args:
                lora_int_id: Unique identifier for this LoRA adapter.
                state_dict: LoRA weight tensors (already on CPU).
                peft_config: PEFT adapter config dict.

            Returns:
                Path where adapter was saved on worker node.
            """
            self._bind_traceparent(traceparent)
            import json
            import os
            import tempfile

            from safetensors.torch import save_file
            from vllm.lora.request import LoRARequest

            # Create temp dir on THIS worker node
            temp_dir = tempfile.mkdtemp(prefix=f"tinker_lora_{lora_int_id}_")
            adapter_path = temp_dir

            # DEBUG: Print tensor norms to trace LoRA values on vLLM side
            debug = os.environ.get("MINT_VLLM_LORA_DEBUG", "0").strip() in {"1", "true", "yes"}
            if debug:
                print(
                    f"[DEBUG vLLM add_lora_with_id] lora_int_id={lora_int_id}, state_dict has {len(state_dict)} keys",
                    flush=True,
                )
                total_norm = 0.0
                nonzero_count = 0
                for k, v in list(state_dict.items())[:10]:
                    import torch

                    norm = float(v.norm().item()) if isinstance(v, torch.Tensor) else 0.0
                    total_norm += norm
                    if norm > 1e-8:
                        nonzero_count += 1
                    print(f"[DEBUG vLLM add_lora_with_id] {k}: norm={norm:.6f}", flush=True)
                print(
                    f"[DEBUG vLLM add_lora_with_id] Summary: {nonzero_count}/10 tensors non-zero, total_norm={total_norm:.6f}",
                    flush=True,
                )

            # Save adapter files locally on worker node
            save_file(state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, indent=2)
            if debug:
                print(f"[DEBUG vLLM add_lora_with_id] Saved to {adapter_path}", flush=True)

            # Track path for this lora_int_id (needed for GPU/CPU swap in generate)
            self._lora_paths[lora_int_id] = adapter_path

            # Create LoRARequest with the specific ID
            lora_request = LoRARequest(
                lora_name=str(lora_int_id),
                lora_int_id=lora_int_id,
                lora_path=adapter_path,
            )

            # Add to engine (no need to remove - this is a new unique ID)
            await self._maybe_ensure_pack_moe_patched_for_state_dict(
                state_dict,
                base_model_name_or_path=peft_config.get("base_model_name_or_path"),
            )
            await self.engine.add_lora(lora_request)
            return adapter_path

        @traced_async_from_traceparent(
            "sampling.vllm_actor.add_lora_from_path",
            component="vllm_actor",
            op="sampling.add_lora_from_path",
            request_id_arg="lora_name",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "lora_path": a.get("lora_path"),
            },
        )
        async def add_lora_from_path(
            self,
            lora_int_id: int,
            lora_path: str,
            lora_name: str,
            traceparent: str | None = None,
        ) -> None:
            """Add LoRA from filesystem path with specific lora_int_id.

            For multi-LoRA: loads directly from shared filesystem.
            File-based loading supports vLLM's GPU/CPU swapping
            (unlike TensorLoRARequest which fails with "stub" path).

            Args:
                lora_int_id: Unique identifier for this LoRA adapter.
                lora_path: Path to PEFT adapter directory.
                lora_name: Human-readable name for the adapter.
            """
            self._bind_traceparent(traceparent)
            from vllm.lora.request import LoRARequest

            debug = os.environ.get("MINT_VLLM_LORA_DEBUG", "0").strip() in {"1", "true", "yes"}
            if debug:
                print(
                    f"[DEBUG add_lora_from_path] lora_int_id={lora_int_id}, lora_path={lora_path}",
                    flush=True,
                )
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            await self._maybe_ensure_pack_moe_patched_for_adapter_dir(lora_path)
            await self.engine.add_lora(lora_request)

            # Track path for generate_with_lora (needed for GPU/CPU swap)
            self._lora_paths[lora_int_id] = lora_path
            if debug:
                print(f"[DEBUG add_lora_from_path] Stored path for lora_int_id={lora_int_id}", flush=True)

        @traced_async_from_traceparent(
            "sampling.vllm_actor.generate_with_lora",
            component="vllm_actor",
            op="sampling.generate_with_lora",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "max_tokens": a.get("max_tokens"),
                "num_samples": a.get("n"),
            },
        )
        async def generate_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
            max_tokens: int,
            stop: object | None = None,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
            n: int = 1,
            traceparent: str | None = None,
        ) -> dict | list[dict]:
            """Generate with a specific LoRA adapter.

            For multi-LoRA: routes request to session-specific adapter.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                lora_int_id: The LoRA adapter ID to use.
                max_tokens: Maximum tokens to generate.
                temperature: Sampling temperature.
                top_k: Top-k sampling parameter.
                top_p: Top-p sampling parameter.
                logprobs: Whether to return log probabilities.
                n: Number of sequences to sample for the same prompt.

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest
            from .vllm_stop import vllm_stop_kwargs

            # Compute effective max_tokens
            verl_max_tokens = self.config.max_model_len - len(prompt_ids)
            effective_max_tokens = min(max_tokens, verl_max_tokens)

            # Validate prompt fits in context window
            if effective_max_tokens < 1:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids):,} tokens) exceeds model context limit "
                    f"({self.config.max_model_len:,} tokens). Reduce prompt or use a model with larger context."
                )

            # Build sampling params
            # vLLM requires n=1 for greedy sampling (temperature=0)
            effective_n = 1 if temperature == 0 else max(1, int(n))
            sampling_params = SamplingParams(
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                n=effective_n,
                **vllm_stop_kwargs(stop, default_stop_token_ids=[151645, 151643, 163586, 163585]),
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Look up local path for this LoRA (needed for GPU/CPU swap)
            lora_path = self._lora_paths.get(lora_int_id)
            if lora_path is None:
                raise ValueError(f"No path found for lora_int_id={lora_int_id}")

            # Create LoRA request with real path for swap support
            lora_request = LoRARequest(
                lora_name=str(lora_int_id),
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            logger.info(
                "[vLLM actor] generate_with_lora ENTER req=%s prompt_len=%d max_tokens=%d effective_max=%d n=%d lora_id=%d",
                request_id, len(prompt_ids), max_tokens, effective_max_tokens, effective_n, lora_int_id,
            )
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            try:
                from tinker_server.backend.future_store import future_store

                future_store.update_meta(
                    request_id,
                    meta={
                        "stage": "prefill",
                        "progress": None,
                        "max_tokens": int(effective_max_tokens),
                    },
                )
            except Exception as e:
                logger.warning(
                    "verl_vllm_prefill_meta_update_failed request_id=%s err=%s: %s",
                    request_id,
                    type(e).__name__,
                    e,
                )

            # Get final response
            if sampling_params.n == 1:
                t0 = time.perf_counter()
                first_tok_s: float | None = None
                final_res = None
                last_progress_log = t0
                async for output in generator:
                    now = time.perf_counter()
                    if first_tok_s is None:
                        first_tok_s = now - t0
                        logger.info(
                            "[vLLM actor] generate_with_lora FIRST_TOKEN req=%s prefill_s=%.3f lora_id=%d",
                            request_id, first_tok_s, lora_int_id,
                        )
                    elif now - last_progress_log >= 30.0:
                        tokens_so_far = len(output.outputs[0].token_ids)
                        logger.info(
                            "[vLLM actor] generate_with_lora PROGRESS req=%s elapsed=%.1fs tokens=%d",
                            request_id, now - t0, tokens_so_far,
                        )
                        last_progress_log = now
                    if first_tok_s is None:
                        first_tok_s = time.perf_counter() - t0
                    try:
                        tokens_generated = len(output.outputs[0].token_ids)
                        await self._update_progress(
                            request_id=request_id,
                            tokens_generated=tokens_generated,
                            max_tokens=effective_max_tokens,
                            stage="decode",
                        )
                    except Exception as e:
                        logger.warning(
                            "verl_vllm_progress_compute_failed request_id=%s err=%s: %s",
                            request_id,
                            type(e).__name__,
                            e,
                        )
                    final_res = output
                assert final_res is not None
                total_s = time.perf_counter() - t0
                if self._timing:
                    print(
                        f"[vLLM timing] generate_with_lora req={request_id} prompt_len={len(prompt_ids)} "
                        f"max_tokens={max_tokens} n={sampling_params.n} lora_id={lora_int_id} "
                        f"total_s={total_s:.3f} first_tok_s={first_tok_s}"
                        ,
                        flush=True,
                    )

                token_ids = list(final_res.outputs[0].token_ids)
                log_probs = None
                if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
                    log_probs = [
                        logprobs[token_ids[i]].logprob
                        for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                    ]
                self._progress_last.pop(request_id, None)
                routed_experts = None
                if self._enable_rollout_routing_replay:
                    raw = getattr(final_res.outputs[0], "routed_experts", None)
                    if raw is None:
                        raise RuntimeError(
                            "enable_rollout_routing_replay is set but vLLM returned no routed_experts"
                        )
                    if raw is not None:
                        routed_experts = raw.tolist() if hasattr(raw, "tolist") else raw

                # Determine stop reason
                stop_reason = "length"
                if final_res.outputs[0].finish_reason == "stop":
                    stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in token_ids[-3:]):
                    stop_reason = "stop"

                return {
                    "token_ids": token_ids,
                    "logprobs": log_probs,
                    "routed_experts": routed_experts,
                    "stop_reason": stop_reason,
                    "_timing_total_s": float(total_s),
                    "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                }

            t0 = time.perf_counter()
            first_tok_s: float | None = None
            by_index: dict[int, Any] = {}
            async for output in generator:
                if first_tok_s is None:
                    first_tok_s = time.perf_counter() - t0
                try:
                    lengths = [len(oo.token_ids) for oo in output.outputs]
                    tokens_generated = min(lengths) if lengths else 0
                    await self._update_progress(
                        request_id=request_id,
                        tokens_generated=tokens_generated,
                        max_tokens=effective_max_tokens,
                        stage="decode",
                    )
                except Exception as e:
                    logger.warning(
                        "verl_vllm_progress_compute_failed request_id=%s err=%s: %s",
                        request_id,
                        type(e).__name__,
                        e,
                    )
                for out in output.outputs:
                    by_index[int(out.index)] = out
            total_s = time.perf_counter() - t0
            if self._timing:
                print(
                    f"[vLLM timing] generate_with_lora req={request_id} prompt_len={len(prompt_ids)} "
                    f"max_tokens={max_tokens} n={sampling_params.n} lora_id={lora_int_id} "
                    f"total_s={total_s:.3f} first_tok_s={first_tok_s}"
                    ,
                    flush=True,
                )

            if len(by_index) != sampling_params.n:
                raise RuntimeError(f"vLLM returned {len(by_index)} sequences for n={sampling_params.n}")

            outs: list[dict] = []
            for idx in range(sampling_params.n):
                out = by_index[idx]
                out_token_ids = list(out.token_ids)
                out_log_probs = None
                if sampling_params.logprobs is not None and out.logprobs:
                    out_log_probs = [
                        lp[out_token_ids[i]].logprob
                        for i, lp in enumerate(out.logprobs)
                    ]
                out_routed_experts = None
                if self._enable_rollout_routing_replay:
                    raw = getattr(out, "routed_experts", None)
                    if raw is None:
                        raise RuntimeError(
                            "enable_rollout_routing_replay is set but vLLM returned no routed_experts"
                        )
                    if raw is not None:
                        out_routed_experts = raw.tolist() if hasattr(raw, "tolist") else raw

                out_stop_reason = "length"
                if out.finish_reason == "stop":
                    out_stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in out_token_ids[-3:]):
                    out_stop_reason = "stop"

                outs.append(
                    {
                        "token_ids": out_token_ids,
                        "logprobs": out_log_probs,
                        "routed_experts": out_routed_experts,
                        "stop_reason": out_stop_reason,
                        "_timing_total_s": float(total_s),
                        "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                    }
                )
            self._progress_last.pop(request_id, None)
            return outs

        @traced_async_from_traceparent(
            "sampling.vllm_actor.generate_base",
            component="vllm_actor",
            op="sampling.generate_base",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "max_tokens": a.get("max_tokens"),
                "num_samples": a.get("n"),
            },
        )
        async def generate_base(
            self,
            prompt_ids: list[int],
            request_id: str,
            max_tokens: int,
            stop: object | None = None,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
            n: int = 1,
            traceparent: str | None = None,
        ) -> dict | list[dict]:
            """Generate using base model without any LoRA adapter.

            For multi-LoRA engine: generates with base model weights only.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                max_tokens: Maximum tokens to generate.
                temperature: Sampling temperature.
                top_k: Top-k sampling parameter.
                top_p: Top-p sampling parameter.
                logprobs: Whether to return log probabilities.
                n: Number of sequences to sample for the same prompt.

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from .vllm_stop import vllm_stop_kwargs

            # Compute effective max_tokens
            verl_max_tokens = self.config.max_model_len - len(prompt_ids)
            effective_max_tokens = min(max_tokens, verl_max_tokens)

            # Validate prompt fits in context window
            if effective_max_tokens < 1:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids):,} tokens) exceeds model context limit "
                    f"({self.config.max_model_len:,} tokens). Reduce prompt or use a model with larger context."
                )

            # Build sampling params
            # vLLM requires n=1 for greedy sampling (temperature=0)
            effective_n = 1 if temperature == 0 else max(1, int(n))
            sampling_params = SamplingParams(
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                n=effective_n,
                **vllm_stop_kwargs(stop, default_stop_token_ids=[151645, 151643, 163586, 163585]),
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Generate WITHOUT LoRA request (base model)
            logger.info(
                "[vLLM actor] generate_base ENTER req=%s prompt_len=%d max_tokens=%d effective_max=%d n=%d",
                request_id, len(prompt_ids), max_tokens, effective_max_tokens, effective_n,
            )
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=None,  # No LoRA = base model
            )
            try:
                from tinker_server.backend.future_store import future_store

                future_store.update_meta(
                    request_id,
                    meta={
                        "stage": "prefill",
                        "progress": None,
                        "max_tokens": int(effective_max_tokens),
                    },
                )
            except Exception as e:
                logger.warning(
                    "verl_vllm_prefill_meta_update_failed request_id=%s err=%s: %s",
                    request_id,
                    type(e).__name__,
                    e,
                )

            # Get final response
            if sampling_params.n == 1:
                t0 = time.perf_counter()
                first_tok_s: float | None = None
                final_res = None
                last_progress_log = t0
                async for output in generator:
                    now = time.perf_counter()
                    if first_tok_s is None:
                        first_tok_s = now - t0
                        logger.info(
                            "[vLLM actor] generate_base FIRST_TOKEN req=%s prefill_s=%.3f",
                            request_id, first_tok_s,
                        )
                    elif now - last_progress_log >= 30.0:
                        tokens_so_far = len(output.outputs[0].token_ids)
                        logger.info(
                            "[vLLM actor] generate_base PROGRESS req=%s elapsed=%.1fs tokens=%d",
                            request_id, now - t0, tokens_so_far,
                        )
                        last_progress_log = now
                    if first_tok_s is None:
                        first_tok_s = time.perf_counter() - t0
                    try:
                        tokens_generated = len(output.outputs[0].token_ids)
                        await self._update_progress(
                            request_id=request_id,
                            tokens_generated=tokens_generated,
                            max_tokens=effective_max_tokens,
                            stage="decode",
                        )
                    except Exception as e:
                        logger.warning(
                            "verl_vllm_progress_compute_failed request_id=%s err=%s: %s",
                            request_id,
                            type(e).__name__,
                            e,
                        )
                    final_res = output
                assert final_res is not None
                total_s = time.perf_counter() - t0
                if self._timing:
                    print(
                        f"[vLLM timing] generate_base req={request_id} prompt_len={len(prompt_ids)} "
                        f"max_tokens={max_tokens} n={sampling_params.n} total_s={total_s:.3f} "
                        f"first_tok_s={first_tok_s}"
                        ,
                        flush=True,
                    )

                token_ids = list(final_res.outputs[0].token_ids)
                log_probs = None
                if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
                    log_probs = [
                        logprobs[token_ids[i]].logprob
                        for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                    ]
                self._progress_last.pop(request_id, None)
                routed_experts = None
                if self._enable_rollout_routing_replay:
                    raw = getattr(final_res.outputs[0], "routed_experts", None)
                    if raw is None:
                        raise RuntimeError(
                            "enable_rollout_routing_replay is set but vLLM returned no routed_experts"
                        )
                    if raw is not None:
                        routed_experts = raw.tolist() if hasattr(raw, "tolist") else raw

                # Determine stop reason
                stop_reason = "length"
                if final_res.outputs[0].finish_reason == "stop":
                    stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in token_ids[-3:]):
                    stop_reason = "stop"

                return {
                    "token_ids": token_ids,
                    "logprobs": log_probs,
                    "routed_experts": routed_experts,
                    "stop_reason": stop_reason,
                    "_timing_total_s": float(total_s),
                    "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                }

            t0 = time.perf_counter()
            first_tok_s: float | None = None
            by_index: dict[int, Any] = {}
            async for output in generator:
                if first_tok_s is None:
                    first_tok_s = time.perf_counter() - t0
                try:
                    lengths = [len(oo.token_ids) for oo in output.outputs]
                    tokens_generated = min(lengths) if lengths else 0
                    await self._update_progress(
                        request_id=request_id,
                        tokens_generated=tokens_generated,
                        max_tokens=effective_max_tokens,
                        stage="decode",
                    )
                except Exception as e:
                    logger.warning(
                        "verl_vllm_progress_compute_failed request_id=%s err=%s: %s",
                        request_id,
                        type(e).__name__,
                        e,
                    )
                for out in output.outputs:
                    by_index[int(out.index)] = out
            total_s = time.perf_counter() - t0
            if self._timing:
                print(
                    f"[vLLM timing] generate_base req={request_id} prompt_len={len(prompt_ids)} "
                    f"max_tokens={max_tokens} n={sampling_params.n} total_s={total_s:.3f} "
                    f"first_tok_s={first_tok_s}"
                    ,
                    flush=True,
                )

            if len(by_index) != sampling_params.n:
                raise RuntimeError(f"vLLM returned {len(by_index)} sequences for n={sampling_params.n}")

            outs: list[dict] = []
            for idx in range(sampling_params.n):
                out = by_index[idx]
                out_token_ids = list(out.token_ids)
                out_log_probs = None
                if sampling_params.logprobs is not None and out.logprobs:
                    out_log_probs = [
                        lp[out_token_ids[i]].logprob
                        for i, lp in enumerate(out.logprobs)
                    ]
                out_routed_experts = None
                if self._enable_rollout_routing_replay:
                    raw = getattr(out, "routed_experts", None)
                    if raw is None:
                        raise RuntimeError(
                            "enable_rollout_routing_replay is set but vLLM returned no routed_experts"
                        )
                    if raw is not None:
                        out_routed_experts = raw.tolist() if hasattr(raw, "tolist") else raw

                out_stop_reason = "length"
                if out.finish_reason == "stop":
                    out_stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in out_token_ids[-3:]):
                    out_stop_reason = "stop"

                outs.append(
                    {
                        "token_ids": out_token_ids,
                        "logprobs": out_log_probs,
                        "routed_experts": out_routed_experts,
                        "stop_reason": out_stop_reason,
                        "_timing_total_s": float(total_s),
                        "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                    }
                )
            self._progress_last.pop(request_id, None)
            return outs

        async def generate(
            self,
            prompt_ids: list[int],
            sampling_params: dict,
            request_id: str,
            image_data: list | None = None,
            traceparent: str | None = None,
        ):
            """Generate with user's max_tokens respected.

            Override of verl's generate() which ignores user's max_tokens.
            Uses min(user_max_tokens, max_model_len - prompt_len).
            """
            self._bind_traceparent(traceparent)

            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            from verl.workers.rollout.vllm_rollout.utils import (
                VLLM_LORA_INT_ID,
                VLLM_LORA_NAME,
                VLLM_LORA_PATH,
            )
            from verl.workers.rollout.replica import TokenOutput

            # Extract user's max_tokens before verl overwrites it
            user_max_tokens = sampling_params.pop("max_tokens", None)
            verl_max_tokens = self.config.max_model_len - len(prompt_ids)

            if user_max_tokens is not None:
                max_tokens = min(user_max_tokens, verl_max_tokens)
            else:
                max_tokens = verl_max_tokens

            # Validate prompt fits in context window
            if max_tokens < 1:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids):,} tokens) exceeds model context limit "
                    f"({self.config.max_model_len:,} tokens). Reduce prompt or use a model with larger context."
                )

            # Rest of verl's generate() logic
            sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
            sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
            sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

            prompt = TokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data={"image": image_data} if image_data else None,
            )

            # Add lora request
            lora_request = None
            if self.model_config.lora_rank > 0:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                if lora_loaded:
                    lora_request = LoRARequest(
                        lora_name=VLLM_LORA_NAME,
                        lora_int_id=VLLM_LORA_INT_ID,
                        lora_path=VLLM_LORA_PATH,
                    )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            token_ids = final_res.outputs[0].token_ids
            log_probs = None
            if sampling_params.logprobs is not None:
                log_probs = [
                    logprobs[token_ids[i]].logprob
                    for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                ]
            routed_experts = None
            if self._enable_rollout_routing_replay:
                raw = getattr(final_res.outputs[0], "routed_experts", None)
                if raw is None:
                    raise RuntimeError(
                        "enable_rollout_routing_replay is set but vLLM returned no routed_experts"
                    )
                if raw is not None:
                    routed_experts = raw.tolist() if hasattr(raw, "tolist") else raw
            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                routed_experts=routed_experts,
            )

        async def compute_prompt_logprobs(
            self,
            prompt_ids: list[int],
            request_id: str,
            traceparent: str | None = None,
        ) -> list[float | None]:
            """Compute logprobs for each token in the prompt.

            Returns a list of length len(prompt_ids), where:
            - logprobs[0] is None (first token has no conditioning context)
            - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.

            Returns:
                List of logprobs, length = len(prompt_ids).
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            from verl.workers.rollout.vllm_rollout.utils import (
                VLLM_LORA_INT_ID,
                VLLM_LORA_NAME,
                VLLM_LORA_PATH,
            )

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

            # Use max_tokens=1 with prompt_logprobs to get logprobs for prompt tokens
            # prompt_logprobs=1 returns top-1 logprob for each position
            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Add lora request if enabled
            lora_request = None
            if self.model_config.lora_rank > 0:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                if lora_loaded:
                    lora_request = LoRARequest(
                        lora_name=VLLM_LORA_NAME,
                        lora_int_id=VLLM_LORA_INT_ID,
                        lora_path=VLLM_LORA_PATH,
                    )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            # Extract prompt logprobs
            # prompt_logprobs is a list where element i contains logprob info for token i
            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            out: list[float | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    out.append(None)
                    continue
                token_id = prompt_ids[i]
                token_lp = prompt_logprobs[i].get(token_id)
                out.append(token_lp.logprob if token_lp is not None else None)

            return out

        async def compute_prompt_topk(
            self,
            prompt_ids: list[int],
            request_id: str,
            k: int = 10,
        ) -> list[list[tuple[int, float]] | None]:
            """Get top-K tokens and logprobs at each prompt position.

            Returns topk[i] = list of (token_id, logprob) pairs for position i.
            Output length is len(prompt_ids), with topk[0]=None.
            Each non-None entry is sorted by logprob descending and truncated to <= k.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                k: Number of top tokens to return (default 10).

            Returns:
                List of per-position top-k lists.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            from verl.workers.rollout.vllm_rollout.utils import (
                VLLM_LORA_PATH,
            )

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]
            if k <= 0:
                return [None] * len(prompt_ids)

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=k,  # Get top-K at each position
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Add lora request if enabled
            lora_request = None
            loaded_loras = await self.engine.list_loras()
            print(f"[compute_prompt_topk] lora_rank={self.model_config.lora_rank}, loaded_loras={loaded_loras}", flush=True)
            if self.model_config.lora_rank > 0 and loaded_loras:
                # Use the first loaded LoRA (multi-LoRA uses dynamic IDs starting from 1)
                lora_id = list(loaded_loras)[0]
                lora_path = self._lora_paths.get(lora_id, VLLM_LORA_PATH)
                print(f"[compute_prompt_topk] Using lora_id={lora_id}, path={lora_path}", flush=True)
                lora_request = LoRARequest(
                    lora_name=f"lora_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path=lora_path,
                )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            result: list[list[tuple[int, float]] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos = [(int(token_id), float(lp.logprob)) for token_id, lp in prompt_logprobs[i].items()]
                pos.sort(key=lambda x: x[1], reverse=True)
                result.append(pos[:k])

            return result

        @traced_async_from_traceparent(
            "sampling.vllm_actor.compute_prompt_logprobs_with_lora",
            component="vllm_actor",
            op="sampling.compute_prompt_logprobs_with_lora",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
            },
        )
        async def compute_prompt_logprobs_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
            traceparent: str | None = None,
        ) -> list[float | None]:
            """Compute logprobs with specific LoRA adapter.

            For multi-LoRA: routes request to session-specific adapter.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                lora_int_id: The LoRA adapter ID to use.

            Returns:
                List of logprobs, length = len(prompt_ids).
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Look up local path for this LoRA
            lora_path = self._lora_paths.get(lora_int_id)
            if lora_path is None:
                raise ValueError(f"No path found for lora_int_id={lora_int_id}")

            # DEBUG: Log which LoRA is being used
            print(f"[DEBUG vLLM compute_logprobs] Using lora_int_id={lora_int_id}, path={lora_path}", flush=True)

            lora_request = LoRARequest(
                lora_name=str(lora_int_id),
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            # Extract prompt logprobs
            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            out: list[float | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    out.append(None)
                    continue
                token_id = prompt_ids[i]
                token_lp = prompt_logprobs[i].get(token_id)
                out.append(token_lp.logprob if token_lp is not None else None)

            return out

        @traced_async_from_traceparent(
            "sampling.vllm_actor.compute_prompt_topk_with_lora",
            component="vllm_actor",
            op="sampling.compute_prompt_topk_with_lora",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "topk": a.get("k"),
            },
        )
        async def compute_prompt_topk_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
            k: int = 10,
            traceparent: str | None = None,
        ) -> list[list[tuple[int, float]] | None]:
            """Get top-K tokens with specific LoRA adapter.

            Returns topk[i] = list of (token_id, logprob) pairs for position i.
            Output length is len(prompt_ids), with topk[0]=None.
            Each non-None entry is sorted by logprob descending and truncated to <= k.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                lora_int_id: The LoRA adapter ID to use.
                k: Number of top tokens to return.

            Returns:
                List of per-position top-k lists.
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]
            if k <= 0:
                return [None] * len(prompt_ids)

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=k,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            lora_path = self._lora_paths.get(lora_int_id)
            if lora_path is None:
                raise ValueError(f"No path found for lora_int_id={lora_int_id}")

            print(f"[compute_prompt_topk_with_lora] Using lora_int_id={lora_int_id}, path={lora_path}", flush=True)

            lora_request = LoRARequest(
                lora_name=str(lora_int_id),
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            result: list[list[tuple[int, float]] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos = [(int(token_id), float(lp.logprob)) for token_id, lp in prompt_logprobs[i].items()]
                pos.sort(key=lambda x: x[1], reverse=True)
                result.append(pos[:k])

            return result

        @traced_async_from_traceparent(
            "sampling.vllm_actor.compute_prompt_logprobs_base",
            component="vllm_actor",
            op="sampling.compute_prompt_logprobs_base",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "prompt_tokens": len(a.get("prompt_ids") or []),
            },
        )
        async def compute_prompt_logprobs_base(
            self,
            prompt_ids: list[int],
            request_id: str,
            traceparent: str | None = None,
        ) -> list[float | None]:
            """Compute logprobs using base model without any LoRA adapter.

            For multi-LoRA engine: computes logprobs with base model weights only.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.

            Returns:
                List of logprobs, length = len(prompt_ids).
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Generate WITHOUT LoRA request (base model)
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=None,  # No LoRA = base model
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            # Extract prompt logprobs
            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            out: list[float | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    out.append(None)
                    continue
                token_id = prompt_ids[i]
                token_lp = prompt_logprobs[i].get(token_id)
                out.append(token_lp.logprob if token_lp is not None else None)

            return out

        @traced_async_from_traceparent(
            "sampling.vllm_actor.compute_prompt_topk_base",
            component="vllm_actor",
            op="sampling.compute_prompt_topk_base",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "topk": a.get("k"),
            },
        )
        async def compute_prompt_topk_base(
            self,
            prompt_ids: list[int],
            request_id: str,
            k: int = 10,
            traceparent: str | None = None,
        ) -> list[list[tuple[int, float]] | None]:
            """Get top-K tokens using base model without any LoRA adapter.

            For multi-LoRA engine: computes top-K with base model weights only.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                k: Number of top tokens to return.

            Returns:
                List of per-position top-k lists.
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]
            if k <= 0:
                return [None] * len(prompt_ids)

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=k,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Generate WITHOUT LoRA request (base model)
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=None,  # No LoRA = base model
            )

            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            result: list[list[tuple[int, float]] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos = [(int(token_id), float(lp.logprob)) for token_id, lp in prompt_logprobs[i].items()]
                pos.sort(key=lambda x: x[1], reverse=True)
                result.append(pos[:k])

            return result


        async def get_debug_env_info(self) -> dict:
            """Return environment info for debugging PYTHONPATH issues."""
            import os
            import sys
            import vllm
            return {
                "pythonpath": os.environ.get("PYTHONPATH", "NOT SET")[:500],
                "vllm_file": vllm.__file__,
                "sys_path_first_5": sys.path[:5],
            }

        async def get_kv_debug_info(self) -> dict:
            def _collect_kv_info(worker_wrapper):
                worker = getattr(worker_wrapper, "worker", None)
                target = worker if worker is not None else worker_wrapper
                model_runner = getattr(target, "model_runner", None)
                kv_cfg = getattr(model_runner, "kv_cache_config", None)
                return {
                    "available_kv_cache_memory_bytes": int(
                        getattr(target, "available_kv_cache_memory_bytes", 0) or 0
                    ),
                    "requested_memory_bytes": int(
                        getattr(target, "requested_memory", 0) or 0
                    ),
                    "non_torch_memory_bytes": int(
                        getattr(target, "non_torch_memory", 0) or 0
                    ),
                    "peak_activation_memory_bytes": int(
                        getattr(target, "peak_activation_memory", 0) or 0
                    ),
                    "kv_cache_num_blocks": int(
                        getattr(kv_cfg, "num_blocks", 0) or 0
                    ) if kv_cfg is not None else 0,
                    "kv_cache_groups": len(
                        getattr(kv_cfg, "kv_cache_groups", []) or []
                    ) if kv_cfg is not None else 0,
                }

            infos = await self.engine.collective_rpc(_collect_kv_info)
            return {
                "per_worker": infos,
                "min_available_kv_cache_memory_bytes": min(
                    int(x.get("available_kv_cache_memory_bytes", 0) or 0) for x in infos
                ) if infos else 0,
                "max_available_kv_cache_memory_bytes": max(
                    int(x.get("available_kv_cache_memory_bytes", 0) or 0) for x in infos
                ) if infos else 0,
            }


        async def test_mp_spawn_from_actor(self) -> dict:
            """Test multiprocessing spawn from within actor to debug PYTHONPATH."""
            import subprocess
            
            # Run the test script
            result = subprocess.run(
                ["python3", "/vePFS-Mindverse/share/code/test_mp_spawn.py"],
                capture_output=True, text=True, timeout=120
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }


        async def test_vllm_spawn_direct(self) -> dict:
            """Test vLLM's spawn mechanism directly from within actor."""
            import os
            import sys
            from vllm.utils.system_utils import get_mp_context
            
            result_file = "/vePFS-Mindverse/share/code/vllm_direct_spawn_result.txt"
            if os.path.exists(result_file):
                os.remove(result_file)
            
            # Get vLLM's multiprocessing context
            context = get_mp_context()
            
            # Worker function must be defined at module level for pickling
            # So we use a simple script approach
            worker_code = '''
import os, sys
import vllm
result_file = "/vePFS-Mindverse/share/code/vllm_direct_spawn_result.txt"
with open(result_file, "w") as f:
    f.write(f"vllm.__file__: {vllm.__file__}\\n")
    f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')[:200]}\\n")
    f.write(f"sys.path[:3]: {sys.path[:3]}\\n")
'''
            import subprocess
            proc = context.Process(
                target=subprocess.run,
                args=(["python3", "-c", worker_code],),
                daemon=True,
            )
            proc.start()
            proc.join(timeout=30)
            
            if os.path.exists(result_file):
                with open(result_file) as f:
                    content = f.read()
                return {
                    "context_method": context._name,
                    "parent_pythonpath": os.environ.get("PYTHONPATH", "NOT SET")[:100],
                    "parent_vllm": sys.modules.get("vllm", {}).__file__ if "vllm" in sys.modules else "not imported",
                    "worker_result": content,
                }
            return {"error": "Worker did not create result file"}


        async def test_actual_worker_spawn(self) -> dict:
            """Test spawning with module-level function (like vLLM does)."""
            import os
            from vllm.utils.system_utils import get_mp_context
            from tinker_server.spawn_worker_test import spawn_worker_check, RESULT_FILE
            
            if os.path.exists(RESULT_FILE):
                os.remove(RESULT_FILE)
            
            context = get_mp_context()
            proc = context.Process(target=spawn_worker_check, daemon=True)
            proc.start()
            proc.join(timeout=30)
            
            if os.path.exists(RESULT_FILE):
                with open(RESULT_FILE) as f:
                    content = f.read()
                return {
                    "context_method": context._name,
                    "parent_pythonpath": os.environ.get("PYTHONPATH", "NOT SET")[:100],
                    "worker_result": content,
                }
            return {"error": "Worker did not create result file"}


        async def dump_raw_logits(
            self,
            prompt_ids: list[int],
            request_id: str,
            dump_path: str = "/vePFS-Mindverse/share/code/vllm_raw_logits.pt",
        ) -> str:
            """Dump raw logits from vLLM forward pass using LogitsProcessor.
            
            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                dump_path: Where to save the logits.
                
            Returns:
                Path to dumped file or error message.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            import torch
            
            # Storage for captured logits
            captured_logits = []
            
            def capture_logits(past_tokens: list[int], logits: torch.Tensor) -> torch.Tensor:
                """LogitsProcessor that captures raw logits."""
                captured_logits.append({
                    "past_tokens": list(past_tokens),
                    "logits": logits.detach().cpu().clone(),
                    "shape": list(logits.shape),
                })
                return logits  # Return unchanged
            
            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,  # Need this to trigger logits computation for prompt
                temperature=1.0,
                logits_processors=[capture_logits],
            )
            
            prompt = TokensPrompt(prompt_token_ids=prompt_ids)
            
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            
            async for output in generator:
                pass  # Consume the generator
            
            if captured_logits:
                torch.save(captured_logits, dump_path)
                return f"Saved {len(captured_logits)} logit tensors to {dump_path}"
            else:
                return "No logits captured"

    # Apply ray.remote() dynamically, like verl does
    return ray.remote(num_cpus=1)(ExtendedVLLMHttpServer)


@dataclass
class GenerateResult:
    """Result from a generate call."""

    token_ids: list[int]
    log_probs: list[float] | None
    routed_experts: list | None = None


class VerlInferenceEngine:
    """Wraps verl's vLLMHttpServer for inference.

    Provides a simple interface for text generation using verl's
    distributed vLLM infrastructure.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        lora_rank: int = 0,
        lora_adapter_path: str | None = None,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.lora_rank = lora_rank
        self.lora_adapter_path = lora_adapter_path
        self.server: vLLMHttpServer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Ray and launch vLLM server."""
        if self._initialized:
            return

        if not ray.is_initialized():
            address = os.environ.get("RAY_ADDRESS", "").strip()
            if not address:
                raise RuntimeError("RAY_ADDRESS must be set before initializing VerlInferenceEngine")
            init_ray(
                address=address,
                namespace=RAY_NAMESPACE,
                ignore_reinit_error=True,
            )

        # Compute total GPUs needed for MoE models
        # For EP (expert parallelism), total_gpus = TP * DP
        total_gpus = self.tensor_parallel_size * self.data_parallel_size

        # Build engine_kwargs for expert parallelism
        # Pass enable_expert_parallel directly to vLLM, bypassing verl's worker-based EP
        # This allows vLLM to handle EP internally via multiprocessing
        engine_kwargs = {}
        if self.data_parallel_size > 1:
            engine_kwargs = {
                "vllm": {
                    "enable_expert_parallel": True,
                }
            }
            logger.info(f"Enabling expert parallelism via vLLM (DP={self.data_parallel_size})")

        # Use model registry as single source of truth for memory constraints.
        cfg = get_model_config(self.model_path)
        if cfg.max_loras is not None:
            max_loras = int(cfg.max_loras)
        else:
            max_loras = 1 if cfg.is_moe else 64
        if cfg.max_cpu_loras is not None:
            max_cpu_loras = int(cfg.max_cpu_loras)
        else:
            max_cpu_loras = max_loras
        # Use our extended server with add_lora support
        ExtendedVLLMHttpServer = _create_extended_server_class(
            max_loras=max_loras,
            max_cpu_loras=max_cpu_loras,
        )
        max_model_len = cfg.max_model_len
        max_num_seqs = cfg.max_num_seqs or 256
        gpu_util = cfg.gpu_memory_utilization or self.gpu_memory_utilization
        # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
        max_num_batched_tokens = cfg.max_num_batched_tokens
        if max_num_batched_tokens is None:
            max_num_batched_tokens = 4096 if max_model_len >= 32768 else 8192
        # vLLM v1 SchedulerConfig requires max_num_batched_tokens >= max_model_len.
        # With enable_chunked_prefill=True (default), vLLM still chunks prefill internally,
        # so memory behavior is preserved — this floor only prevents the hard validation rejection.
        if max_num_batched_tokens < max_model_len:
            max_num_batched_tokens = max_model_len
        # verl calculates max_model_len = prompt_length + response_length
        # We split evenly, but this does NOT restrict actual prompt/response sizes:
        # - The split only affects verl's default max_new_tokens (response_length)
        # - Our generate() computes actual limit dynamically: max_model_len - len(prompt)
        # - A 32K model can do 25K prompt + 7K response, or 5K prompt + 27K response
        # The sum (total context window) is what matters, not the individual values.
        prompt_length = max_model_len // 2
        response_length = max_model_len - prompt_length
        logger.info(f"vLLM max_model_len={max_model_len} (prompt={prompt_length}, response={response_length})")

        enable_rollout_routing_replay = (server_config.router_replay_mode == "R3")
        logger.info(
            f"Launching vLLMHttpServer for {self.model_path} "
            f"(TP={self.tensor_parallel_size}, DP={self.data_parallel_size}, total_gpus={total_gpus}, "
            f"lora_rank={self.lora_rank}, adapter_path={self.lora_adapter_path})"
        )

        # Create ExtendedVLLMHttpServer as Ray actor
        # Request total_gpus (TP * DP) via .options() for MoE expert parallelism
        # runtime_env prepends vLLM 0.12.0 from PFS for MoE LoRA support
        from ..config import actor_ld_library_path, actor_runtime_env_vars, otel_env_vars
        from dataclasses import asdict

        from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
        from verl.workers.rollout.replica import RolloutMode

        # Create rollout config using dataclass
        # NOTE: Keep expert_parallel_size=1 to avoid verl's worker-based EP assertion
        # Expert parallelism is enabled via engine_kwargs instead
        rollout_config = RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=gpu_util,
            prompt_length=prompt_length,
            response_length=response_length,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype="auto",
            load_format="auto",
            enforce_eager=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=True,
            disable_log_stats=True,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            data_parallel_size=1,
            expert_parallel_size=1,
            engine_kwargs=engine_kwargs,
            checkpoint_engine=CheckpointEngineConfig(backend="naive"),
            enable_rollout_routing_replay=enable_rollout_routing_replay,
        )
        model_config = HFModelConfig(
            path=self.model_path,
            trust_remote_code=True,
            lora_rank=self.lora_rank,
            lora_adapter_path=self.lora_adapter_path,
        )
        rollout_payload = asdict(rollout_config)
        if not rollout_payload.get("_target_"):
            rollout_payload["_target_"] = "verl.workers.config.RolloutConfig"
        remote_kwargs: dict[str, object] = {
            "config": rollout_payload,
            "model_config": model_config,
            "rollout_mode": RolloutMode.STANDALONE,
            "workers": [],
            "replica_rank": 0,
            "node_rank": 0,
            "gpus_per_node": total_gpus,
            "nnodes": 1,
            "cuda_visible_devices": ",".join(str(i) for i in range(total_gpus)),
        }
        self.server = ExtendedVLLMHttpServer.options(
            num_gpus=total_gpus,
            max_concurrency=int(os.environ.get("MINT_VLLM_ACTOR_MAX_CONCURRENCY", "64")),
            runtime_env={
                "env_vars": actor_runtime_env_vars(
                    pythonpath=PFS_PYTHONPATH,
                    extra={
                    "LD_LIBRARY_PATH": actor_ld_library_path(),
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "VLLM_ATTENTION_BACKEND": server_config.vllm_attention_backend,
                    **otel_env_vars(),
                    },
                )
            },
        ).remote(**remote_kwargs)

        # Launch the server
        await self.server.launch_server.remote()
        self._initialized = True
        logger.info("vLLMHttpServer initialized")

    async def generate(
        self,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> GenerateResult:
        """Generate tokens using verl's vLLM server.

        Args:
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter (-1 to disable).
            top_p: Top-p (nucleus) sampling parameter.
            logprobs: Whether to return log probabilities.

        Returns:
            GenerateResult with token_ids and optional log_probs.
        """
        if not self._initialized:
            await self.initialize()

        from .vllm_stop import vllm_stop_kwargs

        # Pass max_tokens to our overridden generate() in ExtendedVLLMHttpServer
        # which uses min(user_max_tokens, max_model_len - prompt_len)
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "logprobs": logprobs,
        }
        sampling_params.update(vllm_stop_kwargs(stop, default_stop_token_ids=[151645, 151643, 163586, 163585]))
        traceparent = get_current_traceparent()

        # Call the Ray actor's generate method
        result = await self.server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            traceparent=traceparent,
        )

        return GenerateResult(
            token_ids=list(result.token_ids),
            log_probs=list(result.log_probs) if result.log_probs else None,
            routed_experts=(
                result.routed_experts.tolist()
                if getattr(result, "routed_experts", None) is not None and hasattr(result.routed_experts, "tolist")
                else getattr(result, "routed_experts", None)
            ),
        )

    async def compute_logprobs(
        self,
        prompt_ids: list[int],
        request_id: str,
    ) -> list[float | None]:
        """Compute logprobs for each token in the sequence.

        Returns a list of length len(prompt_ids), where:
        - logprobs[0] is None (first token has no conditioning context)
        - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1

        Args:
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.

        Returns:
            List of logprobs, length = len(prompt_ids).
        """
        if not self._initialized:
            await self.initialize()
        traceparent = get_current_traceparent()

        result = await self.server.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            traceparent=traceparent,
        )
        return list(result)

    async def load_lora_from_path(self, adapter_path: str) -> None:
        """Hot-reload LoRA adapter from filesystem path.

        Loads trained LoRA weights into the running vLLM engine without restart.
        Transfers tensors via Ray object store to handle distributed deployments
        where API server and inference worker are on different nodes.

        Args:
            adapter_path: Path to adapter directory containing
                          adapter_model.safetensors and adapter_config.json.
        """
        import json
        import os

        from safetensors.torch import load_file

        if not self._initialized:
            await self.initialize()

        # Load tensors and config from local files (on API server)
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        config_path = os.path.join(adapter_path, "adapter_config.json")

        state_dict = load_file(weights_path)
        with open(config_path, "r") as f:
            peft_config = json.load(f)
        traceparent = get_current_traceparent()

        # Pass tensors via Ray to inference worker, which saves locally and loads
        # This handles distributed deployments where nodes have different filesystems
        worker_path = await self.server.add_lora_from_tensors.remote(
            state_dict,
            peft_config,
            traceparent=traceparent,
        )

        logger.info(f"Hot-loaded LoRA adapter (API: {adapter_path} -> Worker: {worker_path})")

    async def shutdown(self) -> None:
        """Cleanup Ray actors."""
        if self.server:
            ray_kill.kill(self.server, reason="verl_inference_shutdown", namespace=RAY_NAMESPACE)
            self.server = None
        self._initialized = False
        logger.info("VerlInferenceEngine shutdown")
