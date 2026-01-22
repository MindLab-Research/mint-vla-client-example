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
import torch
from omegaconf import OmegaConf

from . import ray_kill

if TYPE_CHECKING:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH, RAY_NAMESPACE

# Import model registry
from tinker_server.backend.model_registry import get_model_config

# Apply verl's hijack for TensorLoRARequest support
# Must be done before engine initialization
def _apply_vllm_hijack():
    """Apply verl's vLLM hijack for TensorLoRARequest support."""
    try:
        from verl.utils.vllm import VLLMHijack, is_version_ge
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()
            logger.info("Applied VLLMHijack for TensorLoRARequest support")
    except Exception as e:
        logger.warning(f"Could not apply VLLMHijack: {e}")

_apply_vllm_hijack()


# Extended vLLMHttpServer with add_lora support
# Must be defined after hijack is applied
def _create_extended_server_class(max_loras: int = 1, max_cpu_loras: int = 0):
    """Create extended vLLMHttpServer class with add_lora method.

    Args:
        max_loras: Maximum LoRAs in a single batch (default: 1).
                   Set > 1 for multi-LoRA concurrent inference.
        max_cpu_loras: Maximum LoRAs in CPU cache for swap (default: 0).
    """
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID

    # Capture in closure
    _max_loras = max_loras
    _max_cpu_loras = max_cpu_loras

    @ray.remote(num_cpus=1)
    class ExtendedVLLMHttpServer(vLLMHttpServerBase):
        """Extended vLLMHttpServer with hot LoRA loading support."""

        # Class-level config for multi-LoRA (captured from factory)
        MULTI_LORA_MAX_LORAS = _max_loras
        MULTI_LORA_MAX_CPU_LORAS = _max_cpu_loras

        def __init__(self, *args, **kwargs):
            """Initialize with VLLMHijack applied first."""
            # Set PYTHONPATH in OS environment so vLLM's TP workers inherit it
            # Ray's runtime_env only sets it for this process, not multiprocessing children
            import os
            import sys
            pfs_pythonpath = PFS_PYTHONPATH
            os.environ["PYTHONPATH"] = pfs_pythonpath + ":" + os.environ.get("PYTHONPATH", "")
            for p in reversed(pfs_pythonpath.split(":")):
                if p and p not in sys.path:
                    sys.path.insert(0, p)
            print(f"[ExtendedVLLMHttpServer] Set PYTHONPATH, vLLM path: {sys.path[0]}", flush=True)

            # Apply hijack BEFORE engine creation (in parent __init__)
            # This runs inside the Ray actor process on GPU node
            try:
                from verl.utils.vllm import VLLMHijack, is_version_ge
                if is_version_ge(pkg="vllm", minver="0.7.3"):
                    VLLMHijack.hijack()
            except Exception:
                pass

            super().__init__(*args, **kwargs)
            # Track local paths for multi-LoRA (needed for GPU/CPU swap)
            self._lora_paths: dict[int, str] = {}

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
            await self.engine.add_lora(lora_request)

        async def add_lora_from_tensors(
            self,
            state_dict: dict,
            peft_config: dict,
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
            import json
            import os
            import tempfile

            from safetensors.torch import save_file
            from vllm.lora.request import LoRARequest

            # Create temp dir on THIS worker node
            temp_dir = tempfile.mkdtemp(prefix=f"tinker_lora_{lora_int_id}_")
            adapter_path = temp_dir

            # DEBUG: Print tensor norms to trace LoRA values on vLLM side
            print(f"[DEBUG vLLM add_lora_with_id] lora_int_id={lora_int_id}, state_dict has {len(state_dict)} keys", flush=True)
            total_norm = 0.0
            nonzero_count = 0
            for k, v in list(state_dict.items())[:10]:
                import torch
                norm = float(v.norm().item()) if isinstance(v, torch.Tensor) else 0.0
                total_norm += norm
                if norm > 1e-8:
                    nonzero_count += 1
                print(f"[DEBUG vLLM add_lora_with_id] {k}: norm={norm:.6f}", flush=True)
            print(f"[DEBUG vLLM add_lora_with_id] Summary: {nonzero_count}/10 tensors non-zero, total_norm={total_norm:.6f}", flush=True)

            # Save adapter files locally on worker node
            save_file(state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, indent=2)
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
            await self.engine.add_lora(lora_request)
            return adapter_path

        async def add_lora_from_path(
            self,
            lora_int_id: int,
            lora_path: str,
            lora_name: str,
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
            from vllm.lora.request import LoRARequest

            print(f"[DEBUG add_lora_from_path] lora_int_id={lora_int_id}, lora_path={lora_path}", flush=True)
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            await self.engine.add_lora(lora_request)

            # Track path for generate_with_lora (needed for GPU/CPU swap)
            self._lora_paths[lora_int_id] = lora_path
            print(f"[DEBUG add_lora_from_path] Stored path for lora_int_id={lora_int_id}", flush=True)

        async def generate_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
            max_tokens: int,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
        ) -> dict:
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

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

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
            sampling_params = SamplingParams(
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                # EOS token handling for Qwen
                stop_token_ids=[151645, 151643],
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

            token_ids = list(final_res.outputs[0].token_ids)
            log_probs = None
            if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
                log_probs = [
                    logprobs[token_ids[i]].logprob
                    for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                ]

            # Determine stop reason
            stop_reason = "length"
            if final_res.outputs[0].finish_reason == "stop":
                stop_reason = "stop"
            elif any(tid in [151645, 151643] for tid in token_ids[-3:]):
                stop_reason = "stop"

            return {
                "token_ids": token_ids,
                "logprobs": log_probs,
                "stop_reason": stop_reason,
            }

        async def generate_base(
            self,
            prompt_ids: list[int],
            request_id: str,
            max_tokens: int,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
        ) -> dict:
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

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

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
            sampling_params = SamplingParams(
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                # EOS token handling for Qwen
                stop_token_ids=[151645, 151643],
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

            token_ids = list(final_res.outputs[0].token_ids)
            log_probs = None
            if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
                log_probs = [
                    logprobs[token_ids[i]].logprob
                    for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                ]

            # Determine stop reason
            stop_reason = "length"
            if final_res.outputs[0].finish_reason == "stop":
                stop_reason = "stop"
            elif any(tid in [151645, 151643] for tid in token_ids[-3:]):
                stop_reason = "stop"

            return {
                "token_ids": token_ids,
                "logprobs": log_probs,
                "stop_reason": stop_reason,
            }

        async def generate(
            self,
            prompt_ids: list[int],
            sampling_params: dict,
            request_id: str,
            image_data: list | None = None,
        ):
            """Generate with user's max_tokens respected.

            Override of verl's generate() which ignores user's max_tokens.
            Uses min(user_max_tokens, max_model_len - prompt_len).
            """
            from typing import Optional

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
            return TokenOutput(token_ids=token_ids, log_probs=log_probs)

        async def compute_prompt_logprobs(
            self,
            prompt_ids: list[int],
            request_id: str,
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
        ) -> list[dict[int, float] | None]:
            """Get top-K tokens and logprobs at each prompt position.

            Returns topk[i] = dict of {token_id: logprob} for top-K tokens
            at position i (predicting token i+1 given tokens 0..i).
            Output length is len(prompt_ids), with topk[0]=None.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                k: Number of top tokens to return (default 10).

            Returns:
                List of dicts, each mapping token_id to logprob.
            """
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

            result: list[dict[int, float] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos_dict: dict[int, float] = {}
                for token_id, logprob_obj in prompt_logprobs[i].items():
                    pos_dict[token_id] = logprob_obj.logprob
                result.append(pos_dict)

            return result

        async def compute_prompt_logprobs_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
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

        async def compute_prompt_topk_with_lora(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int,
            k: int = 10,
        ) -> list[dict[int, float] | None]:
            """Get top-K tokens with specific LoRA adapter.

            Returns topk[i] = dict of {token_id: logprob} for position i.
            Output length is len(prompt_ids), with topk[0]=None.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                lora_int_id: The LoRA adapter ID to use.
                k: Number of top tokens to return.

            Returns:
                List of dicts mapping token_id to logprob.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

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

            result: list[dict[int, float] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos_dict: dict[int, float] = {}
                for token_id, logprob_obj in prompt_logprobs[i].items():
                    pos_dict[token_id] = logprob_obj.logprob
                result.append(pos_dict)

            return result

        async def compute_prompt_logprobs_base(
            self,
            prompt_ids: list[int],
            request_id: str,
        ) -> list[float | None]:
            """Compute logprobs using base model without any LoRA adapter.

            For multi-LoRA engine: computes logprobs with base model weights only.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.

            Returns:
                List of logprobs, length = len(prompt_ids).
            """
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

        async def compute_prompt_topk_base(
            self,
            prompt_ids: list[int],
            request_id: str,
            k: int = 10,
        ) -> list[dict[int, float] | None]:
            """Get top-K tokens using base model without any LoRA adapter.

            For multi-LoRA engine: computes top-K with base model weights only.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                k: Number of top tokens to return.

            Returns:
                List of dicts mapping token_id to logprob.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

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

            result: list[dict[int, float] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    result.append(None)
                    continue
                pos_dict: dict[int, float] = {}
                for token_id, logprob_obj in prompt_logprobs[i].items():
                    pos_dict[token_id] = logprob_obj.logprob
                result.append(pos_dict)

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


        async def test_mp_spawn_from_actor(self) -> dict:
            """Test multiprocessing spawn from within actor to debug PYTHONPATH."""
            import os
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
            import sys
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

    return ExtendedVLLMHttpServer


@dataclass
class GenerateResult:
    """Result from a generate call."""

    token_ids: list[int]
    log_probs: list[float] | None


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

        # Import verl components
        from verl.workers.config import HFModelConfig, RolloutConfig
        from verl.workers.rollout.replica import RolloutMode

        # Use our extended server with add_lora support
        ExtendedVLLMHttpServer = _create_extended_server_class()

        if not ray.is_initialized():
            # Use 'auto' to connect to existing cluster if available
            # If no cluster, this falls back to starting a local Ray instance
            # Use fixed namespace for persistent vLLM actor support
            ray.init(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

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
        max_model_len = cfg.max_model_len
        max_num_seqs = cfg.max_num_seqs or 256
        gpu_util = cfg.gpu_memory_utilization or self.gpu_memory_utilization
        # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
        max_num_batched_tokens = cfg.max_num_batched_tokens
        if max_num_batched_tokens is None:
            max_num_batched_tokens = 4096 if max_model_len >= 32768 else 8192
        # verl calculates max_model_len = prompt_length + response_length
        # We split evenly, but this does NOT restrict actual prompt/response sizes:
        # - The split only affects verl's default max_new_tokens (response_length)
        # - Our generate() computes actual limit dynamically: max_model_len - len(prompt)
        # - A 32K model can do 25K prompt + 7K response, or 5K prompt + 27K response
        # The sum (total context window) is what matters, not the individual values.
        prompt_length = max_model_len // 2
        response_length = max_model_len - prompt_length
        logger.info(f"vLLM max_model_len={max_model_len} (prompt={prompt_length}, response={response_length})")

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
            data_parallel_size=1,  # Keep at 1 to avoid verl's worker assertion
            expert_parallel_size=1,  # Keep at 1 to avoid verl's worker assertion
            engine_kwargs=engine_kwargs,
        )

        # Create model config using dataclass
        model_config = HFModelConfig(
            path=self.model_path,
            trust_remote_code=True,
            lora_rank=self.lora_rank,
            lora_adapter_path=self.lora_adapter_path,
        )

        logger.info(
            f"Launching vLLMHttpServer for {self.model_path} "
            f"(TP={self.tensor_parallel_size}, DP={self.data_parallel_size}, total_gpus={total_gpus}, "
            f"lora_rank={self.lora_rank}, adapter_path={self.lora_adapter_path})"
        )

        # Create ExtendedVLLMHttpServer as Ray actor
        # Request total_gpus (TP * DP) via .options() for MoE expert parallelism
        # runtime_env prepends vLLM 0.12.0 from PFS for MoE LoRA support
        self.server = ExtendedVLLMHttpServer.options(
            num_gpus=total_gpus,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": PFS_PYTHONPATH,
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                }
            },
        ).remote(
            config=rollout_config,
            model_config=model_config,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],  # No external workers for standalone
            replica_rank=0,
            node_rank=0,
            gpus_per_node=total_gpus,
            nnodes=1,
        )

        # Launch the server
        await self.server.launch_server.remote()
        self._initialized = True
        logger.info("vLLMHttpServer initialized")

    async def generate(
        self,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
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

        # Pass max_tokens to our overridden generate() in ExtendedVLLMHttpServer
        # which uses min(user_max_tokens, max_model_len - prompt_len)
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "logprobs": logprobs,
            "stop_token_ids": [151645, 151643],  # <|im_end|>, <|endoftext|> for Qwen2.5
        }

        # Call the Ray actor's generate method
        result = await self.server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        return GenerateResult(
            token_ids=list(result.token_ids),
            log_probs=list(result.log_probs) if result.log_probs else None,
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

        result = await self.server.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
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

        # Pass tensors via Ray to inference worker, which saves locally and loads
        # This handles distributed deployments where nodes have different filesystems
        worker_path = await self.server.add_lora_from_tensors.remote(state_dict, peft_config)

        logger.info(f"Hot-loaded LoRA adapter (API: {adapter_path} -> Worker: {worker_path})")

    async def shutdown(self) -> None:
        """Cleanup Ray actors."""
        if self.server:
            ray_kill.kill(self.server, reason="verl_inference_shutdown", namespace=RAY_NAMESPACE)
            self.server = None
        self._initialized = False
        logger.info("VerlInferenceEngine shutdown")
