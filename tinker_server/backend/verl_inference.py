"""Wrapper for verl's vLLMHttpServer for inference.

Uses verl's Ray-based vLLM infrastructure for scalable inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = logging.getLogger(__name__)

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

            # Save adapter files locally on worker node
            save_file(state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, indent=2)

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

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            await self.engine.add_lora(lora_request)

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
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        lora_rank: int = 0,
        lora_adapter_path: str | None = None,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
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
            ray.init(address='auto', ignore_reinit_error=True)

        # Create rollout config using dataclass
        rollout_config = RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            prompt_length=2048,
            response_length=2048,
            max_num_seqs=256,
            dtype="auto",
            load_format="auto",
            enforce_eager=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
            enable_prefix_caching=True,
            disable_log_stats=True,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            data_parallel_size=1,
        )
        if self.max_model_len is not None:
            rollout_config.max_model_len = self.max_model_len

        # Create model config using dataclass
        model_config = HFModelConfig(
            path=self.model_path,
            trust_remote_code=True,
            lora_rank=self.lora_rank,
            lora_adapter_path=self.lora_adapter_path,
        )

        logger.info(
            f"Launching vLLMHttpServer for {self.model_path} "
            f"(lora_rank={self.lora_rank}, adapter_path={self.lora_adapter_path})"
        )

        # Create ExtendedVLLMHttpServer as Ray actor
        # For MVP: single node, standalone mode
        # Request GPUs via .options() since vLLMHttpServer doesn't request them by default
        self.server = ExtendedVLLMHttpServer.options(
            num_gpus=self.tensor_parallel_size
        ).remote(
            config=rollout_config,
            model_config=model_config,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],  # No external workers for standalone
            replica_rank=0,
            node_rank=0,
            gpus_per_node=self.tensor_parallel_size,
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
            ray.kill(self.server)
            self.server = None
        self._initialized = False
        logger.info("VerlInferenceEngine shutdown")
