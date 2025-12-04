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
def _create_extended_server_class():
    """Create extended vLLMHttpServer class with add_lora method."""
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID

    @ray.remote(num_cpus=1)
    class ExtendedVLLMHttpServer(vLLMHttpServerBase):
        """Extended vLLMHttpServer with hot LoRA loading support."""

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

        async def add_lora(self, lora_request) -> None:
            """Add LoRA adapter to running engine.

            Args:
                lora_request: TensorLoRARequest with peft_config and lora_tensors.
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

        async def list_loras(self) -> set[int]:
            """List loaded LoRA adapter IDs."""
            return await self.engine.list_loras()

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

        # Note: verl computes max_tokens internally as (max_model_len - prompt_len)
        # so we don't pass it here. For MVP, user's max_tokens is ignored.
        # TODO: Upstream fix to verl to support user-specified max_tokens
        sampling_params = {
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

        Args:
            adapter_path: Path to adapter directory containing
                          adapter_model.safetensors and adapter_config.json.
        """
        import json
        import os

        from safetensors.torch import load_file

        from verl.utils.vllm import TensorLoRARequest
        from verl.workers.rollout.vllm_rollout.utils import (
            VLLM_LORA_INT_ID,
            VLLM_LORA_NAME,
            VLLM_LORA_PATH,
        )

        if not self._initialized:
            await self.initialize()

        # Load config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, "r") as f:
            peft_config = json.load(f)

        # Load tensors
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        lora_tensors = load_file(weights_path)

        # Create TensorLoRARequest
        lora_request = TensorLoRARequest(
            lora_name=VLLM_LORA_NAME,
            lora_int_id=VLLM_LORA_INT_ID,
            lora_path=VLLM_LORA_PATH,
            peft_config=peft_config,
            lora_tensors=lora_tensors,
        )

        # Hot-load into running engine
        await self.server.add_lora.remote(lora_request)

        logger.info(f"Hot-loaded LoRA adapter from {adapter_path}")

    async def shutdown(self) -> None:
        """Cleanup Ray actors."""
        if self.server:
            ray.kill(self.server)
            self.server = None
        self._initialized = False
        logger.info("VerlInferenceEngine shutdown")
