"""Wrapper for verl's vLLMHttpServer for inference.

Uses verl's Ray-based vLLM infrastructure for scalable inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = logging.getLogger(__name__)


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
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.lora_rank = lora_rank
        self.server: vLLMHttpServer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Ray and launch vLLM server."""
        if self._initialized:
            return

        # Import verl components
        from verl.workers.config import HFModelConfig, RolloutConfig
        from verl.workers.rollout.replica import RolloutMode
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

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
        )

        logger.info(f"Launching vLLMHttpServer for {self.model_path} (lora_rank={self.lora_rank})")

        # Create vLLMHttpServer as Ray actor
        # For MVP: single node, standalone mode
        # Request GPUs via .options() since vLLMHttpServer doesn't request them by default
        self.server = vLLMHttpServer.options(
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

    async def shutdown(self) -> None:
        """Cleanup Ray actors."""
        if self.server:
            ray.kill(self.server)
            self.server = None
        self._initialized = False
        logger.info("VerlInferenceEngine shutdown")
