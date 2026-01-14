#!/usr/bin/env python3
"""Standalone K2 Megatron memory profiling.

This script runs Megatron training with artificial data to measure
actual GPU memory usage at each phase:
1. Model initialization
2. Forward pass
3. Loss computation
4. Backward pass
5. Optimizer step

Run on the Ray cluster head node to launch workers across GPUs.
"""

import argparse
import asyncio
import logging
import time

import ray

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def run_memory_profile(
    base_model: str,
    lora_rank: int,
    seq_len: int,
    batch_size: int,
    num_iterations: int = 3,
):
    """Run memory profiling with Megatron workers."""

    from tinker_server.backend.model_registry import get_training_parallelism, requires_fp8
    from tinker_server.backend.megatron_distributed import (
        DistributedConfig,
        async_get_or_create_megatron_worker_group,
    )

    # Get model-specific parallelism config
    train_tp, train_ep, train_etp = get_training_parallelism(base_model)
    use_fp8 = requires_fp8(base_model)

    world_size = train_tp * train_ep
    logger.info(f"Model: {base_model}")
    logger.info(f"Parallelism: TP={train_tp}, EP={train_ep}, ETP={train_etp}")
    logger.info(f"World size: {world_size} GPUs")
    logger.info(f"FP8: {use_fp8}")
    logger.info(f"LoRA rank: {lora_rank}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"Batch size: {batch_size}")

    config = DistributedConfig(
        tensor_parallel_size=train_tp,
        expert_parallel_size=train_ep,
        expert_tensor_parallel_size=train_etp,
        use_fp8=use_fp8,
    )

    logger.info("=" * 60)
    logger.info("Phase 1: Creating/Getting MegatronWorkerGroup")
    logger.info("=" * 60)

    t0 = time.time()
    worker_group = await async_get_or_create_megatron_worker_group(
        base_model=base_model,
        lora_rank=lora_rank,
        learning_rate=1e-4,
        distributed_config=config,
    )
    t1 = time.time()
    logger.info(f"Worker group ready in {t1 - t0:.1f}s")

    # Get memory after initialization
    logger.info("=" * 60)
    logger.info("Phase 2: Memory after model load")
    logger.info("=" * 60)

    memory_init = await asyncio.to_thread(
        ray.get,
        worker_group.log_memory_breakdown.remote("after_init")
    )
    logger.info(f"Memory after init: {memory_init}")

    # Create artificial data - just token IDs
    import numpy as np
    num_tokens = batch_size * seq_len
    input_ids = np.random.randint(0, 163840, size=num_tokens, dtype=np.int64).tolist()
    labels = np.random.randint(0, 163840, size=num_tokens, dtype=np.int64).tolist()

    logger.info("=" * 60)
    logger.info(f"Phase 3: Running {num_iterations} forward_backward iterations")
    logger.info("=" * 60)

    for i in range(num_iterations):
        logger.info(f"\n--- Iteration {i+1}/{num_iterations} ---")

        t0 = time.time()
        result = await asyncio.to_thread(
            ray.get,
            worker_group.forward_backward.remote(
                session_id="profiling",
                input_ids=input_ids,
                labels=labels,
                loss_fn="sft",  # Use simple SFT loss
            )
        )
        t1 = time.time()

        logger.info(f"forward_backward completed in {t1 - t0:.2f}s")
        logger.info(f"Result: loss={result.get('loss', 'N/A')}, grad_norm={result.get('grad_norm', 'N/A')}")

        # Get memory after forward_backward
        memory_fb = await asyncio.to_thread(
            ray.get,
            worker_group.log_memory_breakdown.remote(f"after_forward_backward_{i+1}")
        )
        logger.info(f"Memory: {memory_fb}")

    logger.info("=" * 60)
    logger.info("Phase 4: Running optimizer step")
    logger.info("=" * 60)

    t0 = time.time()
    optim_result = await asyncio.to_thread(
        ray.get,
        worker_group.optim_step.remote(session_id="profiling")
    )
    t1 = time.time()
    logger.info(f"optim_step completed in {t1 - t0:.2f}s")
    logger.info(f"Result: {optim_result}")

    # Get memory after optimizer step
    memory_optim = await asyncio.to_thread(
        ray.get,
        worker_group.log_memory_breakdown.remote("after_optim_step")
    )
    logger.info(f"Memory after optimizer: {memory_optim}")

    logger.info("=" * 60)
    logger.info("Memory profiling complete")
    logger.info("=" * 60)

    return {
        "memory_init": memory_init,
        "memory_final": memory_optim,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile K2 Megatron memory usage")
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/Kimi-K2-Thinking",
        help="Model name or path"
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--iterations", type=int, default=3, help="Number of forward_backward iterations")
    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    logger.info("=" * 60)
    logger.info("K2 Megatron Memory Profiling")
    logger.info("=" * 60)

    result = asyncio.run(run_memory_profile(
        base_model=args.model,
        lora_rank=args.lora_rank,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
    ))

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Initial memory: {result['memory_init']}")
    print(f"Final memory: {result['memory_final']}")


if __name__ == "__main__":
    main()
