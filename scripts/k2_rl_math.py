#!/usr/bin/env python3
"""K2 RL Training on MATH (Hendrycks) using tinker-cookbook.

Configuration (calibrated 2025-12-30):
- Training: 96 GPUs (TP=16, EP=96, ETP=1)
- Inference: 16 GPUs (TP=16)
- lora_rank=16 (must be divisible by TP=16)
- max_context=15,360 tokens (verified working limit)
- vLLM throughput: ~4 tok/s (cross-node bottleneck)

Usage:
    ssh -L 8000:localhost:8000 volcano  # SSH tunnel first
    cd /home/yiwen/tinker_project/tinker-server
    python scripts/k2_rl_math.py --hours 1
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime

# Add tinker-cookbook to path
COOKBOOK_PATH = "/home/yiwen/tinker_project/tinker-cookbook"
if COOKBOOK_PATH not in sys.path:
    sys.path.insert(0, COOKBOOK_PATH)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# K2 configuration
K2_MODEL = "moonshotai/Kimi-K2-Thinking"

# Global for signal handler
_training_task = None
_start_time = None
_target_duration = None


def timeout_handler(signum, frame):
    """Handler for training duration timeout."""
    if _training_task and not _training_task.done():
        elapsed = (time.time() - _start_time) / 3600 if _start_time else 0
        logger.info(f"Duration limit reached ({elapsed:.2f}h). Stopping training gracefully...")
        _training_task.cancel()


async def run_training(config, hours: float):
    """Run the cookbook training with timeout."""
    global _training_task, _start_time, _target_duration

    from tinker_cookbook.recipes.math_rl.train import cli_main

    _start_time = time.time()
    _target_duration = hours * 3600

    # Set up alarm for duration limit
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(_target_duration))

    try:
        _training_task = asyncio.current_task()
        await cli_main(config)
    except asyncio.CancelledError:
        logger.info("Training cancelled (duration limit reached)")
    finally:
        signal.alarm(0)  # Cancel alarm

    elapsed = (time.time() - _start_time) / 3600
    logger.info(f"Training completed after {elapsed:.2f} hours")


def main():
    from tinker_cookbook.recipes.math_rl.train import CLIConfig

    parser = argparse.ArgumentParser(description="K2 MATH RL Training")
    parser.add_argument("--hours", type=float, default=1.0, help="Training duration limit (hours)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (must be divisible by TP=16)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--group-size", type=int, default=4, help="Samples per problem")
    parser.add_argument("--groups-per-batch", type=int, default=4, help="Problems per batch (reduced for slow vLLM)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--eval-every", type=int, default=0, help="Eval frequency (0=skip eval)")
    args = parser.parse_args()

    # Validate lora_rank
    if args.lora_rank % 16 != 0:
        print(f"Error: lora_rank ({args.lora_rank}) must be divisible by TP=16")
        sys.exit(1)

    print("=" * 70)
    print("K2 RL Training on MATH (Hendrycks)")
    print("=" * 70)
    print(f"Model: {K2_MODEL}")
    print(f"Training: 96 GPUs (TP=16, EP=96, ETP=1) - Megatron backend")
    print(f"Inference: 16 GPUs (TP=16) - vLLM (~4 tok/s)")
    print(f"API: forward_backward_async + optim_step_async")
    print(f"Dataset: MATH (Hendrycks - competition math)")
    print(f"Reward: 1 if boxed answer correct, 0 otherwise")
    print(f"Target duration: {args.hours} hours")
    print()

    # Verify environment
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
    print(f"TINKER_BASE_URL: {base_url}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = f"/tmp/k2_rl_math_{timestamp}"

    # Configuration for K2 RL on MATH
    config = CLIConfig(
        model_name=K2_MODEL,
        lora_rank=args.lora_rank,
        renderer_name="kimi_k2",  # Dedicated renderer for Kimi-K2-Thinking
        env="math",  # Hendrycks MATH dataset
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,  # K2 needs long context for thinking tokens
        temperature=1.0,
        kl_penalty_coef=0.0,
        num_substeps=1,
        log_path=log_path,
        base_url=base_url,  # Pass the base URL from environment
        wandb_project=None,
        compute_post_kl=False,
        eval_every=args.eval_every,
        save_every=0,  # Disable checkpoint-based sampling (bug: base_model not passed)
        behavior_if_log_dir_exists="delete",
        loss_fn="importance_sampling",
    )

    print(f"Log path: {config.log_path}")
    print(f"Config:")
    print(f"  lora_rank: {config.lora_rank} (divisible by TP=16)")
    print(f"  groups_per_batch: {config.groups_per_batch}")
    print(f"  group_size: {config.group_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  max_tokens: {config.max_tokens}")
    print(f"  eval_every: {config.eval_every} (0=skip)")
    print()
    print("Expected throughput:")
    tokens_per_step = config.groups_per_batch * config.group_size * config.max_tokens
    time_per_step = tokens_per_step / 4  # ~4 tok/s vLLM throughput
    print(f"  Tokens per step: {tokens_per_step}")
    print(f"  Est. inference time: {time_per_step:.0f}s per step")
    print("=" * 70)
    print()

    # Run training
    try:
        asyncio.run(run_training(config, args.hours))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
