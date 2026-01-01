#!/usr/bin/env python3
"""Megatron memory profiling with proper Datum format.

This script runs forward_backward on the existing K2 Megatron actor
with varying sequence lengths to profile memory usage.
"""

import argparse
import asyncio
import logging
import time

import ray

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_datum(seq_len: int, vocab_size: int = 163840) -> dict:
    """Create a single Datum dict in the expected format.

    Format:
        model_input.chunks[0].tokens: input token IDs
        loss_fn_inputs.target_tokens: target token IDs (shifted by 1)
        loss_fn_inputs.weights: per-token weights
    """
    import numpy as np

    # Generate random tokens
    tokens = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int64).tolist()

    # Target tokens are shifted by 1 (next token prediction)
    target_tokens = tokens[1:] + [tokens[0]]  # Shift left, wrap first token

    # All tokens have weight 1.0
    weights = [1.0] * seq_len

    return {
        "model_input": {
            "chunks": [{"tokens": tokens}]
        },
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [seq_len], "dtype": "int64"},
            "weights": {"data": weights, "shape": [seq_len], "dtype": "float32"},
        }
    }


async def profile_megatron_memory(
    seq_lengths: list[int],
    batch_size: int = 1,
    num_warmup: int = 1,
    num_iterations: int = 2,
):
    """Profile Megatron memory at different sequence lengths."""

    # Get existing Megatron actor
    actor = ray.get_actor("megatron_kimi_k2_thinking", namespace="tinker")

    # Get diagnostics
    diagnostics = ray.get(actor.get_diagnostics.remote())
    logger.info(f"Megatron config: {diagnostics}")

    results = []

    for seq_len in seq_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing seq_len={seq_len}, batch_size={batch_size}")
        logger.info(f"{'='*60}")

        # Create batch of data items
        data_items = [create_datum(seq_len) for _ in range(batch_size)]
        total_tokens = seq_len * batch_size
        logger.info(f"Total tokens: {total_tokens}")

        # Warmup iterations
        for i in range(num_warmup):
            logger.info(f"Warmup {i+1}/{num_warmup}...")
            try:
                t0 = time.time()
                result = ray.get(actor.forward_backward.remote(
                    data_items=data_items,
                    loss_fn="cross_entropy",  # "sft" not supported, use "cross_entropy"
                    session_id="memory_profile",
                ))
                t1 = time.time()
                logger.info(f"Warmup {i+1} done: loss={result.get('metrics', {}).get('loss:mean', 'N/A'):.4f}, time={t1-t0:.2f}s")
            except Exception as e:
                logger.error(f"Warmup failed: {e}")
                results.append({
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "status": "OOM" if "out of memory" in str(e).lower() else "ERROR",
                    "error": str(e)[:200],
                })
                break
        else:
            # Run timed iterations
            times = []
            for i in range(num_iterations):
                logger.info(f"Iteration {i+1}/{num_iterations}...")
                t0 = time.time()
                result = ray.get(actor.forward_backward.remote(
                    data_items=data_items,
                    loss_fn="cross_entropy",  # "sft" not supported, use "cross_entropy"
                    session_id="memory_profile",
                ))
                t1 = time.time()
                times.append(t1 - t0)
                logger.info(f"Iteration {i+1}: loss={result.get('metrics', {}).get('loss:mean', 'N/A'):.4f}, time={t1-t0:.2f}s")

            avg_time = sum(times) / len(times)
            tokens_per_sec = total_tokens / avg_time

            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "status": "OK",
                "avg_time_s": avg_time,
                "tokens_per_sec": tokens_per_sec,
            })

            logger.info(f"seq_len={seq_len}: avg_time={avg_time:.2f}s, throughput={tokens_per_sec:.0f} tokens/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile K2 Megatron memory")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to test"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=2, help="Timed iterations")
    args = parser.parse_args()

    # Initialize Ray
    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    logger.info("K2 Megatron Memory Profiling")
    logger.info(f"Sequence lengths: {args.seq_lengths}")

    results = asyncio.run(profile_megatron_memory(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    ))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'seq_len':<10} {'tokens':<10} {'status':<10} {'time(s)':<10} {'tok/s':<10}")
    print("-"*60)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['seq_len']:<10} {r['total_tokens']:<10} {r['status']:<10} {r['avg_time_s']:<10.2f} {r['tokens_per_sec']:<10.0f}")
        else:
            print(f"{r['seq_len']:<10} {r.get('total_tokens', 'N/A'):<10} {r['status']:<10} {'N/A':<10} {'N/A':<10}")
            if "error" in r:
                print(f"  Error: {r['error']}")


if __name__ == "__main__":
    main()
